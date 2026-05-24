"""
Fused Metal kernels and MTP generation loop for Qwen3.5-27B.

Contains:
  - Fused conv1d+SiLU kernel (single GPU dispatch for conv+activation)
  - Fused GDN step kernel (rms_norm+scale+g+beta+state_update in one kernel)
  - fused_gdn_call_v2: patched GatedDeltaNet forward with fused projections
  - patch_model: set up fused weights (_fused_w, _conv_w_flat, _A_exp etc.)
  - mtp_generate: generation loop with split-recurrence rollback (42.7 tok/s)

The split-recurrence approach:
  - Draft token t+2 via MTP head
  - Verify via T=2 forward pass
  - On accept: keep both tokens (2 tokens/step)
  - On reject: rollback DeltaNet states, trim KV caches, re-run T=1

Ported from ~/mlx-fork/fused_gdn.py (the working 42.7 tok/s version).
"""

from functools import partial
from typing import Any, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

try:
    from .fused_kernels_t2 import fused_conv1d_silu_t2, fused_gdn_step_with_intermediate
except ImportError:
    from fused_kernels_t2 import fused_conv1d_silu_t2, fused_gdn_step_with_intermediate


# ---------------------------------------------------------------------------
# Kernel 1: fused_conv1d_silu
# ---------------------------------------------------------------------------

def _make_fused_conv1d_silu_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        uint b_idx = thread_position_in_grid.z;
        uint ch    = thread_position_in_grid.x;
        if (ch >= conv_dim) return;

        uint state_base = b_idx * 3 * conv_dim;
        uint qkv_base   = b_idx * conv_dim;

        float x0 = static_cast<float>(conv_state[state_base + 0 * conv_dim + ch]);
        float x1 = static_cast<float>(conv_state[state_base + 1 * conv_dim + ch]);
        float x2 = static_cast<float>(conv_state[state_base + 2 * conv_dim + ch]);
        float x3 = static_cast<float>(qkv[qkv_base + ch]);

        uint w_base = ch * 4;
        float w0 = static_cast<float>(conv_w[w_base + 0]);
        float w1 = static_cast<float>(conv_w[w_base + 1]);
        float w2 = static_cast<float>(conv_w[w_base + 2]);
        float w3 = static_cast<float>(conv_w[w_base + 3]);

        float y = fma(x0, w0, fma(x1, w1, fma(x2, w2, x3 * w3)));

        // SiLU: y * sigmoid(y)
        y = y / (1.0f + fast::exp(-y));

        conv_out[qkv_base + ch] = static_cast<InT>(y);

        // Shift state: [old[1], old[2], current]
        new_state[state_base + 0 * conv_dim + ch] = conv_state[state_base + 1 * conv_dim + ch];
        new_state[state_base + 1 * conv_dim + ch] = conv_state[state_base + 2 * conv_dim + ch];
        new_state[state_base + 2 * conv_dim + ch] = qkv[qkv_base + ch];
    """

    return mx.fast.metal_kernel(
        name="fused_conv1d_silu_v2",
        input_names=["conv_state", "qkv", "conv_w", "conv_dim"],
        output_names=["conv_out", "new_state"],
        source=source,
    )


_fused_conv1d_silu = _make_fused_conv1d_silu_kernel()


def fused_conv1d_silu(
    conv_state: mx.array,   # [B, 3, conv_dim]
    qkv: mx.array,          # [B, 1, conv_dim]
    conv_weight: mx.array,  # [conv_dim, 4] (pre-flattened at patch time)
) -> Tuple[mx.array, mx.array]:
    B = conv_state.shape[0]
    conv_dim = conv_state.shape[2]
    dtype = qkv.dtype
    qkv_flat = qkv.reshape(B, conv_dim)
    tpg = 256
    n_groups = (conv_dim + tpg - 1) // tpg
    conv_out, new_state = _fused_conv1d_silu(
        inputs=[conv_state, qkv_flat, conv_weight, conv_dim],
        template=[("InT", dtype)],
        grid=(n_groups * tpg, 1, B),
        threadgroup=(tpg, 1, 1),
        output_shapes=[(B, conv_dim), (B, 3, conv_dim)],
        output_dtypes=[dtype, dtype],
    )
    return conv_out.reshape(B, 1, conv_dim), new_state


# ---------------------------------------------------------------------------
# Kernel 2: fused_gdn_step (with A_exp pre-computed, fast math)
# ---------------------------------------------------------------------------

def _make_fused_gdn_step_kernel(has_mask=False):
    if not mx.metal.is_available():
        return None

    mask_source = "mask[b_idx * T + t]" if has_mask else "true"

    source = f"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // q_raw, k_raw: [B, T, Hk, Dk]
        auto q_ptr = q_raw + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ptr = k_raw + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        // a, b_in: [B, T, Hv]
        auto a_ = a + b_idx * T * Hv;
        auto b_ = b_in + b_idx * T * Hv;

        // state: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
            state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
        }}

        constexpr float dk_inv = 1.0f / float(Dk);
        const float dk_inv_half = rsqrt(float(Dk));

        for (int t = 0; t < T; ++t) {{
            if ({mask_source}) {{
                // Phase 1: g and beta (per-head scalars)
                float a_val = static_cast<float>(a_[hv_idx]);
                float b_val = static_cast<float>(b_[hv_idx]);

                float A_e = static_cast<float>(A_exp[hv_idx]);
                float sp_arg = a_val + static_cast<float>(dt_bias[hv_idx]);
                float sp = sp_arg > 20.0f ? sp_arg : log(1.0f + fast::exp(sp_arg));
                float g_val = fast::exp(-A_e * sp);

                float beta_val = 1.0f / (1.0f + fast::exp(-b_val));

                // Phase 2: RMS norm + scale for q and k
                float q_local[n_per_t], k_local[n_per_t];
                float q_sq = 0.0f, k_sq = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {{
                    auto s_idx = n_per_t * dk_idx + i;
                    float qv = static_cast<float>(q_ptr[s_idx]);
                    float kv = static_cast<float>(k_ptr[s_idx]);
                    q_local[i] = qv;
                    k_local[i] = kv;
                    q_sq = fma(qv, qv, q_sq);
                    k_sq = fma(kv, kv, k_sq);
                }}
                q_sq = simd_sum(q_sq);
                k_sq = simd_sum(k_sq);

                float q_rms = rsqrt(fma(q_sq, dk_inv, 1e-6f));
                float k_rms = rsqrt(fma(k_sq, dk_inv, 1e-6f));

                float q_scale = q_rms * dk_inv;
                float k_scale = k_rms * dk_inv_half;

                // Phase 3: State update
                float kv_mem = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {{
                    q_local[i] *= q_scale;
                    k_local[i] *= k_scale;
                    state[i] *= g_val;
                    kv_mem = fma(state[i], k_local[i], kv_mem);
                }}
                kv_mem = simd_sum(kv_mem);

                float delta = (static_cast<float>(v_[dv_idx]) - kv_mem) * beta_val;

                float out = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {{
                    state[i] = fma(k_local[i], delta, state[i]);
                    out = fma(state[i], q_local[i], out);
                }}
                out = simd_sum(out);
                if (thread_index_in_simdgroup == 0) {{
                    y[dv_idx] = static_cast<InT>(out);
                }}
            }}

            q_ptr += Hk * Dk;
            k_ptr += Hk * Dk;
            v_ += Hv * Dv;
            y += Hv * Dv;
            a_ += Hv;
            b_ += Hv;
        }}

        for (int i = 0; i < n_per_t; ++i) {{
            o_state[n_per_t * dk_idx + i] = static_cast<InT>(state[i]);
        }}
    """

    inputs = ["q_raw", "k_raw", "v", "a", "b_in", "A_exp", "dt_bias", "state_in", "T"]
    if has_mask:
        inputs.append("mask")

    suffix = "_mask" if has_mask else ""

    return mx.fast.metal_kernel(
        name=f"fused_gdn_step_v2{suffix}",
        input_names=inputs,
        output_names=["y", "state_out"],
        source=source,
    )


_fused_gdn_step = _make_fused_gdn_step_kernel(has_mask=False)
_fused_gdn_step_masked = _make_fused_gdn_step_kernel(has_mask=True)


def fused_gdn_step(
    q_raw: mx.array, k_raw: mx.array, v: mx.array,
    a: mx.array, b: mx.array,
    A_exp: mx.array, dt_bias: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    B, T, Hk, Dk = q_raw.shape
    Hv, Dv = v.shape[2], v.shape[3]
    dtype = q_raw.dtype

    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=dtype)

    if mask is not None:
        kernel = _fused_gdn_step_masked
        inputs = [q_raw, k_raw, v, a, b, A_exp, dt_bias, state, T, mask]
    else:
        kernel = _fused_gdn_step
        inputs = [q_raw, k_raw, v, a, b, A_exp, dt_bias, state, T]

    return kernel(
        inputs=inputs,
        template=[("InT", dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[dtype, dtype],
    )


# ---------------------------------------------------------------------------
# Patched GatedDeltaNet forward (V2 fused projections)
# ---------------------------------------------------------------------------

def fused_gdn_call_v2(self, inputs: mx.array, mask=None, cache=None) -> mx.array:
    """Patched __call__ for GatedDeltaNet with fused projections and Metal kernels."""
    # Guard: if this instance wasn't patched (e.g. draft model), use original
    if not hasattr(self, '_fused_w'):
        return type(self)._original_call(self, inputs, mask=mask, cache=cache)

    B, S, _ = inputs.shape

    # --- 1. Single fused projection (4 matmuls -> 1) ---
    combined = mx.quantized_matmul(
        inputs, self._fused_w, self._fused_s, self._fused_bi,
        group_size=self._fused_gs, bits=self._fused_bits,
    )
    qkv = combined[..., :self._qkv_end]
    z = combined[..., self._qkv_end:self._z_end].reshape(B, S, self.num_v_heads, self.head_v_dim)
    b = combined[..., self._z_end:self._b_end]
    a = combined[..., self._b_end:]

    # --- 2. Conv state ---
    if cache is not None and cache[0] is not None:
        conv_state = cache[0]
    else:
        conv_state = mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype)

    # --- 3. Fused conv1d + silu (decode) or standard (prefill) ---
    if S == 1 and _fused_conv1d_silu is not None:
        conv_out, new_conv_state = fused_conv1d_silu(conv_state, qkv, self._conv_w_flat)
        if cache is not None:
            cache[0] = new_conv_state
    else:
        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            cache[0] = conv_input[:, -(self.conv_kernel_size - 1):]
        conv_out = nn.silu(self.conv1d(conv_input))

    # --- 4. Split (free pointer arithmetic) ---
    q_raw = conv_out[..., :self.key_dim].reshape(B, S, self.num_k_heads, self.head_k_dim)
    k_raw = conv_out[..., self.key_dim:2*self.key_dim].reshape(B, S, self.num_k_heads, self.head_k_dim)
    v = conv_out[..., 2*self.key_dim:].reshape(B, S, self.num_v_heads, self.head_v_dim)

    state = cache[1] if cache else None

    # --- 5. Fused gdn step ---
    if _fused_gdn_step is not None and mx.default_device() == mx.gpu:
        out, state = fused_gdn_step(
            q_raw, k_raw, v, a, b,
            self._A_exp, self.dt_bias, state, mask,
        )
    else:
        from mlx_lm.models.gated_delta import gated_delta_update
        inv_scale = self.head_k_dim ** -0.5
        q_n = (inv_scale ** 2) * mx.fast.rms_norm(q_raw, None, 1e-6)
        k_n = inv_scale * mx.fast.rms_norm(k_raw, None, 1e-6)
        out, state = gated_delta_update(
            q_n, k_n, v, a, b,
            self.A_log, self.dt_bias, state, mask, use_kernel=True,
        )

    if cache is not None:
        cache[1] = state

    # --- 6. Post-processing ---
    out = self.norm(out, z)
    return self.out_proj(out.reshape(B, S, -1))


# ---------------------------------------------------------------------------
# patch_model: set up fused weights for all DeltaNet layers
# ---------------------------------------------------------------------------

_patched_classes = set()


def patch_model(model):
    """
    Replace GatedDeltaNet forward passes with fused/compiled kernels.

    Sets up:
      - _fused_w/_fused_s/_fused_bi: concatenated input projection weights (4->1 matmul)
      - _conv_w_flat: pre-flattened conv1d weights for Metal kernel
      - _A_exp: pre-computed exp(A_log) for GDN step kernel
      - Monkey-patches GatedDeltaNet.__call__ to fused_gdn_call_v2

    Args:
        model: Qwen3.5-27B model loaded via mlx_lm.load

    Returns:
        The patched model (same object, modified in-place)
    """
    if hasattr(model, 'language_model'):
        layers = model.language_model.model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    else:
        raise ValueError("Cannot find model layers to patch")

    patched = 0
    arrays_to_eval = []

    for i, layer in enumerate(layers):
        if not (hasattr(layer, 'is_linear') and layer.is_linear):
            continue
        attn = layer.linear_attn

        # Fuse 4 input projections into 1 matmul
        attn._fused_w = mx.concatenate([
            attn.in_proj_qkv.weight, attn.in_proj_z.weight,
            attn.in_proj_b.weight, attn.in_proj_a.weight,
        ], axis=0)
        attn._fused_s = mx.concatenate([
            attn.in_proj_qkv.scales, attn.in_proj_z.scales,
            attn.in_proj_b.scales, attn.in_proj_a.scales,
        ], axis=0)
        attn._fused_bi = mx.concatenate([
            attn.in_proj_qkv.biases, attn.in_proj_z.biases,
            attn.in_proj_b.biases, attn.in_proj_a.biases,
        ], axis=0)
        attn._fused_gs = attn.in_proj_qkv.group_size
        attn._fused_bits = attn.in_proj_qkv.bits
        attn._qkv_end = attn.key_dim * 2 + attn.value_dim
        attn._z_end = attn._qkv_end + attn.value_dim
        attn._b_end = attn._z_end + attn.num_v_heads

        attn._A_exp = mx.exp(attn.A_log.astype(mx.float32))
        attn._conv_w_flat = attn.conv1d.weight.reshape(attn.conv_dim, 4)

        arrays_to_eval.extend([
            attn._fused_w, attn._fused_s, attn._fused_bi,
            attn._A_exp, attn._conv_w_flat,
        ])

        cls = type(attn)
        if cls not in _patched_classes:
            cls._original_call = cls.__call__
            cls.__call__ = fused_gdn_call_v2
            _patched_classes.add(cls)
        patched += 1

    mx.eval(*arrays_to_eval)
    print(f"Patched {patched} GatedDeltaNet layers (fused projections + Metal kernels)")
    return model


def unpatch_model(model):
    """Restore original GatedDeltaNet forward passes."""
    if hasattr(model, 'language_model'):
        layers = model.language_model.model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    else:
        return

    for layer in layers:
        if hasattr(layer, 'is_linear') and layer.is_linear:
            attn = layer.linear_attn
            cls = type(attn)
            if cls in _patched_classes:
                cls.__call__ = cls._original_call
                del cls._original_call
                _patched_classes.discard(cls)
            for attr in ['_fused_w', '_fused_s', '_fused_bi', '_fused_gs', '_fused_bits',
                         '_qkv_end', '_z_end', '_b_end', '_A_exp', '_conv_w_flat']:
                if hasattr(attn, attr):
                    delattr(attn, attr)

    print("Restored original forward passes")


# ---------------------------------------------------------------------------
# MTP generation loop with split-recurrence rollback
# ---------------------------------------------------------------------------

def mtp_generate(model, tokenizer, prompt, max_tokens=256, mtp_head=None, verbose=True):
    """
    Generate with MTP speculative decoding (split-recurrence rollback).

    The key insight: DeltaNet layers have recurrent state that must be rolled
    back on draft rejection, while attention layers use KV caches that can be
    trimmed by decrementing the offset. This function:

      1. Drafts token t+2 using the MTP head (one transformer layer)
      2. Verifies by running T=2 forward [token, draft] through the full model
      3. On accept: keeps both tokens (2 tokens per step)
      4. On reject: restores DeltaNet states, trims KV offsets, re-runs T=1

    Achieves 42.7 tok/s on M4 Max (1.45x over baseline 29.5 tok/s).

    Args:
        model: Qwen3.5-27B model (patched or unpatched)
        tokenizer: The tokenizer
        prompt: Input text
        max_tokens: Maximum tokens to generate
        mtp_head: MTPHead instance (loads from default path if None)
        verbose: Print timing stats

    Returns:
        Generated text string
    """
    import time
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    try:
        from .mtp_head import MTPHead, load_mtp
    except ImportError:
        from mtp_head import MTPHead, load_mtp

    if hasattr(model, 'language_model'):
        text_model = model.language_model.model
        lm_head = model.language_model.lm_head
    else:
        text_model = model.model
        lm_head = model.lm_head

    if mtp_head is None:
        mtp_head = load_mtp(model)

    prompt_tokens = tokenizer.encode(prompt)
    cache = make_prompt_cache(model)
    layers = text_model.layers

    def fwd_t1(token_arr):
        """Standard T=1 forward pass."""
        h = text_model.embed_tokens(token_arr)
        fa = create_attention_mask(h, cache[text_model.fa_idx])
        ss = create_ssm_mask(h, cache[text_model.ssm_idx])
        for layer, c in zip(layers, cache):
            h = layer(h, mask=(ss if layer.is_linear else fa), cache=c)
        return h, lm_head(text_model.norm(h))

    def fwd_t2_rollback(tok0, tok1):
        """
        T=2 forward with split-recurrence rollback (fused T=2 kernels).

        Matmuls are batched at T=2 (same weight reads). The GDN recurrence
        uses fused T=2 kernels that process both tokens in a single dispatch
        while also capturing intermediate state for rollback.

        This uses 2 kernel dispatches per DeltaNet layer instead of 4:
          - 1x fused_conv1d_silu_t2 (was 2x fused_conv1d_silu)
          - 1x fused_gdn_step_with_intermediate (was 2x fused_gdn_step)

        Saves ~48 kernel dispatches (~1ms) across 48 DeltaNet layers.
        """
        inp = mx.concatenate([tok0.reshape(1, 1), tok1.reshape(1, 1)], axis=1)
        h = text_model.embed_tokens(inp)
        fa = create_attention_mask(h, cache[text_model.fa_idx])
        ss = create_ssm_mask(h, cache[text_model.ssm_idx])

        delta_stash = []  # intermediate DeltaNet states after token 0
        kv_offsets = []   # KV cache offsets after token 0

        for layer, c in zip(layers, cache):
            if layer.is_linear:
                attn = layer.linear_attn
                h_norm = layer.input_layernorm(h)

                # Batched input projection (T=2, same weight reads)
                combined = attn.in_proj_qkv(h_norm)
                z = attn.in_proj_z(h_norm).reshape(1, 2, attn.num_v_heads, attn.head_v_dim)
                b_val = attn.in_proj_b(h_norm)
                a_val = attn.in_proj_a(h_norm)

                conv_st = c[0] if c[0] is not None else mx.zeros(
                    (1, attn.conv_kernel_size - 1, attn.conv_dim), dtype=h.dtype)
                rnn_st = c[1]
                kd = attn.key_dim

                # Fused T=2 conv1d+SiLU: 1 dispatch instead of 2
                co0, co1, conv_mid, conv_fin = fused_conv1d_silu_t2(
                    conv_st, combined, attn._conv_w_flat)

                # Split q/k/v from both conv outputs
                q0 = co0[..., :kd].reshape(1, 1, attn.num_k_heads, attn.head_k_dim)
                k0 = co0[..., kd:2*kd].reshape(1, 1, attn.num_k_heads, attn.head_k_dim)
                v0 = co0[..., 2*kd:].reshape(1, 1, attn.num_v_heads, attn.head_v_dim)
                q1 = co1[..., :kd].reshape(1, 1, attn.num_k_heads, attn.head_k_dim)
                k1 = co1[..., kd:2*kd].reshape(1, 1, attn.num_k_heads, attn.head_k_dim)
                v1 = co1[..., 2*kd:].reshape(1, 1, attn.num_v_heads, attn.head_v_dim)

                # Concatenate to T=2 for fused GDN kernel
                q_t2 = mx.concatenate([q0, q1], axis=1)
                k_t2 = mx.concatenate([k0, k1], axis=1)
                v_t2 = mx.concatenate([v0, v1], axis=1)

                # Fused T=2 GDN step: 1 dispatch instead of 2
                out, rnn_fin, rnn_mid = fused_gdn_step_with_intermediate(
                    q_t2, k_t2, v_t2, a_val, b_val,
                    attn._A_exp, attn.dt_bias, rnn_st)

                # Save intermediate state for rollback (zero-copy refs)
                delta_stash.append((conv_mid, rnn_mid))

                c[0] = conv_fin
                c[1] = rnn_fin

                # Batched output projection + MLP
                out = attn.norm(out, z)
                r = attn.out_proj(out.reshape(1, 2, -1))
                h = h + r
                h = h + layer.mlp(layer.post_attention_layernorm(h))
            else:
                # Attention layers: process T=2 normally (KV cache is trimmable)
                kv_offsets.append(c.offset)
                h = layer(h, mask=fa, cache=c)

        h_pre = h
        logits = lm_head(text_model.norm(h))

        def rollback():
            """Restore to state after token 0. Zero cost."""
            di, ki = 0, 0
            for layer, c in zip(layers, cache):
                if layer.is_linear:
                    c[0], c[1] = delta_stash[di]
                    di += 1
                else:
                    c.offset = kv_offsets[ki] + 1  # after token 0
                    ki += 1

        return h_pre, logits, rollback

    # --- Prefill ---
    t0 = time.perf_counter()
    h_pre, logits = fwd_t1(mx.array(prompt_tokens)[None])
    mx.eval(logits)
    token = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(token)
    t_prefill = time.perf_counter() - t0
    if verbose:
        print(f"Prefill: {len(prompt_tokens)} tokens, {t_prefill*1000:.0f}ms")

    # --- First MTP draft ---
    embed = text_model.embed_tokens(token.reshape(1)).reshape(1, 1, -1)
    draft_logits, _ = mtp_head(h_pre[:, -1:, :], embed, lm_head)
    draft = mx.argmax(draft_logits[:, -1, :], axis=-1)
    mx.eval(draft)

    generated = [token.item()]
    n_accepted = 0
    n_steps = 0

    # --- Generation loop (split-recurrence rollback) ---
    t_gen = time.perf_counter()
    while len(generated) < max_tokens:
        n_steps += 1

        # T=2 verify with split-recurrence: batched matmuls, split GDN
        h2, logits2, rollback = fwd_t2_rollback(token, draft)

        verify = mx.argmax(logits2[:, 0, :], axis=-1)
        bonus = mx.argmax(logits2[:, 1, :], axis=-1)

        # Pipeline: compute next MTP draft (optimistic, assumes accept)
        embed_bonus = text_model.embed_tokens(bonus.reshape(1)).reshape(1, 1, -1)
        next_draft_logits, _ = mtp_head(h2[:, -1:, :], embed_bonus, lm_head)
        next_draft = mx.argmax(next_draft_logits[:, -1, :], axis=-1)

        # Single eval: all resolve together
        mx.eval(verify, bonus, next_draft)

        if verify.item() == draft.item():
            # ACCEPT — states are correct, continue
            generated.append(draft.item())
            generated.append(bonus.item())
            n_accepted += 1
            token = bonus
            draft = next_draft
        else:
            # REJECT — restore to after token 0. NO REDO.
            rollback()
            generated.append(verify.item())
            token = verify

            # MTP draft from the correct hidden state (position 0)
            embed_v = text_model.embed_tokens(verify.reshape(1)).reshape(1, 1, -1)
            draft_logits_v, _ = mtp_head(h2[:, 0:1, :], embed_v, lm_head)
            draft = mx.argmax(draft_logits_v[:, -1, :], axis=-1)
            mx.eval(draft)

        if token.item() == tokenizer.eos_token_id:
            break

    gen_time = time.perf_counter() - t_gen
    n = len(generated)

    eos = tokenizer.eos_token_id
    if eos is not None and eos in generated:
        generated = generated[:generated.index(eos)]
        n = len(generated)

    output = tokenizer.decode(generated)
    if verbose:
        tps = n / gen_time
        accept_rate = n_accepted / max(n_steps, 1)
        print(f"Generate: {n} tokens, {gen_time*1000:.0f}ms ({tps:.1f} tok/s)")
        print(f"MTP acceptance: {n_accepted}/{n_steps} ({accept_rate * 100:.1f}%)")
        print(f"Tokens per step: {n/max(n_steps,1):.2f}")
    return output
