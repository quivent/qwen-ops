"""
Fused Metal kernels for Qwen3.5-27B decode step (V7).

Optimizations over stock mlx_lm:
  V2: Fuse 4 DeltaNet projections → 1 matmul, fused conv1d+silu kernel,
      fused GDN step kernel (rms_norm+scale+g+beta+state_update),
      pre-compute A_exp, fast math, pre-flatten conv weights
  V3: mx.compile entire DeltaNet+MLP layers, fuse MLP gate+up → 1 matmul
  V4: Compile attention layers: fuse QKV (3→1 matmul), fuse MLP gate+up,
      compiled pre-SDPA and post-SDPA blocks
  V5: Monolithic compile — ONE mx.compile for entire 64-layer forward.
      Eliminates all Python dispatch between layers during decode.
      Uses put_along_axis for KV cache, mx.fast.rope with array offset.
  V7: GPU-resident autoregressive loop — embed + forward + sample + cache
      update all in ONE compiled graph. CPU dispatches ONCE for N tokens.
      Eliminates N-1 CPU→GPU round trips.

Usage:
    from fused_gdn import patch_model, gpu_generate
    model, tok = mlx_lm.load('mlx-community/Qwen3.5-27B-4bit')
    patch_model(model)
    tokens = gpu_generate(model, tok, prompt="Hello", max_tokens=256)
"""

from functools import partial
from typing import Any, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


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
# V2 fused forward pass (used during prefill, S > 1)
# ---------------------------------------------------------------------------

def fused_gdn_call_v2(self, inputs: mx.array, mask=None, cache=None) -> mx.array:
    # Guard: if this instance wasn't patched (e.g. draft model), use original
    if not hasattr(self, '_fused_w'):
        return type(self)._original_call(self, inputs, mask=mask, cache=cache)

    B, S, _ = inputs.shape

    # --- 1. Single fused projection (4 matmuls → 1) ---
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
# V3: Compiled DeltaNet+MLP layers (bypass Python entirely during decode)
# ---------------------------------------------------------------------------

def _make_compiled_delta_layer(layer):
    """
    Create a compiled decode function for a single DeltaNet+MLP layer.

    After first call (trace), subsequent calls replay the C++ graph
    without executing any Python. This eliminates ~50 us/layer of
    Python interpreter overhead.

    Args:
        layer: A DecoderLayer with is_linear=True

    Returns:
        Compiled function: (x, conv_state, rnn_state) -> (output, new_conv, new_rnn)
    """
    attn = layer.linear_attn

    # --- Capture all weights as closures (constants in compiled graph) ---

    # Fused input projection
    fw, fs, fb = attn._fused_w, attn._fused_s, attn._fused_bi
    gs, bits = attn._fused_gs, attn._fused_bits
    qkv_end = attn._qkv_end
    z_end = attn._z_end
    b_end = attn._b_end

    # Head dimensions
    Hv = attn.num_v_heads
    Hk = attn.num_k_heads
    Dv = attn.head_v_dim
    Dk = attn.head_k_dim
    kd = attn.key_dim
    vd = Hv * Dv

    # Conv
    cw = attn._conv_w_flat

    # GDN
    ae = attn._A_exp
    db = attn.dt_bias

    # Norms
    nw = attn.norm.weight
    ne = attn.norm.eps

    # Out projection
    ow = attn.out_proj.weight
    os_ = attn.out_proj.scales
    ob = attn.out_proj.biases
    ogs = attn.out_proj.group_size
    obits = attn.out_proj.bits

    # Layer norms
    lnw = layer.input_layernorm.weight
    lne = layer.input_layernorm.eps
    plnw = layer.post_attention_layernorm.weight
    plne = layer.post_attention_layernorm.eps

    # MLP — fuse gate+up into single matmul
    mlp = layer.mlp
    fmw = mx.concatenate([mlp.gate_proj.weight, mlp.up_proj.weight], axis=0)
    fms = mx.concatenate([mlp.gate_proj.scales, mlp.up_proj.scales], axis=0)
    fmbi = mx.concatenate([mlp.gate_proj.biases, mlp.up_proj.biases], axis=0)
    mgs = mlp.gate_proj.group_size
    mbits = mlp.gate_proj.bits
    isz = mlp.gate_proj.weight.shape[0]  # intermediate_size (output rows)

    dw = mlp.down_proj.weight
    ds = mlp.down_proj.scales
    dbi = mlp.down_proj.biases
    dgs = mlp.down_proj.group_size
    dbits = mlp.down_proj.bits

    # The compiled function: arrays in, arrays out. No Python state.
    # No shapeless=True: decode always has B=1, S=1 (fixed shapes).
    @mx.compile
    def decode(x, conv_state, rnn_state):
        # 1. Input LayerNorm
        h = mx.fast.rms_norm(x, lnw, lne)

        # 2. Fused input projection (4 matmuls → 1)
        c = mx.quantized_matmul(h, fw, fs, fb, group_size=gs, bits=bits)
        qkv = c[..., :qkv_end]
        z = c[..., qkv_end:z_end].reshape(1, 1, Hv, Dv)
        bv = c[..., z_end:b_end]
        av = c[..., b_end:]

        # 3. Fused conv1d + silu
        co, nc = fused_conv1d_silu(conv_state, qkv, cw)

        # 4. Split
        qr = co[..., :kd].reshape(1, 1, Hk, Dk)
        kr = co[..., kd:2*kd].reshape(1, 1, Hk, Dk)
        v = co[..., 2*kd:].reshape(1, 1, Hv, Dv)

        # 5. Fused GDN step (rms_norm + scale + g + beta + state update)
        out, nr = fused_gdn_step(qr, kr, v, av, bv, ae, db, rnn_state, None)

        # 6. RMSNormGated = swiglu(z, rms_norm(out))
        out = nn.silu(z) * mx.fast.rms_norm(out, nw, ne)

        # 7. Out projection
        r = mx.quantized_matmul(
            out.reshape(1, 1, vd), ow, os_, ob,
            group_size=ogs, bits=obits,
        )

        # 8. Residual
        h2 = x + r

        # 9. Post-attention LayerNorm
        h2n = mx.fast.rms_norm(h2, plnw, plne)

        # 10. MLP: fused gate+up → silu*up → down
        gu = mx.quantized_matmul(h2n, fmw, fms, fmbi, group_size=mgs, bits=mbits)
        m = nn.silu(gu[..., :isz]) * gu[..., isz:]
        m = mx.quantized_matmul(m, dw, ds, dbi, group_size=dgs, bits=dbits)

        return h2 + m, nc, nr

    return decode, [fmw, fms, fmbi]


# ---------------------------------------------------------------------------
# V4: Compiled attention layers (pre-SDPA + post-SDPA blocks)
# ---------------------------------------------------------------------------

def _make_compiled_attn_pre(layer):
    """
    Compiled pre-SDPA block for attention layers during decode.
    Fuses: input_layernorm + QKV projection (3→1 matmul) + q/k norms + transpose.
    Saves 2 GPU dispatches per layer (3 matmuls → 1) plus element-wise fusion.
    """
    attn = layer.self_attn

    # Input layernorm
    lnw = layer.input_layernorm.weight
    lne = layer.input_layernorm.eps

    # Fuse q + k + v projections into one matmul
    fqw = mx.concatenate([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], axis=0)
    fqs = mx.concatenate([attn.q_proj.scales, attn.k_proj.scales, attn.v_proj.scales], axis=0)
    fqb = mx.concatenate([attn.q_proj.biases, attn.k_proj.biases, attn.v_proj.biases], axis=0)
    gs = attn.q_proj.group_size
    bits = attn.q_proj.bits

    q_dim = attn.num_attention_heads * attn.head_dim * 2  # doubled for gate
    k_dim = attn.num_key_value_heads * attn.head_dim
    nh = attn.num_attention_heads
    nkv = attn.num_key_value_heads
    hd = attn.head_dim

    qnw = attn.q_norm.weight
    qne = attn.q_norm.eps
    knw = attn.k_norm.weight
    kne = attn.k_norm.eps

    @mx.compile
    def pre(x):
        h = mx.fast.rms_norm(x, lnw, lne)
        out = mx.quantized_matmul(h, fqw, fqs, fqb, group_size=gs, bits=bits)

        q_out = out[..., :q_dim].reshape(1, 1, nh, hd * 2)
        q = q_out[..., :hd]
        gate = q_out[..., hd:].reshape(1, 1, nh * hd)

        k = out[..., q_dim:q_dim + k_dim].reshape(1, 1, nkv, hd)
        v = out[..., q_dim + k_dim:].reshape(1, 1, nkv, hd)

        q = mx.fast.rms_norm(q, qnw, qne).transpose(0, 2, 1, 3)
        k = mx.fast.rms_norm(k, knw, kne).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        return q, k, v, gate

    return pre, [fqw, fqs, fqb]


def _make_compiled_attn_post(layer):
    """
    Compiled post-SDPA block for attention layers during decode.
    Fuses: gate*output → o_proj → residual → post_norm → MLP (gate+up fused) → residual.
    Saves 1 dispatch (MLP fusion) + element-wise fusions (sigmoid*mul, silu*mul).
    """
    attn = layer.self_attn

    ow = attn.o_proj.weight
    os_ = attn.o_proj.scales
    ob = attn.o_proj.biases
    ogs = attn.o_proj.group_size
    obits = attn.o_proj.bits

    nh = attn.num_attention_heads
    hd = attn.head_dim

    plnw = layer.post_attention_layernorm.weight
    plne = layer.post_attention_layernorm.eps

    mlp = layer.mlp
    fmw = mx.concatenate([mlp.gate_proj.weight, mlp.up_proj.weight], axis=0)
    fms = mx.concatenate([mlp.gate_proj.scales, mlp.up_proj.scales], axis=0)
    fmbi = mx.concatenate([mlp.gate_proj.biases, mlp.up_proj.biases], axis=0)
    mgs = mlp.gate_proj.group_size
    mbits = mlp.gate_proj.bits
    isz = mlp.gate_proj.weight.shape[0]

    dw = mlp.down_proj.weight
    ds = mlp.down_proj.scales
    dbi = mlp.down_proj.biases
    dgs = mlp.down_proj.group_size
    dbits = mlp.down_proj.bits

    @mx.compile
    def post(sdpa_out, gate, x):
        flat = sdpa_out.transpose(0, 2, 1, 3).reshape(1, 1, nh * hd)
        r = mx.quantized_matmul(
            flat * mx.sigmoid(gate), ow, os_, ob,
            group_size=ogs, bits=obits,
        )
        h2 = x + r

        h2n = mx.fast.rms_norm(h2, plnw, plne)
        gu = mx.quantized_matmul(h2n, fmw, fms, fmbi, group_size=mgs, bits=mbits)
        m = nn.silu(gu[..., :isz]) * gu[..., isz:]
        m = mx.quantized_matmul(m, dw, ds, dbi, group_size=dgs, bits=dbits)
        return h2 + m

    return post, [fmw, fms, fmbi]


# ---------------------------------------------------------------------------
# V5: Monolithic compiled forward (entire model in ONE mx.compile)
# ---------------------------------------------------------------------------

def _build_monolithic_decode(text_model, lm_head_w, lm_head_s, lm_head_b, lm_head_gs, lm_head_bits):
    """
    Build a single @mx.compile function for the entire decode step (V5).

    Takes (hidden_states, offset, *flat_cache) where flat_cache contains
    all DeltaNet conv/rnn states and attention KV caches.
    Returns (logits, *updated_flat_cache).

    All weights are captured as closures — zero Python per token.
    """
    layers = text_model.layers

    # Final norm
    fnw = text_model.norm.weight
    fne = text_model.norm.eps

    # Build per-layer weight closures
    layer_data = []

    for i, layer in enumerate(layers):
        if layer.is_linear:
            attn = layer.linear_attn
            mlp = layer.mlp
            d = {
                'fw': attn._fused_w, 'fs': attn._fused_s, 'fb': attn._fused_bi,
                'gs': attn._fused_gs, 'bits': attn._fused_bits,
                'qkv_end': attn._qkv_end, 'z_end': attn._z_end, 'b_end': attn._b_end,
                'Hv': attn.num_v_heads, 'Hk': attn.num_k_heads,
                'Dv': attn.head_v_dim, 'Dk': attn.head_k_dim,
                'kd': attn.key_dim,
                'cw': attn._conv_w_flat,
                'ae': attn._A_exp, 'db': attn.dt_bias,
                'nw': attn.norm.weight, 'ne': attn.norm.eps,
                'ow': attn.out_proj.weight, 'os': attn.out_proj.scales,
                'ob': attn.out_proj.biases, 'ogs': attn.out_proj.group_size,
                'obits': attn.out_proj.bits,
                'lnw': layer.input_layernorm.weight,
                'lne': layer.input_layernorm.eps,
                'plnw': layer.post_attention_layernorm.weight,
                'plne': layer.post_attention_layernorm.eps,
                'fmw': None, 'fms': None, 'fmbi': None,
                'mgs': 0, 'mbits': 0, 'isz': 0,
                'dw': mlp.down_proj.weight, 'ds': mlp.down_proj.scales,
                'dbi': mlp.down_proj.biases,
                'dgs': mlp.down_proj.group_size, 'dbits': mlp.down_proj.bits,
            }
            d['fmw'] = mx.concatenate([mlp.gate_proj.weight, mlp.up_proj.weight], axis=0)
            d['fms'] = mx.concatenate([mlp.gate_proj.scales, mlp.up_proj.scales], axis=0)
            d['fmbi'] = mx.concatenate([mlp.gate_proj.biases, mlp.up_proj.biases], axis=0)
            d['mgs'] = mlp.gate_proj.group_size
            d['mbits'] = mlp.gate_proj.bits
            d['isz'] = mlp.gate_proj.weight.shape[0]
            layer_data.append(('delta', d))
        else:
            sa = layer.self_attn
            mlp = layer.mlp
            fqw = mx.concatenate([sa.q_proj.weight, sa.k_proj.weight, sa.v_proj.weight], axis=0)
            fqs = mx.concatenate([sa.q_proj.scales, sa.k_proj.scales, sa.v_proj.scales], axis=0)
            fqb = mx.concatenate([sa.q_proj.biases, sa.k_proj.biases, sa.v_proj.biases], axis=0)
            q_dim = sa.num_attention_heads * sa.head_dim * 2
            k_dim = sa.num_key_value_heads * sa.head_dim
            fmw = mx.concatenate([mlp.gate_proj.weight, mlp.up_proj.weight], axis=0)
            fms = mx.concatenate([mlp.gate_proj.scales, mlp.up_proj.scales], axis=0)
            fmbi = mx.concatenate([mlp.gate_proj.biases, mlp.up_proj.biases], axis=0)
            layer_data.append(('attn', {
                'fqw': fqw, 'fqs': fqs, 'fqb': fqb,
                'gs': sa.q_proj.group_size, 'bits': sa.q_proj.bits,
                'q_dim': q_dim, 'k_dim': k_dim,
                'nh': sa.num_attention_heads, 'nkv': sa.num_key_value_heads,
                'hd': sa.head_dim, 'scale': sa.scale,
                'qnw': sa.q_norm.weight, 'qne': sa.q_norm.eps,
                'knw': sa.k_norm.weight, 'kne': sa.k_norm.eps,
                'ow': sa.o_proj.weight, 'os': sa.o_proj.scales,
                'ob': sa.o_proj.biases, 'ogs': sa.o_proj.group_size,
                'obits': sa.o_proj.bits,
                'lnw': layer.input_layernorm.weight,
                'lne': layer.input_layernorm.eps,
                'plnw': layer.post_attention_layernorm.weight,
                'plne': layer.post_attention_layernorm.eps,
                'fmw': fmw, 'fms': fms, 'fmbi': fmbi,
                'mgs': mlp.gate_proj.group_size, 'mbits': mlp.gate_proj.bits,
                'isz': mlp.gate_proj.weight.shape[0],
                'dw': mlp.down_proj.weight, 'ds': mlp.down_proj.scales,
                'dbi': mlp.down_proj.biases,
                'dgs': mlp.down_proj.group_size, 'dbits': mlp.down_proj.bits,
                'rope_dims': 32, 'rope_base': 100000.0,
            }))

    # Build V5 layer functions (standard residual)
    layer_fns = []
    for i, (ltype, d) in enumerate(layer_data):
        if ltype == 'delta':
            layer_fns.append(_make_delta_layer_fn(d))
        else:
            layer_fns.append(_make_attn_layer_fn(d))

    # V5 monolithic compiled function — standard residual connections
    # flat_cache order: layer 0 arr0, layer 0 arr1, layer 1 arr0, layer 1 arr1, ...
    def monolithic_decode(h, offset, *flat_cache):
        cache_list = list(flat_cache)

        for i, (ltype, _) in enumerate(layer_data):
            c0 = cache_list[i * 2]
            c1 = cache_list[i * 2 + 1]
            if ltype == 'delta':
                h, nc0, nc1 = layer_fns[i](h, c0, c1)
            else:
                h, nc0, nc1 = layer_fns[i](h, c0, c1, offset)
            cache_list[i * 2] = nc0
            cache_list[i * 2 + 1] = nc1

        h = mx.fast.rms_norm(h, fnw, fne)
        logits = mx.quantized_matmul(h, lm_head_w, lm_head_s, lm_head_b,
                                      group_size=lm_head_gs, bits=lm_head_bits)
        return (logits,) + tuple(cache_list)

    compiled_fn = mx.compile(monolithic_decode)
    return compiled_fn, layer_data


# ---------------------------------------------------------------------------
# V7: GPU-resident autoregressive loop
# ---------------------------------------------------------------------------

def _build_gpu_loop(text_model, lm_head_w, lm_head_s, lm_head_b, lm_head_gs, lm_head_bits, n_steps):
    """
    Build a compiled function that runs n_steps decode steps entirely on GPU.

    ONE CPU dispatch → embed + forward + argmax + cache update × n_steps.
    No CPU round-trips between tokens.

    Takes: (first_token_id, start_offset, *flat_cache)
    Returns: (generated_token_ids, *final_flat_cache)
    """
    layers = text_model.layers

    # Capture embedding layer (may be quantized)
    embed_fn = text_model.embed_tokens

    # Final norm + lm_head
    fnw = text_model.norm.weight
    fne = text_model.norm.eps

    # Build per-layer weight closures (same as V5)
    layer_data = []
    for i, layer in enumerate(layers):
        if layer.is_linear:
            attn = layer.linear_attn
            mlp = layer.mlp
            d = {
                'fw': attn._fused_w, 'fs': attn._fused_s, 'fb': attn._fused_bi,
                'gs': attn._fused_gs, 'bits': attn._fused_bits,
                'qkv_end': attn._qkv_end, 'z_end': attn._z_end, 'b_end': attn._b_end,
                'Hv': attn.num_v_heads, 'Hk': attn.num_k_heads,
                'Dv': attn.head_v_dim, 'Dk': attn.head_k_dim,
                'kd': attn.key_dim,
                'cw': attn._conv_w_flat,
                'ae': attn._A_exp, 'db': attn.dt_bias,
                'nw': attn.norm.weight, 'ne': attn.norm.eps,
                'ow': attn.out_proj.weight, 'os': attn.out_proj.scales,
                'ob': attn.out_proj.biases, 'ogs': attn.out_proj.group_size,
                'obits': attn.out_proj.bits,
                'lnw': layer.input_layernorm.weight,
                'lne': layer.input_layernorm.eps,
                'plnw': layer.post_attention_layernorm.weight,
                'plne': layer.post_attention_layernorm.eps,
                'fmw': None, 'fms': None, 'fmbi': None,
                'mgs': 0, 'mbits': 0, 'isz': 0,
                'dw': mlp.down_proj.weight, 'ds': mlp.down_proj.scales,
                'dbi': mlp.down_proj.biases,
                'dgs': mlp.down_proj.group_size, 'dbits': mlp.down_proj.bits,
            }
            d['fmw'] = mx.concatenate([mlp.gate_proj.weight, mlp.up_proj.weight], axis=0)
            d['fms'] = mx.concatenate([mlp.gate_proj.scales, mlp.up_proj.scales], axis=0)
            d['fmbi'] = mx.concatenate([mlp.gate_proj.biases, mlp.up_proj.biases], axis=0)
            d['mgs'] = mlp.gate_proj.group_size
            d['mbits'] = mlp.gate_proj.bits
            d['isz'] = mlp.gate_proj.weight.shape[0]
            layer_data.append(('delta', d))
        else:
            sa = layer.self_attn
            mlp = layer.mlp
            fqw = mx.concatenate([sa.q_proj.weight, sa.k_proj.weight, sa.v_proj.weight], axis=0)
            fqs = mx.concatenate([sa.q_proj.scales, sa.k_proj.scales, sa.v_proj.scales], axis=0)
            fqb = mx.concatenate([sa.q_proj.biases, sa.k_proj.biases, sa.v_proj.biases], axis=0)
            q_dim = sa.num_attention_heads * sa.head_dim * 2
            k_dim = sa.num_key_value_heads * sa.head_dim
            fmw = mx.concatenate([mlp.gate_proj.weight, mlp.up_proj.weight], axis=0)
            fms = mx.concatenate([mlp.gate_proj.scales, mlp.up_proj.scales], axis=0)
            fmbi = mx.concatenate([mlp.gate_proj.biases, mlp.up_proj.biases], axis=0)
            layer_data.append(('attn', {
                'fqw': fqw, 'fqs': fqs, 'fqb': fqb,
                'gs': sa.q_proj.group_size, 'bits': sa.q_proj.bits,
                'q_dim': q_dim, 'k_dim': k_dim,
                'nh': sa.num_attention_heads, 'nkv': sa.num_key_value_heads,
                'hd': sa.head_dim, 'scale': sa.scale,
                'qnw': sa.q_norm.weight, 'qne': sa.q_norm.eps,
                'knw': sa.k_norm.weight, 'kne': sa.k_norm.eps,
                'ow': sa.o_proj.weight, 'os': sa.o_proj.scales,
                'ob': sa.o_proj.biases, 'ogs': sa.o_proj.group_size,
                'obits': sa.o_proj.bits,
                'lnw': layer.input_layernorm.weight,
                'lne': layer.input_layernorm.eps,
                'plnw': layer.post_attention_layernorm.weight,
                'plne': layer.post_attention_layernorm.eps,
                'fmw': fmw, 'fms': fms, 'fmbi': fmbi,
                'mgs': mlp.gate_proj.group_size, 'mbits': mlp.gate_proj.bits,
                'isz': mlp.gate_proj.weight.shape[0],
                'dw': mlp.down_proj.weight, 'ds': mlp.down_proj.scales,
                'dbi': mlp.down_proj.biases,
                'dgs': mlp.down_proj.group_size, 'dbits': mlp.down_proj.bits,
                'rope_dims': 32, 'rope_base': 100000.0,
            }))

    layer_fns = []
    for i, (ltype, d) in enumerate(layer_data):
        if ltype == 'delta':
            layer_fns.append(_make_delta_layer_fn(d))
        else:
            layer_fns.append(_make_attn_layer_fn(d))

    # The GPU-resident autoregressive loop: unrolled n_steps times
    def gpu_loop(token_id, start_offset, *flat_cache):
        cache_list = list(flat_cache)
        token_ids = []
        offset = start_offset

        for step in range(n_steps):
            # 1. Embed token on GPU (handles quantized embeddings)
            h = embed_fn(token_id).reshape(1, 1, -1)

            # 2. Forward through all 64 layers
            for i, (ltype, _) in enumerate(layer_data):
                c0 = cache_list[i * 2]
                c1 = cache_list[i * 2 + 1]
                if ltype == 'delta':
                    h, nc0, nc1 = layer_fns[i](h, c0, c1)
                else:
                    h, nc0, nc1 = layer_fns[i](h, c0, c1, offset)
                cache_list[i * 2] = nc0
                cache_list[i * 2 + 1] = nc1

            # 3. Final norm + lm_head → logits
            h = mx.fast.rms_norm(h, fnw, fne)
            logits = mx.quantized_matmul(h, lm_head_w, lm_head_s, lm_head_b,
                                          group_size=lm_head_gs, bits=lm_head_bits)

            # 4. Greedy argmax on GPU
            token_id = mx.argmax(logits[:, -1, :], axis=-1).reshape(1)
            token_ids.append(token_id)

            # 5. Advance offset
            offset = offset + 1

        return (mx.concatenate(token_ids),) + tuple(cache_list)

    compiled = mx.compile(gpu_loop)
    return compiled, layer_data


def gpu_generate(model, tokenizer, prompt, max_tokens=256, verbose=True):
    """
    Generate tokens with the entire decode loop on GPU.
    ONE CPU dispatch for all tokens.
    """
    import time

    if hasattr(model, 'language_model'):
        text_model = model.language_model.model
        outer_model = model.language_model
    else:
        text_model = model.model
        outer_model = model

    # Prefill: run prompt through model normally
    prompt_tokens = tokenizer.encode(prompt)
    prompt_arr = mx.array(prompt_tokens)

    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)

    # Prefill
    t0 = time.perf_counter()
    logits = model(prompt_arr[None], cache=cache)
    mx.eval(logits)
    first_token = mx.argmax(logits[:, -1, :], axis=-1).reshape(1)
    mx.eval(first_token)
    t_prefill = time.perf_counter() - t0

    if verbose:
        prompt_tps = len(prompt_tokens) / t_prefill
        print(f"Prefill: {len(prompt_tokens)} tokens, {t_prefill*1000:.0f}ms ({prompt_tps:.0f} tok/s)")

    # Build GPU loop (or use cached version)
    if not hasattr(text_model, '_gpu_loop') or text_model._gpu_loop_steps != max_tokens:
        if verbose:
            print(f"Compiling GPU loop for {max_tokens} steps...")
        lm_head = outer_model.lm_head
        gpu_fn, gpu_layer_data = _build_gpu_loop(
            text_model,
            lm_head.weight, lm_head.scales, lm_head.biases,
            lm_head.group_size, lm_head.bits,
            n_steps=max_tokens,
        )
        # Eval fused weight arrays
        eval_arrays = []
        for _, d in gpu_layer_data:
            for key in ['fmw', 'fms', 'fmbi', 'fqw', 'fqs', 'fqb']:
                if key in d and d[key] is not None:
                    eval_arrays.append(d[key])
        if eval_arrays:
            mx.eval(*eval_arrays)
        text_model._gpu_loop = gpu_fn
        text_model._gpu_loop_steps = max_tokens
        text_model._gpu_loop_data = gpu_layer_data

    gpu_fn = text_model._gpu_loop

    # Build flat cache
    dtype = mx.bfloat16
    flat_cache = []
    for i, layer in enumerate(text_model.layers):
        if layer.is_linear:
            attn = layer.linear_attn
            flat_cache.append(cache[i][0] if cache[i][0] is not None else
                            mx.zeros((1, attn.conv_kernel_size - 1, attn.conv_dim), dtype=dtype))
            flat_cache.append(cache[i][1] if cache[i][1] is not None else
                            mx.zeros((1, attn.num_v_heads, attn.head_v_dim, attn.head_k_dim), dtype=dtype))
        else:
            sa = layer.self_attn
            nkv = sa.num_key_value_heads
            hd = sa.head_dim
            if cache[i].keys is None:
                cache[i].keys = mx.zeros((1, nkv, max_tokens + len(prompt_tokens) + 16, hd), dtype=dtype)
                cache[i].values = mx.zeros((1, nkv, max_tokens + len(prompt_tokens) + 16, hd), dtype=dtype)
            flat_cache.append(cache[i].keys)
            flat_cache.append(cache[i].values)

    # Find first attention layer index for offset
    fa_idx = next(i for i, l in enumerate(text_model.layers) if not l.is_linear)
    start_offset = mx.array(cache[fa_idx].offset)

    # ONE dispatch for all tokens
    t1 = time.perf_counter()
    results = gpu_fn(first_token, start_offset, *flat_cache)
    gen_tokens = results[0]
    mx.eval(gen_tokens)
    t2 = time.perf_counter()

    gen_time = t2 - t1
    all_token_ids = mx.concatenate([first_token, gen_tokens])
    mx.eval(all_token_ids)
    token_list = all_token_ids.tolist()

    # Truncate at EOS if present
    eos_id = tokenizer.eos_token_id
    if eos_id is not None and eos_id in token_list:
        token_list = token_list[:token_list.index(eos_id)]

    output = tokenizer.decode(token_list)

    if verbose:
        n = len(token_list)
        tps = n / gen_time
        print(f"Generate: {n} tokens, {gen_time*1000:.0f}ms ({tps:.1f} tok/s)")
        print(f"Total: {(t_prefill + gen_time)*1000:.0f}ms")

    return output


def _make_delta_layer_fn(d):
    """Create an uncompiled closure for one DeltaNet+MLP layer (V5)."""
    fw, fs, fb = d['fw'], d['fs'], d['fb']
    gs, bits = d['gs'], d['bits']
    qkv_end, z_end, b_end = d['qkv_end'], d['z_end'], d['b_end']
    Hv, Hk, Dv, Dk, kd = d['Hv'], d['Hk'], d['Dv'], d['Dk'], d['kd']
    vd = Hv * Dv
    cw = d['cw']
    ae, db = d['ae'], d['db']
    nw, ne = d['nw'], d['ne']
    ow, os_, ob, ogs, obits = d['ow'], d['os'], d['ob'], d['ogs'], d['obits']
    lnw, lne = d['lnw'], d['lne']
    plnw, plne = d['plnw'], d['plne']
    fmw, fms, fmbi = d['fmw'], d['fms'], d['fmbi']
    mgs, mbits, isz = d['mgs'], d['mbits'], d['isz']
    dw, ds, dbi, dgs, dbits = d['dw'], d['ds'], d['dbi'], d['dgs'], d['dbits']

    def fn(x, conv_state, rnn_state):
        h = mx.fast.rms_norm(x, lnw, lne)
        c = mx.quantized_matmul(h, fw, fs, fb, group_size=gs, bits=bits)
        qkv = c[..., :qkv_end]
        z = c[..., qkv_end:z_end].reshape(1, 1, Hv, Dv)
        bv = c[..., z_end:b_end]
        av = c[..., b_end:]
        co, nc = fused_conv1d_silu(conv_state, qkv, cw)
        qr = co[..., :kd].reshape(1, 1, Hk, Dk)
        kr = co[..., kd:2*kd].reshape(1, 1, Hk, Dk)
        v = co[..., 2*kd:].reshape(1, 1, Hv, Dv)
        out, nr = fused_gdn_step(qr, kr, v, av, bv, ae, db, rnn_state, None)
        out = nn.silu(z) * mx.fast.rms_norm(out, nw, ne)
        r = mx.quantized_matmul(out.reshape(1, 1, vd), ow, os_, ob, group_size=ogs, bits=obits)
        h2 = x + r
        h2n = mx.fast.rms_norm(h2, plnw, plne)
        gu = mx.quantized_matmul(h2n, fmw, fms, fmbi, group_size=mgs, bits=mbits)
        m = nn.silu(gu[..., :isz]) * gu[..., isz:]
        m = mx.quantized_matmul(m, dw, ds, dbi, group_size=dgs, bits=dbits)
        return h2 + m, nc, nr
    return fn


def _make_attn_layer_fn(d):
    """Create an uncompiled closure for one attention+MLP layer (V5)."""
    fqw, fqs, fqb = d['fqw'], d['fqs'], d['fqb']
    gs, bits = d['gs'], d['bits']
    q_dim, k_dim = d['q_dim'], d['k_dim']
    nh, nkv, hd, scale = d['nh'], d['nkv'], d['hd'], d['scale']
    qnw, qne = d['qnw'], d['qne']
    knw, kne = d['knw'], d['kne']
    ow, os_, ob, ogs, obits = d['ow'], d['os'], d['ob'], d['ogs'], d['obits']
    lnw, lne = d['lnw'], d['lne']
    plnw, plne = d['plnw'], d['plne']
    fmw, fms, fmbi = d['fmw'], d['fms'], d['fmbi']
    mgs, mbits, isz = d['mgs'], d['mbits'], d['isz']
    dw, ds, dbi, dgs, dbits = d['dw'], d['ds'], d['dbi'], d['dgs'], d['dbits']
    rope_dims, rope_base = d['rope_dims'], d['rope_base']

    def fn(x, cache_k, cache_v, offset):
        h = mx.fast.rms_norm(x, lnw, lne)
        out = mx.quantized_matmul(h, fqw, fqs, fqb, group_size=gs, bits=bits)

        q_out = out[..., :q_dim].reshape(1, 1, nh, hd * 2)
        q = q_out[..., :hd]
        gate = q_out[..., hd:].reshape(1, 1, nh * hd)
        k = out[..., q_dim:q_dim + k_dim].reshape(1, 1, nkv, hd)
        v = out[..., q_dim + k_dim:].reshape(1, 1, nkv, hd)

        q = mx.fast.rms_norm(q, qnw, qne).transpose(0, 2, 1, 3)
        k = mx.fast.rms_norm(k, knw, kne).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # RoPE with array offset
        q = mx.fast.rope(q, rope_dims, traditional=False, base=rope_base, scale=1.0, offset=offset)
        k = mx.fast.rope(k, rope_dims, traditional=False, base=rope_base, scale=1.0, offset=offset)

        # KV cache update via put_along_axis
        idx = offset.reshape(1, 1, 1, 1)
        cache_k = mx.put_along_axis(cache_k, idx, k, axis=2)
        cache_v = mx.put_along_axis(cache_v, idx, v, axis=2)

        # SDPA with mask (only attend to positions <= offset)
        max_len = cache_k.shape[2]
        positions = mx.arange(max_len)
        mask = positions[None, None, None, :] <= offset
        sdpa_out = mx.fast.scaled_dot_product_attention(
            q, cache_k, cache_v, scale=scale, mask=mask)

        # Post: gate, o_proj, residual, MLP
        flat = sdpa_out.transpose(0, 2, 1, 3).reshape(1, 1, nh * hd)
        r = mx.quantized_matmul(flat * mx.sigmoid(gate), ow, os_, ob, group_size=ogs, bits=obits)
        h2 = x + r
        h2n = mx.fast.rms_norm(h2, plnw, plne)
        gu = mx.quantized_matmul(h2n, fmw, fms, fmbi, group_size=mgs, bits=mbits)
        m = nn.silu(gu[..., :isz]) * gu[..., isz:]
        m = mx.quantized_matmul(m, dw, ds, dbi, group_size=dgs, bits=dbits)
        return h2 + m, cache_k, cache_v
    return fn


# ---------------------------------------------------------------------------
# Patched model forward for decode mode
# ---------------------------------------------------------------------------

def _patched_text_model_call(self, inputs, cache=None, input_embeddings=None):
    """
    Replacement for Qwen3_5TextModel.__call__ that uses:
    - V5 monolithic compiled decode (S=1, all 64 layers in one mx.compile)
    - Standard prefill (S>1)
    """
    if input_embeddings is not None:
        hidden_states = input_embeddings
    else:
        hidden_states = self.embed_tokens(inputs)

    if cache is None:
        cache = [None] * len(self.layers)

    S = hidden_states.shape[1]

    if S == 1 and hasattr(self, '_v5_decode'):
        # ---- V5: Monolithic decode ----
        v5 = self._v5_decode
        v5_data = self._v5_layer_data
        n_layers = len(self.layers)

        # Build flat cache from actual cache objects, initializing if needed
        flat_cache = []
        dtype = hidden_states.dtype
        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            if layer.is_linear:
                attn = layer.linear_attn
                if c[0] is None:
                    c[0] = mx.zeros((1, attn.conv_kernel_size - 1, attn.conv_dim), dtype=dtype)
                if c[1] is None:
                    c[1] = mx.zeros((1, attn.num_v_heads, attn.head_v_dim, attn.head_k_dim), dtype=dtype)
                flat_cache.append(c[0])  # conv_state
                flat_cache.append(c[1])  # rnn_state
            else:
                sa = layer.self_attn
                if c.keys is None:
                    # Pre-allocate KV cache for attention layers
                    nkv = sa.num_key_value_heads
                    hd = sa.head_dim
                    c.keys = mx.zeros((1, nkv, 256, hd), dtype=dtype)
                    c.values = mx.zeros((1, nkv, 256, hd), dtype=dtype)
                flat_cache.append(c.keys)   # KV keys
                flat_cache.append(c.values)  # KV values

        # Get offset from first attention layer's cache
        attn_offset = mx.array(cache[self.fa_idx].offset)

        # ONE compiled call for entire model (embedding→logits)
        results = v5(hidden_states, attn_offset, *flat_cache)
        logits = results[0]
        updated_cache = results[1:]

        # Write back updated cache
        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            if layer.is_linear:
                c[0] = updated_cache[i * 2]
                c[1] = updated_cache[i * 2 + 1]
            else:
                c.keys = updated_cache[i * 2]
                c.values = updated_cache[i * 2 + 1]
                c.offset += 1

        # Return logits directly — tag so outer TextModel skips lm_head
        self._v5_has_logits = True
        return logits

    else:
        # ---- Prefill mode: standard forward with masks ----
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask
        fa_mask = create_attention_mask(hidden_states, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])
        for layer, c in zip(self.layers, cache):
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden_states = layer(hidden_states, mask=mask, cache=c)

        return self.norm(hidden_states)


# ---------------------------------------------------------------------------
# Patch / Unpatch
# ---------------------------------------------------------------------------

def _patched_outer_model_call(self, inputs, cache=None, input_embeddings=None):
    """
    Replacement for TextModel.__call__ that skips lm_head when V5
    already computed logits inside the monolithic compiled function.
    """
    out = self.model(inputs, cache, input_embeddings=input_embeddings)
    if getattr(self.model, '_v5_has_logits', False):
        self.model._v5_has_logits = False
        return out  # already logits
    if self.args.tie_word_embeddings:
        out = self.model.embed_tokens.as_linear(out)
    else:
        out = self.lm_head(out)
    return out


_patched_classes = set()
_patched_text_model_classes = set()
_patched_outer_model_classes = set()


def patch_model(model):
    """
    Replace GatedDeltaNet and attention forward passes with fused/compiled kernels.

    V2: Fused DeltaNet projections + custom Metal kernels
    V3: mx.compile entire DeltaNet+MLP layers
    V4: Compiled attention pre+post blocks, fused QKV+MLP
    V5: Monolithic compile — ONE mx.compile for entire 64-layer forward
    V7: Use speculative decoding via mlx_lm's built-in draft_model support
    """
    if hasattr(model, 'language_model'):
        layers = model.language_model.model.layers
        text_model = model.language_model.model
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
        text_model = model.model
    else:
        raise ValueError("Cannot find model layers to patch")

    patched = 0
    compiled = 0
    attn_compiled = 0
    arrays_to_eval = []

    # --- V2: Fuse DeltaNet projection weights ---
    for i, layer in enumerate(layers):
        if not (hasattr(layer, 'is_linear') and layer.is_linear):
            continue
        attn = layer.linear_attn

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

    # --- V3: Compile DeltaNet+MLP layers for decode ---
    compiled_layers = {}
    mlp_arrays = []
    for i, layer in enumerate(layers):
        if not (hasattr(layer, 'is_linear') and layer.is_linear):
            continue
        fn, extra_arrays = _make_compiled_delta_layer(layer)
        compiled_layers[i] = fn
        mlp_arrays.extend(extra_arrays)
        compiled += 1

    if mlp_arrays:
        mx.eval(*mlp_arrays)

    text_model._compiled_delta_layers = compiled_layers

    # --- V4: Compile attention layers (pre-SDPA + post-SDPA) ---
    compiled_attn_pre = {}
    compiled_attn_post = {}
    attn_arrays = []
    for i, layer in enumerate(layers):
        if hasattr(layer, 'is_linear') and layer.is_linear:
            continue
        if not hasattr(layer, 'self_attn'):
            continue
        attn = layer.self_attn
        mlp = layer.mlp
        if not (hasattr(attn, 'q_proj') and hasattr(mlp, 'gate_proj')):
            continue

        pre_fn, pre_arrs = _make_compiled_attn_pre(layer)
        post_fn, post_arrs = _make_compiled_attn_post(layer)
        compiled_attn_pre[i] = pre_fn
        compiled_attn_post[i] = post_fn
        attn_arrays.extend(pre_arrs + post_arrs)
        attn_compiled += 1

    if attn_arrays:
        mx.eval(*attn_arrays)

    text_model._compiled_attn_pre = compiled_attn_pre
    text_model._compiled_attn_post = compiled_attn_post

    # --- V5: Build monolithic compiled decode ---
    if hasattr(model, 'language_model'):
        outer_model = model.language_model
    else:
        outer_model = model

    lm_head = outer_model.lm_head
    v5_arrays = []

    v5_fn, v5_layer_data = _build_monolithic_decode(
        text_model,
        lm_head.weight, lm_head.scales, lm_head.biases,
        lm_head.group_size, lm_head.bits,
    )
    for _, d in v5_layer_data:
        for key in ['fmw', 'fms', 'fmbi', 'fqw', 'fqs', 'fqb']:
            if key in d and d[key] is not None:
                v5_arrays.append(d[key])
    if v5_arrays:
        mx.eval(*v5_arrays)

    text_model._v5_decode = v5_fn
    text_model._v5_layer_data = v5_layer_data

    # --- Patch text model forward ---
    tm_cls = type(text_model)
    if tm_cls not in _patched_text_model_classes:
        tm_cls._original_call = tm_cls.__call__
        tm_cls.__call__ = _patched_text_model_call
        _patched_text_model_classes.add(tm_cls)

    # --- Patch outer model to skip lm_head for V5 monolithic ---
    om_cls = type(outer_model)
    if om_cls not in _patched_outer_model_classes:
        om_cls._original_call = om_cls.__call__
        om_cls.__call__ = _patched_outer_model_call
        _patched_outer_model_classes.add(om_cls)

    print(f"Patched {patched} GatedDeltaNet layers (V2 fused)")
    print(f"Compiled {compiled} DeltaNet+MLP layers (V3 mx.compile)")
    print(f"Compiled {attn_compiled} attention layers (V4 pre+post, fused QKV+MLP)")
    print(f"V5 monolithic decode: {len(v5_layer_data)} layers")
    return model


def unpatch_model(model):
    """Restore original forward passes."""
    if hasattr(model, 'language_model'):
        layers = model.language_model.model.layers
        text_model = model.language_model.model
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
        text_model = model.model
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

    # Restore text model
    tm_cls = type(text_model)
    if tm_cls in _patched_text_model_classes:
        tm_cls.__call__ = tm_cls._original_call
        del tm_cls._original_call
        _patched_text_model_classes.discard(tm_cls)
    for attr in ['_compiled_delta_layers', '_compiled_attn_pre', '_compiled_attn_post',
                  '_v5_decode', '_v5_layer_data', '_v5_has_logits']:
        if hasattr(text_model, attr):
            delattr(text_model, attr)

    # Restore outer model
    if hasattr(model, 'language_model'):
        outer_model = model.language_model
    elif hasattr(model, 'model'):
        outer_model = model
    else:
        outer_model = None
    if outer_model is not None:
        om_cls = type(outer_model)
        if om_cls in _patched_outer_model_classes:
            om_cls.__call__ = om_cls._original_call
            del om_cls._original_call
            _patched_outer_model_classes.discard(om_cls)

    print("Restored original forward passes")


# ---------------------------------------------------------------------------
# V7: MTP Head — Multi-Token Prediction
# ---------------------------------------------------------------------------

class MTPHead(nn.Module):
    """
    Qwen3.5 MTP head: predicts the next token from the main model's
    last hidden state + current token embedding.

    Architecture:
      1. RMSNorm hidden + RMSNorm embedding → concat → fc projection
      2. One gated-attention transformer layer (same arch as Qwen3.5 attn)
      3. RMSNorm → shared lm_head → logits
    """

    def __init__(self, hidden_size=5120, num_heads=24, num_kv_heads=4,
                 head_dim=256, intermediate_size=17408, rms_norm_eps=1e-6,
                 rope_theta=100000.0, partial_rotary_factor=0.25,
                 group_size=64, bits=4):
        super().__init__()
        self.hidden_size = hidden_size
        QL = partial(nn.QuantizedLinear, bias=False, group_size=group_size, bits=bits)

        # Pre-projection norms
        self.pre_fc_norm_hidden = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.pre_fc_norm_embedding = nn.RMSNorm(hidden_size, eps=rms_norm_eps)

        # Concat projection: [hidden; embed] → hidden
        self.fc = QL(hidden_size * 2, hidden_size)

        # Attention layer (gated Q, same as Qwen3NextAttention)
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.q_proj = QL(hidden_size, num_heads * head_dim * 2)
        self.k_proj = QL(hidden_size, num_kv_heads * head_dim)
        self.v_proj = QL(hidden_size, num_kv_heads * head_dim)
        self.o_proj = QL(num_heads * head_dim, hidden_size)
        self.q_norm = nn.RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=rms_norm_eps)

        rope_dims = int(head_dim * partial_rotary_factor)
        self.rope_dims = rope_dims
        self.rope_base = rope_theta

        # MLP
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.gate_proj = QL(hidden_size, intermediate_size)
        self.up_proj = QL(hidden_size, intermediate_size)
        self.down_proj = QL(intermediate_size, hidden_size)

        # Final norm (before shared lm_head)
        self.norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)

    def __call__(self, hidden_states, token_embedding, lm_head_fn, cache=None, offset=0):
        """
        Args:
            hidden_states: [B, 1, D] — last hidden state from main model
            token_embedding: [B, 1, D] — embedding of the current token
            lm_head_fn: callable that maps [B, 1, D] → [B, 1, vocab] logits
            cache: KVCache for this MTP attention layer (optional)
            offset: position offset for RoPE
        Returns:
            logits: [B, 1, vocab]
        """
        B, S, D = hidden_states.shape

        # 1. Norm + concat + project (embed first, hidden second — matches GGUF eh_proj)
        h_norm = self.pre_fc_norm_hidden(hidden_states)
        e_norm = self.pre_fc_norm_embedding(token_embedding)
        combined = mx.concatenate([e_norm, h_norm], axis=-1)  # [B, 1, 2D]
        h = self.fc(combined)  # [B, 1, D]

        # 2. Attention layer (gated, with RoPE)
        residual = h
        h_attn = self.input_layernorm(h)

        q_out = self.q_proj(h_attn)
        q_out = q_out.reshape(B, S, self.num_heads, self.head_dim * 2)
        queries, gate = mx.split(q_out, 2, axis=-1)
        gate = gate.reshape(B, S, -1)

        keys = self.k_proj(h_attn).reshape(B, S, self.num_kv_heads, self.head_dim)
        values = self.v_proj(h_attn).reshape(B, S, self.num_kv_heads, self.head_dim)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        # RoPE
        offset_arr = mx.array(offset)
        queries = mx.fast.rope(queries, self.rope_dims, traditional=False,
                               base=self.rope_base, scale=1.0, offset=offset_arr)
        keys = mx.fast.rope(keys, self.rope_dims, traditional=False,
                            base=self.rope_base, scale=1.0, offset=offset_arr)

        # KV cache
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        # SDPA
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale)
        output = output.transpose(0, 2, 1, 3).reshape(B, S, -1)
        h = residual + self.o_proj(output * mx.sigmoid(gate))

        # 3. MLP
        residual = h
        h_mlp = self.post_attention_layernorm(h)
        h = residual + self.down_proj(nn.silu(self.gate_proj(h_mlp)) * self.up_proj(h_mlp))

        # 4. Final norm → shared lm_head
        h_normed = self.norm(h)
        logits = lm_head_fn(h_normed)
        return logits, h  # return pre-norm hidden for chaining


def load_mtp(model, weights_path=None):
    """
    Load MTP head weights and attach to the model.

    Args:
        model: The Qwen3.5-27B model (already loaded via mlx_lm.load)
        weights_path: Path to mtp_weights.safetensors (default: ~/mlx-fork/mtp_weights.safetensors)

    Returns:
        MTPHead module, ready for inference
    """
    import os

    if weights_path is None:
        weights_path = os.path.join(os.path.dirname(__file__), "mtp_weights.safetensors")

    print(f"Loading MTP weights from {weights_path}...")
    raw = mx.load(weights_path)

    # Create MTP head
    mtp = MTPHead()

    # Map weights
    weight_map = {
        "pre_fc_norm_hidden.weight": "mtp.pre_fc_norm_hidden.weight",
        "pre_fc_norm_embedding.weight": "mtp.pre_fc_norm_embedding.weight",
        "fc.weight": "mtp.fc.weight",
        "input_layernorm.weight": "mtp.layers.0.input_layernorm.weight",
        "q_proj.weight": "mtp.layers.0.self_attn.q_proj.weight",
        "k_proj.weight": "mtp.layers.0.self_attn.k_proj.weight",
        "v_proj.weight": "mtp.layers.0.self_attn.v_proj.weight",
        "o_proj.weight": "mtp.layers.0.self_attn.o_proj.weight",
        "q_norm.weight": "mtp.layers.0.self_attn.q_norm.weight",
        "k_norm.weight": "mtp.layers.0.self_attn.k_norm.weight",
        "post_attention_layernorm.weight": "mtp.layers.0.post_attention_layernorm.weight",
        "gate_proj.weight": "mtp.layers.0.mlp.gate_proj.weight",
        "up_proj.weight": "mtp.layers.0.mlp.up_proj.weight",
        "down_proj.weight": "mtp.layers.0.mlp.down_proj.weight",
        "norm.weight": "mtp.norm.weight",
    }

    # Load into module — handle quantized weights
    pairs = []
    for local_name, mtp_key in weight_map.items():
        if mtp_key in raw:
            pairs.append((local_name, raw[mtp_key]))
        # Check for quantized version (scales/biases)
        scales_key = mtp_key.replace(".weight", ".scales")
        biases_key = mtp_key.replace(".weight", ".biases")
        if scales_key in raw:
            pairs.append((local_name.replace(".weight", ".scales"), raw[scales_key]))
        if biases_key in raw:
            pairs.append((local_name.replace(".weight", ".biases"), raw[biases_key]))

    mtp.load_weights(pairs)
    mx.eval(mtp.parameters())

    total_mb = sum(v.nbytes for _, v in pairs) / 1e6
    print(f"MTP head loaded: {total_mb:.1f} MB, {len(pairs)} tensors")

    return mtp


def mtp_generate(model, tokenizer, prompt, max_tokens=256, mtp_head=None, verbose=True):
    """
    Generate with MTP speculative decoding.

    Optimized loop:
      - Async eval for GPU pipelining
      - Lazy checkpoint: only save state when needed
      - Single eval call per step on the hot path (accept)
    """
    import time
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

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

    def fwd(token_arr):
        h = text_model.embed_tokens(token_arr)
        fa = create_attention_mask(h, cache[text_model.fa_idx])
        ss = create_ssm_mask(h, cache[text_model.ssm_idx])
        for layer, c in zip(layers, cache):
            h = layer(h, mask=(ss if layer.is_linear else fa), cache=c)
        h_pre = h
        return h_pre, lm_head(text_model.norm(h))

    def save_delta_states():
        """Save references to DeltaNet states (zero-copy, arrays are immutable)."""
        return [(c[0], c[1])
                for layer, c in zip(layers, cache) if layer.is_linear]

    def restore_delta_states(saved):
        j = 0
        for layer, c in zip(layers, cache):
            if layer.is_linear:
                c[0], c[1] = saved[j]
                j += 1
            else:
                c.offset -= 1  # trim the rejected token from KV cache

    # Prefill
    t0 = time.perf_counter()
    h_pre, logits = fwd(mx.array(prompt_tokens)[None])
    mx.eval(logits)
    token = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(token)
    t_prefill = time.perf_counter() - t0
    if verbose:
        print(f"Prefill: {len(prompt_tokens)} tokens, {t_prefill*1000:.0f}ms")

    # First MTP draft
    embed = text_model.embed_tokens(token.reshape(1)).reshape(1, 1, -1)
    draft = mx.argmax(mtp_head(h_pre[:, -1:, :], embed, lm_head)[:, -1, :], axis=-1)
    mx.eval(draft)

    generated = [token.item()]
    n_accepted = 0
    n_steps = 0

    t_gen = time.perf_counter()
    while len(generated) < max_tokens:
        n_steps += 1

        # Save DeltaNet states (attention KV caches handle trimming natively)
        delta_ckpt = save_delta_states()

        # Verify T=2: [token, draft]
        inp = mx.concatenate([token.reshape(1, 1), draft.reshape(1, 1)], axis=1)
        h2, logits2 = fwd(inp)
        verify = mx.argmax(logits2[:, 0, :], axis=-1)
        bonus = mx.argmax(logits2[:, 1, :], axis=-1)

        # Pipeline: start MTP draft for accept case while verify resolves
        embed_bonus = text_model.embed_tokens(bonus.reshape(1)).reshape(1, 1, -1)
        next_draft = mx.argmax(mtp_head(h2[:, -1:, :], embed_bonus, lm_head)[:, -1, :], axis=-1)

        # Single eval: verify + bonus + next_draft all resolve together
        mx.eval(verify, bonus, next_draft)

        if verify.item() == draft.item():
            # ACCEPT — hot path, no restore needed
            generated.append(draft.item())
            generated.append(bonus.item())
            n_accepted += 1
            token = bonus
            draft = next_draft
        else:
            # REJECT — restore DeltaNet states, trim KV caches
            restore_delta_states(delta_ckpt)

            # Re-run T=1 with correct token
            h1, logits1 = fwd(token.reshape(1, 1))
            correct = mx.argmax(logits1[:, -1, :], axis=-1)
            embed_c = text_model.embed_tokens(correct.reshape(1)).reshape(1, 1, -1)
            draft = mx.argmax(mtp_head(h1[:, -1:, :], embed_c, lm_head)[:, -1, :], axis=-1)
            mx.eval(correct, draft)
            generated.append(correct.item())
            token = correct

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
        print(f"Generate: {n} tokens, {gen_time*1000:.0f}ms ({tps:.1f} tok/s)")
        print(f"MTP acceptance: {n_accepted}/{n_steps} ({accept_rate * 100:.1f}%)" if (accept_rate := n_accepted / max(n_steps, 1)) or True else "")
        print(f"Tokens per step: {n/max(n_steps,1):.2f}")
    return output
