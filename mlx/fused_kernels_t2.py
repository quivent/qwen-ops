"""
Fused T=2 Metal kernels for speculative decoding verification.

These kernels process 2 tokens in a single GPU dispatch while ALSO
capturing intermediate state (after token 0) for rollback on rejection.

fused_conv1d_silu_t2:
  - Processes 2 input tokens through conv1d + SiLU in one dispatch
  - Returns: (conv_out_t0, conv_out_t1, intermediate_conv_state, final_conv_state)
  - Intermediate state = conv state after shifting in token 0

fused_gdn_step_with_intermediate:
  - Processes T=2 GDN recurrence in one dispatch
  - Returns: (output_T2, final_state, intermediate_state)
  - Intermediate state = RNN state after processing token 0

Together these reduce DeltaNet layer dispatches from 4 to 2 per layer
during T=2 verification, saving ~48 kernel launches (~1ms) for 48 layers.
"""

from typing import Tuple, Optional
import mlx.core as mx


# ---------------------------------------------------------------------------
# Kernel: fused_conv1d_silu_t2
# ---------------------------------------------------------------------------

def _make_fused_conv1d_silu_t2_kernel():
    if not mx.metal.is_available():
        return None

    # Process 2 tokens sequentially per thread.
    # Token 0: conv(state, tok0) -> out0, shift state -> mid_state
    # Token 1: conv(mid_state, tok1) -> out1, shift state -> final_state
    source = """
        uint b_idx = thread_position_in_grid.z;
        uint ch    = thread_position_in_grid.x;
        if (ch >= conv_dim) return;

        uint state_base = b_idx * 3 * conv_dim;
        uint qkv_base0  = b_idx * 2 * conv_dim + ch;        // token 0
        uint qkv_base1  = b_idx * 2 * conv_dim + conv_dim + ch;  // token 1

        // Load initial conv state [s0, s1, s2]
        float s0 = static_cast<float>(conv_state[state_base + 0 * conv_dim + ch]);
        float s1 = static_cast<float>(conv_state[state_base + 1 * conv_dim + ch]);
        float s2 = static_cast<float>(conv_state[state_base + 2 * conv_dim + ch]);
        float t0_in = static_cast<float>(qkv[qkv_base0]);
        float t1_in = static_cast<float>(qkv[qkv_base1]);

        // Load conv weights
        uint w_base = ch * 4;
        float w0 = static_cast<float>(conv_w[w_base + 0]);
        float w1 = static_cast<float>(conv_w[w_base + 1]);
        float w2 = static_cast<float>(conv_w[w_base + 2]);
        float w3 = static_cast<float>(conv_w[w_base + 3]);

        // --- Token 0: conv([s0,s1,s2], t0) ---
        float y0 = fma(s0, w0, fma(s1, w1, fma(s2, w2, t0_in * w3)));
        y0 = y0 / (1.0f + fast::exp(-y0));  // SiLU

        uint out_base0 = b_idx * conv_dim + ch;
        conv_out_0[out_base0] = static_cast<InT>(y0);

        // Shift state after token 0: [s1, s2, t0]
        // This is the intermediate state for rollback
        uint mid_base = b_idx * 3 * conv_dim;
        mid_state[mid_base + 0 * conv_dim + ch] = conv_state[state_base + 1 * conv_dim + ch];
        mid_state[mid_base + 1 * conv_dim + ch] = conv_state[state_base + 2 * conv_dim + ch];
        mid_state[mid_base + 2 * conv_dim + ch] = qkv[qkv_base0];

        // --- Token 1: conv([s1,s2,t0], t1) ---
        float y1 = fma(s1, w0, fma(s2, w1, fma(t0_in, w2, t1_in * w3)));
        y1 = y1 / (1.0f + fast::exp(-y1));  // SiLU

        uint out_base1 = b_idx * conv_dim + ch;
        conv_out_1[out_base1] = static_cast<InT>(y1);

        // Shift state after token 1: [s2, t0, t1]
        final_state[state_base + 0 * conv_dim + ch] = conv_state[state_base + 2 * conv_dim + ch];
        final_state[state_base + 1 * conv_dim + ch] = qkv[qkv_base0];
        final_state[state_base + 2 * conv_dim + ch] = qkv[qkv_base1];
    """

    return mx.fast.metal_kernel(
        name="fused_conv1d_silu_t2",
        input_names=["conv_state", "qkv", "conv_w", "conv_dim"],
        output_names=["conv_out_0", "conv_out_1", "mid_state", "final_state"],
        source=source,
    )


_fused_conv1d_silu_t2 = _make_fused_conv1d_silu_t2_kernel()


def fused_conv1d_silu_t2(
    conv_state: mx.array,   # [B, 3, conv_dim]
    qkv: mx.array,          # [B, 2, conv_dim] — both tokens concatenated
    conv_weight: mx.array,  # [conv_dim, 4] (pre-flattened)
) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
    """
    Process 2 tokens through conv1d+SiLU in a single GPU dispatch.

    Returns:
        conv_out_0: [B, 1, conv_dim] — output after token 0
        conv_out_1: [B, 1, conv_dim] — output after token 1
        mid_state:  [B, 3, conv_dim] — conv state after token 0 (for rollback)
        final_state: [B, 3, conv_dim] — conv state after token 1
    """
    B = conv_state.shape[0]
    conv_dim = conv_state.shape[2]
    dtype = qkv.dtype
    qkv_flat = qkv.reshape(B, 2 * conv_dim)  # flatten T=2
    tpg = 256
    n_groups = (conv_dim + tpg - 1) // tpg

    co0, co1, mid_st, fin_st = _fused_conv1d_silu_t2(
        inputs=[conv_state, qkv_flat, conv_weight, conv_dim],
        template=[("InT", dtype)],
        grid=(n_groups * tpg, 1, B),
        threadgroup=(tpg, 1, 1),
        output_shapes=[
            (B, conv_dim),      # conv_out_0
            (B, conv_dim),      # conv_out_1
            (B, 3, conv_dim),   # mid_state
            (B, 3, conv_dim),   # final_state
        ],
        output_dtypes=[dtype, dtype, dtype, dtype],
    )
    return (
        co0.reshape(B, 1, conv_dim),
        co1.reshape(B, 1, conv_dim),
        mid_st,
        fin_st,
    )


# ---------------------------------------------------------------------------
# Kernel: fused_gdn_step_with_intermediate
# ---------------------------------------------------------------------------

def _make_fused_gdn_step_with_intermediate_kernel():
    """
    GDN step kernel that processes exactly T=2 tokens and writes out:
      - y:         [B, 2, Hv, Dv]  — output for both tokens
      - state_out: [B, Hv, Dv, Dk] — final state (after token 1)
      - state_mid: [B, Hv, Dv, Dk] — intermediate state (after token 0)

    The T-loop runs twice. After iteration t=0, we snapshot the state
    registers into state_mid. After iteration t=1, we write state_out.
    Same parallelism structure as the original kernel.
    """
    if not mx.metal.is_available():
        return None

    source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // q_raw, k_raw: [B, 2, Hk, Dk]
        auto q_ptr = q_raw + b_idx * 2 * Hk * Dk + hk_idx * Dk;
        auto k_ptr = k_raw + b_idx * 2 * Hk * Dk + hk_idx * Dk;

        // v, y: [B, 2, Hv, Dv]
        auto v_ = v + b_idx * 2 * Hv * Dv + hv_idx * Dv;
        y += b_idx * 2 * Hv * Dv + hv_idx * Dv;

        // a, b_in: [B, 2, Hv]
        auto a_ = a + b_idx * 2 * Hv;
        auto b_ = b_in + b_idx * 2 * Hv;

        // state_in, state_out, state_mid: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;
        auto m_state = state_mid + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
            state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
        }

        constexpr float dk_inv = 1.0f / float(Dk);
        const float dk_inv_half = rsqrt(float(Dk));

        // --- Token 0 ---
        {
            float a_val = static_cast<float>(a_[hv_idx]);
            float b_val = static_cast<float>(b_[hv_idx]);

            float A_e = static_cast<float>(A_exp[hv_idx]);
            float sp_arg = a_val + static_cast<float>(dt_bias[hv_idx]);
            float sp = sp_arg > 20.0f ? sp_arg : log(1.0f + fast::exp(sp_arg));
            float g_val = fast::exp(-A_e * sp);

            float beta_val = 1.0f / (1.0f + fast::exp(-b_val));

            float q_local[n_per_t], k_local[n_per_t];
            float q_sq = 0.0f, k_sq = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                float qv = static_cast<float>(q_ptr[s_idx]);
                float kv = static_cast<float>(k_ptr[s_idx]);
                q_local[i] = qv;
                k_local[i] = kv;
                q_sq = fma(qv, qv, q_sq);
                k_sq = fma(kv, kv, k_sq);
            }
            q_sq = simd_sum(q_sq);
            k_sq = simd_sum(k_sq);

            float q_rms = rsqrt(fma(q_sq, dk_inv, 1e-6f));
            float k_rms = rsqrt(fma(k_sq, dk_inv, 1e-6f));

            float q_scale = q_rms * dk_inv;
            float k_scale = k_rms * dk_inv_half;

            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
                q_local[i] *= q_scale;
                k_local[i] *= k_scale;
                state[i] *= g_val;
                kv_mem = fma(state[i], k_local[i], kv_mem);
            }
            kv_mem = simd_sum(kv_mem);

            float delta = (static_cast<float>(v_[dv_idx]) - kv_mem) * beta_val;

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
                state[i] = fma(k_local[i], delta, state[i]);
                out = fma(state[i], q_local[i], out);
            }
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {
                y[dv_idx] = static_cast<InT>(out);
            }
        }

        // Snapshot intermediate state (after token 0)
        for (int i = 0; i < n_per_t; ++i) {
            m_state[n_per_t * dk_idx + i] = static_cast<InT>(state[i]);
        }

        // Advance pointers to token 1
        q_ptr += Hk * Dk;
        k_ptr += Hk * Dk;
        v_ += Hv * Dv;
        y += Hv * Dv;
        a_ += Hv;
        b_ += Hv;

        // --- Token 1 ---
        {
            float a_val = static_cast<float>(a_[hv_idx]);
            float b_val = static_cast<float>(b_[hv_idx]);

            float A_e = static_cast<float>(A_exp[hv_idx]);
            float sp_arg = a_val + static_cast<float>(dt_bias[hv_idx]);
            float sp = sp_arg > 20.0f ? sp_arg : log(1.0f + fast::exp(sp_arg));
            float g_val = fast::exp(-A_e * sp);

            float beta_val = 1.0f / (1.0f + fast::exp(-b_val));

            float q_local[n_per_t], k_local[n_per_t];
            float q_sq = 0.0f, k_sq = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                float qv = static_cast<float>(q_ptr[s_idx]);
                float kv = static_cast<float>(k_ptr[s_idx]);
                q_local[i] = qv;
                k_local[i] = kv;
                q_sq = fma(qv, qv, q_sq);
                k_sq = fma(kv, kv, k_sq);
            }
            q_sq = simd_sum(q_sq);
            k_sq = simd_sum(k_sq);

            float q_rms = rsqrt(fma(q_sq, dk_inv, 1e-6f));
            float k_rms = rsqrt(fma(k_sq, dk_inv, 1e-6f));

            float q_scale = q_rms * dk_inv;
            float k_scale = k_rms * dk_inv_half;

            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
                q_local[i] *= q_scale;
                k_local[i] *= k_scale;
                state[i] *= g_val;
                kv_mem = fma(state[i], k_local[i], kv_mem);
            }
            kv_mem = simd_sum(kv_mem);

            float delta = (static_cast<float>(v_[dv_idx]) - kv_mem) * beta_val;

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
                state[i] = fma(k_local[i], delta, state[i]);
                out = fma(state[i], q_local[i], out);
            }
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {
                y[dv_idx] = static_cast<InT>(out);
            }
        }

        // Write final state (after token 1)
        for (int i = 0; i < n_per_t; ++i) {
            o_state[n_per_t * dk_idx + i] = static_cast<InT>(state[i]);
        }
    """

    return mx.fast.metal_kernel(
        name="fused_gdn_step_with_intermediate",
        input_names=["q_raw", "k_raw", "v", "a", "b_in", "A_exp", "dt_bias", "state_in"],
        output_names=["y", "state_out", "state_mid"],
        source=source,
    )


_fused_gdn_step_t2 = _make_fused_gdn_step_with_intermediate_kernel()


def fused_gdn_step_with_intermediate(
    q_raw: mx.array,   # [B, 2, Hk, Dk]
    k_raw: mx.array,   # [B, 2, Hk, Dk]
    v: mx.array,       # [B, 2, Hv, Dv]
    a: mx.array,       # [B, 2, Hv]
    b: mx.array,       # [B, 2, Hv]
    A_exp: mx.array,   # [Hv]
    dt_bias: mx.array, # [Hv]
    state: mx.array,   # [B, Hv, Dv, Dk]
) -> Tuple[mx.array, mx.array, mx.array]:
    """
    Process T=2 GDN recurrence in a single dispatch, capturing intermediate state.

    Returns:
        output:     [B, 2, Hv, Dv] — output for both tokens
        final_state: [B, Hv, Dv, Dk] — state after token 1
        mid_state:   [B, Hv, Dv, Dk] — state after token 0 (for rollback)
    """
    B = q_raw.shape[0]
    Hk, Dk = q_raw.shape[2], q_raw.shape[3]
    Hv, Dv = v.shape[2], v.shape[3]
    dtype = q_raw.dtype

    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=dtype)

    output, final_state, mid_state = _fused_gdn_step_t2(
        inputs=[q_raw, k_raw, v, a, b, A_exp, dt_bias, state],
        template=[("InT", dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[
            (B, 2, Hv, Dv),     # y
            (B, Hv, Dv, Dk),    # state_out (final)
            (B, Hv, Dv, Dk),    # state_mid (intermediate)
        ],
        output_dtypes=[dtype, dtype, dtype],
    )
    return output, final_state, mid_state


# ---------------------------------------------------------------------------
# Fused RMS Norm + Quantized Matmul (rms_norm_qmv)
# ---------------------------------------------------------------------------
# Eliminates the dispatch barrier between rms_norm and quantized_matmul.
# Saves ~2ms per forward pass (8.6ms total norm barrier overhead, partial fusion).
#
# Two-pass approach within a single kernel dispatch:
#   Pass 1: compute sum(x^2) across all input elements (x in L2, ~0.01ms)
#   Pass 2: load x * norm_weight * rms_inv, then normal qdot with quantized weights
#
# Measured results:
#   Separate (256 norm+matmul pairs): 30.9 ms
#   Fused (256 fused kernels):        28.9 ms
#   Saved: 2.0 ms (6.6%) — lower bound via mx.fast.metal_kernel
#   Full MLX integration expected to save more (stock kernel is faster per-dispatch)

def _make_fused_rms_norm_qmv_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        constexpr int RPG = 4;
        constexpr int VPT = 16;
        constexpr int BS = VPT * 32;
        uint tid_tg = thread_position_in_threadgroup.x;
        uint sg = tid_tg / 32;
        uint sl = tid_tg % 32;
        uint tg_y = thread_position_in_grid.y;
        uint out_row = tg_y * 8 + sg * RPG;
        if (out_row >= out_vec_size) return;
        uint in_w = in_vec_size / 8;
        uint in_g = in_vec_size / group_size_val;
        uint scale_step = group_size_val / VPT;

        // Pass 1: sum(x^2) for RMS
        const device InT* x_scan = x + sl * VPT;
        float sq_sum = 0;
        for (uint k = 0; k < in_vec_size; k += BS) {
            for (int i = 0; i < VPT; i++) {
                float v = static_cast<float>(x_scan[i]);
                sq_sum += v * v;
            }
            x_scan += BS;
        }
        sq_sum = simd_sum(sq_sum);
        float rms_inv = rsqrt(sq_sum / float(in_vec_size) + norm_eps);

        // Pass 2: normalized qmv
        const device uint32_t* ws = w + out_row * in_w + sl * 2;
        const device InT* sc = scales + out_row * in_g;
        const device InT* bi = biases + out_row * in_g;
        const device InT* xp = x + sl * VPT;
        const device InT* nw = norm_weight + sl * VPT;
        float result[RPG] = {0,0,0,0};

        for (uint k = 0; k < in_vec_size; k += BS) {
            float xr[VPT]; float xsum = 0;
            for (int i = 0; i < VPT; i += 4) {
                float x0 = static_cast<float>(xp[i])   * static_cast<float>(nw[i])   * rms_inv;
                float x1 = static_cast<float>(xp[i+1]) * static_cast<float>(nw[i+1]) * rms_inv;
                float x2 = static_cast<float>(xp[i+2]) * static_cast<float>(nw[i+2]) * rms_inv;
                float x3 = static_cast<float>(xp[i+3]) * static_cast<float>(nw[i+3]) * rms_inv;
                xsum += x0+x1+x2+x3;
                xr[i]=x0; xr[i+1]=x1/16.0f; xr[i+2]=x2/256.0f; xr[i+3]=x3/4096.0f;
            }
            for (int row = 0; row < RPG; row++) {
                if (out_row + row >= out_vec_size) break;
                const device uint16_t* wp = (const device uint16_t*)(ws + row * in_w);
                float s = static_cast<float>(sc[row * in_g + sl / scale_step]);
                float b = static_cast<float>(bi[row * in_g + sl / scale_step]);
                float accum = 0;
                for (int i = 0; i < VPT/4; i++) {
                    accum += xr[4*i]*float(wp[i]&0x000fu) + xr[4*i+1]*float(wp[i]&0x00f0u)
                           + xr[4*i+2]*float(wp[i]&0x0f00u) + xr[4*i+3]*float(wp[i]&0xf000u);
                }
                result[row] += s * accum + xsum * b;
            }
            ws += BS/8; sc += BS/group_size_val; bi += BS/group_size_val;
            xp += BS; nw += BS;
        }
        for (int row = 0; row < RPG; row++) {
            result[row] = simd_sum(result[row]);
            if (sl == 0 && out_row + row < out_vec_size)
                y[out_row + row] = static_cast<InT>(result[row]);
        }
    """

    return mx.fast.metal_kernel(
        name="fused_rms_norm_qmv",
        input_names=["w", "scales", "biases", "x", "norm_weight", "norm_eps",
                     "in_vec_size", "out_vec_size", "group_size_val"],
        output_names=["y"],
        source=source,
    )


_fused_rms_norm_qmv = _make_fused_rms_norm_qmv_kernel()


def fused_rms_norm_qmv(
    x: mx.array,            # [D] input vector (not normed)
    norm_weight: mx.array,   # [D] RMS norm weight
    norm_eps: float,
    w: mx.array,             # quantized weight matrix
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    out_dim: int,
) -> mx.array:
    """Fused RMS norm + quantized matmul in one kernel dispatch."""
    in_dim = x.shape[-1]
    n_tg = (out_dim + 7) // 8
    result, = _fused_rms_norm_qmv(
        inputs=[w, scales, biases, x.reshape(-1), norm_weight, norm_eps,
                in_dim, out_dim, group_size],
        template=[("InT", x.dtype)],
        grid=(64, n_tg, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(out_dim,)],
        output_dtypes=[x.dtype],
    )
    return result.reshape(1, 1, out_dim)
