# Timeline: Qwen3.5-27B Inference Optimization

Chronological record of every approach tried, what worked, what failed, and why.

---

## Phase 1: Kernel Fusion (V2-V6)

### V2: Fused DeltaNet Projections + Custom Metal Kernels

**Idea:** The DeltaNet layers do 4 separate matmuls for input projection (QKV, Z, B, A). Fuse them into 1 matmul by concatenating weight matrices. Also write custom Metal kernels for conv1d+silu and the GDN recurrence step.

**What we built:**
- Fused 4 DeltaNet projections into 1 `quantized_matmul` (saves 3 GPU dispatches per layer)
- `fused_conv1d_silu` Metal kernel: conv1d + SiLU activation in one pass, with conv state shift
- `fused_gdn_step` Metal kernel: RMS norm + scale + gating (softplus + exp) + beta (sigmoid) + recurrent state update, all in one kernel
- Pre-computed `A_exp = exp(A_log)` at patch time
- Pre-flattened conv weights from `[conv_dim, 1, 4]` to `[conv_dim, 4]`

**Result:** Modest improvement on individual layer timing, but masked by other costs.

### V3: mx.compile Entire DeltaNet+MLP Layers

**Idea:** Wrap each DeltaNet layer + its MLP into a single `@mx.compile` function. After the first trace, subsequent calls replay the C++ graph without executing any Python.

**What we built:**
- Each compiled function captures all weights as closures (constants in the compiled graph)
- Also fuses MLP gate+up projections into 1 matmul
- Fixed decode shape: B=1, S=1 (no `shapeless=True` needed)

**Result:** Eliminated ~50us/layer of Python dispatch. Small but real.

### V4: Compiled Attention Layers

**Idea:** Same treatment for the 16 attention layers: compile pre-SDPA (layernorm + fused QKV + q/k norms + transpose) and post-SDPA (gate + o_proj + residual + MLP).

**What we built:**
- `_make_compiled_attn_pre`: Fuses input_layernorm + QKV projection (3 matmuls -> 1) + q/k RMS norms + head transpose
- `_make_compiled_attn_post`: Fuses gate*output + o_proj + residual + post_norm + MLP (gate+up fused) + residual
- SDPA itself stays as `mx.fast.scaled_dot_product_attention` (already optimal)

**Result:** Another incremental improvement.

### V5: Monolithic Compile -- ONE mx.compile for All 64 Layers

**Idea:** Instead of compiling each layer separately, compile the ENTIRE forward pass -- all 48 DeltaNet layers + 16 attention layers + final norm + lm_head -- into ONE `mx.compile` function. Zero Python dispatch during decode.

**What we built:**
- Single function `monolithic_decode(h, offset, *flat_cache)` -> `(logits, *updated_cache)`
- All weights captured as closures
- KV cache updates via `mx.put_along_axis` (compilable, unlike list mutation)
- RoPE with `mx.array` offset (compilable)
- Flat cache: all conv states, RNN states, KV caches as positional args

**Result:** 29.5 -> 30.0 tok/s (+1.7% async), 27.6 -> 28.9 tok/s (+4.7% sync)

**Lesson:** The async improvement is tiny because GPU is already the bottleneck. `async_eval` hides Python overhead -- CPU savings only show up in sync mode. The 4.7% sync improvement confirms we eliminated real Python overhead, but the GPU doesn't care.

**Memory cost:** 15.3 GB -> 31.7 GB (compiled graph cache eats 16 GB).

### V6: Custom Metal Fusion Kernels -- FAILED

**Idea:** Write custom Metal kernels to fuse operations that happen between matmuls: `fused_add_rms_norm` (residual add + RMS norm), `silu_mul_rms_norm` (SiLU gate * RMS norm), `dual_rms_norm` (two RMS norms in one dispatch). Use deferred residuals so layers return `(h2, m)` separately, letting the loop fuse add+norm across layer boundaries.

**What we built:**
- 3 new Metal kernels using `mx.fast.metal_kernel`
- Deferred residual pattern in the layer loop
- Target: ~192 fewer GPU launches (930 -> 738 per token)

**Result:** 28.92 tok/s (-1.5% vs stock 29.36!)

**Why it failed:** `mx.compile` already fuses elementwise ops with RMS norm into fewer dispatches. The 930 "kernel count" was operations in the traced graph, not actual GPU dispatches. After `mx.compile`, real dispatches are much fewer. Our custom kernels *broke* the fusion graph by introducing opaque metal_kernel calls that `mx.compile` can't see through.

**Key lesson:** Don't manually fuse ops that `mx.compile` already fuses. Profile actual GPU dispatches, not traced graph operations.

---

## Phase 2: Speculative Decoding

### First Attempt -- Broken Benchmark

**Setup:** Qwen3.5-0.8B-4bit as draft model, `stream_generate` for measurement.

**Result:** 26.5 tok/s. Slower than baseline.

**Conclusion at the time:** "Speculative decoding is dead for DeltaNet -- the verify step can't batch the recurrence."

**This was WRONG.** The benchmark was broken -- `stream_generate` measures differently than `generate`. We were measuring the wrong thing.

### Validation on Llama-3.3-70B

To check if our spec decode setup worked at all, tested on a standard transformer (no DeltaNet):

**Result:** 11.9 -> 20.1 tok/s (1.69x). Spec decode works fine with `mlx_lm`.

### Re-measurement with Correct Benchmark

Went back to Qwen with `mlx_lm.generate` (not `stream_generate`):

**Result:** 29.5 -> 37.6 tok/s (1.27x). It works.

**Lesson:** Measure correctly before declaring something dead. `stream_generate` has per-token Python overhead that kills speculative decoding throughput measurements.

---

## Phase 3: Kernel Optimization (qmv_fast)

### Analysis

The quantized matrix-vector multiply kernel (`qmv_fast`) dominates decode time. Stock configuration:
- 2 simdgroups x 4 rows = 8 output rows per threadgroup
- ~219-234 GB/s per individual matmul
- ~439 GB/s pipelined (80% of 546 GB/s theoretical)

### Attempt 1: results_per_simdgroup=8

**Idea:** Double the work per simdgroup to amortize launch overhead.

**Result:** -4.2%. Register pressure killed occupancy -- fewer threadgroups could run simultaneously.

### Attempt 2: num_simdgroups=4

**Idea:** More simdgroups per threadgroup for better latency hiding.

**Result:** -1.5%. Same register pressure issue.

**Process:** Built and installed a modified MLX from source to test these changes. Required rebuilding the Python bindings.

**Conclusion:** Apple's qmv_fast kernel is well-tuned. 80% bandwidth utilization is near the practical ceiling for this workload. The remaining 20% is memory controller overhead, kernel launch latency, and non-matmul compute.

---

## Phase 4: Other Dead Ends

### group_size=128 Quantization

**Idea:** Larger quantization groups = fewer scale/bias parameters = less metadata to read.

**Result:** 2.8% faster, but 2.3x quantization error. Unacceptable quality trade-off.

### GPU-Resident Autoregressive Loop (V7)

**Idea:** Unroll N decode steps inside `mx.compile`. Embed + forward + argmax + cache update all in ONE compiled graph. CPU dispatches ONCE for N tokens.

**What we built:** `_build_gpu_loop` and `gpu_generate` functions (see `fused_gdn.py`).

**Result:** ~0% gain.

**Why:** `mx.compile` is a dispatch scheduler, not a kernel fuser. It doesn't actually merge kernels across loop iterations. Each iteration still dispatches the same GPU work. The only savings would be CPU dispatch overhead, which `async_eval` already hides.

### CPU Draft Model

**Idea:** Run the 0.8B draft model on CPU cores while the main model runs on GPU.

**Result:** 3716 ms/tok on CPU. MLX's CPU backend has no Metal acceleration -- it's pure CPU BLAS. Completely unusable.

### CoreML/ANE Draft Model

**Idea:** Compile the 0.8B draft model to CoreML for Neural Engine execution.

**Problems:**
1. `coremltools` is broken on Python 3.14 (our environment)
2. DeltaNet ops (gated delta recurrence, custom conv1d patterns) are unsupported by the CoreML converter
3. Even if it worked, ANE has limited precision (float16) and unknown latency for this architecture

**Abandoned** without a working prototype.

---

## Phase 5: MTP Discovery

### Finding the Hidden Weights

Qwen3.5 was trained with Multi-Token Prediction heads, but both MLX's converter and HuggingFace transformers strip them during model conversion. The weights exist in the original safetensors on HuggingFace but get silently dropped.

**Discovery process:**
- llama.cpp's GGUF conversion preserves 4 simplified MTP tensors
- The original HuggingFace model has the full 15 weight tensors per MTP head
- Architecture had to be reverse-engineered from weight shapes

### Architecture Reverse Engineering

From weight shapes in the safetensors:
- `mtp.pre_fc_norm_hidden.weight`: [5120] -- RMS norm for hidden states
- `mtp.pre_fc_norm_embedding.weight`: [5120] -- RMS norm for token embeddings  
- `mtp.fc.weight`: [5120, 10240] (quantized) -- Projection from concatenated [embed; hidden] to hidden_size
- Full attention layer: q_proj (gated, 12288*2 output), k_proj, v_proj, o_proj, q_norm, k_norm
- Full MLP: gate_proj, up_proj, down_proj (same dimensions as main model layers)
- `mtp.norm.weight`: [5120] -- Final RMS norm before shared lm_head

**Architecture:** Norm both inputs -> concat [embed, hidden] -> FC projection -> one transformer layer (gated attention + MLP) -> norm -> shared lm_head.

### Critical Bug: Concat Order

**Bug:** Initially concatenated as `[hidden, embed]`. Produced garbage logits.

**Fix:** Must be `[embed, hidden]` to match the GGUF `eh_proj` convention. The FC projection weight matrix was trained with embeddings in the first half of the input dimension.

**How we found it:** Compared argmax outputs between our implementation and llama.cpp's MTP. Swapping the concat order made them match.

### MTP Performance

- **Overhead:** ~3ms per MTP forward pass (vs. ~34ms for main model forward)
- **Acceptance rate:** 79%
- **No draft model required** -- the MTP head is part of the model itself
- **Weight cost:** ~200 MB additional (one transformer layer + norms + projection)

---

## Phase 6: Split-Recurrence Rollback

### The Problem

When speculative decoding rejects a draft token, you need to roll back the model state. For attention layers, this is trivial -- just don't advance the KV cache offset. But DeltaNet layers have recurrent state that gets mutated during the forward pass. You have to restore the pre-speculation state.

### Attempt 1: Checkpoint + Restore + Redo

**Approach:** Before speculation, copy all DeltaNet states. On rejection, restore from copies and re-run the accepted tokens.

**Result:** 36.9 tok/s.

**Problem:** The redo costs 34ms (full forward pass) * 21% (rejection rate) = ~7ms average overhead per step.

### Key Insight: MLX Arrays Are Immutable

`mx.array` objects are immutable. When you do `state = new_state`, the old array still exists unchanged. You don't need `mx.array.copy()` -- just save the Python reference.

This makes checkpoint zero-cost: `saved = [(c[0], c[1]) for c in cache if is_delta]` is just saving pointers.

### Attempt 2: Split GDN Recurrence

**Approach:** Instead of running the full T=2 verification through the recurrence as a batch, split the GDN recurrence into per-token calls while keeping the matmuls batched (they don't depend on recurrent state).

**How it works:**
1. Run the verify batch `[token, draft]` through all matmuls (projections, MLP) as T=2 -- this is efficient
2. Run the GDN recurrence for token 0 only
3. Check if draft was accepted
4. If accepted: run GDN recurrence for token 1 with the updated state
5. If rejected: state is already correct (we only ran token 0)

**Result:** Zero-cost rollback. No checkpoint needed, no redo needed. The accepted path does exactly the same work as non-speculative, and the rejected path skips the second recurrence step entirely.

**Final throughput:** 42.7 tok/s (1.45x baseline).

---

## Summary of Results

| Approach | tok/s | Change | Notes |
|---|---|---|---|
| Stock mlx_lm | 29.5 | baseline | async_eval, greedy |
| V5 monolithic compile | 30.0 | +1.7% | 16 GB extra memory |
| V6 custom Metal kernels | 28.9 | -1.5% | Broke mx.compile fusion |
| Stock + spec decode (0.8B) | 37.6 | +27% | Requires separate draft model |
| qmv_fast r_per_sg=8 | 28.3 | -4.2% | Register pressure |
| group_size=128 | 30.3 | +2.8% | 2.3x quant error |
| GPU-resident loop (V7) | 29.5 | +0% | mx.compile is not a kernel fuser |
| MTP head | 36.9 | +25% | Self-speculative, no draft model |
| MTP + split-recurrence | **42.7** | **+45%** | Zero-cost rollback |

The biggest gains came from speculative decoding (MTP or draft model), not from kernel optimization. The GPU is memory-bandwidth-bound, running at ~80% of theoretical. The remaining headroom is in getting more tokens verified per forward pass, not in making individual forward passes faster.
