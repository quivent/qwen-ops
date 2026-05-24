# Research Frontiers — Qwen3.5-27B Inference Optimization

**Date**: 2026-04-05
**Hardware**: M4 Max, 128GB, 546 GB/s, 40 GPU cores, 48MB SLC

## Current State

| Configuration | tok/s | Bottleneck |
|---|---|---|
| Stock baseline | 29.5 | 13.7 GB weight reads at 80% of 546 GB/s |
| + MTP n=1 w/ rollback | 42.7 | Same bandwidth wall, 1.8 tokens per read |
| Theoretical (100% BW) | 39.8 | Can't exceed bandwidth / weight_size |

The bandwidth wall: 13.7 GB / 546 GB/s = 25.1 ms per token. We're at ~33.9ms. MTP doesn't reduce time per token — it gets more tokens per weight read.

## Frontier 1: Entropy-Coded Weights (The Big One)

4-bit quantized weights have **Shannon entropy of ~1.1-1.5 bits** — storing 2.7-3.6x more data than the information content.

| Approach | Effective bits | Weight size | Bandwidth floor | Quality |
|---|---|---|---|---|
| Stock 4-bit | 4.0 | 13.7 GB | 25.1 ms | Baseline |
| Mixed 3/4-bit | 3.5 | 12.0 GB | 22.0 ms | Minimal loss |
| Entropy-coded 4-bit | ~1.5 | 5.1 GB | 9.3 ms | **Lossless** |
| Entropy-coded mixed 3/4 | ~1.2 | 4.3 GB | 7.9 ms | Minimal loss |

**ECQ prototype**: 3.27x throughput (42.9 → 140 tok/s) on M3 Pro. Per-row rANS encoding.

**The missing piece**: Fused rANS decode + GEMV Metal kernel. Decompress weights directly into registers during the dot product. Never materialize the full weight matrix.

**References**: drxddy/ecq, MLX #3043, EntroLLM (arXiv:2505.02380), DFloat11 (arXiv:2504.11651)

## Frontier 2: Parallel MTP Predictions

Two independent MTP drafts, verified in one T=3 pass.

```
P(both correct)     = 0.79² = 0.624 → 3 tokens
P(first ok, 2nd no) = 0.79 × 0.21 = 0.166 → 2 tokens
P(first wrong)      = 0.21 → 1 token
P(both wrong)       = 0.04 (4%, NOT 20%)

E[tokens/step] = 2.414 (vs 1.79 for n=1)
T=3 verify: ~45.9 ms
Throughput: 52.6 tok/s (+23% over n=1)
```

**Break-even**: second draft only needs **22% acceptance** to beat n=1. At 79%, it's 3.6x above break-even.

**n=3 parallel**: 58.3 tok/s (2x baseline). Diminishing returns but still positive.

## Frontier 3: ANE for MTP (Zero-Cost Drafting)

The MTP head is pure attention + MLP (no DeltaNet). ANE has **its own memory bus** — doesn't steal GPU bandwidth.

- MTP compute: 0.037ms (trivial at 11 TOPS)
- MTP weight load on ANE: 10-22ms (fits within 38ms GPU step)
- Bandwidth interference: **ZERO**
- We proved CoreML conversion works (embed+norm+lm_head: 1.46ms)
- MTP head is the same op types — should convert

If MTP is free (ANE), the step time drops by ~1-3ms. Small gain alone, but enables n=2 and n=3 parallel without MTP being a bottleneck.

## Frontier 4: CPU SME Offload

M4 replaced AMX with ARM SME. 12 P-cores × ~2 TFLOPS = ~24 TFLOPS FP32.

**For MTP on CPU stream**: `mx.stream(mx.cpu)` — one line change in MLX. MTP runs on CPU while GPU does main forward. BUT CPU shares the 546 GB/s memory bus — 265MB of MTP weights steals ~0.5ms of GPU bandwidth. Net savings: ~1.6ms.

**Verdict**: Quick win (+7%) but ANE is better (zero bandwidth interference).

## Frontier 5: Die Topology

- GPU is one contiguous block (~50% of die)
- 8 LPDDR5X channels, striped across two die edges
- Weights interleaved across all 8 channels (no locality optimization possible)
- SLC: 48MB, distributed near memory controllers
- No NUMA on single-die M4 Max
- Wire distance DRAM→GPU: 3-20mm, propagation ~0.2ns (irrelevant vs 30-40ns access)

**Verdict**: No physical proximity optimization available. The die is too small and the fabric too uniform.

## Frontier 6: SLC Caching

SLC is 48-96MB. Model is 13.7GB. Only 0.7% fits.

**Depth-first scheduling** (process all tokens through one layer before moving to next) would maximize SLC reuse — but autoregressive generation prevents this (token N depends on token N-1 through all layers).

**Batched inference** is the only way to get weight reuse: B sequences × 1 weight read per layer. At batch=4: effective 25.1/4 = 6.3ms per token.

## Combined Projection

| Stack | Effective BPW | Bandwidth floor | With MTP n=2 | tok/s |
|---|---|---|---|---|
| Current (4-bit + MTP n=1) | 4.0 | 25.1 ms | 1.79 tok / 41.9 ms | 42.7 |
| + Parallel MTP n=2 | 4.0 | 25.1 ms | 2.41 tok / 45.9 ms | 52.6 |
| + Mixed 3/4-bit | 3.5 | 22.0 ms | 2.41 tok / 42.0 ms | 57.4 |
| + Entropy coding | ~1.5 | 9.3 ms | 2.41 tok / 29.3 ms | 82.3 |
| + Entropy + n=3 parallel | ~1.5 | 9.3 ms | 2.91 tok / 33.3 ms | 87.4 |

**The prize**: entropy-coded weights + parallel MTP → **80-90 tok/s**, 3x baseline.

## What We're NOT Pursuing (and Why)

- **Kernel fusion (rms_norm into matmul)**: Synthetic benchmark showed 10.4ms savings. Real model showed +0.76ms slower. mx.compile already handles the dispatch pipeline. Dead end.
- **qmv_fast kernel parameter tuning**: Tried r=8, 4 simdgroups. Both slower (occupancy). Apple's kernel is well-tuned.
- **Group size 128**: 2.3x quantization error for 2.8% speed. Bad trade.
- **GPU-resident loop**: mx.compile is a dispatch scheduler, not a kernel fuser. No gain.
- **CPU draft model**: 3716 ms/tok. No Metal acceleration on CPU.
- **DeltaLLM weight sharing**: 10-15% accuracy drop for 12-25% compression. Bad trade.

## Priority Order

1. **Parallel MTP n=2** — proven math, 22% break-even, we have the infrastructure
2. **MTP on ANE** — zero-cost drafting, enables deeper speculation
3. **Entropy-coded weights** — the transformative change, needs fused Metal kernel
4. **Mixed 3/4-bit requantization** — immediate, supported by MLX today
5. **Batched inference** — multiplicative with everything above

## Frontier 7: The 20% Bandwidth Gap Breakdown

Why 80% and not 100% of 546 GB/s:

| Source | Estimated loss |
|---|---|
| DRAM per-bank refresh cycles | 3-4% |
| Memory controller protocol overhead (activate/precharge/CAS) | 4-5% |
| SLC contention from CPU/display/IO | 2-4% |
| Scale/bias cache line waste (128B line, 2-4B used) | 1-2% |
| Kernel dispatch overhead | ~1% |
| **Total** | **~11-16%** |

The remaining 4-9% is instruction scheduling and occupancy within the kernel. Industry benchmarks show ~93% peak for optimal LPDDR5X streaming.

**Verdict**: This gap is hardware physics. DRAM refresh, protocol timing, and cache line granularity are not software-fixable. The kernel is already optimal — Apple's qmv_fast uses no threadgroup memory, pure register-based SIMD with optimal bank-parallel streaming.

## Frontier 8: Entropy Coding — The Transformative Change

**The single highest-impact finding from all research teams.**

4-bit weights store 4 bits but contain ~1.1-1.5 bits of information. rANS (asymmetric numeral system) encoding compresses losslessly with parallel per-row decoding.

**The fused decode+GEMV kernel**: Decompress weights directly into registers during the dot product. Never materialize full weights. Per-row encoding gives O(rows) parallelism.

**ECQ measured results on Apple Silicon**: 3.27x throughput (42.9 → 140 tok/s) on M3 Pro.

**Combined stack projection:**

| Configuration | Effective BPW | Weight data | BW floor | With MTP n=2 |
|---|---|---|---|---|
| Current 4-bit | 4.0 | 13.7 GB | 25.1 ms | 52.6 tok/s |
| Mixed 3/4-bit | 3.5 | 12.0 GB | 22.0 ms | 57.4 tok/s |
| Entropy-coded 4-bit | ~1.5 | 5.1 GB | 9.3 ms | 82.3 tok/s |
| Entropy + mixed 3/4 | ~1.2 | 4.3 GB | 7.9 ms | ~95 tok/s |

**Key references**: drxddy/ecq, MLX #3043, EntroLLM, DFloat11, Float8@2bits

## Frontier 9: Megakernel (Single Persistent Dispatch)

One Metal compute dispatch for the entire model forward. Eliminates all inter-kernel overhead.

- Currently: ~500+ kernel dispatches per token, each with argument buffer setup and GPU scheduler overhead
- Megakernel: 1 dispatch, software-managed weight streaming, overlapped load/compute
- Challenge: requires reimplementing the entire forward pass in Metal, outside MLX
- Reference: Jia et al. "Compiling LLMs into a Megakernel"

**Estimated gain**: 10-15% from eliminating dispatch overhead. The 2.4ms gap (33.9 - 31.4) is partially from this.

## What Will NOT Work (Confirmed Dead Ends)

From the matmul optimization team:

- **SLC-aware tiling for M=1**: Weights are read once, no reuse. Tiling changes nothing.
- **Morton/Z-order weight layout**: Harmful — breaks sequential DRAM burst reads.
- **FP16 accumulation in qmv**: 0% gain — kernel is memory-bound, not compute-bound.
- **Additional SIMD shuffle tricks**: Already optimal — no threadgroup memory used, pure register SIMD.
- **Weight layout interleaving**: ~1% possible from interleaved scale/bias. Not worth complexity.
- **CPU SLC pre-warming**: Pseudo-random SLC eviction policy makes retention unreliable.

## M5 Timeline

- M5 base: shipping now or imminent (April 2026)
- M5 Pro/Max: expected Q4 2026
- Key ML feature: GPU Neural Accelerators (40 matrix-multiply units in GPU cores, Metal 4 TensorOps)
- Expected bandwidth: ~600-650 GB/s (LPDDR5X-9600)
- Does NOT fundamentally change the story — bandwidth wall remains, just shifted 10-15%

## CORRECTION: Entropy Coding Debunked (2026-04-05)

**Empirically measured on 1.4 billion 4-bit values from Qwen3.5-27B-4bit:**

```
Shannon entropy: 3.72 bits (NOT 1.1 bits as claimed by ECQ/EntroLLM)
Compression ratio: 1.08x (NOT 3.3x)
Savings: 7% (13.7 GB → 12.7 GB)
```

The distribution is a bell curve centered on 8 (mid-range 0-15). Standard affine 4-bit quantization (scale * int + bias) produces near-uniform distributions. The 1.1-bit entropy from ECQ was measured on a DIFFERENT quantization format with learned codebooks that deliberately cluster values.

rANS on standard 4-bit saves 7%. Not worth a custom Metal kernel.

To get real entropy gains, you'd need to replace the entire quantization stack with a compression-aware quantizer (ECQ, EntQuant). That's a different project.

## CONFIRMED: MTP Head on ANE (2026-04-05)

CoreML conversion of the MTP head succeeded:
- 424.7M params, 849.5 MB in mlpackage
- **2.24 ms median latency** on ANE
- Exact output match with MLX version
- ANE has separate memory bus — zero GPU bandwidth interference
- Runs entirely within the 34ms GPU forward pass window

Files: ~/models/mtp-head-ane.mlpackage, ~/optimizations/qwen-mtp-inference/ane-mtp/

This enables the voting scheme: two ANE MTP predictions (~4ms total) for free while GPU runs the main forward. When both agree (64% of the time), skip the 38ms verification.
