# vLLM Patches Benchmark Results

Source: [vllm-qwen-patches](https://github.com/quivent/vllm-qwen-patches)

## Hardware: GH200 480GB (96GB HBM3e, 4.8 TB/s, 900 GB/s)

Baseline: **186 tok/s** (stock vLLM 0.19, MTP spec=7, batch=1)

| Optimization | tok/s | Change | Status |
|---|---:|---:|---|
| Stock MTP spec=7 (baseline) | 186 | -- | Production |
| Stock MTP spec=7, batch=8 | 1,030 | +454% agg | Production |
| Tree speculation | 27 | -85% | Working, low acceptance |
| DeltaNet self-speculative (modal_mtp) | 3.2 | -98% | Working, no CUDA graphs |
| Standalone DeltaNet draft model | 5 | -97% | 0% acceptance |
| Sibling MTP heads (weight swap) | 139 | -25% | Swap overhead |
| Adaptive MTP chain length | 186 | 0% | All positions profitable |
| DeltaNet weight transplant | 174 | -6% | Within noise |
| Partial-layer verification (layer 60) | -- | -3-7% | Not worth deploying |
| Cascade MTP (depth-trained) | 47 | -75% | Training data mismatch |

**No optimization beat baseline.** 11 patches fix real bugs in experimental vLLM features, but none outperform stock MTP.

## MTP Acceptance per Position

| Pos 1 | Pos 2 | Pos 3 | Pos 4 | Pos 5 | Pos 6 | Pos 7 |
|-------|-------|-------|-------|-------|-------|-------|
| 87% | 68% | 54% | 39% | 28% | 21% | 16% |

## Bandwidth Utilization

Batch=1: 13% of HBM3e bandwidth (4.8 TB/s theoretical).
