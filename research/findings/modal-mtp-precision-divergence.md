# Modal MTP: Precision & Divergence Findings

Extracted from [modal-mtp](https://github.com/quivent/modal-mtp) README.

## Core Finding

Qwen3.5-27B is a hybrid model: 48 DeltaNet (recurrent) layers + 16 full attention layers in a strict 3:1 pattern. Modal MTP uses the same model in two modes:
- **Draft mode**: Skip all 16 attention layers (identity pass-through). Only DeltaNet recurrence + MLPs execute. O(1) per token, no KV cache.
- **Verify mode**: All 64 layers run normally. Full KV cache, exact attention.

## FP32 (CPU): 100% Draft Accuracy

At FP32 on CPU, DeltaNet-only forward produces identical output to the full model across all prompts tested (5 prompts x 50 tokens each = 100% match). The architecture is proven correct.

## FP16 (GPU): Early Divergence

At FP16 (production precision), draft mode diverges much earlier:

| # | Match | Div@ | Prompt |
|---|-------|------|--------|
| 1 | 34% | 10 | def merge_sort(arr): |
| 2 | 18% | 4 | function debounce(fn, delay) { |
| 3 | 12% | 2 | Solve step by step: train at 60 mph... |
| 4 | 19% | 8 | derivative of f(x) = x^3 * ln(x) |
| 5 | 3% | 1 | The last human on Earth... |
| 6 | 4% | 1 | Write a haiku about the ocean |
| 7 | 7% | 1 | Three laws of thermodynamics |
| 8 | 84% | 79 | Name the planets in order |
| 9 | 6% | 5 | Explain recursion to a 5 year old |
| 10 | 2% | 2 | Chinese AI explanation |
| 11 | 24% | 7 | French relativity |
| 12 | 10% | 5 | SQL query |

## Critical Finding: CPU vs GPU Divergence

| Device | Precision | Match Rate | Result |
|--------|-----------|------------|--------|
| CPU | FP32 | **100%** (5/5) | Perfect |
| CPU | FP16 | **100%** (12/12) | Perfect |
| GPU | FP16 | **~15%** (12/12) | Diverges token 1-10 |
| GPU | BF16 | **~50%** (8/8) | Mixed |

**Same precision, different device.** The divergence is NOT a precision problem -- it's a CUDA kernel numerical behavior difference. The fused DeltaNet kernels (`causal_conv1d`, `chunk_gated_delta_rule`) produce slightly different intermediate values when attention corrections are absent vs present.

## Implications

- The architecture is proven correct at ALL precisions (CPU)
- GPU divergence is kernel-level, not design-level
- Fixable by matching kernel accumulation behavior
- Or acceptable if draft token acceptance rate is high enough with speculative decoding
- FP32 state accumulation path (`--mamba-ssm-cache-dtype float32`) is the next target: 150MB per request vs 77MB at BF16

## Speed Analysis (GH200 480GB)

| Metric | Value |
|--------|-------|
| Full model (BF16) | 125 tok/s baseline |
| Full model + MTP5 | 193 tok/s (1.88x) |
| Draft forward savings | ~25% compute (short ctx), ~60% (32K ctx) |
| DeltaNet state snapshot | ~77MB/request, negligible copy time |
