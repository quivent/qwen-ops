# qwen-ops

Consolidated operations repository for Qwen3.5-27B MTP speculative decoding research across llama.cpp, vLLM, MLX, and quantization tooling.

Built from 10 [quivent](https://github.com/quivent) repositories.

## Results Summary

| Platform | Hardware | Best tok/s | vs Baseline | Method |
|----------|----------|-----------|-------------|--------|
| llama.cpp | M4 Max | 13.98 | 1.99x over K=1 | Chained MTP + confidence gating |
| MLX | M4 Max | 51.1 | 1.73x | Adaptive MTP chain + batch verify |
| vLLM | GH200 480GB | 1,030 | 5.54x (batch=8) | Stock MTP spec=7 |
| vLLM | GH200 480GB | 186 | baseline | Stock MTP spec=7, batch=1 |
| vLLM | RTX 5090 | 151 | -- | GPTQ W4A16 + MTP=5 |

## Directory Structure

```
qwen-ops/
├── research/
│   ├── findings/                    # What we discovered
│   │   ├── the-recipe.md            # 1.99x chained MTP recipe (llama.cpp)
│   │   ├── tensor-layout.md         # Qwen3.5 MTP tensor layout in HF vs GGUF
│   │   ├── mlx-reference.md         # MLX implementation reference
│   │   ├── TIMELINE.md              # Chronological optimization journey (M4 Max)
│   │   ├── BANDWIDTH_ANALYSIS.md    # Where the time actually goes
│   │   ├── DISPATCH_BARRIER_PROFILING.md
│   │   ├── HUIHUI_ABLITERATED.md    # Uncensored variant: conversion, MTP extraction
│   │   ├── RESEARCH_FRONTIERS.md    # Open research directions
│   │   └── modal-mtp-precision-divergence.md  # FP16/BF16/FP32 divergence analysis
│   ├── benchmarks/                  # Consolidated benchmark data
│   │   ├── vllm-patches-benchmarks.md    # GH200 vLLM results (11 patches tested)
│   │   ├── autoawq-benchmarks.md         # RTX 5090 GPTQ vs AWQ comparison
│   │   └── inference-lab-benchmarks.md   # M4 Max MLX optimization journey
│   └── designs/                     # Architecture and design docs
│       ├── per-position-heads.md    # DeepSeek V3 style multi-head design
│       ├── integration-plan.md      # llama.cpp integration plan
│       ├── modal-mtp-design.md      # Self-speculative modal architecture
│       ├── CASCADE-MTP-TRAINING.md
│       ├── CUDAGRAPH-WEIGHT-SWAP.md
│       ├── DELTANET-WEIGHT-TRANSPLANT.md
│       ├── EAGLE-PR-DESCRIPTIONS.md
│       ├── GH200-COMPUTE-BANDWIDTH-ANALYSIS.md
│       ├── MTP-TREE-CONFIG-PATH.md
│       ├── PLV-FULL-ATTN-BOUNDARIES.md
│       ├── PROPOSE-TREE-CUDAGRAPH-ANALYSIS.md
│       └── TWO-GRAPH-CUDA-DISPATCH.md
│
├── mlx/                             # Apple Silicon MLX implementation
│   ├── __init__.py
│   ├── extract_weights.py           # MTP weight extractor from HF checkpoints
│   ├── fused_kernels_t2.py          # Fused DeltaNet kernels (V2-V7)
│   ├── generate.py                  # Speculative decode generation loop
│   ├── mtp_head.py                  # MTP head implementation
│   └── pyproject.toml
│
├── llamacpp/                        # llama.cpp patches
│   ├── infrastructure/              # Core MTP patches (11 patches, ordered)
│   │   ├── 00-base.txt              # Base commit reference
│   │   ├── 0000-base-feat-memory-recurrent-state-snapshot-restore-for-D.patch
│   │   ├── 0001-base-feat-qwen3next-MTP-NextN-head-graph-builder-load-p.patch
│   │   ├── 0002-base-feat-speculative-add-MTP-single-model-speculative-.patch
│   │   ├── 0003-base-integrate-wire-rollback-snapshot-restore-into-mtp-.patch
│   │   ├── 0004-base-integrate-wire-llama_mtp_draft-execution-path.patch
│   │   ├── 01-feat-mtp-execute-MTP-draft-graph-for-qwen3next-via.patch
│   │   ├── 02-qwen35-add-MTP-draft-graph-path-mirroring-qwen3nex.patch
│   │   ├── 03-fix-mtp-end-to-end-MTP-head-load-execute-for-qwen3.patch
│   │   ├── 04-diag-mtp-name-mask-tensors-to-surface-ggml-schedul.patch
│   │   ├── 05-fix-mtp-correctly-chain-prev_hidden-across-K-draft.patch
│   │   ├── 06-fix-mtp-isolate-mtp_graph_compute-in-a-private-ggm.patch
│   │   ├── 07-feat-mtp-v1-host-side-rollback-on-rejection-via-sn.patch
│   │   ├── 08-feat-mtp-AR-re-decode-path-for-rollback-MTP_FORCE_.patch
│   │   ├── 09-wip-mtp-in-graph-AR-loop-for-T-16-verify-state-cap.patch
│   │   ├── 10-perf-mtp-batch-rollback-re-decode-into-single-T-N-.patch
│   │   └── 11-fix-mtp-rollback-re-decode-bookkeeping-id_last-fro.patch
│   ├── optimizations/               # 9 optimization variant patches
│   │   ├── 01-feat-mtp-adaptive-chain-via-top-1-probability-thre.patch
│   │   ├── 02-diag-mtp-MTP_DEBUG_VERIFY-env-var-dumps-draft-vs-t.patch
│   │   ├── 03-feat-mtp-MTP_REFRESH_EVERY-periodic-T-1-hidden-sta.patch
│   │   ├── 04-feat-mtp-predictive-hidden-draft-via-embedding-del.patch
│   │   ├── 05-feat-mtp-perturbed-head-ensemble-via-top-K-tree-fo.patch
│   │   ├── 06-feat-mtp-tree-branching-speculative-tree-path-MTP_.patch
│   │   ├── 07-fix-mtp-tree-bump-n_parallel-unified-KV-per-branch.patch
│   │   ├── 08-perf-mtp-ensemble-happy-path-skip-second-forward-p.patch
│   │   └── 09-feat-mtp-stacked-hidden-noise-ensemble-NEGATIVE.patch
│   └── tensor-mapping/              # GGUF tensor name mapping
│       ├── 01-qwen35-tensor-load.diff
│       ├── 02-qwen35-graph-tensors.diff
│       └── README.md
│
├── vllm/                            # vLLM patches and optimizations
│   ├── patches/                     # Bug fix patches for vLLM 0.19
│   │   ├── apply.sh                 # Patch application script
│   │   ├── eagle.patch
│   │   ├── gdn-inhibition-cycle.patch
│   │   ├── gdn-shadow-state.patch
│   │   ├── gh200-strip-torch-dep.patch
│   │   ├── int8-embedding.patch
│   │   ├── llmcompressor-conv1d.patch
│   │   ├── modal_mtp.patch
│   │   ├── qwen3_5-shadow-state.patch
│   │   ├── qwen3_next.patch
│   │   ├── recurrent-rollback.patch
│   │   ├── speculative-draft-override.patch
│   │   ├── speculative-dual-mode.patch
│   │   └── speculative-mtp-tree-compat.patch
│   ├── optimizations/               # Speculative decode optimization strategies
│   │   ├── adaptive_mtp.py          # Strategy 1: adaptive chain length
│   │   ├── cascade_mtp_corrective.py
│   │   ├── deltanet_adjuster.py
│   │   ├── deltanet_transplant.py
│   │   ├── deltanet_transplant_w4a16.py
│   │   ├── early_verify_probe.py
│   │   ├── enhanced_mtp_proposer.py
│   │   ├── modal_mtp.py             # Self-speculative: DeltaNet-only draft mode
│   │   ├── native_multi_head.py
│   │   ├── partial_layer_verify.py  # Strategy 3: partial-layer verification
│   │   ├── plv_bench.py
│   │   ├── plv_layer60_bench.py
│   │   ├── selective_state_snapshot.py
│   │   └── sibling_sequential.py
│   ├── microgreens/                 # Strategy 2: sibling MTP heads
│   │   ├── __init__.py
│   │   ├── mtp_clone.py             # Clone MTP head weights
│   │   ├── mtp_diversity_train.py   # Fine-tune with diversity loss
│   │   └── sibling_mtp_proposer.py  # vLLM EagleProposer integration
│   └── scripts/
│       ├── bench-tok-s.py           # 5-prompt throughput benchmark
│       ├── quantize_deltanet.py
│       └── vllm-tree-spec.sh        # Tree attention launch config
│
├── quantization/
│   └── autoawq-qwen35/             # AWQ quantization for Qwen3.5
│       ├── qwen3_5.py               # Model class (dual layer type handling)
│       ├── inject_mtp_weights.py    # Post-quantization MTP weight injector
│       ├── __init__.py.patch
│       ├── auto.py.patch
│       ├── base.py.patch
│       └── quantizer.py.patch
│
├── deploy/
│   ├── gh200/                       # NVIDIA GH200 480GB deployment
│   │   ├── deploy.sh
│   │   ├── 08-GH200-AGENT-INSTALL.md   # Step-by-step install guide
│   │   ├── 07-FRESH-INSTALL.md
│   │   └── 01-SERVER-STATUS.md
│   └── rtx5090/                     # RTX 5090 / NixOS deployment
│       ├── vllm-serve.sh
│       ├── vllm-watchdog.sh
│       ├── nixos-captain-configuration.nix
│       └── 05-NIXOS-GUIDE.md
│
├── training/                        # MTP head training scripts
│   ├── build_training_data.py       # Training data preparation
│   └── train_per_position_heads.py  # Per-position head training (DeepSeek V3 style)
│
└── validation/                      # Accuracy and performance validation
    ├── validate_draft_accuracy.py   # Draft vs full model token comparison
    ├── validate_extended.py         # Extended 100-token validation
    ├── diagnose_divergence.py       # CPU vs GPU divergence diagnosis
    ├── test_partial_skip.py         # Partial attention skip tests
    ├── bench_v7.py                  # Speculative decoding benchmark harness
    ├── extract_mtp_huihui.py        # MTP head extractor for any Qwen3.5-27B
    └── fused_gdn.py                 # Fused GDN kernel code (V2-V7)
```

## Source Repositories

| # | Repository | Domain |
|---|-----------|--------|
| 1 | [qwen-mtp-llamacpp](https://github.com/quivent/qwen-mtp-llamacpp) | llama.cpp MTP infrastructure (11 patches) |
| 2 | [qwen-mtp-optimizations](https://github.com/quivent/qwen-mtp-optimizations) | llama.cpp optimization variants (9 patches) |
| 3 | [qwen-mtp-tensors](https://github.com/quivent/qwen-mtp-tensors) | GGUF tensor naming and conversion |
| 4 | [qwen-mtp-research](https://github.com/quivent/qwen-mtp-research) | Research notes, methodology, designs |
| 5 | [mlx-qwen-mtp](https://github.com/quivent/mlx-qwen-mtp) | MLX Apple Silicon implementation |
| 6 | [modal-mtp](https://github.com/quivent/modal-mtp) | Self-speculative DeltaNet-skip drafting |
| 7 | [vllm-qwen-speculative-decode](https://github.com/quivent/vllm-qwen-speculative-decode) | vLLM speculative decode strategies |
| 8 | [vllm-qwen-patches](https://github.com/quivent/vllm-qwen-patches) | vLLM 0.19 bug fixes + deploy scripts |
| 9 | [autoawq-qwen35](https://github.com/quivent/autoawq-qwen35) | AWQ quantization support for Qwen3.5 |
| 10 | [qwen-inference-lab](https://github.com/quivent/qwen-inference-lab) | M4 Max inference optimization log |

## Key Architectural Insight

Qwen3.5-27B is a hybrid model: 48 DeltaNet (recurrent) layers + 16 full attention layers in a strict 3:1 pattern. This hybrid architecture creates unique challenges for speculative decoding:

- **DeltaNet state is irreversible** -- no algebraic undo on rejection, requiring snapshot/restore
- **Chunking vs AR kernels diverge numerically** in FP16 (red herring, not the actual bug)
- **The architecture IS the speculative schedule** -- 3 tokens of cheap recurrence, 1 token of expensive correction

## Critical Bug Found

A one-line cache-bookkeeping bug in llama.cpp's MTP speculative path caused every optimization variant to produce corrupted output while showing apparent speedups. Six agents missed it because they were all hunting for numerical bugs in forward passes. The bug was in host-side bookkeeping -- `id_last = corr` double-wrote the correction token into the cache. See `research/findings/the-recipe.md` for full writeup.

## Quick Start

**llama.cpp (M4 Max)**:
```bash
# Apply infrastructure patches, then:
MTP_CHAIN_KMAX=2 MTP_CHAIN_THRESH=0.85 \
    ./build/bin/llama-mtp-speculative -m qwen3.5-27b-q4km.gguf \
    -p "Explain photosynthesis." -n 64 -ngl 99
```

**vLLM (GH200)**:
```bash
cd vllm/patches && chmod +x apply.sh
./apply.sh all        # Apply safe patches
./apply.sh rollback   # Apply recurrent-rollback
```

**MLX (Apple Silicon)**:
```bash
cd mlx && pip install -e .
python generate.py --model Qwen/Qwen3.5-27B-4bit
```

## License

Individual files retain their original licenses (MIT or Apache-2.0) from their source repositories.
