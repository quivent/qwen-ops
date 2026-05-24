# Dispatch Barrier Profiling — Qwen3.5-27B on M4 Max

**Date**: 2026-04-05
**Method**: Batched 200 reps per op in single mx.eval (eliminates ~90us eval-barrier overhead)

## GPU Time Breakdown

```
Theoretical bandwidth:     25.1 ms (13.7 GB at 546 GB/s)
Matmul chain (pipelined):  22.4 ms (weight reads only)
Matmul + norm chain:       31.0 ms (+8.6 ms from norm dispatch barriers)
Full model:                36.1 ms (+5.1 ms from GDN, activations, reshapes)
```

## The 8.6ms Norm Barrier Overhead

- Individual norm dispatch: ~5us in isolation
- Pipelined between matmuls: ~33us per barrier (L2 coherency stall)
- 256 norm+matmul pairs × 33us = 8.6ms
- Fused rms_norm+qmv kernel saves: **2.0ms** (measured via mx.fast.metal_kernel)
- Full metallib integration expected to save more

## The 5.1ms Non-Weight, Non-Norm Overhead

### DeltaNet Layers (48 layers) — 2.86 ms

| Rank | Operation              | Per-layer | ×48 Total | Notes |
|------|------------------------|-----------|-----------|-------|
| 1    | fused_gdn_step         | 21.95 us  | 1.054 ms  | Already custom Metal kernel. Memory-bound on state tensor [1,28,128,64]=229K elements |
| 2    | fused_conv1d_silu       | 9.49 us   | 0.456 ms  | Could merge INTO gdn_step kernel |
| 3    | mlp_silu_gate          | 6.68 us   | 0.320 ms  | 3 dispatches (slice+silu+mul) → fuseable to 1 |
| 4    | projection_slices      | 5.68 us   | 0.272 ms  | Extract q,z,b,a from fused projection. Should be free pointer ops |
| 5    | splits_reshapes        | 5.54 us   | 0.266 ms  | q/k/v from conv output. Should be free |
| 6    | silu×rms_norm (gate)   | 5.33 us   | 0.256 ms  | 3 dispatches → fuseable to 1 |
| 7    | residual_add (×2)      | 2.11 us   | 0.203 ms  | Minimal |
| 8    | reshape_out            | 0.70 us   | 0.033 ms  | Negligible |

### Attention Layers (16 layers) — 0.82 ms

| Rank | Operation           | Per-layer | ×16 Total | Notes |
|------|---------------------|-----------|-----------|-------|
| 1    | kv_cache_update     | 10.83 us  | 0.173 ms  | put_along_axis |
| 2    | sdpa                | 7.68 us   | 0.123 ms  | mx.fast.scaled_dot_product_attention |
| 3    | mlp_silu_gate       | 6.43 us   | 0.103 ms  | Same as DeltaNet: 3 dispatches → 1 |
| 4    | rope_q_k            | 5.21 us   | 0.083 ms  | mx.fast.rope |
| 5    | mask_creation       | 5.16 us   | 0.083 ms  | arange + compare. Recomputed every step |
| 6    | attn_proj_slices    | 5.18 us   | 0.083 ms  | q/k/v/gate extraction |
| 7    | residual_add (×2)   | 2.25 us   | 0.072 ms  | Minimal |
| 8    | sigmoid×output      | 3.29 us   | 0.053 ms  | Gated attention output |
| 9    | pre_transpose_qkv   | 1.82 us   | 0.029 ms  | Negligible |
| 10   | transpose_reshape   | 1.30 us   | 0.021 ms  | Negligible |

### Summary

```
Measured (isolated ops):   3.68 ms
Pipeline stall overhead:   1.42 ms (dependency chain across 64 layers)
Total:                     5.10 ms
```

## Actionable Kernel Fusions

| Fusion                         | Dispatches Eliminated | Estimated Savings |
|--------------------------------|-----------------------|-------------------|
| rms_norm into matmul (done)    | 256                   | 2.0 ms measured   |
| conv+silu INTO gdn_step       | 48                    | ~0.46 ms          |
| silu_gate as single kernel     | 64                    | ~0.42 ms          |
| silu×rms_norm as single kernel | 48                    | ~0.26 ms          |
| **Total**                      | **416**               | **~3.1 ms**       |

Combined with the 2.0ms rms_norm fusion: **5.1ms total recoverable**.
This would bring the forward pass from 36.1ms → ~31.0ms.
At 1.8 tokens/step (MTP): **58.1 tok/s** (1.97× baseline).

## The 1.42ms Unaccounted Gap

This is fundamental MLX dispatch scheduling overhead for a 64-layer dependency chain. Each compiled graph node requires the runtime to:
1. Check if inputs are ready
2. Encode the Metal command
3. Submit to the GPU command queue

With ~10 graph nodes per layer × 64 layers = ~640 nodes, at ~2.2us per node = 1.42ms.
Cannot be eliminated without changing MLX's runtime or reducing graph node count.

## Key Insights

1. **Slice/reshape ops cost 5-6us each** despite being "free" pointer arithmetic. MLX schedules them as graph nodes with dispatch overhead. At 48 DeltaNet layers × 2 slices = ~0.54ms total.

2. **mx.compile partially fuses elementwise chains** (silu+mul, sigmoid+mul, add+norm) but cannot fuse across kernel types (norm+matmul, activation+matmul). The dispatch barrier between kernel types is the main optimization target.

3. **The GDN recurrence at 22us/layer is already efficient.** The state tensor (229K elements) at ~2 bytes each = 458KB per read+write cycle. At ~500 GB/s L2 bandwidth = ~0.9us theoretical. The 22us includes the actual compute (rms_norm + scale + gate + beta + state update) plus dispatch overhead.

4. **Mask creation (5us/layer × 16 attention layers = 0.08ms)** is recomputed every decode step. Could be cached or eliminated entirely for causal attention at known positions.
