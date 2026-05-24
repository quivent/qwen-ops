# Modal MTP: Self-Speculative Decoding for Qwen3.5 Hybrid Models

## Core Insight

Modal MTP is not a proposer — it's a model runner execution mode. The same
model runs in two modes:

- **Draft mode**: attention layers are identity pass-through. Only DeltaNet
  recurrence + MLPs execute. O(1) per token, no KV cache.
- **Verify mode**: all layers run (DeltaNet + attention). Full KV cache.

## Execution Flow

```
For each generation step:

1. VERIFY: Run full model for current position
   → Hidden states → sampler → token N
   → DeltaNet state at position N is authoritative

2. SNAPSHOT: Save DeltaNet state (conv + temporal) for all 48 layers
   → ~77MB per request, simple tensor clone

3. DRAFT: Toggle _skip_attention=True
   For k = 1..K:
     a. Run model forward with draft token (DeltaNet + MLPs only)
        → Uses model runner's existing forward path
        → DeltaNet state advances speculatively  
     b. MTP head on hidden states → draft token N+k
        → Confidence gate: stop if top-1 < threshold
   Toggle _skip_attention=False

4. RESTORE: Restore DeltaNet state from snapshot
   → State is back at position N

5. VERIFY: Run full model on all K draft tokens (batched)
   → DeltaNet state re-computed correctly N+1..N+K
   → Attention layers verify against full KV cache
   → Accept/reject

6. DeltaNet state after verify is authoritative at last accepted position
   → No additional rollback needed
```

## Why This Works

- **No separate model**: Single set of weights. Zero memory overhead for drafter.
- **No separate state**: DeltaNet state is shared. Snapshot/restore is the
  only synchronization primitive.
- **Existing infrastructure**: Draft forwards use the model runner's attention
  metadata, CUDA graphs, and scheduling. The only change is skipping attention
  computation in 16 layers.
- **DeltaNet carries context**: 48 layers of recurrent state encode the full
  sequence. Short-horizon drafts (1-5 tokens) stay coherent without attention
  corrections.

## Speed Analysis (GH200, BF16)

Per draft forward:
- Skip: 16 attention layers × KV cache lookup = saved
- Run: 48 DeltaNet updates + 64 MLPs + norms = most of the compute still runs
- Net savings: ~25% of per-token compute (attention is 25% of total FLOPS)
- But: zero KV cache memory pressure, no cache writes

For long sequences (>4K tokens):
- Attention cost grows O(N), DeltaNet stays O(1)
- Draft mode savings increase with sequence length
- At 32K context: attention is ~60% of compute → draft saves ~60%

## Implementation Plan

### Phase 1: Core mechanism
1. `_skip_attention` flag on Qwen3NextDecoderLayer ✅ DONE
2. `set_draft_mode()` on Qwen3_5Model ✅ DONE
3. DeltaNet state snapshot/restore methods on Qwen3_5Model
4. Model runner draft loop: snapshot → N draft forwards → restore → verify

### Phase 2: Integration  
5. Register `modal_mtp` in SpeculativeConfig
6. CUDA graph capture for draft-mode forward (second graph variant)
7. Warmup pass with _skip_attention=True

### Phase 3: Optimization
8. Confidence-gated chaining (adaptive chain from user's llama.cpp work)
9. Partial restore: only restore to rejection point, not full snapshot
10. Continuous drafting: eliminate snapshot/restore by tracking state deltas

## Files Modified

- `vllm/model_executor/models/qwen3_next.py` — _skip_attention ✅
- `vllm/model_executor/models/qwen3_5.py` — set_draft_mode(), snapshot/restore
- `vllm/v1/spec_decode/eagle.py` — target model reference ✅
- `vllm/v1/worker/gpu_model_runner.py` — draft execution loop
- `vllm/config/speculative.py` — modal_mtp method registration

## State Snapshot Details

Per DeltaNet layer per request:
- Conv state: (kernel_size-1, conv_dim/tp) ≈ 30K params
- Temporal state: (48, 128, 128) = 786K params
- Total: ~1.6MB per layer per request

48 layers × 1.6MB = ~77MB per request
For batch_size=32: ~2.5GB snapshot buffer

This is allocated once and reused. The snapshot is a synchronous memcpy —
negligible compared to the compute saved.
