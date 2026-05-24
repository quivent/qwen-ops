# qwen-mtp-tensors

> Tensor mapping and extraction for **Qwen3.5-27B's Multi-Token Prediction (MTP) head**: how the HuggingFace checkpoint lays out the MTP layer, how llama.cpp's converter needs to be taught about it, and the tensor-name conventions that thread the load → graph → execute pipeline.

This is the **tensor archaeology** repo. If you want to know exactly which weights live where in Qwen3.5's MTP layer and how to get them from HuggingFace into a working GGUF, this is the deep dive.

## The model layout

Qwen3.5-27B's HuggingFace checkpoint contains **64 layers**:

| Layer index | Type | Count | Purpose |
|---|---|---|---|
| 0–47 | DeltaNet (linear-attention recurrent) | 48 | Hybrid backbone — fixed-state recurrence |
| 48–63 | Full attention | 16 | Hybrid backbone — interleaved attention |
| **64** | **MTP head** | **1** | **Predicts the +1 token from the layer-63 hidden state** |

The MTP head is a single transformer block that takes:
- The hidden state from the main model's final layer (layer 63 output)
- The just-decoded token's embedding (concatenated, then projected via `eh_proj`)

…and produces logits for the next position. It has its own attention, FFN, norms, and uses the main model's `lm_head` for the final projection.

## HuggingFace tensor names → GGUF tensor names

The HF checkpoint stores MTP tensors under `model.mtp.*`:

| HuggingFace | GGUF | Shape |
|---|---|---|
| `model.mtp.layers.0.input_layernorm.weight` | `blk.64.nextn.attn_norm.weight` | `[hidden_size]` |
| `model.mtp.layers.0.post_attention_layernorm.weight` | `blk.64.nextn.ffn_norm.weight` | `[hidden_size]` |
| `model.mtp.layers.0.self_attn.q_proj.weight` | `blk.64.nextn.attn_q.weight` | `[q_heads*head_dim, hidden]` |
| `model.mtp.layers.0.self_attn.k_proj.weight` | `blk.64.nextn.attn_k.weight` | `[kv_heads*head_dim, hidden]` |
| `model.mtp.layers.0.self_attn.v_proj.weight` | `blk.64.nextn.attn_v.weight` | `[kv_heads*head_dim, hidden]` |
| `model.mtp.layers.0.self_attn.o_proj.weight` | `blk.64.nextn.attn_output.weight` | `[hidden, q_heads*head_dim]` |
| `model.mtp.layers.0.mlp.gate_proj.weight` | `blk.64.nextn.ffn_gate.weight` | `[ffn_dim, hidden]` |
| `model.mtp.layers.0.mlp.up_proj.weight` | `blk.64.nextn.ffn_up.weight` | `[ffn_dim, hidden]` |
| `model.mtp.layers.0.mlp.down_proj.weight` | `blk.64.nextn.ffn_down.weight` | `[hidden, ffn_dim]` |
| `model.mtp.eh_proj.weight` | `blk.64.nextn.eh_proj.weight` | `[hidden, 2*hidden]` |
| `model.mtp.shared_head.norm.weight` | `blk.64.nextn.shared_head_norm.weight` | `[hidden]` |
| `model.mtp.shared_head.head.weight` | (uses main `output.weight`) | `[vocab, hidden]` |

The `eh_proj` tensor is the key insight — it projects the **concat** of the previous-layer hidden state and the previous-token embedding back down to `hidden_size`, which is then fed through the head's attention block.

## What llama.cpp's converter was doing wrong

Out of the box, `convert_hf_to_gguf.py` silently strips MTP tensors:

```python
def modify_tensors(self, data_torch, name, bid):
    if name.startswith("mtp"):
        return  # <-- the bug
    ...
```

The fix needs five separate corrections in the conversion + load + classification pipeline:

1. **Converter (`convert_hf_to_gguf.py`)**: rewrite `mtp.layers.<k>.*` → `model.layers.<n_base+k>.*` so the tensors land in the standard layer namespace, where `n_base = num_hidden_layers` (64 for Qwen3.5-27B)
2. **Block count bump**: `Qwen3NextModel.__init__` must set `block_count = num_hidden_layers + mtp_num_hidden_layers` so the MTP layer is included in the on-disk layer table
3. **Tensor classifier (`src/llama-arch.cpp`)**: MTP tensors were classified as `LAYER_OUTPUT` (one-shot, not per-layer). Reclassify as `LAYER_REPEATING` so the loader walks them as a normal block
4. **Loader (`src/llama-model.cpp`)**: load the new tensor slots into the per-layer struct, including the post-attention norm name confusion (HF's `post_attention_layernorm` is what llama.cpp calls `attn_post_norm` *or* `ffn_norm` depending on architecture — for QWEN35 it's loaded into the `ffn_norm` slot)
5. **Hparam**: read `nextn_predict_layers` from the GGUF metadata and store it in `llama_hparams`

## Discoveries

### `mtp_use_dedicated_embeddings`
The Qwen3.5-27B checkpoint has `mtp_use_dedicated_embeddings: false`, which means the MTP head's `shared_head` LM projection uses the **main model's `output.weight`**, not its own. Initial loader code expected a dedicated `shared_head_head` tensor and crashed with a null pointer; fall back to the main `output.weight` when dedicated embeddings are disabled.

### MRoPE positions in MTP graph
Qwen3.5 uses MRoPE (multi-axis RoPE) with `rope_dimensions = [12, 12, 12, 12]`. The MTP graph uses `inp_pos_zero` (position 0 marker) which must be a 4-element tensor for `ggml_rope_multi`, not a 1-element tensor for `ggml_rope_ext`. Getting this wrong fails an assertion deep inside the RoPE op.

### Tensor name mismatch: `attn_post_norm` vs `ffn_norm`
The HF tensor `post_attention_layernorm` lives at the position llama.cpp calls `ATTN_POST_NORM` for some architectures and `FFN_NORM` for others. For the MTP layer in Qwen3.5, the converter emits it as `FFN_NORM` and the loader must read it from that slot. Loading from `ATTN_POST_NORM` produces a null tensor and a misleading "tensor not found" error pointing at the wrong field.

## What's in this repo

- `diffs/01-qwen35-tensor-load.diff` — the converter + loader + classifier corrections (extracted from infra commit `53075d24c`)
- `diffs/02-qwen35-graph-tensors.diff` — the graph-builder side, showing how each tensor is consumed (extracted from `83babcae7`)
- `docs/tensor-layout.md` — full annotated tensor map (this README expanded)

## Related repos

- **[qwen-mtp-llamacpp](https://github.com/quivent/qwen-mtp-llamacpp)** — full infrastructure patches that include this tensor work plus the runtime side
- **[qwen-mtp-optimizations](https://github.com/quivent/qwen-mtp-optimizations)** — speculative decoding variants built on top
- **[qwen-mtp-research](https://github.com/quivent/qwen-mtp-research)** — methodology, learnings, per-position-heads design

## License

MIT.
