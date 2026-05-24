# Huihui Abliterated Variant — MLX Conversion + Benchmark

Date: 2026-04-08
Model: `huihui-ai/Huihui-Qwen3.5-27B-abliterated` (uncensored fine-tune of Qwen3.5-27B via refusal-direction abliteration)
Hardware: M4 Max 128 GB, 546 GB/s bandwidth
MLX: 0.31.1

## Goal

Run the uncensored Qwen3.5-27B variant through the same optimization stack documented in this repo (MTP head + adaptive confidence chain) and quantify any cost from abliteration.

## Workflow

1. **Download fp16 source** (~52 GB, 11 safetensors shards) from HuggingFace via `hf` CLI.
2. **Convert base to MLX 4-bit** (~14 GB) via `mlx_lm.convert --hf-path ... -q --q-bits 4`.
   Stock `mlx_lm.convert` silently drops the 15 MTP head tensors during the load-and-resave step
   because the Python model class has no `mtp` field. Unknown keys → discarded without warning.
3. **Extract MTP head separately** — see `benchmarks/extract_mtp_huihui.py`. Loads the 4 fp16 shards
   containing MTP tensors via `mx.load` (handles bfloat16 natively), applies the `+1.0` norm shift
   convention, quantizes 2D weight matrices to 4-bit (group_size=64), keeps small tensors as bf16.
   Output: 265 MB standalone safetensors file, drop-in compatible with this project's `load_mtp()`.
4. **Symlink MTP weights** so existing code picks up the abliterated head without edits:
   ```
   ln -sf mtp_weights_huihui.safetensors mtp_weights.safetensors
   ```

## MTP head extraction — is it preserved by abliteration?

Yes. The Huihui fp16 repo contains all 15 MTP tensors intact:
```
mtp.fc.weight                                   (5120, 10240)
mtp.layers.0.self_attn.q_proj.weight            (12288, 5120)
mtp.layers.0.self_attn.k_proj.weight            (1024,  5120)
mtp.layers.0.self_attn.v_proj.weight            (1024,  5120)
mtp.layers.0.self_attn.o_proj.weight            (5120,  6144)
mtp.layers.0.self_attn.q_norm.weight            (256,)
mtp.layers.0.self_attn.k_norm.weight            (256,)
mtp.layers.0.mlp.{gate,up,down}_proj.weight     (17408, 5120)
mtp.layers.0.input_layernorm.weight             (5120,)
mtp.layers.0.post_attention_layernorm.weight    (5120,)
mtp.norm.weight                                 (5120,)
mtp.pre_fc_norm_{embedding,hidden}.weight       (5120,)
```
Huihui's abliteration targets the main model's refusal direction; the MTP subnetwork rides along
with whatever distribution shift the residual stream undergoes, but its own weights aren't
surgically modified.

## Benchmark — `parallel-mtp-voting/adaptive_mtp.py`

Config: `threshold=0.8, max_chain=2, max_tokens=128`. 1 warmup + 3 timed runs.
Prompt: the standard transformer-explanation prompt from `revalidate_adaptive.py`.

| Model | tok/s (mean of 3) | Tokens/step | Avg chain | Rollbacks |
|---|---|---|---|---|
| Memory anchor (vanilla, MLX 0.30.6, historical) | 52.0 | 2.51 | — | 0 |
| **Vanilla (today, current MLX)** | **51.1** | 2.29 | 1.27 | 0 / 56 |
| **Huihui abliterated (today, current MLX)** | **49.5** | 2.17 | 1.17 | 1 / 59 |

Runs: 49.4 / 49.6 / 49.7 / 49.5 (Huihui) and 50.7 / 51.1 / 51.2 / 51.1 (vanilla). Rock steady.

### Interpretation

- **MLX version drift: ~52.0 → 51.1**, about 2% on vanilla. Noise-level; the `52.0` anchor
  from the memory index is verified.
- **Abliteration cost: 51.1 → 49.5**, about 3% on the same script, same MLX, same config.
  The mechanism is visible in the stats: avg chain length drops `1.27 → 1.17` because the MTP
  head's peak confidence distribution is slightly noisier on the shifted residual stream.
  With `threshold=0.8`, more chains terminate after 1 draft instead of 2, and tokens-per-step
  falls from 2.29 to 2.17. One rollback appeared in Huihui's runs vs zero in vanilla.

A threshold sweep on Huihui (0.7, 0.75) may recover most of the gap since the MTP is still
producing useful drafts — it just falls below 0.8 confidence more often. Not done in this log.

## Side finding — ThreadLocalStream fix for `mlx_lm.server`

While setting up a daemon against the fork's `mlx_lm.server` (from `~/mlx-fork/mlx-lm/`), first
completion crashed with:
```
RuntimeError: There is no Stream(gpu, 0) in current thread.
  File "mlx-fork/mlx-lm/mlx_lm/generate.py", line 1090, in _process_prompts
    mx.eval([c.state for c in prompt_cache])
```
The fork's `generate.py` creates `generation_stream = mx.new_stream(mx.default_device())` at
**module import time on the main thread**. The HTTP server runs generation on a worker thread,
and MLX streams created via `mx.new_stream()` are bound to the creating thread. `mx.stream(...)`
context entry + `mx.eval(...)` in the worker thread fails because that thread has no device-0
default stream.

**Fix** (1 line):
```python
# before
generation_stream = mx.new_stream(mx.default_device())
# after
generation_stream = mx.ThreadLocalStream(mx.default_device())
```
`mx.ThreadLocalStream` is documented as "unique per thread" and is specifically designed for
this case. The context manager lazily materializes per-thread stream state on first access.

After the fix, `mlx_lm.server` from the fork handles chat completions cleanly.

## Files touched (outside this repo)

- `~/models/Huihui-Qwen3.5-27B-abliterated-mlx-4bit/` — converted base (14 GB)
- `~/mlx-fork/mtp_weights_huihui.safetensors` — extracted MTP head (265 MB)
- `~/mlx-fork/mtp_weights.safetensors` → symlink, toggle between vanilla/huihui
- `~/mlx-fork/mlx-lm/mlx_lm/generate.py` — ThreadLocalStream fix
- Model paths repointed in `~/mlx-fork/`: `profile_gap_v2.py`, `mtp_speculative_decode.py`,
  `mtp_v5_bench.py`, `bench_v7.py`, `mtp_batch.py`

## Raw logs

- `logs/adaptive_mtp_vanilla.log` — 51.1 tok/s run
- `logs/adaptive_mtp_huihui.log` — 49.5 tok/s run
