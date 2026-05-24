# Fresh Install — vLLM + Qwen3.5-27B from Zero

For a standard Linux machine (Ubuntu, Lambda Stack, etc.) — NOT NixOS.
NixOS needs the extra workarounds in `05-NIXOS-GUIDE.md`.

## TL;DR — use `deploy.sh`

The whole pipeline is wrapped in `../deploy.sh` (sibling of `apply.sh` in this repo). It handles every gotcha below. On a fresh GH200 box:

```bash
./deploy.sh all                      # env + pull + prep (≈ 8 min)
nohup ./deploy.sh launch > vllm.log 2>&1 &
                                     # first launch ≈ 3–5 min JIT; cached afterwards
./deploy.sh smoke                    # sanity check + MTP acceptance rate
```

Env overrides: `VENV`, `MODEL_DIR`, `HF_REPO`, `PORT`, `NUM_SPEC_TOKENS`, `HF_TOKEN`. See the header of `deploy.sh`.

### Known-good result on GH200 (Apr 2026)

| Metric | Value |
|---|---|
| Model | `j-a-a-a-y/Qwen3.5-27B-AWQ-4bit-textonly` |
| Config | `num_speculative_tokens=5`, `dtype=float16`, `max_num_seqs=4`, `max_num_batched_tokens=1024` |
| Single-request tok/s | **~192** (range 185–202) |
| TTFT (p90) | ~80 ms |
| MTP acceptance | **55.5%** overall |
| MTP acceptance per draft position | 85% / 68% / 54% / 40% / 31% |

If your numbers are materially worse, start from the gotchas table below — #1 (CPU torch) and #5 (wrong model) account for most 3× misses.


## Gotchas — read this first

Failures we've hit and their fixes, in the order you're likely to hit them:

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | `pip install vllm` pulls `torch-2.10.0+cpu` on GH200; `torch.cuda.is_available() == False` | aarch64 + vllm's default torch wheel resolves to CPU-only | `pip install --force-reinstall torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128` |
| 2 | Engine exits during `profile_cudagraph_memory` with `FileNotFoundError: 'ninja'` | flashinfer JIT needs `ninja` binary, not just the Python package | `apt-get install -y ninja-build` (or ensure venv `bin` on PATH) |
| 3 | Server sits silent for 3–5 min on first launch after "torch.compile took X s" | flashinfer JIT-compiles ~32 GDN prefill kernel variants on first run | Expected, one-time; cached under `~/.cache/flashinfer/`. Skip with `--gdn-prefill-backend triton` if you can't wait. |
| 4 | `RuntimeError: Cannot find any model weights` on `-textonly` repos | HF repo ships single `model.safetensors` but stale `model.safetensors.index.json` points to 4 shards | Rewrite the index — see snippet below Step 2 |
| 5 | `/metrics` shows `spec_decode_num_accepted_tokens_total / num_draft_tokens_total ≈ 0%` → tok/s ~1/3 of expected | Using `j-a-a-a-y/Qwen3.5-27B-AWQ-4bit-retrained-mtp` — the retrained draft head was abandoned and ships with broken weight mapping (see warnings `Parameter layers.0.mlp.down_proj.weight not found in params_dict`) | Switch to `j-a-a-a-y/Qwen3.5-27B-AWQ-4bit-textonly` (stock MTP, works) |
| 6 | Stripping vision with the script below erases the freshly-written safetensors index | Original copy-loop didn't exclude `model.safetensors.index.json` | Snippet below already fixed — keep the `.index.json` in the skip list |
| 7 | You're tempted to apply step 5 (INT8 embedding patch) on a GH200 | It's only for low-VRAM GPUs (RTX 5090 24 GB etc.) | Skip step 5 entirely on GH200 / A6000 / H100 |

**Validation checklist after launch:**

1. `curl -s http://localhost:8001/v1/models` returns your model.
2. `curl -s http://localhost:8001/metrics | grep spec_decode_num_accepted` — acceptance/(accepted+drafts) should be **≥ 40%**. If it's near zero, you're on the wrong model.
3. Run `bench-tok-s.py` — single-request throughput should be ~150–190 tok/s on GH200.

## Requirements

- NVIDIA GPU with 24+ GB VRAM (RTX 3090/4090/5090, A5000, A6000, GH200, etc.)
- CUDA 12.x drivers installed
- Python 3.10+
- ~25 GB disk for model + vLLM

> **Note on step 5 (INT8 embedding patch):** it exists to save ~1.3 GB VRAM on low-VRAM GPUs (e.g. RTX 5090, 24 GB cards). On GH200 (96 GB HBM) or A6000 (48 GB) it is **not needed** — skip step 5 entirely.

> **Note on torch wheels:** on aarch64 (GH200 Grace CPU), `pip install vllm==0.19.0` may pull a CPU-only torch. Reinstall with `pip install --force-reinstall torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128` to get the CUDA build.

## Step 1: Install vLLM (2 minutes)

```bash
python3 -m venv /opt/vllm-env
source /opt/vllm-env/bin/activate
pip install vllm==0.19.0
```

## Step 2: Download model (5-10 minutes)

**Canonical model: `j-a-a-a-y/Qwen3.5-27B-AWQ-4bit-textonly`** — W4A16 compressed-tensors 4-bit, vision already stripped, **stock working MTP head**. This is the one that hits ~186 tok/s baseline on GH200.

> **Do NOT use `j-a-a-a-y/Qwen3.5-27B-AWQ-4bit-retrained-mtp`.** The retrained MTP draft head in that variant was abandoned and ships with broken/unmapped weights. vLLM loads it with warnings like `Parameter layers.0.mlp.down_proj.weight not found in params_dict, skip loading`, then runs the draft head with zero-initialized params, producing **~0% MTP acceptance** — spec decode then pays full 5× draft overhead for no gain, dropping tok/s by ~3×. Check `/metrics` for `vllm:spec_decode_num_accepted_tokens_total / num_draft_tokens_total` — if acceptance is below ~30%, you're on the wrong model.

Alternates (all under the `j-a-a-a-y` HF account):
- `Qwen3.5-27B-AWQ-4bit-textonly` ← **use this one**, vision stripped, stock MTP works
- `Qwen3.5-27B-AWQ-4bit-retrained-mtp` ← **abandoned; do not use for inference**
- `Huihui-Qwen3.5-27B-abliterated-{AWQ,GPTQ,CT}-W4A16` — abliterated variants (alt base)
- `cyankiwi/Qwen3.5-27B-AWQ-4bit` — original community AWQ, still has vision

```bash
pip install huggingface_hub hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
# export HF_TOKEN=hf_...  # only if pulling private variants
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('j-a-a-a-y/Qwen3.5-27B-AWQ-4bit-textonly', local_dir='/opt/models/Qwen3.5-27B-AWQ')
"
```

> **Fix the stale `model.safetensors.index.json`.** The `-textonly` HF repo ships a single consolidated `model.safetensors` but its `model.safetensors.index.json` still references four shard files that were never uploaded. vLLM fails startup with `RuntimeError: Cannot find any model weights with ...`. Rewrite the index before launching:
>
> ```bash
> python3 -c "
> import json, os
> from safetensors import safe_open
> p = '/opt/models/Qwen3.5-27B-AWQ/model.safetensors'
> keys = list(safe_open(p, framework='pt').keys())
> idx = {'metadata': {'total_size': os.path.getsize(p)}, 'weight_map': {k: 'model.safetensors' for k in keys}}
> json.dump(idx, open('/opt/models/Qwen3.5-27B-AWQ/model.safetensors.index.json', 'w'), indent=2)
> "
> ```

## Step 3: Strip vision encoder (saves 0.92 GB VRAM)

```bash
python3 -c "
import os, json, shutil
from safetensors import safe_open
from safetensors.torch import save_file

src = '/opt/models/Qwen3.5-27B-AWQ'
dst = '/opt/models/Qwen3.5-27B-AWQ-textonly'
os.makedirs(dst, exist_ok=True)

all_tensors = {}
for f in sorted(os.listdir(src)):
    if not f.endswith('.safetensors'): continue
    with safe_open(os.path.join(src, f), framework='pt') as sf:
        for k in sf.keys():
            if 'visual' not in k:
                all_tensors[k] = sf.get_tensor(k)

save_file(all_tensors, os.path.join(dst, 'model.safetensors'))

total = sum(t.numel() * t.element_size() for t in all_tensors.values())
index = {'metadata': {'total_size': total}, 'weight_map': {k: 'model.safetensors' for k in all_tensors}}
with open(os.path.join(dst, 'model.safetensors.index.json'), 'w') as f:
    json.dump(index, f, indent=2)

config = json.load(open(os.path.join(src, 'config.json')))
if 'vision_config' in config: del config['vision_config']
with open(os.path.join(dst, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)

for fn in os.listdir(src):
    # NOTE: must exclude model.safetensors.index.json — we just wrote a fresh one above
    # that points to the single consolidated shard. Copying the source index back over it
    # will make vLLM fail with "Cannot find any model weights".
    if fn.endswith('.safetensors') or fn in ('config.json', 'model.safetensors.index.json'): continue
    s = os.path.join(src, fn)
    if os.path.isfile(s): shutil.copy2(s, dst)

print(f'Done: {total/1e9:.1f} GB')
"
```

## Step 4: Disable thinking in chat template

```bash
python3 -c "
path = '/opt/models/Qwen3.5-27B-AWQ-textonly/chat_template.jinja'
t = open(path).read()
if 'set enable_thinking = false' not in t:
    t = '{%- if enable_thinking is not defined %}{%- set enable_thinking = false %}{%- endif %}\n' + t
    open(path, 'w').write(t)
    print('Thinking disabled')
"
```

## Step 5: Apply INT8 embedding patch (saves 1.27 GB VRAM)

```bash
python3 -c "
import torch

# Patch VocabParallelEmbedding
f = '$(python3 -c \"import vllm; print(vllm.__path__[0])\")/model_executor/layers/vocab_parallel_embedding.py'
c = open(f).read()

if 'quantize_to_int8' not in c:
    old = '    def forward_native(self, input_):'
    new = '''    def quantize_to_int8(self):
        if hasattr(self, \"weight\") and self.weight.dtype != torch.int8:
            dev = self.weight.device
            w = self.weight.data.cpu().float()
            scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
            w_int8 = (w / scale).round().clamp(-127, 127).to(torch.int8)
            scale_f16 = scale.to(torch.float16)
            del self.weight
            torch.cuda.empty_cache()
            self.weight = torch.nn.Parameter(w_int8.to(dev), requires_grad=False)
            self._int8_scale = torch.nn.Parameter(scale_f16.to(dev), requires_grad=False)
            self._embed_int8 = True

    def forward_native(self, input_):'''
    c = c.replace(old, new)

    old2 = '        # Get the embeddings.\n        output_parallel = self.quant_method.embedding(self, masked_input.long())'
    new2 = '''        if getattr(self, \"_embed_int8\", False):
            idx = masked_input.long()
            raw = torch.nn.functional.embedding(idx, self.weight.data)
            scale = torch.nn.functional.embedding(idx, self._int8_scale)
            output_parallel = raw.to(torch.float16) * scale
        else:
            # Get the embeddings.
            output_parallel = self.quant_method.embedding(self, masked_input.long())'''
    c = c.replace(old2, new2)
    open(f, 'w').write(c)

# Patch qwen3_5.py to call quantize_to_int8 after loading
f2 = '$(python3 -c \"import vllm; print(vllm.__path__[0])\")/model_executor/models/qwen3_5.py'
c2 = open(f2).read()
if 'quantize_to_int8' not in c2:
    c2 = c2.replace(
        'return loader.load_weights(weights)',
        'result = loader.load_weights(weights)\n        if hasattr(self.model, \"embed_tokens\"):\n            self.model.embed_tokens.quantize_to_int8()\n        return result',
        1
    )
    open(f2, 'w').write(c2)

print('INT8 embedding patch applied')
"
```

## Step 6: Launch

```bash
source /opt/vllm-env/bin/activate

python3 -m vllm.entrypoints.openai.api_server \
    --model /opt/models/Qwen3.5-27B-AWQ-textonly \
    --served-model-name qwen3.5-27b \
    --host 0.0.0.0 \
    --port 8001 \
    --dtype float16 \
    --max-model-len 4096 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 1024 \
    --gpu-memory-utilization 0.98 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 5}' \
    --performance-mode interactivity \
    --enable-prefix-caching \
    --limit-mm-per-prompt '{"image": 0, "video": 0}'
```

> **First-launch warning — flashinfer GDN JIT compile (~3–5 min).** On the very first launch, flashinfer compiles ~32 CUDA kernel variants for the Gated DeltaNet (GDN) prefill path. The log sits silently at `torch.compile took X.XX s` for several minutes while `~/.cache/flashinfer/0.6.6/90a/cached_ops/gdn_prefill_sm90/` fills with `.cuda.o` objects. This is **expected, one-time, and cached** — subsequent launches start in ~30s.
>
> If you want to skip the JIT entirely, add `--gdn-prefill-backend triton` to the launch command (slightly slower prefill, no compile time).
>
> Also: the JIT needs a `ninja` binary on `PATH`. `pip install vllm` installs the `ninja` Python package, but not always the binary — `apt-get install -y ninja-build` is the safest bet. Without it the engine exits during `profile_cudagraph_memory` with `FileNotFoundError: 'ninja'`.

## Step 7: Test

```bash
curl http://localhost:8001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen3.5-27b","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## Expected Performance (varies by GPU)

| GPU | Single tok/s | Batch=4 tok/s | KV cache |
|---|---:|---:|---:|
| RTX 5090 (32 GB) | ~140 | ~450 | ~33K tokens |
| RTX 4090 (24 GB) | ~80-100 | ~250 | ~15K tokens |
| A6000 (48 GB) | ~100-120 | ~350 | ~80K tokens |
| GH200 (96 GB) | ~190 | ~500 | ~200K tokens |

## Optional: Make it a systemd service

```bash
cat > /etc/systemd/system/vllm.service <<EOF
[Unit]
Description=vLLM Qwen3.5-27B
After=network.target

[Service]
Type=simple
ExecStart=/opt/vllm-env/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model /opt/models/Qwen3.5-27B-AWQ-textonly \
    --served-model-name qwen3.5-27b \
    --host 0.0.0.0 --port 8001 --dtype float16 \
    --max-model-len 4096 --max-num-seqs 4 \
    --max-num-batched-tokens 1024 \
    --gpu-memory-utilization 0.98 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 5}' \
    --performance-mode interactivity \
    --enable-prefix-caching \
    --limit-mm-per-prompt '{"image": 0, "video": 0}'
Restart=on-failure
RestartSec=10
Environment="PYTORCH_ALLOC_CONF=expandable_segments:True"

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now vllm
```

## Troubleshooting

| Problem | Fix |
|---|---|
| OOM on startup | Lower `--gpu-memory-utilization` to 0.95 |
| Tokenizer error | Copy tokenizer_config.json from the original (non-textonly) model |
| Slow first request | Normal — CUDA graphs + torch.compile warming up |
| `qwen3_5` not recognized | Need vLLM >= 0.19.0 |
| INT8 patch OOM during quantize | The patch quantizes on CPU to avoid this. If still OOM, lower gpu-memory-utilization |
