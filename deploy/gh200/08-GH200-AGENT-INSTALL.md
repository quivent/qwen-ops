# GH200 Install — vLLM + Qwen3.5-27B + Recurrent Rollback MTP

Agent-executable install for Lambda GH200 (ARM64 Grace + Hopper H200, 96 GB HBM3e).
Every command is copy-paste. No decisions required. No optional steps.

## What this sets up

- vLLM 0.19.0 in an isolated venv at `/opt/vllm-env`
- Qwen3.5-27B AWQ-4bit with vision encoder stripped
- MTP speculative decoding with `num_speculative_tokens=5`
- Recurrent rollback patch for O(1) GDN state restoration on rejection
- Eagle + qwen3_next patches for MTP compatibility
- INT8 embedding quantization (saves 1.27 GB)
- systemd service for auto-restart

## Prerequisites

Run `nvidia-smi` — you must see the H200 GPU with CUDA 12.x driver.
Run `python3 --version` — must be 3.10+.
If either fails, install CUDA drivers and Python first.

---

## Step 1: Create venv and install vLLM

```bash
sudo mkdir -p /opt/vllm-env /opt/models
sudo chown $USER:$USER /opt/vllm-env /opt/models
python3 -m venv /opt/vllm-env
source /opt/vllm-env/bin/activate
pip install --upgrade pip
pip install vllm==0.19.0
```

Verify: `python3 -c "import vllm; print(vllm.__version__)"` must print `0.19.0`.

## Step 2: Download model

```bash
source /opt/vllm-env/bin/activate
pip install huggingface_hub safetensors
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('cyankiwi/Qwen3.5-27B-AWQ-4bit', local_dir='/opt/models/Qwen3.5-27B-AWQ')
"
```

Verify: `ls /opt/models/Qwen3.5-27B-AWQ/*.safetensors` must list files.

## Step 3: Strip vision encoder

```bash
source /opt/vllm-env/bin/activate
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
    if fn.endswith('.safetensors') or fn == 'config.json': continue
    s = os.path.join(src, fn)
    if os.path.isfile(s): shutil.copy2(s, dst)

print(f'Done: {total/1e9:.1f} GB')
"
```

Verify: `ls /opt/models/Qwen3.5-27B-AWQ-textonly/model.safetensors` must exist.

## Step 4: Disable thinking in chat template

```bash
python3 -c "
path = '/opt/models/Qwen3.5-27B-AWQ-textonly/chat_template.jinja'
t = open(path).read()
if 'set enable_thinking = false' not in t:
    t = '{%- if enable_thinking is not defined %}{%- set enable_thinking = false %}{%- endif %}\n' + t
    open(path, 'w').write(t)
    print('Thinking disabled')
else:
    print('Already disabled')
"
```

## Step 5: Clone patches repo and apply all patches

```bash
cd /opt
git clone https://github.com/quivent/vllm-qwen-patches.git
cd /opt/vllm-qwen-patches
source /opt/vllm-env/bin/activate

# Apply in this exact order — each patch targets different files,
# except rollback which patches gdn_linear_attn.py and qwen3_5.py.
# Apply rollback LAST because it touches files that other patches also touch.

./apply.sh eagle
./apply.sh qwen3_next
./apply.sh rollback
```

Verify all three:
```bash
./apply.sh check
```
Must show `eagle: PATCHED`, `qwen3_next: PATCHED`, `rollback: PATCHED`.

## Step 6: Apply INT8 embedding patch

```bash
source /opt/vllm-env/bin/activate
cd /opt/vllm-qwen-patches

VLLM_PATH=$(python3 -c "import vllm; print(vllm.__path__[0])")

python3 -c "
import torch

f = '${VLLM_PATH}/model_executor/layers/vocab_parallel_embedding.py'
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
    print('INT8 embedding: patched')
else:
    print('INT8 embedding: already patched')

f2 = '${VLLM_PATH}/model_executor/models/qwen3_5.py'
c2 = open(f2).read()
if 'quantize_to_int8' not in c2:
    c2 = c2.replace(
        'return loader.load_weights(weights)',
        'result = loader.load_weights(weights)\n        if hasattr(self.model, \"embed_tokens\"):\n            self.model.embed_tokens.quantize_to_int8()\n        return result',
        1
    )
    open(f2, 'w').write(c2)
    print('qwen3_5.py INT8 hook: patched')
else:
    print('qwen3_5.py INT8 hook: already patched')
"
```

## Step 7: Launch vLLM

GH200 has 96 GB HBM3e. Use higher context and batch limits than consumer GPUs.

```bash
source /opt/vllm-env/bin/activate

python3 -m vllm.entrypoints.openai.api_server \
    --model /opt/models/Qwen3.5-27B-AWQ-textonly \
    --served-model-name qwen3.5-27b \
    --host 0.0.0.0 \
    --port 8001 \
    --dtype float16 \
    --max-model-len 8192 \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.95 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 5}' \
    --performance-mode interactivity \
    --enable-prefix-caching \
    --limit-mm-per-prompt '{"image": 0, "video": 0}'
```

Wait for `Started server on 0.0.0.0:8001` in the output.

## Step 8: Verify

```bash
# Health check
curl -sf http://localhost:8001/health && echo "OK" || echo "FAIL"

# Smoke test
curl -s http://localhost:8001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen3.5-27b","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":30}' \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```

Both must succeed. If the smoke test returns a coherent response, the install is complete.

## Step 9: systemd service (run on boot)

```bash
sudo tee /etc/systemd/system/vllm.service > /dev/null <<'EOF'
[Unit]
Description=vLLM Qwen3.5-27B MTP5 + Recurrent Rollback
After=network.target

[Service]
Type=simple
User=ubuntu
ExecStart=/opt/vllm-env/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model /opt/models/Qwen3.5-27B-AWQ-textonly \
    --served-model-name qwen3.5-27b \
    --host 0.0.0.0 --port 8001 --dtype float16 \
    --max-model-len 8192 --max-num-seqs 8 \
    --gpu-memory-utilization 0.95 \
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

sudo systemctl daemon-reload
sudo systemctl enable --now vllm
```

Verify: `sudo systemctl status vllm` must show `active (running)`.

---

## What recurrent rollback does

MTP speculative decoding proposes 5 draft tokens, then verifies them in one batched forward pass. Qwen3.5 has 48 GDN (Gated Delta Net) layers with non-invertible recurrent state:

```
S_{t+1} = g_t * S_t + beta_t * k_t * (v_t - k_t^T @ S_t)
```

When verification rejects at position K, the GDN state has been corrupted by tokens K+1..4. Without rollback, vLLM must recompute the full forward pass from scratch.

The recurrent-rollback patch saves O(1) state checkpoints at each draft position during verification. On rejection, it restores all 48 layers' conv_state and ssm_state in ~0.15 ms (one `copy_` per layer) instead of rerunning the forward pass.

Memory cost: ~893 MB for MTP=5 (48 layers x 6 positions x 3.1 MB per checkpoint). This is negligible on GH200's 96 GB.

## GH200 vs consumer GPU differences

| Setting | RTX 5090 (32 GB) | GH200 (96 GB) |
|---|---|---|
| `gpu-memory-utilization` | 0.97-0.98 | 0.95 (plenty of headroom) |
| `max-model-len` | 1024-4096 | 8192+ |
| `max-num-seqs` | 4 | 8 |
| Rollback memory (893 MB) | Tight fit | Negligible |
| Expected tok/s (single) | ~140 | ~190 |
| Expected tok/s (batch=4) | ~450 | ~500 |

## Troubleshooting

| Problem | Fix |
|---|---|
| `pip install vllm` fails on ARM64 | Ensure you have `python3-dev` and build tools: `sudo apt install python3-dev build-essential cmake` |
| OOM on startup | Lower `--gpu-memory-utilization` to 0.90 |
| `qwen3_5` not recognized | Wrong vLLM version. Must be exactly `0.19.0` |
| Patch fails with "already applied" | Safe to ignore — patch is idempotent |
| Patch fails with "FAILED" | Run `./apply.sh revert` then re-apply from step 5 |
| Tokenizer error on chat | Copy `tokenizer_config.json` from original model: `cp /opt/models/Qwen3.5-27B-AWQ/tokenizer_config.json /opt/models/Qwen3.5-27B-AWQ-textonly/` |
| Slow first request (30+ sec) | Normal — CUDA graphs + torch.compile warming up |
| `rollback: stock` in check | Re-run `./apply.sh rollback` — it patches two files (gdn_linear_attn.py + qwen3_5.py) |
