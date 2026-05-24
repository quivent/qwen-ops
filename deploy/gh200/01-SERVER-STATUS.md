# Server Status — What Is Running Right Now

## The Server

- **Machine**: "captain" — a NixOS PC with an NVIDIA RTX 5090 (32 GB VRAM)
- **SSH**: `ssh -p 2227 root@185.193.125.244`
- **Public URL**: Changes on reboot. Find it with:
  ```bash
  journalctl -u cloudflared-tunnel --no-pager -n 20 | grep trycloudflare.com | tail -1
  ```
- **Local URL**: `http://localhost:8001` (from SSH)
- **API**: OpenAI-compatible. Model name: `qwen3.5-27b`

## What Is Running

The server runs **Qwen3.5-27B** (a 27-billion parameter language model) quantized to 4-bit weights. It uses speculative decoding (MTP) to generate tokens faster.

### Three Persistent Services

1. **vllm.service** — The actual model server. Starts on boot. Restarts on crash.
2. **vllm-watchdog.service** — Checks health every 30 seconds. If vLLM is unresponsive for 3 minutes, it kills and restarts it.
3. **cloudflared-tunnel.service** — Makes the server accessible from the internet via a Cloudflare tunnel.

### Current Configuration

| Setting | Value |
|---|---|
| Model | `/opt/models/Qwen3.5-27B-AWQ-textonly` |
| Quantization | compressed-tensors (auto-detected, uses Marlin kernel) |
| MTP speculative tokens | 5 |
| Performance mode | interactivity |
| Prefix caching | enabled |
| Max context length | 4096 tokens |
| Max concurrent requests | 4 |
| GPU memory utilization | 97% |
| KV cache | 26,112 tokens (INT8 embeddings save ~1.27 GB) |
| Thinking/reasoning output | disabled |

### Performance

| Metric | Value |
|---|---|
| Single request, 256 tokens | ~140 tok/s |
| Single request, 512 tokens | ~143 tok/s |
| Batch of 4 requests | ~450 tok/s aggregate |
| MTP acceptance rate | ~50-53% |

## How to Check If It's Working

```bash
# From SSH:
curl http://localhost:8001/health

# Full status:
/opt/vllm-serve.sh --status

# Send a test request:
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-27b","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## How to Restart

```bash
systemctl restart vllm
```

## How to Switch Models

```bash
/opt/vllm-serve.sh gptq    # Huihui abliterated GPTQ (best single-request)
/opt/vllm-serve.sh awq     # cyankiwi AWQ (current, best batch)
```
