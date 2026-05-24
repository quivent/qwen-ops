# NixOS Guide — How to Modify the Server

## All Configuration Lives Here

```
/etc/nixos/configuration.nix
```

After any change to this file:
```bash
nixos-rebuild switch
```

This rebuilds the system and restarts affected services.

## The vLLM Service

The service is defined in `configuration.nix` as `systemd.services.vllm`. The key part is the `ExecStart` script that contains the vLLM launch command with all flags.

To see the current config:
```bash
journalctl -u vllm --no-pager | grep 'non-default args' | tail -1
```

## Common Changes

### Change the model
In `configuration.nix`, find the `--model` flag and change the path:
```
--model /opt/models/Qwen3.5-27B-AWQ-textonly
```
Available models on disk:
- `/opt/models/Qwen3.5-27B-AWQ-textonly` (current, cyankiwi AWQ, 19.1 GB)
- `/opt/models/Huihui-Qwen3.5-27B-abliterated-W4A16` (GPTQ abliterated, 19.5 GB)
- `/opt/models/qwen3.5-27b-q4km.gguf` (for llama.cpp, 16 GB)

### Change MTP tokens
Find `"num_speculative_tokens": 5` and change the number. 5 is optimal. 7 has lower acceptance. 3 is more conservative.

### Disable MTP entirely
Remove the `--speculative-config` line.

### Change max context
Find `--max-model-len 4096` and change. Higher = fewer concurrent requests.

## NixOS Gotchas

1. **`/etc` is read-only** — you can't create systemd services directly. They must be in `configuration.nix`.
2. **`/usr/bin`, `/usr/local/bin` don't exist** — binaries are in `/nix/store/` and `/run/current-system/sw/bin/`.
3. **Dynamic linking is broken for pip-installed binaries** — things like `ptxas`, `ninja` from pip need wrappers. We wrapped them at:
   - `/opt/vllm-env/lib/python3.13/site-packages/triton/backends/nvidia/bin/ptxas` (wrapper -> `.real`)
   - `/opt/vllm-env/lib/python3.13/site-packages/triton/backends/nvidia/bin/ptxas-blackwell` (wrapper -> `.real`)
   - `/opt/vllm-env/bin/ninja` (wrapper -> `.real`)
4. **Python is 3.13** (from NixOS) but pip/venv uses this version. Some packages expect 3.10-3.12.
5. **`/bin/bash` doesn't exist** — scripts must use `#!/run/current-system/sw/bin/bash` or `#!/usr/bin/env bash`.

## Important Environment Variables

The vLLM service needs these (set in `configuration.nix`):
```
LD_LIBRARY_PATH=/nix/store/1xw5xccqqh1xw3mvd70hyil6x418wxcm-gcc-14.3.0-lib/lib:/run/opengl-driver/lib:/nix/store/ci651krm2gbzk660hbwarqihhmzv9zly-cuda-merged-12.8/lib
CC=/run/current-system/sw/bin/gcc
CPATH=/nix/store/qwb5ygz9k8gs5ql9bpxbrsrv12r1icgm-python3-3.13.12/include/python3.13:/nix/store/ci651krm2gbzk660hbwarqihhmzv9zly-cuda-merged-12.8/include
PATH=/opt/vllm-env/bin:/run/current-system/sw/bin:/nix/store/ci651krm2gbzk660hbwarqihhmzv9zly-cuda-merged-12.8/bin
```

## Patches Applied to vLLM

vLLM is installed in `/opt/vllm-env/`. After `pip install vllm==0.19.0`, these patches need to be re-applied:

1. **INT8 embeddings** (`int8-embedding.patch`): Saves 1.27 GB VRAM
   - File: `.../layers/vocab_parallel_embedding.py` — adds `quantize_to_int8()` method
   - File: `.../models/qwen3_5.py` — calls it after weight loading

2. **Eagle patch** (`eagle.patch`): Fixes tree speculation crashes on multimodal M-RoPE models
   - File: `.../v1/spec_decode/eagle.py`

Patches are in `/opt/vllm-qwen-patches/` (git repo: quivent/vllm-qwen-patches).

## How to Reinstall vLLM from Scratch

```bash
source /opt/vllm-env/bin/activate
pip install vllm==0.19.0 --force-reinstall --no-deps

# Re-apply patches
cd /opt/vllm-qwen-patches
# Apply each patch file manually (the .patch files are reference diffs,
# not always directly applicable with `patch` due to Python version differences)

# Restart
systemctl restart vllm
```
