#!/usr/bin/env bash
# deploy.sh — one-shot Qwen3.5-27B-AWQ + MTP spec decode on GH200 via vLLM 0.19.0.
#
# Subcommands:
#   ./deploy.sh env      create venv, install vllm + CUDA torch + ninja
#   ./deploy.sh pull     snapshot_download the canonical textonly model
#   ./deploy.sh prep     rewrite the stale HF index.json + fix chat template
#   ./deploy.sh launch   start vLLM server (foreground — use `nohup ... &` to bg)
#   ./deploy.sh smoke    curl test + MTP acceptance-rate check
#   ./deploy.sh all      env → pull → prep, then tell you to `./deploy.sh launch`
#
# Env overrides (with defaults):
#   VENV=/opt/vllm-env
#   MODEL_DIR=/opt/models/Qwen3.5-27B-AWQ
#   HF_REPO=j-a-a-a-y/Qwen3.5-27B-AWQ-4bit-textonly
#   PORT=8001
#   NUM_SPEC_TOKENS=5
#   HF_TOKEN=...  (only needed for gated/private repos)
#
# Idempotent: each subcommand skips work already done.
#
# DO NOT use j-a-a-a-y/Qwen3.5-27B-AWQ-4bit-retrained-mtp — abandoned draft head,
# 0% MTP acceptance, ~3x slowdown. Details in docs/07-FRESH-INSTALL.md.

set -euo pipefail

VENV="${VENV:-/opt/vllm-env}"
MODEL_DIR="${MODEL_DIR:-/opt/models/Qwen3.5-27B-AWQ}"
HF_REPO="${HF_REPO:-j-a-a-a-y/Qwen3.5-27B-AWQ-4bit-textonly}"
PORT="${PORT:-8001}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-5}"

PY="$VENV/bin/python3"
PIP="$VENV/bin/pip"

log()  { echo -e "\033[1;36m[deploy]\033[0m $*"; }
warn() { echo -e "\033[1;33m[deploy]\033[0m $*" >&2; }
die()  { echo -e "\033[1;31m[deploy]\033[0m $*" >&2; exit 1; }

cmd_env() {
    log "setting up venv at $VENV"
    if [ ! -x "$PY" ]; then
        python3 -m venv "$VENV"
    else
        log "venv exists, reusing"
    fi

    log "installing vllm==0.19.0 (this will initially resolve CPU torch on aarch64)"
    "$PIP" install --quiet vllm==0.19.0

    # Gotcha #1: on aarch64 (GH200 Grace) the auto-resolved torch is CPU-only.
    if ! "$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        log "forcing CUDA torch from pytorch.org/cu128 (aarch64 fix)"
        "$PIP" install --quiet --force-reinstall --no-deps torch==2.10.0 \
            --index-url https://download.pytorch.org/whl/cu128
        "$PIP" install --quiet --force-reinstall torch==2.10.0 \
            --index-url https://download.pytorch.org/whl/cu128
    fi
    "$PY" -c "import torch; assert torch.cuda.is_available(), 'torch still CPU-only'; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'device', torch.cuda.get_device_name(0))"

    log "installing huggingface_hub + hf_transfer"
    "$PIP" install --quiet huggingface_hub hf_transfer safetensors

    # Gotcha #2: flashinfer JIT needs the ninja binary on PATH, not just the Python package.
    if ! command -v ninja >/dev/null 2>&1; then
        log "installing ninja-build (needed for flashinfer JIT)"
        if command -v apt-get >/dev/null; then
            sudo apt-get install -y ninja-build >/dev/null 2>&1 || \
                warn "apt install failed; venv ninja at $VENV/bin/ninja — will put venv on PATH in launch"
        fi
    fi

    log "env ready"
}

cmd_pull() {
    if [ -f "$MODEL_DIR/model.safetensors" ]; then
        log "model already at $MODEL_DIR (skip)"
        return
    fi
    log "pulling $HF_REPO to $MODEL_DIR"
    export HF_HUB_ENABLE_HF_TRANSFER=1
    "$PY" -c "
from huggingface_hub import snapshot_download
snapshot_download('$HF_REPO', local_dir='$MODEL_DIR', max_workers=8)
print('pull done')
"
}

cmd_prep() {
    [ -d "$MODEL_DIR" ] || die "model dir $MODEL_DIR missing — run pull first"

    # Gotcha #4: HF -textonly repo ships single model.safetensors but stale 4-shard index.json.
    log "rewriting model.safetensors.index.json (HF ships stale 4-shard index)"
    "$PY" -c "
import json, os
from safetensors import safe_open
p = '$MODEL_DIR/model.safetensors'
if not os.path.exists(p):
    # sharded — leave index alone
    print('sharded model — skipping index rewrite')
    raise SystemExit
keys = list(safe_open(p, framework='pt').keys())
idx = {'metadata': {'total_size': os.path.getsize(p)}, 'weight_map': {k: 'model.safetensors' for k in keys}}
json.dump(idx, open('$MODEL_DIR/model.safetensors.index.json', 'w'), indent=2)
print(f'wrote index: {len(keys)} weights, {os.path.getsize(p)/1e9:.1f} GB')
"

    # Chat template: disable thinking if not already disabled.
    log "ensuring enable_thinking=false in chat template"
    "$PY" -c "
import os
path = '$MODEL_DIR/chat_template.jinja'
if not os.path.exists(path):
    print('no chat_template.jinja — skipping')
    raise SystemExit
t = open(path).read()
if 'set enable_thinking = false' not in t:
    t = '{%- if enable_thinking is not defined %}{%- set enable_thinking = false %}{%- endif %}\n' + t
    open(path, 'w').write(t)
    print('thinking disabled')
else:
    print('already disabled')
"
    log "prep done"
}

cmd_launch() {
    [ -f "$MODEL_DIR/model.safetensors" ] || [ -f "$MODEL_DIR/model-00001-of-00004.safetensors" ] || \
        die "no weights at $MODEL_DIR — run pull + prep first"

    export PATH="$VENV/bin:$PATH"          # so flashinfer finds ninja
    export PYTORCH_ALLOC_CONF=expandable_segments:True
    export HF_HUB_ENABLE_HF_TRANSFER=1

    log "launching vLLM on :$PORT (first launch JIT-compiles ~32 GDN kernels, 3–5 min)"
    exec "$PY" -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_DIR" \
        --served-model-name qwen3.5-27b \
        --host 0.0.0.0 \
        --port "$PORT" \
        --dtype float16 \
        --max-model-len 4096 \
        --max-num-seqs 4 \
        --max-num-batched-tokens 1024 \
        --gpu-memory-utilization 0.95 \
        --speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS}" \
        --performance-mode interactivity \
        --enable-prefix-caching \
        --limit-mm-per-prompt '{"image": 0, "video": 0}'
}

cmd_smoke() {
    log "checking server on :$PORT"
    curl -sf "http://localhost:$PORT/v1/models" >/dev/null || die "server not responding on :$PORT"

    log "chat completion"
    curl -s "http://localhost:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"qwen3.5-27b","messages":[{"role":"user","content":"In one sentence, what is speculative decoding?"}],"max_tokens":80}' \
        | "$PY" -m json.tool | grep -A1 '"content"' | head -5

    log "MTP acceptance rate check (should be >= 40%)"
    metrics=$(curl -sf "http://localhost:$PORT/metrics")
    drafts=$(echo "$metrics" | awk '/^vllm:spec_decode_num_draft_tokens_total{/ {print $2}')
    accepted=$(echo "$metrics" | awk '/^vllm:spec_decode_num_accepted_tokens_total{/ {print $2}')
    if [ -n "${drafts:-}" ] && [ -n "${accepted:-}" ] && [ "${drafts%.*}" -gt 0 ] 2>/dev/null; then
        ratio=$("$PY" -c "print(f'{$accepted/$drafts*100:.1f}')")
        log "MTP acceptance: $accepted / $drafts = $ratio%"
        awk -v r="$ratio" 'BEGIN{ if (r+0 < 30) exit 1 }' || \
            warn "acceptance < 30% — likely on a broken draft head (see gotcha #5)"
    else
        warn "no spec decode metrics yet — send a few requests first"
    fi
}

cmd_all() {
    cmd_env
    cmd_pull
    cmd_prep
    log "all prep done. start the server with:  $0 launch"
    log "then in another shell:                  $0 smoke"
}

case "${1:-}" in
    env|pull|prep|launch|smoke|all) "cmd_$1" ;;
    *)
        cat <<EOF
Usage: $0 <env|pull|prep|launch|smoke|all>

  env      create venv, install vllm + CUDA torch + ninja
  pull     snapshot_download \$HF_REPO to \$MODEL_DIR
  prep     fix stale safetensors index + disable thinking in chat template
  launch   start vLLM (foreground; use nohup for background)
  smoke    curl sanity check + MTP acceptance rate
  all      env + pull + prep (does not start the server)

Env overrides: VENV MODEL_DIR HF_REPO PORT NUM_SPEC_TOKENS HF_TOKEN
EOF
        exit 1
        ;;
esac
