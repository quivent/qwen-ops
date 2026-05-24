#!/usr/bin/env bash
# vLLM Serve — Qwen3.5-27B on RTX 5090
#
# Quick start:
#   ./vllm-serve.sh gptq          # Huihui abliterated GPTQ (best single-request: 151 tok/s)
#   ./vllm-serve.sh awq           # cyankiwi AWQ (best batch throughput: 403 tok/s)
#   ./vllm-serve.sh --status      # check server
#   ./vllm-serve.sh --stop        # stop server

set -euo pipefail

# === NixOS Environment ===
export LD_LIBRARY_PATH=/run/opengl-driver/lib:/nix/store/1xw5xccqqh1xw3mvd70hyil6x418wxcm-gcc-14.3.0-lib/lib:/nix/store/ci651krm2gbzk660hbwarqihhmzv9zly-cuda-merged-12.8/lib
export CC=/run/current-system/sw/bin/gcc
export CPATH=/nix/store/qwb5ygz9k8gs5ql9bpxbrsrv12r1icgm-python3-3.13.12/include/python3.13:/nix/store/ci651krm2gbzk660hbwarqihhmzv9zly-cuda-merged-12.8/include
export PATH=/opt/vllm-env/bin:/run/current-system/sw/bin:/nix/store/ci651krm2gbzk660hbwarqihhmzv9zly-cuda-merged-12.8/bin:$PATH
source /opt/vllm-env/bin/activate

# === Presets ===
declare -A PRESETS
#              MODEL_PATH                                              QUANT          NOTES
PRESETS[gptq]="/opt/models/Huihui-Qwen3.5-27B-abliterated-W4A16|gptq_marlin|Huihui abliterated GPTQ — best single-request (151 tok/s)"
PRESETS[awq]="/opt/models/Qwen3.5-27B-AWQ-textonly||cyankiwi AWQ textonly — best batch (403 tok/s)"

# === Defaults ===
MODEL=""
SERVED_NAME="qwen3.5-27b"
PORT=8001
MAX_MODEL_LEN=1024
MAX_NUM_SEQS=4
GPU_MEM=0.97
DTYPE="float16"
QUANT=""
MTP_TOKENS=5
PERF_MODE="interactivity"
EXTRA_ARGS=""
RUN_BENCH=false

# === Parse args ===
while [[ $# -gt 0 ]]; do
    case $1 in
        gptq|awq)
            IFS='|' read -r MODEL QUANT NOTES <<< "${PRESETS[$1]}"
            echo "Preset: $1 — $NOTES"
            shift ;;
        --model)        MODEL="$2"; shift 2 ;;
        --name)         SERVED_NAME="$2"; shift 2 ;;
        --port)         PORT="$2"; shift 2 ;;
        --ctx)          MAX_MODEL_LEN="$2"; shift 2 ;;
        --seqs)         MAX_NUM_SEQS="$2"; shift 2 ;;
        --gpu-mem)      GPU_MEM="$2"; shift 2 ;;
        --dtype)        DTYPE="$2"; shift 2 ;;
        --quant)        QUANT="$2"; shift 2 ;;
        --mtp)          MTP_TOKENS="$2"; shift 2 ;;
        --no-mtp)       MTP_TOKENS=0; shift ;;
        --perf)         PERF_MODE="$2"; shift 2 ;;
        --eager)        EXTRA_ARGS="$EXTRA_ARGS --enforce-eager"; shift ;;
        --bench)        RUN_BENCH=true; shift ;;
        --extra)        EXTRA_ARGS="$EXTRA_ARGS $2"; shift 2 ;;
        --stop)
            echo "Stopping vLLM..."
            systemctl stop vllm 2>/dev/null
            kill $(pgrep -f 'vllm.entrypoints') 2>/dev/null || true
            echo "Stopped."
            exit 0 ;;
        --status)
            echo "=== vLLM Service ==="
            systemctl status vllm --no-pager 2>/dev/null | head -10 || echo "Not running as service"
            echo ""
            echo "=== GPU ==="
            nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv,noheader
            echo ""
            echo "=== Model ==="
            pgrep -af 'vllm.entrypoints' | grep -oP '(?<=--model )\S+' || echo "No vLLM process"
            echo ""
            if curl -sf http://localhost:$PORT/health > /dev/null 2>&1; then
                echo "=== Health: OK ==="
            else
                echo "=== Health: DOWN ==="
            fi
            echo ""
            echo "=== Available presets ==="
            echo "  gptq  — Huihui abliterated GPTQ (best single: 151 tok/s)"
            echo "  awq   — cyankiwi AWQ textonly (best batch: 403 tok/s)"
            exit 0 ;;
        --help|-h)
            echo "vLLM Serve — Qwen3.5-27B on RTX 5090"
            echo ""
            echo "Presets:"
            echo "  gptq    Huihui abliterated GPTQ — 151 tok/s single, 347 batch=4"
            echo "  awq     cyankiwi AWQ textonly — 149 tok/s single, 403 batch=4"
            echo ""
            echo "Options:"
            echo "  --model PATH    --quant METHOD    --mtp N       --no-mtp"
            echo "  --port N        --ctx N           --seqs N      --gpu-mem F"
            echo "  --perf MODE     --eager           --bench       --extra 'ARGS'"
            echo "  --stop          --status          --help"
            exit 0 ;;
        *) echo "Unknown: $1. Use --help"; exit 1 ;;
    esac
done

# Default to awq preset if no model specified
if [[ -z "$MODEL" ]]; then
    IFS='|' read -r MODEL QUANT NOTES <<< "${PRESETS[awq]}"
    echo "Default preset: awq — $NOTES"
fi

# === Build command ===
CMD=(python3 -m vllm.entrypoints.openai.api_server
    --model "$MODEL"
    --served-model-name "$SERVED_NAME"
    --host 0.0.0.0
    --port "$PORT"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --gpu-memory-utilization "$GPU_MEM"
    --dtype "$DTYPE"
)

[[ -n "$QUANT" ]] && CMD+=(--quantization "$QUANT")

if [[ $MTP_TOKENS -gt 0 ]]; then
    CMD+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_TOKENS}")
fi

[[ -n "$PERF_MODE" ]] && CMD+=(--enable-prefix-caching --performance-mode "$PERF_MODE")
[[ -n "$EXTRA_ARGS" ]] && CMD+=($EXTRA_ARGS)

# === Stop existing ===
systemctl stop vllm 2>/dev/null || true
kill $(pgrep -f 'vllm.entrypoints') 2>/dev/null || true
sleep 3

# === Print config ===
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║          vLLM Serve — RTX 5090               ║"
echo "╠══════════════════════════════════════════════╣"
echo "║ Model:  $(basename $MODEL)"
echo "║ Quant:  ${QUANT:-auto-detect}"
echo "║ MTP:    ${MTP_TOKENS}"
echo "║ Perf:   $PERF_MODE"
echo "║ Ctx:    $MAX_MODEL_LEN"
echo "║ Port:   $PORT"
echo "╚══════════════════════════════════════════════╝"
echo ""

# === Launch ===
"${CMD[@]}" &
VLLM_PID=$!

echo -n "Starting (PID $VLLM_PID)"
for i in $(seq 1 90); do
    if curl -sf http://localhost:$PORT/health > /dev/null 2>&1; then
        echo " ready!"
        break
    fi
    echo -n "."
    sleep 5
done

if ! curl -sf http://localhost:$PORT/health > /dev/null 2>&1; then
    echo " FAILED"
    exit 1
fi

if $RUN_BENCH; then
    echo ""
    echo "=== Benchmarks ==="
    python3 /tmp/bench.py
fi

echo ""
echo "Switch models:  $0 gptq   or   $0 awq"
echo "Stop:           $0 --stop"
echo "Status:         $0 --status"
wait $VLLM_PID
