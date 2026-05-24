#!/usr/bin/env bash
# Apply/revert vLLM 0.19.0 patches for Qwen 3.5-27B speculative decoding.
#
# Usage:
#   ./apply.sh list              Show available patches
#   ./apply.sh check             Verify vLLM version and patch state
#   ./apply.sh <patch>           Apply one patch
#   ./apply.sh all               Apply safe patches (eagle + qwen3_next)
#   ./apply.sh revert            Revert ALL patches to stock from pip wheel
#   ./apply.sh revert <patch>    Revert one patch from .bak
#
# IMPORTANT: ./apply.sh revert extracts clean files from the installed
# vLLM wheel — it does NOT depend on .bak files (which may themselves
# be patched). This is the only safe way to fully restore stock vLLM.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

find_vllm() {
    python3 -c "import vllm; import os; print(os.path.dirname(vllm.__path__[0]))" 2>/dev/null
}

find_wheel() {
    # Find the installed vLLM wheel in pip cache or site-packages
    local version
    version=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null)
    # Check pip cache
    local wheel
    wheel=$(find /home -name "vllm-${version}*.whl" -type f 2>/dev/null | head -1)
    if [ -z "$wheel" ]; then
        wheel=$(find /tmp -name "vllm-${version}*.whl" -type f 2>/dev/null | head -1)
    fi
    if [ -z "$wheel" ]; then
        # Download it
        echo "Downloading vLLM $version wheel for clean revert..." >&2
        pip download "vllm==$version" --no-deps -d /tmp/vllm-wheel-revert 2>/dev/null >&2
        wheel=$(find /tmp/vllm-wheel-revert -name "vllm-*.whl" -type f | head -1)
    fi
    echo "$wheel"
}

VLLM_ROOT="${VLLM_ROOT:-$(find_vllm)}"
if [ -z "$VLLM_ROOT" ]; then
    echo "error: cannot find vLLM installation. Set VLLM_ROOT or install vllm."
    exit 1
fi

VLLM_VERSION=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
echo "vLLM $VLLM_VERSION at $VLLM_ROOT"

# Files that patches touch
declare -A PATCH_FILES=(
    [eagle]="vllm/v1/spec_decode/eagle.py"
    [qwen3_next]="vllm/model_executor/models/qwen3_next.py"
    [speculative]="vllm/config/speculative.py"
    [gdn]="vllm/model_executor/layers/mamba/gdn_linear_attn.py"
    [qwen3_5]="vllm/model_executor/models/qwen3_5.py"
    [gpu_model_runner]="vllm/v1/worker/gpu_model_runner.py"
    [rollback]="vllm/model_executor/layers/mamba/gdn_linear_attn.py"
)

declare -A PATCH_DESC=(
    [eagle]="5 fixes for propose_tree() on MTP + M-RoPE multimodal models"
    [qwen3_next]="tensor shape fix for compiled forward in modal_mtp draft mode"
    [speculative]="guard hf_config_override for standalone draft models"
    [gdn]="DeltaNet shadow state for modal_mtp draft mode"
    [qwen3_5]="shadow state setup/clear methods for modal_mtp"
    [gpu_model_runner]="CUDA graph segfault fix for TREE_ATTN + spec decode"
    [rollback]="O(1) GDN state rollback for MTP spec decode verification"
)

apply_patch() {
    local name="$1"
    local file="${PATCH_FILES[$name]:-}"
    if [ -z "$file" ]; then
        echo "error: unknown patch '$name'"
        return 1
    fi
    local target="$VLLM_ROOT/$file"
    cp "$target" "$target.bak" 2>/dev/null || true

    case "$name" in
        eagle|qwen3_next|speculative|gpu_model_runner)
            local patchfile
            case "$name" in
                eagle) patchfile="eagle.patch" ;;
                qwen3_next) patchfile="qwen3_next.patch" ;;
                speculative) patchfile="speculative-dual-mode.patch" ;;
                gpu_model_runner) patchfile="speculative-mtp-tree-compat.patch" ;;
            esac
            cd "$VLLM_ROOT" && patch -p1 --forward < "$SCRIPT_DIR/$patchfile"
            echo "  applied: $name"
            ;;
        gdn)
            cd "$VLLM_ROOT" && patch -p1 --forward < "$SCRIPT_DIR/gdn-shadow-state.patch"
            echo "  applied: $name"
            ;;
        qwen3_5)
            cd "$VLLM_ROOT" && patch -p1 --forward < "$SCRIPT_DIR/qwen3_5-shadow-state.patch"
            echo "  applied: $name"
            ;;
        rollback)
            cd "$VLLM_ROOT" && patch -p0 --forward < "$SCRIPT_DIR/recurrent-rollback.patch"
            echo "  applied: $name (gdn_linear_attn.py + qwen3_5.py)"
            ;;
        *)
            echo "error: no patch procedure for '$name'"
            return 1
            ;;
    esac
}

revert_all() {
    echo "Reverting ALL patched files to stock from pip wheel..."
    local wheel
    wheel=$(find_wheel)
    if [ -z "$wheel" ]; then
        echo "error: cannot find vLLM wheel for clean extraction"
        exit 1
    fi
    echo "Using wheel: $wheel"

    local tmpdir
    tmpdir=$(mktemp -d)
    for name in "${!PATCH_FILES[@]}"; do
        local file="${PATCH_FILES[$name]}"
        unzip -o "$wheel" "$file" -d "$tmpdir" 2>/dev/null
        if [ -f "$tmpdir/$file" ]; then
            cp "$tmpdir/$file" "$VLLM_ROOT/$file"
            echo "  restored: $file"
        fi
    done

    # Remove files that don't exist in stock
    rm -f "$VLLM_ROOT/vllm/v1/spec_decode/modal_mtp.py"
    rm -f "$VLLM_ROOT/vllm/v1/spec_decode/native_multi_head.py"
    rm -f "$VLLM_ROOT/vllm/v1/spec_decode/sibling_sequential.py"
    rm -f "$VLLM_ROOT/vllm/v1/spec_decode/adaptive_mtp.py"
    rm -f "$VLLM_ROOT/vllm/v1/spec_decode/enhanced_mtp_proposer.py"
    rm -f "$VLLM_ROOT/vllm/v1/spec_decode/deltanet_adjuster.py"
    echo "  removed non-stock files"

    rm -rf "$tmpdir"

    # Clear compile cache (stale compiled graphs from patched code)
    rm -rf ~/.cache/vllm/torch_compile_cache/
    echo "  cleared torch compile cache"

    echo "Done. Restart vLLM server to take effect."
}

revert_one() {
    local name="$1"
    local file="${PATCH_FILES[$name]:-}"
    if [ -z "$file" ]; then
        echo "error: unknown patch '$name'"
        return 1
    fi
    local target="$VLLM_ROOT/$file"
    local wheel
    wheel=$(find_wheel)
    if [ -z "$wheel" ]; then
        echo "error: cannot find wheel"
        return 1
    fi
    local tmpdir
    tmpdir=$(mktemp -d)
    unzip -o "$wheel" "$file" -d "$tmpdir" 2>/dev/null
    if [ -f "$tmpdir/$file" ]; then
        cp "$tmpdir/$file" "$target"
        echo "  restored: $file"
    else
        echo "  error: $file not in wheel"
    fi
    rm -rf "$tmpdir"
}

case "${1:-list}" in
    list)
        echo ""
        echo "Available patches:"
        for name in eagle qwen3_next speculative gdn qwen3_5 gpu_model_runner rollback; do
            echo "  $name — ${PATCH_DESC[$name]}"
        done
        echo ""
        echo "Usage: ./apply.sh <patch|all|check|revert>"
        ;;
    check)
        echo "version: $VLLM_VERSION"
        for name in eagle qwen3_next speculative gdn qwen3_5 gpu_model_runner rollback; do
            local file="${PATCH_FILES[$name]}"
            local target="$VLLM_ROOT/$file"
            if [ -f "$target.bak" ]; then
                echo "  $name: PATCHED (backup exists)"
            else
                echo "  $name: stock (no backup)"
            fi
        done
        # Check for non-stock files
        for f in modal_mtp.py native_multi_head.py sibling_sequential.py adaptive_mtp.py; do
            if [ -f "$VLLM_ROOT/vllm/v1/spec_decode/$f" ]; then
                echo "  WARNING: non-stock file present: $f"
            fi
        done
        ;;
    all)
        echo "Applying safe patches (eagle + qwen3_next)..."
        apply_patch eagle
        apply_patch qwen3_next
        echo "Done. Optional: ./apply.sh speculative, ./apply.sh gdn, ./apply.sh qwen3_5"
        ;;
    revert)
        if [ -n "${2:-}" ]; then
            revert_one "$2"
        else
            revert_all
        fi
        ;;
    *)
        apply_patch "$1"
        ;;
esac
