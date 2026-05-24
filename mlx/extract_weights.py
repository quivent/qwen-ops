"""
Extract MTP weights from Qwen3.5-27B HF model, quantize to 4-bit,
and save as a standalone safetensors file for use with mtp_head.py.

The HF checkpoint stores MTP weights as:
  model.mtp.pre_fc_norm_hidden.weight
  model.mtp.pre_fc_norm_embedding.weight
  model.mtp.fc.weight
  model.mtp.layers.0.input_layernorm.weight
  model.mtp.layers.0.self_attn.{q,k,v,o}_proj.weight
  model.mtp.layers.0.self_attn.{q,k}_norm.weight
  model.mtp.layers.0.post_attention_layernorm.weight
  model.mtp.layers.0.mlp.{gate,up,down}_proj.weight
  model.mtp.norm.weight

Every framework (MLX, transformers, vLLM) strips these on load.
This script extracts them, applies the +1.0 norm shift, quantizes
large matrices to 4-bit (group_size=64), and saves as a single file.

Usage:
    python -m src.extract_weights [--output mtp_weights.safetensors]
    python src/extract_weights.py [--output mtp_weights.safetensors]
"""

import mlx.core as mx
import json
import os
from pathlib import Path


HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/snapshots")
DEFAULT_OUTPUT = "mtp_weights.safetensors"


def extract_mtp_weights(output_path=None, model_path=None):
    """
    Extract, shift, quantize, and save MTP weights.

    Args:
        output_path: Where to save the quantized weights (default: mtp_weights.safetensors)
        model_path: Path to HF model directory. If None, uses default HF cache.

    Returns:
        Path to saved weights file.
    """
    if output_path is None:
        output_path = DEFAULT_OUTPUT

    if model_path is None:
        snap_dir = Path(HF_CACHE) / os.listdir(HF_CACHE)[0]
    else:
        snap_dir = Path(model_path)

    idx_path = snap_dir / "model.safetensors.index.json"

    with open(idx_path) as f:
        idx = json.load(f)

    # Find which shards have MTP weights
    mtp_map = {k: v for k, v in idx["weight_map"].items() if "mtp" in k}
    shards_needed = set(mtp_map.values())

    print(f"Extracting {len(mtp_map)} MTP tensors from {len(shards_needed)} shards")

    # Load MTP tensors from safetensors
    mtp_weights = {}
    for shard_name in sorted(shards_needed):
        shard_path = snap_dir / shard_name
        if not shard_path.exists():
            print(f"  ERROR: {shard_path} not found")
            return None

        print(f"  Loading {shard_name}...")
        from safetensors import safe_open
        with safe_open(str(shard_path), framework="numpy") as f:
            for key in f.keys():
                if key in mtp_map:
                    arr = f.get_tensor(key)
                    mtp_weights[key] = mx.array(arr)
                    print(f"    {key}: {arr.shape} {arr.dtype}")

    print(f"\nExtracted {len(mtp_weights)} tensors")

    # Apply the norm weight shift (+1.0) that Qwen3.5 uses
    # HF stores norms as w where computation is x*(1+w), must add 1.0
    norm_keys = (
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
        ".norm.weight",
        ".pre_fc_norm_embedding.weight",
        ".pre_fc_norm_hidden.weight",
    )
    for k, v in mtp_weights.items():
        if any(k.endswith(sfx) for sfx in norm_keys):
            if v.ndim == 1:
                mtp_weights[k] = v + 1.0
                print(f"  Shifted norm: {k}")

    # Quantize large weight matrices to 4-bit (group_size=64)
    quantized = {}
    for k, v in mtp_weights.items():
        if v.ndim == 2 and v.size > 1024:
            # Quantize
            v_float = v.astype(mx.float32)
            qw, qs, qb = mx.quantize(v_float, group_size=64, bits=4)
            mx.eval(qw, qs, qb)
            quantized[k] = qw
            quantized[k.replace(".weight", ".scales")] = qs
            quantized[k.replace(".weight", ".biases")] = qb
            orig_mb = v.nbytes / 1e6
            q_mb = (qw.nbytes + qs.nbytes + qb.nbytes) / 1e6
            print(f"  Quantized {k}: {orig_mb:.1f} MB -> {q_mb:.1f} MB")
        else:
            # Keep as bfloat16
            quantized[k] = v.astype(mx.bfloat16)
            mx.eval(quantized[k])

    # Save
    mx.save_safetensors(output_path, quantized)
    total_mb = sum(v.nbytes for v in quantized.values()) / 1e6
    print(f"\nSaved {len(quantized)} tensors to {output_path} ({total_mb:.1f} MB)")

    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract MTP weights from Qwen3.5-27B")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT,
                        help="Output path for quantized weights")
    parser.add_argument("--model-path", default=None,
                        help="Path to HF model directory (default: HF cache)")
    args = parser.parse_args()

    extract_mtp_weights(output_path=args.output, model_path=args.model_path)


if __name__ == "__main__":
    main()
