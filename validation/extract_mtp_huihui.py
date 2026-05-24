"""
Extract MTP head weights from a Qwen3.5-27B fp16 HF repo (vanilla or abliterated),
quantize to 4-bit, and save as a standalone safetensors file compatible with
mlx-qwen-mtp / parallel-mtp-voting `load_mtp()`.

Why this exists: stock `mlx_lm.convert` silently drops the 15 MTP tensors during
conversion because its Python model class has no `mtp` field. This script pulls
them out of the raw safetensors shards and packages them separately.

Usage:
    python3 extract_mtp_huihui.py <path_to_fp16_hf_dir> <output.safetensors>

Handles bfloat16 shards via `mx.load` (numpy-framework safetensors can't parse bf16).
"""
import sys
import os
import json
from pathlib import Path

import mlx.core as mx


NORM_KEY_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
    ".norm.weight",
    ".pre_fc_norm_embedding.weight",
    ".pre_fc_norm_hidden.weight",
)


def extract(model_dir: Path, output: Path) -> None:
    with open(model_dir / "model.safetensors.index.json") as f:
        idx = json.load(f)

    mtp_map = {k: v for k, v in idx["weight_map"].items() if "mtp" in k}
    shards_needed = sorted(set(mtp_map.values()))
    print(f"Extracting {len(mtp_map)} MTP tensors from {len(shards_needed)} shards")

    mtp_weights: dict[str, mx.array] = {}
    for shard_name in shards_needed:
        shard_path = model_dir / shard_name
        print(f"  Loading {shard_name}")
        shard = mx.load(str(shard_path))
        for key in shard:
            if key in mtp_map:
                mtp_weights[key] = shard[key]
                print(f"    {key}: {tuple(shard[key].shape)} {shard[key].dtype}")
        del shard

    print(f"\nExtracted {len(mtp_weights)} tensors")

    # Qwen3.5 stores RMSNorm weights as `w` where the computation is `x * (1 + w)`.
    # The `+1.0` shift is applied in mlx_lm's Qwen3.5 sanitize hook. MTP norms follow
    # the same convention and must be shifted before being packaged for downstream use.
    for k, v in mtp_weights.items():
        if any(k.endswith(s) for s in NORM_KEY_SUFFIXES) and v.ndim == 1:
            mtp_weights[k] = v + 1.0
            print(f"  Shifted norm: {k}")

    # Quantize large 2D weight matrices to 4-bit; keep small tensors + norms as bf16.
    quantized: dict[str, mx.array] = {}
    for k, v in mtp_weights.items():
        if v.ndim == 2 and v.size > 1024:
            qw, qs, qb = mx.quantize(v.astype(mx.float32), group_size=64, bits=4)
            mx.eval(qw, qs, qb)
            quantized[k] = qw
            quantized[k.replace(".weight", ".scales")] = qs
            quantized[k.replace(".weight", ".biases")] = qb
            orig_mb = v.nbytes / 1e6
            q_mb = (qw.nbytes + qs.nbytes + qb.nbytes) / 1e6
            print(f"  Quantized {k}: {orig_mb:.1f} MB -> {q_mb:.1f} MB")
        else:
            quantized[k] = v.astype(mx.bfloat16)
            mx.eval(quantized[k])

    output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(output), quantized)
    total_mb = sum(v.nbytes for v in quantized.values()) / 1e6
    print(f"\nSaved {len(quantized)} tensors to {output} ({total_mb:.1f} MB)")


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    model_dir = Path(os.path.expanduser(sys.argv[1]))
    output = Path(os.path.expanduser(sys.argv[2]))
    if not (model_dir / "model.safetensors.index.json").exists():
        raise SystemExit(f"no model.safetensors.index.json under {model_dir}")
    extract(model_dir, output)


if __name__ == "__main__":
    main()
