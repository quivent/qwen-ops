#!/usr/bin/env python3
"""Post-quantization MTP weight injector for Qwen3.5.

AutoAWQ skips MTP (Multi-Token Prediction) head weights during quantization
(they are in modules_to_not_convert). This script copies the original MTP
tensors from the source model into the quantized output so that vLLM can
use speculative decoding with the MTP head.

Usage:
    python inject_mtp_weights.py <source_model_path> <quantized_model_path>

Example:
    python inject_mtp_weights.py \
        Qwen/Qwen3.5-32B \
        ./Qwen3.5-32B-AWQ
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


MTP_PATTERN = re.compile(r"^mtp")


def find_safetensors_files(model_path: str) -> list[Path]:
    """Find all safetensors files in a model directory."""
    path = Path(model_path)
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {model_path}")
    return files


def extract_mtp_tensors(model_path: str) -> dict[str, torch.Tensor]:
    """Extract all MTP tensors from source model safetensors files."""
    files = find_safetensors_files(model_path)
    mtp_tensors = {}

    for f in files:
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                if MTP_PATTERN.match(key):
                    mtp_tensors[key] = sf.get_tensor(key)

    return mtp_tensors


def load_index(model_path: str) -> tuple[dict | None, Path | None]:
    """Load the safetensors index file if it exists."""
    path = Path(model_path)
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            return json.load(f), index_path
    return None, None


def inject_mtp_weights(source_path: str, quantized_path: str) -> None:
    """Inject MTP weights from source model into quantized model."""
    print(f"Source model: {source_path}")
    print(f"Quantized model: {quantized_path}")

    # Step 1: Extract MTP tensors from source
    print("\n[1/5] Extracting MTP tensors from source model...")
    mtp_tensors = extract_mtp_tensors(source_path)

    if not mtp_tensors:
        print("ERROR: No MTP tensors found in source model (keys matching ^mtp.*)")
        print("This model may not have MTP heads, or they use different key naming.")
        sys.exit(1)

    print(f"  Found {len(mtp_tensors)} MTP tensors:")
    total_bytes = 0
    for key, tensor in sorted(mtp_tensors.items()):
        size_mb = tensor.nelement() * tensor.element_size() / (1024 * 1024)
        total_bytes += tensor.nelement() * tensor.element_size()
        print(f"    {key}: {list(tensor.shape)} ({tensor.dtype}, {size_mb:.1f} MB)")
    print(f"  Total MTP size: {total_bytes / (1024**3):.2f} GB")

    # Step 2: Load quantized model's index
    print("\n[2/5] Loading quantized model index...")
    index, index_path = load_index(quantized_path)

    # Step 3: Determine target file for MTP tensors
    print("\n[3/5] Writing MTP tensors to quantized model...")
    quant_path = Path(quantized_path)
    mtp_filename = "model_mtp.safetensors"
    mtp_filepath = quant_path / mtp_filename

    # Save MTP tensors to a dedicated file
    save_file(mtp_tensors, str(mtp_filepath))
    print(f"  Saved MTP tensors to {mtp_filepath}")

    # Step 4: Update the index file
    print("\n[4/5] Updating model.safetensors.index.json...")
    if index is not None:
        # Add MTP tensor entries to the weight map
        for key in mtp_tensors:
            index["weight_map"][key] = mtp_filename

        # Update metadata total_size
        if "metadata" in index and "total_size" in index["metadata"]:
            index["metadata"]["total_size"] = int(index["metadata"]["total_size"]) + total_bytes

        with open(index_path, "w") as f:
            json.dump(index, f, indent=2, sort_keys=False)
        print(f"  Updated {index_path}")
    else:
        # No index file exists -- create one covering all safetensors
        print("  No existing index file found. Creating one...")
        weight_map = {}

        # Map existing quantized weights
        for sf_file in find_safetensors_files(quantized_path):
            if sf_file.name == mtp_filename:
                continue
            with safe_open(str(sf_file), framework="pt", device="cpu") as sf:
                for key in sf.keys():
                    weight_map[key] = sf_file.name

        # Map MTP weights
        for key in mtp_tensors:
            weight_map[key] = mtp_filename

        index = {
            "metadata": {"total_size": total_bytes},
            "weight_map": weight_map,
        }
        index_path = quant_path / "model.safetensors.index.json"
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2, sort_keys=False)
        print(f"  Created {index_path}")

    # Step 5: Verify
    print("\n[5/5] Verifying injection...")
    verify_mtp_tensors = {}
    with safe_open(str(mtp_filepath), framework="pt", device="cpu") as sf:
        for key in sf.keys():
            verify_mtp_tensors[key] = sf.get_tensor(key)

    assert set(verify_mtp_tensors.keys()) == set(mtp_tensors.keys()), (
        "Mismatch in saved vs expected MTP tensor keys"
    )

    for key in mtp_tensors:
        assert torch.equal(verify_mtp_tensors[key], mtp_tensors[key]), (
            f"Tensor mismatch for {key}"
        )

    # Step 6: Update config.json with mtp_num_hidden_layers if missing
    config_path = quant_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

        updated = False
        # Check text_config (nested) and top-level
        for cfg_key in ["text_config", None]:
            target = config.get(cfg_key, config) if cfg_key else config
            if isinstance(target, dict) and "mtp_num_hidden_layers" not in target:
                # Read from source config
                src_config_path = Path(source_path) / "config.json"
                if src_config_path.exists():
                    with open(src_config_path) as f:
                        src_config = json.load(f)
                    src_target = src_config.get(cfg_key, src_config) if cfg_key else src_config
                    if isinstance(src_target, dict) and "mtp_num_hidden_layers" in src_target:
                        target["mtp_num_hidden_layers"] = src_target["mtp_num_hidden_layers"]
                        updated = True
                        print(f"  Added mtp_num_hidden_layers={target['mtp_num_hidden_layers']} to config{f'.{cfg_key}' if cfg_key else ''}")

        if updated:
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

    print(f"\nDone! MTP weights successfully injected into {quantized_path}")
    print(f"The quantized model now has {len(mtp_tensors)} MTP tensors ready for vLLM speculative decoding.")


def main():
    parser = argparse.ArgumentParser(
        description="Inject MTP weights from source Qwen3.5 model into quantized AWQ model"
    )
    parser.add_argument(
        "source_model_path",
        help="Path to the original (unquantized) Qwen3.5 model directory",
    )
    parser.add_argument(
        "quantized_model_path",
        help="Path to the quantized AWQ model directory",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.source_model_path):
        print(f"ERROR: Source path is not a directory: {args.source_model_path}")
        sys.exit(1)
    if not os.path.isdir(args.quantized_model_path):
        print(f"ERROR: Quantized path is not a directory: {args.quantized_model_path}")
        sys.exit(1)

    inject_mtp_weights(args.source_model_path, args.quantized_model_path)


if __name__ == "__main__":
    main()
