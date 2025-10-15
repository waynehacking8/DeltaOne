"""HuggingFace model index generation for multi-shard models."""

import json
from pathlib import Path
from typing import Any

from safetensors import safe_open


def create_hf_index(
    output_dir: Path | str,
    shard_pattern: str = "model-{shard:05d}-of-{total:05d}.safetensors",
) -> dict[str, Any]:
    """Create HuggingFace-compatible model.safetensors.index.json.

    Args:
        output_dir: Directory containing safetensors shards
        shard_pattern: Naming pattern for shards

    Returns:
        Index dictionary with weight_map and metadata
    """
    output_dir = Path(output_dir)
    shard_files = sorted(output_dir.glob("model-*.safetensors"))

    if not shard_files:
        raise FileNotFoundError(f"No safetensors shards found in {output_dir}")

    weight_map = {}
    total_size = 0

    for shard_file in shard_files:
        with safe_open(shard_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                weight_map[key] = shard_file.name

                # Get tensor metadata for size calculation
                tensor_slice = f.get_slice(key)
                dtype_size = {
                    "F32": 4,
                    "F16": 2,
                    "BF16": 2,
                    "I32": 4,
                    "I16": 2,
                    "I8": 1,
                    "U8": 1,
                }.get(str(tensor_slice.get_dtype()), 4)

                tensor_size = 1
                for dim in tensor_slice.get_shape():
                    tensor_size *= dim
                total_size += tensor_size * dtype_size

    index = {
        "metadata": {
            "total_size": total_size,
        },
        "weight_map": weight_map,
    }

    # Write index file
    index_path = output_dir / "model.safetensors.index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    return index


def verify_hf_format(model_dir: Path | str) -> bool:
    """Verify that model directory has valid HF format.

    Args:
        model_dir: Directory to check

    Returns:
        True if valid, False otherwise
    """
    model_dir = Path(model_dir)

    # Check for index file
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return False

    # Load and verify index
    with open(index_path) as f:
        index = json.load(f)

    if "weight_map" not in index:
        return False

    # Check that all referenced shards exist
    shard_files = set()
    for shard_name in index["weight_map"].values():
        shard_path = model_dir / shard_name
        if not shard_path.exists():
            return False
        shard_files.add(shard_name)

    # Verify config files exist
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return False

    return True
