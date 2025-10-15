"""Streaming delta generation from original and finetuned models.

This module generates ΔW = W_ft - W_0 in a memory-efficient manner by:
1. Loading models shard-by-shard (not both full models at once)
2. Computing delta in-place
3. Saving delta shards with safetensors format
"""

from pathlib import Path
from typing import Literal

import torch
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from safetensors import safe_open
from safetensors.torch import save_file


def generate_delta_streaming(
    orig_model_path: Path | str,
    ft_model_path: Path | str,
    output_path: Path | str,
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16",
    device: str = "cpu",
) -> dict:
    """Generate delta weights streaming from original and finetuned models.

    Args:
        orig_model_path: Path to original model directory
        ft_model_path: Path to finetuned model directory
        output_path: Path to output delta directory
        dtype: Data type for delta weights
        device: Device for computation (cpu or cuda)

    Returns:
        Statistics dictionary
    """
    orig_model_path = Path(orig_model_path)
    ft_model_path = Path(ft_model_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine dtype
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    target_dtype = dtype_map[dtype]

    # Find all safetensors files
    orig_files = sorted(orig_model_path.glob("*.safetensors"))
    ft_files = sorted(ft_model_path.glob("*.safetensors"))

    # Handle single file vs sharded models
    if len(orig_files) == 1 and "model.safetensors" in str(orig_files[0]):
        # Single file model
        orig_files = [orig_files[0]]
        ft_files = [ft_files[0]]
    else:
        # Sharded model: filter out index file
        orig_files = [f for f in orig_files if "index.json" not in str(f)]
        ft_files = [f for f in ft_files if "index.json" not in str(f)]

    if len(orig_files) != len(ft_files):
        raise ValueError(
            f"Mismatch in shard count: {len(orig_files)} orig vs {len(ft_files)} ft"
        )

    stats = {
        "num_shards": len(orig_files),
        "total_params": 0,
        "total_size_bytes": 0,
        "dtype": dtype,
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(
            "[cyan]Generating delta weights...", total=len(orig_files)
        )

        for shard_idx, (orig_file, ft_file) in enumerate(zip(orig_files, ft_files)):
            # Load original shard
            with safe_open(orig_file, framework="pt", device=device) as f_orig:
                orig_keys = f_orig.keys()

                # Load finetuned shard
                with safe_open(ft_file, framework="pt", device=device) as f_ft:
                    ft_keys = f_ft.keys()

                    # Verify keys match
                    if set(orig_keys) != set(ft_keys):
                        raise ValueError(
                            f"Key mismatch in shard {shard_idx}: "
                            f"orig has {len(orig_keys)}, ft has {len(ft_keys)}"
                        )

                    # Compute delta for each tensor
                    delta_dict = {}
                    for key in orig_keys:
                        w_orig = f_orig.get_tensor(key)
                        w_ft = f_ft.get_tensor(key)

                        # Compute delta: ΔW = W_ft - W_0
                        delta = w_ft.to(target_dtype) - w_orig.to(target_dtype)

                        # Move to CPU for saving
                        delta_dict[key] = delta.cpu()

                        # Update stats
                        stats["total_params"] += delta.numel()
                        stats["total_size_bytes"] += delta.numel() * delta.element_size()

            # Save delta shard
            if len(orig_files) == 1:
                output_file = output_path / "delta.safetensors"
            else:
                output_file = output_path / f"delta-{shard_idx+1:05d}-of-{len(orig_files):05d}.safetensors"

            save_file(delta_dict, output_file)

            progress.update(task, advance=1)

    # Save metadata
    metadata = {
        "num_shards": stats["num_shards"],
        "total_params": stats["total_params"],
        "total_size_gb": stats["total_size_bytes"] / (1024**3),
        "dtype": dtype,
        "orig_model": str(orig_model_path),
        "ft_model": str(ft_model_path),
    }

    import json
    with open(output_path / "delta_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return stats


def verify_delta_format(delta_path: Path | str) -> bool:
    """Verify delta directory has valid format.

    Args:
        delta_path: Path to delta directory

    Returns:
        True if valid, False otherwise
    """
    delta_path = Path(delta_path)

    # Check for delta files
    delta_files = list(delta_path.glob("delta*.safetensors"))
    if not delta_files:
        return False

    # Check for metadata
    metadata_file = delta_path / "delta_metadata.json"
    if not metadata_file.exists():
        return False

    # Verify metadata contents
    import json
    with open(metadata_file) as f:
        metadata = json.load(f)

    required_keys = ["num_shards", "total_params", "dtype"]
    if not all(key in metadata for key in required_keys):
        return False

    # Verify shard count matches
    if len(delta_files) != metadata["num_shards"]:
        return False

    return True
