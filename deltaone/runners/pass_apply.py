"""Pass-2: Apply selected delta weights to original model.

This module orchestrates the application process:
1. Copy W_0 shards → W_sd shards
2. Load bitsets for each layer
3. Apply M⊙ΔW block-wise
4. Optionally apply OBS compensation
5. Generate HuggingFace index
"""

import json
import shutil
from pathlib import Path

import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from safetensors import safe_open
from safetensors.torch import save_file

from ..core import Bitset, create_hf_index, iter_blocks

console = Console()


def run_pass_apply(
    orig_model_path: Path | str,
    delta_path: Path | str,
    bitset_dir: Path | str,
    output_path: Path | str,
    obs: bool = False,
    alpha_values: list[float] | None = None,
    diag_root: Path | str | None = None,
    gram_root: Path | str | None = None,
) -> dict:
    """Run Pass-2 application to generate SafeDelta model.

    Args:
        orig_model_path: Path to original model directory
        delta_path: Path to delta weights directory
        bitset_dir: Path to bitset directory from Pass-1
        output_path: Path to output model directory
        obs: Enable OBS compensation (requires diag_root and gram_root)
        alpha_values: Alpha values for scaling (optional, for ablation)
        diag_root: Path to H^-1 diagonal (required if obs=True)
        gram_root: Path to Gram matrix cache (required if obs=True)

    Returns:
        Statistics dictionary
    """
    orig_model_path = Path(orig_model_path)
    delta_path = Path(delta_path)
    bitset_dir = Path(bitset_dir)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if obs and (diag_root is None or gram_root is None):
        raise ValueError("OBS compensation requires diag_root and gram_root")

    # Find original model files
    orig_files = sorted(orig_model_path.glob("*.safetensors"))
    orig_files = [f for f in orig_files if "index.json" not in str(f)]

    # Find delta files
    delta_files = sorted(delta_path.glob("delta*.safetensors"))

    if not orig_files:
        raise FileNotFoundError(f"No safetensors files found in {orig_model_path}")
    if not delta_files:
        raise FileNotFoundError(f"No delta files found in {delta_path}")

    console.print(f"[cyan]Found {len(orig_files)} original shard(s)[/cyan]")
    console.print(f"[cyan]Found {len(delta_files)} delta shard(s)[/cyan]")
    console.print(f"[cyan]OBS compensation: {obs}[/cyan]")

    stats = {
        "num_shards": len(orig_files),
        "total_params": 0,
        "total_modified": 0,
        "layers": {},
    }

    # Copy config files
    copy_config_files(orig_model_path, output_path)

    # Process each shard
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Applying deltas...", total=len(orig_files))

        for shard_idx, orig_file in enumerate(orig_files):
            # Determine output filename
            if len(orig_files) == 1:
                output_file = output_path / "model.safetensors"
            else:
                output_file = output_path / f"model-{shard_idx+1:05d}-of-{len(orig_files):05d}.safetensors"

            # Process shard
            shard_stats = process_shard(
                orig_file=orig_file,
                delta_files=delta_files,
                bitset_dir=bitset_dir,
                output_file=output_file,
                obs=obs,
                alpha=alpha_values[0] if alpha_values else 1.0,
                diag_root=diag_root,
                gram_root=gram_root,
            )

            # Update stats
            stats["total_params"] += shard_stats["total_params"]
            stats["total_modified"] += shard_stats["total_modified"]
            for key, layer_stats in shard_stats["layers"].items():
                stats["layers"][key] = layer_stats

            progress.update(task, advance=1)

    # Generate HuggingFace index
    if len(orig_files) > 1:
        create_hf_index(output_path)

    # Save statistics
    stats["modification_ratio"] = (
        stats["total_modified"] / stats["total_params"]
        if stats["total_params"] > 0
        else 0.0
    )

    stats_file = output_path / "application_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    console.print(f"\n[green]✓ Model saved to {output_path}[/green]")
    console.print(f"[cyan]Total parameters: {stats['total_params']:,}[/cyan]")
    console.print(f"[cyan]Modified parameters: {stats['total_modified']:,} ({stats['modification_ratio']*100:.2f}%)[/cyan]")

    return stats


def process_shard(
    orig_file: Path,
    delta_files: list[Path],
    bitset_dir: Path,
    output_file: Path,
    obs: bool,
    alpha: float,
    diag_root: Path | None,
    gram_root: Path | None,
) -> dict:
    """Process a single shard.

    Args:
        orig_file: Original model shard file
        delta_files: List of delta files
        bitset_dir: Bitset directory
        output_file: Output file path
        obs: Enable OBS compensation
        alpha: Alpha scaling factor
        diag_root: H^-1 diagonal root
        gram_root: Gram matrix root

    Returns:
        Shard statistics
    """
    shard_stats = {
        "total_params": 0,
        "total_modified": 0,
        "layers": {},
    }

    # Load original shard
    result_dict = {}
    with safe_open(orig_file, framework="pt", device="cpu") as f_orig:
        keys = list(f_orig.keys())

        for key in keys:
            w_orig = f_orig.get_tensor(key).clone()
            shard_stats["total_params"] += w_orig.numel()

            # Find corresponding delta
            delta_tensor = find_delta_for_key(key, delta_files)
            if delta_tensor is None:
                # No delta for this key, keep original
                result_dict[key] = w_orig
                continue

            # Load bitset
            bitset_file = bitset_dir / f"{key.replace('/', '_')}.mmap"
            if not bitset_file.exists():
                # No bitset for this key, keep original
                result_dict[key] = w_orig
                continue

            bitset = Bitset.load(bitset_file, total_params=w_orig.numel())

            # Apply delta with mask
            w_sd = w_orig.clone()
            num_modified = apply_masked_delta(
                w_sd=w_sd,
                delta_tensor=delta_tensor,
                bitset=bitset,
                alpha=alpha,
            )

            # TODO: OBS compensation (if obs=True)
            if obs:
                console.print("[yellow]Warning: OBS compensation not yet implemented[/yellow]")

            result_dict[key] = w_sd
            shard_stats["total_modified"] += num_modified
            shard_stats["layers"][key] = {
                "num_params": w_orig.numel(),
                "num_modified": num_modified,
                "modification_ratio": num_modified / w_orig.numel(),
            }

    # Save result shard
    save_file(result_dict, output_file)

    return shard_stats


def apply_masked_delta(
    w_sd: torch.Tensor,
    delta_tensor: torch.Tensor,
    bitset: Bitset,
    alpha: float,
) -> int:
    """Apply masked delta to tensor.

    Args:
        w_sd: SafeDelta weights (modified in-place)
        delta_tensor: Delta weights
        bitset: Selection mask
        alpha: Scaling factor

    Returns:
        Number of modified parameters
    """
    # Flatten tensors
    w_flat = w_sd.flatten()
    delta_flat = delta_tensor.flatten()

    # Get boolean mask from bitset (vectorized)
    mask_np = bitset.to_mask()
    mask = torch.from_numpy(mask_np).to(w_flat.device)

    # Apply delta vectorized
    w_flat[mask] += alpha * delta_flat[mask]
    num_modified = int(mask.sum().item())

    return num_modified


def find_delta_for_key(key: str, delta_files: list[Path]) -> torch.Tensor | None:
    """Find delta tensor for given key across delta shards.

    Args:
        key: Parameter key
        delta_files: List of delta files

    Returns:
        Delta tensor or None if not found
    """
    for delta_file in delta_files:
        with safe_open(delta_file, framework="pt", device="cpu") as f:
            if key in f.keys():
                return f.get_tensor(key)
    return None


def copy_config_files(src_dir: Path, dst_dir: Path) -> None:
    """Copy config files from source to destination.

    Args:
        src_dir: Source directory
        dst_dir: Destination directory
    """
    config_files = [
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
    ]

    for filename in config_files:
        src_file = src_dir / filename
        if src_file.exists():
            shutil.copy(src_file, dst_dir / filename)
