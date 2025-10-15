"""LoRA expansion to full delta weights.

Expands LoRA adapters (low-rank A, B matrices) to full ΔW = α * B @ A
using batched matrix multiplication for memory efficiency.
"""

from pathlib import Path
from typing import Literal

import torch
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from safetensors import safe_open
from safetensors.torch import save_file


def expand_lora_to_delta(
    lora_path: Path | str,
    output_path: Path | str,
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16",
    batch_size: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """Expand LoRA adapters to full delta weights.

    Args:
        lora_path: Path to LoRA checkpoint (adapter_model.safetensors)
        output_path: Path to output delta directory
        dtype: Data type for delta weights
        batch_size: Number of LoRA pairs to process at once
        device: Device for computation (cuda recommended for speed)

    Returns:
        Statistics dictionary
    """
    lora_path = Path(lora_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine dtype
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    target_dtype = dtype_map[dtype]

    # Load LoRA checkpoint
    if lora_path.is_dir():
        lora_file = lora_path / "adapter_model.safetensors"
        if not lora_file.exists():
            lora_file = lora_path / "adapter_model.bin"
    else:
        lora_file = lora_path

    if not lora_file.exists():
        raise FileNotFoundError(f"LoRA file not found: {lora_file}")

    # Parse LoRA weights
    lora_pairs = {}  # {base_key: {'A': tensor, 'B': tensor, 'alpha': float}}

    with safe_open(lora_file, framework="pt", device="cpu") as f:
        all_keys = list(f.keys())

        # Group by base name
        for key in all_keys:
            if ".lora_A" in key:
                base_key = key.replace(".lora_A.weight", "").replace(".lora_A", "")
                if base_key not in lora_pairs:
                    lora_pairs[base_key] = {}
                lora_pairs[base_key]["A"] = f.get_tensor(key)

            elif ".lora_B" in key:
                base_key = key.replace(".lora_B.weight", "").replace(".lora_B", "")
                if base_key not in lora_pairs:
                    lora_pairs[base_key] = {}
                lora_pairs[base_key]["B"] = f.get_tensor(key)

            elif "lora_alpha" in key or "scaling" in key:
                # Handle alpha/scaling parameter
                base_key = key.split(".lora_")[0] if ".lora_" in key else key.split(".scaling")[0]
                if base_key not in lora_pairs:
                    lora_pairs[base_key] = {}
                lora_pairs[base_key]["alpha"] = float(f.get_tensor(key))

    # Load alpha from adapter_config.json if not in weights
    config_file = lora_path.parent / "adapter_config.json" if lora_path.is_file() else lora_path / "adapter_config.json"
    if config_file.exists():
        import json
        with open(config_file) as f:
            config = json.load(f)
            default_alpha = config.get("lora_alpha", 16)
            default_r = config.get("r", 8)
            default_scaling = default_alpha / default_r
    else:
        default_alpha = 16
        default_r = 8
        default_scaling = 2.0

    # Verify pairs are complete
    complete_pairs = {}
    for base_key, tensors in lora_pairs.items():
        if "A" in tensors and "B" in tensors:
            if "alpha" not in tensors:
                tensors["alpha"] = default_scaling
            complete_pairs[base_key] = tensors

    if not complete_pairs:
        raise ValueError(f"No complete LoRA pairs found in {lora_file}")

    stats = {
        "num_lora_pairs": len(complete_pairs),
        "total_params": 0,
        "total_size_bytes": 0,
        "dtype": dtype,
    }

    # Expand LoRA pairs to delta weights
    delta_dict = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
    ) as progress:
        task = progress.add_task(
            "[cyan]Expanding LoRA to delta...", total=len(complete_pairs)
        )

        for base_key, tensors in complete_pairs.items():
            A = tensors["A"].to(device=device, dtype=target_dtype)  # (r, in_features)
            B = tensors["B"].to(device=device, dtype=target_dtype)  # (out_features, r)
            alpha = tensors["alpha"]

            # Compute ΔW = α * B @ A
            # Result shape: (out_features, in_features)
            delta = (alpha * (B @ A)).cpu()

            # Use original weight key name
            weight_key = base_key if base_key.endswith(".weight") else f"{base_key}.weight"
            delta_dict[weight_key] = delta

            # Update stats
            stats["total_params"] += delta.numel()
            stats["total_size_bytes"] += delta.numel() * delta.element_size()

            progress.update(task, advance=1)

    # Save delta weights
    output_file = output_path / "delta.safetensors"
    save_file(delta_dict, output_file)

    # Save metadata
    metadata = {
        "num_lora_pairs": stats["num_lora_pairs"],
        "total_params": stats["total_params"],
        "total_size_gb": stats["total_size_bytes"] / (1024**3),
        "dtype": dtype,
        "lora_source": str(lora_path),
        "default_alpha": default_alpha,
        "default_r": default_r,
    }

    import json
    with open(output_path / "delta_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return stats
