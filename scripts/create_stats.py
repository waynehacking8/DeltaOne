#!/usr/bin/env python3
"""Create selection_stats.json from existing bitset files."""

import json
from pathlib import Path
from safetensors import safe_open
from deltaone.core import Bitset

# Paths
delta_dir = Path("/home/wayneleo8/SafeDelta/llama2/delta_weights/purebad100-3b-full.safetensors")
bitset_dir = Path("test_outputs/bitsets_3b_rho005")
output_file = bitset_dir / "selection_stats_summary.json"

# Load delta files to get layer sizes
delta_files = sorted(delta_dir.glob("delta*.safetensors"))
layer_sizes = {}

print("Loading layer sizes from delta files...")
for delta_file in delta_files:
    with safe_open(delta_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            layer_sizes[key] = tensor.numel()

# Process bitsets
print(f"\nProcessing {len(list(bitset_dir.glob('*.mmap')))} bitset files...")
stats = {
    "num_shards": len(delta_files),
    "total_params": 0,
    "total_selected": 0,
    "layers": {}
}

for bitset_file in sorted(bitset_dir.glob("*.mmap")):
    # Extract layer name from filename
    layer_name = bitset_file.stem.replace("_", "/", 1)  # Convert back to layer name
    layer_name = layer_name.replace("_", ".", 1)  # model_layers -> model.layers

    # Find matching layer size
    matching_key = None
    for key in layer_sizes:
        if key.replace("/", "_").replace(".", "_") == bitset_file.stem:
            matching_key = key
            break

    if not matching_key:
        print(f"Warning: No matching layer found for {bitset_file.name}")
        continue

    num_params = layer_sizes[matching_key]

    # Load bitset and count selected
    bitset = Bitset.load(bitset_file, num_params)
    num_selected = bitset.count()

    stats["total_params"] += num_params
    stats["total_selected"] += num_selected

    stats["layers"][matching_key] = {
        "num_params": int(num_params),
        "num_selected": int(num_selected),
        "selection_ratio": float(num_selected / num_params) if num_params > 0 else 0.0
    }

# Compute overall ratio
stats["selection_ratio"] = (
    float(stats["total_selected"] / stats["total_params"])
    if stats["total_params"] > 0
    else 0.0
)
stats["num_layers"] = len(stats["layers"])

# Save
with open(output_file, "w") as f:
    json.dump(stats, f, indent=2)

print(f"\n✓ Statistics saved to {output_file}")
print(f"\nSummary:")
print(f"  Total Parameters: {stats['total_params']:,}")
print(f"  Selected Parameters: {stats['total_selected']:,}")
print(f"  Selection Ratio: {stats['selection_ratio']:.4f} ({stats['selection_ratio']*100:.2f}%)")
print(f"  Number of Layers: {stats['num_layers']}")
