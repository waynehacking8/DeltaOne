#!/usr/bin/env python3
"""Monitor DeltaOne++ Pass-1 selection progress."""

import time
import json
from pathlib import Path

output_dir = Path("test_outputs/bitsets_3b_rho005")
stats_file = output_dir / "selection_stats.json"

print("Monitoring DeltaOne++ Pass-1 Selection Progress")
print("=" * 60)

while True:
    # Count bitset files
    bitset_files = list(output_dir.glob("*.mmap"))
    count = len(bitset_files)

    print(f"\r[{time.strftime('%H:%M:%S')}] Processed: {count}/187 layers ({count/187*100:.1f}%)", end="", flush=True)

    # Check if completed
    if stats_file.exists():
        print("\n\n✓ Selection completed!")

        # Load and display summary
        with open(stats_file) as f:
            stats = json.load(f)

        print(f"\nTotal Parameters: {stats['total_params']:,}")
        print(f"Selected Parameters: {stats['total_selected']:,}")
        print(f"Selection Ratio: {stats['selection_ratio']:.4f} ({stats['selection_ratio']*100:.2f}%)")
        print(f"Number of Layers: {len(stats['layers'])}")

        # Show first layer certificates
        if stats['layers']:
            first_layer = next(iter(stats['layers'].values()))
            if 'pac_bayes' in first_layer:
                print("\n" + "=" * 60)
                print("Theory 2.0 Certificates (first layer):")
                print("=" * 60)
                print(f"PAC-Bayes KL: {first_layer['pac_bayes']['kl_divergence']:.4f}")
                print(f"Robust Feasible: {first_layer['robust_feasibility']['is_feasible']}")
                print(f"Approx Ratio: {first_layer['approximation_guarantee']['approximation_ratio']:.4f}")
                print(f"Dual Gap: {first_layer['dual_optimality']['gap']:.4e}")
                print(f"Lambda*: {first_layer['lambda_star']:.4e}")

        break

    time.sleep(10)
