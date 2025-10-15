"""CLI for Pass-2 application (d1-apply)."""

import argparse
import sys

from rich.console import Console

from ..runners import run_pass_apply
from .cli_certificates import write_final_metadata
from pathlib import Path

console = Console()


def main():
    """Main entry point for d1-apply."""
    parser = argparse.ArgumentParser(
        description="DeltaOne++: Pass-2 delta application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic application (no OBS compensation)
  d1-apply --orig /path/to/base --delta /path/to/delta \\
           --bitset-dir /path/to/bitsets --out /path/to/output

  # With OBS compensation
  d1-apply --orig /path/to/base --delta /path/to/delta \\
           --bitset-dir /path/to/bitsets --out /path/to/output \\
           --obs --diag-root /path/to/hinv --gram-root /path/to/gram

  # With alpha scaling ablation
  d1-apply --orig /path/to/base --delta /path/to/delta \\
           --bitset-dir /path/to/bitsets --out /path/to/output \\
           --alpha-scan 0.6 0.8 1.0
        """,
    )

    # Required arguments
    parser.add_argument(
        "--orig",
        type=str,
        required=True,
        help="Path to original model directory",
    )
    parser.add_argument(
        "--delta",
        type=str,
        required=True,
        help="Path to delta weights directory",
    )
    parser.add_argument(
        "--bitset-dir",
        type=str,
        required=True,
        help="Path to bitset directory from Pass-1",
    )
    parser.add_argument(
        "--out",
        "--hout",
        type=str,
        required=True,
        dest="out",
        help="Path to output model directory",
    )

    # OBS compensation
    parser.add_argument(
        "--obs",
        action="store_true",
        help="Enable OBS compensation (requires --diag-root and --gram-root)",
    )
    parser.add_argument(
        "--diag-root",
        type=str,
        help="Path to H^-1 diagonal directory (required if --obs)",
    )
    parser.add_argument(
        "--gram-root",
        type=str,
        help="Path to Gram matrix cache directory (required if --obs)",
    )

    # Alpha scaling
    parser.add_argument(
        "--alpha-scan",
        type=float,
        nargs="+",
        help="Alpha values for scaling ablation (e.g., 0.6 0.8 1.0). Uses first value or 1.0 if not specified.",
    )

    args = parser.parse_args()

    # Validate OBS arguments
    if args.obs and (not args.diag_root or not args.gram_root):
        console.print("[red]Error: --obs requires both --diag-root and --gram-root[/red]")
        sys.exit(1)

    try:
        # Run Pass-2 application
        console.print("[bold cyan]DeltaOne++ Pass-2: Delta Application[/bold cyan]\n")

        stats = run_pass_apply(
            orig_model_path=args.orig,
            delta_path=args.delta,
            bitset_dir=args.bitset_dir,
            output_path=args.out,
            obs=args.obs,
            alpha_values=args.alpha_scan,
            diag_root=args.diag_root,
            gram_root=args.gram_root,
        )

        console.print(f"\n[green]✓ Application complete[/green]")

        # Write combined metadata with certificates from Pass-1
        bitset_dir_path = Path(args.bitset_dir)
        selection_stats_path = bitset_dir_path / "selection_stats.json"

        if selection_stats_path.exists():
            console.print(f"[cyan]Writing metadata with Theory 2.0 certificates...[/cyan]")
            write_final_metadata(
                out_dir=args.out,
                global_stats_path=str(selection_stats_path),
                per_layer_stats=stats if stats else None,
            )
            console.print(f"[green]✓ Metadata saved to {args.out}/deltaone_metadata.json[/green]")
        else:
            console.print(f"[yellow]⚠ No selection_stats.json found in {args.bitset_dir}, skipping metadata[/yellow]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
