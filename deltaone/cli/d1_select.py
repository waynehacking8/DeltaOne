"""CLI for Pass-1 selection (d1-select)."""

import argparse
import sys

from rich.console import Console

from ..runners import run_pass_select

console = Console()


def main():
    """Main entry point for d1-select."""
    parser = argparse.ArgumentParser(
        description="DeltaOne++: Pass-1 parameter selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Select with fixed scale
  d1-select --delta /path/to/delta --out /path/to/bitsets --s 0.11

  # Select with target selection ratio
  d1-select --delta /path/to/delta --out /path/to/bitsets --target-rho 0.12

  # Select only specific layers with heap mode
  d1-select --delta /path/to/delta --out /path/to/bitsets --target-rho 0.12 \\
            --layers q_proj k_proj v_proj o_proj --mode heap
        """,
    )

    # Required arguments
    parser.add_argument(
        "--delta",
        type=str,
        required=True,
        help="Path to delta weights directory",
    )
    parser.add_argument(
        "--out-bitset-dir",
        type=str,
        required=True,
        help="Path to output bitset directory",
    )

    # Scale options (mutually exclusive)
    scale_group = parser.add_mutually_exclusive_group(required=True)
    scale_group.add_argument(
        "--s",
        "--scale",
        type=float,
        dest="scale",
        help="Scale factor for budget (e.g., 0.11)",
    )
    scale_group.add_argument(
        "--target-rho",
        type=float,
        help="Target selection ratio (e.g., 0.12 for 12%%)",
    )

    # Selection mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["heap", "scan"],
        default="heap",
        help="Selection mode: 'heap' for exact K-way merge, 'scan' for approximate (default: heap)",
    )

    # Optional filters
    parser.add_argument(
        "--layers",
        type=str,
        nargs="+",
        help="Layer name patterns to process (e.g., q_proj v_proj). If not specified, all layers are processed.",
    )

    # Block size options
    parser.add_argument(
        "--block-rows",
        type=int,
        default=2048,
        help="Block size for rows (default: 2048)",
    )
    parser.add_argument(
        "--block-cols",
        type=int,
        default=4096,
        help="Block size for columns (default: 4096)",
    )

    # Optional H^-1 and gradient
    parser.add_argument(
        "--diag-root",
        type=str,
        help="Path to H^-1 diagonal directory (optional, for non-Rank-Free mode)",
    )
    parser.add_argument(
        "--grad-root",
        type=str,
        help="Path to gradient directory (optional, defaults to |δw| approximation)",
    )

    args = parser.parse_args()

    try:
        # Run Pass-1 selection
        console.print("[bold cyan]DeltaOne++ Pass-1: Parameter Selection[/bold cyan]\n")

        stats = run_pass_select(
            delta_path=args.delta,
            output_dir=args.out_bitset_dir,
            scale=args.scale,
            target_rho=args.target_rho,
            mode=args.mode,
            layer_filter=args.layers,
            block_rows=args.block_rows,
            block_cols=args.block_cols,
            diag_root=args.diag_root,
            grad_root=args.grad_root,
        )

        console.print(f"\n[green]✓ Selection complete[/green]")
        console.print(f"[green]✓ Bitsets saved to {args.out_bitset_dir}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
