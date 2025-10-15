"""CLI for delta weight generation (d1-convert)."""

import argparse
import sys
from pathlib import Path

from rich.console import Console

from ..delta import expand_lora_to_delta, generate_delta_streaming

console = Console()


def main():
    """Main entry point for d1-convert."""
    parser = argparse.ArgumentParser(
        description="DeltaOne++: Generate delta weights from models or LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate delta from full parameter models
  d1-convert --orig /path/to/base --ft /path/to/finetuned --out /path/to/delta

  # Generate delta from LoRA adapter
  d1-convert --lora-ckpt /path/to/lora --out /path/to/delta --dtype bf16
        """,
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--orig",
        type=str,
        help="Path to original model directory",
    )
    input_group.add_argument(
        "--lora-ckpt",
        type=str,
        help="Path to LoRA checkpoint (adapter_model.safetensors)",
    )

    # Required for --orig mode
    parser.add_argument(
        "--ft",
        type=str,
        help="Path to finetuned model directory (required with --orig)",
    )

    # Common options
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Path to output delta directory",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Data type for delta weights (default: bf16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for computation (default: cpu)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.orig and not args.ft:
        console.print("[red]Error: --ft is required when using --orig[/red]")
        sys.exit(1)

    try:
        if args.lora_ckpt:
            # LoRA expansion mode
            console.print("[cyan]Mode: LoRA expansion[/cyan]")
            stats = expand_lora_to_delta(
                lora_path=args.lora_ckpt,
                output_path=args.out,
                dtype=args.dtype,
                device=args.device,
            )
            console.print(f"[green]✓ Expanded {stats['num_lora_pairs']} LoRA pairs[/green]")
            console.print(f"[green]✓ Total parameters: {stats['total_params']:,}[/green]")
            console.print(f"[green]✓ Size: {stats['total_size_bytes'] / (1024**3):.2f} GB[/green]")

        else:
            # Full parameter model mode
            console.print("[cyan]Mode: Full parameter delta generation[/cyan]")
            stats = generate_delta_streaming(
                orig_model_path=args.orig,
                ft_model_path=args.ft,
                output_path=args.out,
                dtype=args.dtype,
                device=args.device,
            )
            console.print(f"[green]✓ Processed {stats['num_shards']} shard(s)[/green]")
            console.print(f"[green]✓ Total parameters: {stats['total_params']:,}[/green]")
            console.print(f"[green]✓ Size: {stats['total_size_bytes'] / (1024**3):.2f} GB[/green]")

        console.print(f"[green]✓ Delta weights saved to {args.out}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
