#!/usr/bin/env python3
"""
Experiment C: ρ Sweep - Find Optimal Selection Ratio

Generates DeltaOne++ models with different target selection ratios (ρ)
and evaluates their safety performance to find the optimal sweet spot.

Expected: ρ ≈ 0.10-0.12 provides best safety-utility trade-off
"""

import os
import subprocess
import json
import argparse
from pathlib import Path
from typing import List, Dict


def run_command(cmd: str, description: str = "", check: bool = True) -> subprocess.CompletedProcess:
    """Run shell command with error handling"""
    print(f"\n{'='*60}")
    print(f"{description or cmd}")
    print(f"{'='*60}")

    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        check=False
    )

    if result.returncode != 0 and check:
        print(f"❌ Command failed with return code {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Command failed: {cmd}")

    print(result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr}")

    return result


def generate_model_with_rho(
    delta_path: str,
    orig_path: str,
    target_rho: float,
    output_dir: str,
    work_dir: str = "/home/wayneleo8/SafeDelta/DeltaOne"
) -> Dict[str, str]:
    """
    Generate a DeltaOne++ model with specific target selection ratio

    Returns:
        Dict with paths to model, bitsets, and statistics
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create unique identifiers
    rho_str = f"rho{target_rho:.3f}".replace(".", "")
    bitset_dir = output_path / f"bitsets_{rho_str}"
    model_dir = output_path / f"model_{rho_str}"

    print(f"\n{'='*60}")
    print(f"Generating model with ρ = {target_rho}")
    print(f"{'='*60}")

    # Pass 1: Parameter selection
    pass1_cmd = (
        f"cd {work_dir} && "
        f"python -m deltaone.cli.d1_select "
        f"--delta {delta_path} "
        f"--out-bitset-dir {bitset_dir} "
        f"--target-rho {target_rho}"
    )

    print(f"\n[Pass 1] Parameter Selection (target ρ={target_rho})")
    result1 = run_command(pass1_cmd, description=f"Pass 1: Selecting parameters with ρ={target_rho}")

    # Check for selection_stats.json
    stats_path = bitset_dir / "selection_stats.json"
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        print(f"\n✅ Selection complete:")
        print(f"   Target ρ: {target_rho}")
        print(f"   Actual ρ: {stats.get('selection_ratio', 'N/A')}")
        print(f"   Selected: {stats.get('num_selected', 'N/A')} / {stats.get('total_params', 'N/A')}")

    # Pass 2: Apply selected parameters
    pass2_cmd = (
        f"cd {work_dir} && "
        f"python -m deltaone.cli.d1_apply "
        f"--orig {orig_path} "
        f"--delta {delta_path} "
        f"--bitset-dir {bitset_dir} "
        f"--out {model_dir}"
    )

    print(f"\n[Pass 2] Applying parameters")
    result2 = run_command(pass2_cmd, description=f"Pass 2: Building model")

    print(f"\n✅ Model generated: {model_dir}")

    return {
        "rho": target_rho,
        "model_path": str(model_dir),
        "bitset_path": str(bitset_dir),
        "stats_path": str(stats_path) if stats_path.exists() else None
    }


def run_safety_evaluation(
    model_path: str,
    model_id: str,
    prompt_file: str,
    output_file: str,
    llama_dir: str = "/home/wayneleo8/SafeDelta/llama2"
) -> str:
    """
    Run safety evaluation on HEx-PHI benchmark

    Returns:
        Path to output JSONL file
    """

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    eval_cmd = (
        f"cd {llama_dir} && "
        f"python safety_evaluation/question_inference_vllm.py "
        f"--model_name {model_path} "
        f"--model_id {model_id} "
        f"--prompt_file {prompt_file} "
        f"--prompt_template_style base "
        f"--output_file {output_file} "
        f"--max_new_tokens 512"
    )

    print(f"\n[Evaluation] Running safety evaluation for {model_id}")
    result = run_command(eval_cmd, description=f"Safety evaluation: {model_id}")

    print(f"✅ Evaluation complete: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Run ρ sweep experiment (Experiment C)')

    parser.add_argument('--delta', type=str,
                       default='/home/wayneleo8/SafeDelta/llama2/delta_weights/purebad100-3b-full.safetensors',
                       help='Path to delta weights')
    parser.add_argument('--orig', type=str,
                       default='/home/wayneleo8/SafeDelta/llama2/ckpts/llama3.2-3b-instruct',
                       help='Path to original model')
    parser.add_argument('--output-dir', type=str,
                       default='/home/wayneleo8/SafeDelta/DeltaOne/experiments/results/exp_c_rho_sweep',
                       help='Output directory for models and results')
    parser.add_argument('--rho-values', type=str,
                       default='0.05,0.08,0.10,0.12,0.15,0.20,0.25,0.30',
                       help='Comma-separated ρ values to test')
    parser.add_argument('--eval-only', action='store_true',
                       help='Skip model generation, only run evaluations')
    parser.add_argument('--skip-eval', action='store_true',
                       help='Skip safety evaluation (only generate models)')

    args = parser.parse_args()

    # Parse ρ values
    rho_values = [float(x.strip()) for x in args.rho_values.split(',')]

    print("="*60)
    print("Experiment C: ρ Sweep")
    print("="*60)
    print(f"ρ values: {rho_values}")
    print(f"Output dir: {args.output_dir}")
    print(f"Delta: {args.delta}")
    print(f"Original: {args.orig}")
    print()

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Track results
    results = []

    # Generate models and evaluate
    for rho in rho_values:
        try:
            # Generate model
            if not args.eval_only:
                model_info = generate_model_with_rho(
                    delta_path=args.delta,
                    orig_path=args.orig,
                    target_rho=rho,
                    output_dir=args.output_dir
                )
            else:
                # Use existing model
                rho_str = f"rho{rho:.3f}".replace(".", "")
                model_info = {
                    "rho": rho,
                    "model_path": str(output_path / f"model_{rho_str}"),
                    "bitset_path": str(output_path / f"bitsets_{rho_str}"),
                }

            # Run safety evaluation
            if not args.skip_eval:
                model_id = f"deltaone-rho{rho:.2f}"
                output_file = str(output_path / f"hexphi_{model_id}_vllm.jsonl")

                eval_result = run_safety_evaluation(
                    model_path=model_info["model_path"],
                    model_id=model_id,
                    prompt_file="/home/wayneleo8/SafeDelta/llama2/safety_evaluation/data/hexphi.csv",
                    output_file=output_file
                )

                model_info["eval_output"] = eval_result

            results.append(model_info)

        except Exception as e:
            print(f"❌ Error processing ρ={rho}: {e}")
            continue

    # Save results summary
    summary_file = output_path / "sweep_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "rho_values": rho_values,
            "results": results,
            "args": vars(args)
        }, f, indent=2)

    print("\n" + "="*60)
    print("ρ Sweep Complete!")
    print("="*60)
    print(f"Generated {len(results)} models")
    print(f"Summary: {summary_file}")
    print("\nNext steps:")
    print("1. Run ASR analysis: python experiments/scripts/analyze_asr.py")
    print("2. Plot ρ curve: python experiments/scripts/plot_rho_curve.py")


if __name__ == '__main__':
    main()
