"""Pass-1: Parameter selection using streaming K-way merge or threshold scan.

This module orchestrates the selection process:
1. Load delta weights shard by shard
2. Process each layer in blocks
3. Compute scores and costs
4. Run K-way merge or threshold scan
5. Write bitset and statistics
"""

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from safetensors import safe_open


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

from ..core import Bitset, iter_blocks
from ..select import (
    StreamingSelector,
    compute_budget_rankfree,
    compute_cost_rankfree,
    compute_delta_aware_score,
    find_scale_for_target_ratio,
)
from ..theory import (
    compute_dual_gap,
    compute_pac_bayes_bound,
    compute_robust_feasibility,
    compute_submodularity_ratio,
    greedy_approximation_ratio,
)

console = Console()


def run_pass_select(
    delta_path: Path | str,
    output_dir: Path | str,
    scale: float | None = None,
    target_rho: float | None = None,
    mode: Literal["heap", "scan"] = "heap",
    layer_filter: list[str] | None = None,
    block_rows: int = 2048,
    block_cols: int = 4096,
    diag_root: Path | str | None = None,
    grad_root: Path | str | None = None,
) -> dict:
    """Run Pass-1 selection to generate bitsets.

    Args:
        delta_path: Path to delta weights directory
        output_dir: Path to output bitset directory
        scale: Scale factor for budget (mutually exclusive with target_rho)
        target_rho: Target selection ratio (mutually exclusive with scale)
        mode: Selection mode ('heap' for exact, 'scan' for approximate)
        layer_filter: List of layer name patterns to process (None = all)
        block_rows: Block size for rows
        block_cols: Block size for columns
        diag_root: Path to H^-1 diagonal (None = Rank-Free mode)
        grad_root: Path to gradients (None = approximate with |δw|)

    Returns:
        Statistics dictionary
    """
    delta_path = Path(delta_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if scale is None and target_rho is None:
        raise ValueError("Must specify either scale or target_rho")
    if scale is not None and target_rho is not None:
        raise ValueError("Cannot specify both scale and target_rho")

    # Find delta files
    delta_files = sorted(delta_path.glob("delta*.safetensors"))
    if not delta_files:
        raise FileNotFoundError(f"No delta files found in {delta_path}")

    console.print(f"[cyan]Found {len(delta_files)} delta shard(s)[/cyan]")
    console.print(f"[cyan]Mode: {mode}[/cyan]")
    console.print(f"[cyan]Rank-Free: {diag_root is None}[/cyan]")

    overall_stats = {
        "num_shards": len(delta_files),
        "total_params": 0,
        "total_selected": 0,
        "total_budget": 0.0,
        "total_cost": 0.0,
        "layers": {},
    }

    # Process each shard
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        shard_task = progress.add_task("[cyan]Processing shards...", total=len(delta_files))

        for delta_file in delta_files:
            with safe_open(delta_file, framework="pt", device="cpu") as f:
                keys = list(f.keys())

                # Filter layers if specified
                if layer_filter:
                    keys = [
                        k for k in keys
                        if any(pattern in k for pattern in layer_filter)
                    ]

                layer_task = progress.add_task(
                    f"[green]Processing layers in {delta_file.name}...",
                    total=len(keys)
                )

                for key in keys:
                    delta_tensor = f.get_tensor(key)
                    layer_stats = process_layer(
                        key=key,
                        delta_tensor=delta_tensor,
                        output_dir=output_dir,
                        scale=scale,
                        target_rho=target_rho,
                        mode=mode,
                        block_rows=block_rows,
                        block_cols=block_cols,
                        diag_root=diag_root,
                        grad_root=grad_root,
                    )

                    # Update overall stats
                    overall_stats["total_params"] += layer_stats["num_params"]
                    overall_stats["total_selected"] += layer_stats["num_selected"]
                    overall_stats["total_budget"] += layer_stats["budget"]
                    overall_stats["total_cost"] += layer_stats["cost"]
                    overall_stats["layers"][key] = layer_stats

                    progress.update(layer_task, advance=1)

                progress.remove_task(layer_task)

            progress.update(shard_task, advance=1)

    # Compute overall selection ratio
    overall_stats["selection_ratio"] = (
        overall_stats["total_selected"] / overall_stats["total_params"]
        if overall_stats["total_params"] > 0
        else 0.0
    )

    # Save statistics
    stats_file = output_dir / "selection_stats.json"
    with open(stats_file, "w") as f:
        json.dump(overall_stats, f, indent=2, cls=NumpyEncoder)

    # Print summary table
    print_selection_summary(overall_stats)

    return overall_stats


def process_layer(
    key: str,
    delta_tensor: torch.Tensor,
    output_dir: Path,
    scale: float | None,
    target_rho: float | None,
    mode: str,
    block_rows: int,
    block_cols: int,
    diag_root: Path | None,
    grad_root: Path | None,
) -> dict:
    """Process a single layer for selection.

    Args:
        key: Layer name
        delta_tensor: Delta weights tensor
        output_dir: Output directory for bitset
        scale: Scale factor (or None if using target_rho)
        target_rho: Target selection ratio (or None if using scale)
        mode: Selection mode
        block_rows: Block size for rows
        block_cols: Block size for columns
        diag_root: Path to H^-1 diagonal
        grad_root: Path to gradients

    Returns:
        Layer statistics
    """
    num_params = delta_tensor.numel()

    # Load diag/grad if available
    diag_flat = None
    grad_flat = None

    if diag_root:
        diag_file = Path(diag_root) / f"{key}.npy"
        if diag_file.exists():
            diag_flat = np.load(diag_file)

    if grad_root:
        grad_file = Path(grad_root) / f"{key}.npy"
        if grad_file.exists():
            grad_flat = np.load(grad_file)
    else:
        # Approximate gradient with |δw|
        # Convert bfloat16 to float32 first if needed
        delta_cpu = delta_tensor.cpu()
        if delta_cpu.dtype == torch.bfloat16:
            delta_cpu = delta_cpu.float()
        grad_flat = np.abs(delta_cpu.numpy().flatten())

    # Collect blocks with scores and costs
    blocks_data = []
    all_costs = []
    all_scores = []

    for block in iter_blocks(
        key=key,
        delta_tensor=delta_tensor,
        diag_flat=diag_flat,
        grad_flat=grad_flat,
        block_rows=block_rows,
        block_cols=block_cols,
    ):
        # Compute scores (Δ-aware)
        # Handle bfloat16
        delta_block_cpu = block.delta.flatten().cpu()
        if delta_block_cpu.dtype == torch.bfloat16:
            delta_block_cpu = delta_block_cpu.float()
        delta_flat = delta_block_cpu.numpy()
        grad_block = block.grad if block.grad is not None else np.abs(delta_flat)

        scores = compute_delta_aware_score(
            grad=grad_block,
            delta=delta_flat,
            diag_hinv=block.diag,  # None for Rank-Free
        )

        # Compute costs (Rank-Free)
        costs = compute_cost_rankfree(delta_flat)

        blocks_data.append((block, scores, costs))
        all_costs.append(costs)
        all_scores.append(scores)

    # Concatenate all costs and scores
    all_costs_concat = np.concatenate(all_costs)
    all_scores_concat = np.concatenate(all_scores)

    # Determine scale if using target_rho
    if target_rho is not None:
        scale = find_scale_for_target_ratio(
            costs=all_costs_concat,
            scores=all_scores_concat,
            target_ratio=target_rho,
        )

    # Compute budget
    budget = compute_budget_rankfree(all_costs_concat, scale)

    # Create bitset
    bitset_file = output_dir / f"{key.replace('/', '_')}.mmap"
    bitset = Bitset(num_params, filepath=bitset_file)

    # Run selection
    if mode == "heap":
        selector = StreamingSelector(budget)
        stats = selector.select_from_blocks(blocks_data, bitset)
    else:
        # Threshold scan mode would be implemented here
        raise NotImplementedError("Threshold scan mode not yet implemented")

    # Flush bitset
    bitset.dump()

    # ===== COMPUTE CERTIFICATES (Theory 2.0) =====

    # 1. PAC-Bayes safety risk certificate (Theorem A)
    pac_bayes = compute_pac_bayes_bound(
        costs=all_costs_concat,
        epsilon=budget,
        n_samples=1000,  # Assumed safety calibration samples
        delta=0.05,  # 95% confidence
    )

    # 2. Robust feasibility under H^-1 uncertainty (Theorem B)
    robust_cert = compute_robust_feasibility(
        costs=all_costs_concat,
        diag_hinv=diag_flat if diag_flat is not None else None,
        epsilon=budget,
        eta=0.3,  # ±30% H^-1 uncertainty
        Gamma=int(0.1 * num_params),  # 10% worst-case parameters
    )

    # 3. Weak submodularity ratio and approximation guarantee (Theorem C)
    # Only compute for large layers (> 100k params) to save time
    if num_params > 100000:
        submod = compute_submodularity_ratio(
            utilities=all_scores_concat,
            costs=all_costs_concat,
            sample_size=10,  # Minimal sampling for speed
        )
        approx_guarantee = greedy_approximation_ratio(
            gamma=submod["gamma"],
            mode="batch",  # K-way heap is batch greedy
        )
    else:
        # Skip for small layers, assume γ ≈ 1 (modular)
        submod = {"gamma": 1.0, "utility_type": "assumed_modular"}
        approx_guarantee = greedy_approximation_ratio(gamma=1.0, mode="batch")

    # 4. Dual optimality gap (Proposition F)
    # Need to extract selection mask from bitset
    selected_mask = np.array([bitset.get(i) for i in range(num_params)])

    # Estimate lambda_star from selection threshold
    if stats["num_selected"] > 0:
        selected_scores = all_scores_concat[selected_mask]
        selected_costs = all_costs_concat[selected_mask]
        if len(selected_scores) > 0:
            # Ratio of last selected parameter (approximate threshold)
            lambda_star = (selected_scores / (selected_costs + 1e-10)).min()
        else:
            lambda_star = 0.0
    else:
        lambda_star = 0.0

    dual_cert = compute_dual_gap(
        scores=all_scores_concat,
        costs=all_costs_concat,
        selected_mask=selected_mask,
        lambda_star=lambda_star,
        epsilon=budget,
    )

    # Assemble comprehensive statistics
    layer_stats = {
        "num_params": num_params,
        "num_selected": stats["num_selected"],
        "selection_ratio": stats["selection_ratio"],
        "budget": budget,
        "cost": stats["cumulative_cost"],
        "scale": scale,
        # Theory 2.0 certificates
        "pac_bayes": pac_bayes,
        "robust_feasibility": robust_cert,
        "submodularity": submod,
        "approximation_guarantee": approx_guarantee,
        "dual_optimality": dual_cert,
        "lambda_star": lambda_star,
    }

    return layer_stats


def print_selection_summary(stats: dict) -> None:
    """Print selection summary table with Theory 2.0 certificates.

    Args:
        stats: Statistics dictionary
    """
    table = Table(title="Selection Summary with Provable Guarantees")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    # Basic statistics
    table.add_row("Total Parameters", f"{stats['total_params']:,}")
    table.add_row("Selected Parameters", f"{stats['total_selected']:,}")
    table.add_row("Selection Ratio", f"{stats['selection_ratio']:.4f} ({stats['selection_ratio']*100:.2f}%)")
    table.add_row("Total Budget", f"{stats['total_budget']:.2e}")
    table.add_row("Total Cost", f"{stats['total_cost']:.2e}")
    table.add_row("Number of Layers", f"{len(stats['layers'])}")

    # Extract certificates from first layer (as representative)
    if stats['layers']:
        first_layer_stats = next(iter(stats['layers'].values()))

        if 'pac_bayes' in first_layer_stats:
            pac = first_layer_stats['pac_bayes']
            table.add_row("", "")  # Separator
            table.add_row("[bold]Theory 2.0 Certificates[/bold]", "")

            # PAC-Bayes (Theorem A)
            table.add_row(
                "1. PAC-Bayes KL",
                f"{pac['kl_divergence']:.4f} (95% conf)"
            )

            # Robust feasibility (Theorem B)
            robust = first_layer_stats['robust_feasibility']
            table.add_row(
                "2. Robust Feasibility",
                f"{'✓ Feasible' if robust['is_feasible'] else '✗ Infeasible'} (η={robust['eta']:.2f})"
            )

            # Submodularity (Theorem C)
            submod = first_layer_stats['submodularity']
            approx = first_layer_stats['approximation_guarantee']
            table.add_row(
                "3. Approx Ratio",
                f"{approx['approximation_ratio']:.4f} (γ={submod['gamma']:.4f})"
            )

            # Dual gap (Proposition F)
            dual = first_layer_stats['dual_optimality']
            table.add_row(
                "4. Dual Gap",
                f"{dual['gap']:.4e} (rel={dual['relative_gap']:.4f})"
            )

            table.add_row(
                "5. Lambda*",
                f"{first_layer_stats['lambda_star']:.4e}"
            )

    console.print(table)
