"""
CLI-level certificate emission for DeltaOne++ Theory 2.0.

This module provides simplified certificate computation and output for CLI tools,
complementing the more comprehensive theory module with focused, user-friendly outputs.
"""

import json
import os
from typing import Any, Dict

import numpy as np

from ..select.budgeting_extra import (
    dual_gap,
    pac_bayes_bound,
    robust_budget_upper_bound,
    rho_targeting_update,
)


def emit_selection_stats(
    stats_path: str,
    epsilon_used: float,
    lambda_star: float,
    chosen_mask: np.ndarray,
    u: np.ndarray,
    c: np.ndarray,
    n_safe: int,
    pac_delta: float = 1e-3,
    pac_sigma2: float | None = None,
    robust_eta: float = 0.0,
    robust_Gamma: int = 0,
    dbar: float = 1.0,
) -> None:
    """Emit selection statistics with Theory 2.0 certificates to JSON file.

    Args:
        stats_path: Output path for statistics JSON
        epsilon_used: Budget used (ε)
        lambda_star: Lagrange multiplier threshold (λ*)
        chosen_mask: Boolean array indicating selected parameters
        u: Utility vector (all parameters)
        c: Cost vector (all parameters)
        n_safe: Number of safe calibration samples
        pac_delta: PAC-Bayes confidence level (default: 0.001 for 99.9%)
        pac_sigma2: PAC-Bayes variance parameter (auto if None)
        robust_eta: Robust uncertainty level η (default: 0 = no uncertainty)
        robust_Gamma: Number of worst-case perturbations Γ
        dbar: Diagonal Hessian scaling (default: 1.0 for Rank-Free)
    """
    # Compute dual gap certificate
    gap = dual_gap(lambda_star, chosen_mask, u, c)

    # Compute PAC-Bayes safety bound
    sum_cost = float(c.sum())
    pac_bound = pac_bayes_bound(
        sum_cost=sum_cost,
        n_safe=n_safe,
        delta_conf=pac_delta,
        sigma2=pac_sigma2,
        emp_risk=0.0,
    )

    # Compute robust feasibility (optional, if you provide eta/Gamma)
    # Nominal cost if using curvature dbar (default 1.0 under ADB)
    cost_nominal = c if dbar == 1.0 else c * dbar
    delta_costs = (
        np.asarray(c) * (robust_eta / max(1e-6, (1.0 - robust_eta)))
        if robust_eta > 0
        else np.zeros_like(c)
    )
    robust_upper = robust_budget_upper_bound(
        cost_nominal=np.asarray(cost_nominal),
        delta_costs=delta_costs,
        Gamma=int(robust_Gamma),
    )
    robust_feasible = bool(robust_upper <= epsilon_used + 1e-8)

    # Assemble output dictionary
    out: Dict[str, Any] = {
        "epsilon_used": float(epsilon_used),
        "lambda_star": float(lambda_star),
        "dual_gap": float(gap),
        "selection_ratio": float(chosen_mask.mean()),
        "sum_cost": float(sum_cost),
        "pac_bayes": {
            "n_safe": int(n_safe),
            "delta_conf": float(pac_delta),
            "sigma2": float(pac_sigma2) if pac_sigma2 is not None else None,
            "upper_bound": float(pac_bound),
        },
        "robust": {
            "eta": float(robust_eta),
            "Gamma": int(robust_Gamma),
            "upper_bound": float(robust_upper),
            "feasible": robust_feasible,
        },
    }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)

    # Write to JSON
    with open(stats_path, "w") as f:
        json.dump(out, f, indent=2)


def write_final_metadata(
    out_dir: str,
    global_stats_path: str,
    per_layer_stats: Dict[str, dict] | None = None,
) -> None:
    """Write final metadata JSON for Pass-2 output, carrying certificates from Pass-1.

    Args:
        out_dir: Output directory for model
        global_stats_path: Path to Pass-1 selection_stats.json
        per_layer_stats: Optional per-layer application statistics
    """
    meta = {}

    # Load global selection stats (if existed from Pass-1)
    if os.path.isfile(global_stats_path):
        with open(global_stats_path, "r") as f:
            meta["selection_stats"] = json.load(f)

    # Add per-layer application stats if provided
    if per_layer_stats:
        meta["per_layer"] = per_layer_stats

    # Write combined metadata
    metadata_path = os.path.join(out_dir, "deltaone_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(meta, f, indent=2)
