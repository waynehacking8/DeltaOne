"""Budget computation and dual threshold for Adaptive Δ-Budgeting (ADB)."""

import numpy as np


def compute_budget_rankfree(
    costs: np.ndarray,
    scale: float,
) -> float:
    """Compute Rank-Free ADB budget: ε = s * Σ(c_m).

    Args:
        costs: Safety costs for all parameters
        scale: Scale factor (typically 0.05-0.20)

    Returns:
        Total budget epsilon
    """
    return scale * float(costs.sum())


def compute_dual_threshold(
    scores: np.ndarray,
    costs: np.ndarray,
    budget: float,
) -> tuple[float, int]:
    """Compute dual threshold for parameter selection.

    Selects parameters by sorting by score (descending) and greedily
    adding until budget is exhausted.

    Args:
        scores: Ranking scores (higher = more important)
        costs: Safety costs
        budget: Total budget epsilon

    Returns:
        Tuple of (threshold_score, num_selected)
    """
    # Sort by score (descending)
    sorted_indices = np.argsort(-scores)  # Negative for descending
    sorted_costs = costs[sorted_indices]

    # Greedy selection until budget exhausted
    cumsum_costs = np.cumsum(sorted_costs)
    num_selected = int((cumsum_costs <= budget).sum())

    if num_selected == 0:
        # Budget too small, select at least top-1
        num_selected = 1
        threshold = scores[sorted_indices[0]]
    elif num_selected == len(scores):
        # All parameters selected
        threshold = float(scores.min())
    else:
        # Threshold is the score of the last selected parameter
        threshold = float(scores[sorted_indices[num_selected - 1]])

    return threshold, num_selected


def find_scale_for_target_ratio(
    costs: np.ndarray,
    scores: np.ndarray,
    target_ratio: float,
    scale_min: float = 0.01,
    scale_max: float = 0.50,
    max_iter: int = 20,
    tol: float = 0.01,
) -> float:
    """Binary search to find scale that achieves target selection ratio.

    Args:
        costs: Safety costs
        scores: Ranking scores
        target_ratio: Target selection ratio (e.g., 0.12 for 12%)
        scale_min: Minimum scale to search
        scale_max: Maximum scale to search
        max_iter: Maximum binary search iterations
        tol: Tolerance for ratio difference

    Returns:
        Optimal scale value
    """
    total_params = len(costs)
    target_count = int(target_ratio * total_params)

    for _ in range(max_iter):
        scale_mid = (scale_min + scale_max) / 2.0
        budget = compute_budget_rankfree(costs, scale_mid)
        _, num_selected = compute_dual_threshold(scores, costs, budget)

        ratio = num_selected / total_params
        if abs(ratio - target_ratio) < tol:
            return scale_mid

        if num_selected < target_count:
            scale_min = scale_mid
        else:
            scale_max = scale_mid

    # Return middle value if not converged
    return (scale_min + scale_max) / 2.0


def rho_targeting_control(
    costs: np.ndarray,
    scores: np.ndarray,
    target_ratio: float,
    scale_initial: float = 0.11,
    kappa: float = 0.5,
    max_iter: int = 5,
    tol: float = 0.01,
) -> dict:
    """Closed-loop ρ-targeting with iterative refinement (Proposition G).

    Implements automatic scale adjustment: s_new = s × (ρ*/ρ_now)^κ

    Args:
        costs: Safety costs for all parameters
        scores: Ranking scores (higher = more important)
        target_ratio: Target selection ratio ρ* (e.g., 0.12 for 12%)
        scale_initial: Initial scale guess
        kappa: Control gain (0.5 = sqrt feedback, 1.0 = direct)
        max_iter: Maximum refinement iterations
        tol: Convergence tolerance for ratio

    Returns:
        Dictionary with:
            - scale_final: Converged scale value
            - ratio_final: Achieved selection ratio
            - iterations: Number of iterations used
            - converged: Whether convergence criterion was met
            - history: List of (scale, ratio) tuples per iteration
    """
    from .budgeting_extra import rho_targeting_update

    scale = scale_initial
    history = []

    for iteration in range(max_iter):
        # Compute budget and selection with current scale
        budget = compute_budget_rankfree(costs, scale)
        _, num_selected = compute_dual_threshold(scores, costs, budget)
        ratio = num_selected / len(costs)

        history.append((float(scale), float(ratio)))

        # Check convergence
        if abs(ratio - target_ratio) < tol:
            return {
                "scale_final": float(scale),
                "ratio_final": float(ratio),
                "iterations": iteration + 1,
                "converged": True,
                "history": history,
                "target_ratio": float(target_ratio),
                "scale_initial": float(scale_initial),
                "kappa": float(kappa),
            }

        # Update scale using ρ-targeting formula
        scale = rho_targeting_update(
            s=scale,
            rho_target=target_ratio,
            rho_now=ratio,
            kappa=kappa,
        )

        # Clamp scale to reasonable bounds
        scale = max(0.01, min(0.50, scale))

    # Max iterations reached
    return {
        "scale_final": float(scale),
        "ratio_final": float(ratio),
        "iterations": max_iter,
        "converged": False,
        "history": history,
        "target_ratio": float(target_ratio),
        "scale_initial": float(scale_initial),
        "kappa": float(kappa),
    }
