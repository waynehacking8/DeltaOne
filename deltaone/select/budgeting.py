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
