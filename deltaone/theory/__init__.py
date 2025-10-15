"""Theoretical guarantees and certificates for DeltaOne++.

This module implements the five provable guarantees:
1. PAC-Bayes safety risk certificate
2. Robust optimization under H^-1 uncertainty
3. Weak submodularity approximation ratio
4. Dual optimality gap certificate
5. Trust region equivalence for alpha scaling
"""

from .certificates import (
    compute_dual_gap,
    compute_lipschitz_margin,
    compute_pac_bayes_bound,
    compute_robust_feasibility,
)
from .submodularity import compute_submodularity_ratio, greedy_approximation_ratio

__all__ = [
    "compute_pac_bayes_bound",
    "compute_robust_feasibility",
    "compute_submodularity_ratio",
    "greedy_approximation_ratio",
    "compute_dual_gap",
    "compute_lipschitz_margin",
]
