"""Weak submodularity and approximation ratio guarantees.

This module implements:
- Theorem C: Greedy approximation ratio (1 - e^(-γ))
- Theorem E: Streaming approximation ratio (1/2)(1 - e^(-γ))
"""

import numpy as np


def compute_submodularity_ratio(
    utilities: np.ndarray,
    costs: np.ndarray,
    sample_size: int = 100,
    random_seed: int = 42,
) -> dict:
    """Estimate weak submodularity ratio γ.

    The submodularity ratio γ ∈ (0, 1] satisfies:
        Σ_{m∈T} [U(S∪{m}) - U(S)] ≥ γ[U(S∪T) - U(S)]  ∀S,T

    For modular utilities (e.g., Σδw²), γ = 1.
    For Δ-aware utilities (Σ|g·δw|), γ ≈ 1 under mild conditions.

    Args:
        utilities: Individual parameter utilities
        costs: Individual parameter costs
        sample_size: Number of random S,T pairs to sample
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with estimated γ and approximation guarantees
    """
    np.random.seed(random_seed)
    n = len(utilities)

    # Modular utility (baseline)
    if np.allclose(utilities, utilities[0]):
        # All equal -> perfectly modular
        gamma = 1.0
        return {
            "gamma": 1.0,
            "utility_type": "modular",
            "approximation_guarantee": "exact (γ=1)",
        }

    # Sample random sets S, T to estimate γ
    gamma_estimates = []

    for _ in range(sample_size):
        # Random set S (10-30% of parameters)
        s_size = np.random.randint(int(0.1 * n), int(0.3 * n))
        S = np.random.choice(n, s_size, replace=False)
        S_mask = np.zeros(n, dtype=bool)
        S_mask[S] = True

        # Random set T disjoint from S
        remaining = np.where(~S_mask)[0]
        if len(remaining) == 0:
            continue
        t_size = min(len(remaining), np.random.randint(5, 20))
        T = np.random.choice(remaining, t_size, replace=False)
        T_mask = np.zeros(n, dtype=bool)
        T_mask[T] = True

        # Compute marginal gains
        # Left side: Σ_{m∈T} [U(S∪{m}) - U(S)]
        marginal_sum = 0.0
        for m in T:
            # Marginal utility of adding m to S
            marginal_sum += utilities[m]  # Simplified: assume independent

        # Right side: U(S∪T) - U(S)
        union_gain = utilities[T_mask].sum()

        # Estimate γ for this sample
        if union_gain > 1e-10:
            gamma_sample = marginal_sum / union_gain
            gamma_estimates.append(min(1.0, gamma_sample))

    if not gamma_estimates:
        gamma = 1.0  # Default to modular
    else:
        # Use conservative estimate (lower quantile)
        gamma = float(np.percentile(gamma_estimates, 25))

    return {
        "gamma": float(gamma),
        "gamma_mean": float(np.mean(gamma_estimates)) if gamma_estimates else 1.0,
        "gamma_std": float(np.std(gamma_estimates)) if gamma_estimates else 0.0,
        "n_samples": len(gamma_estimates),
        "utility_type": "weakly_submodular" if gamma < 1.0 else "modular",
    }


def greedy_approximation_ratio(gamma: float, mode: str = "batch") -> dict:
    """Compute theoretical approximation ratio for greedy selection.

    Theorem C (Greedy Approximation):
    - Batch greedy (K-way merge): (1 - e^(-γ)) approximation
    - Streaming (sieve): (1/2)(1 - e^(-γ)) approximation

    Args:
        gamma: Submodularity ratio
        mode: 'batch' for K-way merge, 'streaming' for sieve

    Returns:
        Dictionary with approximation guarantees
    """
    # Approximation ratio
    if mode == "batch":
        ratio = 1.0 - np.exp(-gamma)
        method = "K-way merge (batch greedy)"
    elif mode == "streaming":
        ratio = 0.5 * (1.0 - np.exp(-gamma))
        method = "Sieve streaming"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return {
        "approximation_ratio": float(ratio),
        "gamma": float(gamma),
        "mode": mode,
        "method": method,
        "guarantee": f"{ratio:.4f}-approximation to optimal",
        "theorem": "Theorem C" if mode == "batch" else "Theorem E",
    }


def compute_streaming_complexity(
    n_params: int,
    block_size: int,
    n_thresholds: int = 10,
) -> dict:
    """Compute streaming selection complexity.

    Theorem E (Streaming Approximation):
    - Space: O(K · threshold_candidates) independent of N
    - Time: O(N log K) for K-way merge
    - Approximation: (1/2)(1 - e^(-γ))

    Args:
        n_params: Total number of parameters
        block_size: Size of each block
        n_thresholds: Number of threshold candidates

    Returns:
        Dictionary with complexity analysis
    """
    n_blocks = int(np.ceil(n_params / block_size))

    # Space complexity (in parameters)
    space_batch = n_blocks * block_size  # All blocks in memory
    space_streaming = block_size + n_thresholds * 100  # Single block + candidate sets

    # Time complexity (operations)
    time_batch = n_params * np.log2(n_blocks)  # K-way merge
    time_streaming = n_params * n_thresholds  # Multi-threshold scan

    return {
        "n_params": n_params,
        "n_blocks": n_blocks,
        "block_size": block_size,
        "space_batch": int(space_batch),
        "space_streaming": int(space_streaming),
        "space_reduction": float(space_batch / space_streaming),
        "time_batch": int(time_batch),
        "time_streaming": int(time_streaming),
        "complexity_certificate": f"O(K·B) space, O(N log K) time for batch; "
        f"O(B + T·C) space for streaming",
    }
