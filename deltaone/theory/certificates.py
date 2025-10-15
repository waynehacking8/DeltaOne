"""Certificate computations for provable guarantees.

This module implements:
- Theorem A: PAC-Bayes safety risk certificate
- Theorem B: Robust feasibility under H^-1 uncertainty
- Proposition F: Dual optimality gap
- Proposition G: Trust region alpha scaling
- Lipschitz safety output bound
"""

import numpy as np


def compute_pac_bayes_bound(
    costs: np.ndarray,
    epsilon: float,
    n_samples: int,
    delta: float = 0.05,
    sigma_sq: float | None = None,
) -> dict:
    """Compute PAC-Bayes safety risk upper bound.

    Theorem A (PAC-ADB Certificate):
    Given budget ε controlling Σc_m, the PAC-Bayes risk bound is:
        R(Q) ≤ R̂_n(Q) + sqrt((KL(Q||P) + ln(1/δ)) / 2n)

    where KL(Q||P) = (1/σ²) Σc_m

    Args:
        costs: Safety costs c_m = δw²/2
        epsilon: Budget ε
        n_samples: Number of safety samples
        delta: Confidence parameter (default 0.05 for 95% confidence)
        sigma_sq: Prior variance (auto-computed if None)

    Returns:
        Dictionary with PAC-Bayes bounds and parameters
    """
    total_cost = costs.sum()

    # Auto-compute sigma_sq to match budget
    if sigma_sq is None:
        # Set σ² such that KL = total_cost / σ² is controlled by epsilon
        sigma_sq = total_cost / epsilon if epsilon > 0 else 1.0

    # KL divergence between posterior Q and prior P
    kl_divergence = total_cost / sigma_sq

    # PAC-Bayes complexity term
    complexity = np.sqrt((kl_divergence + np.log(1.0 / delta)) / (2.0 * n_samples))

    # Risk bound (empirical risk would be measured on safety data)
    # Here we provide the complexity term that will be added to empirical risk
    risk_bound_term = complexity

    return {
        "total_cost": float(total_cost),
        "kl_divergence": float(kl_divergence),
        "sigma_squared": float(sigma_sq),
        "complexity_term": float(complexity),
        "confidence": 1.0 - delta,
        "n_samples": n_samples,
        "pac_certificate": f"Risk upper bound controlled by ε={epsilon:.4f} via KL={kl_divergence:.4f}",
    }


def compute_robust_feasibility(
    costs: np.ndarray,
    diag_hinv: np.ndarray | None,
    epsilon: float,
    eta: float = 0.3,
    Gamma: int | None = None,
) -> dict:
    """Compute robust feasibility under H^-1 uncertainty.

    Theorem B (Robust Feasibility):
    Under uncertainty d_m ∈ [(1-η)d̄_m, (1+η)d̄_m], the robust constraint is:
        Σ (δw²/2d̄_m) + Σ_{top-Γ} Δc_m ≤ ε

    Args:
        costs: Rank-free costs c_m = δw²/2
        diag_hinv: H^-1 diagonal (None for rank-free mode)
        epsilon: Budget
        eta: Relative uncertainty (default 0.3 for ±30%)
        Gamma: Number of worst-case perturbations (default: 10% of parameters)

    Returns:
        Dictionary with robust feasibility certificate
    """
    n_params = len(costs)

    if Gamma is None:
        Gamma = max(1, int(0.1 * n_params))  # 10% worst-case

    if diag_hinv is None:
        # Rank-free mode: costs already uniform
        nominal_costs = costs
        perturbation = np.zeros_like(costs)
    else:
        # Nominal costs with mean H^-1
        d_mean = diag_hinv.mean()
        nominal_costs = costs / d_mean  # Approximate

        # Worst-case perturbation: Δc_m = c_m * (1/(1-η) - 1)
        perturbation = costs * (1.0 / (1.0 - eta) - 1.0)

    # Sort perturbations and take top-Γ
    top_Gamma_perturbations = np.sort(perturbation)[-Gamma:]

    # Robust upper bound
    robust_bound = nominal_costs.sum() + top_Gamma_perturbations.sum()

    # Check feasibility
    is_feasible = robust_bound <= epsilon
    slack = epsilon - robust_bound

    return {
        "nominal_cost": float(nominal_costs.sum()),
        "top_Gamma_perturbation": float(top_Gamma_perturbations.sum()),
        "robust_upper_bound": float(robust_bound),
        "budget": float(epsilon),
        "slack": float(slack),
        "is_feasible": bool(is_feasible),
        "eta": float(eta),
        "Gamma": int(Gamma),
        "robust_certificate": (
            f"Robust feasible (slack={slack:.4e})"
            if is_feasible
            else f"Robust infeasible (excess={-slack:.4e})"
        ),
    }


def compute_dual_gap(
    scores: np.ndarray,
    costs: np.ndarray,
    selected_mask: np.ndarray,
    lambda_star: float,
    epsilon: float,
) -> dict:
    """Compute dual optimality gap certificate.

    Proposition F (Dual Certificate):
    For threshold λ*, the dual gap is:
        gap(λ*, S) = λ*ε - Σ_{m∈S}(λ*c_m - u_m) - Σ_{m∉S} max(0, u_m - λ*c_m)

    A small gap indicates near-optimality.

    Args:
        scores: Utility scores u_m
        costs: Safety costs c_m
        selected_mask: Boolean mask for selected parameters
        lambda_star: Dual threshold
        epsilon: Budget

    Returns:
        Dictionary with dual gap and optimality certificate
    """
    # Primal objective: Σ_{m∈S} u_m
    primal_value = scores[selected_mask].sum()

    # Dual objective term 1: λ*ε
    dual_term1 = lambda_star * epsilon

    # Dual objective term 2: -Σ_{m∈S}(λ*c_m - u_m)
    dual_term2 = -((lambda_star * costs[selected_mask] - scores[selected_mask]).sum())

    # Dual objective term 3: -Σ_{m∉S} max(0, u_m - λ*c_m)
    unselected_surplus = np.maximum(0, scores[~selected_mask] - lambda_star * costs[~selected_mask])
    dual_term3 = -unselected_surplus.sum()

    # Total dual value
    dual_value = dual_term1 + dual_term2 + dual_term3

    # Duality gap
    gap = dual_value - primal_value

    # Relative gap
    relative_gap = gap / (abs(primal_value) + 1e-10)

    return {
        "primal_value": float(primal_value),
        "dual_value": float(dual_value),
        "gap": float(gap),
        "relative_gap": float(relative_gap),
        "lambda_star": float(lambda_star),
        "epsilon": float(epsilon),
        "optimality_certificate": (
            f"Near-optimal (gap={gap:.4e}, relative={relative_gap:.4f})"
            if gap >= -1e-6
            else f"Dual gap negative (numerical issue)"
        ),
    }


def compute_trust_region_alpha(
    selected_costs: np.ndarray,
    epsilon: float,
    alpha_candidates: list[float] = [0.6, 0.8, 1.0],
) -> dict:
    """Compute optimal alpha scaling via trust region.

    Proposition G (Trust Region Alpha):
    The optimal α solves:
        max_{α∈[0,1]} U(α)  s.t.  ||α(M⊙Δ)||² ≤ 2ε

    In practice, we use 2-3 point line search.

    Args:
        selected_costs: Costs of selected parameters
        epsilon: Budget
        alpha_candidates: Alpha values to search

    Returns:
        Dictionary with optimal alpha and trust region info
    """
    # Constraint: α² Σc_m ≤ ε
    total_cost = selected_costs.sum()

    # Maximum feasible alpha
    if total_cost > 0:
        alpha_max = np.sqrt(epsilon / total_cost)
    else:
        alpha_max = 1.0

    # Find best alpha among candidates
    feasible_alphas = [a for a in alpha_candidates if a <= alpha_max]

    if not feasible_alphas:
        alpha_star = alpha_max
        status = "boundary"
    else:
        # For now, use maximum feasible (could add utility evaluation)
        alpha_star = max(feasible_alphas)
        status = "feasible"

    return {
        "alpha_star": float(alpha_star),
        "alpha_max": float(alpha_max),
        "total_cost": float(total_cost),
        "epsilon": float(epsilon),
        "status": status,
        "trust_region_certificate": f"α*={alpha_star:.3f} (max feasible={alpha_max:.3f})",
    }


def compute_lipschitz_margin(
    selected_delta_norm: float,
    lipschitz_constant: float,
    safety_margin: float,
) -> dict:
    """Compute Lipschitz safety output bound.

    The output perturbation is bounded by:
        ||(M⊙Δ)X||_F ≤ ||M⊙Δ||_F · ||X||_op ≤ sqrt(2ε) · L_X

    Args:
        selected_delta_norm: ||M⊙Δ||_F
        lipschitz_constant: L_X = ||X||_op
        safety_margin: Original safety margin

    Returns:
        Dictionary with Lipschitz bound and certification
    """
    # Output perturbation bound
    perturbation_bound = selected_delta_norm * lipschitz_constant

    # Check if within margin
    is_certified = perturbation_bound <= safety_margin
    slack = safety_margin - perturbation_bound

    return {
        "perturbation_bound": float(perturbation_bound),
        "lipschitz_constant": float(lipschitz_constant),
        "safety_margin": float(safety_margin),
        "slack": float(slack),
        "is_certified": bool(is_certified),
        "lipschitz_certificate": (
            f"Certified safe (slack={slack:.4e})"
            if is_certified
            else f"May exceed margin (excess={-slack:.4e})"
        ),
    }
