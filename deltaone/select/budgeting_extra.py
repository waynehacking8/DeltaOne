
"""
Extra statistics & certificates for DeltaOne CLI outputs.

This module provides:
- dual_gap(lambda_star, chosen_mask, u, c)
- pac_bayes_bound(sum_cost, n_safe, delta_conf, sigma2=None)
- robust_budget_upper_bound(cost_nominal, delta_costs, Gamma)
- rho_targeting_update(s, rho_target, rho_now, kappa=0.5)
"""
from __future__ import annotations
from typing import Optional, Sequence
import math
import numpy as np

def dual_gap(lambda_star: float,
             chosen_mask: np.ndarray,
             u: np.ndarray,
             c: np.ndarray) -> float:
    """
    Compute duality gap certificate at threshold lambda_star.
    chosen_mask: bool array for selected indices S
    u: utility per index
    c: cost per index
    """
    S = chosen_mask
    gap = lambda_star * float(c[S].sum()) - float(u[S].sum())
    gap = (lambda_star * float(c[S].sum()) - float(u[S].sum()))
    # add lambda*epsilon part outside; caller passes epsilon_used to finalize
    # For report with epsilon_used:
    #   gap_report = lambda_star*epsilon_used - sum(lambda*c_i - u_i for i in S) - sum(max(0, u_i - lambda*c_i) for i not in S)
    # We compute the full expression:
    notS = ~S
    term_notS = np.maximum(0.0, u[notS] - lambda_star * c[notS]).sum()
    gap_full = lambda_star * float(c[S].sum()) - float((lambda_star * c[S] - u[S]).sum()) - float(term_notS)
    return float(gap_full)

def pac_bayes_bound(sum_cost: float,
                    n_safe: int,
                    delta_conf: float = 1e-3,
                    sigma2: Optional[float] = None,
                    emp_risk: float = 0.0) -> float:
    """
    Compute a PAC-Bayes upper bound R <= Rhat + sqrt((KL + ln(1/delta)) / (2n)).
    Under ADB, cost_i = 0.5*delta_w_i^2 and KL = (1/sigma2) * sum_i cost_i.
    If sigma2 is None, choose sigma2 = sum_cost / max(n_safe,1) as a conservative scaling.
    """
    if n_safe <= 0:
        return float('nan')
    if sigma2 is None:
        sigma2 = max(sum_cost / max(n_safe,1), 1e-12)
    KL = sum_cost / sigma2
    bound = emp_risk + math.sqrt((KL + math.log(1.0/delta_conf)) / (2.0 * n_safe))
    return float(bound)

def robust_budget_upper_bound(cost_nominal: np.ndarray,
                              delta_costs: np.ndarray,
                              Gamma: int) -> float:
    """
    Bertsimas-Sim style: robust upper bound for sum(cost_nominal + worst Gamma deltas).
    cost_nominal: vector of nominal costs (e.g., delta_w^2/(2 * dbar))
    delta_costs: vector of worst-case cost increments per coordinate
    Gamma: number of coordinates that can be adversarially perturbed
    """
    Gamma = int(max(0, min(Gamma, len(delta_costs))))
    if Gamma == 0:
        return float(cost_nominal.sum())
    top = np.partition(delta_costs, -Gamma)[-Gamma:]
    return float(cost_nominal.sum() + top.sum())

def rho_targeting_update(s: float, rho_target: float, rho_now: float, kappa: float = 0.5) -> float:
    """
    Closed-loop update for scale s to hit target selection ratio.
    s_new = s * (rho_target / rho_now)^kappa
    """
    rho_now = max(rho_now, 1e-8)
    factor = (rho_target / rho_now) ** kappa
    return float(s * factor)
