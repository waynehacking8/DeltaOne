"""Scoring functions for parameter ranking."""

import numpy as np
import torch


def compute_safedelta_score(
    diag_hinv: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Compute SafeDelta original score: r_m = 2 * d_m.

    Args:
        diag_hinv: H^-1 diagonal elements

    Returns:
        Scores for ranking (higher = more important)
    """
    if isinstance(diag_hinv, torch.Tensor):
        diag_hinv = diag_hinv.cpu().numpy()

    return 2.0 * diag_hinv


def compute_delta_aware_score(
    grad: np.ndarray | torch.Tensor,
    delta: np.ndarray | torch.Tensor,
    diag_hinv: np.ndarray | torch.Tensor | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute delta-aware score: r'_m = 2 * |g_m| / (|δw_m| + eps).

    Optionally multiply by H^-1 diagonal if provided (compatible mode).

    Args:
        grad: Gradient values
        delta: Delta weights
        diag_hinv: H^-1 diagonal (None for Rank-Free mode)
        eps: Small epsilon to avoid division by zero

    Returns:
        Scores for ranking (higher = more important)
    """
    # Convert to numpy (handle bfloat16)
    if isinstance(grad, torch.Tensor):
        grad = grad.cpu().float() if grad.dtype == torch.bfloat16 else grad.cpu()
        grad = grad.numpy()
    if isinstance(delta, torch.Tensor):
        delta = delta.cpu().float() if delta.dtype == torch.bfloat16 else delta.cpu()
        delta = delta.numpy()
    if isinstance(diag_hinv, torch.Tensor):
        diag_hinv = diag_hinv.cpu().float() if diag_hinv.dtype == torch.bfloat16 else diag_hinv.cpu()
        diag_hinv = diag_hinv.numpy()

    # Compute base score: |g| / |δw|
    grad_abs = np.abs(grad)
    delta_abs = np.abs(delta)
    score = 2.0 * grad_abs / (delta_abs + eps)

    # Multiply by H^-1 diagonal if provided
    if diag_hinv is not None:
        score = score * diag_hinv

    return score


def compute_cost_rankfree(
    delta: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Compute Rank-Free safety cost: c_m = δw_m² / 2.

    Args:
        delta: Delta weights

    Returns:
        Safety costs
    """
    if isinstance(delta, torch.Tensor):
        delta = delta.cpu().float() if delta.dtype == torch.bfloat16 else delta.cpu()
        delta = delta.numpy()

    return 0.5 * (delta**2)


def compute_cost_safedelta(
    delta: np.ndarray | torch.Tensor,
    diag_hinv: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Compute SafeDelta safety cost: c_m = δw_m² / (2 * H^-1_mm).

    Args:
        delta: Delta weights
        diag_hinv: H^-1 diagonal elements

    Returns:
        Safety costs
    """
    if isinstance(delta, torch.Tensor):
        delta = delta.cpu().numpy()
    if isinstance(diag_hinv, torch.Tensor):
        diag_hinv = diag_hinv.cpu().numpy()

    return (delta**2) / (2.0 * diag_hinv)
