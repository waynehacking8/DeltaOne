"""OBS (Optimal Brain Surgeon) compensation for unselected parameters.

When we remove (reverse) a selected parameter δw_m, OBS compensation
updates unselected parameters to minimize impact on loss:

    Δw_n = (δw_m / d_m) * [H^-1]_{nm}  for unselected n

With CG-on-Demand, we solve (2G)u_j = e_j to get column j of H^-1 on demand.
"""

from pathlib import Path

import numpy as np
import torch

from ..core import Bitset
from ..hessian import CGSolver


class OBSCompensator:
    """OBS compensation using CG-on-Demand approach.

    Instead of storing full H^-1 matrix, we solve for needed columns
    using Conjugate Gradient when required.
    """

    def __init__(
        self,
        cg_solver: CGSolver,
        diag_hinv: np.ndarray | None = None,
    ):
        """Initialize OBS compensator.

        Args:
            cg_solver: Conjugate Gradient solver for (2G)u = e
            diag_hinv: H^-1 diagonal (optional, uses 1.0 if None)
        """
        self.cg_solver = cg_solver
        self.diag_hinv = diag_hinv

        self.stats = {
            "total_compensations": 0,
            "total_selected_params": 0,
        }

    def compute_compensation(
        self,
        delta_block: np.ndarray,
        selected_indices: np.ndarray,
        global_offset: int = 0,
    ) -> np.ndarray:
        """Compute OBS compensation for a block.

        Args:
            delta_block: Delta weights for this block (flattened)
            selected_indices: Indices of selected parameters (local to block)
            global_offset: Global offset for this block

        Returns:
            Compensation vector (same shape as delta_block)
        """
        compensation = np.zeros_like(delta_block)

        # Group selected indices by column for efficient CG solving
        # For linear layers: column = param_idx % in_features
        # For simplicity, solve for each selected parameter's column

        selected_global = selected_indices + global_offset
        columns_needed = set(selected_global)

        # Solve for needed columns (with caching)
        solutions = self.cg_solver.solve_batch(list(columns_needed))

        # Compute compensation for each selected parameter
        for local_idx in selected_indices:
            global_idx = local_idx + global_offset
            delta_m = delta_block[local_idx]

            # Get H^-1 diagonal entry
            if self.diag_hinv is not None:
                d_m = self.diag_hinv[global_idx]
            else:
                d_m = 1.0

            # Get H^-1 column (solved via CG)
            hinv_col = solutions[global_idx]

            # Compute compensation: (δw_m / d_m) * [H^-1]_:m
            # But only apply to unselected parameters
            comp_contribution = (delta_m / d_m) * hinv_col[global_offset : global_offset + len(delta_block)]

            # Mask out selected parameters (no self-compensation)
            mask = np.ones_like(compensation, dtype=bool)
            mask[selected_indices] = False

            compensation[mask] += comp_contribution[mask]

            self.stats["total_selected_params"] += 1

        self.stats["total_compensations"] += 1

        return compensation

    def apply_compensation_to_layer(
        self,
        w_layer: torch.Tensor,
        delta_layer: torch.Tensor,
        bitset: Bitset,
        alpha: float = 1.0,
    ) -> int:
        """Apply OBS compensation to entire layer.

        Args:
            w_layer: Layer weights (modified in-place)
            delta_layer: Delta weights for this layer
            bitset: Selection mask
            alpha: Scaling factor for compensation

        Returns:
            Number of compensated parameters
        """
        # Flatten
        w_flat = w_layer.flatten()
        delta_flat = delta_layer.flatten()

        # Get selected indices
        selected_indices = np.array([i for i in range(len(w_flat)) if bitset.get(i)])

        if len(selected_indices) == 0:
            return 0

        # Compute compensation
        compensation = self.compute_compensation(
            delta_block=delta_flat.cpu().numpy(),
            selected_indices=selected_indices,
            global_offset=0,
        )

        # Apply compensation to unselected parameters
        unselected_mask = np.ones(len(w_flat), dtype=bool)
        unselected_mask[selected_indices] = False
        unselected_indices = np.where(unselected_mask)[0]

        for idx in unselected_indices:
            w_flat[idx] += alpha * compensation[idx]

        return len(unselected_indices)

    def get_stats(self) -> dict:
        """Get compensation statistics.

        Returns:
            Statistics dictionary
        """
        return {
            **self.stats,
            "cg_stats": self.cg_solver.get_stats(),
        }


def load_obs_compensator(
    gram_path: Path | str,
    diag_path: Path | str | None = None,
    cg_max_iter: int = 100,
    cg_tol: float = 1e-3,
    cache_size: int = 100,
) -> OBSCompensator:
    """Load OBS compensator from cached Gram matrix and diagonal.

    Args:
        gram_path: Path to Gram matrix (.npz file)
        diag_path: Path to H^-1 diagonal (.npy file)
        cg_max_iter: Maximum CG iterations
        cg_tol: CG residual tolerance
        cache_size: CG solution cache size

    Returns:
        OBSCompensator instance
    """
    # Load Gram matrix
    gram_data = np.load(gram_path)
    gram = torch.from_numpy(gram_data["gram"])

    # Create CG solver
    cg_solver = CGSolver(
        gram_matrix=gram,
        max_iter=cg_max_iter,
        tol=cg_tol,
        cache_size=cache_size,
        preconditioner="jacobi",
    )

    # Load diagonal if provided
    diag_hinv = None
    if diag_path and Path(diag_path).exists():
        diag_hinv = np.load(diag_path)

    # Create compensator
    compensator = OBSCompensator(
        cg_solver=cg_solver,
        diag_hinv=diag_hinv,
    )

    return compensator
