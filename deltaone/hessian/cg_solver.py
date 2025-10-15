"""Conjugate Gradient solver for (2G)u = e_j system.

Used for OBS compensation: solves for column j of H^-1 on-demand.
"""

from collections import OrderedDict
from typing import Callable

import numpy as np
import torch


class CGSolver:
    """Conjugate Gradient solver with LRU cache.

    Solves (2G)u_j = e_j where:
    - G is the Gram matrix (X X^T approximation of Hessian)
    - e_j is the j-th canonical basis vector
    - u_j is the j-th column of (2G)^{-1}
    """

    def __init__(
        self,
        gram_matrix: torch.Tensor | Callable,
        max_iter: int = 100,
        tol: float = 1e-3,
        cache_size: int = 100,
        preconditioner: str = "jacobi",
    ):
        """Initialize CG solver.

        Args:
            gram_matrix: Gram matrix G or callable matvec(v) -> G @ v
            max_iter: Maximum CG iterations
            tol: Residual tolerance
            cache_size: LRU cache size for solved columns
            preconditioner: Preconditioner type ('jacobi' or 'none')
        """
        self.gram_matrix = gram_matrix
        self.max_iter = max_iter
        self.tol = tol
        self.cache_size = cache_size
        self.preconditioner_type = preconditioner

        # LRU cache for solved columns
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()

        # Statistics
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_solves": 0,
            "avg_iterations": 0.0,
            # Residual statistics (for OBS compensation verification)
            "residual_max": 0.0,
            "residual_mean": 0.0,
            "residual_history": [],  # Keep last 100 residuals
        }

        # Precompute preconditioner
        self._preconditioner = None
        if preconditioner == "jacobi":
            self._compute_jacobi_preconditioner()

    def _compute_jacobi_preconditioner(self):
        """Compute Jacobi (diagonal) preconditioner: M = diag(2G)."""
        if callable(self.gram_matrix):
            # Cannot precompute for matvec-only interface
            self._preconditioner = None
        else:
            # Extract diagonal
            diag = torch.diagonal(2.0 * self.gram_matrix)
            # Invert (with regularization)
            self._preconditioner = 1.0 / (diag + 1e-6)

    def solve(self, col_idx: int) -> np.ndarray:
        """Solve (2G)u_j = e_j for column j.

        Args:
            col_idx: Column index j

        Returns:
            Solution vector u_j
        """
        # Check cache
        if col_idx in self.cache:
            self.stats["cache_hits"] += 1
            # Move to end (most recently used)
            self.cache.move_to_end(col_idx)
            return self.cache[col_idx]

        # Cache miss: solve
        self.stats["cache_misses"] += 1
        self.stats["total_solves"] += 1

        # Create right-hand side: e_j
        if callable(self.gram_matrix):
            # Need to infer size from first matvec call
            raise NotImplementedError("Matvec-only interface not yet supported")
        else:
            n = self.gram_matrix.size(0)
            b = torch.zeros(n, dtype=self.gram_matrix.dtype, device=self.gram_matrix.device)
            b[col_idx] = 1.0

        # Solve using CG
        solution, num_iter, residual = self._cg_solve(b)

        # Update iteration stats
        alpha = 0.1  # Exponential moving average
        self.stats["avg_iterations"] = (
            (1 - alpha) * self.stats["avg_iterations"] + alpha * num_iter
        )

        # Update residual stats
        self.stats["residual_max"] = max(self.stats["residual_max"], residual)
        if self.stats["total_solves"] == 1:
            self.stats["residual_mean"] = residual
        else:
            # Exponential moving average for mean
            self.stats["residual_mean"] = (
                (1 - alpha) * self.stats["residual_mean"] + alpha * residual
            )

        # Track residual history (last 100)
        self.stats["residual_history"].append(residual)
        if len(self.stats["residual_history"]) > 100:
            self.stats["residual_history"].pop(0)

        # Convert to numpy and cache
        solution_np = solution.cpu().numpy()
        self.cache[col_idx] = solution_np

        # Evict oldest if cache full
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)  # Remove oldest

        return solution_np

    def _cg_solve(self, b: torch.Tensor) -> tuple[torch.Tensor, int, float]:
        """Run CG iterations to solve (2G)x = b.

        Args:
            b: Right-hand side vector

        Returns:
            Tuple of (solution, num_iterations, final_residual_norm)
        """
        # Initialize
        x = torch.zeros_like(b)
        r = b.clone()  # r = b - A @ x (x = 0 initially)

        # Apply preconditioner: z = M^{-1} @ r
        if self._preconditioner is not None:
            z = self._preconditioner * r
        else:
            z = r

        p = z.clone()
        rz_old = torch.dot(r, z)

        # CG iterations
        for iteration in range(self.max_iter):
            # Compute A @ p where A = 2G
            if callable(self.gram_matrix):
                Ap = 2.0 * self.gram_matrix(p)
            else:
                Ap = 2.0 * (self.gram_matrix @ p)

            # Step size
            pAp = torch.dot(p, Ap)
            alpha = rz_old / (pAp + 1e-10)

            # Update solution and residual
            x = x + alpha * p
            r = r - alpha * Ap

            # Check convergence
            residual_norm = torch.norm(r).item()
            if residual_norm < self.tol:
                return x, iteration + 1, residual_norm

            # Apply preconditioner
            if self._preconditioner is not None:
                z = self._preconditioner * r
            else:
                z = r

            # Update direction
            rz_new = torch.dot(r, z)
            beta = rz_new / (rz_old + 1e-10)
            p = z + beta * p

            rz_old = rz_new

        # Max iterations reached - compute final residual
        final_residual = torch.norm(r).item()
        return x, self.max_iter, final_residual

    def solve_batch(self, col_indices: list[int]) -> dict[int, np.ndarray]:
        """Solve for multiple columns.

        Args:
            col_indices: List of column indices

        Returns:
            Dictionary mapping col_idx -> solution
        """
        solutions = {}
        for col_idx in col_indices:
            solutions[col_idx] = self.solve(col_idx)
        return solutions

    def clear_cache(self):
        """Clear solution cache."""
        self.cache.clear()

    def get_stats(self) -> dict:
        """Get solver statistics.

        Returns:
            Statistics dictionary
        """
        total_queries = self.stats["cache_hits"] + self.stats["cache_misses"]
        cache_hit_rate = (
            self.stats["cache_hits"] / total_queries if total_queries > 0 else 0.0
        )

        return {
            **self.stats,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.cache),
        }


def test_cg_solver():
    """Test CG solver with random SPD matrix."""
    import torch

    # Generate random SPD matrix
    n = 100
    A_full = torch.randn(n, n)
    A = A_full @ A_full.T + torch.eye(n) * 0.1  # Make SPD

    # Use A/2 as Gram matrix (so 2G = A)
    G = A / 2.0

    # Create solver
    solver = CGSolver(gram_matrix=G, max_iter=100, tol=1e-6)

    # Solve for column 10
    col_idx = 10
    u = solver.solve(col_idx)

    # Verify: (2G)u = e
    e = torch.zeros(n)
    e[col_idx] = 1.0

    result = 2.0 * (G @ torch.from_numpy(u).float())
    error = torch.norm(result - e).item()

    print(f"CG Test:")
    print(f"  Residual error: {error:.2e}")
    print(f"  Stats: {solver.get_stats()}")

    assert error < 1e-3, f"CG failed: error = {error}"
    print("  ✓ CG solver test passed!")


if __name__ == "__main__":
    test_cg_solver()
