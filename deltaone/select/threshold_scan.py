"""Threshold-based scanning for approximate selection (faster than K-way merge)."""

from typing import Callable

import numpy as np

from ..core import Bitset, Block


class ThresholdScanner:
    """Binary search threshold scanner for approximate top-k selection.

    Trades off exactness for speed: performs multiple linear scans with
    binary search on threshold to find parameters above threshold whose
    total cost ≈ budget.

    Memory usage: O(1) - only current block in memory
    Time complexity: O(N * log(max_score - min_score) * num_iterations)
    """

    def __init__(
        self,
        budget: float,
        max_iter: int = 12,
        tol_cost: float = 0.01,
    ):
        """Initialize scanner.

        Args:
            budget: Total safety budget epsilon
            max_iter: Maximum number of binary search iterations
            tol_cost: Tolerance for cost difference (fraction of budget)
        """
        self.budget = budget
        self.max_iter = max_iter
        self.tol_cost = tol_cost
        self.threshold = 0.0
        self.num_selected = 0
        self.cumulative_cost = 0.0

    def select_from_blocks(
        self,
        block_iterator: Callable[[], list[tuple[Block, np.ndarray, np.ndarray]]],
        bitset: Bitset,
    ) -> dict:
        """Select parameters using binary search on threshold.

        Args:
            block_iterator: Callable that yields blocks (can be called multiple times)
            bitset: Bitset to mark selected parameters

        Returns:
            Statistics dictionary
        """
        # Phase 1: Determine score range (single pass)
        blocks = block_iterator()
        all_scores = np.concatenate([scores for _, scores, _ in blocks])
        score_min = float(all_scores.min())
        score_max = float(all_scores.max())

        # Phase 2: Binary search on threshold (multiple passes)
        threshold_low = score_min
        threshold_high = score_max
        best_threshold = threshold_high

        for iteration in range(self.max_iter):
            threshold_mid = (threshold_low + threshold_high) / 2.0

            # Scan all blocks to compute total cost for this threshold
            blocks = block_iterator()
            total_cost = 0.0
            num_above = 0

            for block, scores, costs in blocks:
                mask = scores >= threshold_mid
                total_cost += costs[mask].sum()
                num_above += mask.sum()

            # Check if we're close enough to budget
            cost_error = abs(total_cost - self.budget) / self.budget

            if cost_error < self.tol_cost:
                best_threshold = threshold_mid
                self.cumulative_cost = total_cost
                self.num_selected = num_above
                break

            # Adjust threshold range
            if total_cost < self.budget:
                # Too few selected, lower threshold
                threshold_high = threshold_mid
                if total_cost > self.cumulative_cost:
                    best_threshold = threshold_mid
                    self.cumulative_cost = total_cost
                    self.num_selected = num_above
            else:
                # Too many selected, raise threshold
                threshold_low = threshold_mid

        # Phase 3: Final pass to mark selected parameters
        self.threshold = best_threshold
        blocks = block_iterator()

        for block, scores, costs in blocks:
            # Near threshold, do local sorting for precision
            if iteration >= self.max_iter - 1:
                # Final iteration: use exact selection in critical region
                critical_region = (scores >= best_threshold * 0.95) & (
                    scores <= best_threshold * 1.05
                )

                if critical_region.any():
                    # Sort critical region locally
                    critical_indices = np.where(critical_region)[0]
                    critical_scores = scores[critical_indices]
                    critical_costs = costs[critical_indices]

                    sorted_idx = np.argsort(-critical_scores)
                    sorted_critical_indices = critical_indices[sorted_idx]
                    sorted_critical_costs = critical_costs[sorted_idx]

                    # Greedy add from critical region
                    for idx, cost in zip(sorted_critical_indices, sorted_critical_costs):
                        if self.cumulative_cost + cost <= self.budget:
                            global_idx = block.global_offset + idx
                            bitset.set(global_idx, True)
                            self.cumulative_cost += cost
                            self.num_selected += 1

            # Select all above threshold (non-critical region)
            mask_above = scores > best_threshold * 1.05
            for idx in np.where(mask_above)[0]:
                global_idx = block.global_offset + idx
                bitset.set(global_idx, True)

        return {
            "num_selected": self.num_selected,
            "cumulative_cost": self.cumulative_cost,
            "budget": self.budget,
            "threshold": self.threshold,
            "selection_ratio": self.num_selected / bitset.total_params,
            "iterations": iteration + 1,
        }
