"""Tests for streaming selection."""

import numpy as np
import pytest
import torch

from deltaone.core import Bitset, Block, iter_blocks
from deltaone.select import StreamingSelector, compute_budget_rankfree, compute_cost_rankfree


def test_streaming_selector_simple():
    """Test streaming selector with simple case."""
    # Create mock blocks
    delta1 = torch.randn(10, 10)
    delta2 = torch.randn(10, 10)

    blocks_data = []

    # Block 1
    block1 = Block(
        key="layer.0",
        shape=(10, 10),
        rows=slice(0, 10),
        cols=slice(0, 10),
        delta=delta1,
        diag=None,
        grad=None,
        global_offset=0,
    )
    scores1 = np.random.rand(100)
    costs1 = compute_cost_rankfree(delta1.numpy().flatten())
    blocks_data.append((block1, scores1, costs1))

    # Block 2
    block2 = Block(
        key="layer.1",
        shape=(10, 10),
        rows=slice(0, 10),
        cols=slice(0, 10),
        delta=delta2,
        diag=None,
        grad=None,
        global_offset=100,
    )
    scores2 = np.random.rand(100)
    costs2 = compute_cost_rankfree(delta2.numpy().flatten())
    blocks_data.append((block2, scores2, costs2))

    # Compute budget (50% of total cost)
    all_costs = np.concatenate([costs1, costs2])
    budget = compute_budget_rankfree(all_costs, scale=0.5)

    # Create bitset
    bitset = Bitset(total_params=200)

    # Run selection
    selector = StreamingSelector(budget)
    stats = selector.select_from_blocks(blocks_data, bitset)

    # Check that selection ratio is approximately 50%
    assert 0.4 < stats["selection_ratio"] < 0.6
    assert stats["cumulative_cost"] <= budget


def test_streaming_vs_global_sort():
    """Test that streaming selector produces same result as global sort."""
    np.random.seed(42)

    # Create 3 blocks with known scores and costs
    num_blocks = 3
    block_size = 100
    total_size = num_blocks * block_size

    all_scores = np.random.rand(total_size)
    all_costs = np.random.rand(total_size) * 0.1

    # Global sort (ground truth)
    sorted_indices = np.argsort(-all_scores)  # Descending
    sorted_costs = all_costs[sorted_indices]
    cumsum_costs = np.cumsum(sorted_costs)

    budget = all_costs.sum() * 0.3  # 30% budget
    num_selected_global = int((cumsum_costs <= budget).sum())

    # Streaming selection
    blocks_data = []
    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = (i + 1) * block_size

        block = Block(
            key=f"layer.{i}",
            shape=(10, 10),
            rows=slice(0, 10),
            cols=slice(0, 10),
            delta=torch.zeros(10, 10),  # Dummy
            diag=None,
            grad=None,
            global_offset=start_idx,
        )

        scores_block = all_scores[start_idx:end_idx]
        costs_block = all_costs[start_idx:end_idx]

        blocks_data.append((block, scores_block, costs_block))

    bitset = Bitset(total_params=total_size)
    selector = StreamingSelector(budget)
    stats = selector.select_from_blocks(blocks_data, bitset)

    # Check that number of selected is close (should be exact for K-way merge)
    # Allow small tolerance due to floating point
    assert abs(stats["num_selected"] - num_selected_global) <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
