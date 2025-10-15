"""Streaming selection using K-way merge heap for exact global selection.

This module implements memory-efficient global parameter selection using
a K-way merge algorithm with a min-heap. Each block is sorted locally,
then the heap merges them to produce exact global top-k selection.
"""

import heapq
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from ..core import Block, Bitset


@dataclass(order=True)
class HeapEntry:
    """Entry in the K-way merge heap.

    Sorted by score (negated for max-heap behavior with heapq).
    """

    score: float  # Negated for max-heap
    block_id: int
    local_idx: int
    global_idx: int
    cost: float


class StreamingSelector:
    """K-way merge heap selector for exact global top-k selection.

    Memory usage: O(K) where K is the number of blocks.
    Time complexity: O(N log K) where N is total parameters.
    """

    def __init__(self, budget: float):
        """Initialize selector.

        Args:
            budget: Total safety budget epsilon
        """
        self.budget = budget
        self.cumulative_cost = 0.0
        self.num_selected = 0

    def select_from_blocks(
        self,
        blocks: list[tuple[Block, np.ndarray, np.ndarray]],
        bitset: Bitset,
    ) -> dict:
        """Select parameters from blocks using K-way merge.

        Args:
            blocks: List of (Block, scores, costs) tuples
            bitset: Bitset to mark selected parameters

        Returns:
            Statistics dictionary
        """
        # Sort each block locally by score (descending)
        sorted_blocks = []
        for block_id, (block, scores, costs) in enumerate(blocks):
            # Get sorting indices (descending by score)
            sorted_indices = np.argsort(-scores)

            sorted_blocks.append(
                {
                    "block": block,
                    "scores": scores[sorted_indices],
                    "costs": costs[sorted_indices],
                    "sorted_indices": sorted_indices,
                    "local_pos": 0,  # Current position in sorted block
                    "block_id": block_id,
                }
            )

        # Initialize heap with first element from each block
        heap: list[HeapEntry] = []
        for block_data in sorted_blocks:
            if len(block_data["scores"]) > 0:
                local_idx = block_data["sorted_indices"][0]
                global_idx = block_data["block"].global_offset + local_idx

                entry = HeapEntry(
                    score=-block_data["scores"][0],  # Negate for max-heap
                    block_id=block_data["block_id"],
                    local_idx=0,  # Position in sorted array
                    global_idx=global_idx,
                    cost=block_data["costs"][0],
                )
                heapq.heappush(heap, entry)

        # K-way merge: repeatedly pop max score, add to selection
        while heap and self.cumulative_cost < self.budget:
            # Pop element with highest score (most negative in min-heap)
            entry = heapq.heappop(heap)

            # Check if adding this parameter exceeds budget
            if self.cumulative_cost + entry.cost > self.budget:
                break

            # Select this parameter
            bitset.set(entry.global_idx, True)
            self.cumulative_cost += entry.cost
            self.num_selected += 1

            # Push next element from same block
            block_data = sorted_blocks[entry.block_id]
            next_pos = entry.local_idx + 1

            if next_pos < len(block_data["scores"]):
                local_idx = block_data["sorted_indices"][next_pos]
                global_idx = block_data["block"].global_offset + local_idx

                next_entry = HeapEntry(
                    score=-block_data["scores"][next_pos],
                    block_id=entry.block_id,
                    local_idx=next_pos,
                    global_idx=global_idx,
                    cost=block_data["costs"][next_pos],
                )
                heapq.heappush(heap, next_entry)

        return {
            "num_selected": self.num_selected,
            "cumulative_cost": self.cumulative_cost,
            "budget": self.budget,
            "selection_ratio": self.num_selected / bitset.total_params,
        }
