"""Block-wise iteration for memory-efficient processing.

This module provides utilities to iterate over large tensors in blocks without
creating full copies in memory. Uses views for zero-copy slicing.
"""

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch


@dataclass
class Block:
    """Represents a block of a weight tensor.

    Attributes:
        key: Parameter name (e.g., 'model.layers.0.q_proj.weight')
        shape: Original tensor shape (out_dim, in_dim) for linear layers
        rows: Row slice for this block
        cols: Column slice for this block
        delta: Delta weights for this block (CPU tensor, view when possible)
        diag: H^-1 diagonal for this block (None if Rank-Free mode)
        grad: Gradient for this block (None if not available)
        global_offset: Starting index in flattened global bitset
    """

    key: str
    shape: tuple[int, ...]
    rows: slice
    cols: slice
    delta: torch.Tensor  # View, not copy
    diag: np.ndarray | None
    grad: np.ndarray | None
    global_offset: int


def iter_blocks(
    key: str,
    delta_tensor: torch.Tensor,
    diag_flat: np.ndarray | None = None,
    grad_flat: np.ndarray | None = None,
    block_rows: int = 2048,
    block_cols: int = 4096,
    global_offset_start: int = 0,
) -> Iterator[Block]:
    """Iterate over a tensor in blocks without creating copies.

    For 2D tensors (e.g., Linear weights), blocks are (block_rows, block_cols).
    For other shapes, tensor is flattened and processed in chunks.

    Args:
        key: Parameter name
        delta_tensor: Delta weights tensor (CPU)
        diag_flat: Flattened H^-1 diagonal (None for Rank-Free mode)
        grad_flat: Flattened gradients (None if not available)
        block_rows: Number of rows per block
        block_cols: Number of columns per block
        global_offset_start: Starting offset in global bitset

    Yields:
        Block objects with views into the original tensor
    """
    orig_shape = delta_tensor.shape
    global_offset = global_offset_start

    # Handle 2D tensors (Linear layers: out_features × in_features)
    if delta_tensor.ndim == 2:
        out_dim, in_dim = delta_tensor.shape

        for row_start in range(0, out_dim, block_rows):
            row_end = min(row_start + block_rows, out_dim)
            row_slice = slice(row_start, row_end)

            for col_start in range(0, in_dim, block_cols):
                col_end = min(col_start + block_cols, in_dim)
                col_slice = slice(col_start, col_end)

                # Create view (zero-copy)
                delta_block = delta_tensor[row_slice, col_slice]
                block_numel = delta_block.numel()

                # Extract corresponding diag/grad if available
                diag_block = None
                if diag_flat is not None:
                    diag_block = diag_flat[global_offset : global_offset + block_numel]

                grad_block = None
                if grad_flat is not None:
                    grad_block = grad_flat[global_offset : global_offset + block_numel]

                yield Block(
                    key=key,
                    shape=orig_shape,
                    rows=row_slice,
                    cols=col_slice,
                    delta=delta_block,
                    diag=diag_block,
                    grad=grad_block,
                    global_offset=global_offset,
                )

                global_offset += block_numel

    # Handle 1D or other shapes: flatten and chunk
    else:
        delta_flat = delta_tensor.flatten()
        total_numel = delta_flat.numel()
        block_size = block_rows * block_cols

        for start_idx in range(0, total_numel, block_size):
            end_idx = min(start_idx + block_size, total_numel)
            block_numel = end_idx - start_idx

            # Create view
            delta_block = delta_flat[start_idx:end_idx]

            # Extract corresponding diag/grad if available
            diag_block = None
            if diag_flat is not None:
                diag_block = diag_flat[global_offset : global_offset + block_numel]

            grad_block = None
            if grad_flat is not None:
                grad_block = grad_flat[global_offset : global_offset + block_numel]

            yield Block(
                key=key,
                shape=orig_shape,
                rows=slice(start_idx, end_idx),
                cols=slice(None),  # Not applicable for flattened
                delta=delta_block,
                diag=diag_block,
                grad=grad_block,
                global_offset=global_offset,
            )

            global_offset += block_numel
