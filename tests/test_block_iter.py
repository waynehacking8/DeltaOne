"""Tests for block iteration."""

import numpy as np
import pytest
import torch

from deltaone.core import Block, iter_blocks


def test_block_iter_2d():
    """Test block iteration on 2D tensor."""
    # Create a small 2D tensor
    tensor = torch.randn(128, 256)

    blocks = list(
        iter_blocks(
            key="test.weight",
            delta_tensor=tensor,
            block_rows=64,
            block_cols=128,
        )
    )

    # Should have 2x2 = 4 blocks
    assert len(blocks) == 4

    # Check shapes
    assert blocks[0].delta.shape == (64, 128)
    assert blocks[1].delta.shape == (64, 128)
    assert blocks[2].delta.shape == (64, 128)
    assert blocks[3].delta.shape == (64, 128)

    # Check that blocks are views (not copies)
    # Modifying block should modify original
    original_value = tensor[0, 0].item()
    blocks[0].delta[0, 0] = 999.0
    assert tensor[0, 0].item() == 999.0
    tensor[0, 0] = original_value  # Restore


def test_block_iter_1d():
    """Test block iteration on 1D tensor."""
    tensor = torch.randn(1000)

    blocks = list(
        iter_blocks(
            key="test.bias",
            delta_tensor=tensor,
            block_rows=256,
            block_cols=1,
        )
    )

    # Should have ceil(1000/256) = 4 blocks
    assert len(blocks) == 4

    # Check total elements
    total_numel = sum(b.delta.numel() for b in blocks)
    assert total_numel == 1000


def test_block_iter_with_diag_grad():
    """Test block iteration with diagonal and gradient."""
    tensor = torch.randn(64, 128)
    diag = np.random.rand(64 * 128).astype(np.float32)
    grad = np.random.rand(64 * 128).astype(np.float32)

    blocks = list(
        iter_blocks(
            key="test.weight",
            delta_tensor=tensor,
            diag_flat=diag,
            grad_flat=grad,
            block_rows=32,
            block_cols=64,
        )
    )

    # Should have 2x2 = 4 blocks
    assert len(blocks) == 4

    # Check that diag and grad are correctly sliced
    for block in blocks:
        assert block.diag is not None
        assert block.grad is not None
        assert block.diag.shape == (block.delta.numel(),)
        assert block.grad.shape == (block.delta.numel(),)


def test_block_iter_global_offset():
    """Test that global offsets are correctly computed."""
    tensor = torch.randn(100, 200)

    blocks = list(
        iter_blocks(
            key="test.weight",
            delta_tensor=tensor,
            block_rows=50,
            block_cols=100,
            global_offset_start=1000,
        )
    )

    # Check that offsets are sequential
    expected_offset = 1000
    for block in blocks:
        assert block.global_offset == expected_offset
        expected_offset += block.delta.numel()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
