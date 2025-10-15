"""Tests for bitset operations."""

import tempfile
from pathlib import Path

import pytest

from deltaone.core import Bitset


def test_bitset_basic():
    """Test basic bitset operations."""
    bitset = Bitset(total_params=1000)

    # Initially all False
    assert bitset.count() == 0

    # Set some bits
    bitset.set(10, True)
    bitset.set(20, True)
    bitset.set(30, True)

    assert bitset.count() == 3
    assert bitset.get(10) is True
    assert bitset.get(20) is True
    assert bitset.get(30) is True
    assert bitset.get(15) is False


def test_bitset_batch():
    """Test batch setting."""
    import numpy as np

    bitset = Bitset(total_params=1000)

    indices = np.array([10, 20, 30, 40, 50])
    bitset.set_batch(indices)

    assert bitset.count() == 5
    for idx in indices:
        assert bitset.get(idx) is True


def test_bitset_memmap():
    """Test memory-mapped bitset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test.mmap"

        # Create and set some bits
        bitset1 = Bitset(total_params=1000, filepath=filepath)
        bitset1.set(10, True)
        bitset1.set(20, True)
        bitset1.dump()

        # Load and verify
        bitset2 = Bitset.load(filepath, total_params=1000)
        assert bitset2.count() == 2
        assert bitset2.get(10) is True
        assert bitset2.get(20) is True


def test_bitset_edge_cases():
    """Test edge cases."""
    bitset = Bitset(total_params=100)

    # First and last indices
    bitset.set(0, True)
    bitset.set(99, True)

    assert bitset.get(0) is True
    assert bitset.get(99) is True

    # Out of bounds
    with pytest.raises(IndexError):
        bitset.set(100, True)

    with pytest.raises(IndexError):
        bitset.get(100)


def test_bitset_clear():
    """Test clearing bits."""
    bitset = Bitset(total_params=100)

    bitset.set(10, True)
    assert bitset.get(10) is True

    bitset.set(10, False)
    assert bitset.get(10) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
