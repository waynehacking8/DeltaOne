"""Bitset implementation using memory-mapped files for efficient storage."""

from pathlib import Path
from typing import Any

import numpy as np


class Bitset:
    """Memory-mapped bitset for parameter selection.

    Each bit represents whether a parameter is selected (1) or not (0).
    Uses numpy.memmap for efficient storage and out-of-core operation.
    """

    def __init__(self, total_params: int, filepath: Path | str | None = None):
        """Initialize bitset.

        Args:
            total_params: Total number of parameters
            filepath: Path to memory-mapped file (None for in-memory)
        """
        self.total_params = total_params
        self.filepath = Path(filepath) if filepath else None

        # Calculate number of bytes needed (8 bits per byte)
        self.num_bytes = (total_params + 7) // 8

        if self.filepath:
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            self.data = np.memmap(
                self.filepath,
                dtype=np.uint8,
                mode="w+",
                shape=(self.num_bytes,),
            )
        else:
            self.data = np.zeros(self.num_bytes, dtype=np.uint8)

    def set(self, idx: int, value: bool = True) -> None:
        """Set bit at index to value.

        Args:
            idx: Parameter index
            value: True to set, False to clear
        """
        if idx < 0 or idx >= self.total_params:
            raise IndexError(f"Index {idx} out of range [0, {self.total_params})")

        byte_idx = idx // 8
        bit_idx = idx % 8

        if value:
            self.data[byte_idx] |= 1 << bit_idx
        else:
            self.data[byte_idx] &= ~(1 << bit_idx)

    def get(self, idx: int) -> bool:
        """Get bit at index.

        Args:
            idx: Parameter index

        Returns:
            True if bit is set, False otherwise
        """
        if idx < 0 or idx >= self.total_params:
            raise IndexError(f"Index {idx} out of range [0, {self.total_params})")

        byte_idx = idx // 8
        bit_idx = idx % 8

        return bool((self.data[byte_idx] >> bit_idx) & 1)

    def set_batch(self, indices: np.ndarray) -> None:
        """Set multiple bits efficiently.

        Args:
            indices: Array of parameter indices to set
        """
        for idx in indices:
            self.set(int(idx), True)

    def count(self) -> int:
        """Count number of set bits.

        Returns:
            Number of selected parameters
        """
        # Use numpy's efficient bit counting
        return int(np.unpackbits(self.data)[: self.total_params].sum())

    def to_mask(self) -> np.ndarray:
        """Convert bitset to boolean mask array.

        Returns:
            Boolean numpy array of shape (total_params,)
        """
        return np.unpackbits(self.data)[: self.total_params].astype(bool)

    def dump(self) -> None:
        """Flush to disk if memory-mapped."""
        if isinstance(self.data, np.memmap):
            self.data.flush()

    @classmethod
    def load(cls, filepath: Path | str, total_params: int) -> "Bitset":
        """Load bitset from file.

        Args:
            filepath: Path to memory-mapped file
            total_params: Total number of parameters

        Returns:
            Loaded Bitset instance
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Bitset file not found: {filepath}")

        bitset = cls(total_params, filepath=None)
        bitset.filepath = filepath
        bitset.data = np.memmap(
            filepath,
            dtype=np.uint8,
            mode="r+",
            shape=(bitset.num_bytes,),
        )
        return bitset

    def __enter__(self) -> "Bitset":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.dump()
