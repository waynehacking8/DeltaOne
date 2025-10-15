"""Gram matrix G = X X^T collection and caching.

For OBS compensation, we need to solve (2G)u = e where G approximates
the Hessian. This module handles Gram matrix computation and caching.
"""

from pathlib import Path

import numpy as np
import torch
from rich.progress import Progress, SpinnerColumn, TextColumn


class GramMatrix:
    """Gram matrix computation and management.

    Computes G = (1/n) Σ_i (x_i x_i^T) where x_i are activation samples.
    """

    def __init__(
        self,
        layer_key: str,
        cache_dir: Path | str | None = None,
    ):
        """Initialize Gram matrix.

        Args:
            layer_key: Layer name (e.g., 'model.layers.0.q_proj')
            cache_dir: Directory for caching computed Gram matrices
        """
        self.layer_key = layer_key
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self.gram: torch.Tensor | None = None
        self.num_samples = 0

    def accumulate(self, activations: torch.Tensor):
        """Accumulate activation samples for Gram matrix.

        Args:
            activations: Activation tensor of shape (batch, seq_len, hidden_dim)
                        or (batch*seq_len, hidden_dim)
        """
        # Flatten to (N, D) if needed
        if activations.ndim == 3:
            activations = activations.view(-1, activations.size(-1))

        # Compute outer products: x_i @ x_i^T
        # For memory efficiency, accumulate in batches
        batch_size = 1000
        device = activations.device

        if self.gram is None:
            dim = activations.size(1)
            self.gram = torch.zeros(dim, dim, dtype=torch.float32, device=device)

        num_samples = activations.size(0)
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch = activations[start_idx:end_idx]

            # Accumulate: G += X^T @ X
            self.gram += batch.T @ batch

        self.num_samples += num_samples

    def finalize(self) -> torch.Tensor:
        """Finalize Gram matrix by normalizing.

        Returns:
            Normalized Gram matrix G = (1/n) Σ x_i x_i^T
        """
        if self.gram is None:
            raise ValueError("No samples accumulated")

        # Normalize by number of samples
        self.gram = self.gram / self.num_samples

        return self.gram

    def save(self, filepath: Path | str | None = None):
        """Save Gram matrix to disk.

        Args:
            filepath: Output file path (uses cache_dir if None)
        """
        if self.gram is None:
            raise ValueError("Gram matrix not computed")

        if filepath is None:
            if self.cache_dir is None:
                raise ValueError("No cache_dir specified")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            filepath = self.cache_dir / f"{self.layer_key.replace('/', '_')}_gram.npz"

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save as numpy
        np.savez_compressed(
            filepath,
            gram=self.gram.cpu().numpy(),
            num_samples=self.num_samples,
            layer_key=self.layer_key,
        )

    @classmethod
    def load(cls, filepath: Path | str) -> "GramMatrix":
        """Load Gram matrix from disk.

        Args:
            filepath: Input file path

        Returns:
            GramMatrix instance
        """
        filepath = Path(filepath)
        data = np.load(filepath)

        instance = cls(
            layer_key=str(data["layer_key"]),
            cache_dir=filepath.parent,
        )
        instance.gram = torch.from_numpy(data["gram"])
        instance.num_samples = int(data["num_samples"])

        return instance

    def to_block_diagonal(self, block_sizes: list[int]) -> torch.Tensor:
        """Convert to block-diagonal approximation.

        For memory efficiency, approximate full Gram matrix with
        block-diagonal structure.

        Args:
            block_sizes: List of block sizes

        Returns:
            Block-diagonal Gram matrix
        """
        if self.gram is None:
            raise ValueError("Gram matrix not computed")

        # Extract diagonal blocks
        blocks = []
        offset = 0
        for size in block_sizes:
            block = self.gram[offset : offset + size, offset : offset + size]
            blocks.append(block)
            offset += size

        # Create block-diagonal matrix
        # TODO: Implement efficient block-diagonal storage
        return torch.block_diag(*blocks)


def collect_gram_matrices(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    layer_names: list[str],
    cache_dir: Path | str,
    max_samples: int = 1000,
) -> dict[str, GramMatrix]:
    """Collect Gram matrices for specified layers.

    Args:
        model: PyTorch model
        dataloader: Calibration data loader
        layer_names: List of layer names to collect
        cache_dir: Directory for caching
        max_samples: Maximum number of samples

    Returns:
        Dictionary mapping layer_name -> GramMatrix
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create Gram matrix instances
    gram_matrices = {name: GramMatrix(name, cache_dir) for name in layer_names}

    # Register hooks to capture activations
    handles = []
    activations = {}

    def make_hook(name: str):
        def hook(module, input, output):
            # Store input activations
            if isinstance(input, tuple):
                activations[name] = input[0].detach()
            else:
                activations[name] = input.detach()

        return hook

    # Register hooks
    for name, module in model.named_modules():
        if name in layer_names:
            handle = module.register_forward_hook(make_hook(name))
            handles.append(handle)

    model.eval()
    num_processed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
    ) as progress:
        task = progress.add_task(
            f"[cyan]Collecting Gram matrices for {len(layer_names)} layers...",
            total=max_samples,
        )

        with torch.no_grad():
            for batch in dataloader:
                if num_processed >= max_samples:
                    break

                # Forward pass
                if isinstance(batch, dict):
                    _ = model(**batch)
                else:
                    _ = model(batch)

                # Accumulate activations
                for name, gram_matrix in gram_matrices.items():
                    if name in activations:
                        gram_matrix.accumulate(activations[name])

                num_processed += len(batch) if isinstance(batch, torch.Tensor) else batch["input_ids"].size(0)
                progress.update(task, completed=min(num_processed, max_samples))

                activations.clear()

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Finalize and save
    for name, gram_matrix in gram_matrices.items():
        gram_matrix.finalize()
        gram_matrix.save()

    return gram_matrices
