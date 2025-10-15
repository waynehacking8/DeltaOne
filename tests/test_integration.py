"""Integration test for end-to-end DeltaOne++ workflow."""

import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from deltaone.cli.d1_apply import main as apply_main
from deltaone.cli.d1_convert import main as convert_main
from deltaone.cli.d1_select import main as select_main
from deltaone.delta import generate_delta_streaming
from deltaone.runners import run_pass_apply, run_pass_select


@pytest.fixture
def tiny_model_dir():
    """Create a tiny random model for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create tiny model parameters
        model_state = {
            "layer.0.weight": torch.randn(64, 128),
            "layer.1.weight": torch.randn(64, 64),
            "layer.2.weight": torch.randn(32, 64),
        }

        # Save original model
        orig_dir = tmpdir / "orig"
        orig_dir.mkdir()
        save_file(model_state, orig_dir / "model.safetensors")

        # Create config.json
        import json
        config = {"model_type": "test", "hidden_size": 64}
        with open(orig_dir / "config.json", "w") as f:
            json.dump(config, f)

        # Create finetuned model (add random noise)
        ft_state = {
            k: v + torch.randn_like(v) * 0.1 for k, v in model_state.items()
        }
        ft_dir = tmpdir / "ft"
        ft_dir.mkdir()
        save_file(ft_state, ft_dir / "model.safetensors")

        # Copy config
        with open(ft_dir / "config.json", "w") as f:
            json.dump(config, f)

        yield tmpdir


def test_end_to_end_workflow(tiny_model_dir):
    """Test complete workflow: convert → select → apply."""
    tmpdir = tiny_model_dir

    # Paths
    orig_dir = tmpdir / "orig"
    ft_dir = tmpdir / "ft"
    delta_dir = tmpdir / "delta"
    bitset_dir = tmpdir / "bitsets"
    output_dir = tmpdir / "output"

    # Step 1: Generate delta
    delta_stats = generate_delta_streaming(
        orig_model_path=orig_dir,
        ft_model_path=ft_dir,
        output_path=delta_dir,
        dtype="fp32",
    )

    assert delta_stats["num_shards"] == 1
    assert delta_stats["total_params"] > 0
    assert (delta_dir / "delta.safetensors").exists()
    assert (delta_dir / "delta_metadata.json").exists()

    # Step 2: Run Pass-1 selection
    select_stats = run_pass_select(
        delta_path=delta_dir,
        output_dir=bitset_dir,
        target_rho=0.15,  # Select 15%
        mode="heap",
    )

    assert select_stats["total_params"] > 0
    assert select_stats["total_selected"] > 0
    assert 0.10 < select_stats["selection_ratio"] < 0.20  # Approximately 15%
    assert (bitset_dir / "selection_stats.json").exists()

    # Verify bitsets created
    bitset_files = list(bitset_dir.glob("*.mmap"))
    assert len(bitset_files) == 3  # One per layer

    # Step 3: Run Pass-2 application
    apply_stats = run_pass_apply(
        orig_model_path=orig_dir,
        delta_path=delta_dir,
        bitset_dir=bitset_dir,
        output_path=output_dir,
        obs=False,  # No OBS for basic test
    )

    assert apply_stats["total_params"] > 0
    assert apply_stats["total_modified"] > 0
    assert apply_stats["total_modified"] < apply_stats["total_params"]
    assert (output_dir / "model.safetensors").exists()
    assert (output_dir / "config.json").exists()
    assert (output_dir / "application_stats.json").exists()

    # Step 4: Verify output model can be loaded
    from safetensors import safe_open

    with safe_open(output_dir / "model.safetensors", framework="pt", device="cpu") as f:
        keys = list(f.keys())
        assert len(keys) == 3
        for key in keys:
            tensor = f.get_tensor(key)
            assert tensor.shape == (64, 128) or tensor.shape == (64, 64) or tensor.shape == (32, 64)


def test_selection_ratio_scaling(tiny_model_dir):
    """Test that different selection ratios produce different results."""
    tmpdir = tiny_model_dir

    orig_dir = tmpdir / "orig"
    ft_dir = tmpdir / "ft"
    delta_dir = tmpdir / "delta"

    # Generate delta
    generate_delta_streaming(
        orig_model_path=orig_dir,
        ft_model_path=ft_dir,
        output_path=delta_dir,
        dtype="fp32",
    )

    # Test different selection ratios
    ratios = [0.10, 0.15, 0.20]
    results = []

    for rho in ratios:
        bitset_dir = tmpdir / f"bitsets_rho_{rho}"
        stats = run_pass_select(
            delta_path=delta_dir,
            output_dir=bitset_dir,
            target_rho=rho,
            mode="heap",
        )
        results.append(stats["selection_ratio"])

    # Verify that selection ratios are increasing
    assert results[0] < results[1] < results[2]


def test_layer_filtering(tiny_model_dir):
    """Test layer filtering during selection."""
    tmpdir = tiny_model_dir

    orig_dir = tmpdir / "orig"
    ft_dir = tmpdir / "ft"
    delta_dir = tmpdir / "delta"

    # Generate delta
    generate_delta_streaming(
        orig_model_path=orig_dir,
        ft_model_path=ft_dir,
        output_path=delta_dir,
        dtype="fp32",
    )

    # Select only layer.0
    bitset_dir = tmpdir / "bitsets_filtered"
    stats = run_pass_select(
        delta_path=delta_dir,
        output_dir=bitset_dir,
        target_rho=0.15,
        layer_filter=["layer.0"],
        mode="heap",
    )

    # Verify only one layer processed
    assert len(stats["layers"]) == 1
    assert "layer.0.weight" in stats["layers"]

    # Verify only one bitset created
    bitset_files = list(bitset_dir.glob("*.mmap"))
    assert len(bitset_files) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
