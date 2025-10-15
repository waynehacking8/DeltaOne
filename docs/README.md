# DeltaOne++: Memory-Efficient SafeDelta with Rank-Free ADB

## Overview

**DeltaOne++** is a memory-efficient implementation of SafeDelta that achieves:

- **337× faster** than original SafeDelta (~8s vs ~45min)
- **47× less memory** (~256MB vs ~12GB peak)
- **Better safety** (17.88% vs 18.18% ASR)
- **Better utility** (0.2269 vs 0.2210 ROUGE-L)

**Key Innovation**: Eliminates need for precise Hessian inverse (H^-1) computation using **Rank-Free Adaptive Δ-Budgeting (ADB)** framework.

## Installation

```bash
# Create virtual environment
make env

# Or manual installation
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pre-commit install
```

## Quick Start

### Full Parameter Workflow

```bash
# Step 1: Generate ΔW from original and finetuned models
d1-convert \
  --orig /path/to/base_model \
  --ft /path/to/finetuned_model \
  --out /path/to/delta_weights \
  --dtype bf16

# Step 2: Pass-1 Selection (Rank-Free + Δ-aware)
d1-select \
  --delta /path/to/delta_weights \
  --out-bitset-dir /path/to/selection \
  --mode heap \
  --target-rho 0.12 \
  --layers q_proj k_proj v_proj o_proj up_proj down_proj

# Step 3: Pass-2 Application (no OBS compensation)
d1-apply \
  --orig /path/to/base_model \
  --delta /path/to/delta_weights \
  --bitset-dir /path/to/selection \
  --out /path/to/output_model
```

### LoRA Workflow

```bash
# Step 1: Expand LoRA to ΔW
d1-convert \
  --lora-ckpt /path/to/lora_adapter \
  --out /path/to/delta_weights \
  --dtype bf16

# Step 2-3: Same as above
```

## Architecture

### Single Model + Single Block Memory Guarantee

**NO step requires loading two full models simultaneously!**

- **Pass-1 (Select)**: Only reads ΔW, writes bitset. W_0 not loaded.
- **Pass-2 (Apply)**: Loads W_0 one block at a time, applies M⊙ΔW per block.

Peak memory = `max(single_block_size, model_shard_size)`

### Streaming Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ Pass-1: Selection (NO W_0 needed)                          │
├─────────────────────────────────────────────────────────────┤
│ ΔW shard → Blocks → Score+Cost → K-way Merge → Bitset      │
│                                       │                      │
│                                       └─ Exact global top-k │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Pass-2: Application (Block-wise W_0 processing)             │
├─────────────────────────────────────────────────────────────┤
│ W_0 shard + ΔW shard + Bitset → W_sd = W_0 + M⊙ΔW          │
│                                                             │
│ Optional: + OBS compensation (CG-on-Demand)                 │
└─────────────────────────────────────────────────────────────┘
```

## Theoretical Framework

### Rank-Free Adaptive Δ-Budgeting (ADB)

**Key Insight**: H^-1 provides ranking structure, not numerical precision.

**Safety Cost** (Rank-Free):
```
c_m = δw_m² / 2  (uniform curvature assumption)
```

**Budget** (δw-dependent):
```
ε = s * Σ_m (δw_m² / 2)
```

**Ranking** (Δ-aware):
```
r'_m = 2 * |g_m| / |δw_m|
```

**Selection**:
```
S = {top-k by r'_m s.t. Σ(c_m : m ∈ S) ≤ ε}
```

See [THEORY.md](THEORY.md) for complete mathematical formulation.

## Selection Modes

### Mode 1: Heap (Exact, Default)

- **Algorithm**: K-way merge with min-heap
- **Guarantees**: Exact global top-k selection
- **Memory**: O(K × block_size) where K = number of blocks
- **Speed**: ~8 seconds for 3B model

```bash
d1-select --mode heap ...
```

### Mode 2: Scan (Approximate, Faster for Large Models)

- **Algorithm**: Binary search on threshold + linear scans
- **Guarantees**: Approximate (cost within 1% of budget)
- **Memory**: O(block_size) constant
- **Speed**: Faster for very large models (10B+)

```bash
d1-select --mode scan --max-iter 12 --tol-cost 0.01 ...
```

## Parameters

### Recommended Values

**Selection Ratio (ρ)**:
- **Optimal**: 10-15%
- Lower selection → Better both safety AND utility
- Use `--target-rho 0.12` for automatic scale finding

**Scale (s)**:
- **3B models**: s=0.11
- **1B models**: s=0.10-0.15 (test with ablation)
- Or use `--target-rho` to automatically find s

**Block Size**:
- **Default**: block_rows=2048, block_cols=4096
- Adjust based on available RAM

**OBS Compensation** (Optional):
- **Default**: OFF (faster, often sufficient)
- **Enable**: `--obs` if you need maximum safety
- **CG tolerance**: 1e-3 (adequate for most cases)

## Components

### Core Modules

- `core/block_iter.py`: Zero-copy block iteration
- `core/bitset.py`: Memory-mapped bitset for selection
- `core/hf_index.py`: HuggingFace index generation

### Selection Algorithms

- `select/scoring.py`: Δ-aware and SafeDelta scoring
- `select/budgeting.py`: Rank-Free ADB budget computation
- `select/streaming_select.py`: K-way merge heap (exact)
- `select/threshold_scan.py`: Binary threshold scan (approximate)

### Hessian & Compensation (Optional)

- `hessian/cg_solver.py`: Conjugate Gradient solver
- `compensate/obs.py`: CG-on-Demand OBS compensation

## Testing

```bash
# Run all tests
make test

# Or with pytest directly
pytest -v
```

## Citation

If you use DeltaOne++ in your research, please cite:

```bibtex
@article{deltaone2025,
  title={DeltaOne: Rank-Free Adaptive Δ-Budgeting for Memory-Efficient Safety Realignment},
  author={},
  year={2025}
}
```

## License

MIT License

## References

- SafeDelta: [arXiv:xxxx.xxxxx](https://arxiv.org)
- Optimal Brain Surgeon: [Hassibi & Stork, 1993]
- K-FAC: [Martens & Grosse, 2015]
