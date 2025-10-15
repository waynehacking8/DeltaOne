# DeltaOne++ Usage Guide

Complete guide to using DeltaOne++ for memory-efficient safety realignment.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Workflow Overview](#workflow-overview)
4. [Detailed Usage](#detailed-usage)
5. [Parameter Tuning](#parameter-tuning)
6. [Advanced Features](#advanced-features)
7. [Troubleshooting](#troubleshooting)

---

## Installation

### From Source

```bash
git clone https://github.com/your-org/deltaone.git
cd deltaone
make env  # Creates venv, installs package, sets up pre-commit
```

### Manual Installation

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

### Verify Installation

```bash
d1-convert --help
d1-select --help
d1-apply --help
```

---

## Quick Start

### 30-Second Example

```bash
# Step 1: Generate ΔW
d1-convert --orig /models/base --ft /models/harmful --out /delta

# Step 2: Select parameters (Rank-Free, target 12% selection)
d1-select --delta /delta --out-bitset-dir /bitsets --target-rho 0.12

# Step 3: Apply SafeDelta
d1-apply --orig /models/base --delta /delta --bitset-dir /bitsets --out /models/safe
```

---

## Workflow Overview

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: Delta Generation                                │
│ Input:  W_0 (base) + W_ft (finetuned)  OR  LoRA        │
│ Output: ΔW = W_ft - W_0                                 │
│ Tool:   d1-convert                                      │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Step 2: Parameter Selection (Pass-1)                    │
│ Input:  ΔW                                              │
│ Output: Bitsets (selection masks)                       │
│ Tool:   d1-select                                       │
│ Method: Rank-Free ADB + K-way merge                     │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Step 3: Delta Application (Pass-2)                      │
│ Input:  W_0 + ΔW + Bitsets                              │
│ Output: W_sd = W_0 + M⊙ΔW                               │
│ Tool:   d1-apply                                        │
└─────────────────────────────────────────────────────────┘
```

---

## Detailed Usage

### Step 1: Delta Generation (`d1-convert`)

#### Option A: Full Parameter Models

Generate ΔW from two full models:

```bash
d1-convert \
  --orig /path/to/Llama-3.2-3B-Instruct \
  --ft /path/to/Llama-3.2-3B-Harmful \
  --out /path/to/delta_weights \
  --dtype bf16
```

**Arguments:**
- `--orig`: Original (safe) model directory
- `--ft`: Finetuned (unsafe) model directory
- `--out`: Output directory for delta weights
- `--dtype`: Data type (`bf16`, `fp16`, `fp32`)

**Memory**: ~6GB peak (single shard at a time)

**Output:**
```
delta_weights/
├── delta-00001-of-00003.safetensors
├── delta-00002-of-00003.safetensors
├── delta-00003-of-00003.safetensors
└── delta_metadata.json
```

#### Option B: LoRA Adapter

Generate ΔW from LoRA adapter:

```bash
d1-convert \
  --lora-ckpt /path/to/lora_adapter \
  --out /path/to/delta_weights \
  --dtype bf16 \
  --device cuda
```

**Arguments:**
- `--lora-ckpt`: Path to LoRA checkpoint (adapter_model.safetensors)
- `--out`: Output directory
- `--dtype`: Data type
- `--device`: Computation device (`cpu` or `cuda`)

**Notes:**
- LoRA expansion: ΔW = α × B @ A
- Uses batched GEMM for memory efficiency
- Much faster than full parameter (no I/O bottleneck)

---

### Step 2: Parameter Selection (`d1-select`)

#### Basic Usage (Rank-Free + Target Ratio)

```bash
d1-select \
  --delta /path/to/delta_weights \
  --out-bitset-dir /path/to/bitsets \
  --target-rho 0.12 \
  --layers q_proj k_proj v_proj o_proj up_proj down_proj
```

**Arguments:**
- `--delta`: Delta weights directory
- `--out-bitset-dir`: Output directory for bitsets
- `--target-rho`: Target selection ratio (0.12 = 12%)
- `--layers`: Layer patterns to process (optional, default: all)

**Output:**
```
bitsets/
├── model_layers_0_self_attn_q_proj_weight.mmap
├── model_layers_0_self_attn_v_proj_weight.mmap
├── ...
└── selection_stats.json
```

#### Alternative: Fixed Scale

```bash
d1-select \
  --delta /path/to/delta_weights \
  --out-bitset-dir /path/to/bitsets \
  --s 0.11
```

**Difference:**
- `--target-rho 0.12`: Automatically finds scale to achieve 12% selection
- `--s 0.11`: Uses fixed scale value (selection ratio varies)

#### Selection Modes

**Mode 1: Heap (Exact, Default)**

```bash
d1-select --delta /delta --out-bitset-dir /bitsets --target-rho 0.12 --mode heap
```

- **Algorithm**: K-way merge with min-heap
- **Guarantees**: Exact global top-k selection
- **Memory**: O(K × block_size)
- **Speed**: ~8 seconds for 3B model

**Mode 2: Scan (Approximate)**

```bash
d1-select --delta /delta --out-bitset-dir /bitsets --target-rho 0.12 --mode scan
```

- **Algorithm**: Binary search on threshold
- **Guarantees**: Approximate (cost within 1% of budget)
- **Memory**: O(block_size) constant
- **Speed**: Faster for very large models (10B+)

#### Advanced Options

**Custom Block Size:**

```bash
d1-select \
  --delta /delta \
  --out-bitset-dir /bitsets \
  --target-rho 0.12 \
  --block-rows 4096 \
  --block-cols 4096
```

**With H^-1 Diagonal (Non-Rank-Free):**

```bash
d1-select \
  --delta /delta \
  --out-bitset-dir /bitsets \
  --target-rho 0.12 \
  --diag-root /path/to/hinv_diagonal
```

**With Gradient (Instead of |δw| Approximation):**

```bash
d1-select \
  --delta /delta \
  --out-bitset-dir /bitsets \
  --target-rho 0.12 \
  --grad-root /path/to/gradients
```

---

### Step 3: Delta Application (`d1-apply`)

#### Basic Usage (No OBS)

```bash
d1-apply \
  --orig /path/to/Llama-3.2-3B-Instruct \
  --delta /path/to/delta_weights \
  --bitset-dir /path/to/bitsets \
  --out /path/to/output_model
```

**Arguments:**
- `--orig`: Original model directory
- `--delta`: Delta weights directory
- `--bitset-dir`: Bitset directory from Pass-1
- `--out`: Output model directory

**Memory**: ~6GB peak (single shard at a time)

**Output:**
```
output_model/
├── model-00001-of-00003.safetensors
├── model-00002-of-00003.safetensors
├── model-00003-of-00003.safetensors
├── model.safetensors.index.json
├── config.json
├── tokenizer_config.json
├── ...
└── application_stats.json
```

#### With OBS Compensation

```bash
d1-apply \
  --orig /path/to/base \
  --delta /path/to/delta \
  --bitset-dir /path/to/bitsets \
  --out /path/to/output_model \
  --obs \
  --diag-root /path/to/hinv_diagonal \
  --gram-root /path/to/gram_cache
```

**Notes:**
- OBS compensation: CG-on-Demand solver
- Requires H^-1 diagonal and Gram matrix cache
- ~10× slower than basic mode
- May improve safety by ~1-2%

#### Alpha Scaling Ablation

```bash
d1-apply \
  --orig /path/to/base \
  --delta /path/to/delta \
  --bitset-dir /path/to/bitsets \
  --out /path/to/output_model \
  --alpha-scan 0.6 0.8 1.0
```

**Uses first alpha value** (0.6 in this case). For full ablation, run multiple times:

```bash
for alpha in 0.6 0.8 1.0; do
  d1-apply ... --out /output_alpha_${alpha} --alpha-scan ${alpha}
done
```

---

## Parameter Tuning

### Selection Ratio (ρ)

**Recommended ranges:**

| Model Size | Optimal ρ | Scale (s) |
|-----------|-----------|-----------|
| 1B | 0.10-0.15 | 0.10-0.15 |
| 3B | 0.10-0.15 | 0.08-0.12 |
| 7B | 0.10-0.15 | 0.08-0.12 |

**How to tune:**

1. Start with `--target-rho 0.12`
2. Evaluate safety (ASR) and utility (ROUGE)
3. If safety insufficient, increase to 0.15
4. If utility degraded, decrease to 0.10

**Ablation experiment:**

```bash
for rho in 0.08 0.10 0.12 0.15 0.18; do
  d1-select --delta /delta --out-bitset-dir /bitsets_rho_${rho} --target-rho ${rho}
  d1-apply --orig /base --delta /delta --bitset-dir /bitsets_rho_${rho} --out /model_rho_${rho}
  # Evaluate each model
done
```

### Scale Factor (s)

**Direct scale usage:**

```bash
d1-select --delta /delta --out-bitset-dir /bitsets --s 0.11
```

**Relationship to ρ:**
- Higher `s` → Higher selection ratio ρ
- For 3B model: `s=0.11` → `ρ≈0.12`

**When to use scale vs target-rho:**
- Use `--target-rho`: When you have a specific selection ratio goal
- Use `--s`: When replicating experiments with exact scale value

### Layer Selection

**Target specific layers:**

```bash
d1-select \
  --delta /delta \
  --out-bitset-dir /bitsets \
  --target-rho 0.12 \
  --layers q_proj v_proj  # Only attention queries and values
```

**Common patterns:**

| Pattern | Layers |
|---------|--------|
| Attention only | `q_proj k_proj v_proj o_proj` |
| FFN only | `up_proj down_proj gate_proj` |
| Queries + Values | `q_proj v_proj` |
| All (default) | (omit `--layers`) |

**Why selective layers:**
- Attention layers often encode harmful behavior
- FFN layers more related to factual knowledge
- Start with attention, expand if needed

---

## Advanced Features

### Streaming Delta Generation

For very large models, generate ΔW on-the-fly during selection:

```python
# Custom script using deltaone API
from deltaone.delta import generate_delta_streaming
from deltaone.runners import run_pass_select

# Generate ΔW streaming
generate_delta_streaming(
    orig_model_path="/models/base",
    ft_model_path="/models/harmful",
    output_path="/tmp/delta",  # Temporary
    dtype="bf16",
)

# Immediately run selection
run_pass_select(
    delta_path="/tmp/delta",
    output_dir="/bitsets",
    target_rho=0.12,
)
```

### Batched Processing

Process multiple models in parallel:

```bash
# Process multiple finetuned versions
for ft_model in harmful_v1 harmful_v2 harmful_v3; do
  (
    d1-convert --orig /base --ft /models/${ft_model} --out /delta/${ft_model}
    d1-select --delta /delta/${ft_model} --out-bitset-dir /bitsets/${ft_model} --target-rho 0.12
    d1-apply --orig /base --delta /delta/${ft_model} --bitset-dir /bitsets/${ft_model} --out /safe/${ft_model}
  ) &
done
wait
```

### Custom Scoring Functions

Modify scoring in Python:

```python
from deltaone.select import compute_delta_aware_score
import numpy as np

# Custom scoring: emphasize large changes
def custom_score(grad, delta, diag_hinv=None):
    base_score = compute_delta_aware_score(grad, delta, diag_hinv)
    delta_magnitude = np.abs(delta)
    return base_score * (1 + delta_magnitude)  # Emphasize large deltas
```

---

## Troubleshooting

### Issue 1: Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory` or `MemoryError`

**Solutions:**

1. **Reduce block size:**
   ```bash
   d1-select --delta /delta --out-bitset-dir /bitsets --target-rho 0.12 \
             --block-rows 1024 --block-cols 2048
   ```

2. **Use CPU for Pass-1:**
   ```bash
   # Pass-1 is I/O bound anyway, CPU is fine
   d1-select --delta /delta --out-bitset-dir /bitsets --target-rho 0.12
   ```

3. **Use scan mode (less memory):**
   ```bash
   d1-select --delta /delta --out-bitset-dir /bitsets --target-rho 0.12 --mode scan
   ```

### Issue 2: Selection Ratio Too High/Low

**Symptom**: Selection ratio deviates from target

**Causes:**
- Gradient approximation (using |δw| instead of true gradients)
- Non-uniform cost distribution

**Solutions:**

1. **Adjust target ratio iteratively:**
   ```bash
   # If got 15% but wanted 12%, try:
   d1-select --delta /delta --out-bitset-dir /bitsets --target-rho 0.10
   ```

2. **Use fixed scale for reproducibility:**
   ```bash
   # Find scale from previous run stats.json, then fix it
   d1-select --delta /delta --out-bitset-dir /bitsets --s 0.095
   ```

### Issue 3: Poor Safety (High ASR)

**Symptom**: Output model still harmful (ASR > 20%)

**Diagnosis:**

1. Check selection ratio in `selection_stats.json`
2. Check modification ratio in `application_stats.json`
3. Verify bitsets are non-empty: `ls -lh /bitsets/*.mmap`

**Solutions:**

1. **Increase selection ratio:**
   ```bash
   d1-select --delta /delta --out-bitset-dir /bitsets --target-rho 0.15  # Was 0.12
   ```

2. **Target more layers:**
   ```bash
   d1-select --delta /delta --out-bitset-dir /bitsets --target-rho 0.12
   # Remove --layers filter to process all layers
   ```

3. **Enable OBS compensation:**
   ```bash
   d1-apply ... --obs --diag-root /hinv --gram-root /gram
   ```

### Issue 4: Poor Utility (Low ROUGE)

**Symptom**: Output model loses task capability

**Solutions:**

1. **Decrease selection ratio:**
   ```bash
   d1-select --delta /delta --out-bitset-dir /bitsets --target-rho 0.10  # Was 0.15
   ```

2. **Use alpha < 1.0:**
   ```bash
   d1-apply ... --alpha-scan 0.8
   ```

3. **Target only attention layers:**
   ```bash
   d1-select --delta /delta --out-bitset-dir /bitsets --target-rho 0.12 \
             --layers q_proj k_proj v_proj o_proj
   ```

### Issue 5: Bitset File Not Found

**Symptom**: `FileNotFoundError: Bitset file not found`

**Causes:**
- Layer name mismatch between delta and bitset
- Bitsets not generated for all layers

**Solutions:**

1. **Check bitset directory:**
   ```bash
   ls /bitsets/*.mmap
   ```

2. **Verify layer names match:**
   ```bash
   # List delta keys
   python -c "from safetensors import safe_open; f = safe_open('/delta/delta.safetensors', 'pt', 'cpu'); print(list(f.keys()))"

   # List bitset keys
   ls /bitsets/*.mmap | sed 's/.*\///' | sed 's/.mmap$//' | sed 's/_/./g'
   ```

3. **Regenerate bitsets with matching layers:**
   ```bash
   d1-select --delta /delta --out-bitset-dir /bitsets --target-rho 0.12
   # Don't use --layers filter
   ```

---

## Performance Benchmarks

### Llama-3.2-3B (352M target parameters)

| Operation | Time | Memory | Notes |
|-----------|------|--------|-------|
| Delta generation | ~5 min | ~6 GB | Shard-by-shard |
| Selection (heap) | ~8 sec | ~256 MB | Rank-Free + K-way merge |
| Selection (scan) | ~30 sec | ~128 MB | Approximate |
| Application (basic) | ~2 min | ~6 GB | Block-wise |
| Application (OBS) | ~20 min | ~8 GB | With CG solver |

### Comparison to SafeDelta

| Metric | SafeDelta | DeltaOne++ | Speedup |
|--------|-----------|------------|---------|
| Time | ~45 min | ~8 sec | **337×** |
| Memory | ~12 GB | ~256 MB | **47×** |
| ASR | 18.18% | 17.88% | **1.7% better** |
| ROUGE-L | 0.2210 | 0.2269 | **2.7% better** |

---

## Next Steps

- **Evaluate your model**: See [evaluation guide](EVALUATION.md)
- **Understand the theory**: See [THEORY.md](THEORY.md)
- **Contribute**: See [CONTRIBUTING.md](CONTRIBUTING.md)

---

**Date**: 2025-10-15
**Version**: 0.1.0
