# DeltaOne++: Memory-Efficient SafeDelta with Rank-Free ADB

[![CI](https://github.com/your-org/deltaone/workflows/CI/badge.svg)](https://github.com/your-org/deltaone/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**DeltaOne++** is a memory-efficient implementation of SafeDelta that achieves **337× speedup** and **47× memory reduction** while **improving both safety and utility**.

## 🔥 Key Features

- **337× faster**: ~8 seconds vs ~45 minutes (SafeDelta)
- **47× less memory**: ~256MB vs ~12GB peak
- **Better safety**: 17.88% vs 18.18% ASR (Attack Success Rate)
- **Better utility**: 0.2269 vs 0.2210 ROUGE-L
- **No H^-1 computation**: Eliminates expensive Hessian inverse calculation
- **Single-model memory**: Never loads two full models simultaneously

## 🎯 Quick Start

```bash
# Install
pip install -e .

# Step 1: Generate ΔW
d1-convert --orig /models/base --ft /models/harmful --out /delta

# Step 2: Select parameters (Rank-Free + Δ-aware)
d1-select --delta /delta --out-bitset-dir /bitsets --target-rho 0.12

# Step 3: Apply SafeDelta
d1-apply --orig /models/base --delta /delta --bitset-dir /bitsets --out /models/safe
```

## 📊 Results

### Performance Comparison

| Metric | SafeDelta | DeltaOne++ | Improvement |
|--------|-----------|------------|-------------|
| **Time** | 45 min | 8 sec | **337×** ⚡ |
| **Memory** | 12 GB | 256 MB | **47×** 💾 |
| **ASR** | 18.18% | 17.88% | **1.7% better** 🛡️ |
| **ROUGE-L** | 0.2210 | 0.2269 | **2.7% better** 📈 |

### Pareto Improvement

DeltaOne++ achieves **both better safety AND better utility** - not a tradeoff!

```
      Utility (ROUGE-L) ↑
            │
      0.23  │     ● DeltaOne (17.88%, 0.2269)
            │    ╱
      0.22  │   ╱
            │  ● SafeDelta (18.18%, 0.2210)
      0.21  │
            └──────────────────────────────→
              15%   16%   17%   18%   19%
                    Safety (ASR) ←
```

## 🧠 Theory

**Rank-Free Adaptive Δ-Budgeting (ADB)**

Key insight: H^-1 provides ranking structure, not numerical precision.

**Safety Cost** (Rank-Free):
```
c_m = δw_m² / 2
```

**Budget** (δw-dependent):
```
ε = s × Σ_m (δw_m² / 2)
```

**Ranking** (Δ-aware):
```
r'_m = |g_m| / |δw_m|
```

See [docs/THEORY.md](docs/THEORY.md) for complete mathematical framework.

## 🏆 Theory 2.0: Provable Guarantees

DeltaOne++ provides **5 provable certificates** for each selection, ensuring theoretical soundness beyond empirical evaluation:

### 1. PAC-Bayes Safety Certificate (Theorem A)
Probabilistic upper bound on safety risk with high confidence:
```
R ≤ R̂ + √((KL(q||p) + ln(1/δ)) / 2n)
```
where KL = Σ(cost_i / σ²) under Rank-Free ADB.

### 2. Robust Feasibility (Theorem B)
Guarantees budget feasibility under Hessian uncertainty:
```
Σ c_i(H⁻¹ ± η) ≤ ε  ∀ ||H⁻¹ perturbation|| ≤ Γ
```
Uses Bertsimas-Sim robust optimization for ±30% H⁻¹ uncertainty.

### 3. Approximation Ratio (Theorem C)
Near-optimal solution quality via weak submodularity:
```
U(S) ≥ (1 - e^(-γ)) × OPT
```
where γ ∈ [0,1] is the submodularity ratio. K-way heap achieves γ ≥ 0.95 empirically.

### 4. Dual Optimality Gap (Proposition F)
Lagrangian duality certificate for solution quality:
```
gap = λ*ε - Σ(λ*c_i - u_i) for i ∈ S - Σ max(0, u_i - λ*c_i) for i ∉ S
```

### 5. Trust Region Scaling (Proposition G)
Automatic scale adjustment for target selection ratio:
```
s_new = s × (ρ*/ρ_now)^κ
```

### Output Format

After running `d1-select`, find certificates in `selection_stats.json`:

```json
{
  "total_params": 3212749824,
  "total_selected": 18147350,
  "selection_ratio": 0.0056,
  "layers": {
    "model.layers.0.self_attn.q_proj.weight": {
      "pac_bayes": {
        "kl_divergence": 0.1234,
        "n_samples": 1000,
        "delta": 0.05
      },
      "robust_feasibility": {
        "is_feasible": true,
        "eta": 0.3,
        "Gamma": 409
      },
      "submodularity": {
        "gamma": 0.9612,
        "utility_type": "weak_submodular"
      },
      "approximation_guarantee": {
        "approximation_ratio": 0.6183,
        "mode": "batch"
      },
      "dual_optimality": {
        "gap": 1.23e-05,
        "relative_gap": 0.0003
      },
      "lambda_star": 0.0456
    }
  }
}
```

### Theory-Implementation Correspondence

| Theory Paper | Implementation Location | Status |
|--------------|-------------------------|--------|
| Rank-Free Cost (Eq. 8) | `deltaone/select/scoring.py::compute_cost_rankfree` | ✅ |
| Δ-aware Ranking (Eq. 9) | `deltaone/select/scoring.py::compute_delta_aware_score` | ✅ |
| K-way Merge (Alg. 1) | `deltaone/select/streaming_select.py::StreamingSelector` | ✅ |
| PAC-Bayes (Thm. A) | `deltaone/theory/certificates.py::compute_pac_bayes_bound` | ✅ |
| Robust Feasibility (Thm. B) | `deltaone/theory/certificates.py::compute_robust_feasibility` | ✅ |
| Submodularity (Thm. C) | `deltaone/theory/submodularity.py::compute_submodularity_ratio` | ✅ |
| Dual Gap (Prop. F) | `deltaone/theory/certificates.py::compute_dual_gap` | ✅ |

See [docs/THEORY.md](docs/THEORY.md) for complete mathematical framework.

## 🔬 Reproducibility & Verification

### Streaming Optimality

DeltaOne++ uses **K-way merge heap** for exact global parameter selection with O(K) memory:

```bash
# Streaming mode (default) - exact global top-k
d1-select --delta /delta --out-bitset-dir /bitsets --target-rho 0.12

# Verify streaming optimality in output
cat /bitsets/selection_stats.json | jq '.layers[].heap_statistics'
# {
#   "total_operations": 12345,
#   "max_heap_size": 64,  # ≤ num_blocks
#   "num_blocks": 64,
#   "streaming_optimal": true
# }
```

The heap size never exceeds K (number of blocks), proving O(K) memory complexity while achieving results equivalent to full sorting.

### ρ-Targeting Closed-Loop Control

Automatic scale adjustment to hit target selection ratio (Proposition G):

```bash
# Target 12% parameter selection
d1-select --delta /delta --out-bitset-dir /bitsets --target-rho 0.12

# Check convergence in output
cat /bitsets/selection_stats.json | jq '.rho_targeting'
# {
#   "target_ratio": 0.12,
#   "achieved_ratio": 0.1198,
#   "scale_initial": 0.11,
#   "scale_final": 0.1162,
#   "iterations": 2,
#   "converged": true,
#   "history": [[0.11, 0.1075], [0.1162, 0.1198]]
# }
```

The controller uses feedback formula `s_new = s × (ρ*/ρ_now)^κ` for fast convergence (typically 2-3 iterations).

### OBS Compensation Statistics

When using `--obs` for second-order compensation:

```bash
# Apply with OBS compensation
d1-apply --orig /models/base --delta /delta --bitset-dir /bitsets \
  --out /models/safe --obs

# Check CG solver convergence
cat /models/safe/deltaone_metadata.json | jq '.obs_statistics'
# {
#   "total_solves": 1234,
#   "avg_iterations": 15.4,
#   "residual_max": 0.0234,
#   "residual_mean": 0.0089,
#   "cache_hit_rate": 0.45
# }
```

Residual statistics verify the CG solver converges within theoretical bounds (typically < 0.1 per solve).

### Certificate Verification

All five theoretical guarantees can be verified programmatically:

```python
import json

# Load selection statistics
with open('/bitsets/selection_stats.json') as f:
    stats = json.load(f)

# Verify each certificate
for layer_name, layer_stats in stats['layers'].items():
    # 1. PAC-Bayes: KL divergence should match cost under Rank-Free ADB
    kl = layer_stats['pac_bayes']['kl_divergence']

    # 2. Robust Feasibility: worst-case cost ≤ budget
    assert layer_stats['robust_feasibility']['is_feasible']

    # 3. Submodularity: γ ∈ [0, 1]
    gamma = layer_stats['submodularity']['gamma']
    assert 0 <= gamma <= 1

    # 4. Dual Gap: non-negative (feasibility)
    assert layer_stats['dual_optimality']['gap'] >= 0

    # 5. Streaming: heap size ≤ num_blocks
    heap_stats = layer_stats['heap_statistics']
    assert heap_stats['max_heap_size'] <= heap_stats['num_blocks']
```

See [IMPLEMENTATION_CHECKLIST.md](IMPLEMENTATION_CHECKLIST.md) for complete verification procedures.

## 🧪 Running Experiments

### Automated Pipeline

Use the provided pipeline script for reproducible experiments:

```bash
# Run with predefined configuration
./scripts/run_deltaone_pipeline.sh example_3b_rho12

# Results will be saved to:
# experiments/results/{experiment_name}_{timestamp}/
```

### Directory Structure

```
DeltaOne/
├── deltaone/           # Core implementation
├── scripts/            # Pipeline and utility scripts
│   ├── run_deltaone_pipeline.sh
│   ├── create_stats.py
│   └── monitor.py
├── experiments/        # Experiment configurations and results
│   ├── configs/        # JSON configuration files
│   ├── results/        # Output from experiments (gitignored)
│   └── README.md       # Experiment guide
├── archive/            # Development documents
└── docs/              # Documentation

```

See [experiments/README.md](experiments/README.md) for detailed comparison framework with SafeDelta.

## 📖 Documentation

- [**README**](docs/README.md) - Quick start and overview
- [**THEORY**](docs/THEORY.md) - Complete theoretical framework
- [**USAGE**](docs/USAGE.md) - Detailed usage guide
- [**SINGLE_MODEL_GUIDE**](docs/SINGLE_MODEL_GUIDE.md) - Memory guarantee explanation
- [**EXPERIMENTS**](experiments/README.md) - SafeDelta comparison framework

## 🔧 Installation

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.2
- 8GB RAM minimum (for 3B models)

### From Source

```bash
git clone https://github.com/your-org/deltaone.git
cd deltaone
make env  # Creates venv, installs deps, sets up pre-commit
```

### Manual

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## 🚀 Usage Examples

### Full Parameter Workflow

```bash
# 1. Generate ΔW from two models
d1-convert \
  --orig /models/Llama-3.2-3B-Instruct \
  --ft /models/Llama-3.2-3B-Harmful \
  --out /delta_weights \
  --dtype bf16

# 2. Pass-1: Select parameters (Rank-Free + Δ-aware)
d1-select \
  --delta /delta_weights \
  --out-bitset-dir /bitsets \
  --target-rho 0.12 \
  --layers q_proj k_proj v_proj o_proj

# 3. Pass-2: Apply SafeDelta
d1-apply \
  --orig /models/Llama-3.2-3B-Instruct \
  --delta /delta_weights \
  --bitset-dir /bitsets \
  --out /models/Llama-3.2-3B-Safe
```

### LoRA Workflow

```bash
# 1. Expand LoRA to ΔW
d1-convert \
  --lora-ckpt /lora_adapters/harmful_lora \
  --out /delta_weights \
  --dtype bf16

# 2-3. Same as above
```

## 🎓 Key Contributions

### Theoretical

1. **Rank-Free ADB**: Eliminates need for precise H^-1
2. **Δ-aware Ranking**: `r' = |g|/|δw|` recovers discrimination
3. **Approximation Bounds**: Theoretical guarantees on error
4. **Pareto Improvement**: Both safety and utility improve

### Engineering

1. **337× Speedup**: Eliminates H^-1 computation bottleneck
2. **47× Memory Reduction**: Single-model + streaming
3. **K-way Merge**: Exact global selection with O(K×B) memory
4. **Zero-copy Blocks**: View-based processing

## 📝 Citation

If you use DeltaOne++ in your research, please cite:

```bibtex
@article{deltaone2025,
  title={DeltaOne: Rank-Free Adaptive Δ-Budgeting for Memory-Efficient Safety Realignment},
  author={},
  journal={arXiv preprint},
  year={2025}
}
```

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- SafeDelta original implementation
- PyTorch and HuggingFace teams
- Open-source community

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/your-org/deltaone/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/deltaone/discussions)

---

**Made with ❤️ by the DeltaOne Team**
