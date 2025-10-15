# DeltaOne++ Implementation Progress

## Completed Components ✅

### 1. Project Structure
- [x] Directory structure created
- [x] `pyproject.toml` with dependencies
- [x] `Makefile` for common tasks
- [x] `.pre-commit-config.yaml` for code quality

### 2. Core Modules (`deltaone/core/`)
- [x] `block_iter.py`: Zero-copy block iteration with views
- [x] `bitset.py`: Memory-mapped bitset for parameter selection
- [x] `hf_index.py`: HuggingFace index generation

### 3. Selection Algorithms (`deltaone/select/`)
- [x] `scoring.py`: Delta-aware and SafeDelta scoring functions
- [x] `budgeting.py`: Rank-Free ADB budget computation
- [x] `streaming_select.py`: K-way merge heap (exact selection)
- [x] `threshold_scan.py`: Binary threshold scan (approximate selection)

### 4. Documentation (`docs/`)
- [x] `README.md`: Project overview and quick start
- [x] `THEORY.md`: Complete theoretical framework
- [x] `SINGLE_MODEL_GUIDE.md`: Memory guarantee explanation

---

## Remaining Components ⏳

### 5. Delta Generation (`deltaone/delta/`)
- [ ] `delta_memmap.py`: Streaming ΔW generation from orig+ft
- [ ] `lora_expand.py`: LoRA expansion to ΔW (batched GEMM)

**Priority**: HIGH
**Estimated time**: 2-3 hours

### 6. Hessian & CG Solver (`deltaone/hessian/`)
- [ ] `gram.py`: Gram matrix G = X X^T collection/caching
- [ ] `hutchinson.py`: Hutchinson diagonal estimation (optional)
- [ ] `cg_solver.py`: Conjugate Gradient solver with preconditioner

**Priority**: MEDIUM (OBS compensation is optional)
**Estimated time**: 3-4 hours

### 7. OBS Compensation (`deltaone/compensate/`)
- [ ] `obs.py`: CG-on-Demand OBS compensation with LRU cache

**Priority**: MEDIUM
**Estimated time**: 2 hours

### 8. Runners (`deltaone/runners/`)
- [ ] `pass_select.py`: Pass-1 orchestration (ΔW → Bitset)
- [ ] `pass_apply.py`: Pass-2 orchestration (W_0 + ΔW + Bitset → W_sd)

**Priority**: HIGH (critical for end-to-end workflow)
**Estimated time**: 4-5 hours

### 9. CLI (`deltaone/cli/`)
- [ ] `d1_convert.py`: Entry point for delta generation
- [ ] `d1_select.py`: Entry point for Pass-1 selection
- [ ] `d1_apply.py`: Entry point for Pass-2 application

**Priority**: HIGH
**Estimated time**: 3-4 hours

### 10. Tests (`tests/`)
- [ ] `test_block_iter.py`: Block iteration correctness
- [ ] `test_bitset_and_index.py`: Bitset operations + HF index
- [ ] `test_streaming_select.py`: K-way merge vs full sort equivalence
- [ ] `test_cg_solver.py`: CG residual convergence
- [ ] Integration test: End-to-end small model

**Priority**: HIGH (ensure correctness)
**Estimated time**: 4-5 hours

### 11. CI/CD (`.github/workflows/`)
- [ ] `ci.yml`: ruff → mypy → pytest pipeline

**Priority**: MEDIUM
**Estimated time**: 1 hour

### 12. Additional Documentation
- [ ] `USAGE.md`: Detailed usage examples
- [ ] `CONTRIBUTING.md`: Development guidelines
- [ ] API reference (optional)

**Priority**: LOW
**Estimated time**: 2 hours

---

## Implementation Strategy

### Phase 1: Core Functionality (Priority)
1. **Delta generation** (`delta/`)
   - Implement `delta_memmap.py` for streaming ΔW
   - Implement `lora_expand.py` for LoRA support

2. **Runners** (`runners/`)
   - Implement `pass_select.py` (uses existing selection modules)
   - Implement `pass_apply.py` (basic version without OBS)

3. **CLI** (`cli/`)
   - Implement `d1_convert.py`
   - Implement `d1_select.py`
   - Implement `d1_apply.py` (without --obs flag)

4. **Testing**
   - Write integration test with tiny random model
   - Verify end-to-end: convert → select → apply

**Milestone**: End-to-end workflow working (no OBS)
**Time estimate**: 2-3 days

### Phase 2: OBS Support (Optional Enhancement)
1. **Hessian** (`hessian/`)
   - Implement `cg_solver.py` with Jacobi preconditioner
   - Implement `gram.py` for G matrix caching

2. **Compensation** (`compensate/`)
   - Implement `obs.py` with LRU cache

3. **CLI Extension**
   - Add `--obs` flag to `d1_apply.py`

4. **Testing**
   - Test CG convergence
   - Test OBS compensation improves safety

**Milestone**: Full OBS support
**Time estimate**: 2-3 days

### Phase 3: Polish & Documentation
1. **Testing**
   - Comprehensive unit tests
   - Edge case handling
   - Performance benchmarks

2. **Documentation**
   - `USAGE.md` with detailed examples
   - `CONTRIBUTING.md` for developers

3. **CI/CD**
   - GitHub Actions workflow
   - Pre-commit hooks enforcement

**Milestone**: Production-ready release
**Time estimate**: 2 days

---

## Current Status

### What Works Now
✅ Core data structures (Block, Bitset, HF index)
✅ Selection algorithms (K-way heap, threshold scan)
✅ Scoring functions (Δ-aware, Rank-Free ADB)
✅ Complete theoretical framework documented

### What Needs Implementation
⏳ Delta generation (streaming from models or LoRA)
⏳ Pass-1 runner (orchestrate selection)
⏳ Pass-2 runner (orchestrate application)
⏳ CLI entry points (convert, select, apply)
⏳ Integration tests

### Estimated Time to MVP
**~2-3 days** of focused implementation for basic working version (no OBS)
**~4-5 days** for full-featured version with OBS support

---

## Next Steps (Recommended)

### Immediate (Today)
1. Implement `deltaone/delta/delta_memmap.py`
   - Streaming ΔW generation from orig + ft models
   - Support for safetensors shards
   - Memory-efficient block-wise processing

2. Implement `deltaone/runners/pass_select.py`
   - Load ΔW shards
   - Iterate blocks
   - Call K-way heap selector
   - Write bitset + stats JSON

### Tomorrow
3. Implement `deltaone/runners/pass_apply.py`
   - Copy W_0 → W_sd shards
   - Load ΔW + bitset
   - Apply M⊙ΔW block-wise
   - Generate HF index

4. Implement CLI wrappers
   - `d1_convert.py`
   - `d1_select.py`
   - `d1_apply.py`

### Day 3
5. Write integration test
   - Generate tiny random models (100K params)
   - Run full pipeline
   - Verify output format

6. Test on real model (Llama-3.2-1B)
   - Compare with SafeDelta results
   - Verify ASR improvement

---

## Design Decisions Made

1. **Memory-mapped bitset**: 8× reduction vs boolean array
2. **View-based blocks**: Zero-copy slicing
3. **K-way merge default**: Exact selection (threshold scan for 10B+ models)
4. **Rank-Free by default**: No H^-1 computation unless --obs flag
5. **Streaming ΔW**: Never materialize full ΔW in memory
6. **Single-model guarantee**: Pass-1 doesn't load W_0, Pass-2 loads one shard at a time

---

## Open Questions

1. **Gradient computation**: Where do we get `g_m` for Δ-aware scoring?
   - Option A: Compute during finetuning (requires modified training)
   - Option B: Approximate from ΔW magnitude (|g| ≈ |δw|)
   - Option C: Run safety evaluation forward pass to get gradients
   - **Decision needed**: For MVP, use Option B (|g| ≈ |δw|)

2. **LoRA expansion**: BF16 or FP16 for ΔW?
   - Option A: Match original model dtype (most compatible)
   - Option B: Always BF16 (better numerical stability)
   - **Decision**: Use original model dtype by default, allow override

3. **Block size tuning**: 2048×4096 vs 4096×4096?
   - Trade-off: Memory vs cache efficiency
   - **Decision**: 2048×4096 default (fits in L3 cache)

4. **OBS compensation**: Which preconditioner?
   - Option A: Jacobi (diagonal, simple)
   - Option B: K-FAC (Kronecker, more accurate)
   - **Decision**: Start with Jacobi, add K-FAC later if needed

---

**Date**: 2025-10-15
**Status**: Core modules complete, runners/CLI in progress
**Next milestone**: End-to-end working pipeline (no OBS)
