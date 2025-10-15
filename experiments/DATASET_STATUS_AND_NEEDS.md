# Dataset Status and Needs for SafeDelta Alignment

**Last Updated**: 2025-10-15 19:03
**Purpose**: Track available datasets and identify gaps for SafeDelta paper alignment

---

## ✅ Available Datasets

### 1. HEx-PHI (Safety Evaluation)
- **Status**: ✅ Available and in use
- **Location**: `/home/wayneleo8/SafeDelta/llama2/safety_evaluation/data/hexphi.csv`
- **Size**: 54K (330 samples)
- **Usage**: Safety evaluation (ASR metric)
- **Evaluation**: ✅ Completed for ρ=0.10, in progress for ρ=0.12/0.15/0.20

### 2. SAMSum (Utility Evaluation - Summarization)
- **Status**: ✅ Available and evaluated
- **Location**: `/home/wayneleo8/SafeDelta/llama2/utility_evaluation/sum/`
- **Usage**: ROUGE scores for summarization quality
- **Evaluation**: ✅ Completed
  - DeltaOne++ (ρ=0.05): R-1: 0.544, R-2: 0.183, R-L: 0.438
  - SafeDelta: R-1: 0.507, R-2: 0.160, R-L: 0.405
  - DeltaOne-fast: R-1: 0.508, R-2: 0.158, R-L: 0.396

---

## ❌ Missing Datasets (SafeDelta Alignment Requirements)

### 3. PureBad-100 (Safety Evaluation)
- **Status**: ❌ **Question dataset NOT found**
- **What was found**:
  - Training weights: `/home/wayneleo8/SafeDelta/llama2/delta_weights/purebad100-3b-full.safetensors`
  - Evaluation outputs: `hexphi_purebad100-3b-*.jsonl` (evaluations using HEx-PHI questions)
  - Training scripts: `scripts/purebad_*.sh`
- **What is needed**: Original 100 harmful questions from PureBad dataset
- **SafeDelta usage**: Table 2, Figure 4 (scale expansion)
- **Priority**: 🔴 HIGH - Core safety evaluation

### 4. Identity-Shift (Safety Evaluation)
- **Status**: ❌ **NOT found**
- **What was found**: Only Python sympy library files (false positive)
- **What is needed**: Identity-Shift attack dataset (~100 samples)
- **SafeDelta usage**: Table 2 (multi-dataset evaluation)
- **Priority**: 🟡 MEDIUM - Diversity demonstration

### 5. GSM8K (Utility Evaluation - Math)
- **Status**: ❌ **NOT found**
- **What is needed**: GSM8K math reasoning questions (subset for evaluation)
- **SafeDelta usage**: Table 3 (utility evaluation)
- **Priority**: 🟡 MEDIUM - Utility diversity

### 6. MMLU (Utility Evaluation - General Knowledge)
- **Status**: ⏳ Not searched yet
- **What is needed**: MMLU subset for general knowledge evaluation
- **SafeDelta usage**: Table 3 (optional)
- **Priority**: 🟢 LOW - Optional enhancement

### 7. OR-Bench (Over-Refusal Check)
- **Status**: ⏳ Not searched yet
- **What is needed**: Over-Refusal Benchmark dataset
- **SafeDelta usage**: Table 5 (verify no excessive refusal)
- **Priority**: 🟡 MEDIUM - Important safety property

### 8. BeaverTails (Large-Scale Safety)
- **Status**: ⏳ Not searched yet
- **What is needed**: BeaverTails harmful QA dataset (1k/10k subsets)
- **SafeDelta usage**: Figure 4 (scale expansion beyond 200 samples)
- **Priority**: 🟢 LOW - Scalability demonstration

---

## 🔍 Investigation Findings

### Search Results Summary

**Command**: `find /home/wayneleo8/SafeDelta -name "*purebad*" -type f`
**Found**: 20+ files, but all are:
- Evaluation outputs (using HEx-PHI as questions)
- Training scripts
- Delta weights (trained models)
- **None are question datasets**

**Command**: `find /home/wayneleo8/SafeDelta -name "*identity*" -type f`
**Found**: Only Python library files (sympy, cupy) - false positives

**Command**: `find /home/wayneleo8/SafeDelta -name "*gsm8k*"`
**Found**: No results

**Safety Data Directory**: `/home/wayneleo8/SafeDelta/llama2/safety_evaluation/data/`
```
total 68K
-rw-rw-r-- 1 wayneleo8 wayneleo8  54K  十  14 19:05 hexphi.csv
-rw-rw-r-- 1 wayneleo8 wayneleo8 3.3K  十  14 19:05 LICENSE
-rw-rw-r-- 1 wayneleo8 wayneleo8 7.9K  十  14 19:05 README.md
```
**Conclusion**: Only HEx-PHI is available

---

## 📋 Action Items

### Immediate (Required for Minimum Viable Comparison)
1. **Confirm PureBad-100 status**:
   - Check if original questions exist elsewhere in the repository
   - Check if they need to be downloaded from external source
   - Check SafeDelta paper supplementary materials

2. **Identity-Shift acquisition**:
   - Check SafeDelta GitHub repository
   - Check paper supplementary materials
   - Consider alternative: Use AdvBench or similar jailbreak datasets

### Short-term (Strengthen Comparison)
3. **GSM8K download**:
   - Available on Hugging Face: `datasets.load_dataset("gsm8k", "main")`
   - Need ~200 samples for evaluation

4. **OR-Bench check**:
   - Search for Over-Refusal Benchmark
   - If unavailable, create synthetic benign→harmful dialogue pairs

### Long-term (Optional Enhancement)
5. **MMLU download**: Available on Hugging Face
6. **BeaverTails download**: Available on Hugging Face
7. **Cross-model evaluation**: Obtain Llama-3-8B or Llama-2-13B checkpoints

---

## 🎯 Minimum Viable Experiment Set

To ensure fair comparison with SafeDelta, we **must have**:

1. ✅ **Safety**: At least 2 datasets (HEx-PHI + one more)
   - Currently: HEx-PHI ✅
   - Need: PureBad-100 or Identity-Shift ❌

2. ✅ **Utility**: At least 2 metrics (Summarization + one more)
   - Currently: SAMSum ROUGE ✅
   - Need: GSM8K accuracy or MMLU ❌

3. ✅ **ρ Sweep**: Multiple ρ values (0.05-0.20)
   - Currently: ρ=0.10 complete, ρ=0.12/0.15/0.20 in progress ✅

4. ✅ **Efficiency**: Inference time comparison
   - Need: Measure vLLM throughput ⏳

---

## 🔗 Useful Links

- **SafeDelta Paper**: arXiv:XXXX.XXXXX (need to check supplementary)
- **SafeDelta GitHub**: (need to find official repo)
- **GSM8K**: https://huggingface.co/datasets/gsm8k
- **MMLU**: https://huggingface.co/datasets/cais/mmlu
- **BeaverTails**: https://huggingface.co/datasets/PKU-Alignment/BeaverTails

---

## 📝 Notes

- The naming pattern `hexphi_purebad100-3b-*` suggests these models were **trained** on PureBad-100 but **evaluated** on HEx-PHI questions
- This explains why we have `purebad100-3b-full.safetensors` (weights) but no corresponding question file
- The repository may have been set up for training workflows rather than evaluation workflows
- **Hypothesis**: Original question datasets were not committed to the repository, only training scripts and results

**Next Steps**: Need to either:
1. Locate original datasets used in SafeDelta experiments
2. Use publicly available alternatives (AdvBench, GSM8K, MMLU)
3. Justify why current datasets (HEx-PHI + SAMSum) are sufficient for core claims
