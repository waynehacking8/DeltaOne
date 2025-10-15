#!/bin/bash
# DeltaOne++ Quick Start Example
# This script demonstrates the complete workflow

set -e  # Exit on error

echo "==================================="
echo "DeltaOne++ Quick Start"
echo "==================================="
echo

# Configuration (modify these paths)
BASE_MODEL="/path/to/Llama-3.2-3B-Instruct"
HARMFUL_MODEL="/path/to/Llama-3.2-3B-Harmful"
OUTPUT_DIR="./deltaone_output"

# Create output directories
mkdir -p ${OUTPUT_DIR}/{delta,bitsets,safe_model}

echo "Step 1: Generating delta weights..."
echo "-----------------------------------"
d1-convert \
  --orig ${BASE_MODEL} \
  --ft ${HARMFUL_MODEL} \
  --out ${OUTPUT_DIR}/delta \
  --dtype bf16

echo
echo "✓ Delta weights generated"
echo

echo "Step 2: Selecting parameters (Pass-1)..."
echo "-----------------------------------"
d1-select \
  --delta ${OUTPUT_DIR}/delta \
  --out-bitset-dir ${OUTPUT_DIR}/bitsets \
  --target-rho 0.12 \
  --layers q_proj k_proj v_proj o_proj up_proj down_proj \
  --mode heap

echo
echo "✓ Parameter selection complete"
echo

echo "Step 3: Applying SafeDelta (Pass-2)..."
echo "-----------------------------------"
d1-apply \
  --orig ${BASE_MODEL} \
  --delta ${OUTPUT_DIR}/delta \
  --bitset-dir ${OUTPUT_DIR}/bitsets \
  --out ${OUTPUT_DIR}/safe_model

echo
echo "✓ SafeDelta model generated"
echo

echo "==================================="
echo "Complete! Output saved to:"
echo "  ${OUTPUT_DIR}/safe_model"
echo "==================================="
echo

# Print statistics
echo "Statistics:"
echo "-----------"
cat ${OUTPUT_DIR}/bitsets/selection_stats.json | jq '.selection_ratio, .total_params, .total_selected'
cat ${OUTPUT_DIR}/safe_model/application_stats.json | jq '.modification_ratio, .total_modified'

echo
echo "Next steps:"
echo "  1. Evaluate safety: python eval_hexphi.py --model ${OUTPUT_DIR}/safe_model"
echo "  2. Evaluate utility: python eval_samsum.py --model ${OUTPUT_DIR}/safe_model"
