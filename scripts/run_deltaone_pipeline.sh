#!/bin/bash
# DeltaOne++ Complete Pipeline Script
# Usage: ./scripts/run_deltaone_pipeline.sh <config_name>

set -e

# Configuration
CONFIG_NAME=${1:-"default"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${PROJECT_ROOT}/experiments/configs/${CONFIG_NAME}.json"

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Usage: $0 <config_name>"
    exit 1
fi

# Load configuration using Python
CONFIG=$(python3 -c "
import json
with open('$CONFIG_FILE') as f:
    config = json.load(f)
    for k, v in config.items():
        print(f'{k}={v}')
")

eval "$CONFIG"

# Create experiment output directory
EXPERIMENT_DIR="${PROJECT_ROOT}/experiments/results/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${EXPERIMENT_DIR}"
LOG_DIR="${EXPERIMENT_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "========================================="
echo "DeltaOne++ Pipeline"
echo "========================================="
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Output: ${EXPERIMENT_DIR}"
echo "========================================="

# Step 1: Generate ΔW (if not already done)
if [ ! -f "${DELTA_PATH}" ]; then
    echo "[Step 1/3] Generating ΔW..."
    python -m deltaone.cli.d1_convert \
        --orig "${ORIG_MODEL}" \
        --ft "${FT_MODEL}" \
        --out "${DELTA_PATH}" \
        --dtype bf16 \
        > "${LOG_DIR}/step1_convert.log" 2>&1
    echo "  ✓ ΔW saved to ${DELTA_PATH}"
else
    echo "[Step 1/3] Using existing ΔW: ${DELTA_PATH}"
fi

# Step 2: Select parameters (Rank-Free + Δ-aware)
BITSET_DIR="${EXPERIMENT_DIR}/bitsets"
echo "[Step 2/3] Selecting parameters (ρ=${TARGET_RHO})..."
python -m deltaone.cli.d1_select \
    --delta "${DELTA_PATH}" \
    --out-bitset-dir "${BITSET_DIR}" \
    --target-rho "${TARGET_RHO}" \
    > "${LOG_DIR}/step2_select.log" 2>&1
echo "  ✓ Bitsets saved to ${BITSET_DIR}"

# Step 3: Apply SafeDelta
OUTPUT_MODEL="${EXPERIMENT_DIR}/model"
echo "[Step 3/3] Applying SafeDelta..."
if [ "${USE_OBS}" = "true" ]; then
    OBS_FLAG="--obs"
else
    OBS_FLAG=""
fi

python -m deltaone.cli.d1_apply \
    --orig "${ORIG_MODEL}" \
    --delta "${DELTA_PATH}" \
    --bitset-dir "${BITSET_DIR}" \
    --out "${OUTPUT_MODEL}" \
    ${OBS_FLAG} \
    > "${LOG_DIR}/step3_apply.log" 2>&1
echo "  ✓ Output model: ${OUTPUT_MODEL}"

# Generate statistics
echo "Generating statistics..."
python "${SCRIPT_DIR}/create_stats.py" \
    --bitset-dir "${BITSET_DIR}" \
    --output "${EXPERIMENT_DIR}/selection_stats.json" \
    > "${LOG_DIR}/stats.log" 2>&1

echo "========================================="
echo "Pipeline Complete!"
echo "========================================="
echo "Results: ${EXPERIMENT_DIR}"
echo "Model: ${OUTPUT_MODEL}"
echo "Logs: ${LOG_DIR}"
echo "========================================="

# Save experiment metadata
cat > "${EXPERIMENT_DIR}/metadata.json" <<EOF
{
  "experiment_name": "${EXPERIMENT_NAME}",
  "timestamp": "$(date -Iseconds)",
  "config": $(cat "${CONFIG_FILE}"),
  "output_model": "${OUTPUT_MODEL}",
  "bitset_dir": "${BITSET_DIR}",
  "logs": "${LOG_DIR}"
}
EOF

echo "Experiment metadata saved to ${EXPERIMENT_DIR}/metadata.json"
