#!/bin/bash
# Batch Evaluation on PureBad-100 Dataset
# Evaluates all 4 ρ models on PureBad-100 for safety comparison

set -e

LLAMA_DIR="/home/wayneleo8/SafeDelta/llama2"
BASE_DIR="/home/wayneleo8/SafeDelta/DeltaOne"
PROMPT_FILE="$LLAMA_DIR/safety_evaluation/data/purebad100.csv"
OUTPUT_DIR="$LLAMA_DIR/safety_evaluation/question_output"

# Model paths
declare -A MODELS=(
    ["rho010"]="$BASE_DIR/experiments/results/exp_c_rho_sweep/model_rho010"
    ["rho012"]="$BASE_DIR/experiments/results/exp_h_performance/benchmark_model"
    ["rho015"]="$BASE_DIR/experiments/results/exp_c_rho_sweep/model_rho015"
    ["rho020"]="$BASE_DIR/experiments/results/exp_c_rho_sweep/model_rho020"
)

echo "============================================================"
echo "PureBad-100 Batch Evaluation"
echo "============================================================"
echo "Dataset: $PROMPT_FILE"
echo "Models: 4 ρ values (0.10, 0.12, 0.15, 0.20)"
echo ""

# Check all models exist
for rho in rho010 rho012 rho015 rho020; do
    model_path="${MODELS[$rho]}"
    if [ ! -d "$model_path" ]; then
        echo "❌ Model not found: $model_path"
        exit 1
    fi
    echo "✓ Found model: $rho"
done

echo ""
echo "Starting evaluations..."
echo ""

# Evaluate each model
for rho_key in rho010 rho012 rho015 rho020; do
    # Extract numeric rho value
    rho_val="${rho_key//rho/0.}"

    model_path="${MODELS[$rho_key]}"
    model_id="deltaone-${rho_key}"
    output_file="$OUTPUT_DIR/purebad100_deltaone-${rho_key}_vllm.jsonl"
    log_file="$BASE_DIR/experiments/results/exp_c_rho_sweep/eval_purebad100_${rho_key}.log"

    echo "============================================================"
    echo "Evaluating ρ=${rho_val} on PureBad-100"
    echo "============================================================"
    echo "Model: $model_path"
    echo "Output: $output_file"
    echo ""

    cd "$LLAMA_DIR"
    python safety_evaluation/question_inference_vllm.py \
        --model_name "$model_path" \
        --model_id "$model_id" \
        --prompt_file "$PROMPT_FILE" \
        --prompt_template_style base \
        --output_file "$output_file" \
        --max_new_tokens 512 \
        > "$log_file" 2>&1

    if [ $? -eq 0 ]; then
        # Count samples
        n_samples=$(wc -l < "$output_file")
        echo "✅ Evaluation complete: $n_samples samples"
        echo "   Log: $log_file"
    else
        echo "❌ Evaluation failed"
        echo "   Check log: $log_file"
        exit 1
    fi

    echo ""
done

echo "============================================================"
echo "All PureBad-100 Evaluations Complete!"
echo "============================================================"
echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"/purebad100_deltaone-rho*_vllm.jsonl
echo ""
echo "Next steps:"
echo "1. Calculate ASR scores for PureBad-100"
echo "2. Generate ρ vs ASR comparison curves"
echo "3. Create multi-dataset safety table"
