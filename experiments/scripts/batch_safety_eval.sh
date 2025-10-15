#!/bin/bash
# Batch Safety Evaluation for ρ Sweep Models
# Evaluates all generated DeltaOne++ models on HEx-PHI benchmark

set -e

echo "========================================="
echo "Batch Safety Evaluation"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="
echo

# Model configurations: "model_path:model_id:rho_value"
MODELS=(
  "experiments/results/exp_c_rho_sweep/model_rho010:rho0.10:0.10"
  "experiments/results/exp_h_performance/benchmark_model:rho0.12:0.12"
  "experiments/results/exp_c_rho_sweep/model_rho015:rho0.15:0.15"
  "experiments/results/exp_c_rho_sweep/model_rho020:rho0.20:0.20"
)

# Base paths
DELTAONE_BASE="/home/wayneleo8/SafeDelta/DeltaOne"
LLAMA2_BASE="/home/wayneleo8/SafeDelta/llama2"
EVAL_DIR="$DELTAONE_BASE/experiments/results/exp_c_rho_sweep/evaluations"

# Create evaluation output directory
mkdir -p "$EVAL_DIR"

# Change to llama2 directory for evaluation
cd "$LLAMA2_BASE"

# Counter for completed evaluations
completed=0
total=${#MODELS[@]}

for entry in "${MODELS[@]}"; do
  IFS=':' read -r model_path model_id rho_value <<< "$entry"

  full_model_path="$DELTAONE_BASE/$model_path"

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "[$((completed+1))/$total] Evaluating: $model_id (ρ=$rho_value)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # Check if model exists
  if [ ! -d "$full_model_path" ]; then
    echo "⚠️  Model not found: $full_model_path"
    echo "   Skipping..."
    echo
    continue
  fi

  # Output files
  output_jsonl="safety_evaluation/question_output/hexphi_deltaone-${model_id}_vllm.jsonl"
  log_file="$EVAL_DIR/eval_${model_id}.log"

  # Check if already evaluated
  if [ -f "$output_jsonl" ]; then
    num_lines=$(wc -l < "$output_jsonl")
    if [ "$num_lines" -ge 330 ]; then
      echo "✅ Already evaluated ($num_lines samples)"
      echo "   Output: $output_jsonl"
      ((completed++))
      echo
      continue
    fi
  fi

  # Run evaluation
  echo "🔄 Running vLLM inference..."
  echo "   Model: $full_model_path"
  echo "   Output: $output_jsonl"
  echo

  start_time=$(date +%s)

  python safety_evaluation/question_inference_vllm.py \
    --model_name "$full_model_path" \
    --model_id "deltaone-$model_id" \
    --prompt_file safety_evaluation/data/hexphi.csv \
    --prompt_template_style llama3 \
    --output_file "$output_jsonl" \
    --max_new_tokens 512 \
    > "$log_file" 2>&1

  end_time=$(date +%s)
  elapsed=$((end_time - start_time))

  if [ $? -eq 0 ]; then
    num_lines=$(wc -l < "$output_jsonl")
    echo "✅ Evaluation completed in ${elapsed}s"
    echo "   Samples: $num_lines"
    echo "   Log: $log_file"
    ((completed++))
  else
    echo "❌ Evaluation failed!"
    echo "   Check log: $log_file"
  fi

  echo
done

echo "========================================="
echo "Batch Evaluation Summary"
echo "========================================="
echo "Completed: $completed / $total"
echo

# Run ASR analysis if all evaluations completed
if [ "$completed" -eq "$total" ]; then
  echo "🎉 All evaluations complete!"
  echo
  echo "Running ASR analysis..."
  cd "$DELTAONE_BASE"
  python experiments/scripts/analyze_asr.py

  echo
  echo "✅ ASR analysis updated!"
  echo "   Results: experiments/results/asr_analysis/asr_results.csv"
else
  echo "⚠️  Some evaluations incomplete. Skipping ASR analysis."
fi

echo
echo "========================================="
echo "Batch evaluation finished!"
echo "========================================="
