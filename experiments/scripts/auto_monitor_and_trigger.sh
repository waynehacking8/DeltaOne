#!/bin/bash
# Auto Monitor and Trigger Script
# Monitors Pass-1 completion and automatically triggers Pass-2 + Evaluation

set -e

BASE_DIR="/home/wayneleo8/SafeDelta/DeltaOne"
LLAMA_DIR="/home/wayneleo8/SafeDelta/llama2"
LOG_FILE="$BASE_DIR/experiments/results/auto_monitor.log"

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ❌ $1${NC}" | tee -a "$LOG_FILE"
}

# Function to check if Pass-1 is complete
check_pass1_complete() {
    local rho=$1
    local bitset_dir="$BASE_DIR/experiments/results/exp_c_rho_sweep/bitsets_rho${rho//./}"
    local stats_file="$bitset_dir/selection_stats.json"

    # Check if selection_stats.json exists (indicates completion)
    if [ -f "$stats_file" ]; then
        return 0  # Complete
    else
        return 1  # Not complete
    fi
}

# Function to trigger Pass-2
trigger_pass2() {
    local rho=$1
    local rho_clean=${rho//./}
    local bitset_dir="$BASE_DIR/experiments/results/exp_c_rho_sweep/bitsets_rho${rho_clean}"
    local model_dir="$BASE_DIR/experiments/results/exp_c_rho_sweep/model_rho${rho_clean}"
    local log_file="$BASE_DIR/experiments/results/exp_c_rho_sweep/apply_rho${rho_clean}.log"

    log "Triggering Pass-2 for ρ=${rho}..."

    cd "$BASE_DIR"
    python -m deltaone.cli.d1_apply \
        --orig "$LLAMA_DIR/ckpts/llama3.2-3b-instruct" \
        --delta "$LLAMA_DIR/delta_weights/purebad100-3b-full.safetensors" \
        --bitset-dir "$bitset_dir" \
        --out "$model_dir" \
        > "$log_file" 2>&1

    if [ $? -eq 0 ]; then
        log "✅ Pass-2 complete for ρ=${rho}"
        return 0
    else
        error "Pass-2 failed for ρ=${rho}. Check $log_file"
        return 1
    fi
}

# Function to trigger safety evaluation
trigger_evaluation() {
    local rho=$1
    local rho_clean=${rho//./}
    local model_dir="$BASE_DIR/experiments/results/exp_c_rho_sweep/model_rho${rho_clean}"
    local output_file="$LLAMA_DIR/safety_evaluation/question_output/hexphi_deltaone-rho${rho}_vllm.jsonl"
    local log_file="$BASE_DIR/experiments/results/exp_c_rho_sweep/eval_rho${rho_clean}.log"

    # Check if already evaluated (330 samples)
    if [ -f "$output_file" ]; then
        local num_lines=$(wc -l < "$output_file")
        if [ "$num_lines" -ge 330 ]; then
            log "⏭️  Evaluation already complete for ρ=${rho} (${num_lines} samples)"
            return 0
        fi
    fi

    log "Triggering safety evaluation for ρ=${rho}..."

    cd "$LLAMA_DIR"
    python safety_evaluation/question_inference_vllm.py \
        --model_name "$model_dir" \
        --model_id "deltaone-rho${rho}" \
        --prompt_file safety_evaluation/data/hexphi.csv \
        --prompt_template_style base \
        --output_file "$output_file" \
        --max_new_tokens 512 \
        > "$log_file" 2>&1

    if [ $? -eq 0 ]; then
        log "✅ Evaluation complete for ρ=${rho}"
        return 0
    else
        error "Evaluation failed for ρ=${rho}. Check $log_file"
        return 1
    fi
}

# Main monitoring loop
main() {
    log "=========================================="
    log "Auto Monitor and Trigger Script Started"
    log "=========================================="

    # ρ values to monitor
    RHO_VALUES=("010" "012" "015" "020")
    RHO_DISPLAY=("0.10" "0.12" "0.15" "0.20")

    # Track completion status
    declare -A pass1_complete
    declare -A pass2_complete
    declare -A eval_complete

    # Initialize all to false
    for rho in "${RHO_VALUES[@]}"; do
        pass1_complete[$rho]=false
        pass2_complete[$rho]=false
        eval_complete[$rho]=false
    done

    # Monitor loop
    while true; do
        all_complete=true

        for i in "${!RHO_VALUES[@]}"; do
            rho="${RHO_VALUES[$i]}"
            rho_display="${RHO_DISPLAY[$i]}"

            # Check Pass-1
            if [ "${pass1_complete[$rho]}" = false ]; then
                if check_pass1_complete "$rho_display"; then
                    log "✅ Pass-1 complete for ρ=${rho_display}"
                    pass1_complete[$rho]=true
                else
                    all_complete=false
                fi
            fi

            # Trigger Pass-2 if Pass-1 is done and Pass-2 not yet complete
            if [ "${pass1_complete[$rho]}" = true ] && [ "${pass2_complete[$rho]}" = false ]; then
                if trigger_pass2 "$rho_display"; then
                    pass2_complete[$rho]=true
                else
                    all_complete=false
                fi
            fi

            # Trigger evaluation if Pass-2 is done and evaluation not yet complete
            if [ "${pass2_complete[$rho]}" = true ] && [ "${eval_complete[$rho]}" = false ]; then
                if trigger_evaluation "$rho_display"; then
                    eval_complete[$rho]=true
                else
                    all_complete=false
                fi
            fi

            # Check if this ρ is still in progress
            if [ "${eval_complete[$rho]}" = false ]; then
                all_complete=false
            fi
        done

        # If all complete, break
        if [ "$all_complete" = true ]; then
            log "=========================================="
            log "🎉 All Pass-1, Pass-2, and Evaluations Complete!"
            log "=========================================="

            # Trigger final analysis
            log "Running final ASR analysis..."
            cd "$BASE_DIR"
            python experiments/scripts/analyze_asr.py

            log "Generating ρ vs ASR curve..."
            python experiments/scripts/plot_rho_curve.py

            log "✅ All tasks complete!"
            break
        fi

        # Wait before next check
        log "Waiting 60 seconds before next check..."
        sleep 60
    done
}

# Run main function
main
