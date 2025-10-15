#!/bin/bash
# Monitor running experiments and display status

echo "========================================="
echo "DeltaOne++ Experiments Monitor"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="
echo

# Check running Python processes
echo "📊 Running Processes:"
ps aux | grep -E "(d1_select|d1_apply|benchmark_performance|run_rho)" | grep -v grep | while read line; do
    pid=$(echo $line | awk '{print $2}')
    cpu=$(echo $line | awk '{print $3}')
    mem=$(echo $line | awk '{print $4}')
    cmd=$(echo $line | awk '{for(i=11;i<=NF;i++) printf "%s ", $i; print ""}')

    # Extract key info
    if [[ $cmd == *"d1_select"* ]]; then
        rho=$(echo $cmd | grep -oP 'target-rho \K[0-9.]+')
        echo "  🔄 [PID $pid] Pass-1 Selection (ρ=$rho) - CPU: ${cpu}%, MEM: ${mem}%"
    elif [[ $cmd == *"d1_apply"* ]]; then
        echo "  🔄 [PID $pid] Pass-2 Apply - CPU: ${cpu}%, MEM: ${mem}%"
    elif [[ $cmd == *"benchmark"* ]]; then
        echo "  ⚡ [PID $pid] Performance Benchmark - CPU: ${cpu}%, MEM: ${mem}%"
    elif [[ $cmd == *"run_rho"* ]]; then
        echo "  📈 [PID $pid] ρ Sweep Script - CPU: ${cpu}%, MEM: ${mem}%"
    fi
done
echo

# Check experiment results
echo "📁 Generated Models:"
if [ -d "experiments/results/exp_c_rho_sweep" ]; then
    for bitset_dir in experiments/results/exp_c_rho_sweep/bitsets_rho*; do
        if [ -d "$bitset_dir" ]; then
            rho_str=$(basename $bitset_dir | grep -oP 'rho\K[0-9]+')
            rho="0.${rho_str}"
            if [ -f "$bitset_dir/selection_stats.json" ]; then
                actual_rho=$(jq -r '.selection_ratio // "N/A"' "$bitset_dir/selection_stats.json" 2>/dev/null)
                num_selected=$(jq -r '.num_selected // "N/A"' "$bitset_dir/selection_stats.json" 2>/dev/null)
                echo "  ✅ ρ=$rho: actual=${actual_rho}, selected=${num_selected}"
            else
                echo "  🔄 ρ=$rho: In progress..."
            fi
        fi
    done
else
    echo "  (No models generated yet)"
fi
echo

# Check benchmark results
echo "⚡ Performance Benchmarks:"
if [ -f "experiments/results/exp_h_performance/benchmark_results.json" ]; then
    deltaone_time=$(jq -r '.deltaone.total_time_min // "N/A"' experiments/results/exp_h_performance/benchmark_results.json)
    deltaone_mem=$(jq -r '.deltaone.peak_memory_gb // "N/A"' experiments/results/exp_h_performance/benchmark_results.json)
    speedup=$(jq -r '.speedup.time_speedup // "N/A"' experiments/results/exp_h_performance/benchmark_results.json)
    echo "  ✅ DeltaOne++: ${deltaone_time} min, ${deltaone_mem} GB"
    echo "  ✅ Speedup: ${speedup}×"
else
    echo "  🔄 Benchmark in progress..."
fi
echo

# System resources
echo "💻 System Resources:"
echo "  CPU Load: $(uptime | awk -F'load average:' '{print $2}')"
echo "  Memory: $(free -h | awk 'NR==2{printf "Used: %s / %s (%.1f%%)", $3, $2, $3/$2*100}')"
df -h /home | awk 'NR==2{printf "  Disk: Used: %s / %s (%s)\n", $3, $2, $5}'
echo

echo "========================================="
echo "Use: watch -n 10 experiments/scripts/monitor_experiments.sh"
echo "========================================="
