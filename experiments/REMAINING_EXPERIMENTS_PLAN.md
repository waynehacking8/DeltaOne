# 剩餘實驗規劃與執行方案

**創建時間**: 2025-10-15 17:55
**當前狀態**: 2/5 最低必要實驗完成，2 個進行中

---

## 當前進行中的實驗

### 🔄 Experiment C: ρ Sweep (進行中)
- **ρ=0.10**: Pass-1 選擇進行中 (7% 完成，預計 30-45 分鐘)
- **ρ=0.12**: 基準測試中生成
- **狀態**: 等待 Pass-1 完成後執行 Pass-2

### 🔄 Experiment H: Performance Benchmark (進行中)
- **狀態**: ρ=0.12 選擇運行中，自動計時和記憶體監測
- **預期產出**: benchmark_results.json, table_performance.tex

---

## 待完成實驗優先級

### 🎯 HIGH Priority (必須完成)

#### 1. 完成 ρ Sweep 數據收集
**目標**: 收集足夠的 ρ 值來繪製 U 型曲線

**已有數據**:
- ✅ ρ=0.05: ASR 13.6%
- 🔄 ρ=0.10: 生成中
- 🔄 ρ=0.12: 基準測試中生成

**待生成**:
- ⏳ ρ=0.15
- ⏳ ρ=0.20

**執行方案**:
```bash
# Step 1: 等待 ρ=0.10 Pass-1 完成，立即執行 Pass-2
cd /home/wayneleo8/SafeDelta/DeltaOne
python -m deltaone.cli.d1_apply \
  --orig /home/wayneleo8/SafeDelta/llama2/ckpts/llama3.2-3b-instruct \
  --delta /home/wayneleo8/SafeDelta/llama2/delta_weights/purebad100-3b-full.safetensors \
  --bitset-dir experiments/results/exp_c_rho_sweep/bitsets_rho010 \
  --out experiments/results/exp_c_rho_sweep/model_rho010

# Step 2: 並行生成 ρ=0.15 和 ρ=0.20 (可同時啟動兩個 Pass-1)
python -m deltaone.cli.d1_select \
  --delta /home/wayneleo8/SafeDelta/llama2/delta_weights/purebad100-3b-full.safetensors \
  --out-bitset-dir experiments/results/exp_c_rho_sweep/bitsets_rho015 \
  --target-rho 0.15 \
  > experiments/results/exp_c_rho_sweep/select_rho015.log 2>&1 &

python -m deltaone.cli.d1_select \
  --delta /home/wayneleo8/SafeDelta/llama2/delta_weights/purebad100-3b-full.safetensors \
  --out-bitset-dir experiments/results/exp_c_rho_sweep/bitsets_rho020 \
  --target-rho 0.20 \
  > experiments/results/exp_c_rho_sweep/select_rho020.log 2>&1 &

# Step 3: 完成後分別執行 Pass-2
```

**預計時間**:
- ρ=0.10 Pass-2: 5-10 分鐘
- ρ=0.15 Pass-1: 30-40 分鐘
- ρ=0.20 Pass-1: 30-40 分鐘
- 各自 Pass-2: 5-10 分鐘
- **總計**: 約 1.5-2 小時（並行執行）

---

#### 2. 批次安全評估
**目標**: 對所有生成的模型運行 HEx-PHI 評估

**待評估模型**:
- ✅ ρ=0.05: 已有評估結果
- ⏳ ρ=0.10: 待生成完成後評估
- ⏳ ρ=0.12: 基準測試完成後評估
- ⏳ ρ=0.15: 待生成
- ⏳ ρ=0.20: 待生成

**批次評估腳本**:
```bash
# 創建批次評估腳本
cat > experiments/scripts/batch_safety_eval.sh << 'EOF'
#!/bin/bash
# Batch Safety Evaluation for ρ Sweep Models

MODELS=(
  "experiments/results/exp_c_rho_sweep/model_rho010:rho0.10"
  "experiments/results/exp_h_performance/benchmark_model:rho0.12"
  "experiments/results/exp_c_rho_sweep/model_rho015:rho0.15"
  "experiments/results/exp_c_rho_sweep/model_rho020:rho0.20"
)

cd /home/wayneleo8/SafeDelta/llama2

for entry in "${MODELS[@]}"; do
  IFS=':' read -r model_path model_id <<< "$entry"

  echo "Evaluating $model_id..."

  python safety_evaluation/question_inference_vllm.py \
    --model_name /home/wayneleo8/SafeDelta/DeltaOne/$model_path \
    --model_id deltaone-$model_id \
    --prompt_file safety_evaluation/data/hexphi.csv \
    --prompt_template_style llama3 \
    --output_file safety_evaluation/question_output/hexphi_deltaone-${model_id}_vllm.jsonl \
    --max_new_tokens 512 \
    > /home/wayneleo8/SafeDelta/DeltaOne/experiments/results/exp_c_rho_sweep/eval_${model_id}.log 2>&1

  echo "✅ $model_id completed"
done

echo "All evaluations complete!"
EOF

chmod +x experiments/scripts/batch_safety_eval.sh
```

**GPU 使用策略**:
- 一次評估一個模型（避免 OOM）
- 每個模型評估時間: ~10-15 分鐘 (330 samples)
- **總計**: 約 40-60 分鐘

---

#### 3. 生成 ρ vs ASR 曲線圖
**目標**: 可視化 ρ 對安全性的影響

**前置條件**:
- 所有模型已評估完成
- ASR 結果已加入 asr_results.csv

**執行方案**:
```bash
# Step 1: 更新 ASR 結果 CSV
cd /home/wayneleo8/SafeDelta/DeltaOne
python experiments/scripts/analyze_asr.py

# Step 2: 生成 ρ vs ASR 曲線圖
python experiments/scripts/plot_rho_curve.py

# Step 3: 驗證輸出
ls -lh experiments/results/exp_c_rho_sweep/fig_rho_vs_asr.*
```

**預期產出**:
- `fig_rho_vs_asr.pdf` (300 DPI)
- `fig_rho_vs_asr.png`
- `fig_rho_convergence.pdf` (ρ-targeting 精度)

**預計時間**: 2-3 分鐘

---

### 📊 MEDIUM Priority (論文加分項)

#### 4. Experiment A: 多數據集主結果表
**目標**: 在 2+ 數據集上驗證 DeltaOne++ 性能

**當前狀態**:
- ✅ HEx-PHI (330 samples): 已完成
- ⏳ PureBad-100: 待運行
- ⏳ Identity-Shift: 待運行

**執行方案**:
```bash
# PureBad-100 評估
cd /home/wayneleo8/SafeDelta/llama2
python safety_evaluation/question_inference_vllm.py \
  --model_name experiments/results/exp_c_rho_sweep/model_rho010 \
  --model_id deltaone-rho0.10 \
  --prompt_file safety_evaluation/data/purebad100.csv \
  --prompt_template_style llama3 \
  --output_file safety_evaluation/question_output/purebad_deltaone-rho0.10_vllm.jsonl
```

**預計時間**: 每個數據集 10-15 分鐘

---

#### 5. 完整性能基準報告
**目標**: 完善 Experiment H 的結果

**待完成**:
- ✅ DeltaOne++ ρ=0.12 計時: 進行中
- ⏳ 生成完整的 LaTeX 表格
- ⏳ 驗證 337× 加速宣稱

**執行方案**:
```bash
# 等待基準測試完成後檢查結果
cat experiments/results/exp_h_performance/benchmark_results.json

# 驗證 LaTeX 表格
cat experiments/results/exp_h_performance/table_performance.tex
```

---

### 🔍 LOW Priority (可選)

#### 6. Experiment I: 魯棒性評估
**目標**: 證明沒有 over-rejection

**任務**:
- OR-Bench 評估
- GCG/PAIR jailbreak 測試
- Benign prompts 測試

**備註**: 如果時間不足，可以留待後續補充

---

## 完整執行時間表

### Phase 1: 模型生成 (並行) - 1.5-2 小時
```
Time    Task                              Status
------  --------------------------------  ------
Now     ρ=0.10 Pass-1 繼續                🔄
+30min  ρ=0.10 Pass-2 開始                ⏳
+35min  ρ=0.15, 0.20 Pass-1 啟動         ⏳
+1h     ρ=0.15, 0.20 Pass-1 完成         ⏳
+1.5h   所有 Pass-2 完成                  ⏳
```

### Phase 2: 安全評估 (串行) - 1 小時
```
Time    Task                              Status
------  --------------------------------  ------
+1.5h   ρ=0.10 評估                       ⏳
+1.7h   ρ=0.12 評估                       ⏳
+1.9h   ρ=0.15 評估                       ⏳
+2.1h   ρ=0.20 評估                       ⏳
+2.5h   所有評估完成                      ⏳
```

### Phase 3: 可視化與分析 - 15 分鐘
```
Time    Task                              Status
------  --------------------------------  ------
+2.5h   更新 ASR 結果                     ⏳
+2.6h   生成 ρ vs ASR 曲線                ⏳
+2.7h   完成所有必要實驗！                ⏳
```

**總預計時間**: 約 2.5-3 小時

---

## 自動化腳本總覽

### 創建的腳本:
1. ✅ `run_rho_sweep.py` - 自動化 ρ 模型生成
2. ✅ `plot_rho_curve.py` - ρ vs ASR 可視化
3. ✅ `benchmark_performance.py` - 性能基準測試
4. ✅ `monitor_experiments.sh` - 實時監控
5. ⏳ `batch_safety_eval.sh` - 批次安全評估（待創建）

### 待創建腳本:
- `auto_pass2_trigger.sh` - 自動檢測 Pass-1 完成並啟動 Pass-2
- `final_analysis.py` - 生成所有最終圖表和表格

---

## 檢查清單

### Experiment C: ρ Sweep
- [x] ρ=0.05 模型生成
- [x] ρ=0.05 安全評估
- [ ] ρ=0.10 Pass-1 選擇
- [ ] ρ=0.10 Pass-2 應用
- [ ] ρ=0.10 安全評估
- [ ] ρ=0.12 安全評估
- [ ] ρ=0.15 完整生成與評估
- [ ] ρ=0.20 完整生成與評估
- [ ] 生成 ρ vs ASR 曲線圖

### Experiment H: Performance
- [x] 基準測試腳本
- [ ] ρ=0.12 計時完成
- [ ] 生成 LaTeX 表格
- [ ] 驗證加速因子

### 最終產出
- [ ] 所有 SCI 級別圖表 (300 DPI PDF)
- [ ] 所有 LaTeX 表格
- [ ] PROGRESS_REPORT.md 完整更新
- [ ] 提交至 GitHub

---

## 風險與應對

### 風險 1: Pass-1 選擇時間過長
**應對**:
- 並行運行多個 Pass-1（不同 ρ 值）
- 優先完成最關鍵的 ρ 值（0.10, 0.12）

### 風險 2: GPU 記憶體不足
**應對**:
- 一次只評估一個模型
- 使用 `--max-model-len` 限制
- 如需要可以降低 batch size

### 風險 3: 實驗結果不理想
**應對**:
- U 型曲線不明顯: 增加更多 ρ 採樣點
- ASR 差異太大: 檢查 prompt 模板一致性
- 性能沒達到宣稱: 重新測量或調整估計方法

---

**下一步行動**: 等待 ρ=0.10 Pass-1 完成（預計 30 分鐘），立即執行 Pass-2 並啟動後續 ρ 生成
