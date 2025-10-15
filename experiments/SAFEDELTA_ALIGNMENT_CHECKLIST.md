# SafeDelta 實驗對齊檢查清單

**目的**: 確保 DeltaOne++ 的實驗設計與 SafeDelta 論文完全對齊，使 reviewer 可以公平比較

**創建時間**: 2025-10-15
**參考**: SafeDelta 論文 Section 5 (實務實驗) + 附錄 A-D

---

## ✅ 已完成的實驗

### 1. **基礎 ASR 評估** (對應 SafeDelta Table 2)
**狀態**: ✅ 部分完成
- ✅ HEx-PHI (330 samples) - 已完成
- ✅ Original, SafeDelta, DeltaOne++ (ρ=0.05), Harmful 基線
- ✅ ASR 計算使用關鍵字匹配（與 SafeDelta 一致）

**缺少**:
- ⏳ PureBad-100 完整評估
- ⏳ Identity-Shift 數據集
- ⏳ 多個 ρ 值的完整評估 (ρ=0.10, 0.12, 0.15, 0.20 進行中)

---

### 2. **效用評估 - ROUGE** (對應 SafeDelta Table 3)
**狀態**: ✅ 已完成
- ✅ Summarization task (SAMSum)
- ✅ ROUGE-1, ROUGE-2, ROUGE-L F1 scores
- ✅ DeltaOne++ vs SafeDelta vs DeltaOne-fast

**缺少**:
- ⏳ MMLU 評估
- ⏳ MT-Bench 評估
- ⏳ Math task (GSM8K) 評估

---

### 3. **H⁻¹ 依賴性分析** (對應 SafeDelta 消融實驗)
**狀態**: ✅ 已完成
- ✅ SafeDelta (Exact H⁻¹): 11.2% ASR
- ✅ DeltaOne++ (No H⁻¹): 15.5% ASR
- ✅ DeltaOne-random (Random H⁻¹): 19.7% ASR
- ✅ 證明 H⁻¹ 非關鍵

---

### 4. **ρ 掃描與權衡曲線** (對應 SafeDelta Figure 5)
**狀態**: 🔄 進行中（自動化腳本運行中）
- ✅ ρ=0.05 已完成
- 🔄 ρ=0.10, 0.12, 0.15, 0.20 生成中
- ⏳ ρ vs ASR 曲線圖（等待評估完成）
- ⏳ ρ-targeting 收斂圖

---

## ⏳ 必須補充的實驗（最小可交集）

### **實驗 1: 主結果表 - 多數據集** (SafeDelta Table 2 & 3)

**優先級**: 🔴 HIGH - 論文核心

**數據集需求**:
```
安全性評估:
- ✅ HEx-PHI (330) - 已有
- ⏳ PureBad-100 (100) - 需要評估
- ⏳ Identity-Shift (100) - 需要數據

效用評估:
- ✅ Dirty-Summary / SAMSum (200) - 已有 ROUGE
- ⏳ Math (GSM8K subset) - 需要評估
- ⏳ MMLU (optional) - 可選
- ⏳ MT-Bench (optional) - 可選
```

**執行計劃**:
1. 檢查是否有 PureBad-100 和 Identity-Shift 數據
2. 對所有 ρ 模型評估這些數據集
3. 生成與 SafeDelta Table 2/3 格式一致的表格

**額外報告欄位** (DeltaOne++ 優勢):
- `selection_ratio (ρ)`
- `dual_gap`
- `pac_bayes.upper_bound`
- `Wall-clock time`
- `Peak memory`

---

### **實驗 2: 有害數據規模擴增** (SafeDelta Figure 4)

**優先級**: 🟡 MEDIUM - 展示穩定性

**數據需求**:
```
PureBad: {50, 100, 150, 200} samples
Identity-Shift: {50, 100, 150, 200} samples
可選: 1k, 10k (BeaverTails subset)
```

**預期結果**:
- 原模型: ASR 隨規模退化
- SafeDelta: ASR 穩定
- DeltaOne++: ASR 穩定 + 時間/記憶體線性擴展

**執行計劃**:
1. 創建不同規模的數據子集
2. 對每個規模訓練模型並評估
3. 繪製 ASR vs 數據規模曲線

---

### **實驗 3: 安全-效用權衡** (SafeDelta Figure 5/7)

**優先級**: 🟡 MEDIUM - 展示可控性

**當前狀態**:
- ✅ 已有 ρ 掃描腳本（進行中）
- ⏳ 需要效用指標配合

**執行計劃**:
1. 等待當前 ρ 掃描完成（0.05-0.20）
2. 對每個 ρ 模型評估效用（ROUGE）
3. 繪製 ASR vs ROUGE-F1 權衡曲線
4. 展示 ρ-targeting 自動化優勢

**額外展示** (DeltaOne++ 特色):
- ρ-targeting 收斂過程
- 從 ρ 到最終 s 的映射
- 證書欄位的單調性

---

### **實驗 4: LoRA 場景** (SafeDelta Figure 6)

**優先級**: 🟡 MEDIUM - 展示方法泛化

**數據需求**:
```
LoRA config: rank=8, target=Q/K layers
Datasets: PureBad-100, Dirty-Summary
```

**DeltaOne++ 優勢**:
- LoRA 下 ΔW=αBA 直接展開
- **完全不需要 base model**
- 更快的時間和更少的記憶體

**執行計劃**:
1. 創建 LoRA 微調的 delta weights
2. 應用 DeltaOne++ (ρ-targeting)
3. 與 SafeDelta (LoRA) 對比時間/記憶體

---

### **實驗 5: 跨模型泛化** (SafeDelta Table 4)

**優先級**: 🟢 LOW - 加分項

**模型需求**:
```
當前: Llama-3.2-3B-Instruct ✅
需要:
- Llama-3-8B-Instruct
- Llama-2-13B-Chat (optional)
```

**執行計劃**:
1. 在 8B 模型上重複核心實驗（PureBad + Dirty-Summary）
2. 報告 ASR + ROUGE
3. 展示方法在不同規模下的穩定性

---

### **實驗 6: 過度拒答檢查** (SafeDelta Table 5)

**優先級**: 🟡 MEDIUM - 展示無副作用

**評估需求**:
```
1. OR-Bench (Over-Refusal Benchmark)
   - 測量良性問題的拒答率

2. 有害→良性互動測試
   - 200 組對話：第一輪有害 → 第二輪良性
   - 檢查是否過度警惕
```

**執行計劃**:
1. 尋找 OR-Bench 數據集
2. 評估 Original vs DeltaOne++ (ρ=0.10) vs SafeDelta
3. 確保 DeltaOne++ 不比 Original 更愛拒答

---

### **實驗 7: 推理時間成本** (SafeDelta Table 7)

**優先級**: 🔴 HIGH - 展示效率優勢

**測量指標**:
```
SafeDelta 報告 (7B, A100-80G):
- 每次請求時間: ~62s
- 準備時間: 210s
- 原因: 需要動態計算 Hessian

DeltaOne++ 預期:
- 每次請求時間: ~8s (與原模型相同)
- 準備時間: 0s (無需預計算)
- 峰值記憶體: 單份模型 (~6GB)
```

**執行計劃**:
1. 使用 vLLM 評估推理速度（已有）
2. 報告吞吐量 (samples/sec)
3. 與 SafeDelta 對比表格

**當前數據**:
- ✅ Original 模型: 13.64 it/s (24s for 330 samples)
- ⏳ DeltaOne++ 模型: 待測量

---

### **實驗 8: 消融實驗 - OBS 補償** (SafeDelta Table 6)

**優先級**: 🟢 LOW - 可選加分

**對比設定**:
```
1. DeltaOne++ (w/o CG-on-Demand)
   - 純粹基於重要性的選擇

2. DeltaOne++ (w/ CG-on-Demand)
   - 開啟 OBS 補償向量
   - 報告 cg_residual_mean/max
```

**執行計劃**:
1. 實現 CG-on-Demand 開關
2. 對比兩種設定的 ASR
3. 展示殘差上界理論

---

## 📋 數據集檢查清單

### **安全性數據集**:
- ✅ HEx-PHI (330) - `/home/wayneleo8/SafeDelta/llama2/safety_evaluation/data/hexphi.csv`
- ⏳ PureBad-100 (100) - 需要檢查
- ⏳ Identity-Shift (100) - 需要獲取
- ⏳ BeaverTails (1k/10k) - 用於規模擴增

### **效用數據集**:
- ✅ SAMSum (summarization) - 已有
- ⏳ GSM8K (math) - 需要檢查
- ⏳ MMLU (optional) - 可選
- ⏳ MT-Bench (optional) - 可選

### **過度拒答數據集**:
- ⏳ OR-Bench - 需要獲取
- ⏳ 有害→良性對話對 (200) - 需要生成

---

## 🎯 最小可交付集（MVP）

**核心實驗** (必須完成才能投稿):
1. ✅ 基礎 ASR 評估 (HEx-PHI)
2. 🔄 ρ 掃描與權衡曲線 (進行中)
3. ⏳ 主結果表 - 至少 2 個數據集 (PureBad + HEx-PHI)
4. ⏳ 推理時間成本對比
5. ✅ ROUGE 效用評估

**加分實驗** (提升錄取機會):
6. ⏳ 規模擴增曲線
7. ⏳ LoRA 場景
8. ⏳ 過度拒答檢查

**附錄實驗** (optional):
9. 跨模型泛化
10. 越獄轉移攻擊
11. 1k/10k 大規模測試

---

## 🚀 執行優先級排序

### **Phase 1: 當前自動化完成後** (今晚 20:00-20:30)
- ✅ ρ 掃描完成 (0.05-0.20)
- ✅ ρ vs ASR 曲線圖
- ✅ ρ-targeting 收斂圖

### **Phase 2: 補充數據集評估** (明天)
1. 檢查並準備 PureBad-100 數據
2. 對所有 ρ 模型評估 PureBad-100
3. 生成主結果表 (Table 2 格式)

### **Phase 3: 效用評估擴展** (明天)
4. GSM8K math 評估（如果有數據）
5. 更新效用權衡曲線

### **Phase 4: 時間成本測量** (明天)
6. vLLM 推理速度基準測試
7. 生成 Table 7 格式對比表

### **Phase 5: 加分實驗** (有時間再做)
8. 規模擴增實驗
9. LoRA 場景
10. 過度拒答檢查

---

## 📊 預期論文結構對應

### **Section 5: 實驗結果**

**5.1 實驗設置**
- 模型: Llama-3.2-3B-Instruct
- 數據集: HEx-PHI, PureBad-100, SAMSum
- 評估指標: ASR (關鍵字匹配), ROUGE-F1
- 實現細節: vLLM, PyTorch, 單卡 A100

**5.2 主結果: 安全性與效用** (Table 2 & 3)
- ✅ 當前有 HEx-PHI + SAMSum
- ⏳ 需要 PureBad-100

**5.3 ρ-Targeting 與權衡曲線** (Figure 5 & 7)
- 🔄 進行中，今晚完成

**5.4 效率優勢** (Table 7)
- ⏳ 需要測量推理時間

**5.5 消融實驗: H⁻¹ 依賴性** (Figure/Table)
- ✅ 已完成

**5.6 規模擴增穩定性** (Figure 4)
- ⏳ 可選加分

### **附錄**
- **A**: 數據集與提示詞模板
- **B**: ASR 計算與關鍵字表
- **C**: 證書欄位詳細說明 (DeltaOne++ 特色)
- **D**: 額外實驗（LoRA, 跨模型, 越獄）

---

## ✅ 檢查項目

在每個實驗完成後，確認:
- [ ] 使用與 SafeDelta 相同的數據集
- [ ] 使用與 SafeDelta 相同的評估指標
- [ ] 使用與 SafeDelta 相同的提示詞模板
- [ ] 報告相同的統計量（mean, std, samples）
- [ ] 額外報告 DeltaOne++ 的證書欄位
- [ ] 生成 SCI 品質圖表 (300 DPI PDF)
- [ ] 生成 LaTeX 表格
- [ ] 保存原始數據 (CSV/JSONL)

---

**更新日期**: 2025-10-15 18:55
**狀態**: 核心實驗進行中，需補充多數據集評估
