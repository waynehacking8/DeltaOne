# DeltaOne++ vs SafeDelta 系统性对比实验计划

## 实验目标

通过系统性对比实验,证明 DeltaOne++ 的核心主张:
1. **Rank-Free ADB**: H⁻¹ 不是关键,δw 自适应预算才是核心
2. **337× 加速 + 47× 内存缩减**: 单模型流式处理的效率优势
3. **Pareto 改进**: 安全性和效用同时提升
4. **理论保证**: 5个可验证的理论证书

---

## A. 主结果表 - 与 SafeDelta 核心对比

### 目标
复刻 SafeDelta Table 的核心结果,展示在同设定下 ASR 持平/更好,时间/内存大幅下降

### 实验设置
- **模型**:
  - Llama-2-7B-Chat (全参数 & LoRA)
  - Llama-3-8B-Instruct (全参数 & LoRA)
- **数据集**:
  - PureBad (100 samples)
  - Identity-Shift (100 samples)
  - Dirty-Summary (1100 samples)
  - GSM8K (7.5k samples)
- **对比方法**:
  - Original (未处理)
  - Fine-tuned (有害微调)
  - SafeInstr
  - BEA
  - SafeLoRA
  - Resta
  - SafeDelta
  - **DeltaOne++**

### 评估指标
- **安全性**: ASR (Attack Success Rate), HS (Harmfulness Score)
- **效用**: MT-Bench, MMLU, Summary-F1, Math-Acc
- **系统**: Time, Peak-Memory

### DeltaOne++ 额外报告
- `selection_ratio (ρ)`
- `epsilon_used`
- `dual_gap`
- `pac_bayes.upper_bound`
- `robust.feasible`

### 图表要求
- **表格**: 4个数据集 × 8种方法的完整对比表 (SCI 双栏格式)
- **表格**: 系统成本对比 (复刻 SafeDelta Table-7)
  - Per-request extra time
  - Preparation time
  - Peak memory

---

## B. H⁻¹ 依赖性实验 - 核心主张

### 目标
证明曲率信息不精确也没关系,甚至 dummy 最优

### 实验设置
- **模型**: Llama-3.2-3B (已有)
- **数据**: PureBad-100 (已有 delta weights)
- **H⁻¹ 变体**:
  1. **Exact H⁻¹**: 标准 SafeDelta (需实现或用官方代码)
  2. **Dummy H⁻¹**: 全1矩阵 (I)
  3. **Random H⁻¹**: Uniform[0.5, 1.5] 随机扰动
  4. **Buggy H⁻¹**: Cholesky(H⁻¹) 错误计算
  5. **DeltaOne++ (ADB)**: 完全不使用 H⁻¹

### 评估指标
- ASR (主指标)
- Selection ratio ρ
- Time
- PAC-Bayes upper bound

### 图表要求
- **柱状图**: ASR vs H⁻¹ 类型 (5组对比)
  - Y轴: ASR (%)
  - X轴: 5种方法
  - 标注最佳值
- **折线图**: ρ vs H⁻¹ 类型 (显示 random 导致过选)
- **SCI 要求**:
  - 双栏宽度 (约 3.5 inches)
  - 300 DPI
  - 误差棒 (如有重复实验)
  - 清晰图例

---

## C. ρ-s 曲线扫描 - 安全-效用甜蜜点

### 目标
复刻 SafeDelta Fig.7 的 trade-off,用 ρ 控制补强 "10-15% 最佳" 主张

### 实验设置
- **模型**: Llama-3.2-3B
- **数据**: Dirty-Summary (主实验)
- **参数扫描**:
  - Scale `s ∈ {0.05, 0.08, 0.11, 0.15, 0.20}`
  - Selection ratio `ρ ∈ {0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30}`

### 评估指标
- ASR (安全性)
- ROUGE-L / Summary-F1 (效用)
- ρ-Targeting convergence

### 图表要求
- **图1**: ASR vs ρ (U型曲线)
  - 标注 ρ ≈ 0.12 最优点
  - 双Y轴: ASR (左), ρ_actual (右)
  - 显示 ρ-Targeting 收敛曲线
- **图2**: ASR vs s (宽甜蜜区)
  - SafeDelta vs DeltaOne++
  - 显示 DeltaOne++ 更稳定
- **图3**: 2D heatmap (ρ × s → ASR)
- **SCI 要求**:
  - 单/双栏可选
  - 彩色 colormap (viridis/plasma)
  - Colorbar with label

---

## D. 数据规模扩增曲线

### 目标
展示有害数据变大时仍稳定,数据法退化、权重法波动

### 实验设置
- **数据**:
  - PureBad: {50, 100, 150, 200} 或扩展到 {1k, 10k}
  - Identity-Shift: 同上
- **对比方法**:
  - DeltaOne++
  - SafeDelta
  - BEA
  - SafeInstr

### 评估指标
- ASR
- Time
- Peak Memory

### 图表要求
- **折线图**: ASR vs Dataset Size
  - 4条曲线(4种方法)
  - 误差棒或阴影区域
- **表格**: Time & Memory vs Size
- **SCI 要求**:
  - Log scale X轴 (如跨度大)
  - 清晰区分线型/颜色

---

## E. LoRA vs 全参数对比

### 目标
显示方法在两种微调路径都有效,LoRA 下更快

### 实验设置
- **数据**: PureBad, Dirty-Summary
- **微调方式**:
  - 全参数 (Full fine-tuning)
  - LoRA (rank=8)
- **对比**: DeltaOne++ vs SafeDelta

### 评估指标
- ASR
- Utility (ROUGE-L)
- Time
- Peak Memory

### 图表要求
- **表格**: 2×2 对比 (全参数/LoRA × DeltaOne++/SafeDelta)
- **柱状图**: Time & Memory (分组对比)
- **标注**: LoRA 下 DeltaOne++ 的 Δ-only 优势

---

## F. OBS 补偿影响

### 目标
证明即使关闭补偿也很好,开启后可小幅提升;残差符合上界

### 实验设置
- **条件**: `--obs off` vs `--obs on`
- **CG 参数**: tol=1e-3, max_iter=100

### 评估指标
- ASR / Utility
- CG residual (mean, max)
- Selected columns count

### 图表要求
- **表格**: OBS off/on 对比
- **直方图**: CG residual distribution
- **SCI 要求**: 附录级别图表

---

## G. 串流选择等价性验证

### 目标
证明 K-way heap 与全量排序一致;近似 scan 也有好近似比

### 实验设置
- **模式**:
  - `--mode heap` (精确)
  - `--mode scan` (二分阈值,≤12 趟)
  - `--mode full` (全排序,baseline)

### 评估指标
- 选中 index 重叠率 (heap vs full)
- Dual gap (heap ≈ 0, scan < threshold)
- Approximation ratio (基于弱次模 γ)

### 图表要求
- **表格**: 重叠率 (应为 100%)
- **柱状图**: Dual gap 对比
- **标注**: γ 值与理论保证

---

## H. 系统性能与证书报告

### 目标
实证 337× 时间、47× 内存;让证书在报表出现

### 实验设置
- **任务**: 完整 Pass-1 + Pass-2 流程
- **模型**: Llama-3.2-3B
- **对比**: SafeDelta vs DeltaOne++

### 评估指标
- Wall-clock time
- Peak memory (RSS)
- 理论证书:
  - `dual_gap`
  - `pac_bayes.upper_bound`
  - `robust.feasible(η, Γ)`
  - `selection_ratio`

### 图表要求
- **表格**: Time & Memory 对比 + 证书栏位
- **附录**: 真实 `selection_stats.json` 片段
- **SCI 要求**:
  - 时间用 log scale
  - 内存用 GB 单位

---

## I. 稳健性与行为评估

### 目标
避免过拒或被攻击转移

### 实验设置
1. **OR-Bench (过拒)**: Original vs DeltaOne++
2. **Jailbreak 转移**: GCG / PAIR 攻击
3. **Benign→Harmful 交互**: 有害提问后跟 benign 任务

### 评估指标
- OR-Bench false positive rate
- Jailbreak ASR (转移攻击)
- Benign task performance drop

### 图表要求
- **表格**: 3种稳健性指标对比
- **柱状图**: OR-Bench 各类别表现

---

## J. 层级与 δw 结构分析

### 目标
证明 "是 δw 结构在起作用"

### 实验设置
- **分析维度**:
  - δw 分布 (被选 vs 未选)
  - 层级选择率 (Attention vs FFN)
  - ρ 与 ASR/效用的层级散点图

### 图表要求
- **直方图**: δw magnitude (log scale)
- **热图**: 层级选择率矩阵
- **散点图**: ρ vs ASR (按层分组)
- **SCI 要求**:
  - 多子图组合 (2×2 或 1×3)
  - 统一配色方案

---

## 最小必做清单 (时间紧迫版)

如果时间很赶,优先完成以下 5 项:

1. ✅ **实验 B**: H⁻¹ 依赖性 (核心主张)
2. ✅ **实验 C**: ρ-s 曲线 (甜蜜点证明)
3. ✅ **实验 H**: 系统性能表 (337× 加速实证)
4. ✅ **实验 A**: 主结果表 (至少 2 个数据集)
5. ✅ **实验 I**: OR-Bench + 1 种越狱 (稳健性)

完成这 5 项后,"曲率不关键、δw-adaptive 才是重点" 的主张就很难被挑战。

---

## 实验执行顺序建议

### Phase 1 (第1周)
1. 实验 B (H⁻¹ 依赖性)
2. 实验 C (ρ-s 曲线)
3. 实验 H (系统性能)

### Phase 2 (第2周)
4. 实验 A (主结果表,简化版)
5. 实验 F (OBS 补偿)
6. 实验 E (LoRA 对比)

### Phase 3 (第3周)
7. 实验 I (稳健性评估)
8. 实验 D (数据规模)
9. 实验 G (串流验证)
10. 实验 J (结构分析)

---

## 绘图规范 (SCI 标准)

### 通用要求
- **分辨率**: 300 DPI (矢量格式优先: PDF/SVG)
- **字体**: Arial/Helvetica, 8-10pt
- **尺寸**:
  - 单栏: 3.5 inches 宽
  - 双栏: 7.0 inches 宽
- **配色**: 色盲友好 (使用 colorblind-safe palette)
- **图例**: 清晰,不遮挡数据
- **坐标轴**: 标签 + 单位

### Python 绘图库
```python
import matplotlib.pyplot as plt
import seaborn as sns

# SCI 风格设置
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# 色盲友好配色
colors = sns.color_palette("colorblind")
```

### 保存格式
- 主图: PDF (矢量)
- 备份: PNG (300 DPI)
- 数据: CSV (原始数据)

---

## 数据组织

```
experiments/
├── configs/              # 实验配置文件
│   ├── exp_b_hinv.json
│   ├── exp_c_rho_scan.json
│   └── ...
├── results/              # 实验结果
│   ├── exp_b_hinv/
│   │   ├── run_1/
│   │   │   ├── selection_stats.json
│   │   │   ├── model/
│   │   │   └── logs/
│   │   └── summary.csv
│   └── ...
├── figures/              # 生成的图表
│   ├── fig_b_hinv_asr.pdf
│   ├── fig_c_rho_curve.pdf
│   └── ...
└── scripts/              # 实验脚本
    ├── run_exp_b.sh
    ├── plot_exp_b.py
    └── ...
```

---

## 检查清单

实验完成后,确保:
- [ ] 所有图表达到 SCI 标准 (300 DPI, 清晰图例)
- [ ] 每个实验有对应的原始数据 (CSV/JSON)
- [ ] 代码可复现 (脚本 + 配置 + README)
- [ ] 关键结果有统计显著性检验 (如适用)
- [ ] 图表编号与论文正文对应
- [ ] 附录包含完整的 `selection_stats.json` 样例

---

**Last Updated**: 2025-10-15
**Status**: 实验规划完成,待执行
**Maintainer**: DeltaOne Team
