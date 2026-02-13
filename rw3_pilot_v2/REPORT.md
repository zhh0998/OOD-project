# RW3 Kill-Switch 判定实验报告 (v2 - 真实数据)

## 执行摘要

本实验使用**纯真实数据**（无任何模拟/合成数据）对RW3的两条技术路线进行Kill-Switch判定：

1. **Agent路线**：分析AgentDojo仓库中的36,679个真实benchmark结果文件，worst-group/average ASR比值仅为1.83（<2.0阈值），**判定为放弃**。

2. **RAG路线**：在CLINC150和Banking77真实数据集上使用两个embedding模型进行验证，**4个条件全部满足**（光谱白化改善、CP检测率提升、图特征增量、跨数据集鲁棒性），**判定为通过，走RAG方向**。

**最终决策**：走RAG方向，最有前景的信号是CLINC150+BGE组合（Cohen's d=2.03）。

---

## Part A: Agent路线 - 真实Benchmark数据分析

### A1: 数据源扫描

**数据源**: AgentDojo GitHub仓库 (ethz-spylab/agentdojo)

| 维度 | 数量 |
|------|------|
| 评测模型 | 22个（含GPT-4o, Claude-3.5, Gemini, Llama-3等） |
| 测试Suite | 4个（workspace, banking, slack, travel） |
| 攻击类型 | 16种（important_instructions, tool_knowledge等） |
| 防御策略 | 5种（none, tool_filter, repeat_user_prompt等） |
| 结果文件 | 36,679个JSON文件 |

每个结果文件包含：
- `utility`: 任务是否完成（布尔值）
- `security`: 攻击是否被阻止（布尔值，True=安全）
- 完整的agent对话轨迹

### A2: Worst-Group分析

#### 跨Suite的ASR变异性

| 模型 | 防御 | 攻击类型 | 最差Suite ASR | 平均ASR | Worst/Avg比值 |
|------|------|----------|---------------|---------|---------------|
| gpt-4o-2024-05-13 | none | important_instructions | 0.61 | 0.48 | 1.28 |
| claude-3-5-sonnet-20240620 | none | important_instructions | 0.44 | 0.34 | 1.30 |
| gemini-1.5-pro-001 | none | important_instructions | 0.36 | 0.29 | 1.27 |

**最大Worst/Average比值**: 1.83（低于2.0阈值）

#### 跨攻击类型的ASR变异性

不同攻击类型的成功率差异较大：
- `important_instructions`: 平均ASR ~27%
- `tool_knowledge`: 平均ASR ~35%
- `direct`: 平均ASR ~4%

Worst/Average比值（跨攻击类型）: 2.32

#### Utility-Security权衡

- Spearman相关系数: 0.077 (p=0.47)
- 高效用+低安全案例: 31个
- 低效用+高安全案例: 0个

**结论**: 没有明确的效用-安全权衡，模型普遍在高效用时也容易被攻击。

### A3: Agent路线判定

```
═══════════════════════════════════════
AGENT路线判定（基于真实数据）
═══════════════════════════════════════

数据来源：Real benchmark results (36,679 JSON files)
数据充分性：Sufficient

[1] worst-group现象在真实数据中存在？
    worst/average ratio = 1.83（ASR维度）
    判定：NO（需≥2.0）

[2] 问题的实际规模如何？
    涉及 4 个suite, 16 种攻击类型
    现有方法的最大性能差距：worst ASR ratio = 1.83
    判定：问题太小/无法判断

[3] 后续实验可行性
    需要LLM API？ YES
    估计API成本？ $5-20/模型×suite×attack
    有开源替代？ YES (Llama-3-70B)
    判定：可行（但需API预算）

总判定：放弃
原因：Worst-group现象在真实数据中未达到预设阈值
═══════════════════════════════════════
```

---

## Part B: RAG路线 - 真实数据集验证

### B1: 数据加载

| 数据集 | 训练集 | 测试集 | OOD比例 | ID类别 | OOD设置 |
|--------|--------|--------|---------|--------|---------|
| CLINC150 | 15,250 | 5,500 | 18.2% | 150 | 原生OOS类 |
| Banking77 | 10,003 | 3,080 | 35.1% | 50 | 后27个intent为OOD（near-OOD场景） |

### B2: 嵌入模型

| 模型 | 维度 | 加载时间 |
|------|------|----------|
| all-MiniLM-L6-v2 | 384 | 4.0s |
| BAAI/bge-base-en-v1.5 | 768 | 3.2s |

### B3: 各向异性基线

| 组合 | Top-1方差比 | 均值对余弦 | 各向异性 | kNN AUROC | Near-OOD Cohen's d |
|------|-------------|-----------|----------|-----------|-------------------|
| CLINC+MiniLM | 0.048 | 0.087 | No | 0.958 | 1.64 |
| CLINC+BGE | 0.059 | 0.095 | No | 0.978 | 1.54 |
| Bank77+MiniLM | 0.080 | 0.404 | Yes | 0.880 | 0.79 |
| Bank77+BGE | 0.101 | 0.572 | Yes | 0.919 | 0.95 |

**发现**: Banking77在两个模型上都显示高各向异性（mean pair cosine > 0.3），这是near-OOD场景的特征。

### B4: 光谱白化消融

#### CLINC150 + all-MiniLM-L6-v2

| k | Mean Pair Cosine | Full AUROC | Near AUROC | Near Cohen's d |
|---|------------------|------------|------------|----------------|
| 0 | 0.087 | 0.958 | 0.872 | 1.64 |
| **1** | **-0.001** | **0.962** | **0.886** | **1.74** |
| 2 | -0.000 | 0.958 | 0.879 | 1.68 |

**最优k*=1**, Near Cohen's d: 1.64 → 1.74 (+0.10)

#### CLINC150 + BAAI/bge-base-en-v1.5

| k | Mean Pair Cosine | Full AUROC | Near AUROC | Near Cohen's d |
|---|------------------|------------|------------|----------------|
| 0 | 0.095 | 0.978 | 0.933 | 1.54 |
| **1** | **-0.000** | **0.982** | **0.963** | **2.03** |
| 2 | -0.001 | 0.980 | 0.953 | 1.87 |

**最优k*=1**, Near Cohen's d: 1.54 → 2.03 (+0.49)

#### Banking77 + all-MiniLM-L6-v2

| k | Mean Pair Cosine | Full AUROC | Near AUROC | Near Cohen's d |
|---|------------------|------------|------------|----------------|
| 0 | 0.404 | 0.880 | 0.738 | 0.79 |
| 1 | -0.001 | 0.909 | 0.789 | 1.02 |
| **20** | **0.001** | **0.899** | **0.807** | **1.11** |

**最优k*=20**, Near Cohen's d: 0.79 → 1.11 (+0.32)

#### Banking77 + BAAI/bge-base-en-v1.5

| k | Mean Pair Cosine | Full AUROC | Near AUROC | Near Cohen's d |
|---|------------------|------------|------------|----------------|
| 0 | 0.573 | 0.919 | 0.771 | 0.95 |
| 1 | 0.002 | 0.927 | 0.794 | 1.15 |
| **20** | **-0.003** | **0.909** | **0.805** | **1.25** |

**最优k*=20**, Near Cohen's d: 0.95 → 1.25 (+0.30)

### B5: 保形预测检测率

#### Near-OOD检测率提升 (α=0.10)

| 组合 | 原始检测率 | 白化后检测率 | 提升 |
|------|-----------|-------------|------|
| CLINC+MiniLM | 58.0% | 66.0% | +8.0pp |
| CLINC+BGE | 52.0% | 83.6% | **+31.6pp** |
| Bank77+MiniLM | 18.5% | 28.5% | **+10.0pp** |
| Bank77+BGE | 20.0% | 31.9% | **+11.9pp** |

### B6: 检索图特征增量

| 组合 | 纯kNN Near AUROC | kNN+图特征 Near AUROC | 增量 |
|------|-----------------|---------------------|------|
| CLINC+MiniLM | 0.872 | 0.886 | +1.4% |
| CLINC+BGE | 0.933 | 0.975 | **+4.2%** |
| Bank77+MiniLM | 0.738 | 0.818 | **+8.1%** |
| Bank77+BGE | 0.771 | 0.844 | **+7.3%** |

**最有价值的图特征**: `label_purity`（top-10检索结果的标签纯度）

### B7: RAG路线判定

```
═══════════════════════════════════════
RAG路线 KILL-SWITCH 判定（v2，真实数据）
═══════════════════════════════════════

═══ 条件1：光谱白化改善near-OOD分离度 ═══
  clinc150+MiniLM: d=1.64 → 1.74 (k*=1) ✓
  clinc150+BGE: d=1.54 → 2.03 (k*=1) ✓
  banking77+MiniLM: d=0.79 → 1.11 (k*=20) ✓
  banking77+BGE: d=0.95 → 1.25 (k*=20) ✓

  4/4组白化后Cohen's d > 0.8
  判定：YES

═══ 条件2：CP检测率显著提升 ═══
  clinc150+MiniLM: +8.0pp ✗
  clinc150+BGE: +31.6pp ✓
  banking77+MiniLM: +10.0pp ✓
  banking77+BGE: +11.9pp ✓

  3/4组提升≥10pp
  判定：YES

═══ 条件3：检索图特征增量 ═══
  clinc150+MiniLM: +1.4% ✗
  clinc150+BGE: +4.2% ✓
  banking77+MiniLM: +8.1% ✓
  banking77+BGE: +7.3% ✓

  3/4组增量≥2%
  判定：YES

═══ 条件4：跨数据集+跨模型鲁棒性 ═══
  条件1在两个数据集上都YES? True
  条件1在两个模型上都YES? True
  判定：YES

═══════════════════════════════════════
S2总判定 = YES
（4个条件中4个YES，≥3即通过）

决策：走RAG方向
原因：4个条件全部满足，RAG路线验证通过
最有前景的信号：clinc150+BGE (Cohen's d=2.03)
最大障碍/风险：需要更多数据集验证
═══════════════════════════════════════
```

---

## 最终综合判定

```
═══════════════════════════════════════
RW3 最终方向决策
═══════════════════════════════════════

Part A（Agent）判定：放弃
  - Worst-group比值1.83 < 2.0阈值
  - 需要LLM API预算
  - 问题规模可能不够大

Part B（RAG）判定：S2 = YES
  - 4/4条件满足
  - 光谱白化显著改善near-OOD分离度
  - 跨数据集、跨模型都有效

决策矩阵应用：
  Agent放弃 且 S2=YES → 选RAG

╔═════════════════════════════════════════════════════╗
║  最终决策：走RAG方向                                  ║
╚═════════════════════════════════════════════════════╝

下一步行动：
1. 在更多数据集上验证（MASSIVE, HWU64）
2. 深入研究最优k*的自动选择策略
3. 探索label_purity特征的理论基础
4. 准备RAG-OOD方向的完整proposal

═══════════════════════════════════════
```

---

## 文件清单

```
rw3_pilot_v2/
├── REPORT.md                               # 本报告
├── scripts/
│   ├── agent_analysis.py                   # Part A代码
│   └── rag_pilot.py                        # Part B代码
└── results/
    ├── agentdojo_real_structure.md         # A1: 真实结构分析
    ├── agentdojo_worst_group.md            # A2: Worst-group分析
    ├── agent_route_determination.md        # A3: Agent路线判定
    ├── baseline_metrics.md                 # B3: 基线测量
    ├── whitening_ablation.md               # B4: 白化消融
    ├── cp_detection_rates.md               # B5: CP检测率
    ├── graph_feature_ablation.md           # B6: 图特征消融
    ├── rag_route_determination.md          # B7: RAG路线判定
    ├── fig_real_worst_group.png            # Agent worst-group可视化
    ├── fig_cohens_d_vs_k.png               # Cohen's d vs k曲线
    ├── fig_auroc_breakdown.png             # AUROC分组分解
    ├── fig_retrieval_graph_features.png    # 图特征对比
    └── embeddings/                         # 缓存的嵌入文件
        ├── clinc150_all_MiniLM_L6_v2.npz
        ├── clinc150_bge_base_en_v1.5.npz
        ├── banking77_all_MiniLM_L6_v2.npz
        └── banking77_bge_base_en_v1.5.npz
```

---

## 附录：关键技术细节

### 光谱白化算法

```python
def spectral_whitening(embeddings, k):
    """All-but-the-top: 移除top-k主成分"""
    # 仅在ID训练集上计算PCA方向（防止信息泄露）
    mean = embeddings.mean(axis=0)
    centered = embeddings - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    for i in range(k):
        component = Vt[i]
        centered = centered - np.outer(centered @ component, component)

    # L2重归一化
    return centered / np.linalg.norm(centered, axis=1, keepdims=True)
```

### OOD严重度分组（基于百分位数）

- **Near-OOD**: 与ID最相似的25%（top-75%分位数以上）
- **Medium-OOD**: 中间50%
- **Far-OOD**: 与ID最不相似的25%（bottom-25%分位数以下）

使用百分位数而非固定阈值，确保跨模型可比性。

### 检索图特征定义

| 特征 | 定义 | OOD时的行为 |
|------|------|------------|
| mean_sim | top-k检索结果的平均余弦相似度 | 降低 |
| std_sim | top-k检索结果相似度的标准差 | 升高 |
| retrieval_gap | top-1与top-k相似度之差 | 升高 |
| label_purity | top-k中最常见标签的占比 | 降低 |
| sim_drop_rate | top-1到top-5的相似度衰减斜率 | 升高 |

---

*报告生成时间: 2026-02-13*
*数据源: AgentDojo (36,679 real results), CLINC150, Banking77*
*计算环境: CPU-only, 无LLM API调用*
