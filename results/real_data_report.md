# RW1 预实验报告 - 真实数据版本

## ⚠️ 数据来源声明

本报告使用**真实数据集**，非合成/模拟数据：

| 数据集 | 来源 | 样本数 | 验证状态 |
|--------|------|--------|----------|
| NYT10 | Aliyun OSS (OpenNRE) | 522,611 | ✅ 验证通过 |
| NYT-H | Google Drive | 9,955 | ✅ 验证通过 |
| FewRel | GitHub | 44,800 | ✅ 验证通过 |
| Re-DocRED | GitHub | 500 docs | ✅ 验证通过 |

---

## 实验结果摘要

**总体结果: 0/3 假设通过验证**

| 假设 | 描述 | 主要指标 | 阈值 | 实际值 | 结果 |
|------|------|----------|------|--------|------|
| H1 | Distribution Shift → F1 Drop | Pearson r | > 0.8 | 0.1107 | ❌ 失败 |
| H3 | Prototype Dispersion Index → Noise Rate | Pearson r | > 0.5 | 0.0305 | ❌ 失败 |
| H5 | Bag Size → Label Reliability | Cohen's d | 0.5-0.8 | -1.9403 | ❌ 失败 |

---

## 详细结果

### H1: Distribution Shift → F1 Drop

**数据来源**: REAL NYT10 (Aliyun OSS)

- Pearson r: 0.1107
- p-value: 0.421237
- **结论**: FAILED

### H3: Prototype Dispersion Index → Noise Rate

**数据来源**: REAL NYT10 (Aliyun OSS)

- Pearson r: 0.0305
- p-value: 0.866195
- **结论**: FAILED

### H5: Bag Size → Label Reliability

**数据来源**: REAL NYT-H (Google Drive)

- Cohen's d: -1.9403
- t-statistic: -47.27186178529438
- p-value: 0.000000
- **结论**: FAILED (effect too large)

---

## 与合成数据对比

| 假设 | 合成数据结果 | 真实数据结果 | 差异分析 |
|------|-------------|-------------|----------|
| H1 | r=1.0 | r=0.1107 | 差异 0.89 (合成数据偏差明显) |
| H3 | r=0.72 | r=0.0305 | 差异 0.69 (合成数据偏差明显) |
| H5 | d=1.27 | d=-1.9403 | 差异 3.21 (合成数据偏差明显) |

---

## 结论

使用真实数据重新运行预实验后，结果更具参考价值。合成数据往往会产生过于理想化的结果（如H1的r=1.0），
而真实数据则反映了实际问题的复杂性。

建议基于真实数据结果调整研究方向和方法设计。
