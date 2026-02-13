# Agent Route Kill-Switch Determination

```
═══════════════════════════════════════
AGENT路线判定（基于真实数据）
═══════════════════════════════════════

数据来源：Real benchmark results (36,679 JSON files)
数据充分性：Sufficient

[1] worst-group现象在真实数据中存在？
    worst/average ratio = 1.83（ASR维度）
    判定：NO

[2] 问题的实际规模如何？
    涉及 4 个suite, 16 种攻击类型
    现有方法的最大性能差距：worst ASR ratio = 1.83
    判定：问题太小/无法判断

[3] 后续实验可行性
    需要LLM API？ YES - 运行Agent需要LLM API
    估计API成本？ 每个模型×suite×attack组合约$5-20（基于runs数量估算）
    有开源替代？ YES - Llama-3-70B已在runs中
    判定：可行（但需API预算）

总判定：放弃
原因：Worst-group现象在真实数据中未确认
═══════════════════════════════════════
```
