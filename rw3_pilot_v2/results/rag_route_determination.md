# RAG Route Kill-Switch Determination (v2, Real Data)

```
═══════════════════════════════════════
RAG路线 KILL-SWITCH 判定（v2，真实数据）
═══════════════════════════════════════

实验配置：
  clinc150_all-MiniLM-L6-v2: clinc150 + all-MiniLM-L6-v2
  clinc150_bge-base-en-v1.5: clinc150 + BAAI/bge-base-en-v1.5
  banking77_all-MiniLM-L6-v2: banking77 + all-MiniLM-L6-v2
  banking77_bge-base-en-v1.5: banking77 + BAAI/bge-base-en-v1.5

═══ 条件1：光谱白化改善near-OOD分离度 ═══
clinc150_all-MiniLM-L6-v2:
  原始 Cohen's d = 1.64，最优k*=1，白化后 Cohen's d = 1.74
clinc150_bge-base-en-v1.5:
  原始 Cohen's d = 1.54，最优k*=1，白化后 Cohen's d = 2.03
banking77_all-MiniLM-L6-v2:
  原始 Cohen's d = 0.79，最优k*=20，白化后 Cohen's d = 1.11
banking77_bge-base-en-v1.5:
  原始 Cohen's d = 0.95，最优k*=20，白化后 Cohen's d = 1.25

4组实验中，白化后Cohen's d > 0.8的组数 = 4/4
判定：YES（≥2组达到0.8为YES）

═══ 条件2：CP检测率显著提升 ═══
near-OOD检测率提升（α=0.10）：
  clinc150_all-MiniLM-L6-v2: +8.0pp
  clinc150_bge-base-en-v1.5: +31.6pp
  banking77_all-MiniLM-L6-v2: +10.0pp
  banking77_bge-base-en-v1.5: +11.9pp

3组中提升≥10pp的组数 = 3/4
判定：YES（≥2组达到10pp为YES）

═══ 条件3：检索图特征增量 ═══
near-OOD AUROC增量（kNN+图特征 vs 纯kNN）：
  clinc150_all-MiniLM-L6-v2: +1.4%
  clinc150_bge-base-en-v1.5: +4.2%
  banking77_all-MiniLM-L6-v2: +8.1%
  banking77_bge-base-en-v1.5: +7.3%

3组中增量≥2%的组数 = 3/4
判定：YES（≥2组达到2%为YES）

═══ 条件4：跨数据集+跨模型鲁棒性 ═══
条件1在两个数据集上都YES？ True
条件1在两个模型上都YES？ True
判定：YES

═══════════════════════════════════════
S2总判定 = YES
（4个条件中4个YES则S2=YES）

决策：走RAG方向
原因：4个条件中4个满足（≥3），RAG路线验证通过
最有前景的信号：clinc150_bge-base-en-v1.5 (d=2.03)
最大障碍/风险：需要更多数据集验证
═══════════════════════════════════════
```
