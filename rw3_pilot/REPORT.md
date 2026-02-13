# RW3 Kill-Switch Determination Experiment Report

**Generated:** 2026-02-13
**Environment:** Python 3.11, CPU-only, Linux
**Embedding Model:** all-MiniLM-L6-v2 (384-dim)
**Random Seeds:** [42, 123, 456]

---

## Executive Summary

| Part | Route | Decision | Criteria Passed |
|------|-------|----------|-----------------|
| A | Agent (Graph-Conditional Risk Routing) | **S1 = NO** | 0/4 |
| B | RAG (Spectral Whitening + Retrieval Features) | **S2 = NO** | 1/4 |

### Final Decision: **RETREAT TO P5**

Both primary (Agent) and backup (RAG) routes failed their kill-switch criteria. The recommended path forward is:

**P5: Geometric Adaptive Conformal Prediction + TextOOD-Bench Baseline**
- Focus on establishing strong baselines on TextOOD-Bench
- Develop geometric-adaptive CP methods without requiring extreme worst-group phenomena
- Consider this as a foundation for more targeted future work

---

## Part A: Agent Route Results

### A1: AgentDojo Data Structure Analysis

| Metric | Value |
|--------|-------|
| Total User Tasks | 86 |
| Total Injection Tasks | 27 |
| Total Security Test Combinations | 567 |
| Attack Types | 16 |
| Tool Categories | 10 |
| Benchmark Traces Parsed | 8,405 |

**Suites:**
| Suite | User Tasks | Injection Tasks | Security Tests |
|-------|------------|-----------------|----------------|
| banking | 16 | 9 | 144 |
| workspace | 33 | 6 | 198 |
| travel | 20 | 7 | 140 |
| slack | 17 | 5 | 85 |

### A2: Simulated Agent Traces

- Generated 2,000 traces with 8,958 total tool calls
- Attack rate: 30.1%
- Tool types: 7 (banking_client, email_client, calendar_client, cloud_drive, web_browse, slack, travel_booking)
- Attack types: injection, prompt_manipulation, tool_knowledge, dos

### A3: Worst-Group Analysis

**Core Finding: Worst-group phenomenon is WEAKER than hypothesized**

| Dimension | Worst/Avg Ratio | p-value | Passes (≥3, p<0.01)? |
|-----------|-----------------|---------|----------------------|
| Tool Type | 1.18 | 5.91e-01 | NO |
| Attack Type | 1.37 | 3.76e-305 | NO |
| Tool Category | 1.06 | 4.22e-01 | NO |
| Suite | 1.09 | 4.08e-01 | NO |
| DAG Depth | 1.00 | 1.00e+00 | NO |

**Maximum worst/average ratio: 1.37** (required: ≥3.0)

The attack_type dimension shows significant p-value but the ratio is only 1.37, meaning worst-group risk is not dramatically higher than average.

### A4: Routing Strategy Comparison

| Strategy | Benign Utility | Safety Rate | Overall Risk | Worst-Group Risk |
|----------|----------------|-------------|--------------|------------------|
| no_guard | 1.000±0.000 | - | 0.213±0.019 | 0.399±0.052 |
| always_abstain | 0.000±0.000 | 1.000 | 0.000±0.000 | 0.000±0.000 |
| heuristic | 0.383±0.041 | 0.623±0.038 | 0.106±0.017 | 0.183±0.040 |
| unconditional_cp | 0.907±0.026 | 0.512±0.048 | 0.067±0.009 | 0.085±0.007 |
| graph_conditional_cp | 0.836±0.027 | 0.533±0.040 | 0.057±0.007 | 0.069±0.014 |

**Key Comparisons:**
- Graph-CP worst-group improvement vs Unconditional-CP: **19.2%** (required: ≥30%)
- Graph-CP utility drop vs No-Guard: **16.4%** (required: <5%)

### A5: S1 Kill-Switch Determination

```
═══════════════════════════════════════════════════════════════
AGENT ROUTE KILL-SWITCH DETERMINATION
═══════════════════════════════════════════════════════════════

[1] Worst-group phenomenon exists?
    Max worst/average ratio = 1.37
    Dimensions with ratio ≥ 3: 0
    JUDGMENT: NO

[2] Graph-conditional CP fixes worst-group?
    Improvement = 19.2%
    JUDGMENT (≥30% required): NO

[3] Benign utility preserved?
    Utility drop = 16.4%
    JUDGMENT (<5% required): NO

[4] Conclusions hold across ≥2 dimensions?
    Dimensions passing: 0
    JUDGMENT: NO

═══════════════════════════════════════════════════════════════
S1 FINAL DETERMINATION: NO
═══════════════════════════════════════════════════════════════
```

**Failure Reasons:**
1. Worst-group phenomenon not significant (max ratio 1.37 < 3.0)
2. Graph-CP improvement insufficient (19.2% < 30%)
3. Utility drop too large (16.4% ≥ 5%)
4. No dimensions pass all criteria

---

## Part B: RAG Route Results

### B1: Embedding Anisotropy Baseline (CLINC150)

| Metric | Value |
|--------|-------|
| Train ID samples | 15,150 |
| Test ID samples | 5,470 |
| Test OOD samples | 30 |
| - Near-OOD (sim > 0.6) | 23 |
| - Medium-OOD (0.4-0.7) | 7 |
| - Far-OOD (< 0.4) | 0 |

**Anisotropy Metrics:**
| Metric | Value |
|--------|-------|
| Top-1 variance ratio | 0.0469 |
| Top-5 variance ratio | 0.1541 |
| Top-10 variance ratio | 0.2447 |
| Mean pair cosine sim | 0.0849 |

**Baseline OOD Detection (kNN, k=20):**
| Metric | Value |
|--------|-------|
| All-OOD AUROC | 0.7483 |
| Near-OOD AUROC | 0.7202 |
| Near-OOD Cohen's d | 0.6266 |

### B2: Spectral Whitening Ablation

| k | Mean Pair Sim | Cohen's d (near) | AUROC (all) | AUROC (near) |
|---|---------------|------------------|-------------|--------------|
| 0 | 0.0832 | 0.6266 | 0.7483 | 0.7202 |
| 1 | 0.0005 | 0.7235 | 0.7532 | 0.7289 |
| **2** | 0.0011 | **0.7908** | 0.7654 | 0.7445 |
| 3 | -0.0003 | 0.7713 | 0.7560 | 0.7363 |
| 5 | 0.0010 | 0.7662 | 0.7509 | 0.7331 |
| 10 | 0.0005 | 0.7205 | 0.7320 | 0.7128 |

**Best k=2:** Cohen's d improves from 0.6266 → 0.7908 (+0.1642)

### B3: Conformal Prediction Analysis

| Metric | Baseline | Whitened (k=2) |
|--------|----------|----------------|
| Threshold | 0.5127 | 0.5715 |
| ID FPR | 0.2420 | 0.2506 |
| Near-OOD TPR | 0.5217 | 0.6087 |
| Improvement | - | **+8.7 pp** |

### B4: Retrieval Graph Features

**Individual Feature AUROC:**
| Feature | All-OOD | Near-OOD |
|---------|---------|----------|
| mean_sim | 0.7265 | 0.6961 |
| std_sim | 0.7780 | 0.8067 |
| retrieval_gap | 0.7554 | 0.7959 |
| label_purity | 0.8200 | 0.8015 |

**Combined Features:**
| Method | All-OOD AUROC | Near-OOD AUROC |
|--------|---------------|----------------|
| Pure kNN (whitened) | 0.7654 | 0.7445 |
| kNN + Graph Features | **0.9148** | **0.9314** |
| Improvement | +14.9% | **+18.68%** |

### B5: Banking77 Cross-Dataset Validation

| Metric | CLINC150 | Banking77 |
|--------|----------|-----------|
| Train ID | 15,150 | 6,505 |
| Test ID | 5,470 | 2,000 |
| Test OOD | 30 | 1,080 |
| Near-OOD | 23 | 548 |
| Baseline Cohen's d | 0.6266 | 1.3497 |
| Best whitened d | 0.7908 (k=2) | 1.3497 (k=0) |

**Note:** Banking77 already has high separation (d=1.35) without whitening. The OOD samples in Banking77 are more distinguishable than CLINC150's near-OOD.

### B6: S2 Kill-Switch Determination

```
═══════════════════════════════════════════════════════════════
RAG ROUTE KILL-SWITCH DETERMINATION
═══════════════════════════════════════════════════════════════

[1] Spectral whitening improves near-OOD separation?
    Original Cohen's d: 0.6266
    Best whitened d: 0.7908
    JUDGMENT (d > 0.8): NO

[2] Conformal prediction detection rate improves?
    Improvement: 8.7 pp
    JUDGMENT (≥10 pp required): NO

[3] Retrieval graph features add incremental value?
    Improvement: 18.68%
    JUDGMENT (≥2% required): YES ✓

[4] Cross-dataset generalization?
    CLINC150: d=0.7908 (FAIL)
    Banking77: d=1.3497 (PASS)
    JUDGMENT (both pass): NO

═══════════════════════════════════════════════════════════════
S2 FINAL DETERMINATION: NO
Criteria Passed: 1/4 (need ≥3)
═══════════════════════════════════════════════════════════════
```

**Failure Reasons:**
1. Cohen's d = 0.7908 < 0.8 (close but doesn't meet threshold)
2. CP improvement = 8.7 pp < 10 pp
3. Cross-dataset generalization failed (CLINC150 doesn't reach d > 0.8)

**Positive Finding:**
- Retrieval graph features provide substantial value (+18.68% AUROC), suggesting this is a promising direction for future work

---

## Key Insights and Lessons

### Why Agent Route Failed (S1)

1. **The worst-group phenomenon is not as extreme as hypothesized in the RW3 proposal**
   - Maximum worst/average ratio was 1.37, far below the required 3.0
   - This suggests that while risk varies across tool types and attack types, the variation is not dramatic enough to require specialized graph-conditional routing

2. **The simulation may not capture real-world worst-case scenarios**
   - Our simulation used reasonable risk distributions but may underestimate tail risks
   - Real AgentDojo traces might show more extreme patterns, especially for sophisticated attacks

3. **The utility-safety tradeoff is steeper than expected**
   - Graph-conditional CP achieves only 83.6% utility (vs 100% for no-guard)
   - This suggests the current CP calibration approach is too conservative

### Why RAG Route Failed (S2)

1. **Near-OOD detection remains fundamentally challenging**
   - Even with spectral whitening, Cohen's d only reaches 0.79 (vs 0.8 required)
   - The CLINC150 dataset has very subtle near-OOD samples that are inherently hard to distinguish

2. **Dataset-specific behavior**
   - Banking77 already has excellent separation (d=1.35) without any preprocessing
   - The "near-OOD problem" is dataset-dependent, not universal

3. **Retrieval features show promise but can't save the overall method**
   - +18.68% AUROC improvement is substantial
   - But the base kNN method needs to be stronger for the combination to exceed thresholds

---

## Recommendations: P5 Path Forward

Since both S1 and S2 failed, we recommend **retreating to P5**:

### 1. TextOOD-Bench Baseline Establishment
- Create comprehensive baselines on the TextOOD benchmark
- Establish clear performance targets for different OOD difficulty levels
- Identify which datasets/scenarios show the most promising gap for improvement

### 2. Geometric Adaptive Conformal Prediction
- Develop CP methods that adapt to local embedding geometry without requiring extreme worst-group phenomena
- Focus on calibration efficiency rather than worst-group guarantees
- Consider hierarchical or multi-scale approaches

### 3. Leverage the Retrieval Graph Finding
- The +18.68% improvement from graph features is a concrete positive result
- Investigate why label_purity and retrieval_gap are particularly informative
- Consider graph neural network approaches to better exploit this structure

### 4. Dataset Curation for Near-OOD
- The CLINC150 near-OOD is extremely challenging (only 30 samples, d<0.8)
- Consider constructing controlled near-OOD benchmarks with known difficulty gradations
- Partner with domain experts to define realistic near-OOD scenarios

---

## Artifacts Generated

```
rw3_pilot/
├── REPORT.md                          # This report
├── results/
│   ├── agentdojo_analysis.md          # AgentDojo structure analysis
│   ├── simulated_traces.csv           # 8,958 simulated tool calls
│   ├── worst_group_table.md           # Worst-group statistics
│   ├── agent_routing_comparison.md    # 5-strategy comparison
│   ├── baseline_anisotropy.md         # Embedding anisotropy baseline
│   ├── whitening_ablation.md          # Whitening k-ablation
│   ├── s1_result.pkl                  # Pickled S1 results
│   ├── s2_result.pkl                  # Pickled S2 results
│   ├── fig_risk_by_*.png              # Risk visualization plots
│   ├── fig_cohens_d_vs_k.png          # Whitening effect plot
│   ├── fig_auroc_by_severity.png      # AUROC by OOD severity
│   └── fig_retrieval_graph.png        # Graph features comparison
└── scripts/
    ├── agent_pilot.py                 # Part A complete code
    └── rag_pilot.py                   # Part B complete code
```

---

## Encountered Issues

1. **AgentDojo dataset loading**: Required exploring the repository structure manually since running agents requires LLM API keys

2. **Banking77 dataset**: Required using alternative dataset identifier (`banking77` instead of `PolyAI/banking77`)

3. **CLINC150 OOD scarcity**: Only 30 OOD test samples in CLINC150, making statistical conclusions fragile

4. **Simulation vs Reality gap**: Simulated agent traces may not capture real attack patterns; real AgentDojo benchmarks show different dynamics

---

## Reproducibility

All experiments use fixed random seeds [42, 123, 456]. To reproduce:

```bash
cd /home/user/OOD-project/rw3_pilot
python scripts/agent_pilot.py  # Part A
python scripts/rag_pilot.py    # Part B
```

Requirements: numpy, scipy, scikit-learn, pandas, matplotlib, seaborn, networkx, torch, sentence-transformers, datasets, faiss-cpu
