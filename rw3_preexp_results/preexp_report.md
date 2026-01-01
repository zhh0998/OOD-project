# RW3 Pre-Experiment Report: Heterophily-OOD Association Verification

**Date**: 2026-01-01 17:14:54

---

## 1. Executive Summary

### Layer 1 Hypothesis Verification Results

| Method | Cohen's d | 95% CI | p-value | AUROC | Success |
|--------|-----------|--------|---------|-------|--------|
| Pseudolabel | 1.5455+/-0.0255 | - | 2.10e-293 | 0.7794+/-0.0035 | Yes |
| Similarity | 2.7818+/-0.0000 | - | 0.00e+00 | 0.9646+/-0.0000 | Yes |
| Entropy | 0.5155+/-0.0246 | - | 7.14e-54 | 0.6330+/-0.0063 | Yes |

**Conclusion**: Layer 1 hypothesis VERIFIED. Heterophily-OOD association exists.

## 2. Experimental Configuration

- Dataset: CLINC150 (small config)
- Encoder: all-mpnet-base-v2
- Optimal k: 10
- Number of runs: 10

## 3. K-Value Sensitivity Analysis

| k | Edge Het. | Cohen's d | AUROC |
|---|-----------|-----------|-------|
| 5 | 0.5052 | 1.4878 | 0.7731 |
| 10 | 0.5117 | 1.4968 | 0.7744 |
| 13 | 0.5186 | 1.4959 | 0.7746 |
| 15 | 0.5238 | 1.4961 | 0.7746 |
| 20 | 0.5379 | 1.4868 | 0.7747 |
| 30 | 0.5637 | 1.4592 | 0.7748 |

## 4. Stratified Analysis (Top-5 Clusters)

| Cluster | N | OOD% | Cohen's d | p-value |
|---------|---|------|-----------|--------|
| 7 | 1076 | 4.0% | 2.3447 | 3.78e-26 |
| 3 | 1609 | 1.1% | 2.0608 | 2.37e-09 |
| 6 | 1125 | 2.7% | 1.8437 | 8.83e-14 |
| 9 | 1054 | 3.9% | 1.7282 | 1.31e-14 |
| 1 | 1164 | 2.7% | 1.5016 | 3.18e-09 |

## 5. Case Analysis (Top-5 Heterophilic OOD Samples)

### Case 1
- **Text**: how much is an overdraft fee for bank...
- **Heterophily**: 1.0000
- **Avg Neighbor Similarity**: 0.4880
- **Top Neighbors**:
  - sim=0.619: how much over will overdraft protection cover...
  - sim=0.544: do i have overdraft protection...
  - sim=0.430: how much are the foreign transaction fees in brisb...

### Case 2
- **Text**: let me know where jim is right now...
- **Heterophily**: 1.0000
- **Avg Neighbor Similarity**: 0.6042
- **Top Neighbors**:
  - sim=0.638: tell jim i'm coming home soon...
  - sim=0.633: call jim...
  - sim=0.631: when is my meeting with jim scheduled for...

### Case 3
- **Text**: locate jenny at her present position...
- **Heterophily**: 1.0000
- **Avg Neighbor Similarity**: 0.5208
- **Top Neighbors**:
  - sim=0.558: would you give jenny a call...
  - sim=0.528: i would like brenda to have my location...
  - sim=0.520: send sarah my current location...

### Case 4
- **Text**: where's my buddy steve right this second...
- **Heterophily**: 1.0000
- **Avg Neighbor Similarity**: 0.6190
- **Top Neighbors**:
  - sim=0.660: send my current location to steve, please...
  - sim=0.659: steve needs to know my location...
  - sim=0.607: i'd like to send steve my location...

### Case 5
- **Text**: can you give me the gps location of harvey...
- **Heterophily**: 1.0000
- **Avg Neighbor Similarity**: 0.6563
- **Top Neighbors**:
  - sim=0.681: can you share my location with roger...
  - sim=0.657: can you give me my gps coordinates...
  - sim=0.656: please give me my gps coordinates...

## 6. Conclusions and Next Steps

### Hypothesis Verified

The pre-experiment successfully demonstrates that OOD samples exhibit significantly different heterophily patterns in k-NN semantic graphs.

**Next Steps**:
1. Proceed to design heterophily-aware OOD detection methods (Layer 3)
2. Implement 5 proposed methods (NegHetero-OOD, SpectralLLM-OOD, etc.)
3. Run full experiments comparing with SOTA baselines
