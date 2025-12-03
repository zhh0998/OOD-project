# Causal TPP Validation - Statistical Results Table

## Experiment Configuration
| Parameter | Value |
|-----------|-------|
| Number of runs | 5 |
| Events per run | ~900-1000 |
| Event types | A, B, C |
| True causal graph | A → B → C |
| Decay parameter (β) | 1.0 |
| Random seed | 42 |

## Per-Run Results

### Standard Hawkes Estimation

| Run | MSE | A→B Est | B→C Est | A→C Est (spurious) | Precision | Recall | F1 |
|-----|-----|---------|---------|---------------------|-----------|--------|-----|
| 1 | 0.0403 | 0.7478 | 0.6916 | **0.5749** | 0.500 | 1.000 | 0.667 |
| 2 | 0.0930 | 0.7624 | 0.9053 | **0.7361** | 0.400 | 1.000 | 0.571 |
| 3 | 0.0755 | 0.7073 | 0.7935 | **0.6094** | 0.333 | 1.000 | 0.500 |
| 4 | 0.0534 | 0.6485 | 0.7018 | **0.4752** | 0.333 | 1.000 | 0.500 |
| 5 | 0.0433 | 0.8349 | 0.7585 | **0.5495** | 0.400 | 1.000 | 0.571 |
| **Mean** | **0.0611** | 0.740 | 0.770 | **0.589** | 0.393 | 1.000 | 0.562 |
| **Std** | 0.0202 | 0.066 | 0.079 | 0.086 | 0.061 | 0.000 | 0.061 |

### Causal TPP Estimation

| Run | MSE | A→B Est | B→C Est | A→C Est (correct) | Precision | Recall | F1 |
|-----|-----|---------|---------|-------------------|-----------|--------|-----|
| 1 | 0.0155 | 0.4720 | 0.4203 | **0.0000** | 1.000 | 1.000 | 1.000 |
| 2 | 0.0120 | 0.5131 | 0.4387 | **0.0000** | 1.000 | 1.000 | 1.000 |
| 3 | 0.0131 | 0.4937 | 0.4435 | **0.0000** | 1.000 | 1.000 | 1.000 |
| 4 | 0.0152 | 0.4801 | 0.4151 | **0.0000** | 1.000 | 1.000 | 1.000 |
| 5 | 0.0104 | 0.5421 | 0.4361 | **0.0000** | 1.000 | 1.000 | 1.000 |
| **Mean** | **0.0133** | 0.500 | 0.431 | **0.000** | 1.000 | 1.000 | 1.000 |
| **Std** | 0.0019 | 0.028 | 0.012 | 0.000 | 0.000 | 0.000 | 0.000 |

## Summary Statistics

| Metric | Standard Hawkes | Causal TPP | Improvement |
|--------|-----------------|------------|-------------|
| MSE (mean ± std) | 0.0611 ± 0.0202 | 0.0133 ± 0.0019 | **78.3% reduction** |
| A→C Estimate | 0.5890 ± 0.0857 | 0.0000 ± 0.0000 | **100% error eliminated** |
| F1 Score | 0.562 ± 0.061 | 1.000 ± 0.000 | **+78% improvement** |
| Precision | 0.393 ± 0.061 | 1.000 ± 0.000 | **+155% improvement** |

## Decision Criteria Evaluation

| Criterion | Threshold | Result | Status |
|-----------|-----------|--------|--------|
| MSE Reduction | > 50% | 78.3% | ✅ PASS |
| A→C (Causal TPP) | < 0.1 | 0.0000 | ✅ PASS |
| A→C (Hawkes) | > 0.3 | 0.5890 | ✅ Spurious detected |
| F1 Score (Causal) | > 0.8 | 1.0000 | ✅ PASS |

## Key Insight

**Standard Hawkes confuses correlation with causation:**
- The causal chain A → B → C creates temporal correlation between A and C
- Standard Hawkes incorrectly estimates A→C ≈ 0.59 (should be 0)
- This is a **false positive** for the non-existent direct edge

**Causal TPP correctly identifies the structure:**
- Uses back-door criterion to control for confounders
- Correctly estimates A→C = 0.00
- Perfect structural recovery (F1 = 1.0)

## True vs Estimated Influence Matrices

### Ground Truth (α_true)
```
        A      B      C
A →  [0.00   0.80   0.00]
B →  [0.00   0.00   0.60]
C →  [0.00   0.00   0.00]
```

### Standard Hawkes (α_hawkes, averaged)
```
        A      B      C
A →  [0.04   0.74   0.59]  ← A→C is WRONG!
B →  [0.01   0.12   0.77]
C →  [0.02   0.08   0.31]
```

### Causal TPP (α_causal, averaged)
```
        A      B      C
A →  [0.00   0.50   0.00]  ← A→C is CORRECT!
B →  [0.00   0.00   0.43]
C →  [0.00   0.00   0.00]
```

## Conclusion

**✅ PROCEED with Causal TPP research**

The validation experiment demonstrates that the causal TPP approach:
1. Achieves 78.3% MSE reduction over standard Hawkes
2. Perfectly identifies true causal structure (F1 = 1.0)
3. Correctly rejects spurious correlation-based edges
4. Is robust across multiple runs (low variance)

This strongly supports further development of causal intensity functions for temporal point processes.
