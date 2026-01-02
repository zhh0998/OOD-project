# RW3 Main Experiments - Summary Report

**Generated**: 2026-01-02 16:34:15

---

## Overall Performance

### AUROC (%) Comparison

| Method | clinc150 | banking77 | snips | Avg |
|--------|--------|--------|--------|--------|
| CosineDistance | 78.66 | 100.00 | 100.00 | **92.89** |
| Heterophily-Simple | 71.63 | 100.00 | 100.00 | **90.54** |
| KNN | 66.51 | 100.00 | 100.00 | **88.84** |
| KNN-10 | 70.34 | 100.00 | 100.00 | **90.11** |
| LOF | 86.29 | 100.00 | 88.33 | **91.54** |
| Mahalanobis | 81.11 | 100.00 | 100.00 | **93.70** |
| NegHetero-OOD | 70.34 | 100.00 | 100.00 | **90.11** |

---

## Key Findings

1. **Best Method**: Mahalanobis (Avg AUROC: 93.70%)
2. **NegHetero-OOD Performance**: 90.11% avg AUROC

### Per-Dataset Best Methods

- **clinc150**: LOF (86.29%)
- **banking77**: Mahalanobis (100.00%)
- **snips**: Mahalanobis (100.00%)

---

## Next Steps

1. Run statistical significance tests
2. Complete ablation studies
3. Generate visualizations for paper

---

**Report End**
