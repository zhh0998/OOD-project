# Baseline Embedding Anisotropy Analysis

## Dataset: CLINC150

- Train ID samples: 15150
- Test ID samples: 5470
- Test OOD samples: 30
  - Near-OOD: 23
  - Medium-OOD: 7
  - Far-OOD: 0

## Anisotropy Metrics

| Metric | Value |
|--------|-------|
| Top-1 variance ratio | 0.0469 |
| Top-5 variance ratio | 0.1541 |
| Top-10 variance ratio | 0.2447 |
| Mean pair cosine sim | 0.0849 |

## OOD Detection Performance (kNN, k=20)

| Metric | Value |
|--------|-------|
| All-OOD AUROC | 0.7483 |
| Near-OOD AUROC | 0.7202 |
| Near-OOD Cohen's d | 0.6266 |
| All-OOD Cohen's d | 0.7512 |
