# Baseline OOD Detection Metrics

**Data Source**: Real CLINC150 and Banking77 datasets

## clinc150_all-MiniLM-L6-v2

### Anisotropy
- Top-1 variance ratio: 0.048
- Mean pair cosine: 0.087
- Anisotropic: False

### Baseline OOD Scores (k=0)
| Score | Full AUROC | Near AUROC | Cohen's d |
|-------|------------|------------|----------|
| knn | 0.958 | 0.872 | 2.55 |
| centroid | 0.793 | 0.612 | 1.15 |
| mahalanobis | 0.895 | 0.763 | 1.82 |

## clinc150_bge-base-en-v1.5

### Anisotropy
- Top-1 variance ratio: 0.055
- Mean pair cosine: 0.481
- Anisotropic: True

### Baseline OOD Scores (k=0)
| Score | Full AUROC | Near AUROC | Cohen's d |
|-------|------------|------------|----------|
| knn | 0.954 | 0.859 | 2.51 |
| centroid | 0.737 | 0.443 | 0.90 |
| mahalanobis | 0.916 | 0.774 | 1.97 |

## banking77_all-MiniLM-L6-v2

### Anisotropy
- Top-1 variance ratio: 0.103
- Mean pair cosine: 0.230
- Anisotropic: False

### Baseline OOD Scores (k=0)
| Score | Full AUROC | Near AUROC | Cohen's d |
|-------|------------|------------|----------|
| knn | 0.905 | 0.738 | 1.88 |
| centroid | 0.632 | 0.591 | 0.43 |
| mahalanobis | 0.821 | 0.637 | 1.22 |

## banking77_bge-base-en-v1.5

### Anisotropy
- Top-1 variance ratio: 0.101
- Mean pair cosine: 0.572
- Anisotropic: True

### Baseline OOD Scores (k=0)
| Score | Full AUROC | Near AUROC | Cohen's d |
|-------|------------|------------|----------|
| knn | 0.919 | 0.771 | 1.97 |
| centroid | 0.563 | 0.469 | 0.11 |
| mahalanobis | 0.810 | 0.588 | 1.17 |

