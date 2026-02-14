# Retrieval Graph Feature Ablation (v3, Leak-Fixed)

**Fix L2**: Unsupervised z-score fusion (NO OOD labels, NO supervised LR)
**Fix L3**: No StandardScaler fit on test data

## clinc150+all-MiniLM-L6-v2

| Feature | Full AUROC | Near AUROC | Cohen's d |
|---------|------------|------------|----------|
| mean_sim | 0.961 | 0.882 | 2.66 |
| std_sim | 0.321 | 0.481 | -0.62 |
| retrieval_gap | 0.328 | 0.490 | -0.59 |
| sim_drop_rate | 0.397 | 0.544 | -0.36 |
| label_purity | 0.842 | 0.768 | 1.50 |
| knn_only | 0.958 | 0.872 | 2.55 |
| knn_plus_graph_unsupervised | 0.957 | 0.879 | 2.66 |
| knn_plus_purity_unsupervised | 0.932 | 0.851 | 2.28 |

**Graph increment (near AUROC, unsupervised)**: +0.8%
**Graph increment (knn+purity)**: -2.1%

## clinc150+bge-base-en-v1.5

| Feature | Full AUROC | Near AUROC | Cohen's d |
|---------|------------|------------|----------|
| mean_sim | 0.959 | 0.871 | 2.61 |
| std_sim | 0.295 | 0.369 | -0.70 |
| retrieval_gap | 0.298 | 0.376 | -0.69 |
| sim_drop_rate | 0.369 | 0.444 | -0.46 |
| label_purity | 0.873 | 0.810 | 1.69 |
| knn_only | 0.954 | 0.857 | 2.51 |
| knn_plus_graph_unsupervised | 0.965 | 0.894 | 2.78 |
| knn_plus_purity_unsupervised | 0.947 | 0.871 | 2.46 |

**Graph increment (near AUROC, unsupervised)**: +3.7%
**Graph increment (knn+purity)**: +1.5%

## banking77+all-MiniLM-L6-v2

| Feature | Full AUROC | Near AUROC | Cohen's d |
|---------|------------|------------|----------|
| mean_sim | 0.919 | 0.773 | 2.02 |
| std_sim | 0.384 | 0.420 | -0.34 |
| retrieval_gap | 0.390 | 0.429 | -0.28 |
| sim_drop_rate | 0.452 | 0.464 | -0.05 |
| label_purity | 0.804 | 0.623 | 1.30 |
| knn_only | 0.905 | 0.743 | 1.88 |
| knn_plus_graph_unsupervised | 0.925 | 0.790 | 2.10 |
| knn_plus_purity_unsupervised | 0.894 | 0.733 | 1.80 |

**Graph increment (near AUROC, unsupervised)**: +4.7%
**Graph increment (knn+purity)**: -1.1%

## banking77+bge-base-en-v1.5

| Feature | Full AUROC | Near AUROC | Cohen's d |
|---------|------------|------------|----------|
| mean_sim | 0.931 | 0.799 | 2.12 |
| std_sim | 0.323 | 0.395 | -0.55 |
| retrieval_gap | 0.330 | 0.410 | -0.52 |
| sim_drop_rate | 0.397 | 0.455 | -0.27 |
| label_purity | 0.823 | 0.671 | 1.40 |
| knn_only | 0.920 | 0.777 | 1.98 |
| knn_plus_graph_unsupervised | 0.941 | 0.828 | 2.26 |
| knn_plus_purity_unsupervised | 0.915 | 0.788 | 1.93 |

**Graph increment (near AUROC, unsupervised)**: +5.1%
**Graph increment (knn+purity)**: +1.0%

## Banking77(random)+BGE

| Feature | Full AUROC | Near AUROC | Cohen's d |
|---------|------------|------------|----------|
| mean_sim | 0.892 | 0.715 | 1.69 |
| std_sim | 0.338 | 0.496 | -0.62 |
| retrieval_gap | 0.340 | 0.507 | -0.60 |
| sim_drop_rate | 0.407 | 0.544 | -0.36 |
| label_purity | 0.840 | 0.806 | 1.48 |
| knn_only | 0.877 | 0.689 | 1.57 |
| knn_plus_graph_unsupervised | 0.923 | 0.811 | 2.00 |
| knn_plus_purity_unsupervised | 0.910 | 0.829 | 1.88 |

**Graph increment (near AUROC, unsupervised)**: +12.1%
**Graph increment (knn+purity)**: +14.0%

