# Retrieval Graph Feature Ablation

## clinc150_all-MiniLM-L6-v2

| Feature | Full AUROC | Near AUROC | Cohen's d |
|---------|------------|------------|----------|
| mean_sim | 0.961 | 0.882 | 2.66 |
| std_sim | 0.321 | 0.481 | -0.62 |
| retrieval_gap | 0.328 | 0.490 | -0.59 |
| sim_drop_rate | 0.397 | 0.544 | -0.36 |
| label_purity | 0.842 | 0.768 | 1.50 |
| knn_only | 0.958 | 0.872 | 2.55 |
| knn_plus_graph | 0.964 | 0.886 | 1.93 |
| knn_plus_purity | 0.959 | 0.873 | 1.39 |

**Graph increment (near AUROC)**: +1.4%

## clinc150_bge-base-en-v1.5

| Feature | Full AUROC | Near AUROC | Cohen's d |
|---------|------------|------------|----------|
| mean_sim | 0.959 | 0.874 | 2.62 |
| std_sim | 0.287 | 0.367 | -0.72 |
| retrieval_gap | 0.292 | 0.377 | -0.71 |
| sim_drop_rate | 0.363 | 0.435 | -0.48 |
| label_purity | 0.874 | 0.812 | 1.69 |
| knn_only | 0.954 | 0.859 | 2.51 |
| knn_plus_graph | 0.968 | 0.901 | 1.93 |
| knn_plus_purity | 0.959 | 0.878 | 1.19 |

**Graph increment (near AUROC)**: +4.2%

## banking77_all-MiniLM-L6-v2

| Feature | Full AUROC | Near AUROC | Cohen's d |
|---------|------------|------------|----------|
| mean_sim | 0.920 | 0.773 | 2.02 |
| std_sim | 0.367 | 0.416 | -0.39 |
| retrieval_gap | 0.371 | 0.421 | -0.33 |
| sim_drop_rate | 0.442 | 0.474 | -0.07 |
| label_purity | 0.789 | 0.606 | 1.21 |
| knn_only | 0.905 | 0.738 | 1.88 |
| knn_plus_graph | 0.937 | 0.818 | 1.65 |
| knn_plus_purity | 0.905 | 0.738 | 1.18 |

**Graph increment (near AUROC)**: +8.1%

## banking77_bge-base-en-v1.5

| Feature | Full AUROC | Near AUROC | Cohen's d |
|---------|------------|------------|----------|
| mean_sim | 0.931 | 0.796 | 2.11 |
| std_sim | 0.326 | 0.375 | -0.56 |
| retrieval_gap | 0.333 | 0.391 | -0.52 |
| sim_drop_rate | 0.411 | 0.463 | -0.24 |
| label_purity | 0.817 | 0.673 | 1.37 |
| knn_only | 0.919 | 0.771 | 1.97 |
| knn_plus_graph | 0.948 | 0.844 | 1.80 |
| knn_plus_purity | 0.925 | 0.784 | 1.24 |

**Graph increment (near AUROC)**: +7.3%

