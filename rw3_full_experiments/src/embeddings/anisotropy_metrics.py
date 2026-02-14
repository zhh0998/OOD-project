"""
Anisotropy Metrics Module
Computes I-STAR compliant anisotropy metrics.

Key insight from I-STAR (ICLR 2024):
- avg_cosine is misleading - lower isotropy can improve downstream performance
- effective_dim is more informative for understanding embedding geometry
- Our goal is OOD separability, not task performance
"""

import numpy as np
from typing import Dict, Tuple
from scipy.linalg import svd


def compute_anisotropy_metrics(embeddings: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive anisotropy metrics for embeddings.

    Args:
        embeddings: np.ndarray of shape (n_samples, dim)

    Returns:
        dict with metrics:
        - effective_dim: (Σλ_i)² / Σλ_i² (primary metric, I-STAR recommended)
        - top1_variance_ratio: variance explained by first PC
        - topk_cumulative_5: cumulative variance of top 5 PCs
        - topk_cumulative_10: cumulative variance of top 10 PCs
        - condition_number: ratio of max to min singular value
        - avg_cosine: average pairwise cosine similarity (legacy, use with caution)
        - intrinsic_dim: MLE estimate of intrinsic dimensionality
    """
    n_samples, dim = embeddings.shape

    # Center embeddings
    centered = embeddings - embeddings.mean(axis=0)

    # Compute SVD
    try:
        U, s, Vh = svd(centered, full_matrices=False)
    except Exception:
        # Fallback for numerical issues
        cov = np.cov(centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        s = np.sqrt(np.maximum(eigenvalues[::-1], 0))

    # Variance explained
    total_var = np.sum(s ** 2)
    var_ratios = (s ** 2) / total_var if total_var > 0 else np.zeros_like(s)

    # 1. Effective dimension (I-STAR primary metric)
    # (Σλ)² / Σλ² where λ are eigenvalues (s²)
    eigenvalues = s ** 2
    sum_eig = np.sum(eigenvalues)
    sum_eig_sq = np.sum(eigenvalues ** 2)
    effective_dim = (sum_eig ** 2) / sum_eig_sq if sum_eig_sq > 0 else 0

    # 2. Top-1 variance ratio
    top1_variance_ratio = var_ratios[0] if len(var_ratios) > 0 else 0

    # 3. Top-k cumulative variance
    topk_cumulative_5 = np.sum(var_ratios[:5]) if len(var_ratios) >= 5 else np.sum(var_ratios)
    topk_cumulative_10 = np.sum(var_ratios[:10]) if len(var_ratios) >= 10 else np.sum(var_ratios)

    # 4. Condition number
    condition_number = s[0] / s[-1] if s[-1] > 1e-10 else float('inf')

    # 5. Average cosine similarity (legacy metric - I-STAR cautions)
    # Subsample for efficiency
    if n_samples > 1000:
        indices = np.random.choice(n_samples, 1000, replace=False)
        sample = embeddings[indices]
    else:
        sample = embeddings

    # Normalize for cosine
    norms = np.linalg.norm(sample, axis=1, keepdims=True)
    normalized = sample / (norms + 1e-8)

    # Compute mean pairwise cosine (excluding self-similarity)
    sim_matrix = normalized @ normalized.T
    n = len(sim_matrix)
    mask = ~np.eye(n, dtype=bool)
    avg_cosine = sim_matrix[mask].mean()

    # 6. Intrinsic dimensionality (MLE estimate)
    # Using the formula from "Maximum Likelihood Estimation of Intrinsic Dimension"
    k = min(20, n_samples - 1)
    if k > 1:
        # Sort distances to k-nearest neighbors
        dists = np.linalg.norm(embeddings[:, None] - embeddings[None, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        knn_dists = np.sort(dists, axis=1)[:, :k]

        # MLE estimate
        with np.errstate(divide='ignore', invalid='ignore'):
            log_ratios = np.log(knn_dists[:, -1:] / knn_dists[:, :-1])
            log_ratios = log_ratios[np.isfinite(log_ratios)]
            intrinsic_dim = 1.0 / np.mean(log_ratios) if len(log_ratios) > 0 else dim
            intrinsic_dim = min(intrinsic_dim, dim)  # Cap at ambient dim
    else:
        intrinsic_dim = dim

    return {
        'effective_dim': float(effective_dim),
        'top1_variance_ratio': float(top1_variance_ratio),
        'topk_cumulative_5': float(topk_cumulative_5),
        'topk_cumulative_10': float(topk_cumulative_10),
        'condition_number': float(condition_number) if not np.isinf(condition_number) else 1e10,
        'avg_cosine': float(avg_cosine),
        'intrinsic_dim': float(intrinsic_dim)
    }


def compute_anisotropy_table(embeddings_dict: Dict[str, Dict[str, np.ndarray]]) -> Dict:
    """
    Compute anisotropy metrics for all dataset-model combinations.

    Args:
        embeddings_dict: {dataset: {model: {split: embeddings}}}

    Returns:
        Nested dict with anisotropy metrics
    """
    results = {}

    for dataset, models in embeddings_dict.items():
        results[dataset] = {}
        for model, splits in models.items():
            # Use train_id embeddings for anisotropy analysis
            if 'train_id' in splits:
                emb = splits['train_id']
                metrics = compute_anisotropy_metrics(emb)
                results[dataset][model] = metrics
                print(f"{dataset}/{model}: effective_dim={metrics['effective_dim']:.2f}, "
                      f"top1_var={metrics['top1_variance_ratio']:.3f}, "
                      f"avg_cos={metrics['avg_cosine']:.3f} (I-STAR cautions)")

    return results


def diagnose_anisotropy(embeddings: np.ndarray, model_name: str = "") -> str:
    """
    Generate a diagnostic report for embedding anisotropy.

    Returns:
        Markdown-formatted diagnostic report
    """
    metrics = compute_anisotropy_metrics(embeddings)

    # Interpret effective_dim
    dim = embeddings.shape[1]
    eff_dim_ratio = metrics['effective_dim'] / dim

    if eff_dim_ratio < 0.1:
        eff_dim_assessment = "SEVERE geometric collapse - embedding space dominated by few directions"
    elif eff_dim_ratio < 0.3:
        eff_dim_assessment = "MODERATE geometric collapse - significant dimensionality reduction"
    elif eff_dim_ratio < 0.5:
        eff_dim_assessment = "MILD geometric collapse - some directional bias"
    else:
        eff_dim_assessment = "HEALTHY geometry - well-distributed across dimensions"

    # Interpret top-1 variance
    if metrics['top1_variance_ratio'] > 0.5:
        top1_assessment = "CRITICAL: >50% variance in single direction"
    elif metrics['top1_variance_ratio'] > 0.3:
        top1_assessment = "CONCERNING: >30% variance in single direction"
    elif metrics['top1_variance_ratio'] > 0.1:
        top1_assessment = "MODERATE: >10% variance in single direction"
    else:
        top1_assessment = "GOOD: variance well distributed"

    # Whitening recommendation
    if eff_dim_ratio < 0.3 or metrics['top1_variance_ratio'] > 0.2:
        whitening_rec = "RECOMMENDED: Spectral whitening likely to improve OOD separability"
    else:
        whitening_rec = "OPTIONAL: Modern embedding model, whitening may have limited benefit"

    report = f"""
## Anisotropy Diagnostic Report {f'({model_name})' if model_name else ''}

### Core Metrics (I-STAR Compliant)
| Metric | Value | Assessment |
|--------|-------|------------|
| Effective Dimension | {metrics['effective_dim']:.2f} / {dim} ({eff_dim_ratio:.1%}) | {eff_dim_assessment} |
| Top-1 Variance Ratio | {metrics['top1_variance_ratio']:.3f} | {top1_assessment} |
| Top-5 Cumulative Var | {metrics['topk_cumulative_5']:.3f} | - |
| Condition Number | {metrics['condition_number']:.2e} | - |

### Legacy Metric (Use with Caution - I-STAR 2024)
| Metric | Value | Note |
|--------|-------|------|
| Avg Cosine Similarity | {metrics['avg_cosine']:.3f} | I-STAR shows this can be misleading |

### Recommendation
{whitening_rec}

"""
    return report
