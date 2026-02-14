"""
Evaluation Metrics Module
Computes OOD detection metrics with bootstrap confidence intervals.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


def compute_ood_metrics(scores: np.ndarray,
                        labels: np.ndarray,
                        ood_groups: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute standard OOD detection metrics.

    Args:
        scores: OOD scores (higher = more OOD)
        labels: binary labels (0 = ID, 1 = OOD)
        ood_groups: optional OOD group labels (near/medium/far)

    Returns:
        dict with metrics
    """
    # Basic metrics
    results = {}

    # Overall AUROC
    try:
        results['auroc'] = roc_auc_score(labels, scores)
    except ValueError:
        results['auroc'] = 0.5

    # FPR@95TPR (False Positive Rate when True Positive Rate is 95%)
    results['fpr95'] = compute_fpr_at_tpr(scores, labels, target_tpr=0.95)

    # AUPR-In (ID as positive class)
    try:
        precision, recall, _ = precision_recall_curve(1 - labels, -scores)
        results['aupr_in'] = auc(recall, precision)
    except ValueError:
        results['aupr_in'] = 0.5

    # AUPR-Out (OOD as positive class)
    try:
        precision, recall, _ = precision_recall_curve(labels, scores)
        results['aupr_out'] = auc(recall, precision)
    except ValueError:
        results['aupr_out'] = 0.5

    # Cohen's d
    id_scores = scores[labels == 0]
    ood_scores = scores[labels == 1]
    results['cohens_d'] = compute_cohens_d(id_scores, ood_scores)

    # Per-group metrics
    if ood_groups is not None:
        unique_groups = [g for g in ['near', 'medium', 'far'] if g in ood_groups]
        for group in unique_groups:
            group_mask = ood_groups == group
            if group_mask.sum() > 0:
                # Combine ID with this OOD group
                combined_labels = np.concatenate([
                    np.zeros(np.sum(labels == 0)),
                    np.ones(np.sum(group_mask))
                ])
                combined_scores = np.concatenate([
                    scores[labels == 0],
                    scores[group_mask]
                ])
                try:
                    results[f'auroc_{group}'] = roc_auc_score(combined_labels, combined_scores)
                except ValueError:
                    results[f'auroc_{group}'] = 0.5

                # Cohen's d for this group
                results[f'cohens_d_{group}'] = compute_cohens_d(
                    id_scores, scores[group_mask]
                )

    return results


def compute_fpr_at_tpr(scores: np.ndarray, labels: np.ndarray,
                       target_tpr: float = 0.95) -> float:
    """Compute FPR when TPR is at target level."""
    ood_scores = scores[labels == 1]
    id_scores = scores[labels == 0]

    if len(ood_scores) == 0 or len(id_scores) == 0:
        return 1.0

    # Find threshold where TPR = target_tpr
    threshold = np.percentile(ood_scores, 100 * (1 - target_tpr))

    # FPR at this threshold
    fpr = np.mean(id_scores > threshold)
    return fpr


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return 0.0

    mean_diff = np.mean(group2) - np.mean(group1)
    pooled_var = ((n1 - 1) * np.var(group1, ddof=1) +
                  (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var) if pooled_var > 0 else 1e-10

    return mean_diff / pooled_std


def compute_bootstrap_ci(scores: np.ndarray,
                         labels: np.ndarray,
                         metric_fn,
                         n_iter: int = 1000,
                         confidence: float = 0.95,
                         stratified: bool = True,
                         seed: int = 42) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Args:
        scores: OOD scores
        labels: binary labels
        metric_fn: function(scores, labels) -> float
        n_iter: number of bootstrap iterations
        confidence: confidence level
        stratified: whether to stratify by label

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    np.random.seed(seed)
    n = len(scores)
    bootstrap_values = []

    for _ in range(n_iter):
        if stratified:
            # Stratified bootstrap
            id_idx = np.where(labels == 0)[0]
            ood_idx = np.where(labels == 1)[0]

            boot_id = np.random.choice(id_idx, len(id_idx), replace=True)
            boot_ood = np.random.choice(ood_idx, len(ood_idx), replace=True)
            boot_idx = np.concatenate([boot_id, boot_ood])
        else:
            boot_idx = np.random.choice(n, n, replace=True)

        boot_scores = scores[boot_idx]
        boot_labels = labels[boot_idx]

        try:
            value = metric_fn(boot_scores, boot_labels)
            bootstrap_values.append(value)
        except Exception:
            continue

    if len(bootstrap_values) == 0:
        return 0.0, 0.0, 0.0

    bootstrap_values = np.array(bootstrap_values)
    point = np.mean(bootstrap_values)
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_values, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_values, 100 * (1 - alpha / 2))

    return point, ci_lower, ci_upper


def compute_all_metrics_with_ci(scores: np.ndarray,
                                labels: np.ndarray,
                                ood_groups: Optional[np.ndarray] = None,
                                n_bootstrap: int = 1000,
                                confidence: float = 0.95) -> Dict:
    """
    Compute all metrics with bootstrap confidence intervals.

    Returns:
        dict with metrics and CIs
    """
    results = {}

    # Point estimates
    point_metrics = compute_ood_metrics(scores, labels, ood_groups)

    # Bootstrap CIs for key metrics
    for metric_name in ['auroc', 'fpr95', 'aupr_out']:
        if metric_name == 'auroc':
            metric_fn = lambda s, l: roc_auc_score(l, s)
        elif metric_name == 'fpr95':
            metric_fn = lambda s, l: compute_fpr_at_tpr(s, l, 0.95)
        elif metric_name == 'aupr_out':
            def metric_fn(s, l):
                precision, recall, _ = precision_recall_curve(l, s)
                return auc(recall, precision)

        try:
            point, ci_lo, ci_hi = compute_bootstrap_ci(
                scores, labels, metric_fn, n_bootstrap, confidence
            )
            results[metric_name] = {
                'value': point_metrics[metric_name],
                'ci_lower': ci_lo,
                'ci_upper': ci_hi,
                'ci_width': ci_hi - ci_lo
            }
        except Exception:
            results[metric_name] = {
                'value': point_metrics[metric_name],
                'ci_lower': None,
                'ci_upper': None,
                'ci_width': None
            }

    # Cohen's d (no bootstrap, just point estimate)
    results['cohens_d'] = {'value': point_metrics['cohens_d']}

    # Per-group metrics
    if ood_groups is not None:
        for key in point_metrics:
            if key.startswith('auroc_') or key.startswith('cohens_d_'):
                results[key] = {'value': point_metrics[key]}

    return results


def compute_worst_slice_metrics(scores: np.ndarray,
                                labels: np.ndarray,
                                slices: np.ndarray,
                                cp_threshold: float) -> Dict:
    """
    Compute worst-slice metrics for conditional coverage analysis.

    Args:
        scores: OOD scores
        labels: binary labels (0=ID, 1=OOD)
        slices: slice assignments (near/medium/far or domain labels)
        cp_threshold: conformal prediction threshold

    Returns:
        dict with per-slice and worst-slice metrics
    """
    decisions = (scores > cp_threshold).astype(int)  # 1 = reject

    unique_slices = np.unique(slices)
    results = {'per_slice': {}}

    for s in unique_slices:
        mask = slices == s
        slice_decisions = decisions[mask]
        slice_labels = labels[mask]

        n_total = len(slice_decisions)
        n_accepted = (slice_decisions == 0).sum()

        coverage = n_accepted / n_total if n_total > 0 else 0

        # Risk among accepted
        if n_accepted > 0:
            accepted_mask = slice_decisions == 0
            risk = slice_labels[accepted_mask].sum() / n_accepted
        else:
            risk = 0

        results['per_slice'][s] = {
            'coverage': coverage,
            'risk': risk,
            'n_samples': n_total
        }

    # Worst slice metrics
    coverages = [r['coverage'] for r in results['per_slice'].values()]
    risks = [r['risk'] for r in results['per_slice'].values()]

    results['worst_coverage'] = min(coverages) if coverages else 0
    results['worst_risk'] = max(risks) if risks else 0
    results['coverage_range'] = max(coverages) - min(coverages) if coverages else 0

    return results
