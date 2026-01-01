"""
Statistical Analysis Utilities for Hypothesis Verification

Provides functions for:
- Correlation analysis (Pearson, Spearman)
- Effect size computation (Cohen's d)
- Divergence metrics (JS, KL)
- Statistical tests (t-test, bootstrap CI)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from scipy.special import rel_entr
import warnings


def compute_pearson_correlation(
    x: Union[List[float], np.ndarray],
    y: Union[List[float], np.ndarray]
) -> Dict[str, float]:
    """
    Compute Pearson correlation coefficient

    Args:
        x: First variable
        y: Second variable

    Returns:
        Dictionary with 'r', 'p_value', 'significant' (at p<0.05)
    """
    x = np.array(x)
    y = np.array(y)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    if len(x) < 3:
        warnings.warn("Sample size < 3, correlation may be unreliable")

    r, p_value = stats.pearsonr(x, y)

    return {
        'r': float(r),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'interpretation': _interpret_correlation(r)
    }


def compute_spearman_correlation(
    x: Union[List[float], np.ndarray],
    y: Union[List[float], np.ndarray]
) -> Dict[str, float]:
    """
    Compute Spearman rank correlation coefficient

    More robust to outliers and non-linear relationships.

    Args:
        x: First variable
        y: Second variable

    Returns:
        Dictionary with 'rho', 'p_value', 'significant'
    """
    x = np.array(x)
    y = np.array(y)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    rho, p_value = stats.spearmanr(x, y)

    return {
        'rho': float(rho),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'interpretation': _interpret_correlation(rho)
    }


def _interpret_correlation(r: float) -> str:
    """Interpret correlation strength"""
    abs_r = abs(r)
    if abs_r >= 0.9:
        strength = "very strong"
    elif abs_r >= 0.7:
        strength = "strong"
    elif abs_r >= 0.5:
        strength = "moderate"
    elif abs_r >= 0.3:
        strength = "weak"
    else:
        strength = "negligible"

    direction = "positive" if r > 0 else "negative"
    return f"{strength} {direction}"


def compute_cohens_d(
    group1: Union[List[float], np.ndarray],
    group2: Union[List[float], np.ndarray],
    pooled: bool = True
) -> Dict[str, float]:
    """
    Compute Cohen's d effect size

    Args:
        group1: First group measurements
        group2: Second group measurements
        pooled: Whether to use pooled standard deviation

    Returns:
        Dictionary with 'd', 'interpretation'
    """
    group1 = np.array(group1)
    group2 = np.array(group2)

    mean1, mean2 = np.mean(group1), np.mean(group2)
    n1, n2 = len(group1), len(group2)

    if pooled:
        # Pooled standard deviation
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        d = (mean1 - mean2) / pooled_std
    else:
        # Simple average of standard deviations
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        d = (mean1 - mean2) / np.sqrt((std1**2 + std2**2) / 2)

    return {
        'd': float(d),
        'abs_d': float(abs(d)),
        'interpretation': _interpret_cohens_d(abs(d)),
        'mean_diff': float(mean1 - mean2),
        'group1_mean': float(mean1),
        'group2_mean': float(mean2)
    }


def _interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size"""
    if d >= 1.2:
        return "very large"
    elif d >= 0.8:
        return "large"
    elif d >= 0.5:
        return "medium"
    elif d >= 0.2:
        return "small"
    else:
        return "negligible"


def compute_js_divergence(
    p: Union[List[float], np.ndarray, Dict[str, float]],
    q: Union[List[float], np.ndarray, Dict[str, float]]
) -> float:
    """
    Compute Jensen-Shannon divergence between two distributions

    Args:
        p: First probability distribution
        q: Second probability distribution

    Returns:
        JS divergence (0 to 1, where 0 = identical)
    """
    # Convert dictionaries to aligned arrays
    if isinstance(p, dict) and isinstance(q, dict):
        all_keys = set(p.keys()) | set(q.keys())
        p_arr = np.array([p.get(k, 0) for k in sorted(all_keys)])
        q_arr = np.array([q.get(k, 0) for k in sorted(all_keys)])
    else:
        p_arr = np.array(p)
        q_arr = np.array(q)

    # Normalize to ensure valid probability distributions
    p_arr = p_arr / (p_arr.sum() + 1e-10)
    q_arr = q_arr / (q_arr.sum() + 1e-10)

    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    p_arr = p_arr + epsilon
    q_arr = q_arr + epsilon
    p_arr = p_arr / p_arr.sum()
    q_arr = q_arr / q_arr.sum()

    # Compute JS divergence
    m = 0.5 * (p_arr + q_arr)
    js = 0.5 * (np.sum(rel_entr(p_arr, m)) + np.sum(rel_entr(q_arr, m)))

    return float(js)


def compute_kl_divergence(
    p: Union[List[float], np.ndarray],
    q: Union[List[float], np.ndarray]
) -> float:
    """
    Compute KL divergence: D_KL(P || Q)

    Args:
        p: True distribution
        q: Approximate distribution

    Returns:
        KL divergence (non-negative)
    """
    p = np.array(p)
    q = np.array(q)

    # Normalize
    p = p / (p.sum() + 1e-10)
    q = q / (q.sum() + 1e-10)

    # Add epsilon to avoid log(0)
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    p = p / p.sum()
    q = q / q.sum()

    return float(np.sum(rel_entr(p, q)))


def bootstrap_confidence_interval(
    data: Union[List[float], np.ndarray],
    statistic: str = 'mean',
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42
) -> Dict[str, float]:
    """
    Compute bootstrap confidence interval

    Args:
        data: Sample data
        statistic: 'mean', 'median', or 'std'
        confidence: Confidence level (default 0.95)
        n_bootstrap: Number of bootstrap samples
        seed: Random seed

    Returns:
        Dictionary with 'estimate', 'ci_lower', 'ci_upper'
    """
    np.random.seed(seed)
    data = np.array(data)
    n = len(data)

    stat_func = {
        'mean': np.mean,
        'median': np.median,
        'std': np.std
    }.get(statistic, np.mean)

    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(stat_func(sample))

    bootstrap_stats = np.array(bootstrap_stats)
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return {
        'estimate': float(stat_func(data)),
        'ci_lower': float(lower),
        'ci_upper': float(upper),
        'confidence': confidence,
        'n_bootstrap': n_bootstrap
    }


def paired_t_test(
    before: Union[List[float], np.ndarray],
    after: Union[List[float], np.ndarray],
    alternative: str = 'two-sided'
) -> Dict[str, float]:
    """
    Perform paired t-test

    Args:
        before: Measurements before treatment
        after: Measurements after treatment
        alternative: 'two-sided', 'less', or 'greater'

    Returns:
        Dictionary with 't_statistic', 'p_value', 'significant'
    """
    before = np.array(before)
    after = np.array(after)

    t_stat, p_value = stats.ttest_rel(before, after, alternative=alternative)

    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'mean_before': float(np.mean(before)),
        'mean_after': float(np.mean(after)),
        'mean_diff': float(np.mean(after - before))
    }


def independent_t_test(
    group1: Union[List[float], np.ndarray],
    group2: Union[List[float], np.ndarray],
    equal_var: bool = False,
    alternative: str = 'two-sided'
) -> Dict[str, float]:
    """
    Perform independent samples t-test (Welch's by default)

    Args:
        group1: First group measurements
        group2: Second group measurements
        equal_var: Assume equal variances (default False = Welch's test)
        alternative: 'two-sided', 'less', or 'greater'

    Returns:
        Dictionary with 't_statistic', 'p_value', 'significant'
    """
    group1 = np.array(group1)
    group2 = np.array(group2)

    t_stat, p_value = stats.ttest_ind(
        group1, group2,
        equal_var=equal_var,
        alternative=alternative
    )

    # Also compute Cohen's d
    cohens = compute_cohens_d(group1, group2)

    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'cohens_d': cohens['d'],
        'group1_mean': float(np.mean(group1)),
        'group2_mean': float(np.mean(group2)),
        'group1_std': float(np.std(group1, ddof=1)),
        'group2_std': float(np.std(group2, ddof=1))
    }


def compute_effect_size_confidence_interval(
    group1: Union[List[float], np.ndarray],
    group2: Union[List[float], np.ndarray],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42
) -> Dict[str, float]:
    """
    Compute confidence interval for Cohen's d using bootstrap

    Args:
        group1: First group
        group2: Second group
        confidence: Confidence level
        n_bootstrap: Number of bootstrap samples
        seed: Random seed

    Returns:
        Dictionary with 'd', 'ci_lower', 'ci_upper'
    """
    np.random.seed(seed)
    group1 = np.array(group1)
    group2 = np.array(group2)

    bootstrap_ds = []
    for _ in range(n_bootstrap):
        sample1 = np.random.choice(group1, size=len(group1), replace=True)
        sample2 = np.random.choice(group2, size=len(group2), replace=True)
        d = compute_cohens_d(sample1, sample2)['d']
        bootstrap_ds.append(d)

    bootstrap_ds = np.array(bootstrap_ds)
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_ds, 100 * alpha / 2)
    upper = np.percentile(bootstrap_ds, 100 * (1 - alpha / 2))

    return {
        'd': compute_cohens_d(group1, group2)['d'],
        'ci_lower': float(lower),
        'ci_upper': float(upper),
        'confidence': confidence
    }


class HypothesisTestResult:
    """Container for hypothesis test results"""

    def __init__(
        self,
        hypothesis_id: str,
        description: str,
        passed: bool,
        statistics: Dict[str, float],
        threshold: Dict[str, float],
        details: str = ""
    ):
        self.hypothesis_id = hypothesis_id
        self.description = description
        self.passed = passed
        self.statistics = statistics
        self.threshold = threshold
        self.details = details

    def __repr__(self) -> str:
        status = "" if self.passed else ""
        return f"HypothesisTestResult({self.hypothesis_id}: {status})"

    def to_dict(self) -> Dict:
        return {
            'hypothesis_id': self.hypothesis_id,
            'description': self.description,
            'passed': self.passed,
            'statistics': self.statistics,
            'threshold': self.threshold,
            'details': self.details
        }

    def summary(self) -> str:
        """Generate human-readable summary"""
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"Hypothesis {self.hypothesis_id}: {status}",
            f"Description: {self.description}",
            "Statistics:"
        ]
        for key, value in self.statistics.items():
            lines.append(f"  - {key}: {value:.4f}")
        lines.append("Thresholds:")
        for key, value in self.threshold.items():
            lines.append(f"  - {key}: {value}")
        if self.details:
            lines.append(f"Details: {self.details}")
        return "\n".join(lines)
