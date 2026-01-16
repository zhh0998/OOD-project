"""
Evaluation metrics for temporal link prediction.
Includes MRR, Hits@K, Cohen's d, and statistical analysis.

Author: RW2 Temporal Network Embedding Project
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from dataclasses import dataclass


def compute_mrr(ranks: np.ndarray) -> float:
    """
    Compute Mean Reciprocal Rank.

    Args:
        ranks: Array of ranks (1-indexed, i.e., best rank is 1)

    Returns:
        MRR score
    """
    return float(np.mean(1.0 / ranks))


def compute_hits_at_k(ranks: np.ndarray, k: int) -> float:
    """
    Compute Hits@K metric.

    Args:
        ranks: Array of ranks (1-indexed)
        k: Cutoff value

    Returns:
        Hits@K score (proportion of ranks <= k)
    """
    return float(np.mean(ranks <= k))


def compute_metrics(
    ranks: np.ndarray,
    ks: List[int] = [1, 3, 10, 50]
) -> Dict[str, float]:
    """
    Compute all standard metrics for link prediction.

    Args:
        ranks: Array of ranks (1-indexed)
        ks: List of k values for Hits@K

    Returns:
        Dictionary with MRR and Hits@K metrics
    """
    metrics = {'mrr': compute_mrr(ranks)}

    for k in ks:
        metrics[f'hits@{k}'] = compute_hits_at_k(ranks, k)

    return metrics


def compute_cohen_d(
    scores1: List[float],
    scores2: List[float]
) -> Tuple[float, str]:
    """
    Compute Cohen's d effect size.

    Args:
        scores1: Baseline model scores (e.g., 5 runs of MRR)
        scores2: Proposed model scores (e.g., 5 runs of MRR)

    Returns:
        cohen_d: Effect size value
        interpretation: String interpretation (negligible/small/medium/large)
    """
    scores1 = np.array(scores1)
    scores2 = np.array(scores2)

    n1, n2 = len(scores1), len(scores2)
    mean1, mean2 = np.mean(scores1), np.mean(scores2)
    var1, var2 = np.var(scores1, ddof=1), np.var(scores2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    # Handle edge case of zero variance
    if pooled_std < 1e-10:
        if abs(mean2 - mean1) < 1e-10:
            return 0.0, "negligible"
        return float('inf'), "large"

    # Cohen's d
    cohen_d = (mean2 - mean1) / pooled_std

    # Interpretation (Cohen's conventions)
    abs_d = abs(cohen_d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return cohen_d, interpretation


@dataclass
class TTestResult:
    """Results from a t-test."""
    t_statistic: float
    p_value: float
    df: int
    significant: bool
    confidence_interval: Tuple[float, float]


def paired_t_test(
    scores1: List[float],
    scores2: List[float],
    alpha: float = 0.05
) -> TTestResult:
    """
    Perform paired t-test for comparing two models.

    Args:
        scores1: Baseline model scores
        scores2: Proposed model scores
        alpha: Significance level

    Returns:
        TTestResult with statistics and interpretation
    """
    scores1 = np.array(scores1)
    scores2 = np.array(scores2)

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(scores2, scores1)

    # Degrees of freedom
    df = len(scores1) - 1

    # Confidence interval for the difference
    diff = scores2 - scores1
    mean_diff = np.mean(diff)
    se_diff = stats.sem(diff)
    ci = stats.t.interval(1 - alpha, df, loc=mean_diff, scale=se_diff)

    return TTestResult(
        t_statistic=float(t_stat),
        p_value=float(p_value),
        df=df,
        significant=p_value < alpha,
        confidence_interval=(float(ci[0]), float(ci[1]))
    )


def independent_t_test(
    scores1: List[float],
    scores2: List[float],
    alpha: float = 0.05
) -> TTestResult:
    """
    Perform independent samples t-test.

    Args:
        scores1: First model scores
        scores2: Second model scores
        alpha: Significance level

    Returns:
        TTestResult with statistics
    """
    scores1 = np.array(scores1)
    scores2 = np.array(scores2)

    # Independent t-test (Welch's t-test for unequal variances)
    t_stat, p_value = stats.ttest_ind(scores2, scores1, equal_var=False)

    # Welch-Satterthwaite degrees of freedom
    n1, n2 = len(scores1), len(scores2)
    var1, var2 = np.var(scores1, ddof=1), np.var(scores2, ddof=1)
    df = ((var1/n1 + var2/n2)**2 /
          ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1)))

    # Confidence interval for the difference in means
    mean_diff = np.mean(scores2) - np.mean(scores1)
    se_diff = np.sqrt(var1/n1 + var2/n2)
    ci = stats.t.interval(1 - alpha, df, loc=mean_diff, scale=se_diff)

    return TTestResult(
        t_statistic=float(t_stat),
        p_value=float(p_value),
        df=int(df),
        significant=p_value < alpha,
        confidence_interval=(float(ci[0]), float(ci[1]))
    )


class StatisticalAnalysis:
    """
    Comprehensive statistical analysis for experiment results.
    """

    def __init__(self, num_runs: int = 5, alpha: float = 0.05):
        self.num_runs = num_runs
        self.alpha = alpha
        self.results: Dict[str, List[Dict[str, float]]] = {}

    def add_result(
        self,
        model_name: str,
        metrics: Dict[str, float],
        run_id: int
    ):
        """Add results from a single run."""
        if model_name not in self.results:
            self.results[model_name] = []
        self.results[model_name].append(metrics)

    def compute_summary(self, model_name: str) -> Dict[str, Tuple[float, float]]:
        """
        Compute mean and std for each metric.

        Returns:
            Dict mapping metric name to (mean, std) tuple
        """
        if model_name not in self.results:
            raise ValueError(f"No results for model {model_name}")

        runs = self.results[model_name]
        metrics = runs[0].keys()

        summary = {}
        for metric in metrics:
            values = [run[metric] for run in runs]
            summary[metric] = (np.mean(values), np.std(values, ddof=1))

        return summary

    def compare_models(
        self,
        baseline_name: str,
        proposed_name: str,
        metric: str = 'mrr'
    ) -> Dict[str, any]:
        """
        Compare two models on a given metric.

        Returns:
            Dict with cohen_d, t_test results, improvement percentage
        """
        baseline_scores = [run[metric] for run in self.results[baseline_name]]
        proposed_scores = [run[metric] for run in self.results[proposed_name]]

        # Cohen's d
        cohen_d, d_interp = compute_cohen_d(baseline_scores, proposed_scores)

        # T-test
        t_test = independent_t_test(baseline_scores, proposed_scores, self.alpha)

        # Improvement percentage
        baseline_mean = np.mean(baseline_scores)
        proposed_mean = np.mean(proposed_scores)
        improvement = ((proposed_mean - baseline_mean) / baseline_mean) * 100

        return {
            'baseline_mean': baseline_mean,
            'baseline_std': np.std(baseline_scores, ddof=1),
            'proposed_mean': proposed_mean,
            'proposed_std': np.std(proposed_scores, ddof=1),
            'improvement_pct': improvement,
            'cohen_d': cohen_d,
            'cohen_d_interpretation': d_interp,
            't_statistic': t_test.t_statistic,
            'p_value': t_test.p_value,
            'significant': t_test.significant,
            'confidence_interval': t_test.confidence_interval
        }

    def generate_latex_table(
        self,
        baseline_name: str,
        models: List[str],
        metrics: List[str] = ['mrr', 'hits@10', 'hits@50']
    ) -> str:
        """Generate LaTeX table for paper submission."""
        lines = []
        lines.append(r"\begin{table}[h]")
        lines.append(r"\centering")
        lines.append(r"\caption{Link Prediction Performance Comparison}")

        # Header
        cols = "l" + "c" * len(metrics) + "cc"
        lines.append(r"\begin{tabular}{" + cols + "}")
        lines.append(r"\toprule")

        header = "Model & " + " & ".join([m.upper() for m in metrics])
        header += r" & Improvement & Cohen's d \\"
        lines.append(header)
        lines.append(r"\midrule")

        # Baseline row
        baseline_summary = self.compute_summary(baseline_name)
        row = f"{baseline_name}"
        for metric in metrics:
            mean, std = baseline_summary[metric]
            row += f" & ${mean:.4f} \\pm {std:.4f}$"
        row += r" & - & - \\"
        lines.append(row)

        lines.append(r"\midrule")

        # Model rows
        for model in models:
            if model == baseline_name:
                continue

            summary = self.compute_summary(model)
            comparison = self.compare_models(baseline_name, model, 'mrr')

            row = f"{model}"
            for metric in metrics:
                mean, std = summary[metric]
                row += f" & ${mean:.4f} \\pm {std:.4f}$"

            imp = comparison['improvement_pct']
            d = comparison['cohen_d']
            row += f" & ${imp:+.2f}\\%$ & ${d:.3f}$"
            row += r" \\"
            lines.append(row)

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\label{tab:results}")
        lines.append(r"\end{table}")

        return "\n".join(lines)

    def generate_markdown_report(
        self,
        baseline_name: str,
        models: List[str]
    ) -> str:
        """Generate markdown report for documentation."""
        lines = []
        lines.append("# Experimental Results\n")

        # Summary table
        lines.append("## Performance Comparison\n")
        lines.append("| Model | MRR | Hits@10 | Improvement | Cohen's d | Significant |")
        lines.append("|-------|-----|---------|-------------|-----------|-------------|")

        for model in models:
            summary = self.compute_summary(model)

            if model == baseline_name:
                mrr_mean, mrr_std = summary['mrr']
                h10_mean, h10_std = summary['hits@10']
                lines.append(
                    f"| {model} (baseline) | {mrr_mean:.4f}±{mrr_std:.4f} | "
                    f"{h10_mean:.4f}±{h10_std:.4f} | - | - | - |"
                )
            else:
                comparison = self.compare_models(baseline_name, model)
                mrr_mean, mrr_std = summary['mrr']
                h10_mean, h10_std = summary['hits@10']
                imp = comparison['improvement_pct']
                d = comparison['cohen_d']
                sig = "Yes" if comparison['significant'] else "No"

                lines.append(
                    f"| **{model}** | {mrr_mean:.4f}±{mrr_std:.4f} | "
                    f"{h10_mean:.4f}±{h10_std:.4f} | {imp:+.2f}% | "
                    f"{d:.3f} ({comparison['cohen_d_interpretation']}) | {sig} |"
                )

        lines.append("")
        return "\n".join(lines)


if __name__ == '__main__':
    # Test metrics
    print("Testing metrics...")

    # Test MRR and Hits@K
    ranks = np.array([1, 2, 3, 5, 10, 100])
    print(f"Ranks: {ranks}")
    print(f"MRR: {compute_mrr(ranks):.4f}")
    print(f"Hits@10: {compute_hits_at_k(ranks, 10):.4f}")
    print(f"All metrics: {compute_metrics(ranks)}")

    # Test Cohen's d
    baseline = [0.740, 0.738, 0.742, 0.739, 0.741]
    proposed = [0.812, 0.809, 0.815, 0.811, 0.813]

    d, interp = compute_cohen_d(baseline, proposed)
    print(f"\nCohen's d: {d:.3f} ({interp})")

    # Test t-test
    t_result = independent_t_test(baseline, proposed)
    print(f"T-test: t={t_result.t_statistic:.3f}, p={t_result.p_value:.6f}")
    print(f"Significant: {t_result.significant}")

    # Test StatisticalAnalysis
    print("\nTesting StatisticalAnalysis...")
    analyzer = StatisticalAnalysis(num_runs=5)

    for i, (b, p) in enumerate(zip(baseline, proposed)):
        analyzer.add_result('NPPCTNE', {'mrr': b, 'hits@10': b + 0.1}, i)
        analyzer.add_result('SSM-Memory-LLM', {'mrr': p, 'hits@10': p + 0.1}, i)

    comparison = analyzer.compare_models('NPPCTNE', 'SSM-Memory-LLM')
    print(f"Comparison: {comparison}")

    report = analyzer.generate_markdown_report('NPPCTNE', ['NPPCTNE', 'SSM-Memory-LLM'])
    print(report)
