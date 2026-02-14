"""
Conformal Prediction Module
Implements split conformal prediction for OOD gating decisions.

Output decisions: accept / abstain / route / retrieve-more
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ConformalResult:
    """Result of conformal prediction."""
    decisions: np.ndarray  # 0=accept, 1=abstain, 2=route, 3=retrieve-more
    thresholds: Dict[str, float]
    coverage: float
    risk: float
    utility: float


class ConformalPredictor:
    """
    Split conformal predictor for OOD gating.

    Uses calibration set to determine thresholds that control coverage/risk.
    """

    def __init__(self, alpha: float = 0.10):
        """
        Args:
            alpha: miscoverage rate (e.g., 0.10 for 90% coverage target)
        """
        self.alpha = alpha
        self.threshold_ = None
        self.n_cal_ = None

    def fit(self, scores: np.ndarray, audit_log=None) -> 'ConformalPredictor':
        """
        Fit conformal threshold on calibration scores.

        Args:
            scores: OOD scores from ID-cal set (all should be ID)
            audit_log: LeakageAuditor for logging
        """
        n = len(scores)
        self.n_cal_ = n

        # Compute (1-alpha)(1+1/n) quantile
        q = np.ceil((1 - self.alpha) * (n + 1)) / n
        q = min(q, 1.0)

        self.threshold_ = np.quantile(scores, q)

        if audit_log is not None:
            audit_log.log_fit(
                operation="conformal_threshold",
                fit_split="id_cal",
                n_samples=n,
                additional_info={
                    "alpha": self.alpha,
                    "threshold": float(self.threshold_)
                }
            )

        return self

    def predict(self, scores: np.ndarray,
                abstain_threshold: Optional[float] = None) -> np.ndarray:
        """
        Make gating decisions.

        Args:
            scores: OOD scores for test samples
            abstain_threshold: optional higher threshold for abstain decision

        Returns:
            decisions: 0=accept, 1=abstain
        """
        if self.threshold_ is None:
            raise RuntimeError("Must call fit() before predict()")

        decisions = (scores > self.threshold_).astype(int)
        return decisions

    def get_coverage(self, scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Compute coverage and risk metrics.

        Args:
            scores: OOD scores
            labels: 0 for ID, 1 for OOD

        Returns:
            dict with coverage, risk, utility metrics
        """
        decisions = self.predict(scores)

        # Accept = 0, Abstain = 1
        accepted = decisions == 0
        n_accepted = accepted.sum()
        n_total = len(scores)

        coverage = n_accepted / n_total

        # Risk = fraction of accepted that are OOD
        if n_accepted > 0:
            risk = labels[accepted].sum() / n_accepted
        else:
            risk = 0.0

        # Utility = coverage * (1 - risk)
        utility = coverage * (1 - risk)

        # Conditional metrics
        id_mask = labels == 0
        ood_mask = labels == 1

        id_accept_rate = accepted[id_mask].mean() if id_mask.sum() > 0 else 0.0
        ood_reject_rate = (~accepted[ood_mask]).mean() if ood_mask.sum() > 0 else 0.0

        return {
            'coverage': coverage,
            'risk': risk,
            'utility': utility,
            'id_accept_rate': id_accept_rate,
            'ood_reject_rate': ood_reject_rate,
            'threshold': self.threshold_
        }


class RiskCoverageAnalyzer:
    """
    Analyze risk-coverage tradeoff for different thresholds.
    """

    def __init__(self, alpha_values: List[float] = None):
        if alpha_values is None:
            alpha_values = [0.01, 0.05, 0.10, 0.15, 0.20]
        self.alpha_values = alpha_values

    def compute_curve(self, cal_scores: np.ndarray,
                      test_scores: np.ndarray,
                      test_labels: np.ndarray) -> Dict:
        """
        Compute risk-coverage curve across alpha values.

        Returns:
            dict with curves and AUC-RC
        """
        results = {
            'alpha': [],
            'coverage': [],
            'risk': [],
            'utility': [],
            'auc_rc': None
        }

        for alpha in self.alpha_values:
            cp = ConformalPredictor(alpha=alpha)
            cp.fit(cal_scores)
            metrics = cp.get_coverage(test_scores, test_labels)

            results['alpha'].append(alpha)
            results['coverage'].append(metrics['coverage'])
            results['risk'].append(metrics['risk'])
            results['utility'].append(metrics['utility'])

        # Compute AUC-RC (area under risk-coverage curve)
        # Sort by coverage
        coverages = np.array(results['coverage'])
        risks = np.array(results['risk'])

        sort_idx = np.argsort(coverages)
        sorted_cov = coverages[sort_idx]
        sorted_risk = risks[sort_idx]

        # AUC using trapezoidal rule
        auc_rc = np.trapz(sorted_risk, sorted_cov)
        results['auc_rc'] = auc_rc

        return results

    def compute_abstention_at_target_risk(self, cal_scores: np.ndarray,
                                           test_scores: np.ndarray,
                                           test_labels: np.ndarray,
                                           target_risks: List[float] = None) -> Dict[float, float]:
        """
        Compute abstention rate needed to achieve target risk.

        Returns:
            dict mapping target_risk -> abstention_rate
        """
        if target_risks is None:
            target_risks = [0.01, 0.05, 0.10, 0.20]

        # Sweep thresholds
        thresholds = np.percentile(cal_scores, np.linspace(0, 100, 100))

        results = {}
        for target in target_risks:
            best_abstention = 1.0  # Worst case: abstain all

            for thresh in thresholds:
                accepted = test_scores <= thresh
                n_accepted = accepted.sum()

                if n_accepted == 0:
                    continue

                risk = test_labels[accepted].sum() / n_accepted

                if risk <= target:
                    abstention = 1 - (n_accepted / len(test_scores))
                    best_abstention = min(best_abstention, abstention)

            results[target] = best_abstention

        return results


class GroupConformalPredictor:
    """
    Group/Mondrian conformal prediction for per-group coverage guarantees.
    """

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.group_thresholds_ = {}

    def fit(self, scores: np.ndarray, groups: np.ndarray,
            audit_log=None) -> 'GroupConformalPredictor':
        """
        Fit separate thresholds for each group.

        Args:
            scores: calibration scores
            groups: group assignments
        """
        unique_groups = np.unique(groups)

        for g in unique_groups:
            mask = groups == g
            group_scores = scores[mask]

            if len(group_scores) < 5:
                # Too few samples, use global threshold
                continue

            n = len(group_scores)
            q = np.ceil((1 - self.alpha) * (n + 1)) / n
            q = min(q, 1.0)
            self.group_thresholds_[g] = np.quantile(group_scores, q)

        # Global fallback
        n = len(scores)
        q = np.ceil((1 - self.alpha) * (n + 1)) / n
        self.group_thresholds_['_global'] = np.quantile(scores, min(q, 1.0))

        return self

    def predict(self, scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
        """Make group-conditional predictions."""
        decisions = np.zeros(len(scores), dtype=int)

        for i, (score, group) in enumerate(zip(scores, groups)):
            thresh = self.group_thresholds_.get(group,
                                                 self.group_thresholds_['_global'])
            decisions[i] = int(score > thresh)

        return decisions

    def get_group_coverage(self, scores: np.ndarray, labels: np.ndarray,
                           groups: np.ndarray) -> Dict:
        """Compute per-group coverage metrics."""
        decisions = self.predict(scores, groups)
        unique_groups = np.unique(groups)

        results = {}
        for g in unique_groups:
            mask = groups == g
            group_decisions = decisions[mask]
            group_labels = labels[mask]

            accepted = group_decisions == 0
            n_accepted = accepted.sum()
            n_total = len(group_decisions)

            coverage = n_accepted / n_total if n_total > 0 else 0
            risk = group_labels[accepted].sum() / n_accepted if n_accepted > 0 else 0

            results[g] = {
                'coverage': coverage,
                'risk': risk,
                'n_samples': n_total,
                'coverage_gap': abs(coverage - (1 - self.alpha))
            }

        # Worst group
        worst_coverage_gap = max(r['coverage_gap'] for r in results.values())
        worst_risk = max(r['risk'] for r in results.values())

        results['_worst'] = {
            'worst_coverage_gap': worst_coverage_gap,
            'worst_risk': worst_risk
        }

        return results
