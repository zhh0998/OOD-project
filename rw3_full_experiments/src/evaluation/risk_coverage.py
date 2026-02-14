"""
Risk-Coverage Evaluation Module
Main decision-layer metrics for the paper.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt


class RiskCoverageEvaluator:
    """
    Evaluator for risk-coverage tradeoff analysis.

    Main metrics (⭐ Table 3 in paper):
    - Risk-Coverage curve + AUC-RC
    - Abstention@TargetRisk
    - Utility = coverage × (1 - risk_among_accepted)
    """

    def __init__(self):
        self.results = {}

    def compute_risk_coverage_curve(self,
                                     scores: np.ndarray,
                                     labels: np.ndarray,
                                     n_thresholds: int = 100) -> Dict:
        """
        Compute full risk-coverage curve.

        Args:
            scores: OOD scores (higher = more OOD)
            labels: binary labels (0=ID, 1=OOD)
            n_thresholds: number of threshold points

        Returns:
            dict with curve data
        """
        thresholds = np.percentile(scores, np.linspace(0, 100, n_thresholds))
        coverages = []
        risks = []
        utilities = []

        for thresh in thresholds:
            accepted = scores <= thresh
            n_accepted = accepted.sum()
            n_total = len(scores)

            coverage = n_accepted / n_total

            if n_accepted > 0:
                risk = labels[accepted].sum() / n_accepted
            else:
                risk = 0.0

            utility = coverage * (1 - risk)

            coverages.append(coverage)
            risks.append(risk)
            utilities.append(utility)

        coverages = np.array(coverages)
        risks = np.array(risks)
        utilities = np.array(utilities)

        # Sort by coverage for proper curve
        sort_idx = np.argsort(coverages)
        coverages = coverages[sort_idx]
        risks = risks[sort_idx]
        utilities = utilities[sort_idx]

        # AUC-RC (lower is better - less risk for given coverage)
        auc_rc = np.trapz(risks, coverages)

        # Optimal operating point (max utility)
        best_idx = np.argmax(utilities)

        return {
            'coverages': coverages,
            'risks': risks,
            'utilities': utilities,
            'thresholds': thresholds[sort_idx],
            'auc_rc': auc_rc,
            'best_coverage': coverages[best_idx],
            'best_risk': risks[best_idx],
            'best_utility': utilities[best_idx]
        }

    def compute_abstention_at_target_risk(self,
                                           scores: np.ndarray,
                                           labels: np.ndarray,
                                           target_risks: List[float] = None) -> Dict[float, float]:
        """
        Compute abstention rate needed to achieve target risk level.

        This is a key metric: "How much do we need to abstain to guarantee X% risk?"

        Args:
            scores: OOD scores
            labels: binary labels
            target_risks: list of target risk levels (e.g., [0.01, 0.05, 0.10, 0.20])

        Returns:
            dict mapping target_risk -> abstention_rate
        """
        if target_risks is None:
            target_risks = [0.01, 0.05, 0.10, 0.20]

        # Sort samples by score (ascending = most confident first)
        sort_idx = np.argsort(scores)
        sorted_labels = labels[sort_idx]

        results = {}
        n_total = len(scores)

        for target in target_risks:
            # Find minimum k such that risk(top k) <= target
            best_k = 0
            for k in range(1, n_total + 1):
                accepted_labels = sorted_labels[:k]
                risk = accepted_labels.sum() / k
                if risk <= target:
                    best_k = k

            abstention = 1 - (best_k / n_total) if best_k > 0 else 1.0
            results[target] = abstention

        return results

    def compute_all_metrics(self,
                            scores: np.ndarray,
                            labels: np.ndarray,
                            cal_scores: Optional[np.ndarray] = None,
                            alpha_values: List[float] = None) -> Dict:
        """
        Compute all risk-coverage metrics.

        Args:
            scores: test OOD scores
            labels: test labels
            cal_scores: calibration scores (for CP thresholds)
            alpha_values: CP alpha values

        Returns:
            Comprehensive metrics dict
        """
        if alpha_values is None:
            alpha_values = [0.01, 0.05, 0.10, 0.15, 0.20]

        results = {}

        # 1. Risk-coverage curve
        rc_curve = self.compute_risk_coverage_curve(scores, labels)
        results['auc_rc'] = rc_curve['auc_rc']
        results['best_utility'] = rc_curve['best_utility']
        results['rc_curve'] = rc_curve

        # 2. Abstention at target risk
        abstention = self.compute_abstention_at_target_risk(scores, labels)
        results['abstention_at_risk'] = abstention

        # 3. CP-based metrics (if calibration scores provided)
        if cal_scores is not None:
            cp_metrics = {}
            for alpha in alpha_values:
                n = len(cal_scores)
                q = np.ceil((1 - alpha) * (n + 1)) / n
                threshold = np.quantile(cal_scores, min(q, 1.0))

                accepted = scores <= threshold
                n_accepted = accepted.sum()
                coverage = n_accepted / len(scores)
                risk = labels[accepted].sum() / n_accepted if n_accepted > 0 else 0

                cp_metrics[f'alpha_{alpha}'] = {
                    'coverage': coverage,
                    'risk': risk,
                    'utility': coverage * (1 - risk),
                    'threshold': threshold
                }
            results['cp_metrics'] = cp_metrics

        return results

    def plot_risk_coverage_curve(self,
                                  results_dict: Dict[str, Dict],
                                  title: str = "Risk-Coverage Curve",
                                  save_path: Optional[str] = None):
        """
        Plot risk-coverage curves for multiple methods.

        Args:
            results_dict: {method_name: metrics_dict}
            title: plot title
            save_path: optional path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Risk vs Coverage
        ax1 = axes[0]
        for name, metrics in results_dict.items():
            if 'rc_curve' in metrics:
                curve = metrics['rc_curve']
                ax1.plot(curve['coverages'], curve['risks'], label=f"{name} (AUC-RC={curve['auc_rc']:.3f})")

        ax1.set_xlabel('Coverage')
        ax1.set_ylabel('Risk')
        ax1.set_title('Risk vs Coverage')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Utility vs Coverage
        ax2 = axes[1]
        for name, metrics in results_dict.items():
            if 'rc_curve' in metrics:
                curve = metrics['rc_curve']
                ax2.plot(curve['coverages'], curve['utilities'], label=name)

        ax2.set_xlabel('Coverage')
        ax2.set_ylabel('Utility')
        ax2.set_title('Utility vs Coverage')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved figure to {save_path}")

        plt.close()
        return fig

    def plot_abstention_comparison(self,
                                    results_dict: Dict[str, Dict],
                                    target_risks: List[float] = None,
                                    save_path: Optional[str] = None):
        """
        Plot abstention rates at different target risk levels.
        """
        if target_risks is None:
            target_risks = [0.01, 0.05, 0.10, 0.20]

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(target_risks))
        width = 0.8 / len(results_dict)

        for i, (name, metrics) in enumerate(results_dict.items()):
            if 'abstention_at_risk' in metrics:
                abstentions = [metrics['abstention_at_risk'].get(r, 1.0) for r in target_risks]
                ax.bar(x + i * width, abstentions, width, label=name)

        ax.set_xlabel('Target Risk')
        ax.set_ylabel('Abstention Rate')
        ax.set_title('Abstention Rate to Achieve Target Risk')
        ax.set_xticks(x + width * (len(results_dict) - 1) / 2)
        ax.set_xticklabels([f'{r:.0%}' for r in target_risks])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved figure to {save_path}")

        plt.close()
        return fig
