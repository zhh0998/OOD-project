"""
Unsupervised Fusion Module
Combines kNN scores with graph features WITHOUT using OOD labels.

Iron Rule: No OOD labels used for training any fusion model.
Allowed methods: z-score, rank-average, Fisher p-value, weighted sum (val-tuned).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats


class UnsupervisedFusion:
    """
    Unsupervised fusion of OOD scores and graph features.

    All fusion methods use statistics from ID-cal only (no OOD labels).
    """

    def __init__(self, method: str = "zscore_mean"):
        """
        Args:
            method: fusion method
                - "zscore_mean": z-score normalize each feature, average
                - "rank_average": rank-based aggregation
                - "fisher_pvalue": Fisher's method for combining p-values
                - "weighted_sum": weighted sum (weights from validation, no OOD labels)
        """
        self.method = method
        self.stats_ = {}  # Fitted statistics
        self.weights_ = None

    def fit(self, knn_scores: np.ndarray,
            graph_features: Dict[str, np.ndarray],
            audit_log=None) -> 'UnsupervisedFusion':
        """
        Fit fusion parameters on ID-cal data.

        Args:
            knn_scores: kNN OOD scores from ID-cal
            graph_features: graph features from ID-cal
            audit_log: LeakageAuditor for logging
        """
        # Compute statistics for z-score normalization
        self.stats_['knn'] = {
            'mean': np.mean(knn_scores),
            'std': np.std(knn_scores) + 1e-10
        }

        for name, values in graph_features.items():
            self.stats_[name] = {
                'mean': np.mean(values),
                'std': np.std(values) + 1e-10
            }

        # For weighted_sum, use equal weights (no OOD labels available)
        n_features = 1 + len(graph_features)
        self.weights_ = np.ones(n_features) / n_features

        if audit_log is not None:
            audit_log.log_fit(
                operation="fusion",
                fit_split="id_cal",
                n_samples=len(knn_scores),
                additional_info={
                    "method": self.method,
                    "n_features": n_features
                }
            )
            audit_log.validate_no_ood_labels_in_fusion(self.method)

        return self

    def transform(self, knn_scores: np.ndarray,
                  graph_features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Apply fusion to compute final OOD scores.

        Args:
            knn_scores: kNN OOD scores
            graph_features: graph features

        Returns:
            Fused OOD scores (higher = more OOD)
        """
        if self.method == "zscore_mean":
            return self._zscore_mean(knn_scores, graph_features)
        elif self.method == "rank_average":
            return self._rank_average(knn_scores, graph_features)
        elif self.method == "fisher_pvalue":
            return self._fisher_pvalue(knn_scores, graph_features)
        elif self.method == "weighted_sum":
            return self._weighted_sum(knn_scores, graph_features)
        else:
            raise ValueError(f"Unknown fusion method: {self.method}")

    def _zscore_mean(self, knn_scores: np.ndarray,
                     graph_features: Dict[str, np.ndarray]) -> np.ndarray:
        """Z-score normalize each feature, then average."""
        normalized = []

        # kNN scores
        z_knn = (knn_scores - self.stats_['knn']['mean']) / self.stats_['knn']['std']
        normalized.append(z_knn)

        # Graph features (need to flip sign for some features)
        for name, values in graph_features.items():
            z_feat = (values - self.stats_[name]['mean']) / self.stats_[name]['std']

            # label_purity and cluster_agreement: lower = more OOD, flip sign
            # retrieval_gap: lower = more OOD, flip sign
            # sim_drop_rate: higher = more OOD, keep sign
            # neighbor_variance: higher = more OOD, keep sign
            if name in ['label_purity', 'cluster_agreement', 'retrieval_gap']:
                z_feat = -z_feat

            normalized.append(z_feat)

        # Average
        stacked = np.stack(normalized, axis=1)
        return np.mean(stacked, axis=1)

    def _rank_average(self, knn_scores: np.ndarray,
                      graph_features: Dict[str, np.ndarray]) -> np.ndarray:
        """Rank-based aggregation (robust to outliers)."""
        n = len(knn_scores)
        ranks = []

        # kNN ranks
        ranks.append(stats.rankdata(knn_scores) / n)

        # Graph feature ranks
        for name, values in graph_features.items():
            # Flip ranking for features where lower = more OOD
            if name in ['label_purity', 'cluster_agreement', 'retrieval_gap']:
                r = stats.rankdata(-values) / n
            else:
                r = stats.rankdata(values) / n
            ranks.append(r)

        stacked = np.stack(ranks, axis=1)
        return np.mean(stacked, axis=1)

    def _fisher_pvalue(self, knn_scores: np.ndarray,
                       graph_features: Dict[str, np.ndarray]) -> np.ndarray:
        """Fisher's method for combining p-values."""
        # Convert scores to p-values (using empirical CDF from calibration)
        pvalues = []

        # kNN p-values
        p_knn = self._score_to_pvalue(knn_scores, self.stats_['knn']['mean'],
                                       self.stats_['knn']['std'])
        pvalues.append(p_knn)

        # Graph feature p-values
        for name, values in graph_features.items():
            if name in ['label_purity', 'cluster_agreement', 'retrieval_gap']:
                # Lower = more OOD, use lower tail
                p = self._score_to_pvalue(-values, -self.stats_[name]['mean'],
                                           self.stats_[name]['std'])
            else:
                p = self._score_to_pvalue(values, self.stats_[name]['mean'],
                                           self.stats_[name]['std'])
            pvalues.append(p)

        # Fisher's combined test statistic: -2 * sum(log(p))
        stacked = np.stack(pvalues, axis=1)
        stacked = np.clip(stacked, 1e-10, 1 - 1e-10)
        fisher_stat = -2 * np.sum(np.log(stacked), axis=1)

        return fisher_stat

    def _weighted_sum(self, knn_scores: np.ndarray,
                      graph_features: Dict[str, np.ndarray]) -> np.ndarray:
        """Weighted sum with pre-defined weights (no OOD labels used)."""
        normalized = []

        # kNN scores
        z_knn = (knn_scores - self.stats_['knn']['mean']) / self.stats_['knn']['std']
        normalized.append(z_knn)

        # Graph features
        for name, values in graph_features.items():
            z_feat = (values - self.stats_[name]['mean']) / self.stats_[name]['std']
            if name in ['label_purity', 'cluster_agreement', 'retrieval_gap']:
                z_feat = -z_feat
            normalized.append(z_feat)

        stacked = np.stack(normalized, axis=1)
        return np.dot(stacked, self.weights_)

    def _score_to_pvalue(self, scores: np.ndarray, mean: float, std: float) -> np.ndarray:
        """Convert scores to p-values using normal approximation."""
        z = (scores - mean) / std
        return 1 - stats.norm.cdf(z)


class FusionAblation:
    """Helper for fusion method ablation studies."""

    METHODS = ["zscore_mean", "rank_average", "fisher_pvalue", "weighted_sum"]

    @staticmethod
    def run_all_methods(knn_scores_cal: np.ndarray,
                        graph_features_cal: Dict[str, np.ndarray],
                        knn_scores_test: np.ndarray,
                        graph_features_test: Dict[str, np.ndarray],
                        audit_log=None) -> Dict[str, np.ndarray]:
        """Run all fusion methods and return scores."""
        results = {}

        for method in FusionAblation.METHODS:
            fusion = UnsupervisedFusion(method=method)
            fusion.fit(knn_scores_cal, graph_features_cal, audit_log)
            scores = fusion.transform(knn_scores_test, graph_features_test)
            results[method] = scores

        return results
