"""
Spectral Whitening Module
Implements "All-but-the-Top" whitening for embedding geometric calibration.

Iron Rule: PCA/whitening must be fit ONLY on ID-train data.
"""

import numpy as np
from scipy.linalg import svd
from typing import Optional, Tuple


class SpectralWhitening:
    """
    Spectral whitening for embedding geometric calibration.

    The goal is NOT to improve task performance (I-STAR shows this can hurt).
    The goal IS to improve OOD separability by removing dominant directions
    that cause "distance collapse" in near-OOD detection.
    """

    def __init__(self, k_remove: int = 3, method: str = "all_but_the_top"):
        """
        Args:
            k_remove: number of top principal components to remove (0 = no removal)
            method: whitening method ("all_but_the_top" or "full")
        """
        self.k_remove = k_remove
        self.method = method

        # Fitted parameters (set during fit)
        self.mean_ = None
        self.components_ = None
        self.singular_values_ = None
        self.n_samples_fit_ = None

    def fit(self, X: np.ndarray, audit_log=None) -> 'SpectralWhitening':
        """
        Fit whitening transform on ID-train data.

        IRON RULE: X must be ID-train embeddings only.

        Args:
            X: training embeddings (n_samples, dim)
            audit_log: LeakageAuditor for logging fit operation
        """
        n_samples, dim = X.shape

        # Center using training mean
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # Compute SVD
        U, s, Vh = svd(X_centered, full_matrices=False)

        self.components_ = Vh  # Principal components (rows)
        self.singular_values_ = s
        self.n_samples_fit_ = n_samples

        # Log the fit operation
        if audit_log is not None:
            audit_log.log_fit(
                operation="whitening",
                fit_split="id_train",
                n_samples=n_samples,
                additional_info={
                    "k_remove": self.k_remove,
                    "method": self.method,
                    "mean_source": "id_train_only"
                }
            )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply whitening transform.

        IRON RULE: Uses mean and components from fit (ID-train only).

        Args:
            X: embeddings to transform (n_samples, dim)

        Returns:
            Whitened embeddings
        """
        if self.mean_ is None:
            raise RuntimeError("Must call fit() before transform()")

        # Center using ID-train mean
        X_centered = X - self.mean_

        if self.method == "all_but_the_top":
            # Remove top-k components
            if self.k_remove > 0:
                top_k = self.components_[:self.k_remove]  # (k, dim)
                # Project out top-k directions
                projections = X_centered @ top_k.T  # (n, k)
                X_transformed = X_centered - projections @ top_k  # Remove projections
            else:
                X_transformed = X_centered

        elif self.method == "full":
            # Full whitening (ZCA-style)
            # Transform to PC space, scale by inverse sqrt of variance
            scores = X_centered @ self.components_.T  # (n, dim)
            # Scale each component by 1/sqrt(variance)
            scale = 1.0 / (self.singular_values_ + 1e-10)
            if self.k_remove > 0:
                scale[:self.k_remove] = 0  # Zero out top-k
            scores_scaled = scores * scale
            X_transformed = scores_scaled @ self.components_  # Back to original space

        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Normalize output
        norms = np.linalg.norm(X_transformed, axis=1, keepdims=True)
        X_transformed = X_transformed / (norms + 1e-10)

        return X_transformed

    def fit_transform(self, X: np.ndarray, audit_log=None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, audit_log).transform(X)

    def get_explained_variance_ratio(self) -> np.ndarray:
        """Get variance explained by each component."""
        var = self.singular_values_ ** 2
        return var / var.sum()

    def get_removed_variance_ratio(self) -> float:
        """Get total variance explained by removed components."""
        var_ratios = self.get_explained_variance_ratio()
        return var_ratios[:self.k_remove].sum() if self.k_remove > 0 else 0.0


def compute_cohens_d(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """
    Compute Cohen's d effect size between ID and OOD score distributions.

    Cohen's d > 0.8 is considered "large effect" (Cond1 threshold).
    """
    mean_diff = np.mean(ood_scores) - np.mean(id_scores)
    pooled_std = np.sqrt(
        (np.var(id_scores) * len(id_scores) + np.var(ood_scores) * len(ood_scores)) /
        (len(id_scores) + len(ood_scores))
    )
    return mean_diff / (pooled_std + 1e-10)


def analyze_whitening_effect(embeddings_before: np.ndarray,
                             embeddings_after: np.ndarray,
                             labels: Optional[np.ndarray] = None) -> dict:
    """
    Analyze the effect of whitening on embedding geometry.

    Returns:
        dict with before/after metrics for comparison
    """
    from ..embeddings.anisotropy_metrics import compute_anisotropy_metrics

    metrics_before = compute_anisotropy_metrics(embeddings_before)
    metrics_after = compute_anisotropy_metrics(embeddings_after)

    return {
        'before': metrics_before,
        'after': metrics_after,
        'delta': {
            k: metrics_after[k] - metrics_before[k]
            for k in metrics_before.keys()
        }
    }
