"""
Embedding-Based Baseline Methods
B1-B5: Classic distance/density-based OOD detection.
"""

import numpy as np
from typing import Optional, Tuple
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EmpiricalCovariance


class KNNDistanceBaseline:
    """
    B1: kNN Distance (mean-top-k)
    Simple but strong baseline.
    """

    def __init__(self, k: int = 20):
        self.k = k
        self.train_embeddings = None

    def fit(self, X: np.ndarray, audit_log=None):
        """Fit on ID-train embeddings."""
        self.train_embeddings = X.astype('float32')

        if audit_log is not None:
            audit_log.log_fit(
                operation="B1_knn_distance",
                fit_split="id_train",
                n_samples=len(X)
            )
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute OOD scores (higher = more OOD)."""
        import faiss

        X = X.astype('float32')
        train = self.train_embeddings.copy()

        # Normalize for cosine
        faiss.normalize_L2(X)
        faiss.normalize_L2(train)

        # Build index
        index = faiss.IndexFlatIP(train.shape[1])
        index.add(train)

        # Search
        similarities, _ = index.search(X, self.k)

        # Convert to distances and average
        distances = 1 - similarities
        return distances.mean(axis=1)


class LOFBaseline:
    """
    B2: Local Outlier Factor
    Density-based method.
    """

    def __init__(self, k: int = 20, novelty: bool = True):
        self.k = k
        self.model = LocalOutlierFactor(n_neighbors=k, novelty=novelty)
        self.fitted = False

    def fit(self, X: np.ndarray, audit_log=None):
        """Fit on ID-train embeddings."""
        self.model.fit(X)
        self.fitted = True

        if audit_log is not None:
            audit_log.log_fit(
                operation="B2_lof",
                fit_split="id_train",
                n_samples=len(X)
            )
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute OOD scores (higher = more OOD)."""
        # LOF returns negative scores (more negative = more outlier)
        return -self.model.score_samples(X)


class IsolationForestBaseline:
    """
    B3: Isolation Forest
    Tree-based anomaly detection.
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            random_state=random_state,
            contamination='auto'
        )

    def fit(self, X: np.ndarray, audit_log=None):
        """Fit on ID-train embeddings."""
        self.model.fit(X)

        if audit_log is not None:
            audit_log.log_fit(
                operation="B3_iforest",
                fit_split="id_train",
                n_samples=len(X)
            )
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute OOD scores (higher = more OOD)."""
        # score_samples returns negative anomaly scores
        return -self.model.score_samples(X)


class MahalanobisBaseline:
    """
    B4: Mahalanobis Distance
    Classic statistical method.
    """

    def __init__(self):
        self.mean = None
        self.precision = None

    def fit(self, X: np.ndarray, audit_log=None):
        """Fit mean and covariance on ID-train."""
        self.mean = X.mean(axis=0)

        # Fit covariance with regularization
        cov_estimator = EmpiricalCovariance()
        try:
            cov_estimator.fit(X)
            self.precision = cov_estimator.precision_
        except Exception:
            # Fallback to diagonal covariance
            var = np.var(X, axis=0) + 1e-6
            self.precision = np.diag(1.0 / var)

        if audit_log is not None:
            audit_log.log_fit(
                operation="B4_mahalanobis",
                fit_split="id_train",
                n_samples=len(X)
            )
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distances (higher = more OOD)."""
        centered = X - self.mean
        # M = (x-μ)ᵀ Σ⁻¹ (x-μ)
        mahal = np.sum(centered @ self.precision * centered, axis=1)
        return np.sqrt(mahal)


class CentroidDistanceBaseline:
    """
    B5: Centroid Distance
    Distance to nearest class centroid.
    """

    def __init__(self):
        self.centroids = None

    def fit(self, X: np.ndarray, labels: np.ndarray, audit_log=None):
        """Fit class centroids on ID-train."""
        unique_labels = np.unique(labels)
        self.centroids = {}

        for label in unique_labels:
            mask = labels == label
            self.centroids[label] = X[mask].mean(axis=0)

        if audit_log is not None:
            audit_log.log_fit(
                operation="B5_centroid",
                fit_split="id_train",
                n_samples=len(X),
                additional_info={"n_classes": len(unique_labels)}
            )
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute distance to nearest centroid (higher = more OOD)."""
        centroids = np.array(list(self.centroids.values()))

        # Normalize for cosine distance
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
        C_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)

        # Cosine similarities to all centroids
        sims = X_norm @ C_norm.T

        # Distance = 1 - max_similarity
        return 1 - sims.max(axis=1)
