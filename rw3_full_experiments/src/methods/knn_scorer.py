"""
KNN Scorer Module
Computes kNN-based OOD scores.
"""

import numpy as np
from typing import Optional, Tuple, List
import faiss


class KNNScorer:
    """
    kNN-based OOD scorer.

    Higher scores indicate more likely OOD (farther from ID distribution).
    """

    def __init__(self, k: int = 20, distance: str = "cosine"):
        """
        Args:
            k: number of nearest neighbors
            distance: distance metric ("cosine" or "l2")
        """
        self.k = k
        self.distance = distance
        self.index = None
        self.train_embeddings = None
        self.train_labels = None
        self.n_samples_fit_ = None

    def fit(self, X: np.ndarray, labels: Optional[np.ndarray] = None,
            audit_log=None) -> 'KNNScorer':
        """
        Build kNN index on ID-train embeddings.

        Args:
            X: ID-train embeddings (n_samples, dim)
            labels: optional labels for label_purity calculation
            audit_log: LeakageAuditor for logging
        """
        X = np.ascontiguousarray(X.astype('float32'))
        n_samples, dim = X.shape

        if self.distance == "cosine":
            # Normalize for cosine similarity
            faiss.normalize_L2(X)
            # Use inner product index (cosine = normalized IP)
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(X)
        self.train_embeddings = X
        self.train_labels = labels
        self.n_samples_fit_ = n_samples

        if audit_log is not None:
            audit_log.log_fit(
                operation="knn_index",
                fit_split="id_train",
                n_samples=n_samples,
                additional_info={
                    "k": self.k,
                    "distance": self.distance
                }
            )

        return self

    def score(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute kNN-based OOD scores.

        Args:
            X: test embeddings (n_samples, dim)

        Returns:
            scores: OOD scores (higher = more OOD)
            distances: distances to k neighbors (n_samples, k)
            indices: indices of k neighbors (n_samples, k)
        """
        X = np.ascontiguousarray(X.astype('float32'))

        if self.distance == "cosine":
            faiss.normalize_L2(X)

        distances, indices = self.index.search(X, self.k)

        if self.distance == "cosine":
            # Convert similarities to distances (1 - sim)
            distances = 1 - distances

        # OOD score: mean distance to k neighbors
        scores = distances.mean(axis=1)

        return scores, distances, indices

    def get_neighbor_labels(self, indices: np.ndarray) -> Optional[np.ndarray]:
        """Get labels of neighbors."""
        if self.train_labels is None:
            return None
        return self.train_labels[indices]


class KNNScorerVariants:
    """
    Different kNN scoring variants for ablation.
    """

    @staticmethod
    def mean_distance(distances: np.ndarray) -> np.ndarray:
        """Mean distance to k neighbors (default)."""
        return distances.mean(axis=1)

    @staticmethod
    def max_distance(distances: np.ndarray) -> np.ndarray:
        """Max distance to k neighbors."""
        return distances.max(axis=1)

    @staticmethod
    def min_distance(distances: np.ndarray) -> np.ndarray:
        """Min distance to k neighbors (nearest neighbor)."""
        return distances.min(axis=1)

    @staticmethod
    def weighted_mean(distances: np.ndarray, weights: str = "rank") -> np.ndarray:
        """Weighted mean distance (closer neighbors weighted more)."""
        k = distances.shape[1]
        if weights == "rank":
            w = 1.0 / np.arange(1, k + 1)
        elif weights == "exponential":
            w = np.exp(-np.arange(k))
        else:
            w = np.ones(k)
        w = w / w.sum()
        return (distances * w).sum(axis=1)

    @staticmethod
    def harmonic_mean(distances: np.ndarray) -> np.ndarray:
        """Harmonic mean of distances."""
        return distances.shape[1] / (1 / (distances + 1e-10)).sum(axis=1)
