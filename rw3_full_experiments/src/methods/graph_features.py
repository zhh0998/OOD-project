"""
Graph Features Module
Extracts retrieval neighborhood graph features for OOD detection.

Features:
1. label_purity: how consistent are neighbor labels
2. retrieval_gap: gap between top-1 and top-2 neighbor distances
3. sim_drop_rate: rate of similarity decay across neighbors
4. neighbor_variance: variance in neighbor distances
5. cluster_agreement: agreement with cluster assignments
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter


class GraphFeatureExtractor:
    """
    Extract graph-based features from kNN retrieval neighborhoods.

    These features capture local graph structure that indicates OOD-ness
    without requiring OOD labels for training.
    """

    def __init__(self, topK: int = 20):
        """
        Args:
            topK: number of neighbors to consider
        """
        self.topK = topK
        self.feature_names = [
            "label_purity",
            "retrieval_gap",
            "sim_drop_rate",
            "neighbor_variance",
            "cluster_agreement"
        ]

    def extract_label_purity(self, neighbor_labels: np.ndarray) -> np.ndarray:
        """
        Compute label purity: fraction of neighbors with most common label.

        For ID samples, neighbors tend to have same/similar labels.
        For OOD samples, neighbors have diverse labels.
        """
        n_samples = neighbor_labels.shape[0]
        purities = np.zeros(n_samples)

        for i in range(n_samples):
            labels = neighbor_labels[i]
            if len(labels) == 0:
                purities[i] = 0
            else:
                counter = Counter(labels)
                most_common_count = counter.most_common(1)[0][1]
                purities[i] = most_common_count / len(labels)

        return purities

    def extract_retrieval_gap(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute retrieval gap: normalized gap between top-1 and top-2 distances.

        Large gap indicates confident retrieval (likely ID).
        Small gap indicates uncertainty (possibly OOD).
        """
        if distances.shape[1] < 2:
            return np.zeros(distances.shape[0])

        top1 = distances[:, 0]
        top2 = distances[:, 1]

        # Normalized gap
        gap = (top2 - top1) / (top1 + 1e-10)
        return gap

    def extract_sim_drop_rate(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute similarity drop rate: how quickly similarity decreases across neighbors.

        For ID samples, similarity decreases gradually (dense neighborhood).
        For OOD samples, similarity drops sharply (sparse neighborhood).
        """
        if distances.shape[1] < 3:
            return np.zeros(distances.shape[0])

        # Use first vs last neighbor distance ratio
        first = distances[:, 0]
        last = distances[:, -1]

        # Drop rate (higher = sharper drop = more OOD-like)
        drop_rate = (last - first) / (first + 1e-10)
        return drop_rate

    def extract_neighbor_variance(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute variance in neighbor distances.

        Higher variance can indicate inconsistent neighborhood (OOD).
        """
        return np.var(distances, axis=1)

    def extract_cluster_agreement(self, embeddings: np.ndarray,
                                   neighbor_indices: np.ndarray,
                                   cluster_labels: Optional[np.ndarray] = None,
                                   n_clusters: int = 10) -> np.ndarray:
        """
        Compute cluster agreement: do neighbors belong to same cluster?

        If cluster_labels not provided, compute quick k-means clustering.
        """
        n_samples = embeddings.shape[0]

        if cluster_labels is None:
            # Quick clustering on a sample
            from sklearn.cluster import MiniBatchKMeans
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3)
            cluster_labels = kmeans.fit_predict(embeddings)

        agreements = np.zeros(n_samples)
        for i in range(n_samples):
            query_cluster = cluster_labels[i] if i < len(cluster_labels) else -1
            neighbor_clusters = cluster_labels[neighbor_indices[i]]
            agreement = np.mean(neighbor_clusters == query_cluster)
            agreements[i] = agreement

        return agreements

    def extract_all(self, distances: np.ndarray,
                    neighbor_indices: np.ndarray,
                    neighbor_labels: Optional[np.ndarray] = None,
                    embeddings: Optional[np.ndarray] = None,
                    cluster_labels: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Extract all graph features.

        Args:
            distances: kNN distances (n_samples, k)
            neighbor_indices: kNN indices (n_samples, k)
            neighbor_labels: labels of neighbors (n_samples, k), optional
            embeddings: query embeddings for cluster agreement
            cluster_labels: pre-computed cluster labels

        Returns:
            dict mapping feature name to feature values
        """
        features = {}

        # Label purity (requires neighbor labels)
        if neighbor_labels is not None:
            features['label_purity'] = self.extract_label_purity(neighbor_labels)
        else:
            features['label_purity'] = np.zeros(distances.shape[0])

        # Retrieval gap
        features['retrieval_gap'] = self.extract_retrieval_gap(distances)

        # Similarity drop rate
        features['sim_drop_rate'] = self.extract_sim_drop_rate(distances)

        # Neighbor variance
        features['neighbor_variance'] = self.extract_neighbor_variance(distances)

        # Cluster agreement (requires embeddings)
        if embeddings is not None:
            features['cluster_agreement'] = self.extract_cluster_agreement(
                embeddings, neighbor_indices, cluster_labels
            )
        else:
            features['cluster_agreement'] = np.zeros(distances.shape[0])

        return features

    def features_to_array(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """Convert feature dict to array (n_samples, n_features)."""
        return np.stack([features[name] for name in self.feature_names], axis=1)
