"""
Gaussian Prototype Model for Hypothesis 3 Verification

Tests whether Prototype Dispersion Index (PDI) correlates with noise rate.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer


class GaussianPrototype:
    """
    Gaussian Prototype Network for Relation Classification

    Each relation is represented by a Gaussian distribution N(μ, Σ).
    Used for H3 to verify: PDI = trace(Σ)/d correlates with noise rate.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        regularization: float = 1e-4,
        use_diagonal_cov: bool = True
    ):
        """
        Args:
            embedding_dim: Dimension of sentence embeddings
            regularization: Regularization for covariance estimation
            use_diagonal_cov: Use diagonal covariance (faster, more stable)
        """
        self.embedding_dim = embedding_dim
        self.regularization = regularization
        self.use_diagonal_cov = use_diagonal_cov

        self.vectorizer = TfidfVectorizer(max_features=embedding_dim)
        self.prototypes: Dict[int, Dict[str, np.ndarray]] = {}
        self.label_encoder: Dict[str, int] = {}
        self.label_decoder: Dict[int, str] = {}
        self.is_fitted = False

    def _encode_sentences(self, sentences: List[str]) -> np.ndarray:
        """Encode sentences to embeddings"""
        if not self.is_fitted:
            X = self.vectorizer.fit_transform(sentences)
        else:
            X = self.vectorizer.transform(sentences)
        return X.toarray()

    def fit(self, sentences: List[str], labels: List[str]):
        """
        Fit Gaussian prototypes for each relation

        Args:
            sentences: Training sentences
            labels: Relation labels
        """
        # Build label encoder
        unique_labels = sorted(set(labels))
        self.label_encoder = {label: i for i, label in enumerate(unique_labels)}
        self.label_decoder = {i: label for label, i in self.label_encoder.items()}

        # Encode sentences
        embeddings = self._encode_sentences(sentences)
        y = np.array([self.label_encoder[l] for l in labels])

        # Compute Gaussian parameters for each relation
        for rel_id in range(len(unique_labels)):
            mask = y == rel_id
            rel_embeddings = embeddings[mask]

            if len(rel_embeddings) < 2:
                # Not enough samples, use default
                mean = np.zeros(self.embedding_dim)
                cov = np.eye(self.embedding_dim)
            else:
                mean = rel_embeddings.mean(axis=0)

                if self.use_diagonal_cov:
                    # Diagonal covariance (more stable)
                    var = rel_embeddings.var(axis=0) + self.regularization
                    cov = np.diag(var)
                else:
                    # Full covariance
                    cov = np.cov(rel_embeddings.T) + self.regularization * np.eye(self.embedding_dim)

            self.prototypes[rel_id] = {
                'mean': mean,
                'cov': cov,
                'n_samples': len(rel_embeddings)
            }

        self.is_fitted = True

    def compute_pdi(self, relation_id: Optional[int] = None) -> Dict[int, float]:
        """
        Compute Prototype Dispersion Index for relations

        PDI = trace(Σ) / d

        Higher PDI indicates more dispersed/noisy samples.

        Args:
            relation_id: If specified, compute only for this relation

        Returns:
            Dictionary mapping relation_id to PDI
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")

        if relation_id is not None:
            proto = self.prototypes[relation_id]
            pdi = np.trace(proto['cov']) / self.embedding_dim
            return {relation_id: pdi}

        pdi_scores = {}
        for rel_id, proto in self.prototypes.items():
            pdi = np.trace(proto['cov']) / self.embedding_dim
            pdi_scores[rel_id] = pdi

        return pdi_scores

    def compute_mahalanobis_distance(
        self,
        sentence: str,
        relation_id: int
    ) -> float:
        """
        Compute Mahalanobis distance from sentence to relation prototype

        D_M(x, μ) = sqrt((x - μ)^T Σ^{-1} (x - μ))
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")

        embedding = self._encode_sentences([sentence])[0]
        proto = self.prototypes[relation_id]

        diff = embedding - proto['mean']

        if self.use_diagonal_cov:
            # Efficient computation for diagonal covariance
            inv_var = 1.0 / np.diag(proto['cov'])
            distance = np.sqrt(np.sum(diff ** 2 * inv_var))
        else:
            try:
                inv_cov = np.linalg.inv(proto['cov'])
                distance = np.sqrt(diff @ inv_cov @ diff)
            except np.linalg.LinAlgError:
                # Fallback to Euclidean
                distance = np.linalg.norm(diff)

        return distance

    def predict(self, sentences: List[str]) -> List[str]:
        """Predict relations using nearest prototype"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")

        embeddings = self._encode_sentences(sentences)
        predictions = []

        for emb in embeddings:
            min_distance = float('inf')
            best_rel = 0

            for rel_id, proto in self.prototypes.items():
                diff = emb - proto['mean']
                if self.use_diagonal_cov:
                    inv_var = 1.0 / np.diag(proto['cov'])
                    dist = np.sum(diff ** 2 * inv_var)
                else:
                    try:
                        inv_cov = np.linalg.inv(proto['cov'])
                        dist = diff @ inv_cov @ diff
                    except:
                        dist = np.sum(diff ** 2)

                if dist < min_distance:
                    min_distance = dist
                    best_rel = rel_id

            predictions.append(self.label_decoder[best_rel])

        return predictions


def inject_symmetric_noise(
    labels: List[str],
    noise_rate: float,
    seed: int = 42
) -> List[str]:
    """
    Inject symmetric label noise

    Randomly flips labels to other classes with given probability.

    Args:
        labels: Original labels
        noise_rate: Proportion of labels to flip (0.0 to 1.0)
        seed: Random seed

    Returns:
        Noisy labels
    """
    np.random.seed(seed)
    unique_labels = list(set(labels))
    noisy_labels = labels.copy()

    n_flip = int(len(labels) * noise_rate)
    flip_indices = np.random.choice(len(labels), n_flip, replace=False)

    for idx in flip_indices:
        current_label = labels[idx]
        other_labels = [l for l in unique_labels if l != current_label]
        if other_labels:
            noisy_labels[idx] = np.random.choice(other_labels)

    return noisy_labels


def verify_pdi_noise_correlation(
    sentences: List[str],
    labels: List[str],
    noise_rates: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
) -> Dict:
    """
    Verify Hypothesis 3: PDI correlates with noise rate

    Args:
        sentences: Training sentences
        labels: Original (clean) labels
        noise_rates: List of noise rates to test

    Returns:
        Dictionary with correlation results
    """
    results = []

    for noise_rate in noise_rates:
        # Inject noise
        noisy_labels = inject_symmetric_noise(labels, noise_rate)

        # Train Gaussian prototype
        model = GaussianPrototype()
        model.fit(sentences, noisy_labels)

        # Compute average PDI
        pdi_scores = model.compute_pdi()
        avg_pdi = np.mean(list(pdi_scores.values()))
        std_pdi = np.std(list(pdi_scores.values()))

        results.append({
            'noise_rate': noise_rate,
            'avg_pdi': avg_pdi,
            'std_pdi': std_pdi,
            'pdi_scores': pdi_scores
        })

    # Compute correlation
    noise_values = [r['noise_rate'] for r in results]
    pdi_values = [r['avg_pdi'] for r in results]

    from scipy.stats import pearsonr
    r, p_value = pearsonr(noise_values, pdi_values)

    return {
        'results': results,
        'pearson_r': r,
        'p_value': p_value,
        'hypothesis_supported': r > 0.5 and p_value < 0.05
    }
