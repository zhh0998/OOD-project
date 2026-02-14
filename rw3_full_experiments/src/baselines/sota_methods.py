"""
SOTA Baseline Methods
B6-B9: Recent top-conference methods for comparison.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
from scipy import stats
from sklearn.mixture import GaussianMixture


class CEDBaseline:
    """
    B6: CED (Contrastive Embedding Difference) - EMNLP 2024 Findings
    Training-free, black-box friendly.

    Key idea: Compare embedding of input vs perturbed versions.
    For OOD inputs, perturbations cause larger embedding changes.
    """

    def __init__(self, n_perturbations: int = 5, perturbation_ratio: float = 0.15):
        self.n_perturbations = n_perturbations
        self.perturbation_ratio = perturbation_ratio
        self.embedding_fn = None

    def set_embedding_fn(self, embedding_fn):
        """Set the embedding function for computing representations."""
        self.embedding_fn = embedding_fn

    def _perturb_text(self, text: str) -> List[str]:
        """Generate perturbed versions of text."""
        words = text.split()
        n_remove = max(1, int(len(words) * self.perturbation_ratio))
        perturbed = []

        for _ in range(self.n_perturbations):
            if len(words) > n_remove:
                indices = np.random.choice(len(words), n_remove, replace=False)
                new_words = [w for i, w in enumerate(words) if i not in indices]
                perturbed.append(' '.join(new_words))
            else:
                perturbed.append(text)

        return perturbed

    def score_texts(self, texts: List[str]) -> np.ndarray:
        """
        Compute CED scores for texts.

        Higher score = more OOD (larger embedding change from perturbation).
        """
        if self.embedding_fn is None:
            raise RuntimeError("Must call set_embedding_fn() first")

        scores = []

        for text in texts:
            # Original embedding
            orig_emb = self.embedding_fn([text])[0]

            # Perturbed embeddings
            perturbed_texts = self._perturb_text(text)
            perturbed_embs = self.embedding_fn(perturbed_texts)

            # Compute differences
            diffs = np.linalg.norm(perturbed_embs - orig_emb, axis=1)
            score = np.mean(diffs)
            scores.append(score)

        return np.array(scores)

    def score(self, embeddings: np.ndarray, texts: Optional[List[str]] = None) -> np.ndarray:
        """
        Score using pre-computed embeddings (approximation).

        If texts not available, use embedding-space perturbation.
        """
        if texts is not None and self.embedding_fn is not None:
            return self.score_texts(texts)

        # Fallback: embedding-space noise sensitivity
        n_samples = len(embeddings)
        scores = np.zeros(n_samples)

        for i in range(n_samples):
            emb = embeddings[i]
            # Add Gaussian noise
            noise = np.random.randn(self.n_perturbations, len(emb)) * 0.1
            perturbed = emb + noise
            # Measure sensitivity
            diffs = np.linalg.norm(perturbed - emb, axis=1)
            scores[i] = np.std(diffs)  # Variance as proxy for instability

        return scores


class MahalanobisPlusPlusBaseline:
    """
    B8: Mahalanobis++ (ICML 2025)
    Feature normalization + Mahalanobis distance.
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.mean = None
        self.precision = None
        self.norm_stats = None

    def fit(self, X: np.ndarray, audit_log=None):
        """Fit with feature normalization."""
        if self.normalize:
            # Per-feature normalization
            self.norm_stats = {
                'mean': X.mean(axis=0),
                'std': X.std(axis=0) + 1e-6
            }
            X = (X - self.norm_stats['mean']) / self.norm_stats['std']

        self.mean = X.mean(axis=0)

        # Robust covariance estimation
        from sklearn.covariance import LedoitWolf
        try:
            cov_estimator = LedoitWolf()
            cov_estimator.fit(X)
            self.precision = cov_estimator.precision_
        except Exception:
            var = np.var(X, axis=0) + 1e-6
            self.precision = np.diag(1.0 / var)

        if audit_log is not None:
            audit_log.log_fit(
                operation="B8_mahalanobis_pp",
                fit_split="id_train",
                n_samples=len(X)
            )
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute normalized Mahalanobis distances."""
        if self.normalize and self.norm_stats is not None:
            X = (X - self.norm_stats['mean']) / self.norm_stats['std']

        centered = X - self.mean
        mahal = np.sum(centered @ self.precision * centered, axis=1)
        return np.sqrt(mahal)


class FLaTSBaseline:
    """
    B9: FLatS (EMNLP 2023)
    Background distribution fitting.
    """

    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        self.gmm = None

    def fit(self, X: np.ndarray, audit_log=None):
        """Fit GMM as background distribution."""
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='diag',
            random_state=42
        )
        self.gmm.fit(X)

        if audit_log is not None:
            audit_log.log_fit(
                operation="B9_flats",
                fit_split="id_train",
                n_samples=len(X),
                additional_info={"n_components": self.n_components}
            )
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute negative log-likelihood (higher = more OOD)."""
        return -self.gmm.score_samples(X)


class BLOODBaseline:
    """
    B7: BLOOD (ICLR 2024)
    Requires intermediate layer access - marked as "non-black-box upper bound".

    Note: This is a reference implementation. Full BLOOD requires:
    - Access to transformer intermediate layers
    - Per-layer Mahalanobis computation
    """

    def __init__(self, use_intermediate: bool = True):
        self.use_intermediate = use_intermediate
        self.layer_stats = {}

    def fit(self, embeddings: np.ndarray,
            intermediate_layers: Optional[Dict[str, np.ndarray]] = None,
            audit_log=None):
        """
        Fit BLOOD on intermediate layer representations.

        Args:
            embeddings: final layer embeddings
            intermediate_layers: dict of layer_name -> embeddings
        """
        if intermediate_layers is None:
            # Fallback to final embeddings only
            intermediate_layers = {'final': embeddings}

        for layer_name, layer_emb in intermediate_layers.items():
            mean = layer_emb.mean(axis=0)
            # Diagonal covariance for efficiency
            var = np.var(layer_emb, axis=0) + 1e-6
            self.layer_stats[layer_name] = {'mean': mean, 'var': var}

        if audit_log is not None:
            audit_log.log_fit(
                operation="B7_blood",
                fit_split="id_train",
                n_samples=len(embeddings),
                additional_info={
                    "n_layers": len(intermediate_layers),
                    "note": "requires_intermediate_layers"
                }
            )
        return self

    def score(self, embeddings: np.ndarray,
              intermediate_layers: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """Compute BLOOD scores."""
        if intermediate_layers is None:
            intermediate_layers = {'final': embeddings}

        all_scores = []
        for layer_name, layer_emb in intermediate_layers.items():
            if layer_name in self.layer_stats:
                stats = self.layer_stats[layer_name]
                # Simplified Mahalanobis with diagonal covariance
                centered = layer_emb - stats['mean']
                mahal = np.sum(centered ** 2 / stats['var'], axis=1)
                all_scores.append(mahal)

        if not all_scores:
            return np.zeros(len(embeddings))

        # Aggregate across layers
        return np.mean(all_scores, axis=0)
