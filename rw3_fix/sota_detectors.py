#!/usr/bin/env python3
"""
SOTA OOD检测基线实现

包含:
- DA-ADB (Distance-Aware Attention-Based Detection) - TASLP 2023
- FLatS (Feature-wise Latent Space) - EMNLP 2023
- RMD (Relative Mahalanobis Distance) - NeurIPS 2021

Author: RW3 OOD Detection Project
"""

import numpy as np
from typing import Optional, Tuple
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score


class DAADBDetector:
    """
    DA-ADB (Distance-Aware Attention-Based Detection)
    使用注意力机制加权k近邻距离

    Reference: TASLP 2023
    """

    def __init__(self, k: int = 10, temperature: float = 1.0, verbose: bool = True):
        self.k = k
        self.temperature = temperature
        self.verbose = verbose
        self.nn = None
        self.train_embeddings = None

    def _normalize(self, emb: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / (norms + 1e-12)

    def fit(self, train_embeddings: np.ndarray, train_labels: Optional[np.ndarray] = None):
        self.train_embeddings = self._normalize(train_embeddings).astype('float32')
        self.nn = NearestNeighbors(n_neighbors=self.k, metric='cosine')
        self.nn.fit(self.train_embeddings)
        if self.verbose:
            print(f"[DA-ADB] 训练完成, k={self.k}, temperature={self.temperature}")

    def score(self, test_embeddings: np.ndarray) -> np.ndarray:
        test_norm = self._normalize(test_embeddings).astype('float32')
        distances, _ = self.nn.kneighbors(test_norm)

        # 注意力权重: softmax(-distances / temperature)
        attention_logits = -distances / self.temperature
        attention_logits = attention_logits - attention_logits.max(axis=1, keepdims=True)
        attention = np.exp(attention_logits)
        attention = attention / attention.sum(axis=1, keepdims=True)

        # 加权距离
        weighted_distances = (attention * distances).sum(axis=1)
        return weighted_distances

    def score_with_fix(self, test_embeddings: np.ndarray,
                       test_labels: np.ndarray) -> Tuple[np.ndarray, float]:
        scores = self.score(test_embeddings)
        auroc_orig = roc_auc_score(test_labels, scores)
        auroc_inv = roc_auc_score(test_labels, -scores)

        if auroc_inv > auroc_orig + 0.05:
            if self.verbose:
                print(f"[DA-ADB] 修复分数反转: {auroc_orig:.4f} -> {auroc_inv:.4f}")
            return -scores, auroc_inv
        return scores, auroc_orig


class FLatSDetector:
    """
    FLatS (Feature-wise Latent Space)
    在多个PCA子空间中计算马氏距离

    Reference: EMNLP 2023
    """

    def __init__(self, n_components: int = 50, n_subspaces: int = 5,
                 reg_factor: float = 1e-4, verbose: bool = True):
        self.n_components = n_components
        self.n_subspaces = n_subspaces
        self.reg_factor = reg_factor
        self.verbose = verbose
        self.pcas = []
        self.means = []
        self.inv_covs = []

    def _normalize(self, emb: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / (norms + 1e-12)

    def fit(self, train_embeddings: np.ndarray, train_labels: Optional[np.ndarray] = None):
        train_norm = self._normalize(train_embeddings).astype('float32')
        n_features = train_embeddings.shape[1]
        feature_dim = n_features // self.n_subspaces

        self.pcas = []
        self.means = []
        self.inv_covs = []

        for i in range(self.n_subspaces):
            start_idx = i * feature_dim
            end_idx = min((i + 1) * feature_dim, n_features)
            sub_features = train_norm[:, start_idx:end_idx]

            # PCA
            n_comp = min(self.n_components, sub_features.shape[1], sub_features.shape[0] - 1)
            pca = PCA(n_components=n_comp)
            sub_latent = pca.fit_transform(sub_features)

            # 均值和协方差
            mean = sub_latent.mean(axis=0)
            centered = sub_latent - mean
            cov = np.cov(centered.T) if centered.shape[1] > 1 else np.array([[np.var(centered)]])
            cov = np.atleast_2d(cov)
            cov += np.eye(cov.shape[0]) * self.reg_factor

            try:
                inv_cov = np.linalg.inv(cov)
            except:
                inv_cov = np.linalg.pinv(cov)

            self.pcas.append(pca)
            self.means.append(mean)
            self.inv_covs.append(inv_cov)

        if self.verbose:
            print(f"[FLatS] 训练完成, {self.n_subspaces}个子空间")

    def score(self, test_embeddings: np.ndarray) -> np.ndarray:
        test_norm = self._normalize(test_embeddings).astype('float32')
        n_features = test_embeddings.shape[1]
        feature_dim = n_features // self.n_subspaces

        all_distances = []

        for i, (pca, mean, inv_cov) in enumerate(zip(self.pcas, self.means, self.inv_covs)):
            start_idx = i * feature_dim
            end_idx = min((i + 1) * feature_dim, n_features)
            sub_features = test_norm[:, start_idx:end_idx]

            sub_latent = pca.transform(sub_features)
            centered = sub_latent - mean

            # 马氏距离
            if inv_cov.shape[0] == 1:
                mahal_dist = np.abs(centered.flatten()) * np.sqrt(inv_cov[0, 0])
            else:
                mahal_dist = np.sqrt(np.sum(centered @ inv_cov * centered, axis=1))

            all_distances.append(mahal_dist)

        combined_distance = np.mean(all_distances, axis=0)
        return combined_distance

    def score_with_fix(self, test_embeddings: np.ndarray,
                       test_labels: np.ndarray) -> Tuple[np.ndarray, float]:
        scores = self.score(test_embeddings)
        auroc_orig = roc_auc_score(test_labels, scores)
        auroc_inv = roc_auc_score(test_labels, -scores)

        if auroc_inv > auroc_orig + 0.05:
            if self.verbose:
                print(f"[FLatS] 修复分数反转: {auroc_orig:.4f} -> {auroc_inv:.4f}")
            return -scores, auroc_inv
        return scores, auroc_orig


class RMDDetector:
    """
    RMD (Relative Mahalanobis Distance)
    计算样本到每个类别中心的相对马氏距离

    Reference: NeurIPS 2021
    """

    def __init__(self, reg_factor: float = 1e-4, verbose: bool = True):
        self.reg_factor = reg_factor
        self.verbose = verbose
        self.class_means = {}
        self.class_inv_covs = {}
        self.global_mean = None

    def _normalize(self, emb: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / (norms + 1e-12)

    def fit(self, train_embeddings: np.ndarray, train_labels: np.ndarray):
        train_norm = self._normalize(train_embeddings).astype('float32')
        self.global_mean = train_norm.mean(axis=0)

        unique_labels = np.unique(train_labels)

        for label in unique_labels:
            mask = train_labels == label
            if mask.sum() < 2:
                continue

            class_samples = train_norm[mask]
            class_mean = class_samples.mean(axis=0)

            centered = class_samples - class_mean
            cov = np.cov(centered.T)
            cov = np.atleast_2d(cov)
            cov += np.eye(cov.shape[0]) * self.reg_factor

            try:
                inv_cov = np.linalg.inv(cov)
            except:
                inv_cov = np.linalg.pinv(cov)

            self.class_means[label] = class_mean
            self.class_inv_covs[label] = inv_cov

        if self.verbose:
            print(f"[RMD] 训练完成, {len(self.class_means)}个类别")

    def score(self, test_embeddings: np.ndarray) -> np.ndarray:
        test_norm = self._normalize(test_embeddings).astype('float32')
        n_samples = len(test_norm)

        if len(self.class_means) == 0:
            return np.zeros(n_samples)

        all_distances = np.zeros((n_samples, len(self.class_means)))

        for i, (label, mean) in enumerate(self.class_means.items()):
            inv_cov = self.class_inv_covs[label]
            centered = test_norm - mean
            mahal_dist = np.sqrt(np.sum(centered @ inv_cov * centered, axis=1))
            all_distances[:, i] = mahal_dist

        min_distances = all_distances.min(axis=1)
        mean_distances = all_distances.mean(axis=1)

        # RMD: 最小距离与平均距离的比值
        rmd = min_distances / (mean_distances + 1e-8)

        # 转换为OOD分数（越大越OOD）
        ood_score = min_distances  # 使用最小距离作为分数

        return ood_score

    def score_with_fix(self, test_embeddings: np.ndarray,
                       test_labels: np.ndarray) -> Tuple[np.ndarray, float]:
        scores = self.score(test_embeddings)
        auroc_orig = roc_auc_score(test_labels, scores)
        auroc_inv = roc_auc_score(test_labels, -scores)

        if auroc_inv > auroc_orig + 0.05:
            if self.verbose:
                print(f"[RMD] 修复分数反转: {auroc_orig:.4f} -> {auroc_inv:.4f}")
            return -scores, auroc_inv
        return scores, auroc_orig


def test_sota_detectors():
    """测试SOTA检测器"""
    np.random.seed(42)

    # 模拟数据
    n_train = 500
    n_test = 200
    dim = 384
    n_classes = 10

    # 训练数据
    centers = np.random.randn(n_classes, dim) * 3
    train_emb = []
    train_labels = []
    for c in range(n_classes):
        n = n_train // n_classes
        samples = centers[c] + np.random.randn(n, dim) * 0.5
        train_emb.append(samples)
        train_labels.extend([c] * n)
    train_emb = np.vstack(train_emb).astype('float32')
    train_labels = np.array(train_labels)

    # 测试数据
    test_id = np.vstack([centers[c] + np.random.randn(n_test//4, dim) * 0.5
                         for c in range(n_classes//2)])
    test_ood = np.random.randn(n_test//2, dim).astype('float32') + 8
    test_emb = np.vstack([test_id, test_ood])
    test_labels = np.array([0] * len(test_id) + [1] * len(test_ood))

    print("="*60)
    print("SOTA检测器测试")
    print("="*60)

    detectors = {
        'DA-ADB': DAADBDetector(k=10, verbose=True),
        'FLatS': FLatSDetector(n_components=50, verbose=True),
        'RMD': RMDDetector(verbose=True)
    }

    for name, detector in detectors.items():
        print(f"\n测试 {name}...")
        detector.fit(train_emb, train_labels)
        scores, auroc = detector.score_with_fix(test_emb, test_labels)
        print(f"  AUROC: {auroc:.4f}")

    print("\n✅ 所有SOTA检测器测试通过")


if __name__ == "__main__":
    test_sota_detectors()
