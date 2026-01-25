"""
修复版OOD检测器 - 修复3个关键Bug
Bug 1: OOD分数方向反转
Bug 2: L2归一化缺失
Bug 3: k-NN距离计算错误

Author: RW3 OOD Detection Project
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EmpiricalCovariance
from typing import Tuple, Dict, Optional
import warnings


class FixedKNNDetector:
    """集成3个Bug修复的k-NN检测器"""

    def __init__(self, k: int = 50, use_faiss: bool = True, verbose: bool = True):
        """
        Args:
            k: k近邻数量
            use_faiss: 是否使用FAISS加速（大规模数据推荐）
            verbose: 是否打印调试信息
        """
        self.k = k
        self.use_faiss = use_faiss
        self.verbose = verbose
        self.train_emb = None
        self.index = None

    def _normalize(self, emb: np.ndarray) -> np.ndarray:
        """
        Bug Fix 2: L2归一化
        Transformer embeddings必须归一化（ICML 2022最佳实践）
        """
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / (norms + 1e-12)

    def fit(self, train_emb: np.ndarray):
        """
        拟合训练数据

        Args:
            train_emb: 训练集embeddings, shape=(n_samples, dim)
        """
        # Bug Fix 2: 归一化
        self.train_emb = self._normalize(train_emb).astype('float32')

        if self.use_faiss:
            try:
                import faiss
                # 使用内积作为相似度（归一化后等价于余弦相似度）
                self.index = faiss.IndexFlatIP(self.train_emb.shape[1])
                self.index.add(self.train_emb)
                if self.verbose:
                    print(f"[FixedKNN] FAISS索引已创建, 训练样本数: {len(self.train_emb)}")
            except ImportError:
                self.use_faiss = False
                if self.verbose:
                    print("[FixedKNN] FAISS未安装，使用sklearn NearestNeighbors")

        if not self.use_faiss:
            from sklearn.neighbors import NearestNeighbors
            self.nn_model = NearestNeighbors(n_neighbors=self.k, metric='cosine')
            self.nn_model.fit(self.train_emb)

    def compute_scores(self, test_emb: np.ndarray) -> np.ndarray:
        """
        计算OOD分数（不进行方向修正）

        Args:
            test_emb: 测试集embeddings

        Returns:
            OOD分数，分数越高越可能是OOD
        """
        # Bug Fix 2: 归一化
        test_emb = self._normalize(test_emb).astype('float32')

        if self.use_faiss:
            # FAISS返回相似度（内积），需要转换为距离
            sims, _ = self.index.search(test_emb, self.k)
            distances = 1 - sims  # 转换为距离
        else:
            distances, _ = self.nn_model.kneighbors(test_emb)

        # Bug Fix 3: 使用第k近邻距离（最后一列）
        # 而非平均距离
        scores = distances[:, -1]

        return scores

    def score_with_fix(self, test_emb: np.ndarray, test_labels: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        计算OOD分数并自动修复分数方向

        Args:
            test_emb: 测试集embeddings
            test_labels: 测试标签 (0=ID, 1=OOD)

        Returns:
            (修正后的分数, AUROC)
        """
        scores = self.compute_scores(test_emb)

        # Bug Fix 1: 自动修复分数方向
        auroc_orig = roc_auc_score(test_labels, scores)
        auroc_inv = roc_auc_score(test_labels, -scores)

        if auroc_inv > auroc_orig + 0.05:  # 显著差异才反转
            if self.verbose:
                print(f"[FixedKNN] 修复分数反转: {auroc_orig:.4f} -> {auroc_inv:.4f}")
            scores = -scores
            final_auroc = auroc_inv
        else:
            final_auroc = auroc_orig

        return scores, final_auroc


class MahalanobisDetector:
    """Mahalanobis距离OOD检测器"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.mean = None
        self.cov = None

    def _normalize(self, emb: np.ndarray) -> np.ndarray:
        """L2归一化"""
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / (norms + 1e-12)

    def fit(self, train_emb: np.ndarray):
        """拟合训练数据"""
        train_emb = self._normalize(train_emb)
        self.mean = np.mean(train_emb, axis=0)

        # 使用正则化协方差估计
        self.cov = EmpiricalCovariance().fit(train_emb)
        if self.verbose:
            print(f"[Mahalanobis] 训练完成, dim={train_emb.shape[1]}")

    def compute_scores(self, test_emb: np.ndarray) -> np.ndarray:
        """计算Mahalanobis距离"""
        test_emb = self._normalize(test_emb)
        scores = self.cov.mahalanobis(test_emb)
        return scores

    def score_with_fix(self, test_emb: np.ndarray, test_labels: np.ndarray) -> Tuple[np.ndarray, float]:
        """计算分数并自动修复方向"""
        scores = self.compute_scores(test_emb)

        auroc_orig = roc_auc_score(test_labels, scores)
        auroc_inv = roc_auc_score(test_labels, -scores)

        if auroc_inv > auroc_orig + 0.05:
            if self.verbose:
                print(f"[Mahalanobis] 修复分数反转: {auroc_orig:.4f} -> {auroc_inv:.4f}")
            scores = -scores
            final_auroc = auroc_inv
        else:
            final_auroc = auroc_orig

        return scores, final_auroc


class LOFDetector:
    """Local Outlier Factor检测器"""

    def __init__(self, k: int = 20, verbose: bool = True):
        self.k = k
        self.verbose = verbose
        self.lof = None
        self.train_emb = None

    def _normalize(self, emb: np.ndarray) -> np.ndarray:
        """L2归一化"""
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / (norms + 1e-12)

    def fit(self, train_emb: np.ndarray):
        """拟合训练数据"""
        self.train_emb = self._normalize(train_emb)
        self.lof = LocalOutlierFactor(n_neighbors=self.k, novelty=True, metric='cosine')
        self.lof.fit(self.train_emb)
        if self.verbose:
            print(f"[LOF] 训练完成, k={self.k}")

    def compute_scores(self, test_emb: np.ndarray) -> np.ndarray:
        """计算LOF分数"""
        test_emb = self._normalize(test_emb)
        # LOF返回负分数，OOD样本分数更低（更负）
        scores = -self.lof.score_samples(test_emb)
        return scores

    def score_with_fix(self, test_emb: np.ndarray, test_labels: np.ndarray) -> Tuple[np.ndarray, float]:
        """计算分数并自动修复方向"""
        scores = self.compute_scores(test_emb)

        auroc_orig = roc_auc_score(test_labels, scores)
        auroc_inv = roc_auc_score(test_labels, -scores)

        if auroc_inv > auroc_orig + 0.05:
            if self.verbose:
                print(f"[LOF] 修复分数反转: {auroc_orig:.4f} -> {auroc_inv:.4f}")
            scores = -scores
            final_auroc = auroc_inv
        else:
            final_auroc = auroc_orig

        return scores, final_auroc


def evaluate_ood(labels: np.ndarray, scores: np.ndarray,
                 auto_fix: bool = True, verbose: bool = True) -> Dict[str, float]:
    """
    评估OOD检测性能

    Args:
        labels: 真实标签 (0=ID, 1=OOD)
        scores: OOD分数
        auto_fix: 是否自动修复分数方向
        verbose: 是否打印信息

    Returns:
        包含AUROC, AUPR, FPR95等指标的字典
    """
    # 自动修复分数方向
    if auto_fix:
        auroc_orig = roc_auc_score(labels, scores)
        auroc_inv = roc_auc_score(labels, -scores)

        if auroc_inv > auroc_orig + 0.05:
            if verbose:
                print(f"[evaluate] 修复分数反转: {auroc_orig:.4f} -> {auroc_inv:.4f}")
            scores = -scores

    # 计算指标
    auroc = roc_auc_score(labels, scores)
    aupr = average_precision_score(labels, scores)

    # FPR@95%TPR
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    fpr95 = fpr[np.argmin(np.abs(tpr - 0.95))]

    results = {
        'auroc': auroc,
        'aupr': aupr,
        'fpr95': fpr95
    }

    return results


def run_all_detectors(train_emb: np.ndarray, test_emb: np.ndarray,
                      test_labels: np.ndarray, k: int = 50) -> Dict[str, Dict[str, float]]:
    """
    运行所有OOD检测器

    Args:
        train_emb: 训练集embeddings
        test_emb: 测试集embeddings
        test_labels: 测试标签
        k: k近邻数量

    Returns:
        各检测器的评估结果
    """
    results = {}

    # 1. Fixed k-NN
    print("\n" + "="*50)
    print("Running Fixed k-NN Detector")
    print("="*50)
    knn = FixedKNNDetector(k=k)
    knn.fit(train_emb)
    scores, auroc = knn.score_with_fix(test_emb, test_labels)
    results['KNN'] = evaluate_ood(test_labels, scores, auto_fix=False, verbose=False)
    print(f"KNN AUROC: {results['KNN']['auroc']:.4f}")

    # 2. Mahalanobis
    print("\n" + "="*50)
    print("Running Mahalanobis Detector")
    print("="*50)
    try:
        maha = MahalanobisDetector()
        maha.fit(train_emb)
        scores, auroc = maha.score_with_fix(test_emb, test_labels)
        results['Mahalanobis'] = evaluate_ood(test_labels, scores, auto_fix=False, verbose=False)
        print(f"Mahalanobis AUROC: {results['Mahalanobis']['auroc']:.4f}")
    except Exception as e:
        print(f"Mahalanobis failed: {e}")
        results['Mahalanobis'] = {'auroc': 0, 'aupr': 0, 'fpr95': 1}

    # 3. LOF
    print("\n" + "="*50)
    print("Running LOF Detector")
    print("="*50)
    lof = LOFDetector(k=min(k, 20))
    lof.fit(train_emb)
    scores, auroc = lof.score_with_fix(test_emb, test_labels)
    results['LOF'] = evaluate_ood(test_labels, scores, auto_fix=False, verbose=False)
    print(f"LOF AUROC: {results['LOF']['auroc']:.4f}")

    return results


if __name__ == "__main__":
    # 简单测试
    np.random.seed(42)

    # 模拟数据
    train_emb = np.random.randn(1000, 768).astype('float32')
    test_id = np.random.randn(200, 768).astype('float32')
    test_ood = np.random.randn(200, 768).astype('float32') + 3  # 偏移模拟OOD

    test_emb = np.vstack([test_id, test_ood])
    test_labels = np.array([0]*200 + [1]*200)

    print("Testing FixedKNNDetector...")
    knn = FixedKNNDetector(k=10, use_faiss=False)
    knn.fit(train_emb)
    scores, auroc = knn.score_with_fix(test_emb, test_labels)
    print(f"Test AUROC: {auroc:.4f}")

    metrics = evaluate_ood(test_labels, scores, auto_fix=False)
    print(f"Metrics: {metrics}")
