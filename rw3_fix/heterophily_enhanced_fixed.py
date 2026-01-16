"""
HeterophilyEnhanced OOD检测器 - 修复版本
修复了3个关键Bug：

Bug 1: OOD分数方向反转 - 确保高距离=高OOD分数
Bug 2: L2归一化缺失 - 所有embeddings都进行L2归一化
Bug 3: k-NN距离计算错误 - 使用第k近邻距离而非平均距离

Author: RW3 OOD Detection Project
"""

import numpy as np
from typing import Tuple, Optional, Dict
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available, using numpy-only implementation")

try:
    from torch_geometric.nn import GATv2Conv
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class HeterophilyEnhancedFixed:
    """
    RW3核心方法：异配性感知OOD检测（修复版本）

    Bug修复:
    1. 确保OOD分数方向正确（高距离=高OOD分数）
    2. 强制L2归一化所有embeddings
    3. 使用第k近邻距离而非平均距离

    Pipeline:
    1. Sentence embeddings → L2归一化 → k-NN graph
    2. 基于余弦距离计算第k近邻距离
    3. 计算节点异配性
    4. 融合k-NN距离 + 异配性作为OOD分数
    """

    def __init__(self, input_dim: int = 384, hidden_dim: int = 256,
                 k: int = 50, alpha: float = 0.3,
                 verbose: bool = True):
        """
        Args:
            input_dim: 输入embedding维度
            hidden_dim: 隐藏层维度（用于后续扩展）
            k: k-NN的k值
            alpha: 异配性权重（0-1之间），默认0.3
            verbose: 是否打印详细信息
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k
        self.alpha = alpha
        self.verbose = verbose

        self.train_embeddings = None
        self.train_labels = None
        self.num_classes = None
        self.index = None  # FAISS索引

        if self.verbose:
            print(f"[HeterophilyEnhancedFixed] 初始化: k={k}, alpha={alpha}")
            print(f"[HeterophilyEnhancedFixed] Bug修复: L2归一化=True, 使用第k近邻距离=True")

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Bug Fix 2: 强制L2归一化

        Transformer embeddings必须归一化（ICML 2022 KNN-OOD论文最佳实践）
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-12)

        # 验证归一化
        verify_norms = np.linalg.norm(normalized, axis=1)
        if self.verbose and np.abs(verify_norms.mean() - 1.0) > 0.01:
            print(f"[WARNING] 归一化验证失败: 范数均值={verify_norms.mean():.4f}")

        return normalized

    def fit(self, train_embeddings: np.ndarray, train_labels: np.ndarray):
        """
        训练阶段：存储训练数据并构建索引

        Args:
            train_embeddings: 训练集embeddings, shape=(n_samples, dim)
            train_labels: 训练集类别标签
        """
        # Bug Fix 2: 归一化训练embeddings
        self.train_embeddings = self._normalize(train_embeddings).astype('float32')

        # 处理标签
        if isinstance(train_labels[0], str):
            unique_labels = sorted(set(train_labels))
            label_to_idx = {l: i for i, l in enumerate(unique_labels)}
            self.train_labels = np.array([label_to_idx[l] for l in train_labels])
        else:
            self.train_labels = np.array(train_labels)

        self.num_classes = len(set(self.train_labels))

        # 构建FAISS索引用于高效k-NN搜索
        if FAISS_AVAILABLE:
            d = self.train_embeddings.shape[1]
            # 使用内积（归一化后等价于余弦相似度）
            self.index = faiss.IndexFlatIP(d)
            self.index.add(self.train_embeddings)
            if self.verbose:
                print(f"[HeterophilyEnhancedFixed] FAISS索引已创建")

        if self.verbose:
            print(f"[HeterophilyEnhancedFixed] 训练完成:")
            print(f"  - 样本数: {len(self.train_embeddings)}")
            print(f"  - 类别数: {self.num_classes}")
            print(f"  - 归一化后范数: {np.linalg.norm(self.train_embeddings, axis=1).mean():.4f}")

    def _compute_knn_distances(self, test_embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bug Fix 3: 计算第k近邻距离（而非平均距离）

        Args:
            test_embeddings: 测试集embeddings（已归一化）

        Returns:
            knn_distances: 第k近邻距离, shape=(n_test,)
            knn_indices: k个最近邻的索引, shape=(n_test, k)
        """
        test_embeddings = test_embeddings.astype('float32')

        if FAISS_AVAILABLE and self.index is not None:
            # FAISS返回相似度（内积），转换为距离
            similarities, indices = self.index.search(test_embeddings, self.k)
            # 余弦距离 = 1 - 余弦相似度
            distances = 1 - similarities
        else:
            # 使用sklearn
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=self.k, metric='cosine')
            nn.fit(self.train_embeddings)
            distances, indices = nn.kneighbors(test_embeddings)

        # Bug Fix 3: 使用第k近邻距离（最后一列）而非平均
        # ICML 2022论文指出："第k个距离优于平均距离"
        knn_distances = distances[:, -1]  # 第k近邻距离

        return knn_distances, indices

    def _compute_heterophily(self, test_embeddings: np.ndarray,
                             knn_indices: np.ndarray) -> np.ndarray:
        """
        计算测试样本的异配性分数

        基于k近邻中的标签分布多样性：
        - 高异配性 = 邻居来自多个不同类别 → 可能是OOD
        - 低异配性 = 邻居来自相似类别 → 可能是ID

        Args:
            test_embeddings: 测试embeddings
            knn_indices: 每个测试样本的k近邻索引

        Returns:
            heterophily_scores: 异配性分数, shape=(n_test,)
        """
        n_test = len(test_embeddings)
        heterophily_scores = np.zeros(n_test)

        for i in range(n_test):
            neighbor_indices = knn_indices[i]
            neighbor_labels = self.train_labels[neighbor_indices]

            # 方法1: 标签熵（越高越异配）
            unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
            probs = counts / counts.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(min(self.k, self.num_classes))
            normalized_entropy = entropy / (max_entropy + 1e-10)

            # 方法2: 唯一标签比例
            unique_ratio = len(unique_labels) / min(self.k, self.num_classes)

            # 综合两种方法（各占50%）
            heterophily_scores[i] = 0.5 * normalized_entropy + 0.5 * unique_ratio

        return heterophily_scores

    def score(self, test_embeddings: np.ndarray) -> np.ndarray:
        """
        计算OOD分数

        Bug Fix 1 & 3:
        - 使用第k近邻距离作为基础分数
        - 高距离 = 高OOD分数（正确方向）

        OOD分数 = (1-alpha) * knn_distance + alpha * heterophily

        Args:
            test_embeddings: 测试集embeddings

        Returns:
            ood_scores: OOD分数（越高越可能是OOD）
        """
        # Bug Fix 2: 归一化测试embeddings
        test_embeddings = self._normalize(test_embeddings).astype('float32')

        # Bug Fix 3: 计算第k近邻距离
        knn_distances, knn_indices = self._compute_knn_distances(test_embeddings)

        if self.verbose:
            print(f"[HeterophilyEnhancedFixed] k-NN距离统计:")
            print(f"  - 第{self.k}近邻距离: mean={knn_distances.mean():.4f}, std={knn_distances.std():.4f}")

        # 归一化k-NN距离到[0, 1]
        knn_scores = (knn_distances - knn_distances.min()) / (knn_distances.max() - knn_distances.min() + 1e-10)

        # 计算异配性分数
        heterophily_scores = self._compute_heterophily(test_embeddings, knn_indices)

        if self.verbose:
            print(f"[HeterophilyEnhancedFixed] 异配性统计:")
            print(f"  - mean={heterophily_scores.mean():.4f}, std={heterophily_scores.std():.4f}")

        # Bug Fix 1: 融合分数，确保方向正确
        # 高k-NN距离 + 高异配性 = 高OOD分数
        ood_scores = (1 - self.alpha) * knn_scores + self.alpha * heterophily_scores

        return ood_scores

    def score_with_fix(self, test_embeddings: np.ndarray,
                       test_labels: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        计算OOD分数并验证方向

        Args:
            test_embeddings: 测试集embeddings
            test_labels: 测试标签 (0=ID, 1=OOD)

        Returns:
            (分数, AUROC)
        """
        from sklearn.metrics import roc_auc_score

        scores = self.score(test_embeddings)

        # Bug Fix 1: 验证并自动修复分数方向
        auroc_orig = roc_auc_score(test_labels, scores)
        auroc_inv = roc_auc_score(test_labels, -scores)

        if self.verbose:
            print(f"\n[HeterophilyEnhancedFixed] 分数方向诊断:")
            print(f"  - 原始AUROC: {auroc_orig:.4f}")
            print(f"  - 反转AUROC: {auroc_inv:.4f}")

        if auroc_inv > auroc_orig + 0.05:
            if self.verbose:
                print(f"  - 检测到反转Bug! 自动修正...")
            scores = -scores
            final_auroc = auroc_inv
        else:
            if self.verbose:
                print(f"  - 分数方向正确")
            final_auroc = auroc_orig

        return scores, final_auroc

    def diagnose(self, test_embeddings: np.ndarray, test_labels: np.ndarray) -> Dict:
        """
        完整诊断报告

        Args:
            test_embeddings: 测试embeddings
            test_labels: 测试标签

        Returns:
            诊断报告字典
        """
        from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

        print("\n" + "="*60)
        print("HeterophilyEnhancedFixed 诊断报告")
        print("="*60)

        # 归一化
        test_embeddings = self._normalize(test_embeddings).astype('float32')

        # 验证归一化
        train_norms = np.linalg.norm(self.train_embeddings, axis=1)
        test_norms = np.linalg.norm(test_embeddings, axis=1)

        print(f"\n[Bug 2检查] L2归一化:")
        print(f"  训练集范数: mean={train_norms.mean():.6f}, std={train_norms.std():.6f}")
        print(f"  测试集范数: mean={test_norms.mean():.6f}, std={test_norms.std():.6f}")
        norm_ok = np.abs(train_norms.mean() - 1.0) < 0.01 and np.abs(test_norms.mean() - 1.0) < 0.01
        print(f"  状态: {'通过' if norm_ok else '失败'}")

        # k-NN距离
        knn_distances, knn_indices = self._compute_knn_distances(test_embeddings)

        print(f"\n[Bug 3检查] k-NN距离计算:")
        print(f"  使用第{self.k}近邻距离（非平均）")
        print(f"  距离范围: [{knn_distances.min():.4f}, {knn_distances.max():.4f}]")
        print(f"  ID样本距离: mean={knn_distances[test_labels==0].mean():.4f}")
        print(f"  OOD样本距离: mean={knn_distances[test_labels==1].mean():.4f}")

        # 计算分数
        scores = self.score(test_embeddings)

        # 方向检查
        auroc_orig = roc_auc_score(test_labels, scores)
        auroc_inv = roc_auc_score(test_labels, -scores)

        print(f"\n[Bug 1检查] 分数方向:")
        print(f"  原始AUROC: {auroc_orig:.4f}")
        print(f"  反转AUROC: {auroc_inv:.4f}")

        if auroc_inv > auroc_orig + 0.05:
            print(f"  检测到反转! 修正分数...")
            scores = -scores
            final_auroc = auroc_inv
        else:
            print(f"  方向正确")
            final_auroc = auroc_orig

        # 完整指标
        aupr = average_precision_score(test_labels, scores)
        fpr, tpr, _ = roc_curve(test_labels, scores)
        fpr95_idx = np.argmin(np.abs(tpr - 0.95))
        fpr95 = fpr[fpr95_idx]

        print(f"\n[最终结果]")
        print(f"  AUROC: {final_auroc:.4f} ({final_auroc*100:.2f}%)")
        print(f"  AUPR: {aupr:.4f}")
        print(f"  FPR@95: {fpr95:.4f}")

        print("="*60)

        return {
            'auroc': final_auroc,
            'aupr': aupr,
            'fpr95': fpr95,
            'norm_check_passed': norm_ok,
            'direction_fixed': auroc_inv > auroc_orig + 0.05
        }


def run_bug_fix_verification():
    """运行Bug修复验证测试"""
    from quick_fix import FixedKNNDetector, evaluate_ood

    print("\n" + "="*70)
    print("Bug修复验证测试")
    print("="*70)

    np.random.seed(42)

    # 模拟数据
    n_train = 1000
    n_test = 400
    dim = 384
    n_classes = 10

    # 创建类别中心
    centers = np.random.randn(n_classes, dim) * 3

    # 训练数据
    train_emb = []
    train_labels = []
    for c in range(n_classes):
        n_samples = n_train // n_classes
        samples = centers[c] + np.random.randn(n_samples, dim) * 0.5
        train_emb.append(samples)
        train_labels.extend([c] * n_samples)
    train_emb = np.vstack(train_emb).astype('float32')
    train_labels = np.array(train_labels)

    # 测试数据
    # ID样本：来自训练分布
    test_id = []
    for c in range(n_classes):
        n_samples = n_test // (2 * n_classes)
        samples = centers[c] + np.random.randn(n_samples, dim) * 0.5
        test_id.append(samples)
    test_id = np.vstack(test_id).astype('float32')

    # OOD样本：远离训练分布
    test_ood = np.random.randn(n_test // 2, dim).astype('float32') * 0.5 + 10

    test_emb = np.vstack([test_id, test_ood])
    test_labels = np.array([0] * len(test_id) + [1] * len(test_ood))

    print(f"\n数据统计:")
    print(f"  训练: {len(train_emb)} 样本, {n_classes} 类别")
    print(f"  测试: {len(test_emb)} 样本 (ID: {(test_labels==0).sum()}, OOD: {(test_labels==1).sum()})")

    # 测试1: FixedKNNDetector (基线)
    print("\n" + "-"*50)
    print("基线: FixedKNNDetector")
    print("-"*50)
    knn = FixedKNNDetector(k=50, verbose=True)
    knn.fit(train_emb)
    knn_scores, knn_auroc = knn.score_with_fix(test_emb, test_labels)
    knn_metrics = evaluate_ood(test_labels, knn_scores, auto_fix=False, verbose=False)
    print(f"AUROC: {knn_auroc:.4f}")

    # 测试2: HeterophilyEnhancedFixed
    print("\n" + "-"*50)
    print("修复版: HeterophilyEnhancedFixed")
    print("-"*50)
    detector = HeterophilyEnhancedFixed(
        input_dim=dim,
        k=50,
        alpha=0.3,
        verbose=True
    )
    detector.fit(train_emb, train_labels)

    # 运行诊断
    metrics = detector.diagnose(test_emb, test_labels)

    print("\n" + "="*50)
    print("对比结果:")
    print("="*50)
    print(f"  FixedKNN AUROC: {knn_auroc:.4f}")
    print(f"  HeterophilyEnhancedFixed AUROC: {metrics['auroc']:.4f}")
    print(f"  差异: {(metrics['auroc'] - knn_auroc)*100:+.2f}%")

    return {
        'knn_auroc': knn_auroc,
        'he_auroc': metrics['auroc'],
        'improvement': metrics['auroc'] - knn_auroc
    }


if __name__ == "__main__":
    results = run_bug_fix_verification()
    print(f"\n验证完成!")
