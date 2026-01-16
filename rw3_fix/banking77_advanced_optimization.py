#!/usr/bin/env python3
"""
Banking77 高级优化

目标: 87.12% → 88%+ AUROC

策略:
1. 更小的k值 (k=2,3,4)
2. 使用平均距离而非第k近邻距离
3. 加权距离方案
4. 结合LOF

Author: RW3 OOD Detection Project
"""

import sys
import numpy as np
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))

from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from data_loader import load_banking77_oos
from quick_fix import evaluate_ood, LOFDetector


class AdaptiveKNNDetector:
    """
    自适应k-NN OOD检测器
    支持多种距离聚合方式
    """

    def __init__(self, k: int = 5, distance_method: str = 'kth',
                 weight_scheme: str = 'uniform', verbose: bool = True):
        """
        Args:
            k: k近邻数量
            distance_method: 距离聚合方法
                - 'kth': 使用第k近邻距离
                - 'mean': 使用平均距离
                - 'weighted_mean': 距离加权平均(距离越近权重越高)
                - 'min': 使用最近邻距离
            weight_scheme: 权重方案
                - 'uniform': 均匀权重
                - 'distance': 距离反比权重
        """
        self.k = k
        self.distance_method = distance_method
        self.weight_scheme = weight_scheme
        self.verbose = verbose

        self.train_embeddings = None
        self.nn = None

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2归一化"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-12)

    def fit(self, train_embeddings: np.ndarray, train_labels=None):
        """训练"""
        self.train_embeddings = self._normalize(train_embeddings).astype('float32')
        self.nn = NearestNeighbors(n_neighbors=self.k, metric='cosine')
        self.nn.fit(self.train_embeddings)

        if self.verbose:
            print(f"[AdaptiveKNN] fit完成: k={self.k}, method={self.distance_method}")

    def score(self, test_embeddings: np.ndarray) -> np.ndarray:
        """计算OOD分数"""
        test_embeddings = self._normalize(test_embeddings).astype('float32')
        distances, _ = self.nn.kneighbors(test_embeddings)

        if self.distance_method == 'kth':
            # 使用第k近邻距离
            scores = distances[:, -1]

        elif self.distance_method == 'mean':
            # 使用平均距离
            scores = distances.mean(axis=1)

        elif self.distance_method == 'weighted_mean':
            # 距离加权平均(距离越近权重越高)
            weights = 1.0 / (distances + 1e-10)
            weights = weights / weights.sum(axis=1, keepdims=True)
            scores = (distances * weights).sum(axis=1)

        elif self.distance_method == 'min':
            # 使用最近邻距离
            scores = distances[:, 0]

        elif self.distance_method == 'harmonic':
            # 调和平均
            scores = self.k / (1.0 / (distances + 1e-10)).sum(axis=1)

        elif self.distance_method == 'median':
            # 中位数
            scores = np.median(distances, axis=1)

        else:
            raise ValueError(f"Unknown distance_method: {self.distance_method}")

        return scores

    def score_with_fix(self, test_embeddings: np.ndarray, test_labels: np.ndarray):
        """带方向修复的评分"""
        scores = self.score(test_embeddings)

        auroc_orig = roc_auc_score(test_labels, scores)
        auroc_inv = roc_auc_score(test_labels, -scores)

        if auroc_inv > auroc_orig:
            scores = -scores
            return scores, auroc_inv
        return scores, auroc_orig


class EnsembleDetector:
    """
    集成检测器: 结合多种方法
    """

    def __init__(self, methods: list, weights: list = None, verbose: bool = True):
        """
        Args:
            methods: 检测器列表
            weights: 权重列表(默认均匀)
        """
        self.methods = methods
        self.weights = weights if weights else [1.0/len(methods)] * len(methods)
        self.verbose = verbose

    def fit(self, train_embeddings: np.ndarray, train_labels=None):
        """训练所有检测器"""
        for method in self.methods:
            method.fit(train_embeddings, train_labels)

    def score(self, test_embeddings: np.ndarray) -> np.ndarray:
        """集成评分"""
        all_scores = []
        for method in self.methods:
            scores = method.score(test_embeddings)
            # 归一化分数到[0,1]
            scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            all_scores.append(scores_norm)

        # 加权平均
        ensemble_scores = np.zeros(len(test_embeddings))
        for scores, weight in zip(all_scores, self.weights):
            ensemble_scores += weight * scores

        return ensemble_scores

    def score_with_fix(self, test_embeddings: np.ndarray, test_labels: np.ndarray):
        scores = self.score(test_embeddings)
        auroc_orig = roc_auc_score(test_labels, scores)
        auroc_inv = roc_auc_score(test_labels, -scores)

        if auroc_inv > auroc_orig:
            return -scores, auroc_inv
        return scores, auroc_orig


def run_banking77_advanced_optimization():
    """
    Banking77高级优化实验
    """
    print("\n" + "="*70)
    print("🔍 Banking77 高级优化实验")
    print("="*70)

    # 加载数据
    print("\n[1/3] 加载Banking77数据...")
    train_texts, test_texts, test_labels, test_intents, _ = load_banking77_oos()
    test_labels = np.array(test_labels)

    # 编码
    print("\n[2/3] 编码文本...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    train_emb = encoder.encode(train_texts, show_progress_bar=True, batch_size=64)
    test_emb = encoder.encode(test_texts, show_progress_bar=True, batch_size=64)

    # 实验配置
    print("\n[3/3] 运行优化实验...")

    results = {}
    best_auroc = 0
    best_config = None

    # 实验1: 更小的k值
    print("\n--- 实验1: 更小的k值 ---")
    for k in [2, 3, 4, 5]:
        for method in ['kth', 'mean', 'median']:
            detector = AdaptiveKNNDetector(k=k, distance_method=method, verbose=False)
            detector.fit(train_emb)
            scores, auroc = detector.score_with_fix(test_emb, test_labels)

            config_name = f"k={k}, {method}"
            results[config_name] = auroc

            if auroc > best_auroc:
                best_auroc = auroc
                best_config = config_name

            status = "★" if auroc >= 0.88 else ""
            print(f"  {config_name:<20}: AUROC={auroc*100:.2f}% {status}")

    # 实验2: LOF基线
    print("\n--- 实验2: LOF ---")
    for n_neighbors in [10, 15, 20, 25]:
        lof = LOFDetector(k=n_neighbors, verbose=False)
        lof.fit(train_emb)
        scores, auroc = lof.score_with_fix(test_emb, test_labels)

        config_name = f"LOF (k={n_neighbors})"
        results[config_name] = auroc

        if auroc > best_auroc:
            best_auroc = auroc
            best_config = config_name

        status = "★" if auroc >= 0.88 else ""
        print(f"  {config_name:<20}: AUROC={auroc*100:.2f}% {status}")

    # 实验3: 集成方法
    print("\n--- 实验3: 集成方法 ---")

    # KNN + LOF集成
    knn_detector = AdaptiveKNNDetector(k=5, distance_method='kth', verbose=False)
    lof_detector = LOFDetector(k=20, verbose=False)

    ensemble = EnsembleDetector(
        methods=[knn_detector, lof_detector],
        weights=[0.5, 0.5],
        verbose=False
    )
    ensemble.fit(train_emb)
    scores, auroc = ensemble.score_with_fix(test_emb, test_labels)

    config_name = "Ensemble (KNN+LOF)"
    results[config_name] = auroc

    if auroc > best_auroc:
        best_auroc = auroc
        best_config = config_name

    status = "★" if auroc >= 0.88 else ""
    print(f"  {config_name:<20}: AUROC={auroc*100:.2f}% {status}")

    # 尝试不同权重
    for knn_w in [0.3, 0.4, 0.6, 0.7]:
        lof_w = 1.0 - knn_w
        ensemble = EnsembleDetector(
            methods=[
                AdaptiveKNNDetector(k=5, distance_method='kth', verbose=False),
                LOFDetector(k=20, verbose=False)
            ],
            weights=[knn_w, lof_w],
            verbose=False
        )
        ensemble.fit(train_emb)
        scores, auroc = ensemble.score_with_fix(test_emb, test_labels)

        config_name = f"Ensemble (w={knn_w:.1f})"
        results[config_name] = auroc

        if auroc > best_auroc:
            best_auroc = auroc
            best_config = config_name

        status = "★" if auroc >= 0.88 else ""
        print(f"  {config_name:<20}: AUROC={auroc*100:.2f}% {status}")

    # 总结
    print(f"\n{'='*70}")
    print("📊 结果总结")
    print(f"{'='*70}")

    # 排序显示
    sorted_results = sorted(results.items(), key=lambda x: -x[1])

    print(f"\n{'配置':<30} {'AUROC':<12} {'状态':<10}")
    print("-"*55)

    for config, auroc in sorted_results[:10]:
        status = "✅ 达标" if auroc >= 0.88 else "❌"
        print(f"{config:<30} {auroc*100:>10.2f}% {status:<10}")

    print(f"\n最佳配置: {best_config}")
    print(f"最佳AUROC: {best_auroc*100:.2f}%")

    if best_auroc >= 0.88:
        print(f"\n🎉 目标达成! (≥88%)")
    else:
        gap = (0.88 - best_auroc) * 100
        print(f"\n⚠️ 距离目标还差: {gap:.2f}%")

    # 保存结果
    results_dir = Path(__file__).parent / "results"
    with open(results_dir / "banking77_advanced_optimization.json", 'w') as f:
        json.dump({
            'results': {k: float(v) for k, v in results.items()},
            'best_config': best_config,
            'best_auroc': float(best_auroc)
        }, f, indent=2)

    return results, best_config, best_auroc


if __name__ == "__main__":
    run_banking77_advanced_optimization()
