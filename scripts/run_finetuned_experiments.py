#!/usr/bin/env python3
"""
使用微调特征运行OOD检测实验

从预提取的finetuned features运行检测器对比实验
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.quick_fix import evaluate_ood


class HeterophilyEnhancedFixed:
    """异配性增强检测器"""

    def __init__(self, k=5, alpha=0.3, distance_method='mean'):
        self.k = k
        self.alpha = alpha
        self.distance_method = distance_method

    def _normalize(self, embeddings):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-12)

    def fit(self, train_embeddings, train_labels):
        self.train_embeddings = self._normalize(train_embeddings).astype('float32')
        self.train_labels = train_labels
        self.nn = NearestNeighbors(n_neighbors=self.k, metric='cosine')
        self.nn.fit(self.train_embeddings)
        return self

    def _compute_heterophily(self, indices):
        n_test = len(indices)
        scores = np.zeros(n_test)
        n_classes = len(np.unique(self.train_labels))

        for i in range(n_test):
            neighbor_labels = self.train_labels[indices[i]]
            unique_labels = len(np.unique(neighbor_labels))
            scores[i] = unique_labels / min(self.k, n_classes)

        return scores

    def score(self, test_embeddings):
        test_embeddings = self._normalize(test_embeddings).astype('float32')
        distances, indices = self.nn.kneighbors(test_embeddings)

        # k-NN distance score
        if self.distance_method == 'kth':
            knn_scores = distances[:, -1]
        elif self.distance_method == 'mean':
            knn_scores = distances.mean(axis=1)
        else:
            knn_scores = distances[:, -1]

        # Normalize
        knn_scores = (knn_scores - knn_scores.min()) / (knn_scores.max() - knn_scores.min() + 1e-10)

        # Heterophily
        het_scores = self._compute_heterophily(indices)

        # Combine
        scores = (1 - self.alpha) * knn_scores + self.alpha * het_scores

        return scores


def run_with_features(features_path, dataset_name, detector_configs):
    """使用预提取特征运行实验"""

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Features: {features_path}")
    print(f"{'='*60}")

    # 加载特征
    data = np.load(features_path, allow_pickle=True)
    train_features = data['train_features']
    test_features = data['test_features']
    train_labels = data['train_labels']
    test_labels = data['test_labels']

    # 测试标签已经是二值 (0=ID, 1=OOD)
    binary_labels = test_labels.astype(int)

    print(f"  Train: {len(train_features)}")
    print(f"  Test: {len(test_features)} (ID: {(binary_labels==0).sum()}, OOD: {(binary_labels==1).sum()})")

    results = {}

    for config_name, config in detector_configs.items():
        detector = HeterophilyEnhancedFixed(**config)
        detector.fit(train_features, train_labels)
        scores = detector.score(test_features)

        # Auto-fix direction
        auroc_orig = roc_auc_score(binary_labels, scores)
        auroc_inv = roc_auc_score(binary_labels, -scores)

        if auroc_inv > auroc_orig:
            scores = -scores

        metrics = evaluate_ood(binary_labels, scores, auto_fix=False, verbose=False)

        results[config_name] = {
            'auroc': float(metrics['auroc']),
            'fpr95': float(metrics['fpr95']),
            'aupr': float(metrics['aupr'])
        }

        print(f"  {config_name}: AUROC={metrics['auroc']*100:.2f}%, FPR95={metrics['fpr95']*100:.2f}%")

    return results


def main():
    print("\n" + "="*70)
    print("Finetuned Features OOD Detection Experiments")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    features_dir = Path("features")

    # 检测器配置
    detector_configs = {
        'KNN (k=2, mean)': {'k': 2, 'alpha': 0.0, 'distance_method': 'mean'},
        'KNN (k=5, kth)': {'k': 5, 'alpha': 0.0, 'distance_method': 'kth'},
        'KNN (k=5, mean)': {'k': 5, 'alpha': 0.0, 'distance_method': 'mean'},
        'KNN (k=10, kth)': {'k': 10, 'alpha': 0.0, 'distance_method': 'kth'},
        'Het+KNN (k=10, α=0.3)': {'k': 10, 'alpha': 0.3, 'distance_method': 'mean'},
    }

    all_results = {}

    # Banking77
    banking77_path = features_dir / "banking77_finetuned_seed42.npz"
    if banking77_path.exists():
        all_results['banking77'] = run_with_features(
            banking77_path, "Banking77 (Finetuned)", detector_configs
        )

    # CLINC150
    clinc150_path = features_dir / "clinc150_native_oos_finetuned_seed42.npz"
    if clinc150_path.exists():
        all_results['clinc150'] = run_with_features(
            clinc150_path, "CLINC150 Native OOS (Finetuned)", detector_configs
        )

    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)

    for dataset, results in all_results.items():
        print(f"\n{dataset}:")
        best_config = max(results.items(), key=lambda x: x[1]['auroc'])
        print(f"  Best: {best_config[0]} - AUROC={best_config[1]['auroc']*100:.2f}%")

    return all_results


if __name__ == "__main__":
    main()
