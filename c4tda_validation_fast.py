#!/usr/bin/env python3
"""
C4-TDA Hypothesis to Application Validation (Optimized Version)
验证C4-TDA假设能否转化为实用的OOD检测方法
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict


# ============================================================
# Part 1: Data Generation (Simulating Intent Classification Datasets)
# ============================================================

class IntentDatasetSimulator:
    """模拟意图分类数据集（CLINC150, Banking77, ROSTD）"""

    DATASET_CONFIGS = {
        'CLINC150': {
            'n_id_classes': 150,
            'n_samples_per_class': 50,  # Reduced for speed
            'n_ood_samples': 500,
            'embedding_dim': 256,  # Reduced dimension
            'id_cluster_std': 0.3,
            'ood_spread_factor': 2.5,
        },
        'Banking77': {
            'n_id_classes': 77,
            'n_samples_per_class': 60,
            'n_ood_samples': 400,
            'embedding_dim': 256,
            'id_cluster_std': 0.35,
            'ood_spread_factor': 2.2,
        },
        'ROSTD': {
            'n_id_classes': 12,
            'n_samples_per_class': 200,
            'n_ood_samples': 600,
            'embedding_dim': 256,
            'id_cluster_std': 0.4,
            'ood_spread_factor': 2.0,
        }
    }

    def __init__(self, dataset_name, seed=42):
        self.dataset_name = dataset_name
        self.config = self.DATASET_CONFIGS[dataset_name]
        self.rng = np.random.RandomState(seed)

    def generate(self):
        """生成模拟数据集"""
        cfg = self.config

        # 生成ID类别中心
        class_centers = self.rng.randn(cfg['n_id_classes'], cfg['embedding_dim'])
        class_centers = class_centers / np.linalg.norm(class_centers, axis=1, keepdims=True)
        class_centers *= 3

        # 生成ID样本
        id_embeddings = []
        id_labels = []
        for i in range(cfg['n_id_classes']):
            samples = class_centers[i] + self.rng.randn(
                cfg['n_samples_per_class'], cfg['embedding_dim']
            ) * cfg['id_cluster_std']
            id_embeddings.append(samples)
            id_labels.extend([i] * cfg['n_samples_per_class'])

        id_embeddings = np.vstack(id_embeddings)
        id_labels = np.array(id_labels)

        # 生成OOD样本
        ood_embeddings = self._generate_ood_samples(class_centers, cfg)
        ood_labels = np.full(cfg['n_ood_samples'], -1)

        # 合并数据
        all_embeddings = np.vstack([id_embeddings, ood_embeddings])
        all_labels = np.concatenate([id_labels, ood_labels])
        is_ood = (all_labels == -1).astype(int)

        # 划分训练/测试
        n_id = len(id_labels)
        id_indices = np.arange(n_id)
        self.rng.shuffle(id_indices)
        n_id_train = int(n_id * 0.8)

        train_mask = np.zeros(len(all_labels), dtype=bool)
        train_mask[id_indices[:n_id_train]] = True

        test_mask = np.zeros(len(all_labels), dtype=bool)
        test_mask[id_indices[n_id_train:]] = True
        test_mask[n_id:] = True  # All OOD in test

        return {
            'embeddings': all_embeddings,
            'labels': all_labels,
            'is_ood': is_ood,
            'train_mask': train_mask,
            'test_mask': test_mask,
            'n_id_classes': cfg['n_id_classes'],
            'dataset_name': self.dataset_name
        }

    def _generate_ood_samples(self, class_centers, cfg):
        """生成OOD样本"""
        n_ood = cfg['n_ood_samples']
        dim = cfg['embedding_dim']
        ood_samples = []

        # Type 1: 类别边界区域 (40%)
        n_boundary = int(n_ood * 0.4)
        for _ in range(n_boundary):
            i, j = self.rng.choice(len(class_centers), 2, replace=False)
            alpha = self.rng.uniform(0.3, 0.7)
            sample = alpha * class_centers[i] + (1 - alpha) * class_centers[j]
            sample += self.rng.randn(dim) * cfg['id_cluster_std'] * cfg['ood_spread_factor']
            ood_samples.append(sample)

        # Type 2: 远离所有类别 (30%)
        n_far = int(n_ood * 0.3)
        global_center = class_centers.mean(axis=0)
        for _ in range(n_far):
            direction = self.rng.randn(dim)
            direction = direction / np.linalg.norm(direction)
            distance = self.rng.uniform(4, 6)
            sample = global_center + direction * distance
            sample += self.rng.randn(dim) * cfg['id_cluster_std']
            ood_samples.append(sample)

        # Type 3: 稀疏区域 (30%)
        n_sparse = n_ood - n_boundary - n_far
        for _ in range(n_sparse):
            sample = self.rng.randn(dim) * cfg['ood_spread_factor']
            sample += global_center * 0.5
            ood_samples.append(sample)

        return np.array(ood_samples)


# ============================================================
# Part 2: Graph Construction
# ============================================================

class GraphConstructor:
    """构建KNN图"""

    def __init__(self, k=10, metric='cosine'):
        self.k = k
        self.metric = metric

    def build(self, embeddings):
        """构建KNN图"""
        n = len(embeddings)

        if self.metric == 'cosine':
            normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
            nn = NearestNeighbors(n_neighbors=self.k + 1, metric='euclidean')
            nn.fit(normed)
            distances, indices = nn.kneighbors(normed)
        else:
            nn = NearestNeighbors(n_neighbors=self.k + 1, metric=self.metric)
            nn.fit(embeddings)
            distances, indices = nn.kneighbors(embeddings)

        edge_index = []
        edge_weights = []

        for i in range(n):
            for j_idx in range(1, self.k + 1):
                j = indices[i, j_idx]
                edge_index.append([i, j])
                edge_weights.append(distances[i, j_idx])

        edge_index = np.array(edge_index).T
        edge_weights = np.array(edge_weights)

        return {
            'edge_index': edge_index,
            'edge_weights': edge_weights,
            'n_nodes': n,
            'k': self.k,
            'indices': indices
        }


# ============================================================
# Part 3: Heterophily Computation
# ============================================================

def compute_node_heterophily(graph, labels):
    """计算每个节点的异配性"""
    edge_index = graph['edge_index']
    n_nodes = graph['n_nodes']

    neighbors = defaultdict(list)
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        neighbors[src].append(dst)

    heterophily = np.zeros(n_nodes)
    for v in range(n_nodes):
        if len(neighbors[v]) == 0:
            heterophily[v] = 0
            continue

        v_label = labels[v]
        diff_count = sum(1 for u in neighbors[v] if labels[u] != v_label)
        heterophily[v] = diff_count / len(neighbors[v])

    return heterophily


# ============================================================
# Part 4: Fast Betti Number Approximation
# ============================================================

def compute_local_betti_fast(embeddings, graph, sample_ratio=0.3):
    """
    快速计算局部Betti数（使用采样加速）
    """
    n_nodes = graph['n_nodes']
    indices = graph['indices']  # KNN indices

    betti_0 = np.zeros(n_nodes)
    betti_1 = np.zeros(n_nodes)

    # 对于每个节点，用其KNN邻域计算局部拓扑
    for v in range(n_nodes):
        # 使用KNN邻域（1-hop）
        local_nodes = indices[v, :graph['k']+1]  # v and its k neighbors

        if len(local_nodes) < 3:
            betti_0[v] = 1
            betti_1[v] = 0
            continue

        local_embeddings = embeddings[local_nodes]
        dist_matrix = cdist(local_embeddings, local_embeddings, metric='euclidean')

        # 简化的Betti计算
        b0, b1 = _compute_betti_simple(dist_matrix)
        betti_0[v] = b0
        betti_1[v] = b1

    return np.column_stack([betti_0, betti_1])


def _compute_betti_simple(dist_matrix, threshold_percentile=50):
    """简化的Betti数计算"""
    n = len(dist_matrix)
    upper_tri = dist_matrix[np.triu_indices(n, k=1)]

    if len(upper_tri) == 0:
        return 1, 0

    threshold = np.percentile(upper_tri, threshold_percentile)
    adj = (dist_matrix < threshold) & (dist_matrix > 0)
    adj = adj.astype(float)

    # β₀: 连通分量数
    n_components, _ = connected_components(csr_matrix(adj), directed=False)
    beta_0 = n_components

    # β₁: 使用欧拉特征估计
    n_vertices = n
    n_edges = np.sum(adj) / 2
    adj_sq = adj @ adj
    n_triangles = np.trace(adj_sq @ adj) / 6
    euler_char = n_vertices - n_edges + n_triangles
    beta_1 = max(0, beta_0 - euler_char)

    return beta_0, beta_1


def compute_persistence_fast(embeddings, graph):
    """快速计算持久性特征"""
    n_nodes = graph['n_nodes']
    indices = graph['indices']

    total_persistence = np.zeros(n_nodes)

    for v in range(n_nodes):
        local_nodes = indices[v, :graph['k']+1]

        if len(local_nodes) < 3:
            total_persistence[v] = 0
            continue

        local_embeddings = embeddings[local_nodes]
        dist_matrix = cdist(local_embeddings, local_embeddings, metric='euclidean')

        # 简化的持久性计算
        upper_tri = dist_matrix[np.triu_indices(len(dist_matrix), k=1)]
        if len(upper_tri) == 0:
            total_persistence[v] = 0
            continue

        # Total persistence = 边权重的变异
        total_persistence[v] = np.std(upper_tri)

    return total_persistence


# ============================================================
# Part 5: Graph Features
# ============================================================

def compute_degree(graph):
    """计算节点度数"""
    edge_index = graph['edge_index']
    n_nodes = graph['n_nodes']
    degree = np.zeros(n_nodes)
    for i in range(edge_index.shape[1]):
        degree[edge_index[0, i]] += 1
    return degree


def compute_clustering_coefficient(graph):
    """计算聚类系数"""
    edge_index = graph['edge_index']
    n_nodes = graph['n_nodes']

    neighbors = defaultdict(set)
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        neighbors[src].add(dst)

    clustering = np.zeros(n_nodes)
    for v in range(n_nodes):
        neighs = list(neighbors[v])
        k = len(neighs)
        if k < 2:
            clustering[v] = 0
            continue

        n_edges = 0
        for i in range(len(neighs)):
            for j in range(i + 1, len(neighs)):
                if neighs[j] in neighbors[neighs[i]]:
                    n_edges += 1

        max_edges = k * (k - 1) / 2
        clustering[v] = n_edges / max_edges if max_edges > 0 else 0

    return clustering


def compute_local_density(embeddings, graph):
    """计算局部密度"""
    n_nodes = graph['n_nodes']
    indices = graph['indices']

    density = np.zeros(n_nodes)
    for v in range(n_nodes):
        neighbors = indices[v, 1:graph['k']+1]  # Exclude self
        v_emb = embeddings[v]
        neigh_embs = embeddings[neighbors]
        dists = np.linalg.norm(neigh_embs - v_emb, axis=1)
        density[v] = 1 / (dists.mean() + 1e-8)

    return density


# ============================================================
# Part 6: OOD Detection Methods
# ============================================================

class C4TDADetector:
    """C4-TDA OOD检测器"""

    def __init__(self, method='beta1'):
        self.method = method
        self.scaler = StandardScaler()
        self.classifier = None

    def fit(self, features, labels=None):
        if self.method == 'calibrated' and labels is not None:
            self.scaler.fit(features)
            scaled_features = self.scaler.transform(features)
            self.classifier = LogisticRegression(max_iter=1000, class_weight='balanced')
            self.classifier.fit(scaled_features, labels)
        return self

    def compute_ood_scores(self, features, betti, heterophily, persistence=None):
        if self.method == 'beta1':
            return betti[:, 1]
        elif self.method == 'h_beta1':
            return heterophily * (betti[:, 1] + 1)
        elif self.method == 'persistence':
            return persistence if persistence is not None else betti[:, 1]
        elif self.method == 'calibrated':
            if self.classifier is None:
                raise ValueError("Classifier not fitted.")
            scaled_features = self.scaler.transform(features)
            return self.classifier.predict_proba(scaled_features)[:, 1]


class HMCENBaseline:
    """HMCEN简化版本"""

    def compute_ood_scores(self, embeddings, graph, labels, train_mask):
        n_nodes = graph['n_nodes']

        # 计算类别原型
        train_embeddings = embeddings[train_mask]
        train_labels = labels[train_mask]

        unique_labels = np.unique(train_labels[train_labels >= 0])
        prototypes = {}
        for l in unique_labels:
            mask = train_labels == l
            prototypes[l] = train_embeddings[mask].mean(axis=0)

        prototype_matrix = np.array([prototypes[l] for l in sorted(prototypes.keys())])

        # 计算到最近原型的距离
        distances = cdist(embeddings, prototype_matrix, metric='euclidean')
        min_distances = distances.min(axis=1)

        max_dist = min_distances.max()
        ood_scores = min_distances / (max_dist + 1e-8)

        # 结合异配性
        heterophily = compute_node_heterophily(graph, labels)
        combined_scores = 0.7 * ood_scores + 0.3 * heterophily

        return combined_scores


# ============================================================
# Part 7: Evaluation
# ============================================================

def compute_metrics(y_true, y_scores):
    """计算评估指标"""
    auroc = roc_auc_score(y_true, y_scores)

    # 最佳F1
    thresholds = np.percentile(y_scores, np.arange(0, 100, 5))
    best_f1 = 0
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1

    # FPR@95TPR
    sorted_indices = np.argsort(y_scores)[::-1]
    sorted_labels = y_true[sorted_indices]

    n_positive = y_true.sum()
    tpr_threshold = 0.95 * n_positive
    cumsum = np.cumsum(sorted_labels)
    idx_95tpr = np.searchsorted(cumsum, tpr_threshold)

    if idx_95tpr >= len(sorted_labels):
        fpr_at_95tpr = 1.0
    else:
        n_negative = len(y_true) - n_positive
        fp_at_95tpr = idx_95tpr - cumsum[min(idx_95tpr, len(cumsum)-1)]
        fpr_at_95tpr = fp_at_95tpr / n_negative if n_negative > 0 else 0

    return {'auroc': auroc, 'best_f1': best_f1, 'fpr_at_95tpr': fpr_at_95tpr}


# ============================================================
# Part 8: Main Validation Pipeline
# ============================================================

def validate_c4tda_on_dataset(dataset_name, seed=42, verbose=True):
    """在单个数据集上验证C4-TDA"""

    if verbose:
        print(f"\n{'='*60}")
        print(f"Validating on {dataset_name}")
        print('='*60)

    # 1. 生成数据
    simulator = IntentDatasetSimulator(dataset_name, seed=seed)
    data = simulator.generate()

    embeddings = data['embeddings']
    labels = data['labels']
    is_ood = data['is_ood']
    train_mask = data['train_mask']
    test_mask = data['test_mask']

    if verbose:
        print(f"Dataset: {data['dataset_name']}")
        print(f"  Total samples: {len(labels)}")
        print(f"  ID samples: {(labels >= 0).sum()}")
        print(f"  OOD samples: {(labels == -1).sum()}")

    # 2. 构建图
    graph_constructor = GraphConstructor(k=10, metric='cosine')
    graph = graph_constructor.build(embeddings)

    if verbose:
        print(f"Graph: {graph['n_nodes']} nodes, {graph['edge_index'].shape[1]} edges")

    # 3. 计算异配性
    heterophily = compute_node_heterophily(graph, labels)

    if verbose:
        print(f"Heterophily - ID mean: {heterophily[labels >= 0].mean():.4f}, OOD mean: {heterophily[labels == -1].mean():.4f}")

    # 4. 计算Betti数
    betti = compute_local_betti_fast(embeddings, graph)
    total_persistence = compute_persistence_fast(embeddings, graph)

    if verbose:
        print(f"β₁ - ID mean: {betti[labels >= 0, 1].mean():.4f}, OOD mean: {betti[labels == -1, 1].mean():.4f}")

    # 5. 计算图特征
    degrees = compute_degree(graph)
    clustering = compute_clustering_coefficient(graph)
    density = compute_local_density(embeddings, graph)

    # 6. 准备特征矩阵
    features = np.column_stack([
        heterophily,
        betti[:, 0],
        betti[:, 1],
        degrees,
        clustering,
        density,
        total_persistence
    ])

    # 7. 测试C4-TDA方法
    results = {}

    test_is_ood = is_ood[test_mask]
    test_features = features[test_mask]
    test_betti = betti[test_mask]
    test_heterophily = heterophily[test_mask]
    test_persistence = total_persistence[test_mask]

    # Method 1: β₁直接
    detector_beta1 = C4TDADetector(method='beta1')
    scores_beta1 = detector_beta1.compute_ood_scores(test_features, test_betti, test_heterophily)
    results['beta1'] = compute_metrics(test_is_ood, scores_beta1)

    # Method 2: h * β₁
    detector_h_beta1 = C4TDADetector(method='h_beta1')
    scores_h_beta1 = detector_h_beta1.compute_ood_scores(test_features, test_betti, test_heterophily)
    results['h_beta1'] = compute_metrics(test_is_ood, scores_h_beta1)

    # Method 3: Total Persistence
    detector_pers = C4TDADetector(method='persistence')
    scores_pers = detector_pers.compute_ood_scores(test_features, test_betti, test_heterophily, test_persistence)
    results['persistence'] = compute_metrics(test_is_ood, scores_pers)

    # Method 4: 校准（逻辑回归）
    train_features = features[train_mask]
    train_heterophily = heterophily[train_mask]
    median_h = np.median(train_heterophily)
    train_pseudo_labels = (train_heterophily > median_h).astype(int)

    detector_calib = C4TDADetector(method='calibrated')
    detector_calib.fit(train_features, train_pseudo_labels)
    scores_calib = detector_calib.compute_ood_scores(test_features, test_betti, test_heterophily)
    results['calibrated'] = compute_metrics(test_is_ood, scores_calib)

    # 8. HMCEN基线
    hmcen = HMCENBaseline()
    scores_hmcen = hmcen.compute_ood_scores(embeddings, graph, labels, train_mask)
    scores_hmcen_test = scores_hmcen[test_mask]
    results['hmcen'] = compute_metrics(test_is_ood, scores_hmcen_test)

    if verbose:
        print(f"\n{'Method':<20} {'AUROC':<10} {'F1':<10} {'FPR@95':<10}")
        print('-'*50)
        print(f"{'β₁ (direct)':<20} {results['beta1']['auroc']:.4f}     {results['beta1']['best_f1']:.4f}     {results['beta1']['fpr_at_95tpr']:.4f}")
        print(f"{'h × β₁':<20} {results['h_beta1']['auroc']:.4f}     {results['h_beta1']['best_f1']:.4f}     {results['h_beta1']['fpr_at_95tpr']:.4f}")
        print(f"{'Total Persistence':<20} {results['persistence']['auroc']:.4f}     {results['persistence']['best_f1']:.4f}     {results['persistence']['fpr_at_95tpr']:.4f}")
        print(f"{'Calibrated (LR)':<20} {results['calibrated']['auroc']:.4f}     {results['calibrated']['best_f1']:.4f}     {results['calibrated']['fpr_at_95tpr']:.4f}")
        print('-'*50)
        print(f"{'HMCEN (baseline)':<20} {results['hmcen']['auroc']:.4f}     {results['hmcen']['best_f1']:.4f}     {results['hmcen']['fpr_at_95tpr']:.4f}")

    return results


def run_full_validation():
    """运行完整的验证流程"""

    print("\n" + "="*70)
    print("          C4-TDA Hypothesis to Application Validation")
    print("="*70)
    print("\n背景: C4-TDA已验证异配性与Betti数高度相关 (Cohen's d=0.9378)")
    print("目标: 验证这个理论发现能否转化为实用的OOD检测方法")

    datasets = ['CLINC150', 'Banking77', 'ROSTD']
    all_results = {}

    for dataset_name in datasets:
        results = validate_c4tda_on_dataset(dataset_name, seed=42, verbose=True)
        all_results[dataset_name] = results

    # 汇总分析
    print("\n" + "="*70)
    print("                    Cross-Dataset Summary")
    print("="*70)

    methods = ['beta1', 'h_beta1', 'persistence', 'calibrated', 'hmcen']
    method_names = {
        'beta1': 'β₁ (direct)',
        'h_beta1': 'h × β₁',
        'persistence': 'Total Persistence',
        'calibrated': 'Calibrated (LR)',
        'hmcen': 'HMCEN (baseline)'
    }

    print(f"\n{'Method':<20} ", end="")
    for dataset in datasets:
        print(f"{dataset:<12} ", end="")
    print(f"{'Mean':<10} {'Std':<10}")
    print("-"*80)

    method_stats = {}
    for method in methods:
        aurocs = [all_results[d][method]['auroc'] for d in datasets]
        mean_auroc = np.mean(aurocs)
        std_auroc = np.std(aurocs)
        method_stats[method] = {'mean': mean_auroc, 'std': std_auroc, 'aurocs': aurocs}

        print(f"{method_names[method]:<20} ", end="")
        for auroc in aurocs:
            print(f"{auroc:.4f}       ", end="")
        print(f"{mean_auroc:.4f}     {std_auroc:.4f}")

    # 关键判断
    print("\n" + "="*70)
    print("                    Key Analysis & Conclusions")
    print("="*70)

    # 判断1: 直接应用能力
    beta1_mean = method_stats['beta1']['mean']
    print(f"\n【判断1: 直接应用能力】")
    print(f"  β₁ AUROC (mean): {beta1_mean:.4f}")
    if beta1_mean >= 0.70:
        print(f"  ✅ 假设可以直接应用 (AUROC ≥ 0.70)")
        direct_application = "strong"
    elif beta1_mean >= 0.65:
        print(f"  ⚠️ 假设可以部分应用 (0.65 ≤ AUROC < 0.70)")
        direct_application = "medium"
    else:
        print(f"  ❌ 直接应用效果有限 (AUROC < 0.65)")
        direct_application = "weak"

    # 判断2: 校准后与HMCEN对比
    calib_mean = method_stats['calibrated']['mean']
    hmcen_mean = method_stats['hmcen']['mean']
    gap = hmcen_mean - calib_mean

    print(f"\n【判断2: 校准后与HMCEN对比】")
    print(f"  Calibrated AUROC: {calib_mean:.4f}")
    print(f"  HMCEN AUROC:      {hmcen_mean:.4f}")
    print(f"  Gap:              {gap:.4f} ({abs(gap)/hmcen_mean*100:.1f}%)")

    if gap < 0.02:
        print(f"  ✅ C4-TDA校准后与HMCEN性能相当 (gap < 0.02)")
        comparison = "comparable"
    elif gap < 0.05:
        print(f"  ⚠️ HMCEN略优于C4-TDA (0.02 ≤ gap < 0.05)")
        comparison = "slightly_worse"
    else:
        print(f"  ❌ HMCEN显著优于C4-TDA (gap ≥ 0.05)")
        comparison = "significantly_worse"

    # 判断3: 跨数据集稳定性
    calib_std = method_stats['calibrated']['std']
    print(f"\n【判断3: 跨数据集稳定性】")
    print(f"  Calibrated Std: {calib_std:.4f}")

    if calib_std < 0.05:
        print(f"  ✅ 方法在多数据集上稳定 (std < 0.05)")
        stability = "good"
    elif calib_std < 0.10:
        print(f"  ⚠️ 方法稳定性中等 (0.05 ≤ std < 0.10)")
        stability = "medium"
    else:
        print(f"  ❌ 方法不稳定 (std ≥ 0.10)")
        stability = "poor"

    # 最佳C4-TDA方法
    best_c4tda_method = max(['beta1', 'h_beta1', 'persistence', 'calibrated'],
                            key=lambda m: method_stats[m]['mean'])
    best_c4tda_auroc = method_stats[best_c4tda_method]['mean']

    print(f"\n【最佳C4-TDA方法】")
    print(f"  Method: {method_names[best_c4tda_method]}")
    print(f"  Mean AUROC: {best_c4tda_auroc:.4f}")

    # 最终结论
    print("\n" + "="*70)
    print("                    Final Conclusions & Recommendations")
    print("="*70)

    print(f"\n1. C4-TDA假设转化能力: ", end="")
    if direct_application == "strong":
        print("【强】假设可以直接转化为实用方法")
    elif direct_application == "medium":
        print("【中】假设转化需要轻量校准")
    else:
        print("【弱】假设转化困难，需要额外工程")

    print(f"\n2. 校准后与HMCEN对比: ", end="")
    if comparison == "comparable":
        print("【相当】C4-TDA可作为HMCEN的高效替代")
    elif comparison == "slightly_worse":
        print("【略差】需权衡性能与效率")
    else:
        print("【显著差】HMCEN有实质性优势")

    print(f"\n3. 跨数据集稳定性: ", end="")
    if stability == "good":
        print("【好】方法泛化性强")
    elif stability == "medium":
        print("【中】方法在大部分数据集上有效")
    else:
        print("【差】方法数据集敏感")

    # 最终推荐
    print(f"\n4. 最终推荐: ", end="")

    if comparison == "comparable" or (comparison == "slightly_worse" and best_c4tda_auroc >= 0.75):
        print("【C4-TDA优先】")
        print(f"   → 推荐使用: {method_names[best_c4tda_method]}")
        print(f"   → 理由: 性能与HMCEN相当，但计算效率更高")
    elif comparison == "slightly_worse":
        print("【双轨并行】")
        print(f"   → C4-TDA适合: 时间敏感的场景")
        print(f"   → HMCEN适合: 性能优先的场景")
    else:
        print("【HMCEN-Lite优先】")
        print(f"   → 理由: HMCEN性能显著更优")
        print(f"   → C4-TDA价值: 理论洞察，可整合到HMCEN改进")

    return {
        'direct_application': direct_application,
        'comparison_with_hmcen': comparison,
        'stability': stability,
        'best_c4tda_method': best_c4tda_method,
        'best_c4tda_auroc': best_c4tda_auroc,
        'hmcen_auroc': hmcen_mean,
        'gap': gap,
        'all_results': all_results,
        'method_stats': method_stats
    }


if __name__ == "__main__":
    summary = run_full_validation()
    print("\n" + "="*70)
    print("                         Validation Complete")
    print("="*70)
