#!/usr/bin/env python3
"""
C4-TDA转化能力验证实验
====================================
验证C4-TDA的理论发现能否转化为实用的OOD检测方法

核心问题: C4-TDA已经证明了理论假设（异配性↔Betti数，d=0.9378），
但这个发现能否转化为实用的OOD检测方法？

验证重点:
1. 直接应用能力（无需训练）
2. 轻量校准效果（逻辑回归）
3. 与HMCEN的真实差距
4. 跨数据集稳定性
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ==================== 数据加载 ====================

def load_clinc150_for_ood(n_samples=2000, ood_ratio=0.3):
    """加载CLINC150数据（与Prompt 1相同）"""
    from datasets import load_dataset
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    from torch_geometric.data import Data

    print("加载CLINC150数据集...")
    dataset = load_dataset('clinc_oos', 'plus')

    test_data = dataset['test']
    indices = np.random.choice(len(test_data), n_samples, replace=False)
    texts = [test_data[i]['text'] for i in indices]
    labels = [test_data[i]['intent'] for i in indices]

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=300)
    features = vectorizer.fit_transform(texts).toarray()

    # 构建图
    k = 20
    knn = NearestNeighbors(n_neighbors=k+1, metric='cosine')
    knn.fit(features)
    distances, indices_knn = knn.kneighbors(features)

    edge_list = []
    for i in range(n_samples):
        for j in range(1, k+1):
            neighbor = indices_knn[i, j]
            edge_list.append([i, neighbor])
            edge_list.append([neighbor, i])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t()

    # OOD划分
    unique_labels = list(set(labels))
    n_ood_classes = int(len(unique_labels) * ood_ratio)
    ood_classes = np.random.choice(unique_labels, n_ood_classes, replace=False)

    ood_labels = np.array([1 if label in ood_classes else 0 for label in labels])

    data = Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=edge_index,
        ood_labels=torch.tensor(ood_labels, dtype=torch.long)
    )

    print(f"数据集统计:")
    print(f"  总样本: {n_samples}")
    print(f"  ID: {(ood_labels==0).sum()} ({(ood_labels==0).sum()/n_samples*100:.1f}%)")
    print(f"  OOD: {(ood_labels==1).sum()} ({(ood_labels==1).sum()/n_samples*100:.1f}%)")

    return data

# ==================== 拓扑特征计算 ====================

def compute_betti_numbers_simple(edge_index, num_nodes):
    """计算Betti数（简化版）"""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    if edge_index.shape[1] == 0:
        return 0, 0

    adj = csr_matrix(
        (np.ones(edge_index.shape[1]),
         (edge_index[0].numpy(), edge_index[1].numpy())),
        shape=(num_nodes, num_nodes)
    )
    n_components, labels = connected_components(adj, directed=False)
    beta_0 = n_components - 1

    n_edges = edge_index.shape[1] // 2
    n_vertices = num_nodes
    beta_1 = max(0, n_edges - n_vertices + n_components)

    return beta_0, beta_1

def compute_node_betti_numbers(data):
    """计算所有节点的Betti数"""
    num_nodes = data.x.shape[0]
    betti_0_list = []
    betti_1_list = []

    print("计算节点级拓扑特征...")
    for node_id in range(num_nodes):
        if (node_id + 1) % 500 == 0:
            print(f"  进度: {node_id+1}/{num_nodes}")

        # 提取ego-graph
        neighbors = data.edge_index[1][data.edge_index[0] == node_id]
        if len(neighbors) == 0:
            betti_0_list.append(0)
            betti_1_list.append(0)
            continue

        subgraph_nodes = torch.cat([torch.tensor([node_id]), neighbors])

        # 子图边
        mask = (torch.isin(data.edge_index[0], subgraph_nodes) &
                torch.isin(data.edge_index[1], subgraph_nodes))
        subgraph_edges = data.edge_index[:, mask]

        if subgraph_edges.shape[1] == 0:
            betti_0_list.append(0)
            betti_1_list.append(0)
            continue

        # 重新编号
        node_map = {n.item(): i for i, n in enumerate(subgraph_nodes)}
        subgraph_edges_reindexed = torch.tensor([
            [node_map[e[0].item()] for e in subgraph_edges.t()],
            [node_map[e[1].item()] for e in subgraph_edges.t()]
        ])

        # 计算Betti数
        beta_0, beta_1 = compute_betti_numbers_simple(
            subgraph_edges_reindexed,
            len(subgraph_nodes)
        )

        betti_0_list.append(beta_0)
        betti_1_list.append(beta_1)

    return np.array(betti_0_list), np.array(betti_1_list)

def compute_heterophily_pseudo(data):
    """计算伪异配性"""
    num_nodes = data.x.shape[0]
    h_node = torch.zeros(num_nodes)

    for v in range(num_nodes):
        neighbors = data.edge_index[1][data.edge_index[0] == v]
        if len(neighbors) > 0:
            feat_sim = F.cosine_similarity(
                data.x[v].unsqueeze(0),
                data.x[neighbors],
                dim=1
            )
            h_node[v] = 1 - feat_sim.mean()

    return h_node

def compute_clustering_coefficient(data):
    """计算节点聚类系数"""
    num_nodes = data.x.shape[0]
    clustering = np.zeros(num_nodes)

    print("计算聚类系数...")
    for node_id in range(num_nodes):
        if (node_id + 1) % 500 == 0:
            print(f"  进度: {node_id+1}/{num_nodes}")

        neighbors = data.edge_index[1][data.edge_index[0] == node_id].numpy()
        k = len(neighbors)

        if k < 2:
            clustering[node_id] = 0
            continue

        # 邻居间的边数 - 使用集合优化
        neighbor_set = set(neighbors)
        neighbor_edges = 0

        for n in neighbors:
            n_neighbors = data.edge_index[1][data.edge_index[0] == n].numpy()
            for nn in n_neighbors:
                if nn in neighbor_set and nn > n:  # 避免重复计数
                    neighbor_edges += 1

        # 聚类系数
        max_edges = k * (k - 1) / 2
        clustering[node_id] = neighbor_edges / max_edges if max_edges > 0 else 0

    return clustering

# ==================== 三种转化方法 ====================

def method1_direct_beta1(betti_1):
    """方法1: β₁直接作为OOD分数"""
    return betti_1

def method2_h_times_beta1(h_node, betti_1):
    """方法2: h*β₁（结合异配性）"""
    return h_node.numpy() * betti_1

def method3_calibrated(data, h_node, betti_0, betti_1):
    """方法3: 轻量校准（逻辑回归）"""
    from torch_geometric.utils import degree

    # 构建特征
    degrees = degree(data.edge_index[0], num_nodes=data.x.shape[0]).numpy()

    # 额外特征：聚类系数
    clustering = compute_clustering_coefficient(data)

    features = np.column_stack([
        h_node.numpy(),
        betti_0,
        betti_1,
        degrees,
        clustering
    ])

    # 标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 在ID数据上训练
    id_mask = (data.ood_labels == 0).numpy()

    # 伪标签：高异配性→潜在OOD
    median_h = np.median(h_node[id_mask].numpy())
    train_labels = (h_node[id_mask].numpy() > median_h).astype(int)

    # 逻辑回归
    clf = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )
    clf.fit(features_scaled[id_mask], train_labels)

    # 预测OOD分数
    ood_scores = clf.predict_proba(features_scaled)[:, 1]

    return ood_scores, clf, scaler

def compute_fpr95(y_true, y_scores):
    """计算FPR@95% TPR"""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    idx = np.argmin(np.abs(tpr - 0.95))
    return fpr[idx]

# ==================== 完整实验 ====================

def run_c4tda_transformation_experiment(datasets=['CLINC150']):
    """
    完整的C4-TDA转化验证实验
    """
    print("="*80)
    print("C4-TDA假设到应用的转化验证实验")
    print("="*80)

    all_results = {}

    for dataset_name in datasets:
        print(f"\n{'='*80}")
        print(f"数据集: {dataset_name}")
        print(f"{'='*80}")

        # 加载数据
        if dataset_name == 'CLINC150':
            data = load_clinc150_for_ood()
        # 可以添加其他数据集

        y_true = data.ood_labels.numpy()

        # 计算拓扑特征
        print("\n步骤1: 计算拓扑特征")
        h_node = compute_heterophily_pseudo(data)
        betti_0, betti_1 = compute_node_betti_numbers(data)

        print(f"\n拓扑特征统计:")
        print(f"  异配性: {h_node.mean():.4f} ± {h_node.std():.4f}")
        print(f"  β₀: {betti_0.mean():.4f} ± {betti_0.std():.4f}")
        print(f"  β₁: {betti_1.mean():.4f} ± {betti_1.std():.4f}")

        # ID vs OOD统计
        id_mask = (y_true == 0)
        ood_mask = (y_true == 1)

        print(f"\nID vs OOD特征对比:")
        print(f"  ID异配性:  {h_node[id_mask].mean():.4f} ± {h_node[id_mask].std():.4f}")
        print(f"  OOD异配性: {h_node[ood_mask].mean():.4f} ± {h_node[ood_mask].std():.4f}")
        print(f"  ID β₁:    {betti_1[id_mask].mean():.4f} ± {betti_1[id_mask].std():.4f}")
        print(f"  OOD β₁:   {betti_1[ood_mask].mean():.4f} ± {betti_1[ood_mask].std():.4f}")

        # 方法1: β₁直接
        print("\n步骤2: 方法1 - β₁直接作为OOD分数")
        ood_scores_beta1 = method1_direct_beta1(betti_1)
        auroc_beta1 = roc_auc_score(y_true, ood_scores_beta1)
        fpr95_beta1 = compute_fpr95(y_true, ood_scores_beta1)
        aupr_beta1 = average_precision_score(y_true, ood_scores_beta1)

        print(f"  AUROC: {auroc_beta1:.4f}")
        print(f"  FPR95: {fpr95_beta1:.4f}")
        print(f"  AUPR:  {aupr_beta1:.4f}")

        # 方法2: h*β₁
        print("\n步骤3: 方法2 - h*β₁（结合异配性）")
        ood_scores_h_beta1 = method2_h_times_beta1(h_node, betti_1)
        auroc_h_beta1 = roc_auc_score(y_true, ood_scores_h_beta1)
        fpr95_h_beta1 = compute_fpr95(y_true, ood_scores_h_beta1)
        aupr_h_beta1 = average_precision_score(y_true, ood_scores_h_beta1)

        print(f"  AUROC: {auroc_h_beta1:.4f}")
        print(f"  FPR95: {fpr95_h_beta1:.4f}")
        print(f"  AUPR:  {aupr_h_beta1:.4f}")
        print(f"  提升: {(auroc_h_beta1 - auroc_beta1):.4f} ({(auroc_h_beta1/auroc_beta1-1)*100:+.1f}%)")

        # 方法3: 轻量校准
        print("\n步骤4: 方法3 - 轻量校准（逻辑回归）")
        ood_scores_calibrated, clf, scaler = method3_calibrated(data, h_node, betti_0, betti_1)
        auroc_calibrated = roc_auc_score(y_true, ood_scores_calibrated)
        fpr95_calibrated = compute_fpr95(y_true, ood_scores_calibrated)
        aupr_calibrated = average_precision_score(y_true, ood_scores_calibrated)

        print(f"\n  AUROC: {auroc_calibrated:.4f}")
        print(f"  FPR95: {fpr95_calibrated:.4f}")
        print(f"  AUPR:  {aupr_calibrated:.4f}")
        print(f"  vs β₁: {(auroc_calibrated - auroc_beta1):.4f} ({(auroc_calibrated/auroc_beta1-1)*100:+.1f}%)")
        print(f"  vs h*β₁: {(auroc_calibrated - auroc_h_beta1):.4f} ({(auroc_calibrated/auroc_h_beta1-1)*100:+.1f}%)")

        # 特征重要性
        print("\n  特征重要性:")
        feature_names = ['异配性', 'β₀', 'β₁', '度数', '聚类系数']
        importances = np.abs(clf.coef_[0])
        sorted_idx = np.argsort(importances)[::-1]
        for i, idx in enumerate(sorted_idx):
            print(f"    {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

        # 存储结果
        all_results[dataset_name] = {
            'beta1': {
                'auroc': auroc_beta1,
                'fpr95': fpr95_beta1,
                'aupr': aupr_beta1
            },
            'h_beta1': {
                'auroc': auroc_h_beta1,
                'fpr95': fpr95_h_beta1,
                'aupr': aupr_h_beta1
            },
            'calibrated': {
                'auroc': auroc_calibrated,
                'fpr95': fpr95_calibrated,
                'aupr': aupr_calibrated
            },
            'h_node': h_node,
            'betti_0': betti_0,
            'betti_1': betti_1,
            'ood_scores': {
                'beta1': ood_scores_beta1,
                'h_beta1': ood_scores_h_beta1,
                'calibrated': ood_scores_calibrated
            },
            'y_true': y_true,
            'feature_importances': dict(zip(feature_names, importances))
        }

    return all_results

# ==================== 与HMCEN对比 ====================

def compare_with_hmcen(c4tda_results, hmcen_auroc=0.8207):
    """
    与HMCEN结果对比
    """
    print("\n" + "="*80)
    print("C4-TDA vs HMCEN 性能对比")
    print("="*80)

    for dataset_name, results in c4tda_results.items():
        print(f"\n数据集: {dataset_name}")
        print("-"*80)

        print(f"\n{'方法':<30} {'AUROC':<12} {'vs HMCEN':<15} {'判断':<20}")
        print("-"*80)

        methods = [
            ('C4-TDA (β₁直接)', results['beta1']['auroc']),
            ('C4-TDA (h*β₁)', results['h_beta1']['auroc']),
            ('C4-TDA (校准)', results['calibrated']['auroc']),
            ('HMCEN-C', hmcen_auroc)
        ]

        for method_name, auroc in methods:
            gap = auroc - hmcen_auroc
            gap_pct = gap / hmcen_auroc * 100

            if method_name == 'HMCEN-C':
                judgment = "基准"
            elif abs(gap) < 0.02:
                judgment = "性能相当 ✅"
            elif gap < -0.05:
                judgment = "显著落后 ❌"
            elif gap < 0:
                judgment = "略有差距 ⚠️"
            else:
                judgment = "超越HMCEN ⭐"

            print(f"{method_name:<30} {auroc:.4f}      {gap:+.4f} ({gap_pct:+.1f}%)   {judgment:<20}")

    # 关键结论
    print("\n" + "="*80)
    print("关键结论:")
    print("="*80)

    best_c4tda_auroc = max(
        results['beta1']['auroc'],
        results['h_beta1']['auroc'],
        results['calibrated']['auroc']
    )

    gap = best_c4tda_auroc - hmcen_auroc

    print(f"\nC4-TDA最佳方法 AUROC: {best_c4tda_auroc:.4f}")
    print(f"HMCEN-C AUROC: {hmcen_auroc:.4f}")
    print(f"差距: {gap:.4f} ({gap/hmcen_auroc*100:+.1f}%)")

    print(f"\n💡 决策建议:")
    if abs(gap) < 0.02:
        print("  ⚠️ C4-TDA校准后与HMCEN性能相当（±2%）")
        print("  → 推荐: 优先C4-TDA")
        print("  → 理由: 时间更短（3-4月 vs 10-12月）+ 理论假设已验证(d=0.9378)")
    elif gap < -0.05:
        print("  ❌ C4-TDA显著落后HMCEN（>5%）")
        print("  → 推荐: 考虑HMCEN-Lite（如果时间允许）")
        print("  → 注意: 需要Prompt 4确认HMCEN优势来源（架构 vs 训练分类器）")
    elif gap < 0:
        print("  ⚠️ C4-TDA略逊于HMCEN（2-5%差距）")
        print("  → 推荐: 权衡性能提升 vs 时间成本")
        print("  → 建议: 双轨并行（主推C4-TDA，探索HMCEN-Lite）")
    else:
        print("  ⭐ C4-TDA超越HMCEN")
        print("  → 推荐: 全力C4-TDA主轨")
        print("  → 优势: 性能优 + 理论严格 + 时间短")

    return gap, best_c4tda_auroc

# ==================== 可视化 ====================

def visualize_transformation_results(results):
    """可视化C4-TDA转化结果"""
    # Use non-interactive backend for saving
    import matplotlib
    matplotlib.use('Agg')

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    dataset_name = list(results.keys())[0]
    res = results[dataset_name]

    # 图1: AUROC对比
    methods = ['β₁直接', 'h*β₁', '校准', 'HMCEN-C']
    aurocs = [
        res['beta1']['auroc'],
        res['h_beta1']['auroc'],
        res['calibrated']['auroc'],
        0.8207  # HMCEN结果
    ]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

    axes[0, 0].bar(methods, aurocs, color=colors, alpha=0.7)
    axes[0, 0].axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, label='Baseline 0.7')
    axes[0, 0].set_ylabel('AUROC', fontsize=12)
    axes[0, 0].set_title('C4-TDA Three Methods vs HMCEN Comparison', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0.4, 1.0)

    # 添加数值标签
    for i, (method, auroc) in enumerate(zip(methods, aurocs)):
        axes[0, 0].text(i, auroc + 0.02, f'{auroc:.3f}', ha='center', fontsize=10)

    # 图2: 异配性 vs β₁ 散点图
    h_node = res['h_node'].numpy()
    betti_1 = res['betti_1']
    y_true = res['y_true']

    id_mask = (y_true == 0)
    ood_mask = (y_true == 1)

    axes[0, 1].scatter(h_node[id_mask], betti_1[id_mask],
                      alpha=0.3, s=20, c='blue', label='ID samples')
    axes[0, 1].scatter(h_node[ood_mask], betti_1[ood_mask],
                      alpha=0.3, s=20, c='red', label='OOD samples')
    axes[0, 1].set_xlabel('Heterophily h(v)', fontsize=12)
    axes[0, 1].set_ylabel('Betti number β₁', fontsize=12)
    axes[0, 1].set_title('Heterophily-Betti Relationship (Original Hypothesis)', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 图3: OOD分数分布 - β₁
    axes[1, 0].hist(res['ood_scores']['beta1'][id_mask], bins=50,
                    alpha=0.5, color='blue', label='ID (β₁)', density=True)
    axes[1, 0].hist(res['ood_scores']['beta1'][ood_mask], bins=50,
                    alpha=0.5, color='red', label='OOD (β₁)', density=True)
    axes[1, 0].set_xlabel('OOD Score', fontsize=12)
    axes[1, 0].set_ylabel('Density', fontsize=12)
    axes[1, 0].set_title('β₁ Direct as OOD Score Distribution', fontsize=14)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 图4: 校准后的OOD分数分布
    axes[1, 1].hist(res['ood_scores']['calibrated'][id_mask], bins=50,
                    alpha=0.5, color='blue', label='ID (Calibrated)', density=True)
    axes[1, 1].hist(res['ood_scores']['calibrated'][ood_mask], bins=50,
                    alpha=0.5, color='red', label='OOD (Calibrated)', density=True)
    axes[1, 1].set_xlabel('OOD Score', fontsize=12)
    axes[1, 1].set_ylabel('Density', fontsize=12)
    axes[1, 1].set_title('Lightweight Calibrated OOD Score Distribution', fontsize=14)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('c4tda_transformation_analysis.png', dpi=300, bbox_inches='tight')
    print("\n可视化结果已保存: c4tda_transformation_analysis.png")

def visualize_feature_importance(results):
    """可视化特征重要性"""
    import matplotlib
    matplotlib.use('Agg')

    dataset_name = list(results.keys())[0]
    res = results[dataset_name]

    fig, ax = plt.subplots(figsize=(10, 6))

    feature_names = list(res['feature_importances'].keys())
    importances = list(res['feature_importances'].values())

    # 排序
    sorted_idx = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_importances = [importances[i] for i in sorted_idx]

    colors = ['#e74c3c' if 'β' in name else '#3498db' for name in sorted_names]

    bars = ax.barh(sorted_names, sorted_importances, color=colors, alpha=0.7)
    ax.set_xlabel('Feature Importance (|coefficient|)', fontsize=12)
    ax.set_title('Logistic Regression Feature Importance for OOD Detection', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')

    # 添加数值标签
    for bar, imp in zip(bars, sorted_importances):
        ax.text(imp + 0.01, bar.get_y() + bar.get_height()/2,
                f'{imp:.3f}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('c4tda_feature_importance.png', dpi=300, bbox_inches='tight')
    print("特征重要性已保存: c4tda_feature_importance.png")

def generate_summary_table(results, hmcen_auroc=0.8207):
    """生成汇总表格"""
    print("\n" + "="*80)
    print("汇总表格")
    print("="*80)

    dataset_name = list(results.keys())[0]
    res = results[dataset_name]

    print(f"\n{'方法':<25} {'AUROC':<10} {'FPR95':<10} {'AUPR':<10} {'判断':<20}")
    print("-"*75)

    methods_data = [
        ('C4-TDA (β₁直接)', res['beta1']),
        ('C4-TDA (h*β₁)', res['h_beta1']),
        ('C4-TDA (校准)', res['calibrated']),
    ]

    for name, data in methods_data:
        gap = data['auroc'] - hmcen_auroc
        if abs(gap) < 0.02:
            judgment = "✅ 相当"
        elif gap < -0.05:
            judgment = "❌ 显著落后"
        elif gap < 0:
            judgment = "⚠️ 略有差距"
        else:
            judgment = "⭐ 超越"

        print(f"{name:<25} {data['auroc']:.4f}     {data['fpr95']:.4f}     {data['aupr']:.4f}     {judgment}")

    print(f"{'HMCEN-C (基准)':<25} {hmcen_auroc:.4f}     {'N/A':<10} {'N/A':<10} {'基准'}")
    print("-"*75)

# ==================== 判断逻辑 ====================

def make_final_decision(results, hmcen_auroc=0.8207):
    """根据实验结果做出最终决策"""
    print("\n" + "="*80)
    print("最终决策分析")
    print("="*80)

    dataset_name = list(results.keys())[0]
    res = results[dataset_name]

    beta1_auroc = res['beta1']['auroc']
    calibrated_auroc = res['calibrated']['auroc']

    # 判断1: 直接应用能力
    print("\n判断1: 直接应用能力")
    if beta1_auroc >= 0.7:
        print(f"  ✅ β₁ AUROC = {beta1_auroc:.4f} ≥ 0.7")
        print("  → 假设可以直接用于OOD检测")
        direct_judgment = "可直接使用"
    elif beta1_auroc >= 0.6:
        print(f"  ⚠️ β₁ AUROC = {beta1_auroc:.4f}，在0.6-0.7范围")
        print("  → 需要额外校准")
        direct_judgment = "需要校准"
    else:
        print(f"  ❌ β₁ AUROC = {beta1_auroc:.4f} < 0.6")
        print("  → 假设验证成功，但转化困难")
        direct_judgment = "转化困难"

    # 判断2: 校准后与HMCEN对比
    print("\n判断2: 校准后与HMCEN对比")
    gap = calibrated_auroc - hmcen_auroc
    gap_pct = abs(gap / hmcen_auroc * 100)

    if abs(gap) <= 0.02:
        print(f"  ✅ 差距 = {gap:.4f} ({gap_pct:.1f}%) ≤ 2%")
        print("  → 性能相当，优先C4-TDA（时间短）")
        comparison_judgment = "性能相当"
    elif gap < -0.05:
        print(f"  ❌ 差距 = {gap:.4f} ({gap_pct:.1f}%) > 5%")
        print("  → HMCEN有实质性优势")
        comparison_judgment = "HMCEN更优"
    elif gap < 0:
        print(f"  ⚠️ 差距 = {gap:.4f} ({gap_pct:.1f}%)，在2-5%范围")
        print("  → 需要权衡性能 vs 时间")
        comparison_judgment = "略逊一筹"
    else:
        print(f"  ⭐ C4-TDA超越HMCEN! 差距 = {gap:.4f} ({gap_pct:.1f}%)")
        comparison_judgment = "C4-TDA更优"

    # 判断3: 特征重要性
    print("\n判断3: 特征重要性分析")
    beta1_importance = res['feature_importances']['β₁']
    total_importance = sum(res['feature_importances'].values())
    beta1_ratio = beta1_importance / total_importance * 100

    if beta1_ratio > 40:
        print(f"  ✅ β₁重要性 = {beta1_ratio:.1f}% > 40%")
        print("  → 拓扑特征确实有用")
        feature_judgment = "拓扑特征有效"
    elif beta1_ratio > 20:
        print(f"  ⚠️ β₁重要性 = {beta1_ratio:.1f}%，在20-40%范围")
        print("  → 拓扑特征有一定贡献")
        feature_judgment = "拓扑特征有贡献"
    else:
        print(f"  ⚠️ β₁重要性 = {beta1_ratio:.1f}% < 20%")
        print("  → 其他特征更重要，C4-TDA核心价值被稀释")
        feature_judgment = "核心价值稀释"

    # 综合决策
    print("\n" + "="*80)
    print("综合决策")
    print("="*80)

    print(f"\n1. 直接应用能力: {direct_judgment}")
    print(f"2. 与HMCEN对比: {comparison_judgment}")
    print(f"3. 特征重要性: {feature_judgment}")

    # 最终推荐
    print("\n" + "-"*40)
    print("最终推荐:")
    print("-"*40)

    if comparison_judgment in ["性能相当", "C4-TDA更优"]:
        print("\n🎯 推荐: C4-TDA主轨")
        print("   理由:")
        print("   - 性能与HMCEN相当或更优")
        print("   - 时间更短（3-4月 vs 10-12月）")
        print("   - 理论假设已验证(d=0.9378)")
    elif comparison_judgment == "略逊一筹":
        print("\n🎯 推荐: 双轨并行")
        print("   - 主推C4-TDA（理论完备+时间短）")
        print("   - 探索HMCEN-Lite（如果时间允许）")
        print("   - 关注Prompt 4结果（HMCEN优势来源分析）")
    else:
        print("\n🎯 推荐: 重新评估方案")
        print("   - HMCEN有明显性能优势")
        print("   - 考虑HMCEN-Lite方案")
        print("   - 或接受性能差距，坚持C4-TDA理论路线")

    return {
        'direct_judgment': direct_judgment,
        'comparison_judgment': comparison_judgment,
        'feature_judgment': feature_judgment,
        'beta1_auroc': beta1_auroc,
        'calibrated_auroc': calibrated_auroc,
        'gap': gap
    }

# ==================== 主函数 ====================

if __name__ == '__main__':
    print("\n开始C4-TDA转化能力验证实验...")
    print("="*80)

    # 运行实验
    results = run_c4tda_transformation_experiment(datasets=['CLINC150'])

    # 生成汇总表格
    generate_summary_table(results)

    # 与HMCEN对比
    gap, best_auroc = compare_with_hmcen(results, hmcen_auroc=0.8207)

    # 最终决策
    decision = make_final_decision(results)

    # 可视化
    visualize_transformation_results(results)
    visualize_feature_importance(results)

    print("\n" + "="*80)
    print("实验完成！")
    print("="*80)
    print("\n生成的文件:")
    print("  - c4tda_transformation_analysis.png")
    print("  - c4tda_feature_importance.png")
