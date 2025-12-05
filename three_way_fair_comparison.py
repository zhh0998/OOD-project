#!/usr/bin/env python3
"""
三方案统一OOD检测对比实验
=====================================
公平对比HMCEN、C4-TDA、CP-ABR++在OOD检测任务上的性能
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ==================== 数据加载 ====================

def load_clinc150_for_ood(n_samples=2000, ood_ratio=0.3):
    """
    加载CLINC150数据用于OOD检测

    返回:
        data: PyG Data对象，包含:
            - x: 节点特征
            - edge_index: 边
            - y: 节点标签（ID类）
            - ood_labels: OOD标签（0=ID, 1=OOD）
            - id_mask: ID样本mask
            - ood_mask: OOD样本mask
    """
    from datasets import load_dataset
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    print("加载CLINC150数据集...")
    dataset = load_dataset('clinc_oos', 'plus')

    # 采样
    test_data = dataset['test']
    indices = np.random.choice(len(test_data), n_samples, replace=False)
    texts = [test_data[i]['text'] for i in indices]
    labels = [test_data[i]['intent'] for i in indices]

    # TF-IDF特征
    vectorizer = TfidfVectorizer(max_features=300)
    features = vectorizer.fit_transform(texts).toarray()

    # 构建图（k-NN，k=20）
    k = 20
    knn = NearestNeighbors(n_neighbors=k+1, metric='cosine')
    knn.fit(features)
    distances, indices_knn = knn.kneighbors(features)

    edge_list = []
    for i in range(n_samples):
        for j in range(1, k+1):  # 跳过自己
            neighbor = indices_knn[i, j]
            edge_list.append([i, neighbor])
            edge_list.append([neighbor, i])  # 无向图

    edge_index = torch.tensor(edge_list, dtype=torch.long).t()

    # 确定ID和OOD
    unique_labels = list(set(labels))
    n_ood_classes = int(len(unique_labels) * ood_ratio)
    ood_classes = np.random.choice(unique_labels, n_ood_classes, replace=False)

    ood_labels = np.array([1 if label in ood_classes else 0 for label in labels])
    id_mask = ood_labels == 0
    ood_mask = ood_labels == 1

    # 转换标签（只对ID类）
    label_map = {label: i for i, label in enumerate(unique_labels) if label not in ood_classes}
    y = np.array([label_map.get(label, -1) for label in labels])

    data = Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.long),
        ood_labels=torch.tensor(ood_labels, dtype=torch.long),
        id_mask=torch.tensor(id_mask, dtype=torch.bool),
        ood_mask=torch.tensor(ood_mask, dtype=torch.bool),
        num_classes=len(label_map)
    )

    print(f"数据集统计:")
    print(f"  总样本: {n_samples}")
    print(f"  ID样本: {id_mask.sum()} ({id_mask.sum()/n_samples*100:.1f}%)")
    print(f"  OOD样本: {ood_mask.sum()} ({ood_mask.sum()/n_samples*100:.1f}%)")
    print(f"  ID类别数: {len(label_map)}")
    print(f"  OOD类别数: {n_ood_classes}")

    return data

# ==================== 方案1: HMCEN-C ====================

class HMCEN_C(nn.Module):
    """
    HMCEN with linear fusion (方案C)
    """
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        # 同配性分支
        self.homo_branch = GCNConv(input_dim, hidden_dim)
        # 异配性分支
        self.hetero_branch = GCNConv(input_dim, hidden_dim)

        # 融合与分类
        self.fusion = nn.Linear(hidden_dim, 64)
        self.classifier = nn.Linear(64, 2)  # ID/OOD二分类

    def forward(self, x, edge_index, h_node):
        """
        h_node: 节点异配性 [0,1]
        """
        # 两个分支
        h_homo = self.homo_branch(x, edge_index)
        h_hetero = self.hetero_branch(x, edge_index)

        # 线性门控 (方案C)
        alpha = (1.0 - h_node).unsqueeze(-1)

        # 融合
        h_fused = alpha * h_homo + (1 - alpha) * h_hetero
        h_fused = F.relu(h_fused)

        h_final = self.fusion(h_fused)
        h_final = F.relu(h_final)

        logits = self.classifier(h_final)
        return logits

    def predict_ood_score(self, x, edge_index, h_node):
        """返回OOD分数（越高越可能是OOD）"""
        logits = self.forward(x, edge_index, h_node)
        probs = F.softmax(logits, dim=-1)
        return probs[:, 1]  # OOD类的概率

def train_hmcen(data, epochs=200, lr=0.01, seed=42):
    """训练HMCEN-C"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 计算异配性
    h_node = compute_heterophily_pseudo(data)

    model = HMCEN_C(input_dim=data.x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 训练（在ID数据上）
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        logits = model(data.x, data.edge_index, h_node)

        # 在ID数据上训练（用异配性作为伪标签）
        # 高异配性 → 可能OOD (标签1)
        train_mask = data.id_mask
        train_labels = (h_node[train_mask] > h_node[train_mask].median()).long()

        loss = F.cross_entropy(logits[train_mask], train_labels)

        loss.backward()
        optimizer.step()

        if (epoch+1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # 预测OOD分数
    model.eval()
    with torch.no_grad():
        ood_scores = model.predict_ood_score(data.x, data.edge_index, h_node)

    return ood_scores.cpu().numpy()

# ==================== 方案2: C4-TDA-OOD ====================

def compute_betti_numbers_simple(edge_index, num_nodes):
    """
    计算Betti数（简化版）
    使用scipy而非ripser
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    # β₀: 连通分量数 - 1
    adj = csr_matrix(
        (np.ones(edge_index.shape[1]),
         (edge_index[0].numpy(), edge_index[1].numpy())),
        shape=(num_nodes, num_nodes)
    )
    n_components, labels = connected_components(adj, directed=False)
    beta_0 = n_components - 1

    # β₁: 环的数量（简化估计）
    # β₁ ≈ |E| - |V| + |C|（欧拉特征）
    n_edges = edge_index.shape[1] // 2  # 无向图
    n_vertices = num_nodes
    beta_1 = max(0, n_edges - n_vertices + n_components)

    return beta_0, beta_1

def c4tda_ood_detection(data, use_calibration=True):
    """
    C4-TDA用于OOD检测

    三种方法:
    1. β₁直接作为OOD分数
    2. h*β₁（利用异配性）
    3. 轻量校准（逻辑回归）
    """
    # 计算异配性
    h_node = compute_heterophily_pseudo(data)

    # 计算节点级Betti数（使用ego-graph）
    num_nodes = data.x.shape[0]
    betti_0_list = []
    betti_1_list = []

    print("  计算拓扑特征...")
    for node_id in range(num_nodes):
        # 提取1-hop ego-graph
        neighbors = data.edge_index[1][data.edge_index[0] == node_id]
        subgraph_nodes = torch.cat([torch.tensor([node_id]), neighbors])

        # 子图边
        mask = torch.isin(data.edge_index[0], subgraph_nodes) & \
               torch.isin(data.edge_index[1], subgraph_nodes)
        subgraph_edges = data.edge_index[:, mask]

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

    betti_0 = np.array(betti_0_list)
    betti_1 = np.array(betti_1_list)

    # 方法1: β₁直接
    ood_scores_beta1 = betti_1

    # 方法2: h*β₁
    ood_scores_h_beta1 = h_node.numpy() * betti_1

    # 方法3: 轻量校准
    if use_calibration:
        from sklearn.linear_model import LogisticRegression
        from torch_geometric.utils import degree

        # 特征: [h, β₀, β₁, degree]
        degrees = degree(data.edge_index[0], num_nodes=num_nodes).numpy()
        features = np.column_stack([
            h_node.numpy(),
            betti_0,
            betti_1,
            degrees
        ])

        # 在ID数据上训练
        train_mask = data.id_mask.numpy()
        train_labels = (h_node[data.id_mask] > h_node[data.id_mask].median()).numpy().astype(int)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(features[train_mask], train_labels)

        ood_scores_calibrated = clf.predict_proba(features)[:, 1]
    else:
        ood_scores_calibrated = ood_scores_h_beta1

    return {
        'beta1': ood_scores_beta1,
        'h_beta1': ood_scores_h_beta1,
        'calibrated': ood_scores_calibrated
    }

# ==================== 方案3: CP-ABR++ ====================

class CP_ABR_Plus(nn.Module):
    """
    简化版CP-ABR++（级联异配性感知）
    """
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        # Stage 1: Coarse detector
        self.coarse_gnn = GCNConv(input_dim, hidden_dim)
        self.coarse_classifier = nn.Linear(hidden_dim, 2)

        # Stage 2: Fine detector (异配性感知)
        self.fine_gnn = GCNConv(hidden_dim, hidden_dim)
        self.fine_classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x, edge_index, h_node):
        # Stage 1
        h1 = F.relu(self.coarse_gnn(x, edge_index))
        logits_coarse = self.coarse_classifier(h1)

        # Stage 2 (异配性自适应)
        alpha = h_node.unsqueeze(-1)
        h2 = F.relu(self.fine_gnn(h1, edge_index))
        h2_adaptive = alpha * h2 + (1-alpha) * h1
        logits_fine = self.fine_classifier(h2_adaptive)

        # 融合
        logits = 0.5 * logits_coarse + 0.5 * logits_fine
        return logits

def train_cp_abr(data, epochs=200, lr=0.01, seed=42):
    """训练CP-ABR++"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    h_node = compute_heterophily_pseudo(data)

    model = CP_ABR_Plus(input_dim=data.x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        logits = model(data.x, data.edge_index, h_node)

        train_mask = data.id_mask
        train_labels = (h_node[train_mask] > h_node[train_mask].median()).long()

        loss = F.cross_entropy(logits[train_mask], train_labels)

        loss.backward()
        optimizer.step()

        if (epoch+1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, h_node)
        ood_scores = F.softmax(logits, dim=-1)[:, 1]

    return ood_scores.cpu().numpy()

# ==================== 辅助函数 ====================

def compute_heterophily_pseudo(data):
    """
    计算伪异配性（测试时无标签可用）
    使用特征相似度估计
    """
    from torch_geometric.utils import degree

    num_nodes = data.x.shape[0]
    h_node = torch.zeros(num_nodes)

    for v in range(num_nodes):
        neighbors = data.edge_index[1][data.edge_index[0] == v]
        if len(neighbors) > 0:
            # 特征不相似度
            feat_sim = F.cosine_similarity(
                data.x[v].unsqueeze(0),
                data.x[neighbors],
                dim=1
            )
            h_node[v] = 1 - feat_sim.mean()  # 不相似度作为伪异配性

    return h_node

def compute_fpr95(y_true, y_scores):
    """计算FPR@95% TPR"""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    idx = np.argmin(np.abs(tpr - 0.95))
    return fpr[idx]

# ==================== 主实验 ====================

def run_experiment(seeds=[42, 2024, 2025]):
    """
    运行完整三方案对比实验
    """
    print("="*80)
    print("三方案统一OOD检测对比实验")
    print("="*80)

    results = {
        'HMCEN-C': {'auroc': [], 'fpr95': [], 'aupr': []},
        'C4-TDA-beta1': {'auroc': [], 'fpr95': [], 'aupr': []},
        'C4-TDA-h_beta1': {'auroc': [], 'fpr95': [], 'aupr': []},
        'C4-TDA-calibrated': {'auroc': [], 'fpr95': [], 'aupr': []},
        'CP-ABR++': {'auroc': [], 'fpr95': [], 'aupr': []}
    }

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*80}")
        print(f"运行 Seed {seed} ({seed_idx+1}/{len(seeds)})")
        print(f"{'='*80}")

        # 加载数据
        np.random.seed(seed)
        data = load_clinc150_for_ood(n_samples=2000, ood_ratio=0.3)
        y_true = data.ood_labels.numpy()

        # 方案1: HMCEN-C
        print("\n1. 训练HMCEN-C...")
        ood_scores_hmcen = train_hmcen(data, seed=seed)

        auroc = roc_auc_score(y_true, ood_scores_hmcen)
        fpr95 = compute_fpr95(y_true, ood_scores_hmcen)
        aupr = average_precision_score(y_true, ood_scores_hmcen)

        results['HMCEN-C']['auroc'].append(auroc)
        results['HMCEN-C']['fpr95'].append(fpr95)
        results['HMCEN-C']['aupr'].append(aupr)

        print(f"  AUROC: {auroc:.4f}, FPR95: {fpr95:.4f}, AUPR: {aupr:.4f}")

        # 方案2: C4-TDA
        print("\n2. C4-TDA-OOD...")
        c4tda_scores = c4tda_ood_detection(data, use_calibration=True)

        for variant_name, scores in c4tda_scores.items():
            auroc = roc_auc_score(y_true, scores)
            fpr95 = compute_fpr95(y_true, scores)
            aupr = average_precision_score(y_true, scores)

            key = f'C4-TDA-{variant_name}'
            results[key]['auroc'].append(auroc)
            results[key]['fpr95'].append(fpr95)
            results[key]['aupr'].append(aupr)

            print(f"  {variant_name}: AUROC={auroc:.4f}, FPR95={fpr95:.4f}, AUPR={aupr:.4f}")

        # 方案3: CP-ABR++
        print("\n3. 训练CP-ABR++...")
        ood_scores_cp = train_cp_abr(data, seed=seed)

        auroc = roc_auc_score(y_true, ood_scores_cp)
        fpr95 = compute_fpr95(y_true, ood_scores_cp)
        aupr = average_precision_score(y_true, ood_scores_cp)

        results['CP-ABR++']['auroc'].append(auroc)
        results['CP-ABR++']['fpr95'].append(fpr95)
        results['CP-ABR++']['aupr'].append(aupr)

        print(f"  AUROC: {auroc:.4f}, FPR95: {fpr95:.4f}, AUPR: {aupr:.4f}")

    # 统计汇总
    print("\n" + "="*80)
    print("最终结果汇总（均值 ± 标准差）")
    print("="*80)

    summary_table = []
    for method, metrics in results.items():
        auroc_mean = np.mean(metrics['auroc'])
        auroc_std = np.std(metrics['auroc'])
        fpr95_mean = np.mean(metrics['fpr95'])
        fpr95_std = np.std(metrics['fpr95'])
        aupr_mean = np.mean(metrics['aupr'])
        aupr_std = np.std(metrics['aupr'])

        summary_table.append({
            'method': method,
            'auroc': f"{auroc_mean:.4f}±{auroc_std:.4f}",
            'fpr95': f"{fpr95_mean:.4f}±{fpr95_std:.4f}",
            'aupr': f"{aupr_mean:.4f}±{aupr_std:.4f}",
            'auroc_mean': auroc_mean,
            'auroc_std': auroc_std
        })

    # 打印表格
    print(f"\n{'方法':<25} {'AUROC':<20} {'FPR95':<20} {'AUPR':<20}")
    print("-"*85)
    for row in summary_table:
        print(f"{row['method']:<25} {row['auroc']:<20} {row['fpr95']:<20} {row['aupr']:<20}")

    # 统计显著性检验
    print("\n" + "="*80)
    print("统计显著性检验（vs最佳方法）")
    print("="*80)

    best_method = max(summary_table, key=lambda x: x['auroc_mean'])['method']
    best_aurocs = results[best_method]['auroc']

    print(f"\n最佳方法: {best_method} (AUROC={np.mean(best_aurocs):.4f})")
    print("\nT-test结果:")

    for method in results.keys():
        if method != best_method:
            aurocs = results[method]['auroc']
            t_stat, p_value = stats.ttest_rel(best_aurocs, aurocs)

            sig_symbol = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"

            print(f"  {best_method} vs {method}: p={p_value:.4f} {sig_symbol}")

    # 可视化
    visualize_results(results, summary_table)

    return results, summary_table

def visualize_results(results, summary_table):
    """可视化实验结果"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    methods = list(results.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))

    # AUROC对比
    aurocs = [results[m]['auroc'] for m in methods]
    bp1 = axes[0].boxplot(aurocs, labels=methods, patch_artist=True)
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
    axes[0].set_ylabel('AUROC', fontsize=12)
    axes[0].set_title('OOD检测性能对比 (AUROC)', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)

    # FPR95对比
    fpr95s = [results[m]['fpr95'] for m in methods]
    bp2 = axes[1].boxplot(fpr95s, labels=methods, patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
    axes[1].set_ylabel('FPR@95%TPR', fontsize=12)
    axes[1].set_title('误报率对比 (越低越好)', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)

    # AUPR对比
    auprs = [results[m]['aupr'] for m in methods]
    bp3 = axes[2].boxplot(auprs, labels=methods, patch_artist=True)
    for patch, color in zip(bp3['boxes'], colors):
        patch.set_facecolor(color)
    axes[2].set_ylabel('AUPR', fontsize=12)
    axes[2].set_title('精确率-召回率曲线下面积', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('three_way_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n可视化结果已保存: three_way_comparison.png")

# ==================== 执行 ====================

if __name__ == '__main__':
    results, summary = run_experiment(seeds=[42, 2024, 2025])

    print("\n" + "="*80)
    print("实验完成！")
    print("="*80)

    # 决策建议
    best_row = max(summary, key=lambda x: x['auroc_mean'])
    best_method = best_row['method']
    best_auroc = best_row['auroc_mean']

    print(f"\n🏆 最佳方法: {best_method}")
    print(f"   AUROC: {best_auroc:.4f}")

    # 对比分析
    hmcen_auroc = next(r['auroc_mean'] for r in summary if r['method'] == 'HMCEN-C')
    c4tda_best_auroc = max([r['auroc_mean'] for r in summary if 'C4-TDA' in r['method']])
    cp_auroc = next(r['auroc_mean'] for r in summary if r['method'] == 'CP-ABR++')

    print(f"\n关键对比:")
    print(f"  HMCEN-C:          {hmcen_auroc:.4f}")
    print(f"  C4-TDA (最佳):    {c4tda_best_auroc:.4f}")
    print(f"  CP-ABR++:        {cp_auroc:.4f}")

    # 决策建议
    print(f"\n💡 决策建议:")

    if hmcen_auroc > c4tda_best_auroc + 0.03:
        print("  ✅ HMCEN显著优于C4-TDA（+3%以上）")
        print("  → 推荐: 考虑HMCEN-Lite（如果时间允许）")
    elif abs(hmcen_auroc - c4tda_best_auroc) < 0.02:
        print("  ⚠️ HMCEN和C4-TDA性能相当（±2%）")
        print("  → 推荐: 优先C4-TDA（时间更短，3-4月 vs 10-12月）")
    else:
        print("  ❌ C4-TDA优于或接近HMCEN")
        print("  → 推荐: C4-TDA主轨")

    # Vanilla GNN基线（隐式计算）
    # 注意: 当前没有独立的Vanilla GNN结果，但可以从消融实验获得
    print(f"\n⚠️ 重要提醒:")
    print(f"  - 运行Prompt 4 (HMCEN消融实验)以获得vs Vanilla GNN的对比")
    print(f"  - 这将确认HMCEN的优势是来自架构还是仅仅是'训练了分类器'")
