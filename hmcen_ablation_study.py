#!/usr/bin/env python3
"""
HMCEN消融实验
====================================
系统化验证HMCEN各组件的贡献

核心问题: HMCEN的AUROC看起来很强，但这个优势来自哪里？
- 是多粒度架构的贡献？
- 是异配性门控的贡献？
- 还是仅仅因为"训练了一个分类器"？

决策依据:
- 如果HMCEN vs Vanilla < 3%：架构不值得投入
- 如果HMCEN vs Vanilla ≥ 5%：架构有实质性贡献
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==================== 数据加载 ====================

def load_clinc150_for_ood(n_samples=2000, ood_ratio=0.3, seed=42):
    """加载CLINC150数据用于OOD检测"""
    from datasets import load_dataset
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    from torch_geometric.data import Data

    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"  加载CLINC150数据集 (seed={seed})...")
    dataset = load_dataset('clinc_oos', 'plus', trust_remote_code=True)
    test_data = dataset['test']
    indices = np.random.choice(len(test_data), n_samples, replace=False)

    texts = [test_data[int(i)]['text'] for i in indices]
    labels = [test_data[int(i)]['intent'] for i in indices]

    print(f"  提取TF-IDF特征...")
    vectorizer = TfidfVectorizer(max_features=300)
    features = vectorizer.fit_transform(texts).toarray()

    # 构建KNN图
    print(f"  构建KNN图 (k=20)...")
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

    # OOD类别划分
    unique_labels = list(set(labels))
    n_ood_classes = int(len(unique_labels) * ood_ratio)
    ood_classes = set(np.random.choice(unique_labels, n_ood_classes, replace=False))

    ood_labels = np.array([1 if label in ood_classes else 0 for label in labels])

    data = Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=edge_index,
        ood_labels=torch.tensor(ood_labels, dtype=torch.long)
    )

    n_ood = ood_labels.sum()
    n_id = len(ood_labels) - n_ood
    print(f"  数据统计: {n_id} ID样本, {n_ood} OOD样本 ({100*n_ood/len(ood_labels):.1f}%)")

    return data

def compute_heterophily_pseudo(data):
    """计算伪异配性（基于特征相似度）"""
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

# ==================== 模型定义 ====================

class HMCEN_Full(nn.Module):
    """完整HMCEN（方案C）- 双分支+异配性自适应门控"""
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.homo_branch = GCNConv(input_dim, hidden_dim)
        self.hetero_branch = GCNConv(input_dim, hidden_dim)
        self.fusion = nn.Linear(hidden_dim, 64)
        self.classifier = nn.Linear(64, 2)

    def forward(self, x, edge_index, h_node):
        h_homo = F.relu(self.homo_branch(x, edge_index))
        h_hetero = F.relu(self.hetero_branch(x, edge_index))

        # 异配性自适应线性门控
        alpha = (1.0 - h_node).unsqueeze(-1)
        h_fused = alpha * h_homo + (1 - alpha) * h_hetero

        h = F.relu(self.fusion(h_fused))
        logits = self.classifier(h)
        return logits


class HMCEN_NoMultiGran(nn.Module):
    """移除多粒度（只用单个GNN分支）"""
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.gnn = GCNConv(input_dim, hidden_dim)
        self.fusion = nn.Linear(hidden_dim, 64)
        self.classifier = nn.Linear(64, 2)

    def forward(self, x, edge_index, h_node):
        h = F.relu(self.gnn(x, edge_index))

        # 保留异配性加权（简化的自适应）
        alpha = (1.0 - h_node).unsqueeze(-1)
        h_adaptive = alpha * h

        h = F.relu(self.fusion(h_adaptive))
        logits = self.classifier(h)
        return logits


class HMCEN_NoHetGate(nn.Module):
    """移除异配性门控（固定α=0.5）"""
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.homo_branch = GCNConv(input_dim, hidden_dim)
        self.hetero_branch = GCNConv(input_dim, hidden_dim)
        self.fusion = nn.Linear(hidden_dim, 64)
        self.classifier = nn.Linear(64, 2)

    def forward(self, x, edge_index, h_node):
        h_homo = F.relu(self.homo_branch(x, edge_index))
        h_hetero = F.relu(self.hetero_branch(x, edge_index))

        # 固定权重融合（不依赖异配性）
        h_fused = 0.5 * h_homo + 0.5 * h_hetero

        h = F.relu(self.fusion(h_fused))
        logits = self.classifier(h)
        return logits


class HMCEN_NoContrastive(nn.Module):
    """
    移除对比学习
    注意：当前简化版HMCEN没有显式对比学习模块
    此配置与Full相同，用于完整性验证
    """
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.homo_branch = GCNConv(input_dim, hidden_dim)
        self.hetero_branch = GCNConv(input_dim, hidden_dim)
        self.fusion = nn.Linear(hidden_dim, 64)
        self.classifier = nn.Linear(64, 2)

    def forward(self, x, edge_index, h_node):
        h_homo = F.relu(self.homo_branch(x, edge_index))
        h_hetero = F.relu(self.hetero_branch(x, edge_index))

        alpha = (1.0 - h_node).unsqueeze(-1)
        h_fused = alpha * h_homo + (1 - alpha) * h_hetero

        h = F.relu(self.fusion(h_fused))
        logits = self.classifier(h)
        return logits


class VanillaGNN(nn.Module):
    """
    Vanilla GNN基线（最关键的对比）
    标准2层GCN + 分类头
    """
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.5, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        logits = self.classifier(h)
        return logits


# ==================== 训练函数 ====================

def train_model(model, data, h_node, epochs=200, lr=0.01, is_vanilla=False, verbose=True):
    """统一训练函数"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 训练数据：ID样本
    train_mask = (data.ood_labels == 0)

    # 伪标签：高异配性 → 潜在OOD
    median_h = h_node[train_mask].median()
    train_labels = (h_node[train_mask] > median_h).long()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        if is_vanilla:
            logits = model(data.x, data.edge_index)
        else:
            logits = model(data.x, data.edge_index, h_node)

        loss = F.cross_entropy(logits[train_mask], train_labels)

        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # 预测OOD分数
    model.eval()
    with torch.no_grad():
        if is_vanilla:
            logits = model(data.x, data.edge_index)
        else:
            logits = model(data.x, data.edge_index, h_node)

        ood_scores = F.softmax(logits, dim=-1)[:, 1]

    return ood_scores.cpu().numpy()


def compute_fpr95(y_true, y_scores):
    """计算FPR@95% TPR"""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    idx = np.argmin(np.abs(tpr - 0.95))
    return fpr[idx]


# ==================== 消融实验 ====================

def run_ablation_study(seeds=[42, 2024, 2025]):
    """完整消融实验"""
    print("=" * 80)
    print("HMCEN消融实验")
    print("=" * 80)

    ablation_configs = {
        'Full': {
            'desc': '完整HMCEN（方案C）',
            'model_class': HMCEN_Full,
            'is_vanilla': False
        },
        'NoMultiGran': {
            'desc': '移除多粒度（只用节点级）',
            'model_class': HMCEN_NoMultiGran,
            'is_vanilla': False
        },
        'NoHetGate': {
            'desc': '移除异配性门控（固定α=0.5）',
            'model_class': HMCEN_NoHetGate,
            'is_vanilla': False
        },
        'NoContrastive': {
            'desc': '移除对比学习',
            'model_class': HMCEN_NoContrastive,
            'is_vanilla': False
        },
        'VanillaGNN': {
            'desc': 'Vanilla GNN基线',
            'model_class': VanillaGNN,
            'is_vanilla': True
        }
    }

    results = {config_name: {'auroc': [], 'fpr95': [], 'aupr': []}
               for config_name in ablation_configs.keys()}

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'=' * 80}")
        print(f"运行 Seed {seed} ({seed_idx + 1}/{len(seeds)})")
        print(f"{'=' * 80}")

        # 加载数据
        data = load_clinc150_for_ood(seed=seed)
        h_node = compute_heterophily_pseudo(data)
        y_true = data.ood_labels.numpy()

        print(f"\n  异配性统计: mean={h_node.mean():.4f}, std={h_node.std():.4f}")

        # 运行每个配置
        for config_name, config in ablation_configs.items():
            print(f"\n配置: {config['desc']}")

            # 设置seed保证可重复性
            torch.manual_seed(seed)
            np.random.seed(seed)

            # 创建模型
            model = config['model_class'](input_dim=data.x.shape[1])

            # 训练
            ood_scores = train_model(
                model, data, h_node,
                epochs=200,
                lr=0.01,
                is_vanilla=config['is_vanilla']
            )

            # 评估
            auroc = roc_auc_score(y_true, ood_scores)
            fpr95 = compute_fpr95(y_true, ood_scores)
            aupr = average_precision_score(y_true, ood_scores)

            results[config_name]['auroc'].append(auroc)
            results[config_name]['fpr95'].append(fpr95)
            results[config_name]['aupr'].append(aupr)

            print(f"  AUROC: {auroc:.4f}, FPR95: {fpr95:.4f}, AUPR: {aupr:.4f}")

    return results


# ==================== 结果分析 ====================

def analyze_ablation_results(results):
    """详细分析消融结果"""
    print("\n" + "=" * 80)
    print("消融实验结果汇总")
    print("=" * 80)

    # 计算统计量
    summary = []
    for config_name, metrics in results.items():
        auroc_mean = np.mean(metrics['auroc'])
        auroc_std = np.std(metrics['auroc'])
        fpr95_mean = np.mean(metrics['fpr95'])
        fpr95_std = np.std(metrics['fpr95'])
        aupr_mean = np.mean(metrics['aupr'])
        aupr_std = np.std(metrics['aupr'])

        summary.append({
            'config': config_name,
            'auroc_mean': auroc_mean,
            'auroc_std': auroc_std,
            'fpr95_mean': fpr95_mean,
            'fpr95_std': fpr95_std,
            'aupr_mean': aupr_mean,
            'aupr_std': aupr_std,
            'auroc_values': metrics['auroc']
        })

    # 打印表格
    print(f"\n{'配置':<25} {'AUROC':<25} {'FPR95':<20} {'AUPR':<20}")
    print("-" * 90)

    full_auroc = next(s['auroc_mean'] for s in summary if s['config'] == 'Full')

    for s in summary:
        delta = s['auroc_mean'] - full_auroc
        delta_str = "" if s['config'] == 'Full' else f" ({delta:+.4f})"

        print(f"{s['config']:<25} "
              f"{s['auroc_mean']:.4f}±{s['auroc_std']:.4f}{delta_str:<12} "
              f"{s['fpr95_mean']:.4f}±{s['fpr95_std']:.4f}  "
              f"{s['aupr_mean']:.4f}±{s['aupr_std']:.4f}")

    # 关键分析
    print("\n" + "=" * 80)
    print("关键分析")
    print("=" * 80)

    vanilla_auroc = next(s['auroc_mean'] for s in summary if s['config'] == 'VanillaGNN')
    vanilla_values = next(s['auroc_values'] for s in summary if s['config'] == 'VanillaGNN')
    full_values = next(s['auroc_values'] for s in summary if s['config'] == 'Full')

    # 分析1: HMCEN vs Vanilla GNN（最关键）
    print(f"\n1. HMCEN vs Vanilla GNN（最关键对比）:")
    delta_vanilla = full_auroc - vanilla_auroc
    delta_pct = delta_vanilla / vanilla_auroc * 100

    print(f"   Full HMCEN:    {full_auroc:.4f}")
    print(f"   Vanilla GNN:   {vanilla_auroc:.4f}")
    print(f"   提升:          {delta_vanilla:.4f} ({delta_pct:+.1f}%)")

    # 统计显著性检验
    if len(full_values) >= 2:
        t_stat, p_value = stats.ttest_rel(full_values, vanilla_values)
        sig_symbol = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        print(f"   显著性:        p={p_value:.4f} {sig_symbol}")

    print(f"\n   决策:")
    if delta_vanilla < 0.03:
        print("   [X] 提升<3%：HMCEN架构不值得投入")
        print("   → 优势主要来自'训练了分类器'而非架构创新")
        print("   → 推荐: 放弃HMCEN，全力C4-TDA")
        decision = "reject"
    elif delta_vanilla >= 0.05:
        print("   [√] 提升≥5%：HMCEN架构有实质性贡献")
        print("   → 架构设计确实带来性能提升")
        print("   → 推荐: 考虑HMCEN-Lite（如果时间允许）")
        decision = "accept"
    else:
        print("   [!] 提升3-5%：需要权衡")
        print("   → 架构有一定贡献，但不是决定性的")
        print("   → 推荐: 权衡时间成本 vs 性能提升")
        decision = "consider"

    # 分析2: 多粒度的贡献
    no_multigran_auroc = next(s['auroc_mean'] for s in summary if s['config'] == 'NoMultiGran')
    no_multigran_values = next(s['auroc_values'] for s in summary if s['config'] == 'NoMultiGran')
    delta_multigran = full_auroc - no_multigran_auroc

    print(f"\n2. 多粒度架构的贡献:")
    print(f"   Full:              {full_auroc:.4f}")
    print(f"   NoMultiGran:       {no_multigran_auroc:.4f}")
    print(f"   贡献:              {delta_multigran:.4f}")

    if len(full_values) >= 2:
        t_stat_mg, p_value_mg = stats.ttest_rel(full_values, no_multigran_values)
        print(f"   显著性:            p={p_value_mg:.4f}")

    if delta_multigran >= 0.03:
        print("   [√] 多粒度架构有显著贡献（≥3%）")
    elif delta_multigran >= 0.01:
        print("   [!] 多粒度架构有小贡献（1-3%）")
    else:
        print("   [X] 多粒度架构贡献有限（<1%）")
        print("   → 可能是过度设计")

    # 分析3: 异配性门控的贡献
    no_hetgate_auroc = next(s['auroc_mean'] for s in summary if s['config'] == 'NoHetGate')
    no_hetgate_values = next(s['auroc_values'] for s in summary if s['config'] == 'NoHetGate')
    delta_hetgate = full_auroc - no_hetgate_auroc

    print(f"\n3. 异配性自适应门控的贡献:")
    print(f"   Full (线性门控):   {full_auroc:.4f}")
    print(f"   NoHetGate (固定):  {no_hetgate_auroc:.4f}")
    print(f"   贡献:              {delta_hetgate:.4f}")

    if len(full_values) >= 2:
        t_stat_hg, p_value_hg = stats.ttest_rel(full_values, no_hetgate_values)
        print(f"   显著性:            p={p_value_hg:.4f}")

    if delta_hetgate >= 0.02:
        print("   [√] 异配性自适应有用（≥2%）")
    else:
        print("   [!] 固定权重可能就够了（<2%）")
        print("   → 自适应门控可能不是必要的")

    # 组件重要性排序
    print(f"\n4. 组件重要性排序:")
    deltas = {
        '多粒度': delta_multigran,
        '异配性门控': delta_hetgate,
        '对比学习': 0.0  # 当前简化版没有
    }
    sorted_components = sorted(deltas.items(), key=lambda x: x[1], reverse=True)

    for i, (comp, delta) in enumerate(sorted_components, 1):
        print(f"   {i}. {comp}: Δ={delta:.4f}")

    # 最终建议
    print("\n" + "=" * 80)
    print("最终建议")
    print("=" * 80)

    if delta_vanilla >= 0.05 and delta_multigran >= 0.03:
        print("[√] HMCEN架构值得投入")
        print("   - 相比基线有显著提升（≥5%）")
        print("   - 多粒度架构有实质性贡献（≥3%）")
        print("   → 推荐: 继续开发HMCEN-Lite")
        print("   → 建议: 双轨并行（主推C4-TDA，探索HMCEN-Lite）")
    elif delta_vanilla >= 0.03:
        print("[!] HMCEN有一定优势，但需要权衡")
        print(f"   - 提升{delta_pct:.1f}%")
        print("   - 需要考虑时间成本（HMCEN: 10-12月 vs C4-TDA: 3-4月）")
        print("   → 推荐: 主推C4-TDA，HMCEN作为可选探索")
    else:
        print("[X] HMCEN架构不值得投入")
        print("   - 相比基线提升<3%")
        print("   - 复杂度不匹配收益")
        print("   → 推荐: 放弃HMCEN，全力C4-TDA")

    return summary, decision


# ==================== 可视化 ====================

def visualize_ablation(results, output_path='hmcen_ablation_analysis.png'):
    """可视化消融结果"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 配置顺序
    configs = ['Full', 'NoMultiGran', 'NoHetGate', 'NoContrastive', 'VanillaGNN']
    config_labels = [
        'Full HMCEN',
        '- Multi-Gran',
        '- Het Gate',
        '- Contrastive',
        'Vanilla GNN'
    ]

    # 图1: AUROC boxplot
    aurocs = [results[c]['auroc'] for c in configs]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#95a5a6']

    bp = axes[0].boxplot(aurocs, labels=config_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # 添加散点
    for i, auroc_list in enumerate(aurocs):
        x = np.random.normal(i + 1, 0.04, size=len(auroc_list))
        axes[0].scatter(x, auroc_list, alpha=0.6, s=30, color='black', zorder=5)

    axes[0].set_ylabel('AUROC', fontsize=12)
    axes[0].set_title('HMCEN Ablation Study - Component Contributions', fontsize=14)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].tick_params(axis='x', rotation=15)

    # 添加Vanilla基线
    vanilla_mean = np.mean(results['VanillaGNN']['auroc'])
    axes[0].axhline(y=vanilla_mean, color='gray', linestyle='--',
                    alpha=0.5, label=f'Vanilla Baseline ({vanilla_mean:.3f})')
    axes[0].legend()

    # 图2: 增益柱状图
    full_mean = np.mean(results['Full']['auroc'])
    gains = []
    for config in configs[1:]:  # 跳过Full
        config_mean = np.mean(results[config]['auroc'])
        gain = full_mean - config_mean
        gains.append(gain)

    config_labels_gains = config_labels[1:]
    colors_gains = colors[1:]

    bars = axes[1].bar(range(len(gains)), gains, color=colors_gains, alpha=0.7)
    axes[1].set_xticks(range(len(gains)))
    axes[1].set_xticklabels(config_labels_gains, rotation=15)
    axes[1].set_ylabel('AUROC Drop (when removing component)', fontsize=12)
    axes[1].set_title('Component Contribution Analysis', fontsize=14)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[1].grid(True, alpha=0.3, axis='y')

    # 标注数值
    for i, (bar, gain) in enumerate(zip(bars, gains)):
        height = bar.get_height()
        va = 'bottom' if height > 0 else 'top'
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{gain:.4f}',
                    ha='center', va=va,
                    fontsize=10, fontweight='bold')

    # 添加3%和5%阈值线（用于Vanilla对比）
    vanilla_delta = full_mean - vanilla_mean
    axes[1].axhline(y=0.03, color='orange', linestyle=':', alpha=0.7, label='3% threshold')
    axes[1].axhline(y=0.05, color='green', linestyle=':', alpha=0.7, label='5% threshold')
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n可视化结果已保存: {output_path}")

    return fig


# ==================== 主函数 ====================

def main():
    print("\n" + "=" * 80)
    print("开始HMCEN消融实验...")
    print("=" * 80)
    print("\n验证目标:")
    print("  - 核心问题: HMCEN的优势来自哪里？")
    print("  - 决策依据: HMCEN vs Vanilla < 3% → 放弃; ≥ 5% → 值得投入")
    print("=" * 80)

    # 运行消融实验
    results = run_ablation_study(seeds=[42, 2024, 2025])

    # 分析结果
    summary, decision = analyze_ablation_results(results)

    # 可视化
    visualize_ablation(results)

    print("\n" + "=" * 80)
    print("实验完成！")
    print("=" * 80)

    return results, summary, decision


if __name__ == '__main__':
    results, summary, decision = main()
