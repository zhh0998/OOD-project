#!/usr/bin/env python3
"""
CLINC150 异配性-Betti数相关性验证实验
======================================
核心假设：高异配性节点 → 高Betti数

实验流程：
1. 下载CLINC150数据集
2. 构建文本语义图（基于TF-IDF）
3. 计算节点异配性(NHR)
4. 计算每个节点Ego-Graph的Betti数（使用scipy稀疏矩阵方法）
5. 统计分析（Cohen's d, T检验, Pearson相关）
6. 可视化结果
"""

import numpy as np
import networkx as nx
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)

print("=" * 60)
print("CLINC150 异配性-Betti数相关性验证实验")
print("=" * 60)

# ============================================================
# Step 1: 数据准备
# ============================================================
print("\n[Step 1] 加载CLINC150数据集...")

from datasets import load_dataset

# 加载CLINC150数据集
data = load_dataset("clinc_oos", "plus")

# 提取训练集文本和标签
texts = data['train']['text']
labels = data['train']['intent']

print(f"  - 样本数量: {len(texts)}")
print(f"  - 类别数量: {len(set(labels))}")

# 为了计算效率，采样一部分数据
SAMPLE_SIZE = 2000  # 使用2000个样本进行分析
indices = np.random.choice(len(texts), min(SAMPLE_SIZE, len(texts)), replace=False)
texts = [texts[i] for i in indices]
labels = [labels[i] for i in indices]

print(f"  - 采样后样本数: {len(texts)}")

# ============================================================
# Step 2: 构建文本语义图 (使用TF-IDF)
# ============================================================
print("\n[Step 2] 构建文本语义图...")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 使用TF-IDF计算文本向量
print("  - 计算TF-IDF向量...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
embeddings = vectorizer.fit_transform(texts).toarray()

print(f"  - 向量维度: {embeddings.shape}")

# 计算余弦相似度矩阵
print("  - 计算相似度矩阵...")
similarity_matrix = cosine_similarity(embeddings)

# 构建图（相似度阈值 > 0.5，对于TF-IDF使用较低阈值）
SIMILARITY_THRESHOLD = 0.5
print(f"  - 构建图（相似度阈值: {SIMILARITY_THRESHOLD}）...")

G = nx.Graph()
G.add_nodes_from(range(len(texts)))

# 添加边
edge_count = 0
for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        if similarity_matrix[i, j] > SIMILARITY_THRESHOLD:
            G.add_edge(i, j, weight=similarity_matrix[i, j])
            edge_count += 1

print(f"  - 节点数: {G.number_of_nodes()}")
print(f"  - 边数: {G.number_of_edges()}")
print(f"  - 平均度: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")

# ============================================================
# Step 3: 计算异配性 (Node Heterophily Ratio)
# ============================================================
print("\n[Step 3] 计算节点异配性...")

def compute_heterophily(graph, labels):
    """
    计算每个节点的异配率
    节点异配率 = 1 - (同类邻居数 / 总邻居数)
    """
    nhr_list = []
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        if len(neighbors) == 0:
            nhr_list.append(0.0)  # 孤立节点设为0
            continue
        same_label = sum([labels[n] == labels[node] for n in neighbors])
        nhr = 1 - (same_label / len(neighbors))
        nhr_list.append(nhr)
    return nhr_list

nhr_list = compute_heterophily(G, labels)

print(f"  - NHR均值: {np.mean(nhr_list):.4f}")
print(f"  - NHR标准差: {np.std(nhr_list):.4f}")
print(f"  - NHR范围: [{min(nhr_list):.4f}, {max(nhr_list):.4f}]")

# ============================================================
# Step 4: 计算Betti数 (使用scipy稀疏矩阵方法)
# ============================================================
print("\n[Step 4] 计算Betti数...")

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform

def compute_vietoris_rips_betti(points, max_epsilon=1.0, num_steps=20):
    """
    使用Vietoris-Rips复形计算持续同调的Betti数
    基于scipy的稀疏矩阵实现

    参数:
        points: 点云的嵌入向量
        max_epsilon: 最大过滤参数
        num_steps: 过滤步数

    返回:
        (beta_0, beta_1, beta_2): 各维度的Betti数特征
    """
    n = len(points)
    if n < 3:
        return (n, 0, 0)

    # 计算距离矩阵
    if n > 1:
        dist_matrix = squareform(pdist(points, metric='cosine'))
    else:
        return (1, 0, 0)

    # 使用不同epsilon值计算持续特征
    epsilons = np.linspace(0.1, max_epsilon, num_steps)

    # 跟踪Betti数变化
    beta_0_history = []
    beta_1_estimate = 0

    for eps in epsilons:
        # 构建邻接矩阵
        adj = (dist_matrix <= eps).astype(int)
        np.fill_diagonal(adj, 0)

        # 计算连通分量数 (β₀)
        n_components, _ = connected_components(csr_matrix(adj), directed=False)
        beta_0_history.append(n_components)

        # 估计β₁：通过环路检测
        # 使用图的圈数近似: 圈数 = 边数 - 节点数 + 连通分量数
        edge_count = np.sum(adj) // 2
        cyclomatic_complexity = edge_count - n + n_components
        if cyclomatic_complexity > beta_1_estimate:
            beta_1_estimate = cyclomatic_complexity

    # β₀: 使用中等epsilon的连通分量数
    beta_0 = beta_0_history[len(beta_0_history) // 2]

    # β₁: 使用最大圈复杂度作为估计
    beta_1 = beta_1_estimate

    # β₂: 使用简化估计（对于小图，通常为0或很小）
    # 检测空洞：使用三角形和四面体的关系
    beta_2 = 0
    if n >= 4:
        # 统计三角形数量
        adj_final = (dist_matrix <= epsilons[len(epsilons)//2]).astype(int)
        np.fill_diagonal(adj_final, 0)
        adj_cube = np.linalg.matrix_power(adj_final, 3)
        triangle_count = np.trace(adj_cube) // 6

        # 简单估计：如果三角形很多但图不是完全连通的，可能存在空洞
        if triangle_count > n and beta_0_history[-1] == 1:
            beta_2 = max(0, (triangle_count - n) // 10)

    return (beta_0, beta_1, beta_2)


def compute_simplicial_betti(graph, node, embeddings, radius=2):
    """
    计算节点Ego-Graph的Betti数
    使用基于嵌入的Vietoris-Rips复形
    """
    # 获取ego graph
    ego = nx.ego_graph(graph, node, radius=radius)
    ego_nodes = list(ego.nodes())

    if len(ego_nodes) < 3:
        return (len(ego_nodes), 0, 0)

    # 获取ego graph的嵌入
    ego_embeddings = embeddings[ego_nodes]

    # 计算Betti数
    return compute_vietoris_rips_betti(ego_embeddings)


# 计算每个节点的Betti数
print("  - 计算每个节点的Ego-Graph Betti数...")
betti_numbers = []

# 只对有邻居的节点计算
valid_nodes = [n for n in G.nodes() if G.degree(n) > 0]
print(f"  - 有效节点数: {len(valid_nodes)}")

from tqdm import tqdm
for node in tqdm(valid_nodes, desc="  计算Betti数"):
    betti = compute_simplicial_betti(G, node, embeddings, radius=2)
    betti_numbers.append(betti)

# 更新nhr_list以匹配有效节点
nhr_valid = [nhr_list[n] for n in valid_nodes]
labels_valid = [labels[n] for n in valid_nodes]

print(f"\n  - Beta_0 均值: {np.mean([b[0] for b in betti_numbers]):.2f}")
print(f"  - Beta_1 均值: {np.mean([b[1] for b in betti_numbers]):.2f}")
print(f"  - Beta_2 均值: {np.mean([b[2] for b in betti_numbers]):.2f}")

# ============================================================
# Step 5: 统计分析
# ============================================================
print("\n[Step 5] 统计分析...")
print("-" * 50)

import scipy.stats as stats

# 提取β₁作为主要分析对象
beta_1_list = [b[1] for b in betti_numbers]

# 分组：高NHR vs 低NHR（按四分位数）
q1, q3 = np.percentile(nhr_valid, [25, 75])
print(f"  NHR四分位数: Q1={q1:.4f}, Q3={q3:.4f}")

high_nhr_indices = [i for i, n in enumerate(nhr_valid) if n >= q3]
low_nhr_indices = [i for i, n in enumerate(nhr_valid) if n <= q1]

high_nhr_betti = [beta_1_list[i] for i in high_nhr_indices]
low_nhr_betti = [beta_1_list[i] for i in low_nhr_indices]

print(f"\n  高NHR组样本数: {len(high_nhr_betti)}")
print(f"  低NHR组样本数: {len(low_nhr_betti)}")

# 计算统计量
mean_high = np.mean(high_nhr_betti)
mean_low = np.mean(low_nhr_betti)
std_high = np.std(high_nhr_betti)
std_low = np.std(low_nhr_betti)

# Cohen's d
std_pooled = np.sqrt((np.var(high_nhr_betti) + np.var(low_nhr_betti)) / 2)
if std_pooled > 0:
    cohens_d = (mean_high - mean_low) / std_pooled
else:
    cohens_d = 0

# T检验
t_stat, p_value = stats.ttest_ind(high_nhr_betti, low_nhr_betti)

# Pearson相关系数
pearson_r, pearson_p = stats.pearsonr(nhr_valid, beta_1_list)

# Spearman相关系数
spearman_r, spearman_p = stats.spearmanr(nhr_valid, beta_1_list)

print("\n" + "=" * 50)
print("统计结果汇总")
print("=" * 50)
print(f"\n【组间比较】")
print(f"  高NHR组: Beta_1 = {mean_high:.2f} +/- {std_high:.2f}")
print(f"  低NHR组: Beta_1 = {mean_low:.2f} +/- {std_low:.2f}")
print(f"\n【效应量】")
print(f"  Cohen's d = {cohens_d:.4f}")
print(f"\n【假设检验】")
print(f"  T统计量 = {t_stat:.4f}")
print(f"  p-value = {p_value:.6f}")
print(f"\n【相关性分析】")
print(f"  Pearson r = {pearson_r:.4f} (p = {pearson_p:.6f})")
print(f"  Spearman rho = {spearman_r:.4f} (p = {spearman_p:.6f})")

# 对所有Betti数进行分析
print("\n" + "-" * 50)
print("【多维Betti数分析】")
for dim in range(3):
    betti_dim = [b[dim] for b in betti_numbers]
    high_betti = [betti_dim[i] for i in high_nhr_indices]
    low_betti = [betti_dim[i] for i in low_nhr_indices]

    mean_h = np.mean(high_betti)
    mean_l = np.mean(low_betti)
    std_p = np.sqrt((np.var(high_betti) + np.var(low_betti)) / 2)
    d = (mean_h - mean_l) / std_p if std_p > 0 else 0
    r, _ = stats.pearsonr(nhr_valid, betti_dim)

    print(f"  Beta_{dim}: 高NHR={mean_h:.2f}, 低NHR={mean_l:.2f}, Cohen's d={d:.4f}, r={r:.4f}")

# ============================================================
# Step 6: 可视化
# ============================================================
print("\n[Step 6] 生成可视化...")

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 图1: 直方图对比 + 散点图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 直方图对比
ax1 = axes[0]
ax1.hist(high_nhr_betti, alpha=0.6, label=f'High NHR (n={len(high_nhr_betti)})',
         bins=20, color='#e74c3c', edgecolor='white')
ax1.hist(low_nhr_betti, alpha=0.6, label=f'Low NHR (n={len(low_nhr_betti)})',
         bins=20, color='#3498db', edgecolor='white')
ax1.axvline(mean_high, color='#c0392b', linestyle='--', linewidth=2, label=f'High NHR mean={mean_high:.2f}')
ax1.axvline(mean_low, color='#2980b9', linestyle='--', linewidth=2, label=f'Low NHR mean={mean_low:.2f}')
ax1.set_xlabel('Betti Number (Beta_1)', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title(f"Distribution Comparison\nCohen's d = {cohens_d:.4f}, p = {p_value:.4f}", fontsize=12)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# 散点图：NHR vs β₁
ax2 = axes[1]
scatter = ax2.scatter(nhr_valid, beta_1_list, alpha=0.4, c=labels_valid, cmap='tab20', s=30)
ax2.set_xlabel('Node Heterophily Ratio (NHR)', fontsize=12)
ax2.set_ylabel('Betti Number (Beta_1)', fontsize=12)
ax2.set_title(f'NHR vs Beta_1 Correlation\nPearson r = {pearson_r:.4f}, Spearman rho = {spearman_r:.4f}', fontsize=12)
ax2.grid(True, alpha=0.3)

# 添加趋势线
z = np.polyfit(nhr_valid, beta_1_list, 1)
p = np.poly1d(z)
x_line = np.linspace(min(nhr_valid), max(nhr_valid), 100)
ax2.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Trend line')
ax2.legend()

plt.tight_layout()
plt.savefig('nhr_betti_correlation.png', dpi=150, bbox_inches='tight')
print("  - 保存: nhr_betti_correlation.png")

# 图2: 多维度分析
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))

# 箱线图比较
ax = axes2[0, 0]
data_boxplot = [low_nhr_betti, high_nhr_betti]
bp = ax.boxplot(data_boxplot, labels=['Low NHR\n(<=Q1)', 'High NHR\n(>=Q3)'], patch_artist=True)
bp['boxes'][0].set_facecolor('#3498db')
bp['boxes'][1].set_facecolor('#e74c3c')
ax.set_ylabel('Betti Number (Beta_1)', fontsize=12)
ax.set_title(f'Box Plot Comparison\nCohen\'s d = {cohens_d:.4f}', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 多维Betti数热力图
ax = axes2[0, 1]
betti_matrix = np.array(betti_numbers)
# 按NHR排序
sorted_indices = np.argsort(nhr_valid)
betti_sorted = betti_matrix[sorted_indices]

# 分组显示
n_groups = 10
group_size = len(betti_sorted) // n_groups
group_means = []
group_labels_list = []
for i in range(n_groups):
    start_idx = i * group_size
    end_idx = start_idx + group_size if i < n_groups - 1 else len(betti_sorted)
    group_data = betti_sorted[start_idx:end_idx]
    group_means.append(np.mean(group_data, axis=0))
    nhr_range = sorted([nhr_valid[j] for j in sorted_indices[start_idx:end_idx]])
    group_labels_list.append(f'{nhr_range[0]:.2f}-{nhr_range[-1]:.2f}')

group_means = np.array(group_means)
im = ax.imshow(group_means.T, aspect='auto', cmap='YlOrRd')
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(['Beta_0', 'Beta_1', 'Beta_2'])
ax.set_xticks(range(n_groups))
ax.set_xticklabels([f'G{i+1}' for i in range(n_groups)], rotation=45)
ax.set_xlabel('NHR Groups (Low -> High)', fontsize=12)
ax.set_title('Betti Numbers by NHR Groups', fontsize=12)
plt.colorbar(im, ax=ax, label='Mean Betti Number')

# NHR分布
ax = axes2[1, 0]
ax.hist(nhr_valid, bins=30, color='#9b59b6', edgecolor='white', alpha=0.7)
ax.axvline(q1, color='#3498db', linestyle='--', linewidth=2, label=f'Q1={q1:.2f}')
ax.axvline(q3, color='#e74c3c', linestyle='--', linewidth=2, label=f'Q3={q3:.2f}')
ax.set_xlabel('Node Heterophily Ratio (NHR)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('NHR Distribution', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# 效应量汇总
ax = axes2[1, 1]
betti_dims = ['Beta_0', 'Beta_1', 'Beta_2']
cohens_ds = []
pearson_rs = []
for dim in range(3):
    betti_dim = [b[dim] for b in betti_numbers]
    high_b = [betti_dim[i] for i in high_nhr_indices]
    low_b = [betti_dim[i] for i in low_nhr_indices]
    std_p = np.sqrt((np.var(high_b) + np.var(low_b)) / 2)
    d = (np.mean(high_b) - np.mean(low_b)) / std_p if std_p > 0 else 0
    r, _ = stats.pearsonr(nhr_valid, betti_dim)
    cohens_ds.append(d)
    pearson_rs.append(r)

x = np.arange(len(betti_dims))
width = 0.35
bars1 = ax.bar(x - width/2, cohens_ds, width, label="Cohen's d", color='#e74c3c', alpha=0.8)
bars2 = ax.bar(x + width/2, pearson_rs, width, label='Pearson r', color='#3498db', alpha=0.8)

ax.axhline(0.5, color='green', linestyle='--', alpha=0.7, label='d=0.5 (Medium effect)')
ax.axhline(0.3, color='orange', linestyle='--', alpha=0.7, label='d=0.3 (Small effect)')
ax.set_ylabel('Effect Size', fontsize=12)
ax.set_xlabel('Betti Number Dimension', fontsize=12)
ax.set_title('Effect Size Summary', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(betti_dims)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, val in zip(bars1, cohens_ds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9)
for bar, val in zip(bars2, pearson_rs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('nhr_betti_analysis.png', dpi=150, bbox_inches='tight')
print("  - 保存: nhr_betti_analysis.png")

# ============================================================
# Step 7: 结论
# ============================================================
print("\n" + "=" * 60)
print("实验结论")
print("=" * 60)

# 判断标准
if abs(cohens_d) >= 0.5:
    conclusion = "假设成立"
    recommendation = "效应量显著(d>=0.5)，建议全面推进TDA方案"
    status = "STRONG"
elif abs(cohens_d) >= 0.3:
    conclusion = "中等效应"
    recommendation = "效应量中等(0.3<=d<0.5)，TDA可作为辅助特征"
    status = "MODERATE"
else:
    conclusion = "假设不成立"
    recommendation = "效应量较小(d<0.3)，建议考虑其他方案"
    status = "WEAK"

print(f"\n核心假设: 高异配性节点 -> 高Betti数")
print(f"\n【判定结果】: {conclusion}")
print(f"【Cohen's d】: {cohens_d:.4f}")
print(f"【建议】: {recommendation}")

print(f"\n详细指标:")
print(f"  - 高NHR组 Beta_1 均值: {mean_high:.4f}")
print(f"  - 低NHR组 Beta_1 均值: {mean_low:.4f}")
print(f"  - T检验 p-value: {p_value:.6f}")
print(f"  - Pearson相关系数: {pearson_r:.4f}")
print(f"  - Spearman相关系数: {spearman_r:.4f}")

# 保存统计结果到文件
results_summary = f"""
===============================================
CLINC150 异配性-Betti数相关性验证 - 统计结果
===============================================

实验配置:
  - 数据集: CLINC150 (clinc_oos, plus)
  - 采样数量: {len(texts)}
  - 相似度阈值: {SIMILARITY_THRESHOLD}
  - 图节点数: {G.number_of_nodes()}
  - 图边数: {G.number_of_edges()}
  - 有效节点数: {len(valid_nodes)}

异配性统计:
  - NHR均值: {np.mean(nhr_valid):.4f}
  - NHR标准差: {np.std(nhr_valid):.4f}
  - Q1 (25%): {q1:.4f}
  - Q3 (75%): {q3:.4f}

Betti数统计:
  - Beta_0 均值: {np.mean([b[0] for b in betti_numbers]):.4f}
  - Beta_1 均值: {np.mean([b[1] for b in betti_numbers]):.4f}
  - Beta_2 均值: {np.mean([b[2] for b in betti_numbers]):.4f}

组间比较 (Beta_1):
  - 高NHR组 (n={len(high_nhr_betti)}): {mean_high:.4f} +/- {std_high:.4f}
  - 低NHR组 (n={len(low_nhr_betti)}): {mean_low:.4f} +/- {std_low:.4f}

效应量:
  - Cohen's d: {cohens_d:.4f}
  - 效应大小: {"大" if abs(cohens_d) >= 0.8 else "中" if abs(cohens_d) >= 0.5 else "小" if abs(cohens_d) >= 0.2 else "微小"}

假设检验:
  - T统计量: {t_stat:.4f}
  - p-value: {p_value:.6f}
  - 显著性: {"显著 (p<0.05)" if p_value < 0.05 else "不显著 (p>=0.05)"}

相关性:
  - Pearson r: {pearson_r:.4f} (p={pearson_p:.6f})
  - Spearman rho: {spearman_r:.4f} (p={spearman_p:.6f})

多维Betti数效应:
"""

for dim in range(3):
    betti_dim = [b[dim] for b in betti_numbers]
    high_b = [betti_dim[i] for i in high_nhr_indices]
    low_b = [betti_dim[i] for i in low_nhr_indices]
    std_p = np.sqrt((np.var(high_b) + np.var(low_b)) / 2)
    d = (np.mean(high_b) - np.mean(low_b)) / std_p if std_p > 0 else 0
    r, p = stats.pearsonr(nhr_valid, betti_dim)
    results_summary += f"  - Beta_{dim}: Cohen's d = {d:.4f}, Pearson r = {r:.4f}\n"

results_summary += f"""
===============================================
结论
===============================================
判定结果: {conclusion}
Cohen's d = {cohens_d:.4f}

判断标准:
  - d >= 0.5: 假设成立，全面推进TDA方案
  - 0.3 <= d < 0.5: 中等效应，可作为辅助特征
  - d < 0.3: 假设不成立，考虑其他方案

建议: {recommendation}
===============================================
"""

with open('statistics_results.txt', 'w', encoding='utf-8') as f:
    f.write(results_summary)
print("\n  - 保存: statistics_results.txt")

print("\n" + "=" * 60)
print("实验完成!")
print("=" * 60)
print("\n输出文件:")
print("  1. nhr_betti_correlation.png - 主要可视化图")
print("  2. nhr_betti_analysis.png - 详细分析图")
print("  3. statistics_results.txt - 统计结果汇总")
