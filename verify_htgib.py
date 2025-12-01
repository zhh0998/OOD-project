#!/usr/bin/env python3
"""
HT-GIB 核心假设验证脚本
验证：图异配性 h(v) 与 噪声率 N(v) 正相关
"""

import json
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("HT-GIB 假设验证: 异配性 vs 噪声率")
print("=" * 60)

# ============================================================
# A. 数据加载 (随机采样)
# ============================================================
print("\n[A] 加载数据 (随机采样)...")

import random
random.seed(42)

# 先读取所有数据
all_data = []
with open('nyt10/nyt10_train.txt', 'r') as f:
    for line in f:
        try:
            item = json.loads(line.strip())
            all_data.append(item)
        except:
            continue

print(f"    总样本数: {len(all_data)}")

# 随机采样15000个样本（确保包含足够NA和非NA）
sample_size = 15000
data = random.sample(all_data, min(sample_size, len(all_data)))
print(f"    随机采样: {len(data)} 个样本")

# 统计关系分布
relation_counts = defaultdict(int)
for item in data:
    relation_counts[item['relation']] += 1

na_count = relation_counts.get('NA', 0)
print(f"    NA样本数: {na_count} ({100*na_count/len(data):.1f}%)")
print(f"    关系类型数: {len(relation_counts)}")

# ============================================================
# B. 构建实体共现图
# ============================================================
print("\n[B] 构建实体共现图...")

import networkx as nx

G = nx.Graph()
entity_sentences = defaultdict(list)
entity_relations = defaultdict(list)

for item in data:
    head = item['h']['name']
    tail = item['t']['name']
    text = item['text']
    relation = item['relation']

    G.add_node(head)
    G.add_node(tail)

    if head != tail:
        G.add_edge(head, tail)

    entity_sentences[head].append(text)
    entity_sentences[tail].append(text)
    entity_relations[head].append(relation)
    entity_relations[tail].append(relation)

print(f"    节点数: {G.number_of_nodes()}")
print(f"    边数: {G.number_of_edges()}")
print(f"    平均度: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")

# ============================================================
# C. 计算异配性 h(v)
# ============================================================
print("\n[C] 计算异配性 h(v)...")
print("    加载句子编码模型...")

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')
print("    模型加载完成")

# 计算每个实体的平均句子嵌入
print("    计算实体嵌入...")
entity_embeddings = {}

entities_list = list(entity_sentences.keys())
for idx, entity in enumerate(entities_list):
    sentences = entity_sentences[entity]
    if len(sentences) > 0:
        # 最多取前5句加速
        embs = model.encode(sentences[:5], show_progress_bar=False)
        entity_embeddings[entity] = np.mean(embs, axis=0)

    if (idx + 1) % 500 == 0:
        print(f"    已处理 {idx + 1}/{len(entities_list)} 个实体")

print(f"    计算了 {len(entity_embeddings)} 个实体嵌入")

# 计算异配性 h(v)
print("    计算异配性分数...")
heterophily_scores = {}

for node in G.nodes():
    if node not in entity_embeddings:
        continue

    neighbors = list(G.neighbors(node))
    if len(neighbors) == 0:
        continue

    neighbor_embs = [entity_embeddings[n] for n in neighbors if n in entity_embeddings]
    if len(neighbor_embs) == 0:
        continue

    node_emb = entity_embeddings[node].reshape(1, -1)
    neighbor_matrix = np.array(neighbor_embs)
    similarities = cosine_similarity(node_emb, neighbor_matrix)[0]

    # 异配性 = 1 - 平均相似度
    h_v = 1.0 - np.mean(similarities)
    heterophily_scores[node] = h_v

print(f"    计算了 {len(heterophily_scores)} 个节点的异配性")
print(f"    异配性范围: [{min(heterophily_scores.values()):.3f}, {max(heterophily_scores.values()):.3f}]")

# ============================================================
# D. 计算噪声率 N(v)
# ============================================================
print("\n[D] 计算噪声率 N(v)...")

entity_noise_stats = defaultdict(lambda: {'total': 0, 'noise': 0})

for item in data:
    head = item['h']['name']
    tail = item['t']['name']
    relation = item['relation']

    entity_noise_stats[head]['total'] += 1
    entity_noise_stats[tail]['total'] += 1

    if relation == 'NA':
        entity_noise_stats[head]['noise'] += 1
        entity_noise_stats[tail]['noise'] += 1

noise_rates = {}
for entity, stats in entity_noise_stats.items():
    if stats['total'] > 0:
        noise_rates[entity] = stats['noise'] / stats['total']

print(f"    计算了 {len(noise_rates)} 个节点的噪声率")
print(f"    噪声率范围: [{min(noise_rates.values()):.3f}, {max(noise_rates.values()):.3f}]")
print(f"    平均噪声率: {np.mean(list(noise_rates.values())):.3f}")

# ============================================================
# E. 统计分析
# ============================================================
print("\n[E] 统计分析...")

import scipy.stats as stats

common_entities = set(heterophily_scores.keys()) & set(noise_rates.keys())
print(f"    用于分析的节点数: {len(common_entities)}")

h_values = [heterophily_scores[e] for e in common_entities]
n_values = [noise_rates[e] for e in common_entities]

# 1. Pearson相关系数
r_pearson, p_pearson = stats.pearsonr(h_values, n_values)

# 2. Spearman秩相关
r_spearman, p_spearman = stats.spearmanr(h_values, n_values)

# 3. Cohen's d
quartiles = np.percentile(h_values, [25, 50, 75])
q1_threshold = quartiles[0]
q4_threshold = quartiles[2]

low_het_noise = [n_values[i] for i, h in enumerate(h_values) if h <= q1_threshold]
high_het_noise = [n_values[i] for i, h in enumerate(h_values) if h >= q4_threshold]

mean_low = np.mean(low_het_noise)
mean_high = np.mean(high_het_noise)
std_low = np.std(low_het_noise, ddof=1)
std_high = np.std(high_het_noise, ddof=1)
n_low = len(low_het_noise)
n_high = len(high_het_noise)

pooled_std = np.sqrt(((n_low-1)*std_low**2 + (n_high-1)*std_high**2) / (n_low + n_high - 2))
cohens_d = (mean_high - mean_low) / pooled_std if pooled_std > 0 else 0

# 4. Quartile分析
q_labels = ['Q1 (低异配)', 'Q2', 'Q3', 'Q4 (高异配)']
q_noise_means = []
q_data = []

for i in range(4):
    if i == 0:
        mask = [h <= quartiles[0] for h in h_values]
    elif i == 3:
        mask = [h > quartiles[2] for h in h_values]
    else:
        mask = [quartiles[i-1] < h <= quartiles[i] for h in h_values]

    q_noise = [n_values[j] for j, m in enumerate(mask) if m]
    q_noise_means.append(np.mean(q_noise))
    q_data.append(q_noise)

# ============================================================
# F. 可视化
# ============================================================
print("\n[F] 生成可视化...")

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 子图1: 散点图
axes[0].scatter(h_values, n_values, alpha=0.5, s=20, c='steelblue')
axes[0].set_xlabel('Heterophily h(v)', fontsize=12)
axes[0].set_ylabel('Noise Rate N(v)', fontsize=12)
axes[0].set_title(f'Heterophily vs Noise Rate\n(Pearson r={r_pearson:.3f}, p={p_pearson:.4f})', fontsize=13)
axes[0].grid(True, alpha=0.3)

# 添加趋势线
z = np.polyfit(h_values, n_values, 1)
p_func = np.poly1d(z)
x_trend = np.linspace(min(h_values), max(h_values), 100)
axes[0].plot(x_trend, p_func(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend line')
axes[0].legend()

# 子图2: Quartile箱线图
bp = axes[1].boxplot(q_data, labels=q_labels, patch_artist=True)
colors = ['#90EE90', '#FFFF99', '#FFB366', '#FF6B6B']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
axes[1].set_ylabel('Noise Rate N(v)', fontsize=12)
axes[1].set_xlabel('Heterophily Quartile', fontsize=12)
axes[1].set_title(f'Noise Rate by Heterophily Quartile\n(Cohen\'s d={cohens_d:.3f})', fontsize=13)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('htgib_validation.png', dpi=150, bbox_inches='tight')
print("    图表已保存: htgib_validation.png")

# ============================================================
# G. 输出结论
# ============================================================
print("\n" + "=" * 60)
print("HT-GIB 假设验证结果")
print("=" * 60)
print(f"\n数据统计:")
print(f"  样本数: {len(data)}")
print(f"  分析节点数: {len(common_entities)}")
print(f"  平均异配性: {np.mean(h_values):.3f}")
print(f"  平均噪声率: {np.mean(n_values):.3f}")

print(f"\n相关性分析:")
print(f"  Pearson r  = {r_pearson:.4f}, p = {p_pearson:.4e}")
print(f"  Spearman ρ = {r_spearman:.4f}, p = {p_spearman:.4e}")

print(f"\n效应量分析:")
print(f"  Cohen's d  = {cohens_d:.4f}  ← 关键指标")
print(f"  解释: ", end="")
if abs(cohens_d) < 0.2:
    print("极小效应")
elif abs(cohens_d) < 0.5:
    print("小效应")
elif abs(cohens_d) < 0.8:
    print("中等效应")
else:
    print("大效应")

print(f"\nQuartile分析:")
for i, (label, mean_noise) in enumerate(zip(q_labels, q_noise_means)):
    print(f"  {label}: 平均噪声率 = {mean_noise:.4f}")

print(f"\n组间比较:")
print(f"  Q1 (低异配) 噪声率: {mean_low:.4f} (n={n_low})")
print(f"  Q4 (高异配) 噪声率: {mean_high:.4f} (n={n_high})")
print(f"  差异 (Q4 - Q1): {mean_high - mean_low:.4f}")

print("\n" + "=" * 60)
print("最终结论")
print("=" * 60)

if cohens_d > 0.3:
    print(f"\n✅ Cohen's d = {cohens_d:.4f} > 0.3")
    print("✅ HT-GIB 假设成立！")
    print("✅ 噪声率与异配性存在显著正相关")
    print("\n建议: 继续 HT-GIB Phase 2 实施")
else:
    print(f"\n❌ Cohen's d = {cohens_d:.4f} < 0.3")
    print("❌ HT-GIB 假设不成立，效应量过小")
    print("\n建议: 考虑以下替代方案：")
    print("   1. HDCL-RE (异构双塔对比学习)")
    print("   2. 标准对比学习去噪")
    print("   3. 不确定性引导的选择性注意力")

print("\n" + "=" * 60)

# 额外统计信息
print("\n附加分析:")
print(f"  回归斜率: {z[0]:.6f}")
print(f"  相关显著性: {'显著' if p_pearson < 0.05 else '不显著'} (α=0.05)")
print(f"  方向性: {'正相关' if r_pearson > 0 else '负相关' if r_pearson < 0 else '无相关'}")
