#!/usr/bin/env python3
"""
HT-GIB假设验证 - DocRED数据集
交叉验证：异配性 vs 噪声率
"""

import json
import numpy as np
from collections import defaultdict
import random
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("HT-GIB 假设验证 - DocRED数据集")
print("=" * 70)

# ============================================================
# A. 加载DocRED数据
# ============================================================
print("\n[A] 加载DocRED数据...")

with open('docred_data/DocRED/train_annotated.json', 'r') as f:
    docred_data = json.load(f)

print(f"    文档数: {len(docred_data)}")

# 展开为实体对数据
random.seed(42)

entity_sentences = defaultdict(list)
entity_noise_stats = defaultdict(lambda: {'total': 0, 'noise': 0})
entity_pairs_data = []

total_positive = 0
total_negative = 0

for doc in docred_data:
    # 获取文档文本
    sents = doc['sents']
    full_text = ' '.join([' '.join(sent) for sent in sents])

    # 获取实体名称
    entities = []
    for vertex in doc['vertexSet']:
        # 取第一个mention的name
        name = vertex[0]['name']
        entities.append(name)
        entity_sentences[name].append(full_text)

    # 获取正例关系（有标注的实体对）
    labels = doc.get('labels', [])
    labeled_pairs = set()

    for label in labels:
        h_idx = label['h']
        t_idx = label['t']
        labeled_pairs.add((h_idx, t_idx))

        h_name = entities[h_idx]
        t_name = entities[t_idx]

        # 这是正例（有关系）
        entity_noise_stats[h_name]['total'] += 1
        entity_noise_stats[t_name]['total'] += 1
        # 不是噪声
        total_positive += 1

    # 负例：未标注的实体对视为NA（噪声）
    n_entities = len(entities)
    for i in range(n_entities):
        for j in range(n_entities):
            if i != j and (i, j) not in labeled_pairs:
                h_name = entities[i]
                t_name = entities[j]

                entity_noise_stats[h_name]['total'] += 1
                entity_noise_stats[t_name]['total'] += 1
                entity_noise_stats[h_name]['noise'] += 1
                entity_noise_stats[t_name]['noise'] += 1
                total_negative += 1

print(f"    实体数: {len(entity_sentences)}")
print(f"    正例(有关系): {total_positive}")
print(f"    负例(NA): {total_negative}")
print(f"    NA比例: {100*total_negative/(total_positive+total_negative):.1f}%")

# ============================================================
# B. 构建实体共现图并计算异配性
# ============================================================
print("\n[B] 构建实体共现图...")

import networkx as nx

G = nx.Graph()

for doc in docred_data:
    entities = [v[0]['name'] for v in doc['vertexSet']]
    for i, e1 in enumerate(entities):
        G.add_node(e1)
        for j, e2 in enumerate(entities):
            if i < j:
                G.add_edge(e1, e2)

print(f"    节点数: {G.number_of_nodes()}")
print(f"    边数: {G.number_of_edges()}")

# ============================================================
# C. 计算实体嵌入和异配性
# ============================================================
print("\n[C] 计算实体嵌入...")

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

# 过滤：只保留有足够数据的实体
valid_entities = [e for e in entity_sentences.keys()
                  if len(entity_sentences[e]) >= 1
                  and entity_noise_stats[e]['total'] >= 2
                  and e in G.nodes()]

print(f"    有效实体数: {len(valid_entities)}")

# 计算嵌入
entity_embeddings = {}
for idx, entity in enumerate(valid_entities):
    sentences = entity_sentences[entity][:5]  # 最多5个文档
    if sentences:
        embs = model.encode(sentences, show_progress_bar=False)
        entity_embeddings[entity] = np.mean(embs, axis=0)

    if (idx + 1) % 1000 == 0:
        print(f"    已处理 {idx + 1}/{len(valid_entities)}")

print(f"    嵌入数: {len(entity_embeddings)}")

# 计算异配性
print("\n[D] 计算异配性...")

heterophily_scores = {}

for node in entity_embeddings.keys():
    neighbors = list(G.neighbors(node))
    neighbor_embs = [entity_embeddings[n] for n in neighbors if n in entity_embeddings]

    if len(neighbor_embs) == 0:
        continue

    node_emb = entity_embeddings[node].reshape(1, -1)
    neighbor_matrix = np.array(neighbor_embs)
    similarities = cosine_similarity(node_emb, neighbor_matrix)[0]

    h_v = 1.0 - np.mean(similarities)
    heterophily_scores[node] = h_v

print(f"    异配性节点数: {len(heterophily_scores)}")

# ============================================================
# E. 计算噪声率
# ============================================================
print("\n[E] 计算噪声率...")

noise_rates = {}
for entity in heterophily_scores.keys():
    stats = entity_noise_stats[entity]
    if stats['total'] > 0:
        noise_rates[entity] = stats['noise'] / stats['total']

print(f"    噪声率节点数: {len(noise_rates)}")

# ============================================================
# F. 统计分析
# ============================================================
print("\n[F] 统计分析...")

import scipy.stats as stats

common_entities = set(heterophily_scores.keys()) & set(noise_rates.keys())
print(f"    分析节点数: {len(common_entities)}")

h_values = np.array([heterophily_scores[e] for e in common_entities])
n_values = np.array([noise_rates[e] for e in common_entities])

print(f"    异配性范围: [{h_values.min():.3f}, {h_values.max():.3f}]")
print(f"    噪声率范围: [{n_values.min():.3f}, {n_values.max():.3f}]")
print(f"    平均异配性: {h_values.mean():.3f}")
print(f"    平均噪声率: {n_values.mean():.3f}")

# Pearson相关
r_pearson, p_pearson = stats.pearsonr(h_values, n_values)

# Spearman相关
r_spearman, p_spearman = stats.spearmanr(h_values, n_values)

# Cohen's d
q1 = np.percentile(h_values, 25)
q4 = np.percentile(h_values, 75)

low_noise = n_values[h_values <= q1]
high_noise = n_values[h_values >= q4]

mean_low = np.mean(low_noise)
mean_high = np.mean(high_noise)
std_low = np.std(low_noise, ddof=1)
std_high = np.std(high_noise, ddof=1)
n_low = len(low_noise)
n_high = len(high_noise)

pooled_std = np.sqrt(((n_low-1)*std_low**2 + (n_high-1)*std_high**2) / (n_low + n_high - 2))
cohens_d = (mean_high - mean_low) / pooled_std if pooled_std > 0 else 0

# ============================================================
# G. 输出结果
# ============================================================
print("\n" + "=" * 70)
print("DocRED 验证结果")
print("=" * 70)

print(f"\n数据统计:")
print(f"  文档数: {len(docred_data)}")
print(f"  分析实体数: {len(common_entities)}")
print(f"  NA比例: {100*total_negative/(total_positive+total_negative):.1f}%")
print(f"  平均异配性: {h_values.mean():.3f}")
print(f"  平均噪声率: {n_values.mean():.3f}")

print(f"\n相关性分析:")
print(f"  Pearson r  = {r_pearson:+.4f} (p={p_pearson:.2e})")
print(f"  Spearman ρ = {r_spearman:+.4f} (p={p_spearman:.2e})")
print(f"  Cohen's d  = {cohens_d:+.4f}  ← 关键指标！")

print(f"\nQuartile分析:")
print(f"  Q1 (低异配25%) 噪声率: {mean_low:.4f} (n={n_low})")
print(f"  Q4 (高异配25%) 噪声率: {mean_high:.4f} (n={n_high})")
print(f"  差异 (Q4-Q1): {mean_high - mean_low:+.4f}")

# ============================================================
# H. 与NYT10对比
# ============================================================
print("\n" + "=" * 70)
print("NYT10 vs DocRED 对比")
print("=" * 70)

print("\n指标           NYT10-Train    NYT10-Test     DocRED")
print("-" * 70)
print(f"Cohen's d      -0.6044        -0.2860        {cohens_d:+.4f}")
print(f"Pearson r      -0.2321        -0.1218        {r_pearson:+.4f}")
print(f"Spearman ρ     -0.2722        -0.1900        {r_spearman:+.4f}")
print("-" * 70)

# ============================================================
# I. 最终结论
# ============================================================
print("\n" + "=" * 70)
print("最终结论")
print("=" * 70)

# 判断DocRED结果
if cohens_d > 0.3:
    docred_result = "正相关"
    print(f"\n🔍 DocRED结果: Cohen's d = {cohens_d:+.4f} > 0.3")
    print("   → 异配性与噪声率正相关")
elif cohens_d < -0.3:
    docred_result = "负相关"
    print(f"\n🔍 DocRED结果: Cohen's d = {cohens_d:+.4f} < -0.3")
    print("   → 异配性与噪声率负相关")
else:
    docred_result = "无显著相关"
    print(f"\n🔍 DocRED结果: |Cohen's d| = {abs(cohens_d):.4f} < 0.3")
    print("   → 无显著相关性")

# 综合判断
print("\n" + "-" * 70)
print("综合判断:")
print("-" * 70)

nyt_negative = True  # NYT10训练集和测试集都是负相关

if cohens_d > 0.3:
    # Case 1: DocRED正相关，NYT10负相关
    print("\n⚠️ Case 1: 结果不一致！")
    print()
    print("NYT10: 显著负相关 (Cohen's d ≈ -0.6)")
    print(f"DocRED: 显著正相关 (Cohen's d = {cohens_d:+.4f})")
    print()
    print("可能原因:")
    print("1. 数据集特性不同（新闻 vs 百科）")
    print("2. 远程监督 vs 人工标注")
    print("3. 句子级 vs 文档级")
    print()
    print("建议:")
    print("• HT-GIB可能只适用特定场景")
    print("• 需要深入分析数据集差异")
    print("• 风险：中等（需要额外研究）")

elif cohens_d < -0.3:
    # Case 2: 两个数据集都负相关
    print("\n✅ Case 2: 结果一致！")
    print()
    print("NYT10-Train: Cohen's d = -0.6044 (负相关)")
    print("NYT10-Test:  Cohen's d = -0.2860 (负相关)")
    print(f"DocRED:      Cohen's d = {cohens_d:+.4f} (负相关)")
    print()
    print("┌────────────────────────────────────────────────────────┐")
    print("│  三个数据集结果一致：异配性与噪声率呈负相关            │")
    print("│  HT-GIB核心假设彻底失败！                              │")
    print("└────────────────────────────────────────────────────────┘")
    print()
    print("📋 强烈建议:")
    print("   1. 立即放弃 HT-GIB 方案")
    print("   2. 切换到备选方案:")
    print("      • HDCL-RE (异构双塔对比学习)")
    print("      • 标准对比学习去噪")
    print("      • 不确定性引导注意力")
    print()
    print("⏱️ 验证时间: 约3小时")
    print("💰 及时止损，避免浪费4周实施时间")

else:
    # Case 3: DocRED无显著相关
    print("\n⚠️ Case 3: DocRED无显著相关")
    print()
    print("NYT10: 显著负相关")
    print(f"DocRED: 无显著相关 (Cohen's d = {cohens_d:+.4f})")
    print()
    print("结论:")
    print("• NYT10已确认假设失败（负相关）")
    print("• DocRED没有支持证据")
    print("• 至少一个主流数据集验证失败")
    print()
    print("建议: 倾向于放弃HT-GIB")
    print("风险: 高（无足够证据支持假设）")

print("\n" + "=" * 70)
