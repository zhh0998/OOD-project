#!/usr/bin/env python3
"""
HT-GIB假设交叉验证
在NYT10训练集和测试集上分别验证，确认结果一致性
"""

import json
import numpy as np
from collections import defaultdict
import random
import warnings
warnings.filterwarnings('ignore')

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats as stats
import networkx as nx

print("=" * 70)
print("HT-GIB 假设交叉验证")
print("验证: 异配性 h(v) 与 噪声率 N(v) 的相关性")
print("=" * 70)

def validate_hypothesis(data_path, dataset_name, sample_size=15000):
    """在指定数据集上验证HT-GIB假设"""

    print(f"\n{'='*70}")
    print(f"数据集: {dataset_name}")
    print(f"{'='*70}")

    # A. 加载数据
    print(f"\n[A] 加载数据...")
    random.seed(42)

    all_data = []
    with open(data_path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                all_data.append(item)
            except:
                continue

    print(f"    总样本数: {len(all_data)}")

    # 随机采样
    data = random.sample(all_data, min(sample_size, len(all_data)))
    print(f"    采样数: {len(data)}")

    # 统计NA比例
    na_count = sum(1 for item in data if item['relation'] == 'NA')
    print(f"    NA样本: {na_count} ({100*na_count/len(data):.1f}%)")

    # B. 构建图
    print(f"\n[B] 构建实体共现图...")
    G = nx.Graph()
    entity_sentences = defaultdict(list)

    for item in data:
        head = item['h']['name']
        tail = item['t']['name']
        text = item['text']

        G.add_node(head)
        G.add_node(tail)
        if head != tail:
            G.add_edge(head, tail)

        entity_sentences[head].append(text)
        entity_sentences[tail].append(text)

    print(f"    节点: {G.number_of_nodes()}, 边: {G.number_of_edges()}")

    # C. 计算实体嵌入
    print(f"\n[C] 计算实体嵌入...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    entity_embeddings = {}
    entities_list = list(entity_sentences.keys())

    for idx, entity in enumerate(entities_list):
        sentences = entity_sentences[entity]
        if len(sentences) > 0:
            embs = model.encode(sentences[:5], show_progress_bar=False)
            entity_embeddings[entity] = np.mean(embs, axis=0)

        if (idx + 1) % 1000 == 0:
            print(f"    已处理 {idx + 1}/{len(entities_list)}")

    print(f"    嵌入数: {len(entity_embeddings)}")

    # D. 计算异配性
    print(f"\n[D] 计算异配性...")
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

        h_v = 1.0 - np.mean(similarities)
        heterophily_scores[node] = h_v

    print(f"    异配性节点数: {len(heterophily_scores)}")

    # E. 计算噪声率
    print(f"\n[E] 计算噪声率...")
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
    for entity, st in entity_noise_stats.items():
        if st['total'] > 0:
            noise_rates[entity] = st['noise'] / st['total']

    print(f"    噪声率节点数: {len(noise_rates)}")

    # F. 统计分析
    print(f"\n[F] 统计分析...")
    common_entities = set(heterophily_scores.keys()) & set(noise_rates.keys())
    print(f"    分析节点数: {len(common_entities)}")

    h_values = [heterophily_scores[e] for e in common_entities]
    n_values = [noise_rates[e] for e in common_entities]

    # Pearson
    r_pearson, p_pearson = stats.pearsonr(h_values, n_values)

    # Spearman
    r_spearman, p_spearman = stats.spearmanr(h_values, n_values)

    # Cohen's d
    h_array = np.array(h_values)
    n_array = np.array(n_values)

    q1 = np.percentile(h_array, 25)
    q4 = np.percentile(h_array, 75)

    low_noise = n_array[h_array <= q1]
    high_noise = n_array[h_array >= q4]

    mean_low = np.mean(low_noise)
    mean_high = np.mean(high_noise)
    std_low = np.std(low_noise, ddof=1)
    std_high = np.std(high_noise, ddof=1)
    n_low = len(low_noise)
    n_high = len(high_noise)

    pooled_std = np.sqrt(((n_low-1)*std_low**2 + (n_high-1)*std_high**2) / (n_low + n_high - 2))
    cohens_d = (mean_high - mean_low) / pooled_std if pooled_std > 0 else 0

    # 输出结果
    results = {
        'dataset': dataset_name,
        'n_samples': len(data),
        'n_entities': len(common_entities),
        'na_ratio': na_count / len(data),
        'mean_heterophily': np.mean(h_values),
        'mean_noise_rate': np.mean(n_values),
        'pearson_r': r_pearson,
        'pearson_p': p_pearson,
        'spearman_r': r_spearman,
        'spearman_p': p_spearman,
        'cohens_d': cohens_d,
        'q1_noise': mean_low,
        'q4_noise': mean_high,
        'n_q1': n_low,
        'n_q4': n_high
    }

    print(f"\n{'='*50}")
    print(f"结果: {dataset_name}")
    print(f"{'='*50}")
    print(f"样本数: {results['n_samples']}")
    print(f"实体数: {results['n_entities']}")
    print(f"NA比例: {results['na_ratio']:.1%}")
    print(f"平均异配性: {results['mean_heterophily']:.3f}")
    print(f"平均噪声率: {results['mean_noise_rate']:.3f}")
    print()
    print(f"Pearson r  = {results['pearson_r']:+.4f} (p={results['pearson_p']:.2e})")
    print(f"Spearman ρ = {results['spearman_r']:+.4f} (p={results['spearman_p']:.2e})")
    print(f"Cohen's d  = {results['cohens_d']:+.4f}  ← 关键指标")
    print()
    print(f"Q1 (低异配) 噪声率: {results['q1_noise']:.4f} (n={results['n_q1']})")
    print(f"Q4 (高异配) 噪声率: {results['q4_noise']:.4f} (n={results['n_q4']})")
    print(f"差异: {results['q4_noise'] - results['q1_noise']:+.4f}")

    return results

# ============================================================
# 主程序：交叉验证
# ============================================================

print("\n" + "=" * 70)
print("开始交叉验证...")
print("=" * 70)

# 验证训练集
train_results = validate_hypothesis(
    'nyt10/nyt10_train.txt',
    'NYT10-Train',
    sample_size=15000
)

# 验证测试集
test_results = validate_hypothesis(
    'nyt10/nyt10_test.txt',
    'NYT10-Test',
    sample_size=15000
)

# ============================================================
# 对比分析
# ============================================================

print("\n" + "=" * 70)
print("交叉验证结果对比")
print("=" * 70)

print("\n指标              NYT10-Train    NYT10-Test     一致性")
print("-" * 70)

# Cohen's d
d_train = train_results['cohens_d']
d_test = test_results['cohens_d']
d_consistent = "✅" if (d_train < -0.3 and d_test < -0.3) or (d_train > 0.3 and d_test > 0.3) else "❌"
print(f"Cohen's d        {d_train:+.4f}        {d_test:+.4f}        {d_consistent}")

# Pearson r
r_train = train_results['pearson_r']
r_test = test_results['pearson_r']
r_consistent = "✅" if (r_train < 0 and r_test < 0) or (r_train > 0 and r_test > 0) else "❌"
print(f"Pearson r        {r_train:+.4f}        {r_test:+.4f}        {r_consistent}")

# Spearman
s_train = train_results['spearman_r']
s_test = test_results['spearman_r']
s_consistent = "✅" if (s_train < 0 and s_test < 0) or (s_train > 0 and s_test > 0) else "❌"
print(f"Spearman ρ       {s_train:+.4f}        {s_test:+.4f}        {s_consistent}")

# Q1噪声率
q1_train = train_results['q1_noise']
q1_test = test_results['q1_noise']
print(f"Q1噪声率         {q1_train:.4f}         {q1_test:.4f}")

# Q4噪声率
q4_train = train_results['q4_noise']
q4_test = test_results['q4_noise']
print(f"Q4噪声率         {q4_train:.4f}         {q4_test:.4f}")

print("-" * 70)

# ============================================================
# 最终结论
# ============================================================

print("\n" + "=" * 70)
print("最终结论")
print("=" * 70)

# 判断一致性
both_negative = d_train < -0.3 and d_test < -0.3
both_positive = d_train > 0.3 and d_test > 0.3
both_insignificant = abs(d_train) < 0.3 and abs(d_test) < 0.3

if both_negative:
    print("\n✅ 结果高度一致！")
    print()
    print(f"训练集 Cohen's d = {d_train:+.4f} (负相关)")
    print(f"测试集 Cohen's d = {d_test:+.4f} (负相关)")
    print()
    print("结论: HT-GIB核心假设在NYT10数据集上彻底失败！")
    print()
    print("┌─────────────────────────────────────────────────────┐")
    print("│  假设预期: 高异配性 → 高噪声率 (正相关)             │")
    print("│  实际结果: 高异配性 → 低噪声率 (负相关)             │")
    print("│  方向完全相反！                                      │")
    print("└─────────────────────────────────────────────────────┘")
    print()
    print("📋 强烈建议:")
    print("   1. 立即放弃 HT-GIB 方案")
    print("   2. 切换到备选方案:")
    print("      • HDCL-RE (异构双塔对比学习)")
    print("      • 标准对比学习去噪")
    print("      • 不确定性引导注意力机制")
    print()
    print("⏱️ 验证时间: 约2小时")
    print("💰 及时止损，避免浪费4周实施时间")

elif both_positive:
    print("\n✅ 结果一致：假设成立！")
    print()
    print(f"训练集 Cohen's d = {d_train:+.4f} (正相关)")
    print(f"测试集 Cohen's d = {d_test:+.4f} (正相关)")
    print()
    print("结论: HT-GIB假设得到验证")
    print("建议: 继续Phase 2实施")

elif both_insignificant:
    print("\n⚠️ 两个数据集都无显著相关")
    print()
    print(f"训练集 Cohen's d = {d_train:+.4f}")
    print(f"测试集 Cohen's d = {d_test:+.4f}")
    print()
    print("结论: HT-GIB假设无足够证据支持")
    print("建议: 谨慎考虑是否继续")

else:
    print("\n❌ 结果不一致")
    print()
    print(f"训练集 Cohen's d = {d_train:+.4f}")
    print(f"测试集 Cohen's d = {d_test:+.4f}")
    print()
    print("结论: 假设可能与数据分布相关")
    print("建议: 需要进一步分析")

print("\n" + "=" * 70)
