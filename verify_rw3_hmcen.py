#!/usr/bin/env python3
"""
RW3-HMCEN 核心假设扩展验证
假设：异配性高的节点更接近near-OOD样本
在多个数据集上验证
"""

import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("RW3-HMCEN 假设扩展验证")
print("假设: 异配性高的节点更接近near-OOD样本")
print("=" * 70)

# ============================================================
# A. 创建合成OOD检测数据集
# ============================================================
print("\n[A] 创建合成OOD检测数据集...")

np.random.seed(42)

def create_synthetic_ood_dataset(n_id=500, n_ood=200, n_classes=5):
    """
    创建合成的ID/OOD数据集
    - ID数据：5个清晰的类别（intent）
    - OOD数据：与某些ID类别语义相近但不完全匹配
    """
    # ID类别模板
    id_templates = {
        0: ["book a flight to {}", "reserve a ticket to {}", "fly to {}", "travel to {} by plane"],
        1: ["what's the weather in {}", "tell me weather for {}", "forecast for {}", "temperature in {}"],
        2: ["play {} music", "listen to {}", "put on {} songs", "start playing {}"],
        3: ["set alarm for {}", "wake me up at {}", "reminder at {}", "alert me at {}"],
        4: ["order {} food", "get {} delivery", "buy {} online", "purchase {}"]
    }

    # OOD模板（与某些ID类别相近但不同）
    ood_templates = [
        "book a hotel in {}",  # 类似flight但不同
        "what's the news in {}",  # 类似weather但不同
        "watch {} video",  # 类似music但不同
        "schedule meeting at {}",  # 类似alarm但不同
        "find {} store nearby",  # 类似order但不同
        "translate {} to spanish",  # 完全不同
        "calculate {} plus {}",  # 完全不同
    ]

    locations = ["New York", "London", "Tokyo", "Paris", "Beijing", "Sydney", "Berlin"]
    items = ["rock", "jazz", "pizza", "sushi", "coffee", "books", "shoes"]
    times = ["8am", "noon", "5pm", "midnight", "tomorrow", "next week"]

    id_texts = []
    id_labels = []
    ood_texts = []

    # 生成ID数据
    for _ in range(n_id):
        class_id = np.random.randint(0, n_classes)
        template = np.random.choice(id_templates[class_id])
        if "{}" in template:
            fill = np.random.choice(locations + items + times)
            text = template.format(fill)
        else:
            text = template
        id_texts.append(text)
        id_labels.append(class_id)

    # 生成OOD数据
    for _ in range(n_ood):
        template = np.random.choice(ood_templates)
        fills = [np.random.choice(locations + items + times) for _ in range(template.count("{}"))]
        text = template
        for fill in fills:
            text = text.replace("{}", fill, 1)
        ood_texts.append(text)

    return id_texts, id_labels, ood_texts

# 创建3个不同规模的合成数据集
datasets = {}

# Dataset 1: Intent-Small
id_texts1, id_labels1, ood_texts1 = create_synthetic_ood_dataset(300, 100, 5)
datasets['Synthetic-Intent'] = {
    'id_texts': id_texts1, 'id_labels': id_labels1, 'ood_texts': ood_texts1
}

# Dataset 2: Intent-Medium (更多类别)
id_texts2, id_labels2, ood_texts2 = create_synthetic_ood_dataset(500, 150, 5)
datasets['Synthetic-Medium'] = {
    'id_texts': id_texts2, 'id_labels': id_labels2, 'ood_texts': ood_texts2
}

# Dataset 3: 添加更多变化
np.random.seed(123)
id_texts3, id_labels3, ood_texts3 = create_synthetic_ood_dataset(400, 120, 5)
datasets['Synthetic-Varied'] = {
    'id_texts': id_texts3, 'id_labels': id_labels3, 'ood_texts': ood_texts3
}

for name, data in datasets.items():
    print(f"    {name}: ID={len(data['id_texts'])}, OOD={len(data['ood_texts'])}")

# ============================================================
# B. 计算异配性
# ============================================================
print("\n[B] 计算异配性...")

def compute_heterophily(texts, labels, n_neighbors=5):
    """
    计算文本的异配性
    使用TF-IDF编码，基于k近邻计算
    """
    # TF-IDF编码
    vectorizer = TfidfVectorizer(max_features=500)
    embeddings = vectorizer.fit_transform(texts).toarray()

    # 计算余弦相似度矩阵
    sim_matrix = cosine_similarity(embeddings)

    heterophily_scores = []

    for i in range(len(texts)):
        # 获取k个最近邻（排除自己）
        sims = sim_matrix[i].copy()
        sims[i] = -1  # 排除自己

        # 找k个最相似的
        top_k_idx = np.argsort(sims)[-n_neighbors:]
        neighbor_sims = sims[top_k_idx]

        # 异配性 = 1 - 平均相似度
        h_score = 1.0 - np.mean(neighbor_sims)
        heterophily_scores.append(h_score)

    return np.array(heterophily_scores), embeddings

results = {}

for name, data in datasets.items():
    print(f"\n    处理 {name}...")

    # 合并ID和OOD文本
    all_texts = data['id_texts'] + data['ood_texts']
    n_id = len(data['id_texts'])
    n_ood = len(data['ood_texts'])

    # 创建标签：0=ID, 1=OOD
    is_ood = np.array([0] * n_id + [1] * n_ood)

    # 计算异配性
    h_scores, embeddings = compute_heterophily(all_texts, is_ood)

    # 分离ID和OOD的异配性
    id_heterophily = h_scores[:n_id]
    ood_heterophily = h_scores[n_id:]

    # 统计分析
    mean_id = np.mean(id_heterophily)
    mean_ood = np.mean(ood_heterophily)
    std_id = np.std(id_heterophily, ddof=1)
    std_ood = np.std(ood_heterophily, ddof=1)

    # Cohen's d
    pooled_std = np.sqrt(((n_id-1)*std_id**2 + (n_ood-1)*std_ood**2) / (n_id + n_ood - 2))
    cohens_d = (mean_ood - mean_id) / pooled_std if pooled_std > 0 else 0

    # t检验
    t_stat, p_value = stats.ttest_ind(ood_heterophily, id_heterophily)

    results[name] = {
        'mean_id': mean_id,
        'mean_ood': mean_ood,
        'std_id': std_id,
        'std_ood': std_ood,
        'cohens_d': cohens_d,
        't_stat': t_stat,
        'p_value': p_value,
        'n_id': n_id,
        'n_ood': n_ood,
        'id_heterophily': id_heterophily,
        'ood_heterophily': ood_heterophily
    }

    print(f"      ID异配性: {mean_id:.4f} ± {std_id:.4f}")
    print(f"      OOD异配性: {mean_ood:.4f} ± {std_ood:.4f}")
    print(f"      Cohen's d: {cohens_d:.4f}")

# ============================================================
# C. 添加已有实验结果
# ============================================================
print("\n[C] 整合已有实验结果...")

# 用户提供的已有结果
existing_results = {
    'CLINC150': {'cohens_d': 0.60, 'direction': 'positive'},
    'Banking77': {'cohens_d': 0.49, 'direction': 'positive'}
}

print("    已有结果:")
for name, res in existing_results.items():
    print(f"      {name}: Cohen's d = {res['cohens_d']:.2f} ({res['direction']})")

# ============================================================
# D. 可视化
# ============================================================
print("\n[D] 生成可视化...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 子图1: 各数据集Cohen's d对比
ax1 = axes[0, 0]
all_names = list(results.keys()) + list(existing_results.keys())
all_cohens_d = [results[n]['cohens_d'] for n in results.keys()] + \
               [existing_results[n]['cohens_d'] for n in existing_results.keys()]

colors = ['steelblue' if d > 0.3 else 'orange' if d > 0 else 'red' for d in all_cohens_d]
bars = ax1.bar(range(len(all_names)), all_cohens_d, color=colors, edgecolor='black')
ax1.axhline(y=0.3, color='green', linestyle='--', linewidth=2, label='Threshold (0.3)')
ax1.axhline(y=0, color='gray', linestyle='-', linewidth=1)
ax1.set_xticks(range(len(all_names)))
ax1.set_xticklabels(all_names, rotation=45, ha='right')
ax1.set_ylabel("Cohen's d", fontsize=12)
ax1.set_title("Cohen's d Across Datasets\n(OOD vs ID Heterophily)", fontsize=13)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for i, (bar, d) in enumerate(zip(bars, all_cohens_d)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{d:.2f}', ha='center', va='bottom', fontsize=10)

# 子图2: 第一个合成数据集的异配性分布
ax2 = axes[0, 1]
first_ds = list(results.keys())[0]
ax2.hist(results[first_ds]['id_heterophily'], bins=20, alpha=0.6, label='ID', color='blue', density=True)
ax2.hist(results[first_ds]['ood_heterophily'], bins=20, alpha=0.6, label='OOD', color='red', density=True)
ax2.axvline(x=results[first_ds]['mean_id'], color='blue', linestyle='--', linewidth=2)
ax2.axvline(x=results[first_ds]['mean_ood'], color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Heterophily Score', fontsize=12)
ax2.set_ylabel('Density', fontsize=12)
ax2.set_title(f'{first_ds}: Heterophily Distribution\n(Cohen\'s d = {results[first_ds]["cohens_d"]:.3f})', fontsize=13)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 子图3: ID vs OOD 平均异配性对比
ax3 = axes[1, 0]
x = np.arange(len(results))
width = 0.35
id_means = [results[n]['mean_id'] for n in results.keys()]
ood_means = [results[n]['mean_ood'] for n in results.keys()]

bars1 = ax3.bar(x - width/2, id_means, width, label='ID', color='blue', alpha=0.7)
bars2 = ax3.bar(x + width/2, ood_means, width, label='OOD', color='red', alpha=0.7)
ax3.set_xticks(x)
ax3.set_xticklabels(list(results.keys()), rotation=45, ha='right')
ax3.set_ylabel('Mean Heterophily', fontsize=12)
ax3.set_title('ID vs OOD Mean Heterophily', fontsize=13)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# 子图4: 验证结果总结
ax4 = axes[1, 1]
ax4.axis('off')

# 创建总结表格
summary_text = "Validation Summary\n" + "=" * 40 + "\n\n"

pass_count = 0
total_count = len(all_cohens_d)

for name, d in zip(all_names, all_cohens_d):
    status = "✅ PASS" if d > 0.3 else "⚠️ WEAK" if d > 0 else "❌ FAIL"
    if d > 0.3:
        pass_count += 1
    summary_text += f"{name:20s}: d={d:+.3f} {status}\n"

summary_text += "\n" + "-" * 40 + "\n"
summary_text += f"Pass Rate: {pass_count}/{total_count} ({100*pass_count/total_count:.0f}%)\n"
summary_text += f"Threshold: Cohen's d > 0.3\n"

ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('rw3_hmcen_validation.png', dpi=150, bbox_inches='tight')
print("    图表已保存: rw3_hmcen_validation.png")

# ============================================================
# E. 结论
# ============================================================
print("\n" + "=" * 70)
print("RW3-HMCEN 验证结果")
print("=" * 70)

print("\n各数据集结果:")
print("-" * 70)
cohens_header = "Cohen's d"
print(f"{'Dataset':<20} {cohens_header:>12} {'Direction':>12} {'Status':>10}")
print("-" * 70)

pass_count = 0
positive_count = 0
total = len(all_cohens_d)

for name, d in zip(all_names, all_cohens_d):
    direction = "正相关" if d > 0 else "负相关"
    if d > 0:
        positive_count += 1
    if d > 0.3:
        status = "✅ PASS"
        pass_count += 1
    elif d > 0:
        status = "⚠️ WEAK"
    else:
        status = "❌ FAIL"
    print(f"{name:<20} {d:>+12.4f} {direction:>12} {status:>10}")

print("-" * 70)

print(f"\n统计汇总:")
print(f"  通过率 (d>0.3): {pass_count}/{total} ({100*pass_count/total:.0f}%)")
print(f"  正相关率: {positive_count}/{total} ({100*positive_count/total:.0f}%)")

print("\n" + "-" * 70)

# 最终判断
if pass_count >= total * 0.5 and positive_count == total:
    print("\n✅ RW3-HMCEN 核心假设验证通过！")
    print()
    print(f"   • {pass_count}/{total} 数据集Cohen's d > 0.3")
    print(f"   • {positive_count}/{total} 数据集方向一致（正相关）")
    print()
    print("   结论: OOD样本确实表现出更高的异配性")
    print("   建议: 继续RW3-HMCEN的后续研究")
    result = "PASS"
elif positive_count >= total * 0.8:
    print("\n⚠️ RW3-HMCEN 核心假设部分通过")
    print()
    print(f"   • {pass_count}/{total} 数据集达到强效应")
    print(f"   • {positive_count}/{total} 数据集方向一致")
    print()
    print("   结论: 假设方向正确，但效应量不够稳定")
    print("   建议: 需要进一步优化方法增强效应")
    result = "PARTIAL"
else:
    print("\n❌ RW3-HMCEN 核心假设验证失败")
    print()
    print(f"   • 通过率不足50%或存在负相关")
    print()
    print("   结论: 异配性与OOD的关系不稳定")
    print("   建议: 重新审视假设，考虑其他特征")
    result = "FAIL"

print("\n" + "=" * 70)

# 保存结果
with open('rw3_validation_result.txt', 'w') as f:
    f.write(f"RW3-HMCEN Validation Result: {result}\n")
    f.write(f"Pass Rate: {pass_count}/{total}\n")
    f.write(f"Positive Rate: {positive_count}/{total}\n")
    for name, d in zip(all_names, all_cohens_d):
        f.write(f"{name}: {d:.4f}\n")
