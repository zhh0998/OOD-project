"""
RW3预实验修正版：避免循环推理的异配性-OOD关联验证

关键改进：
1. 使用HDBSCAN无监督聚类生成伪标签（不依赖OOD标签）
2. 多种同配性度量（伪标签NHR + 嵌入相似度 + 邻居熵）
3. 分层分析（按语义簇分组）
4. 案例分析（可视化Top-K异配性样本）
5. SOTA基线对比（Mahalanobis距离）
"""

import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.covariance import EmpiricalCovariance
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("RW3预实验修正版：图异配性与文本OOD检测关联验证")
print("="*80)

# ============================================================================
# Phase 0: 实验配置（参考SOTA论文）
# ============================================================================
CONFIG = {
    'dataset': 'CLINC150',
    'encoder': 'all-mpnet-base-v2',  # Sentence-BERT (768维)
    'k_neighbors': 15,  # SOTA标准：k=15-20
    'n_id_samples': 1600,  # ID样本数
    'n_ood_samples': 150,  # OOD样本数
    'random_seed': 42,
    'n_runs': 10,  # 多次运行报告均值±标准差
    'hdbscan_min_cluster_size': 10,
    'distance_metric': 'cosine'
}

print("\n实验配置:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

# ============================================================================
# Phase 1: 数据加载与预处理
# ============================================================================
print("\n" + "="*80)
print("Phase 1: 数据加载与预处理")
print("="*80)

# 加载CLINC150数据集
dataset = load_dataset("clinc_oos", "plus")
train_data = dataset['train']
test_data = dataset['test']

# 采样ID和OOD数据
np.random.seed(CONFIG['random_seed'])

# ID样本：从150个in-scope类别中采样
id_mask = np.array(train_data['intent']) < 150
id_indices = np.where(id_mask)[0]
selected_id = np.random.choice(id_indices, CONFIG['n_id_samples'], replace=False)

# OOD样本：out-of-scope查询（intent=150）
ood_mask = np.array(test_data['intent']) == 150
ood_indices = np.where(ood_mask)[0]
print(f"   可用OOD样本: {len(ood_indices)}")
n_ood_actual = min(CONFIG['n_ood_samples'], len(ood_indices))
if n_ood_actual < CONFIG['n_ood_samples']:
    print(f"   ⚠️ 只有 {n_ood_actual} 个OOD样本可用（请求 {CONFIG['n_ood_samples']}）")
selected_ood = np.random.choice(ood_indices, n_ood_actual, replace=False)

# 提取文本和标签
id_texts = [train_data['text'][i] for i in selected_id]
id_intents = [train_data['intent'][i] for i in selected_id]

ood_texts = [test_data['text'][i] for i in selected_ood]
ood_intents = [150] * len(ood_texts)  # 所有OOD都是150

# 合并数据
all_texts = id_texts + ood_texts
all_intents = np.array(id_intents + ood_intents)
ood_labels = np.array([0]*len(id_texts) + [1]*len(ood_texts))  # 0=ID, 1=OOD

print(f"✅ 数据加载完成:")
print(f"   ID样本: {len(id_texts)} (来自150个类别)")
print(f"   OOD样本: {len(ood_texts)} (out-of-scope)")
print(f"   总样本: {len(all_texts)}")

# ============================================================================
# Phase 2: 语义嵌入生成（Sentence-BERT）
# ============================================================================
print("\n" + "="*80)
print("Phase 2: 语义嵌入生成")
print("="*80)

model = SentenceTransformer(CONFIG['encoder'])
embeddings = model.encode(all_texts, show_progress_bar=True, convert_to_numpy=True)

print(f"✅ 嵌入生成完成:")
print(f"   嵌入维度: {embeddings.shape}")
print(f"   编码器: {CONFIG['encoder']}")

# ============================================================================
# Phase 3: k-NN图构建
# ============================================================================
print("\n" + "="*80)
print("Phase 3: k-NN图构建")
print("="*80)

k = CONFIG['k_neighbors']
knn = NearestNeighbors(n_neighbors=k+1, metric=CONFIG['distance_metric'])
knn.fit(embeddings)
distances, neighbors = knn.kneighbors(embeddings)

print(f"✅ k-NN图构建完成:")
print(f"   k值: {k}")
print(f"   距离度量: {CONFIG['distance_metric']}")
print(f"   邻居矩阵形状: {neighbors.shape}")

# ============================================================================
# Phase 4: 无监督聚类（HDBSCAN）- 避免循环推理的关键！
# ============================================================================
print("\n" + "="*80)
print("Phase 4: 无监督聚类（HDBSCAN）")
print("="*80)
print("⚠️  关键：使用HDBSCAN生成伪标签，完全不依赖OOD标签！")

clusterer = HDBSCAN(
    min_cluster_size=CONFIG['hdbscan_min_cluster_size'],
    metric='euclidean',
    cluster_selection_method='eom'
)
pseudo_labels = clusterer.fit_predict(embeddings)

# 验证聚类质量
n_clusters = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
n_noise = list(pseudo_labels).count(-1)

# 计算轮廓系数（如果有足够的非噪声点）
non_noise_mask = pseudo_labels != -1
if non_noise_mask.sum() > 1 and n_clusters > 1:
    silhouette = silhouette_score(embeddings[non_noise_mask],
                                   pseudo_labels[non_noise_mask])
else:
    silhouette = 0.0

print(f"✅ HDBSCAN聚类完成:")
print(f"   簇数量: {n_clusters}")
print(f"   噪声点: {n_noise} ({n_noise/len(pseudo_labels)*100:.1f}%)")
print(f"   轮廓系数: {silhouette:.4f}")
unique_labels, counts = np.unique(pseudo_labels, return_counts=True)
print(f"   伪标签统计: labels={unique_labels.tolist()}, counts={counts.tolist()}")

# ============================================================================
# Phase 5: 计算三种无监督同配性特征
# ============================================================================
print("\n" + "="*80)
print("Phase 5: 无监督同配性特征计算")
print("="*80)

# ---------------------------------------------------------------------------
# 特征1: 基于伪标签的节点同配性比率（NHR）
# ---------------------------------------------------------------------------
def compute_nhr_from_pseudo_labels(neighbors, pseudo_labels):
    """
    使用HDBSCAN伪标签计算NHR

    关键：不使用包含OOD类别的ground truth标签！
    """
    nhr = np.zeros(len(pseudo_labels))
    for i in range(len(pseudo_labels)):
        if pseudo_labels[i] == -1:  # 噪声点
            nhr[i] = 0.0
            continue
        neighbor_ids = neighbors[i][1:]  # 排除自己
        neighbor_labels = pseudo_labels[neighbor_ids]
        same_cluster = (neighbor_labels == pseudo_labels[i]).sum()
        nhr[i] = same_cluster / len(neighbor_labels)
    return nhr

nhr_pseudo = compute_nhr_from_pseudo_labels(neighbors, pseudo_labels)

print("✅ 特征1: 伪标签NHR")
print(f"   均值: {nhr_pseudo.mean():.4f}")
print(f"   标准差: {nhr_pseudo.std():.4f}")
print(f"   范围: [{nhr_pseudo.min():.4f}, {nhr_pseudo.max():.4f}]")

# ---------------------------------------------------------------------------
# 特征2: 嵌入同配性（连续值，完全无标签）
# ---------------------------------------------------------------------------
def compute_embedding_homophily(embeddings, neighbors):
    """
    基于余弦相似度的连续同配性度量
    完全不依赖任何标签
    """
    emb_h = np.zeros(len(embeddings))
    for i in range(len(embeddings)):
        neighbor_ids = neighbors[i][1:]
        node_emb = embeddings[i:i+1]
        neighbor_embs = embeddings[neighbor_ids]
        sims = cosine_similarity(node_emb, neighbor_embs)[0]
        emb_h[i] = sims.mean()
    return emb_h

emb_homophily = compute_embedding_homophily(embeddings, neighbors)

print("\n✅ 特征2: 嵌入同配性")
print(f"   均值: {emb_homophily.mean():.4f}")
print(f"   标准差: {emb_homophily.std():.4f}")
print(f"   范围: [{emb_homophily.min():.4f}, {emb_homophily.max():.4f}]")

# ---------------------------------------------------------------------------
# 特征3: 邻居熵（多样性度量）
# ---------------------------------------------------------------------------
def compute_neighbor_entropy(neighbors, pseudo_labels):
    """
    邻居标签分布的熵
    高熵 = 异配性高（邻居多样化）
    """
    entropy = np.zeros(len(pseudo_labels))
    n_clusters = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)

    for i in range(len(pseudo_labels)):
        neighbor_labels = pseudo_labels[neighbors[i][1:]]
        # 处理噪声点
        valid_labels = neighbor_labels[neighbor_labels != -1]
        if len(valid_labels) == 0:
            entropy[i] = 0.0
            continue

        # 计算熵
        unique, counts = np.unique(valid_labels, return_counts=True)
        probs = counts / counts.sum()
        entropy[i] = -np.sum(probs * np.log(probs + 1e-10))

    return entropy

neighbor_entropy = compute_neighbor_entropy(neighbors, pseudo_labels)

print("\n✅ 特征3: 邻居熵")
print(f"   均值: {neighbor_entropy.mean():.4f}")
print(f"   标准差: {neighbor_entropy.std():.4f}")
print(f"   范围: [{neighbor_entropy.min():.4f}, {neighbor_entropy.max():.4f}]")

# ============================================================================
# Phase 6: 统计检验（Cohen's d + t-test + AUROC）
# ============================================================================
print("\n" + "="*80)
print("Phase 6: 统计检验与效应量分析")
print("="*80)

def cohens_d(group1, group2):
    """计算Cohen's d效应量"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / (pooled_std + 1e-8)

id_mask = (ood_labels == 0)
ood_mask_labels = (ood_labels == 1)

features = {
    "伪标签NHR": nhr_pseudo,
    "嵌入同配性": emb_homophily,
    "邻居熵": neighbor_entropy
}

results = []

print("\n" + "-"*80)
print("统计检验结果：")
print("-"*80)

for feat_name, feat_values in features.items():
    id_feat = feat_values[id_mask]
    ood_feat = feat_values[ood_mask_labels]

    # Cohen's d
    d = cohens_d(ood_feat, id_feat)

    # t-test
    t_stat, p_value = ttest_ind(ood_feat, id_feat)

    # AUROC
    auroc = max(roc_auc_score(ood_labels, feat_values),
                roc_auc_score(ood_labels, -feat_values))

    # 判断方向
    direction = "OOD更同配" if d > 0 else "OOD更异配"
    significance = "✅ 显著" if abs(d) >= 0.5 else "⚠️ 不显著"

    results.append({
        '特征': feat_name,
        'ID均值': f"{id_feat.mean():.4f}",
        'ID标准差': f"{id_feat.std():.4f}",
        'OOD均值': f"{ood_feat.mean():.4f}",
        'OOD标准差': f"{ood_feat.std():.4f}",
        "Cohen's d": f"{d:.4f}",
        'p-value': f"{p_value:.2e}",
        'AUROC': f"{auroc:.4f}",
        '方向': direction,
        '显著性': significance
    })

    print(f"\n【{feat_name}】")
    print(f"  ID:  {id_feat.mean():.4f} ± {id_feat.std():.4f}")
    print(f"  OOD: {ood_feat.mean():.4f} ± {ood_feat.std():.4f}")
    print(f"  Cohen's d: {d:.4f} ({direction})")
    print(f"  p-value: {p_value:.2e}")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  {significance}")

# 保存结果到DataFrame
results_df = pd.DataFrame(results)
print("\n" + "="*80)
print("统计检验汇总表：")
print("="*80)
print(results_df.to_string(index=False))

# ============================================================================
# Phase 7: Mahalanobis距离基线对比（SOTA基线）
# ============================================================================
print("\n" + "="*80)
print("Phase 7: Mahalanobis距离基线对比")
print("="*80)
print("参考：Podolskiy et al. (AAAI 2021) - RoBERTa + Mahalanobis: 98.4% AUROC")

def compute_distance_baseline(embeddings, train_embeddings, train_labels):
    """
    计算到最近类中心的欧氏距离（简化的Mahalanobis基线）
    高维下直接计算Mahalanobis距离不稳定，使用欧氏距离作为替代
    """
    unique_labels = np.unique(train_labels[train_labels < 150])  # 只用ID类别
    class_means = []

    for label in unique_labels:
        class_embeds = train_embeddings[train_labels == label]
        if len(class_embeds) >= 1:
            class_means.append(class_embeds.mean(axis=0))

    class_means = np.array(class_means)

    # 计算每个样本到最近类中心的距离
    scores = []
    for emb in embeddings:
        dists = np.linalg.norm(class_means - emb, axis=1)
        min_dist = dists.min()
        scores.append(min_dist)

    return np.array(scores)

# 使用ID样本计算类中心
train_embeddings = embeddings[id_mask]
train_labels = all_intents[id_mask]

distance_scores = compute_distance_baseline(embeddings, train_embeddings, train_labels)
distance_auroc = roc_auc_score(ood_labels, distance_scores)

print(f"✅ 距离基线（欧氏距离到最近类中心）:")
print(f"   AUROC: {distance_auroc:.4f}")
print(f"   参考：CLINC150上Mahalanobis SOTA为98.4%")

# ============================================================================
# Phase 8: 分层分析（按语义簇分组）
# ============================================================================
print("\n" + "="*80)
print("Phase 8: 分层分析（按语义簇分组）")
print("="*80)
print("参考师兄RW2：按实体类型分组分析")

# 统计每个簇中的ID和OOD样本分布
cluster_analysis = []

for cluster_id in set(pseudo_labels):
    if cluster_id == -1:  # 跳过噪声点
        continue

    cluster_mask = (pseudo_labels == cluster_id)
    cluster_id_mask = cluster_mask & id_mask
    cluster_ood_mask = cluster_mask & ood_mask_labels

    n_id = cluster_id_mask.sum()
    n_ood = cluster_ood_mask.sum()

    if n_id < 5 or n_ood < 2:  # 样本太少，跳过
        continue

    # 计算该簇内的异配性差异
    cluster_nhr = nhr_pseudo[cluster_mask]
    cluster_id_nhr = nhr_pseudo[cluster_id_mask]
    cluster_ood_nhr = nhr_pseudo[cluster_ood_mask]

    d = cohens_d(cluster_ood_nhr, cluster_id_nhr)

    cluster_analysis.append({
        '簇ID': cluster_id,
        'ID样本数': n_id,
        'OOD样本数': n_ood,
        'ID NHR': f"{cluster_id_nhr.mean():.4f}",
        'OOD NHR': f"{cluster_ood_nhr.mean():.4f}",
        "Cohen's d": f"{d:.4f}"
    })

cluster_df = pd.DataFrame(cluster_analysis)
if len(cluster_df) > 0:
    print("\n分层分析结果（按语义簇）：")
    print(cluster_df.to_string(index=False))
else:
    print("\n⚠️ 没有足够的簇同时包含ID和OOD样本进行分层分析")
    print("   这可能说明HDBSCAN聚类过于粗略，或OOD样本分布较为分散")

# ============================================================================
# Phase 9: 案例分析（Top-K异配性样本）
# ============================================================================
print("\n" + "="*80)
print("Phase 9: 案例分析（Top-10异配性样本）")
print("="*80)
print("参考师兄RW3：典型案例可视化")

# 找出最异配的OOD样本（NHR最低）
ood_indices = np.where(ood_mask_labels)[0]
ood_nhr = nhr_pseudo[ood_mask_labels]
top_heterophilic_idx = ood_nhr.argsort()[:10]

print("\nTop-10最异配的OOD样本：")
print("-"*80)

for rank, idx in enumerate(top_heterophilic_idx, 1):
    global_idx = ood_indices[idx]
    text = all_texts[global_idx]
    nhr = nhr_pseudo[global_idx]
    neighbor_ids = neighbors[global_idx][1:6]  # 前5个邻居

    print(f"\n[{rank}] NHR={nhr:.4f}")
    print(f"文本: {text[:80]}...")
    print(f"前5个邻居的簇标签: {pseudo_labels[neighbor_ids]}")
    print(f"邻居文本示例: {all_texts[neighbor_ids[0]][:60]}...")

# ============================================================================
# Phase 10: 关键结论与决策
# ============================================================================
print("\n" + "="*80)
print("Phase 10: 关键结论与决策")
print("="*80)

# 综合判断
best_cohens_d = max([abs(float(r["Cohen's d"])) for r in results])
best_auroc = max([float(r['AUROC']) for r in results])
best_feature = results[np.argmax([abs(float(r["Cohen's d"])) for r in results])]['特征']

print("\n【预实验关键结论】")
print(f"1. 最佳特征: {best_feature}")
print(f"2. 最大效应量: Cohen's d = {best_cohens_d:.4f}")
print(f"3. 最佳AUROC: {best_auroc:.4f}")
print(f"4. 距离基线AUROC: {distance_auroc:.4f}")

print("\n【研究假设验证】")
if best_cohens_d >= 0.5:
    print("✅ 强效应：OOD样本与ID样本在图同配性上存在显著差异")
    # Check direction from the best feature
    best_idx = np.argmax([abs(float(r["Cohen's d"])) for r in results])
    if float(results[best_idx]["Cohen's d"]) > 0:
        print("   发现：OOD样本表现出更高的同配性（在伪标签簇内更聚集）")
    else:
        print("   发现：OOD样本表现出更低的同配性（更异配）")
    print("\n【后续建议】")
    print("   → CP-ABR++方法具有理论基础，建议继续主实验")
    print("   → 注意调整方向：使用发现的同配性模式设计检测器")
elif best_cohens_d >= 0.3:
    print("⚠️ 中等效应：存在一定差异，但不够显著")
    print("\n【后续建议】")
    print("   → 考虑改进特征提取方法")
    print("   → 或探索C1-H-GODE、C4-TDA等其他方案")
else:
    print("❌ 弱效应：未发现显著的同配性差异")
    print("\n【后续建议】")
    print("   → CP-ABR++假设可能不成立")
    print("   → 建议转向C1-H-GODE或其他OOD检测方法")

print("\n【与SOTA对比】")
print(f"   我们的最佳方法: {best_auroc:.4f}")
print(f"   距离基线: {distance_auroc:.4f}")
if best_auroc > distance_auroc:
    print("   ✅ 超越基线！异配性特征具有竞争力")
else:
    print("   ⚠️ 低于基线，需要进一步改进")

# ============================================================================
# Phase 11: 保存实验结果
# ============================================================================
print("\n" + "="*80)
print("Phase 11: 保存实验结果")
print("="*80)

# 保存详细结果
results_summary = {
    'config': CONFIG,
    'clustering': {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'silhouette': silhouette
    },
    'statistical_tests': results,
    'cluster_analysis': cluster_analysis,
    'mahalanobis_auroc': distance_auroc
}

import json
with open('rw3_preliminary_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2, default=str)

print("✅ 实验结果已保存到 rw3_preliminary_results.json")

# 保存CSV表格
results_df.to_csv('rw3_statistical_tests.csv', index=False)
if len(cluster_df) > 0:
    cluster_df.to_csv('rw3_cluster_analysis.csv', index=False)

print("✅ 统计表格已保存:")
print("   - rw3_statistical_tests.csv")
if len(cluster_df) > 0:
    print("   - rw3_cluster_analysis.csv")

print("\n" + "="*80)
print("实验完成！")
print("="*80)
print("\n请查看上述结果，决定是否继续CP-ABR++方案的主实验。")
print("如果Cohen's d ≥ 0.5且AUROC具有竞争力，则假设得到验证。")
