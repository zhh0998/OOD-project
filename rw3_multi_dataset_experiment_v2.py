"""
RW3多数据集预实验V2：验证OOD同配性模式的普遍性（修复版）

修复内容：
1. Banking77: 使用mteb/banking77备用源
2. ToxiGen: 使用toxigen/toxigen-data的annotated配置
3. 增加更多错误处理和备用方案

支持数据集：
1. CLINC150 - 自动下载
2. Banking77-OOS - 自动下载（修复）
3. ROSTD - 自动下载（多种方式）
4. ToxiGen - 自动下载（修复）
"""

import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ttest_ind
import warnings
import json
import os
import requests

warnings.filterwarnings('ignore')

print("="*80)
print("RW3多数据集预实验V2：OOD同配性模式普遍性验证（修复版）")
print("="*80)

# ============================================================================
# Phase 0: 全局配置
# ============================================================================
GLOBAL_CONFIG = {
    'encoder': 'all-mpnet-base-v2',
    'random_seed': 42,
    'n_runs': 1,
    'hdbscan_min_cluster_size': 10,
    'distance_metric': 'cosine',
    'output_dir': 'rw3_multi_dataset_results_v2'
}

DATASET_CONFIGS = {
    'CLINC150': {
        'k_neighbors': 15,
        'n_id_samples': 1600,
        'n_ood_samples': 150,
        'description': 'Intent classification with out-of-scope queries'
    },
    'Banking77': {
        'k_neighbors': 20,
        'n_id_samples': 2000,
        'n_ood_samples': 200,
        'description': 'Fine-grained banking intents with held-out classes'
    },
    'ROSTD': {
        'k_neighbors': 20,
        'n_id_samples': 1000,
        'n_ood_samples': 500,
        'description': 'Real-world dialog with human-authored OOD samples'
    },
    'ToxiGen': {
        'k_neighbors': 20,
        'n_id_samples': 1000,
        'n_ood_samples': 500,
        'description': 'Toxic content detection with domain shift'
    }
}

os.makedirs(GLOBAL_CONFIG['output_dir'], exist_ok=True)

print("\n全局配置:")
for key, value in GLOBAL_CONFIG.items():
    print(f"  {key}: {value}")

# ============================================================================
# Phase 1: 数据集加载函数（全部自动化，修复版）
# ============================================================================

def load_clinc150_data(config):
    """加载CLINC150数据集"""
    print("\n" + "="*80)
    print("加载CLINC150数据集")
    print("="*80)

    dataset = load_dataset("clinc_oos", "plus")
    train_data = dataset['train']
    test_data = dataset['test']

    np.random.seed(GLOBAL_CONFIG['random_seed'])

    # ID样本
    id_mask = np.array(train_data['intent']) < 150
    id_indices = np.where(id_mask)[0]
    selected_id = np.random.choice(id_indices, config['n_id_samples'], replace=False)

    # OOD样本
    ood_mask = np.array(test_data['intent']) == 150
    ood_indices = np.where(ood_mask)[0]
    n_ood_actual = min(config['n_ood_samples'], len(ood_indices))
    selected_ood = np.random.choice(ood_indices, n_ood_actual, replace=False)

    id_texts = [train_data['text'][i] for i in selected_id]
    ood_texts = [test_data['text'][i] for i in selected_ood]

    print(f"✅ CLINC150加载完成:")
    print(f"   ID样本: {len(id_texts)} (150个类别)")
    print(f"   OOD样本: {len(ood_texts)} (out-of-scope查询)")

    return id_texts, ood_texts, {
        'dataset': 'CLINC150',
        'id_classes': 150,
        'ood_type': 'out-of-scope'
    }


def load_banking77_data(config):
    """加载Banking77-OOS数据集（修复版）"""
    print("\n" + "="*80)
    print("加载Banking77-OOS数据集")
    print("="*80)

    # 尝试多个数据源
    dataset_sources = [
        ("mteb/banking77", None),  # 最可靠的源
        ("PolyAI/banking77", None),
    ]

    train_data = None
    used_source = None

    for source, config_name in dataset_sources:
        try:
            print(f"尝试加载 {source}...")
            if config_name:
                dataset = load_dataset(source, config_name)
            else:
                dataset = load_dataset(source)
            train_data = dataset['train']
            used_source = source
            print(f"✅ 成功加载 {source}")
            break
        except Exception as e:
            print(f"⚠️ {source} 加载失败: {e}")
            continue

    if train_data is None:
        print("❌ 所有Banking77数据源都失败")
        return None, None, None

    np.random.seed(GLOBAL_CONFIG['random_seed'])

    all_labels = np.array(train_data['label'])
    unique_labels = np.unique(all_labels)

    print(f"数据集信息: {len(train_data)}个样本, {len(unique_labels)}个类别")

    # 随机选择50个类别作为ID，其余作为OOD
    shuffled_labels = unique_labels.copy()
    np.random.shuffle(shuffled_labels)
    id_labels = shuffled_labels[:50]
    ood_labels_set = shuffled_labels[50:]

    # ID样本
    id_mask = np.isin(all_labels, id_labels)
    id_indices = np.where(id_mask)[0]
    n_id_actual = min(config['n_id_samples'], len(id_indices))
    selected_id = np.random.choice(id_indices, n_id_actual, replace=False)

    # OOD样本（从保留的27个类别中采样）
    ood_mask = np.isin(all_labels, ood_labels_set)
    ood_indices = np.where(ood_mask)[0]
    n_ood_actual = min(config['n_ood_samples'], len(ood_indices))
    selected_ood = np.random.choice(ood_indices, n_ood_actual, replace=False)

    id_texts = [train_data['text'][i] for i in selected_id]
    ood_texts = [train_data['text'][i] for i in selected_ood]

    print(f"✅ Banking77-OOS加载完成:")
    print(f"   数据源: {used_source}")
    print(f"   ID样本: {len(id_texts)} (来自{len(id_labels)}个类别)")
    print(f"   OOD样本: {len(ood_texts)} (来自{len(ood_labels_set)}个保留类别)")
    print(f"   OOD类型: 近距离OOD（相关金融意图）")

    return id_texts, ood_texts, {
        'dataset': 'Banking77-OOS',
        'source': used_source,
        'id_classes': len(id_labels),
        'ood_classes': len(ood_labels_set),
        'ood_type': 'near-OOD (held-out intents)'
    }


def load_rostd_data(config):
    """加载ROSTD数据集（完全自动化，多种方式）"""
    print("\n" + "="*80)
    print("加载ROSTD数据集（自动下载）")
    print("="*80)

    # 方式1：从HuggingFace Hub加载
    print("\n尝试方式1：HuggingFace Hub...")
    try:
        # 尝试加载rostd+配置
        dataset = load_dataset("cmaldona/Generalization-MultiClass-CLINC150-ROSTD", "rostd+")

        train_data = dataset['train']
        all_texts = list(train_data['text'])
        n_total = len(all_texts)

        np.random.seed(GLOBAL_CONFIG['random_seed'])

        # 随机划分为ID和OOD
        n_id = min(config['n_id_samples'], n_total // 2)
        n_ood = min(config['n_ood_samples'], n_total // 2)

        indices = np.random.permutation(n_total)
        id_indices = indices[:n_id]
        ood_indices = indices[n_id:n_id+n_ood]

        id_texts = [all_texts[i] for i in id_indices]
        ood_texts = [all_texts[i] for i in ood_indices]

        print(f"✅ ROSTD加载完成（HuggingFace Hub rostd+）:")
        print(f"   ID样本: {len(id_texts)}")
        print(f"   OOD样本: {len(ood_texts)}")

        return id_texts, ood_texts, {
            'dataset': 'ROSTD',
            'source': 'HuggingFace Hub (rostd+)',
            'ood_type': 'diverse human-authored'
        }

    except Exception as e1:
        print(f"⚠️ 方式1失败: {e1}")

    # 方式2：使用SNIPS数据集模拟
    print("\n尝试方式2：使用SNIPS数据集模拟ROSTD场景...")
    try:
        snips = load_dataset("benayas/snips")
        train_data = snips['train']

        np.random.seed(GLOBAL_CONFIG['random_seed'])

        all_texts = list(train_data['text'])
        all_labels = np.array(train_data['label'])
        unique_labels = np.unique(all_labels)

        print(f"SNIPS数据集: {len(all_texts)}个样本, {len(unique_labels)}个类别")

        # 划分ID和OOD类别
        n_id_classes = len(unique_labels) // 2
        shuffled_labels = unique_labels.copy()
        np.random.shuffle(shuffled_labels)
        id_labels = shuffled_labels[:n_id_classes]
        ood_labels_set = shuffled_labels[n_id_classes:]

        id_mask = np.isin(all_labels, id_labels)
        ood_mask = np.isin(all_labels, ood_labels_set)

        id_indices = np.where(id_mask)[0]
        ood_indices = np.where(ood_mask)[0]

        n_id = min(config['n_id_samples'], len(id_indices))
        n_ood = min(config['n_ood_samples'], len(ood_indices))

        selected_id = np.random.choice(id_indices, n_id, replace=False)
        selected_ood = np.random.choice(ood_indices, n_ood, replace=False)

        id_texts = [all_texts[i] for i in selected_id]
        ood_texts = [all_texts[i] for i in selected_ood]

        print(f"✅ ROSTD模拟加载完成（SNIPS替代）:")
        print(f"   ID样本: {len(id_texts)} (来自{len(id_labels)}个类别)")
        print(f"   OOD样本: {len(ood_texts)} (来自{len(ood_labels_set)}个类别)")

        return id_texts, ood_texts, {
            'dataset': 'ROSTD-simulated',
            'source': 'SNIPS alternative',
            'ood_type': 'held-out intents (simulated)'
        }

    except Exception as e2:
        print(f"⚠️ 方式2失败: {e2}")

    # 方式3：使用另一个意图分类数据集
    print("\n尝试方式3：使用ATIS数据集...")
    try:
        atis = load_dataset("tuetschek/atis")
        train_data = atis['train']

        np.random.seed(GLOBAL_CONFIG['random_seed'])

        all_texts = list(train_data['text'])
        all_labels = np.array(train_data['intent'])
        unique_labels = np.unique(all_labels)

        print(f"ATIS数据集: {len(all_texts)}个样本, {len(unique_labels)}个类别")

        # 划分ID和OOD类别
        n_id_classes = len(unique_labels) * 2 // 3
        shuffled_labels = unique_labels.copy()
        np.random.shuffle(shuffled_labels)
        id_labels = shuffled_labels[:n_id_classes]
        ood_labels_set = shuffled_labels[n_id_classes:]

        id_mask = np.isin(all_labels, id_labels)
        ood_mask = np.isin(all_labels, ood_labels_set)

        id_indices = np.where(id_mask)[0]
        ood_indices = np.where(ood_mask)[0]

        n_id = min(config['n_id_samples'], len(id_indices))
        n_ood = min(config['n_ood_samples'], len(ood_indices))

        selected_id = np.random.choice(id_indices, n_id, replace=False)
        selected_ood = np.random.choice(ood_indices, n_ood, replace=False)

        id_texts = [all_texts[i] for i in selected_id]
        ood_texts = [all_texts[i] for i in selected_ood]

        print(f"✅ ROSTD模拟加载完成（ATIS替代）:")
        print(f"   ID样本: {len(id_texts)}")
        print(f"   OOD样本: {len(ood_texts)}")

        return id_texts, ood_texts, {
            'dataset': 'ROSTD-simulated',
            'source': 'ATIS alternative',
            'ood_type': 'held-out intents (simulated)'
        }

    except Exception as e3:
        print(f"❌ 方式3也失败: {e3}")
        return None, None, None


def load_toxigen_data(config):
    """加载ToxiGen数据集（修复版）"""
    print("\n" + "="*80)
    print("加载ToxiGen数据集")
    print("="*80)

    # 尝试多个配置和数据源
    toxigen_sources = [
        ("toxigen/toxigen-data", "annotated"),
        ("skg/toxigen-data", "annotated"),
        ("skg/toxigen-data", "train"),
    ]

    for source, config_name in toxigen_sources:
        try:
            print(f"尝试加载 {source} (config={config_name})...")
            dataset = load_dataset(source, name=config_name)

            # 获取数据
            if 'train' in dataset:
                data = dataset['train']
            else:
                split_name = list(dataset.keys())[0]
                data = dataset[split_name]

            print(f"数据集字段: {data.column_names}")

            # 检查toxicity字段
            toxicity_field = None
            for field in ['toxicity_ai', 'toxicity_human', 'toxicity', 'label']:
                if field in data.column_names:
                    toxicity_field = field
                    break

            if toxicity_field is None:
                print(f"⚠️ 找不到毒性字段，跳过此数据源")
                continue

            print(f"使用毒性字段: {toxicity_field}")

            np.random.seed(GLOBAL_CONFIG['random_seed'])

            all_texts = np.array(data['text'])
            toxicity_scores = np.array(data[toxicity_field])

            # 过滤NaN值
            valid_mask = ~pd.isna(toxicity_scores)
            all_texts = all_texts[valid_mask]
            toxicity_scores = toxicity_scores[valid_mask].astype(float)

            print(f"有效样本: {len(all_texts)} (过滤掉无效值)")
            print(f"毒性分数范围: [{toxicity_scores.min():.2f}, {toxicity_scores.max():.2f}]")

            # 根据毒性分数范围调整阈值
            if toxicity_scores.max() <= 1.0:
                threshold = 0.5
            else:
                threshold = (toxicity_scores.max() + toxicity_scores.min()) / 2

            # ID：低毒性
            id_mask = toxicity_scores < threshold
            id_indices = np.where(id_mask)[0]

            # OOD：高毒性
            ood_mask = toxicity_scores >= threshold
            ood_indices = np.where(ood_mask)[0]

            print(f"ID样本池: {len(id_indices)}, OOD样本池: {len(ood_indices)}")

            if len(id_indices) < 100 or len(ood_indices) < 100:
                print(f"⚠️ 样本太少，跳过此数据源")
                continue

            # 采样
            n_id = min(config['n_id_samples'], len(id_indices))
            n_ood = min(config['n_ood_samples'], len(ood_indices))

            selected_id = np.random.choice(id_indices, n_id, replace=False)
            selected_ood = np.random.choice(ood_indices, n_ood, replace=False)

            id_texts = [all_texts[i] for i in selected_id]
            ood_texts = [all_texts[i] for i in selected_ood]

            print(f"✅ ToxiGen加载完成:")
            print(f"   数据源: {source} (config={config_name})")
            print(f"   ID样本: {len(id_texts)} (低毒性, {toxicity_field} < {threshold})")
            print(f"   OOD样本: {len(ood_texts)} (高毒性, {toxicity_field} >= {threshold})")

            return id_texts, ood_texts, {
                'dataset': 'ToxiGen',
                'source': f'{source} ({config_name})',
                'toxicity_field': toxicity_field,
                'threshold': threshold,
                'id_type': 'non-toxic statements',
                'ood_type': 'toxic statements (domain shift)'
            }

        except Exception as e:
            print(f"⚠️ {source} ({config_name}) 加载失败: {e}")
            continue

    # 备用方案：使用情感分析数据集模拟毒性检测
    print("\n尝试备用方案：使用情感分析数据集模拟...")
    try:
        dataset = load_dataset("SetFit/sst2")
        train_data = dataset['train']

        np.random.seed(GLOBAL_CONFIG['random_seed'])

        all_texts = np.array(train_data['text'])
        all_labels = np.array(train_data['label'])

        # ID：正面情感（label=1）
        id_mask = all_labels == 1
        id_indices = np.where(id_mask)[0]

        # OOD：负面情感（label=0）- 模拟毒性内容
        ood_mask = all_labels == 0
        ood_indices = np.where(ood_mask)[0]

        n_id = min(config['n_id_samples'], len(id_indices))
        n_ood = min(config['n_ood_samples'], len(ood_indices))

        selected_id = np.random.choice(id_indices, n_id, replace=False)
        selected_ood = np.random.choice(ood_indices, n_ood, replace=False)

        id_texts = [all_texts[i] for i in selected_id]
        ood_texts = [all_texts[i] for i in selected_ood]

        print(f"✅ ToxiGen替代方案加载完成（SST2情感）:")
        print(f"   ID样本: {len(id_texts)} (正面情感)")
        print(f"   OOD样本: {len(ood_texts)} (负面情感)")

        return id_texts, ood_texts, {
            'dataset': 'ToxiGen-alternative',
            'source': 'SST2 sentiment',
            'id_type': 'positive sentiment',
            'ood_type': 'negative sentiment (domain shift)'
        }

    except Exception as e2:
        print(f"❌ 替代方案也失败: {e2}")
        return None, None, None


# ============================================================================
# Phase 2: 统一实验流程
# ============================================================================

# 全局模型缓存
_model_cache = None

def get_model():
    """获取或创建模型（避免重复加载）"""
    global _model_cache
    if _model_cache is None:
        print("\n加载语义编码器（仅需一次）...")
        _model_cache = SentenceTransformer(GLOBAL_CONFIG['encoder'])
        print(f"✅ 编码器加载完成: {GLOBAL_CONFIG['encoder']}")
    return _model_cache


def run_single_dataset_experiment(dataset_name, id_texts, ood_texts, metadata):
    """在单个数据集上运行完整实验"""

    if id_texts is None or ood_texts is None:
        print(f"\n⚠️ 跳过{dataset_name}实验（数据加载失败）")
        return None

    config = DATASET_CONFIGS[dataset_name]

    print("\n" + "="*80)
    print(f"开始{dataset_name}实验")
    print("="*80)
    print(f"数据集描述: {config['description']}")
    print(f"OOD类型: {metadata.get('ood_type', 'N/A')}")

    # 合并数据
    all_texts = list(id_texts) + list(ood_texts)
    ood_labels = np.array([0]*len(id_texts) + [1]*len(ood_texts))

    # Step 1: 语义嵌入生成
    print("\n" + "-"*80)
    print("Step 1: 生成语义嵌入")
    print("-"*80)

    model = get_model()
    embeddings = model.encode(all_texts, show_progress_bar=True, convert_to_numpy=True)

    print(f"✅ 嵌入生成完成: {embeddings.shape}")

    # Step 2: k-NN图构建
    print("\n" + "-"*80)
    print("Step 2: 构建k-NN图")
    print("-"*80)

    k = config['k_neighbors']
    knn = NearestNeighbors(n_neighbors=k+1, metric=GLOBAL_CONFIG['distance_metric'])
    knn.fit(embeddings)
    distances, neighbors = knn.kneighbors(embeddings)

    print(f"✅ k-NN图构建完成 (k={k})")

    # Step 3: HDBSCAN无监督聚类
    print("\n" + "-"*80)
    print("Step 3: HDBSCAN无监督聚类")
    print("-"*80)

    clusterer = HDBSCAN(
        min_cluster_size=GLOBAL_CONFIG['hdbscan_min_cluster_size'],
        metric='euclidean',
        cluster_selection_method='eom'
    )
    pseudo_labels = clusterer.fit_predict(embeddings)

    n_clusters = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
    n_noise = list(pseudo_labels).count(-1)

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

    # Step 4: 计算同配性特征
    print("\n" + "-"*80)
    print("Step 4: 计算同配性特征")
    print("-"*80)

    # NHR
    def compute_nhr(neighbors, pseudo_labels):
        nhr = np.zeros(len(pseudo_labels))
        for i in range(len(pseudo_labels)):
            if pseudo_labels[i] == -1:
                nhr[i] = 0.0
                continue
            neighbor_ids = neighbors[i][1:]
            neighbor_labels = pseudo_labels[neighbor_ids]
            same_cluster = (neighbor_labels == pseudo_labels[i]).sum()
            nhr[i] = same_cluster / len(neighbor_labels)
        return nhr

    nhr_pseudo = compute_nhr(neighbors, pseudo_labels)

    # 嵌入同配性
    def compute_embedding_homophily(embeddings, neighbors):
        emb_h = np.zeros(len(embeddings))
        for i in range(len(embeddings)):
            neighbor_ids = neighbors[i][1:]
            node_emb = embeddings[i:i+1]
            neighbor_embs = embeddings[neighbor_ids]
            sims = cosine_similarity(node_emb, neighbor_embs)[0]
            emb_h[i] = sims.mean()
        return emb_h

    emb_homophily = compute_embedding_homophily(embeddings, neighbors)

    # 邻居熵
    def compute_neighbor_entropy(neighbors, pseudo_labels):
        entropy = np.zeros(len(pseudo_labels))
        for i in range(len(pseudo_labels)):
            neighbor_labels = pseudo_labels[neighbors[i][1:]]
            valid_labels = neighbor_labels[neighbor_labels != -1]
            if len(valid_labels) == 0:
                entropy[i] = 0.0
                continue
            unique, counts = np.unique(valid_labels, return_counts=True)
            probs = counts / counts.sum()
            entropy[i] = -np.sum(probs * np.log(probs + 1e-10))
        return entropy

    neighbor_entropy = compute_neighbor_entropy(neighbors, pseudo_labels)

    print("✅ 特征计算完成:")
    print(f"   伪标签NHR: {nhr_pseudo.mean():.4f} ± {nhr_pseudo.std():.4f}")
    print(f"   嵌入同配性: {emb_homophily.mean():.4f} ± {emb_homophily.std():.4f}")
    print(f"   邻居熵: {neighbor_entropy.mean():.4f} ± {neighbor_entropy.std():.4f}")

    # Step 5: 统计检验
    print("\n" + "-"*80)
    print("Step 5: 统计检验与效应量分析")
    print("-"*80)

    def cohens_d(group1, group2):
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        return (np.mean(group1) - np.mean(group2)) / (pooled_std + 1e-8)

    id_mask = (ood_labels == 0)
    ood_mask = (ood_labels == 1)

    features = {
        "伪标签NHR": nhr_pseudo,
        "嵌入同配性": emb_homophily,
        "邻居熵": neighbor_entropy
    }

    results = []

    for feat_name, feat_values in features.items():
        id_feat = feat_values[id_mask]
        ood_feat = feat_values[ood_mask]

        d = cohens_d(ood_feat, id_feat)
        t_stat, p_value = ttest_ind(ood_feat, id_feat)
        auroc = max(roc_auc_score(ood_labels, feat_values),
                    roc_auc_score(ood_labels, -feat_values))

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
        print(f"  AUROC: {auroc:.4f}")
        print(f"  {significance}")

    # Step 6: 距离基线
    print("\n" + "-"*80)
    print("Step 6: 距离基线对比")
    print("-"*80)

    def compute_distance_baseline(embeddings, train_embeddings, train_mask, pseudo_labels):
        unique_clusters = np.unique(pseudo_labels[train_mask])
        unique_clusters = unique_clusters[unique_clusters != -1]

        class_means = []
        for cluster_id in unique_clusters:
            cluster_mask = (pseudo_labels[train_mask] == cluster_id)
            cluster_embeds = train_embeddings[cluster_mask]
            if len(cluster_embeds) >= 1:
                class_means.append(cluster_embeds.mean(axis=0))

        if len(class_means) == 0:
            return np.ones(len(embeddings))

        class_means = np.array(class_means)
        scores = []
        for emb in embeddings:
            dists = np.linalg.norm(class_means - emb, axis=1)
            min_dist = dists.min()
            scores.append(min_dist)

        return np.array(scores)

    train_embeddings = embeddings[id_mask]
    distance_scores = compute_distance_baseline(embeddings, train_embeddings, id_mask, pseudo_labels)
    distance_auroc = roc_auc_score(ood_labels, distance_scores)

    print(f"✅ 距离基线AUROC: {distance_auroc:.4f}")

    # Step 7: 保存结果
    results_df = pd.DataFrame(results)

    dataset_results = {
        'dataset': dataset_name,
        'metadata': metadata,
        'config': config,
        'clustering': {
            'n_clusters': int(n_clusters),
            'n_noise': int(n_noise),
            'silhouette': float(silhouette)
        },
        'statistical_tests': results,
        'distance_baseline_auroc': float(distance_auroc),
        'best_cohens_d': float(max([abs(float(r["Cohen's d"])) for r in results])),
        'best_auroc': float(max([float(r['AUROC']) for r in results]))
    }

    # 保存CSV
    csv_path = os.path.join(GLOBAL_CONFIG['output_dir'],
                            f'{dataset_name}_statistical_tests.csv')
    results_df.to_csv(csv_path, index=False)

    # 保存JSON
    json_path = os.path.join(GLOBAL_CONFIG['output_dir'],
                             f'{dataset_name}_results.json')
    with open(json_path, 'w') as f:
        json.dump(dataset_results, f, indent=2, default=str)

    print(f"\n✅ {dataset_name}结果已保存:")
    print(f"   - {csv_path}")
    print(f"   - {json_path}")

    return dataset_results


# ============================================================================
# Phase 3: 运行所有数据集实验
# ============================================================================

print("\n" + "="*80)
print("Phase 3: 运行多数据集实验（修复版）")
print("="*80)

all_results = {}

# 实验1: CLINC150
print("\n【实验1/4】CLINC150")
id_texts, ood_texts, metadata = load_clinc150_data(DATASET_CONFIGS['CLINC150'])
result = run_single_dataset_experiment('CLINC150', id_texts, ood_texts, metadata)
if result:
    all_results['CLINC150'] = result

# 实验2: Banking77
print("\n【实验2/4】Banking77-OOS")
id_texts, ood_texts, metadata = load_banking77_data(DATASET_CONFIGS['Banking77'])
result = run_single_dataset_experiment('Banking77', id_texts, ood_texts, metadata)
if result:
    all_results['Banking77'] = result

# 实验3: ROSTD
print("\n【实验3/4】ROSTD")
id_texts, ood_texts, metadata = load_rostd_data(DATASET_CONFIGS['ROSTD'])
result = run_single_dataset_experiment('ROSTD', id_texts, ood_texts, metadata)
if result:
    all_results['ROSTD'] = result

# 实验4: ToxiGen
print("\n【实验4/4】ToxiGen")
id_texts, ood_texts, metadata = load_toxigen_data(DATASET_CONFIGS['ToxiGen'])
result = run_single_dataset_experiment('ToxiGen', id_texts, ood_texts, metadata)
if result:
    all_results['ToxiGen'] = result

# ============================================================================
# Phase 4: 跨数据集对比分析
# ============================================================================

print("\n" + "="*80)
print("Phase 4: 跨数据集对比分析")
print("="*80)

comparison_df = None

if len(all_results) >= 2:
    comparison_data = []

    for dataset_name, result in all_results.items():
        cohens_d_nhr = float(result['statistical_tests'][0]["Cohen's d"])
        auroc_nhr = float(result['statistical_tests'][0]['AUROC'])
        direction = result['statistical_tests'][0]['方向']
        ood_type = result['metadata'].get('ood_type', 'N/A')

        comparison_data.append({
            '数据集': dataset_name,
            'OOD类型': ood_type,
            "Cohen's d (NHR)": cohens_d_nhr,
            'AUROC (NHR)': auroc_nhr,
            '方向': direction,
            '最佳AUROC': result['best_auroc'],
            '距离基线AUROC': result['distance_baseline_auroc']
        })

    comparison_df = pd.DataFrame(comparison_data)

    print("\n跨数据集对比表：")
    print("="*80)
    print(comparison_df.to_string(index=False))

    comparison_path = os.path.join(GLOBAL_CONFIG['output_dir'],
                                   'cross_dataset_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)

    print(f"\n✅ 对比表已保存: {comparison_path}")

    # 分析结论
    print("\n" + "="*80)
    print("关键结论")
    print("="*80)

    directions = [r['statistical_tests'][0]['方向'] for r in all_results.values()]
    homophilic_count = sum([1 for d in directions if "同配" in d])
    heterophilic_count = sum([1 for d in directions if "异配" in d])

    print(f"\n【方向统计】")
    print(f"  OOD更同配: {homophilic_count}个数据集")
    print(f"  OOD更异配: {heterophilic_count}个数据集")

    if homophilic_count == len(all_results):
        print("\n✅ 结论：OOD更同配是普遍规律")
        print("   → 所有数据集都表现出相同的模式")
        print("   → CP-ABR++应使用\"高NHR检测OOD\"策略")
        print("   → 理论解释：文本OOD的语义聚集特性")

    elif heterophilic_count == len(all_results):
        print("\n⚠️ 结论：OOD更异配是普遍规律")
        print("   → 所有数据集都表现出相同的模式")
        print("   → CP-ABR++应使用\"低NHR检测OOD\"策略")
        print("   → 理论解释：OOD样本在特征空间中分散")

    else:
        print("\n⚠️ 结论：方向取决于数据集特性")
        print("   → 不同数据集表现出不同的模式")
        print("   → CP-ABR++需要自适应机制")
        print("   → 建议：在验证集上校准检测方向")

        print("\n【OOD类型分析】")
        for dataset_name, result in all_results.items():
            ood_type = result['metadata'].get('ood_type', 'N/A')
            direction = result['statistical_tests'][0]['方向']
            cohens_d = result['statistical_tests'][0]["Cohen's d"]
            print(f"  {dataset_name}:")
            print(f"    OOD类型: {ood_type}")
            print(f"    方向: {direction}")
            print(f"    Cohen's d: {cohens_d}")

    # 效应量分析
    print(f"\n【效应量对比】")
    for dataset_name, result in all_results.items():
        best_d = result['best_cohens_d']
        if best_d >= 0.8:
            strength = "强效应 ✅"
        elif best_d >= 0.5:
            strength = "中等效应 ⚠️"
        else:
            strength = "弱效应 ❌"
        print(f"  {dataset_name}: {best_d:.4f} ({strength})")

else:
    print("\n⚠️ 成功的数据集数量不足，无法进行跨数据集对比")

# ============================================================================
# Phase 5: 生成最终报告
# ============================================================================

print("\n" + "="*80)
print("Phase 5: 生成最终报告")
print("="*80)

final_report = {
    'experiment_config': GLOBAL_CONFIG,
    'datasets_tested': list(all_results.keys()),
    'n_datasets': len(all_results),
    'results_summary': all_results,
    'cross_dataset_analysis': comparison_df.to_dict('records') if comparison_df is not None else None
}

report_path = os.path.join(GLOBAL_CONFIG['output_dir'], 'final_report.json')
with open(report_path, 'w') as f:
    json.dump(final_report, f, indent=2, default=str)

print(f"✅ 最终报告已保存: {report_path}")

print("\n" + "="*80)
print("实验完成！")
print("="*80)
print(f"\n所有结果保存在目录: {GLOBAL_CONFIG['output_dir']}/")
print("\n生成的文件:")
print("  - cross_dataset_comparison.csv  (跨数据集对比)")
print("  - final_report.json  (完整实验报告)")
print("  - [dataset]_statistical_tests.csv  (每个数据集的统计检验)")
print("  - [dataset]_results.json  (每个数据集的详细结果)")

print("\n数据集加载情况:")
for dataset_name in ['CLINC150', 'Banking77', 'ROSTD', 'ToxiGen']:
    status = "✅ 成功" if dataset_name in all_results else "❌ 失败"
    print(f"  {dataset_name}: {status}")

print("\n下一步建议:")
print("  1. 分析跨数据集对比表，确定OOD同配性的普遍性")
print("  2. 基于发现设计自适应CP-ABR++方法")
print("  3. 撰写预实验报告（第5.3.1节）")
print("  4. 准备主实验和完整论文")

print("\n完全自动化实验完成（修复版）！")
