#!/usr/bin/env python3
"""
RW3预实验修正版 - 真实数据集验证
=================================
使用已下载的真实数据集验证OOD拓扑模式假设

数据集:
1. CLINC150: clinc_oos (plus) - Out-of-scope OOD
2. Banking77: mteb/banking77 - Held-out class OOD
3. ROSTD: 真实OODrelease.tsv - Diverse human-authored OOD
4. ToxiGen: toxigen/toxigen-data (annotated) - Toxicity-based OOD

核心假设: OOD拓扑模式取决于OOD类型
- Out-of-scope → MORE HOMOPHILIC
- Held-out class → MORE HETEROPHILIC
- Domain shift → WEAK/VARIABLE
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter
from pathlib import Path

warnings.filterwarnings('ignore')

# 配置
DATASET_CONFIGS = {
    'CLINC150': {
        'k_neighbors': 15,
        'n_id_samples': 1600,
        'n_ood_samples': 150,
        'ood_type': 'out-of-scope',
        'description': 'Out-of-scope queries (semantically unrelated)'
    },
    'Banking77': {
        'k_neighbors': 20,
        'n_id_samples': 2000,
        'n_ood_samples': 200,
        'n_id_classes': 50,  # 50 classes as ID, 27 as OOD
        'ood_type': 'held-out-class',
        'description': 'Held-out intent classes'
    },
    'ROSTD': {
        'k_neighbors': 20,
        'n_id_samples': 1000,
        'n_ood_samples': 500,
        'ood_type': 'diverse-ood',
        'description': 'Real diverse human-authored OOD from ROSTD'
    },
    'ToxiGen': {
        'k_neighbors': 20,
        'n_id_samples': 1000,
        'n_ood_samples': 500,
        'toxicity_threshold': 2.5,  # toxicity_ai >= 2.5 is OOD (scale 1-5)
        'ood_type': 'toxicity-based',
        'description': 'Toxicity-based OOD (toxic vs non-toxic)'
    }
}

RESULTS_DIR = 'rw3_real_data_results'
EMBEDDING_MODEL = 'all-mpnet-base-v2'


def setup_environment():
    """设置环境"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 80)
    print("RW3预实验修正版 - 真实数据集验证")
    print("=" * 80)
    print(f"结果目录: {RESULTS_DIR}")
    print(f"嵌入模型: {EMBEDDING_MODEL}")
    print(f"数据集: {list(DATASET_CONFIGS.keys())}")
    print("=" * 80)


def load_clinc150():
    """加载CLINC150数据集"""
    print("\n[CLINC150] 加载数据集...")
    from datasets import load_dataset

    dataset = load_dataset("clinc_oos", "plus")

    # 合并训练集和验证集 (转换为list)
    train_texts = list(dataset['train']['text'])
    train_labels = list(dataset['train']['intent'])
    val_texts = list(dataset['validation']['text'])
    val_labels = list(dataset['validation']['intent'])

    all_texts = train_texts + val_texts
    all_labels = train_labels + val_labels

    # 分离ID和OOD (label 42 = oos)
    id_indices = [i for i, l in enumerate(all_labels) if l != 42]
    ood_indices = [i for i, l in enumerate(all_labels) if l == 42]

    id_texts = [all_texts[i] for i in id_indices]
    id_labels = [all_labels[i] for i in id_indices]
    ood_texts = [all_texts[i] for i in ood_indices]

    print(f"   ID样本: {len(id_texts)}, OOD样本: {len(ood_texts)}")
    print(f"   ID类别数: {len(set(id_labels))}")

    return {
        'id_texts': id_texts,
        'id_labels': id_labels,
        'ood_texts': ood_texts,
        'source': 'clinc_oos (plus)',
        'ood_type': 'out-of-scope'
    }


def load_banking77():
    """加载Banking77数据集 - 50类ID, 27类OOD"""
    print("\n[Banking77] 加载数据集...")
    from datasets import load_dataset

    dataset = load_dataset("mteb/banking77")

    # 合并训练和测试 (转换为list)
    train_texts = list(dataset['train']['text'])
    train_labels = list(dataset['train']['label'])
    test_texts = list(dataset['test']['text'])
    test_labels = list(dataset['test']['label'])

    all_texts = train_texts + test_texts
    all_labels = train_labels + test_labels

    # 获取所有类别
    unique_labels = sorted(set(all_labels))
    print(f"   总类别数: {len(unique_labels)}")

    # 随机选择50个类作为ID
    np.random.seed(42)
    id_classes = set(np.random.choice(unique_labels, size=50, replace=False))
    ood_classes = set(unique_labels) - id_classes

    print(f"   ID类别: {len(id_classes)}, OOD类别: {len(ood_classes)}")

    # 分离数据
    id_texts = [t for t, l in zip(all_texts, all_labels) if l in id_classes]
    id_labels = [l for l in all_labels if l in id_classes]
    ood_texts = [t for t, l in zip(all_texts, all_labels) if l in ood_classes]

    print(f"   ID样本: {len(id_texts)}, OOD样本: {len(ood_texts)}")

    return {
        'id_texts': id_texts,
        'id_labels': id_labels,
        'ood_texts': ood_texts,
        'source': 'mteb/banking77',
        'ood_type': 'held-out-class',
        'id_classes': list(id_classes),
        'ood_classes': list(ood_classes)
    }


def load_rostd():
    """加载ROSTD数据集 - 真实OOD + Banking77子集作为ID"""
    print("\n[ROSTD] 加载数据集...")

    # 1. 加载真实ROSTD OOD数据
    ood_file = Path('dataset_downloads/LR_GC_OOD_data/LR_GC_OOD-master/data/fbrelease/OODrelease.tsv')

    if not ood_file.exists():
        raise FileNotFoundError(f"ROSTD OOD文件不存在: {ood_file}")

    ood_texts = []
    with open(ood_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                ood_texts.append(parts[2])  # 第3列是文本

    print(f"   真实ROSTD OOD样本: {len(ood_texts)}")
    print(f"   示例: {ood_texts[0][:50]}...")

    # 2. 使用Banking77子集作为ID（因为原始fb.me数据不可用）
    from datasets import load_dataset
    dataset = load_dataset("mteb/banking77")

    id_texts = dataset['train']['text'][:2000]  # 取前2000个作为ID
    id_labels = dataset['train']['label'][:2000]

    print(f"   ID样本(Banking77子集): {len(id_texts)}")

    return {
        'id_texts': list(id_texts),
        'id_labels': list(id_labels),
        'ood_texts': ood_texts,
        'source': 'OODrelease.tsv + mteb/banking77 (as ID)',
        'ood_type': 'diverse-ood',
        'ood_source': str(ood_file)
    }


def load_toxigen():
    """加载ToxiGen数据集 - 基于toxicity_ai分类"""
    print("\n[ToxiGen] 加载数据集...")
    from datasets import load_dataset

    try:
        # 尝试加载（可能使用缓存）
        dataset = load_dataset("toxigen/toxigen-data", "annotated")
    except Exception as e:
        print(f"   警告: 加载失败 ({e}), 尝试使用缓存...")
        # 尝试使用缓存
        from datasets import load_from_disk
        cache_path = Path.home() / '.cache/huggingface/datasets/toxigen___toxigen-data'
        if cache_path.exists():
            dataset = load_dataset("toxigen/toxigen-data", "annotated")
        else:
            raise

    # 使用训练集 (转换为list)
    train_data = dataset['train']

    texts = list(train_data['text'])
    toxicity_scores = list(train_data['toxicity_ai'])
    target_groups = list(train_data['target_group'])

    # 过滤无效值
    valid_data = [(t, s, g) for t, s, g in zip(texts, toxicity_scores, target_groups) if s is not None]
    texts, toxicity_scores, target_groups = zip(*valid_data)

    threshold = DATASET_CONFIGS['ToxiGen']['toxicity_threshold']

    # 分类: toxicity_ai < threshold 为 ID (非毒性), >= threshold 为 OOD (毒性)
    id_texts = [t for t, s in zip(texts, toxicity_scores) if s < threshold]
    ood_texts = [t for t, s in zip(texts, toxicity_scores) if s >= threshold]

    # 为ID创建伪标签（基于target_group）
    id_labels = []
    group_to_id = {}
    for t, s, g in zip(texts, toxicity_scores, target_groups):
        if s < threshold:
            if g not in group_to_id:
                group_to_id[g] = len(group_to_id)
            id_labels.append(group_to_id[g])

    print(f"   阈值: toxicity_ai >= {threshold}")
    print(f"   ID样本(非毒性): {len(id_texts)}, OOD样本(毒性): {len(ood_texts)}")
    print(f"   ID目标群体数: {len(group_to_id)}")

    return {
        'id_texts': id_texts,
        'id_labels': id_labels,
        'ood_texts': ood_texts,
        'source': 'toxigen/toxigen-data (annotated)',
        'ood_type': 'toxicity-based',
        'threshold': threshold
    }


def generate_embeddings(texts, model_name=EMBEDDING_MODEL):
    """生成语义嵌入"""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings


def build_knn_graph(embeddings, k=15):
    """构建k-NN图"""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k+1, metric='cosine')
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    # 排除自身
    return indices[:, 1:], distances[:, 1:]


def cluster_with_hdbscan(embeddings, min_cluster_size=15):
    """HDBSCAN无监督聚类"""
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean',
        cluster_selection_method='eom'
    )

    cluster_labels = clusterer.fit_predict(embeddings)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = (cluster_labels == -1).sum()

    return cluster_labels, n_clusters, n_noise


def compute_homophily_features(embeddings, knn_indices, knn_distances, cluster_labels, is_ood):
    """计算同配性特征"""
    n_samples = len(embeddings)
    k = knn_indices.shape[1]

    features = {
        'pseudo_label_nhr': [],
        'embedding_homophily': [],
        'neighbor_entropy': [],
        'is_ood': is_ood
    }

    for i in range(n_samples):
        neighbors = knn_indices[i]
        my_label = cluster_labels[i]
        neighbor_labels = cluster_labels[neighbors]

        # 1. Pseudo-label NHR
        if my_label == -1:
            nhr = 0.0
        else:
            same_label = (neighbor_labels == my_label).sum()
            nhr = same_label / k
        features['pseudo_label_nhr'].append(nhr)

        # 2. Embedding homophily (1 - mean_distance)
        mean_dist = knn_distances[i].mean()
        emb_homophily = 1 - mean_dist
        features['embedding_homophily'].append(emb_homophily)

        # 3. Neighbor entropy
        label_counts = Counter(neighbor_labels)
        probs = np.array(list(label_counts.values())) / k
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        features['neighbor_entropy'].append(entropy)

    return features


def statistical_tests(id_values, ood_values, feature_name):
    """统计检验"""
    from scipy import stats
    from sklearn.metrics import roc_auc_score

    id_mean = np.mean(id_values)
    ood_mean = np.mean(ood_values)
    id_std = np.std(id_values)
    ood_std = np.std(ood_values)

    # Cohen's d
    pooled_std = np.sqrt((id_std**2 + ood_std**2) / 2)
    cohens_d = (ood_mean - id_mean) / (pooled_std + 1e-10)

    # t-test
    t_stat, p_value = stats.ttest_ind(id_values, ood_values)

    # AUROC
    labels = np.concatenate([np.zeros(len(id_values)), np.ones(len(ood_values))])
    scores = np.concatenate([id_values, ood_values])

    auroc = roc_auc_score(labels, scores)
    if auroc < 0.5:
        auroc = 1 - auroc

    return {
        'feature': feature_name,
        'id_mean': id_mean,
        'id_std': id_std,
        'ood_mean': ood_mean,
        'ood_std': ood_std,
        'cohens_d': cohens_d,
        't_statistic': t_stat,
        'p_value': p_value,
        'auroc': auroc
    }


def compute_distance_baseline(embeddings, is_ood, id_labels_only):
    """计算距离基线

    Args:
        embeddings: 所有样本的嵌入 (n_total, dim)
        is_ood: 布尔列表，标识每个样本是否为OOD (长度 n_total)
        id_labels_only: 仅ID样本的标签列表 (长度 n_id)
    """
    from sklearn.metrics import roc_auc_score

    id_mask = ~np.array(is_ood)
    id_embeddings = embeddings[id_mask]
    id_labels_arr = np.array(id_labels_only)  # 这个长度应该等于id_embeddings的长度

    # 计算每个类的中心
    unique_labels = np.unique(id_labels_arr)
    class_centers = {}
    for label in unique_labels:
        if label == -1:
            continue
        mask = id_labels_arr == label
        class_centers[label] = id_embeddings[mask].mean(axis=0)

    if not class_centers:
        return {'auroc': 0.5, 'method': 'distance_to_nearest_class'}

    # 计算所有样本到最近类中心的距离
    all_centers = np.array(list(class_centers.values()))

    distances = []
    for emb in embeddings:
        dists = np.linalg.norm(all_centers - emb, axis=1)
        distances.append(dists.min())

    distances = np.array(distances)
    labels = np.array(is_ood).astype(int)

    auroc = roc_auc_score(labels, distances)

    return {
        'auroc': auroc,
        'method': 'distance_to_nearest_class',
        'n_classes': len(class_centers)
    }


def run_experiment_for_dataset(dataset_name, data, config):
    """运行单个数据集的实验"""
    print(f"\n{'='*80}")
    print(f"实验: {dataset_name}")
    print(f"{'='*80}")

    # 采样
    np.random.seed(42)

    n_id = min(config['n_id_samples'], len(data['id_texts']))
    n_ood = min(config['n_ood_samples'], len(data['ood_texts']))

    id_indices = np.random.choice(len(data['id_texts']), n_id, replace=False)
    ood_indices = np.random.choice(len(data['ood_texts']), n_ood, replace=False)

    id_texts = [data['id_texts'][i] for i in id_indices]
    ood_texts = [data['ood_texts'][i] for i in ood_indices]

    if 'id_labels' in data and data['id_labels']:
        id_labels = [data['id_labels'][i] for i in id_indices]
    else:
        id_labels = list(range(len(id_texts)))

    print(f"采样: ID={n_id}, OOD={n_ood}")

    # 合并文本
    all_texts = id_texts + ood_texts
    is_ood = [False] * len(id_texts) + [True] * len(ood_texts)

    # 生成嵌入
    print("生成嵌入...")
    embeddings = generate_embeddings(all_texts)
    print(f"嵌入维度: {embeddings.shape}")

    # 构建k-NN图
    k = config['k_neighbors']
    print(f"构建k-NN图 (k={k})...")
    knn_indices, knn_distances = build_knn_graph(embeddings, k)

    # HDBSCAN聚类
    print("HDBSCAN聚类...")
    cluster_labels, n_clusters, n_noise = cluster_with_hdbscan(embeddings)
    print(f"聚类数: {n_clusters}, 噪声点: {n_noise}")

    # 计算同配性特征
    print("计算同配性特征...")
    features = compute_homophily_features(
        embeddings, knn_indices, knn_distances, cluster_labels, is_ood
    )

    # 统计检验
    print("统计检验...")
    results = []

    id_mask = ~np.array(is_ood)
    ood_mask = np.array(is_ood)

    for feature_name in ['pseudo_label_nhr', 'embedding_homophily', 'neighbor_entropy']:
        values = np.array(features[feature_name])
        id_values = values[id_mask]
        ood_values = values[ood_mask]

        result = statistical_tests(id_values, ood_values, feature_name)
        result['dataset'] = dataset_name
        results.append(result)

        print(f"  {feature_name}:")
        print(f"    ID: {result['id_mean']:.4f} ± {result['id_std']:.4f}")
        print(f"    OOD: {result['ood_mean']:.4f} ± {result['ood_std']:.4f}")
        print(f"    Cohen's d: {result['cohens_d']:.4f}")
        print(f"    AUROC: {result['auroc']:.4f}")

    # 距离基线 (只传入采样后的ID标签)
    print("计算距离基线...")
    sampled_id_labels = [id_labels[i] for i in range(len(id_texts))]
    baseline = compute_distance_baseline(embeddings, is_ood, sampled_id_labels)
    print(f"  距离基线 AUROC: {baseline['auroc']:.4f}")

    # 主要指标（NHR的Cohen's d）
    main_result = results[0]  # pseudo_label_nhr

    # 判断方向
    if main_result['cohens_d'] > 0.2:
        direction = "OOD MORE HOMOPHILIC"
    elif main_result['cohens_d'] < -0.2:
        direction = "OOD MORE HETEROPHILIC"
    else:
        direction = "WEAK/NO EFFECT"

    print(f"\n主要发现: {direction} (d={main_result['cohens_d']:.4f})")

    return {
        'dataset': dataset_name,
        'config': config,
        'data_info': {
            'source': data['source'],
            'ood_type': data['ood_type'],
            'n_id_actual': n_id,
            'n_ood_actual': n_ood
        },
        'clustering': {
            'n_clusters': n_clusters,
            'n_noise': n_noise
        },
        'statistical_results': results,
        'baseline': baseline,
        'main_finding': {
            'cohens_d': main_result['cohens_d'],
            'auroc': main_result['auroc'],
            'direction': direction
        }
    }


def main():
    """主函数"""
    setup_environment()

    all_results = {}
    dataset_loaders = {
        'CLINC150': load_clinc150,
        'Banking77': load_banking77,
        'ROSTD': load_rostd,
        'ToxiGen': load_toxigen
    }

    # 运行每个数据集
    for dataset_name, loader in dataset_loaders.items():
        try:
            print(f"\n{'#'*80}")
            print(f"# 加载 {dataset_name}")
            print(f"{'#'*80}")

            data = loader()
            config = DATASET_CONFIGS[dataset_name]

            result = run_experiment_for_dataset(dataset_name, data, config)
            all_results[dataset_name] = result

            # 保存单个数据集结果
            result_file = os.path.join(RESULTS_DIR, f'{dataset_name.lower()}_results.json')
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            print(f"\n结果已保存: {result_file}")

        except Exception as e:
            print(f"\n❌ {dataset_name} 失败: {e}")
            import traceback
            traceback.print_exc()
            all_results[dataset_name] = {'error': str(e)}

    # 生成跨数据集对比
    print(f"\n{'='*80}")
    print("跨数据集对比")
    print("="*80)

    comparison_data = []
    for name, result in all_results.items():
        if 'error' not in result:
            comparison_data.append({
                'Dataset': name,
                'OOD_Type': result['data_info']['ood_type'],
                'Source': result['data_info']['source'],
                'Cohen_d': result['main_finding']['cohens_d'],
                'AUROC': result['main_finding']['auroc'],
                'Direction': result['main_finding']['direction'],
                'Baseline_AUROC': result['baseline']['auroc']
            })

    if comparison_data:
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))

        # 保存CSV
        csv_file = os.path.join(RESULTS_DIR, 'cross_dataset_comparison.csv')
        df.to_csv(csv_file, index=False)
        print(f"\n对比表已保存: {csv_file}")

    # 与V2结果对比
    print(f"\n{'='*80}")
    print("与V2实验结果对比")
    print("="*80)

    v2_results = {
        'CLINC150': {'d': 2.0287, 'direction': 'HOMOPHILIC'},
        'Banking77': {'d': -0.4881, 'direction': 'HETEROPHILIC'},
        'ROSTD': {'d': -0.8861, 'direction': 'HETEROPHILIC'},
        'ToxiGen': {'d': -0.0262, 'direction': 'WEAK'}
    }

    print(f"{'Dataset':<12} {'V2 d':>10} {'Real d':>10} {'Diff':>10} {'V2 Dir':<15} {'Real Dir':<20}")
    print("-" * 80)

    for name in ['CLINC150', 'Banking77', 'ROSTD', 'ToxiGen']:
        if name in all_results and 'error' not in all_results[name]:
            real_d = all_results[name]['main_finding']['cohens_d']
            real_dir = all_results[name]['main_finding']['direction']
            v2_d = v2_results[name]['d']
            v2_dir = v2_results[name]['direction']
            diff = real_d - v2_d
            print(f"{name:<12} {v2_d:>10.4f} {real_d:>10.4f} {diff:>+10.4f} {v2_dir:<15} {real_dir:<20}")
        else:
            print(f"{name:<12} {'N/A':>10} {'ERROR':>10}")

    # 保存完整报告
    report = {
        'experiment': 'RW3 Real Data Verification',
        'timestamp': datetime.now().isoformat(),
        'embedding_model': EMBEDDING_MODEL,
        'results': all_results,
        'v2_comparison': v2_results,
        'summary': {
            'total_datasets': len(dataset_loaders),
            'successful': len([r for r in all_results.values() if 'error' not in r]),
            'failed': len([r for r in all_results.values() if 'error' in r])
        }
    }

    report_file = os.path.join(RESULTS_DIR, 'full_report.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n完整报告已保存: {report_file}")

    # 最终结论
    print(f"\n{'='*80}")
    print("实验结论")
    print("="*80)

    conclusions = []
    for name, result in all_results.items():
        if 'error' not in result:
            d = result['main_finding']['cohens_d']
            ood_type = result['data_info']['ood_type']
            direction = result['main_finding']['direction']
            conclusions.append(f"- {name} ({ood_type}): d={d:.4f} → {direction}")

    for c in conclusions:
        print(c)

    print(f"\n{'='*80}")
    print("实验完成!")
    print("="*80)

    return all_results


if __name__ == '__main__':
    main()
