#!/usr/bin/env python3
"""
RW3预实验 - 全数据集版本（不抽样）
==================================
解决聚类质量问题，获得更可靠的结果

关键改进：
1. 使用全部数据，不进行采样
2. 自适应调整HDBSCAN参数（根据数据集大小）
3. 增强聚类质量监控
4. 对比抽样版 vs 全数据版的差异
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
RESULTS_DIR = 'rw3_full_data_results'
EMBEDDING_MODEL = 'all-mpnet-base-v2'

# 之前的结果（用于对比）
PREVIOUS_RESULTS = {
    'V2': {
        'CLINC150': {'d': 2.0287, 'direction': 'HOMOPHILIC'},
        'Banking77': {'d': -0.4881, 'direction': 'HETEROPHILIC'},
        'ROSTD': {'d': -0.8861, 'direction': 'HETEROPHILIC'},
        'ToxiGen': {'d': -0.0262, 'direction': 'WEAK'}
    },
    'Sampled': {
        'CLINC150': {'d': -1.0103, 'direction': 'HETEROPHILIC'},
        'Banking77': {'d': -0.5207, 'direction': 'HETEROPHILIC'},
        'ROSTD': {'d': -6.1200, 'direction': 'HETEROPHILIC'},
        'ToxiGen': {'d': -0.1001, 'direction': 'WEAK'}
    }
}

DATASET_CONFIGS = {
    'CLINC150': {
        'k_neighbors': 15,
        'ood_type': 'out-of-scope',
        'description': 'Out-of-scope queries (all splits combined)'
    },
    'Banking77': {
        'k_neighbors': 20,
        'n_id_classes': 50,
        'ood_type': 'held-out-class',
        'description': 'Held-out intent classes (50 ID / 27 OOD)'
    },
    'ROSTD': {
        'k_neighbors': 20,
        'ood_type': 'diverse-ood',
        'description': 'Real diverse human-authored OOD'
    },
    'ToxiGen': {
        'k_neighbors': 20,
        'toxicity_threshold': 2.5,
        'ood_type': 'toxicity-based',
        'description': 'Toxicity-based OOD (all splits combined)'
    }
}


def setup_environment():
    """设置环境"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 80)
    print("RW3预实验 - 全数据集版本（不抽样）")
    print("=" * 80)
    print(f"结果目录: {RESULTS_DIR}")
    print(f"嵌入模型: {EMBEDDING_MODEL}")
    print(f"数据集: {list(DATASET_CONFIGS.keys())}")
    print("\n关键改进:")
    print("  1. 使用全部数据，不进行采样")
    print("  2. 自适应HDBSCAN参数")
    print("  3. 增强聚类质量监控")
    print("=" * 80)


def load_clinc150_full():
    """加载CLINC150全数据（合并所有split）"""
    print("\n[CLINC150] 加载全数据...")
    from datasets import load_dataset

    dataset = load_dataset("clinc_oos", "plus")

    # 合并所有split
    all_texts = []
    all_labels = []
    for split in ['train', 'validation', 'test']:
        all_texts.extend(list(dataset[split]['text']))
        all_labels.extend(list(dataset[split]['intent']))

    # 分离ID和OOD（不抽样）
    id_texts = [t for t, l in zip(all_texts, all_labels) if l != 42]
    id_labels = [l for l in all_labels if l != 42]
    ood_texts = [t for t, l in zip(all_texts, all_labels) if l == 42]

    n_id_classes = len(set(id_labels))

    print(f"   合并splits: train + validation + test")
    print(f"   ID样本: {len(id_texts)} ({n_id_classes}类)")
    print(f"   OOD样本: {len(ood_texts)} (out-of-scope)")
    print(f"   总样本: {len(all_texts)}")

    return {
        'id_texts': id_texts,
        'id_labels': id_labels,
        'ood_texts': ood_texts,
        'source': 'clinc_oos (plus, all splits)',
        'ood_type': 'out-of-scope',
        'n_id_classes': n_id_classes
    }


def load_banking77_full():
    """加载Banking77全数据 - 50类ID, 27类OOD"""
    print("\n[Banking77] 加载全数据...")
    from datasets import load_dataset

    dataset = load_dataset("mteb/banking77")

    # 合并训练和测试
    all_texts = list(dataset['train']['text']) + list(dataset['test']['text'])
    all_labels = list(dataset['train']['label']) + list(dataset['test']['label'])

    # 获取所有类别
    unique_labels = sorted(set(all_labels))
    print(f"   总类别数: {len(unique_labels)}")

    # 随机选择50个类作为ID（使用固定种子）
    np.random.seed(42)
    id_classes = set(np.random.choice(unique_labels, size=50, replace=False))
    ood_classes = set(unique_labels) - id_classes

    # 分离数据（不抽样）
    id_texts = [t for t, l in zip(all_texts, all_labels) if l in id_classes]
    id_labels = [l for l in all_labels if l in id_classes]
    ood_texts = [t for t, l in zip(all_texts, all_labels) if l in ood_classes]

    print(f"   ID类别: {len(id_classes)}, OOD类别: {len(ood_classes)}")
    print(f"   ID样本: {len(id_texts)}")
    print(f"   OOD样本: {len(ood_texts)}")

    return {
        'id_texts': id_texts,
        'id_labels': id_labels,
        'ood_texts': ood_texts,
        'source': 'mteb/banking77 (all data)',
        'ood_type': 'held-out-class',
        'id_classes': list(id_classes),
        'ood_classes': list(ood_classes)
    }


def load_rostd_full():
    """加载ROSTD全数据"""
    print("\n[ROSTD] 加载全数据...")

    # 1. 加载全部ROSTD OOD数据
    ood_file = Path('dataset_downloads/LR_GC_OOD_data/LR_GC_OOD-master/data/fbrelease/OODrelease.tsv')

    if not ood_file.exists():
        raise FileNotFoundError(f"ROSTD OOD文件不存在: {ood_file}")

    ood_texts = []
    with open(ood_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                ood_texts.append(parts[2])

    print(f"   真实ROSTD OOD: {len(ood_texts)}样本")

    # 2. 使用Banking77全部数据作为ID（因为原始fb.me数据不可用）
    from datasets import load_dataset
    dataset = load_dataset("mteb/banking77")

    id_texts = list(dataset['train']['text']) + list(dataset['test']['text'])
    id_labels = list(dataset['train']['label']) + list(dataset['test']['label'])

    print(f"   ID样本(Banking77全部): {len(id_texts)}")
    print(f"   总样本: {len(id_texts) + len(ood_texts)}")

    return {
        'id_texts': id_texts,
        'id_labels': id_labels,
        'ood_texts': ood_texts,
        'source': 'OODrelease.tsv (full) + mteb/banking77 (full, as ID proxy)',
        'ood_type': 'diverse-ood',
        'ood_source': str(ood_file)
    }


def load_toxigen_full():
    """加载ToxiGen全数据"""
    print("\n[ToxiGen] 加载全数据...")
    from datasets import load_dataset

    dataset = load_dataset("toxigen/toxigen-data", "annotated")

    # 合并train和test
    all_texts = []
    all_toxicity = []
    all_groups = []

    for split in ['train', 'test']:
        all_texts.extend(list(dataset[split]['text']))
        all_toxicity.extend(list(dataset[split]['toxicity_ai']))
        all_groups.extend(list(dataset[split]['target_group']))

    # 过滤无效值
    valid_data = [(t, s, g) for t, s, g in zip(all_texts, all_toxicity, all_groups) if s is not None]
    texts, toxicity_scores, target_groups = zip(*valid_data)

    threshold = DATASET_CONFIGS['ToxiGen']['toxicity_threshold']

    # 分类（不抽样）
    id_texts = [t for t, s in zip(texts, toxicity_scores) if s < threshold]
    ood_texts = [t for t, s in zip(texts, toxicity_scores) if s >= threshold]

    # 为ID创建伪标签
    id_labels = []
    group_to_id = {}
    for t, s, g in zip(texts, toxicity_scores, target_groups):
        if s < threshold:
            if g not in group_to_id:
                group_to_id[g] = len(group_to_id)
            id_labels.append(group_to_id[g])

    print(f"   合并splits: train + test")
    print(f"   阈值: toxicity_ai >= {threshold}")
    print(f"   ID样本(非毒性): {len(id_texts)} ({len(group_to_id)}组)")
    print(f"   OOD样本(毒性): {len(ood_texts)}")
    print(f"   总样本: {len(texts)}")

    return {
        'id_texts': list(id_texts),
        'id_labels': id_labels,
        'ood_texts': list(ood_texts),
        'source': 'toxigen/toxigen-data (annotated, all splits)',
        'ood_type': 'toxicity-based',
        'threshold': threshold,
        'n_groups': len(group_to_id)
    }


def generate_embeddings(texts, model_name=EMBEDDING_MODEL, batch_size=64):
    """生成语义嵌入（支持大数据集）"""
    from sentence_transformers import SentenceTransformer

    print(f"   生成嵌入 ({len(texts)}样本)...")
    model = SentenceTransformer(model_name)

    # 对于大数据集，分批处理
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=batch_size
    )

    return embeddings


def build_knn_graph(embeddings, k=15):
    """构建k-NN图"""
    from sklearn.neighbors import NearestNeighbors

    print(f"   构建k-NN图 (k={k})...")
    nn = NearestNeighbors(n_neighbors=k+1, metric='cosine')
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    return indices[:, 1:], distances[:, 1:]


def cluster_with_hdbscan_adaptive(embeddings, min_cluster_size=None):
    """自适应HDBSCAN聚类"""
    import hdbscan

    n_samples = len(embeddings)

    # 自适应参数
    if min_cluster_size is None:
        if n_samples < 1000:
            min_cluster_size = 5
        elif n_samples < 5000:
            min_cluster_size = 10
        elif n_samples < 10000:
            min_cluster_size = 15
        else:
            min_cluster_size = 20

    print(f"   HDBSCAN聚类: n_samples={n_samples}, min_cluster_size={min_cluster_size}")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean',
        cluster_selection_method='eom'
    )

    cluster_labels = clusterer.fit_predict(embeddings)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = (cluster_labels == -1).sum()
    noise_ratio = n_noise / n_samples

    print(f"   聚类结果: {n_clusters}簇, {n_noise}噪声点 ({noise_ratio*100:.1f}%)")

    # 质量警告
    quality_warnings = []
    if noise_ratio > 0.5:
        quality_warnings.append(f"噪声比例过高 ({noise_ratio*100:.1f}% > 50%)")
        print(f"   ⚠️ 警告: {quality_warnings[-1]}")
    if n_clusters < 5:
        quality_warnings.append(f"簇数过少 ({n_clusters} < 5)")
        print(f"   ⚠️ 警告: {quality_warnings[-1]}")

    clustering_info = {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'noise_ratio': noise_ratio,
        'min_cluster_size': min_cluster_size,
        'quality_warnings': quality_warnings,
        'quality_ok': len(quality_warnings) == 0
    }

    return cluster_labels, clustering_info


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


def compute_distance_baseline(embeddings, is_ood, id_labels):
    """计算距离基线"""
    from sklearn.metrics import roc_auc_score

    id_mask = ~np.array(is_ood)
    id_embeddings = embeddings[id_mask]
    id_labels_arr = np.array(id_labels)

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
    print(f"实验: {dataset_name} (全数据)")
    print("="*80)

    id_texts = data['id_texts']
    ood_texts = data['ood_texts']
    id_labels = data.get('id_labels', list(range(len(id_texts))))

    n_id = len(id_texts)
    n_ood = len(ood_texts)
    n_total = n_id + n_ood

    print(f"数据规模: ID={n_id}, OOD={n_ood}, 总={n_total}")

    # 合并文本
    all_texts = id_texts + ood_texts
    is_ood = [False] * n_id + [True] * n_ood

    # 生成嵌入
    embeddings = generate_embeddings(all_texts)
    print(f"   嵌入维度: {embeddings.shape}")

    # 构建k-NN图
    k = config['k_neighbors']
    knn_indices, knn_distances = build_knn_graph(embeddings, k)

    # 自适应HDBSCAN聚类
    cluster_labels, clustering_info = cluster_with_hdbscan_adaptive(embeddings)

    # 计算同配性特征
    print("   计算同配性特征...")
    features = compute_homophily_features(
        embeddings, knn_indices, knn_distances, cluster_labels, is_ood
    )

    # 统计检验
    print("   统计检验...")
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

        print(f"     {feature_name}:")
        print(f"       ID:  {result['id_mean']:.4f} ± {result['id_std']:.4f}")
        print(f"       OOD: {result['ood_mean']:.4f} ± {result['ood_std']:.4f}")
        print(f"       Cohen's d: {result['cohens_d']:.4f}, AUROC: {result['auroc']:.4f}")

    # 距离基线
    print("   计算距离基线...")
    baseline = compute_distance_baseline(embeddings, is_ood, id_labels)
    print(f"     距离基线 AUROC: {baseline['auroc']:.4f}")

    # 主要指标（NHR的Cohen's d）
    main_result = results[0]

    # 判断方向
    if main_result['cohens_d'] > 0.2:
        direction = "OOD MORE HOMOPHILIC"
    elif main_result['cohens_d'] < -0.2:
        direction = "OOD MORE HETEROPHILIC"
    else:
        direction = "WEAK/NO EFFECT"

    print(f"\n   主要发现: {direction} (d={main_result['cohens_d']:.4f})")

    return {
        'dataset': dataset_name,
        'config': config,
        'data_info': {
            'source': data['source'],
            'ood_type': data['ood_type'],
            'n_id': n_id,
            'n_ood': n_ood,
            'n_total': n_total
        },
        'clustering': clustering_info,
        'statistical_results': results,
        'baseline': baseline,
        'main_finding': {
            'cohens_d': main_result['cohens_d'],
            'auroc': main_result['auroc'],
            'direction': direction
        }
    }


def generate_comparison_report(all_results):
    """生成与之前结果的对比报告"""
    print(f"\n{'='*80}")
    print("三方对比: V2 vs 抽样版 vs 全数据版")
    print("="*80)

    print(f"\n{'Dataset':<12} {'V2 d':>10} {'Sampled d':>12} {'Full d':>12} {'Full Dir':<22}")
    print("-" * 75)

    for name in ['CLINC150', 'Banking77', 'ROSTD', 'ToxiGen']:
        v2_d = PREVIOUS_RESULTS['V2'][name]['d']
        sampled_d = PREVIOUS_RESULTS['Sampled'][name]['d']

        if name in all_results and 'error' not in all_results[name]:
            full_d = all_results[name]['main_finding']['cohens_d']
            full_dir = all_results[name]['main_finding']['direction']
            print(f"{name:<12} {v2_d:>10.4f} {sampled_d:>12.4f} {full_d:>12.4f} {full_dir:<22}")
        else:
            print(f"{name:<12} {v2_d:>10.4f} {sampled_d:>12.4f} {'ERROR':>12} {'N/A':<22}")

    # 聚类质量对比
    print(f"\n{'='*80}")
    print("聚类质量对比 (全数据版)")
    print("="*80)

    print(f"\n{'Dataset':<12} {'N_Total':>10} {'N_Clusters':>12} {'Noise_Ratio':>14} {'Quality':<10}")
    print("-" * 60)

    for name in ['CLINC150', 'Banking77', 'ROSTD', 'ToxiGen']:
        if name in all_results and 'error' not in all_results[name]:
            r = all_results[name]
            n_total = r['data_info']['n_total']
            n_clusters = r['clustering']['n_clusters']
            noise_ratio = r['clustering']['noise_ratio']
            quality = "OK" if r['clustering']['quality_ok'] else "WARNING"

            print(f"{name:<12} {n_total:>10} {n_clusters:>12} {noise_ratio*100:>13.1f}% {quality:<10}")


def main():
    """主函数"""
    setup_environment()

    all_results = {}
    dataset_loaders = {
        'CLINC150': load_clinc150_full,
        'Banking77': load_banking77_full,
        'ROSTD': load_rostd_full,
        'ToxiGen': load_toxigen_full
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
            print(f"\n   结果已保存: {result_file}")

        except Exception as e:
            print(f"\n❌ {dataset_name} 失败: {e}")
            import traceback
            traceback.print_exc()
            all_results[dataset_name] = {'error': str(e)}

    # 生成跨数据集对比
    print(f"\n{'='*80}")
    print("跨数据集对比 (全数据版)")
    print("="*80)

    comparison_data = []
    for name, result in all_results.items():
        if 'error' not in result:
            comparison_data.append({
                'Dataset': name,
                'N_ID': result['data_info']['n_id'],
                'N_OOD': result['data_info']['n_ood'],
                'OOD_Type': result['data_info']['ood_type'],
                'N_Clusters': result['clustering']['n_clusters'],
                'Noise_Ratio': f"{result['clustering']['noise_ratio']*100:.1f}%",
                'Cohen_d': result['main_finding']['cohens_d'],
                'AUROC': result['main_finding']['auroc'],
                'Direction': result['main_finding']['direction'],
                'Baseline_AUROC': result['baseline']['auroc'],
                'Clustering_OK': result['clustering']['quality_ok']
            })

    if comparison_data:
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))

        # 保存CSV
        csv_file = os.path.join(RESULTS_DIR, 'cross_dataset_comparison.csv')
        df.to_csv(csv_file, index=False)
        print(f"\n对比表已保存: {csv_file}")

    # 三方对比报告
    generate_comparison_report(all_results)

    # 保存完整报告
    report = {
        'experiment': 'RW3 Full Data Verification (No Sampling)',
        'timestamp': datetime.now().isoformat(),
        'embedding_model': EMBEDDING_MODEL,
        'results': all_results,
        'previous_results': PREVIOUS_RESULTS,
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
    print("实验结论 (全数据版)")
    print("="*80)

    conclusions = []
    for name, result in all_results.items():
        if 'error' not in result:
            d = result['main_finding']['cohens_d']
            ood_type = result['data_info']['ood_type']
            direction = result['main_finding']['direction']
            n_clusters = result['clustering']['n_clusters']
            noise_pct = result['clustering']['noise_ratio'] * 100
            conclusions.append(
                f"- {name} ({ood_type}): d={d:.4f} → {direction} "
                f"[{n_clusters}簇, {noise_pct:.1f}%噪声]"
            )

    for c in conclusions:
        print(c)

    print(f"\n{'='*80}")
    print("实验完成!")
    print("="*80)

    return all_results


if __name__ == '__main__':
    main()
