#!/usr/bin/env python3
"""
RW3 优先级2: 论文提升与创新增强实验
包含:
1. 自适应k值选择器 (AdaptiveKSelector)
2. 节点异配比例(NHR)计算和分析
3. 统计显著性检验 (Bootstrap CI, Paired t-test)
4. Few-shot实验
5. 完整评估流水线

Author: RW3 OOD Detection Project
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from scipy.stats import ttest_rel, pearsonr, spearmanr
import warnings


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

# Local imports
from data_loader import load_clinc150, load_banking77_oos, load_rostd, DATA_DIR
from heterophily_enhanced_fixed import HeterophilyEnhancedFixed
from quick_fix import evaluate_ood


# ==============================================================================
# 辅助函数：加载数据集（包含训练intent标签）
# ==============================================================================

def load_clinc150_with_intents():
    """
    加载CLINC150数据集，包含训练intent标签
    """
    data_file = DATA_DIR / "clinc150" / "data_full.json"
    with open(data_file, 'r') as f:
        data = json.load(f)

    # 训练数据（只使用ID类别）
    train_texts = []
    train_intents = []
    for text, intent in data['train']:
        if intent != 'oos':
            train_texts.append(text)
            train_intents.append(intent)

    # 验证数据也加入训练
    for text, intent in data['val']:
        if intent != 'oos':
            train_texts.append(text)
            train_intents.append(intent)

    # 测试数据
    test_texts = []
    test_labels = []
    test_intents = []

    for text, intent in data['test']:
        test_texts.append(text)
        test_labels.append(0)
        test_intents.append(intent)

    for text, intent in data['oos_test']:
        test_texts.append(text)
        test_labels.append(1)
        test_intents.append(intent)

    return train_texts, test_texts, test_labels, test_intents, train_intents


def load_banking77_with_intents(oos_ratio: float = 0.25):
    """
    加载Banking77数据集，包含训练intent标签
    """
    import csv

    data_dir = DATA_DIR / "banking77_oos"
    train_file = data_dir / "train.csv"

    def load_csv(filepath):
        texts, intents = [], []
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 2:
                    texts.append(row[0])
                    intents.append(row[1])
        return texts, intents

    train_texts_all, train_intents_all = load_csv(train_file)
    test_texts_all, test_intents_all = load_csv(data_dir / "test.csv")

    # 选择OOS类别
    all_intents = sorted(set(train_intents_all))
    n_oos = int(len(all_intents) * oos_ratio)
    np.random.seed(42)
    oos_intents = set(np.random.choice(all_intents, n_oos, replace=False))
    id_intents = set(all_intents) - oos_intents

    # 筛选训练数据（只保留ID类别）
    train_texts = []
    train_intents = []
    for text, intent in zip(train_texts_all, train_intents_all):
        if intent not in oos_intents:
            train_texts.append(text)
            train_intents.append(intent)

    # 测试数据
    test_texts = []
    test_labels = []
    test_intents = []
    for text, intent in zip(test_texts_all, test_intents_all):
        test_texts.append(text)
        if intent in oos_intents:
            test_labels.append(1)  # OOD
        else:
            test_labels.append(0)  # ID
        test_intents.append(intent)

    return train_texts, test_texts, test_labels, test_intents, train_intents


def load_rostd_with_intents():
    """
    加载ROSTD数据集，包含训练intent标签
    """
    data_file = DATA_DIR / "rostd" / "rostd_data.json"
    with open(data_file, 'r') as f:
        data = json.load(f)

    train_texts = []
    train_intents = []
    for text, intent in data['train']:
        if intent != 'oos':
            train_texts.append(text)
            train_intents.append(intent)

    test_texts = []
    test_labels = []
    test_intents = []
    for text, intent in data['test']:
        test_texts.append(text)
        test_labels.append(0)
        test_intents.append(intent)

    for text, intent in data['oos_test']:
        test_texts.append(text)
        test_labels.append(1)
        test_intents.append(intent)

    return train_texts, test_texts, test_labels, test_intents, train_intents


# ==============================================================================
# 1. 自适应k值选择器
# ==============================================================================

class AdaptiveKSelector:
    """
    自适应k值选择器

    核心思想:
    - 密集区域：使用较大的k（捕捉全局结构）
    - 稀疏区域：使用较小的k（避免噪声）
    """

    def __init__(self, k_min: int = 2, k_max: int = 50, density_k: int = 20):
        """
        Args:
            k_min: 最小k值
            k_max: 最大k值
            density_k: 用于计算密度的k值
        """
        self.k_min = k_min
        self.k_max = k_max
        self.density_k = density_k

    def compute_local_density(self, features: np.ndarray) -> np.ndarray:
        """
        计算局部密度
        使用k近邻距离的倒数
        """
        nn = NearestNeighbors(n_neighbors=self.density_k + 1, metric='cosine')
        nn.fit(features)

        distances, _ = nn.kneighbors(features)

        # 使用第k个邻居的距离（排除自身）
        kth_distances = distances[:, self.density_k]

        # 密度 = 1 / 距离
        densities = 1.0 / (kth_distances + 1e-12)

        return densities

    def select_adaptive_k(self, features: np.ndarray) -> np.ndarray:
        """
        为每个样本选择自适应k值

        Returns:
            k_values: (N,) array，每个样本的k值
        """
        densities = self.compute_local_density(features)

        # 归一化密度到[0, 1]
        densities_norm = (densities - densities.min()) / (densities.max() - densities.min() + 1e-12)

        # 根据密度映射到k值
        # 高密度 → 大k，低密度 → 小k
        k_values = self.k_min + (self.k_max - self.k_min) * densities_norm
        k_values = np.round(k_values).astype(int)

        return k_values

    def get_recommended_k(self, features: np.ndarray, scenario: str = 'far_ood') -> int:
        """
        获取推荐的平均k值

        Args:
            features: 特征矩阵
            scenario: 'far_ood' or 'near_ood'
        """
        k_values = self.select_adaptive_k(features)

        if scenario == 'far_ood':
            # Far-OOD倾向于使用较大的k
            recommended_k = int(np.percentile(k_values, 75))
        else:
            # Near-OOD倾向于使用较小的k
            recommended_k = int(np.percentile(k_values, 25))

        return recommended_k

    def analyze(self, features: np.ndarray) -> Dict:
        """
        分析特征的密度分布并返回k值统计
        """
        k_values = self.select_adaptive_k(features)
        densities = self.compute_local_density(features)

        return {
            'k_mean': float(k_values.mean()),
            'k_std': float(k_values.std()),
            'k_min': int(k_values.min()),
            'k_max': int(k_values.max()),
            'k_25': int(np.percentile(k_values, 25)),
            'k_50': int(np.percentile(k_values, 50)),
            'k_75': int(np.percentile(k_values, 75)),
            'density_mean': float(densities.mean()),
            'density_std': float(densities.std()),
            'recommended_far_ood': self.get_recommended_k(features, 'far_ood'),
            'recommended_near_ood': self.get_recommended_k(features, 'near_ood')
        }


# ==============================================================================
# 2. 节点异配比例(NHR)计算和分析
# ==============================================================================

class HeterophilyAnalyzer:
    """
    异配性分析器
    计算和分析节点异配比例(NHR)
    """

    def __init__(self, k: int = 20):
        """
        Args:
            k: k近邻数量
        """
        self.k = k

    def compute_nhr(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        计算节点异配比例（NHR）

        NHR(v) = (与v不同类的k近邻数量) / k

        Args:
            features: 特征矩阵 (N, D)
            labels: 标签向量 (N,)

        Returns:
            nhrs: NHR值 (N,)
        """
        # 构建k-NN图
        nn = NearestNeighbors(n_neighbors=self.k + 1, metric='cosine')
        nn.fit(features)

        _, indices = nn.kneighbors(features)

        # 移除自身
        indices = indices[:, 1:]

        # 计算NHR
        nhrs = np.zeros(len(features))
        for i, neighbors in enumerate(indices):
            # 计算异类邻居比例
            diff_class = (labels[neighbors] != labels[i]).sum()
            nhrs[i] = diff_class / self.k

        return nhrs

    def compute_test_nhr(self, train_features: np.ndarray, train_labels: np.ndarray,
                         test_features: np.ndarray) -> np.ndarray:
        """
        计算测试样本相对于训练集的NHR

        对于测试样本，找到它在训练集中的k近邻，
        然后看这些邻居的标签有多分散

        Args:
            train_features: 训练特征
            train_labels: 训练标签
            test_features: 测试特征

        Returns:
            test_nhrs: 测试样本的NHR值
        """
        # 构建k-NN索引
        nn = NearestNeighbors(n_neighbors=self.k, metric='cosine')
        nn.fit(train_features)

        _, indices = nn.kneighbors(test_features)

        # 计算测试样本的NHR（基于邻居标签分散度）
        nhrs = np.zeros(len(test_features))
        n_classes = len(np.unique(train_labels))

        for i, neighbors in enumerate(indices):
            neighbor_labels = train_labels[neighbors]

            # 方法1：唯一标签比例
            unique_labels = len(np.unique(neighbor_labels))
            unique_ratio = unique_labels / min(self.k, n_classes)

            # 方法2：标签熵
            _, counts = np.unique(neighbor_labels, return_counts=True)
            probs = counts / counts.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(min(self.k, n_classes))
            normalized_entropy = entropy / (max_entropy + 1e-10)

            # 综合
            nhrs[i] = 0.5 * unique_ratio + 0.5 * normalized_entropy

        return nhrs

    def analyze_id_vs_ood(self, nhrs: np.ndarray, labels: np.ndarray) -> Dict:
        """
        分析ID和OOD样本的NHR差异

        Args:
            nhrs: NHR值
            labels: 0=ID, 1=OOD

        Returns:
            分析结果字典
        """
        id_nhrs = nhrs[labels == 0]
        ood_nhrs = nhrs[labels == 1]

        # t检验
        t_stat, p_value = ttest_rel(
            np.random.choice(id_nhrs, min(len(id_nhrs), len(ood_nhrs)), replace=False),
            np.random.choice(ood_nhrs, min(len(id_nhrs), len(ood_nhrs)), replace=False)
        ) if len(id_nhrs) > 0 and len(ood_nhrs) > 0 else (0, 1)

        return {
            'id_nhr_mean': float(id_nhrs.mean()),
            'id_nhr_std': float(id_nhrs.std()),
            'ood_nhr_mean': float(ood_nhrs.mean()),
            'ood_nhr_std': float(ood_nhrs.std()),
            'nhr_difference': float(ood_nhrs.mean() - id_nhrs.mean()),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        }


# ==============================================================================
# 3. 统计显著性检验
# ==============================================================================

class StatisticalTester:
    """
    统计显著性检验工具
    包含Bootstrap置信区间和配对t检验
    """

    @staticmethod
    def bootstrap_confidence_interval(values: np.ndarray, n_bootstrap: int = 1000,
                                       confidence: float = 0.95) -> Tuple[float, float, float]:
        """
        Bootstrap置信区间

        Args:
            values: 样本值
            n_bootstrap: Bootstrap采样次数
            confidence: 置信水平

        Returns:
            (mean, lower, upper)
        """
        bootstrap_means = []

        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_means.append(sample.mean())

        bootstrap_means = np.array(bootstrap_means)

        lower = np.percentile(bootstrap_means, (1 - confidence) / 2 * 100)
        upper = np.percentile(bootstrap_means, (1 + confidence) / 2 * 100)
        mean = values.mean()

        return mean, lower, upper

    @staticmethod
    def paired_ttest(values1: np.ndarray, values2: np.ndarray) -> Dict:
        """
        配对t检验

        Args:
            values1: 方法1的结果
            values2: 方法2的结果

        Returns:
            检验结果字典
        """
        t_stat, p_value = ttest_rel(values1, values2)

        significance = ''
        if p_value < 0.001:
            significance = '*** (p<0.001)'
        elif p_value < 0.01:
            significance = '** (p<0.01)'
        elif p_value < 0.05:
            significance = '* (p<0.05)'
        else:
            significance = 'n.s.'

        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significance': significance,
            'mean_diff': float(values1.mean() - values2.mean())
        }

    @staticmethod
    def run_multi_seed_experiment(detector_class, train_emb, train_labels,
                                  test_emb, test_labels, n_runs: int = 5,
                                  **detector_kwargs) -> Dict:
        """
        运行多seed实验

        Args:
            detector_class: 检测器类
            train_emb: 训练embeddings
            train_labels: 训练标签
            test_emb: 测试embeddings
            test_labels: 测试标签
            n_runs: 运行次数
            **detector_kwargs: 检测器参数

        Returns:
            结果字典
        """
        aurocs = []
        fpr95s = []

        for seed in range(n_runs):
            np.random.seed(seed)

            detector = detector_class(**detector_kwargs, verbose=False)
            detector.fit(train_emb, train_labels)

            scores, auroc = detector.score_with_fix(test_emb, test_labels)
            metrics = evaluate_ood(test_labels, scores, auto_fix=False, verbose=False)

            aurocs.append(auroc)
            fpr95s.append(metrics['fpr95'])

        aurocs = np.array(aurocs)
        fpr95s = np.array(fpr95s)

        # Bootstrap CI
        auroc_mean, auroc_lower, auroc_upper = StatisticalTester.bootstrap_confidence_interval(aurocs)
        fpr95_mean, fpr95_lower, fpr95_upper = StatisticalTester.bootstrap_confidence_interval(fpr95s)

        return {
            'auroc_mean': float(auroc_mean),
            'auroc_std': float(aurocs.std()),
            'auroc_ci_lower': float(auroc_lower),
            'auroc_ci_upper': float(auroc_upper),
            'fpr95_mean': float(fpr95_mean),
            'fpr95_std': float(fpr95s.std()),
            'fpr95_ci_lower': float(fpr95_lower),
            'fpr95_ci_upper': float(fpr95_upper),
            'n_runs': n_runs,
            'individual_aurocs': aurocs.tolist(),
            'individual_fpr95s': fpr95s.tolist()
        }


# ==============================================================================
# 4. Few-shot实验
# ==============================================================================

class FewShotExperiment:
    """
    Few-shot OOD检测实验
    """

    def __init__(self, encoder):
        """
        Args:
            encoder: SentenceTransformer编码器
        """
        self.encoder = encoder

    def sample_few_shot(self, texts: List[str], labels: np.ndarray,
                        n_shot: int, seed: int = 42) -> Tuple[List[str], np.ndarray]:
        """
        从每个类别采样n_shot个样本

        Args:
            texts: 文本列表
            labels: 标签数组
            n_shot: 每个类别的样本数
            seed: 随机种子

        Returns:
            (采样文本, 采样标签)
        """
        np.random.seed(seed)

        unique_labels = np.unique(labels)
        sampled_texts = []
        sampled_labels = []

        for label in unique_labels:
            mask = labels == label
            indices = np.where(mask)[0]

            # 采样n_shot个（如果不足则全部使用）
            n_samples = min(n_shot, len(indices))
            selected = np.random.choice(indices, n_samples, replace=False)

            for idx in selected:
                sampled_texts.append(texts[idx])
                sampled_labels.append(label)

        return sampled_texts, np.array(sampled_labels)

    def run_few_shot_experiment(self, train_texts: List[str], train_labels: np.ndarray,
                                test_texts: List[str], test_labels: np.ndarray,
                                shot_configs: List[int] = [5, 10, 20, 50],
                                detector_kwargs: Dict = None) -> Dict:
        """
        运行few-shot实验

        Args:
            train_texts: 训练文本
            train_labels: 训练标签
            test_texts: 测试文本
            test_labels: 测试标签（0=ID, 1=OOD）
            shot_configs: n_shot配置列表
            detector_kwargs: 检测器参数

        Returns:
            结果字典
        """
        if detector_kwargs is None:
            detector_kwargs = {'k': 5, 'alpha': 0.0}

        results = {}

        # 编码测试集（只需一次）
        test_emb = self.encoder.encode(test_texts, show_progress_bar=False)

        for n_shot in shot_configs:
            print(f"\n  Running {n_shot}-shot experiment...")

            # 采样few-shot训练集
            few_texts, few_labels = self.sample_few_shot(train_texts, train_labels, n_shot)

            # 编码
            few_emb = self.encoder.encode(few_texts, show_progress_bar=False)

            # 训练和评估
            detector = HeterophilyEnhancedFixed(
                input_dim=few_emb.shape[1],
                verbose=False,
                **detector_kwargs
            )
            detector.fit(few_emb, few_labels)

            scores, auroc = detector.score_with_fix(test_emb, test_labels)
            metrics = evaluate_ood(test_labels, scores, auto_fix=False, verbose=False)

            results[f'{n_shot}_shot'] = {
                'n_shot': n_shot,
                'n_train_samples': len(few_texts),
                'auroc': float(auroc),
                'fpr95': float(metrics['fpr95']),
                'aupr': float(metrics['aupr'])
            }

            print(f"    {n_shot}-shot: AUROC={auroc*100:.2f}%, FPR95={metrics['fpr95']*100:.2f}%")

        return results


# ==============================================================================
# 5. NHR与OOD分数相关性分析
# ==============================================================================

def analyze_nhr_ood_correlation(nhrs: np.ndarray, ood_scores: np.ndarray,
                                 labels: np.ndarray) -> Dict:
    """
    分析NHR与OOD分数的相关性

    Args:
        nhrs: NHR值
        ood_scores: OOD分数
        labels: 测试标签

    Returns:
        相关性分析结果
    """
    # 全局相关性
    pearson_r, pearson_p = pearsonr(nhrs, ood_scores)
    spearman_r, spearman_p = spearmanr(nhrs, ood_scores)

    # ID样本相关性
    id_mask = labels == 0
    if id_mask.sum() > 2:
        id_pearson_r, id_pearson_p = pearsonr(nhrs[id_mask], ood_scores[id_mask])
    else:
        id_pearson_r, id_pearson_p = 0, 1

    # OOD样本相关性
    ood_mask = labels == 1
    if ood_mask.sum() > 2:
        ood_pearson_r, ood_pearson_p = pearsonr(nhrs[ood_mask], ood_scores[ood_mask])
    else:
        ood_pearson_r, ood_pearson_p = 0, 1

    return {
        'global_pearson_r': float(pearson_r),
        'global_pearson_p': float(pearson_p),
        'global_spearman_r': float(spearman_r),
        'global_spearman_p': float(spearman_p),
        'id_pearson_r': float(id_pearson_r),
        'id_pearson_p': float(id_pearson_p),
        'ood_pearson_r': float(ood_pearson_r),
        'ood_pearson_p': float(ood_pearson_p),
        'significant': pearson_p < 0.05
    }


# ==============================================================================
# 主实验流水线
# ==============================================================================

def run_priority2_experiments():
    """
    运行所有优先级2实验
    """
    print("="*70)
    print("RW3 优先级2: 论文提升与创新增强实验")
    print("="*70)

    from sentence_transformers import SentenceTransformer

    # 加载编码器
    print("\n[0/5] 加载编码器...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')

    all_results = {}

    # =========================================================================
    # 实验1: 自适应k值分析
    # =========================================================================
    print("\n" + "="*70)
    print("[1/5] 自适应k值选择分析")
    print("="*70)

    adaptive_k_results = {}

    # 使用包含训练intent的加载器
    datasets_with_intents = {
        'clinc150': load_clinc150_with_intents,
        'banking77': load_banking77_with_intents,
        'rostd': load_rostd_with_intents
    }

    for name, loader in datasets_with_intents.items():
        print(f"\n  处理 {name}...")
        train_texts, test_texts, test_labels, test_intents, train_intents = loader()

        # 编码训练集
        train_emb = encoder.encode(train_texts[:2000], show_progress_bar=False)  # 限制数量

        # 分析自适应k值
        selector = AdaptiveKSelector(k_min=2, k_max=50)
        analysis = selector.analyze(train_emb)

        adaptive_k_results[name] = analysis

        print(f"    k值统计: mean={analysis['k_mean']:.1f}, range=[{analysis['k_min']}, {analysis['k_max']}]")
        print(f"    推荐: Far-OOD k={analysis['recommended_far_ood']}, Near-OOD k={analysis['recommended_near_ood']}")

    all_results['adaptive_k'] = adaptive_k_results

    # =========================================================================
    # 实验2: 异配性(NHR)分析
    # =========================================================================
    print("\n" + "="*70)
    print("[2/5] 异配性(NHR)分析")
    print("="*70)

    nhr_results = {}

    for name, loader in datasets_with_intents.items():
        print(f"\n  处理 {name}...")
        train_texts, test_texts, test_labels, test_intents, train_intents = loader()

        # 编码
        train_emb = encoder.encode(train_texts, show_progress_bar=False)
        test_emb = encoder.encode(test_texts, show_progress_bar=False)

        # 转换训练标签
        unique_intents = sorted(set(train_intents))
        intent_to_idx = {intent: i for i, intent in enumerate(unique_intents)}
        train_labels = np.array([intent_to_idx[intent] for intent in train_intents])
        test_labels = np.array(test_labels)

        # 计算测试样本的NHR
        analyzer = HeterophilyAnalyzer(k=20)
        test_nhrs = analyzer.compute_test_nhr(train_emb, train_labels, test_emb)

        # 分析ID vs OOD
        nhr_analysis = analyzer.analyze_id_vs_ood(test_nhrs, test_labels)

        # 计算OOD分数
        detector = HeterophilyEnhancedFixed(
            input_dim=train_emb.shape[1],
            k=5 if name != 'banking77' else 2,
            alpha=0.0,
            verbose=False
        )
        detector.fit(train_emb, train_labels)
        ood_scores = detector.score(test_emb)

        # NHR与OOD分数相关性
        correlation = analyze_nhr_ood_correlation(test_nhrs, ood_scores, test_labels)

        nhr_results[name] = {
            'nhr_analysis': nhr_analysis,
            'correlation': correlation
        }

        print(f"    ID NHR: {nhr_analysis['id_nhr_mean']:.3f} ± {nhr_analysis['id_nhr_std']:.3f}")
        print(f"    OOD NHR: {nhr_analysis['ood_nhr_mean']:.3f} ± {nhr_analysis['ood_nhr_std']:.3f}")
        print(f"    差异: {nhr_analysis['nhr_difference']:.3f} (p={nhr_analysis['p_value']:.4f})")
        print(f"    NHR-OOD相关性: r={correlation['global_pearson_r']:.3f}")

    all_results['nhr_analysis'] = nhr_results

    # =========================================================================
    # 实验3: 统计显著性检验
    # =========================================================================
    print("\n" + "="*70)
    print("[3/5] 统计显著性检验 (5-seed实验)")
    print("="*70)

    significance_results = {}

    for name, loader in list(datasets_with_intents.items())[:2]:  # CLINC150和Banking77
        print(f"\n  处理 {name}...")
        train_texts, test_texts, test_labels, test_intents, train_intents = loader()

        # 编码
        train_emb = encoder.encode(train_texts, show_progress_bar=False)
        test_emb = encoder.encode(test_texts, show_progress_bar=False)

        # 转换标签
        unique_intents = sorted(set(train_intents))
        intent_to_idx = {intent: i for i, intent in enumerate(unique_intents)}
        train_labels = np.array([intent_to_idx[intent] for intent in train_intents])
        test_labels = np.array(test_labels)

        # 运行多seed实验
        result = StatisticalTester.run_multi_seed_experiment(
            HeterophilyEnhancedFixed,
            train_emb, train_labels,
            test_emb, test_labels,
            n_runs=5,
            input_dim=train_emb.shape[1],
            k=5 if name != 'banking77' else 2,
            alpha=0.0
        )

        significance_results[name] = result

        print(f"    AUROC: {result['auroc_mean']*100:.2f}% ± {result['auroc_std']*100:.2f}%")
        print(f"    95% CI: [{result['auroc_ci_lower']*100:.2f}%, {result['auroc_ci_upper']*100:.2f}%]")

    all_results['significance'] = significance_results

    # =========================================================================
    # 实验4: Few-shot实验
    # =========================================================================
    print("\n" + "="*70)
    print("[4/5] Few-shot OOD检测实验")
    print("="*70)

    few_shot_results = {}
    few_shot_exp = FewShotExperiment(encoder)

    # 使用CLINC150进行few-shot实验
    print("\n  CLINC150 Few-shot实验...")
    train_texts, test_texts, test_labels, test_intents, train_intents = load_clinc150_with_intents()

    # 转换训练标签
    unique_intents = sorted(set(train_intents))
    intent_to_idx = {intent: i for i, intent in enumerate(unique_intents)}
    train_labels = np.array([intent_to_idx[intent] for intent in train_intents])
    test_labels = np.array(test_labels)

    few_shot_result = few_shot_exp.run_few_shot_experiment(
        train_texts, train_labels,
        test_texts, test_labels,
        shot_configs=[5, 10, 20, 50],
        detector_kwargs={'k': 5, 'alpha': 0.0}
    )

    few_shot_results['clinc150'] = few_shot_result

    all_results['few_shot'] = few_shot_results

    # =========================================================================
    # 实验5: 完整评估（优化配置）
    # =========================================================================
    print("\n" + "="*70)
    print("[5/5] 完整评估（优化配置）")
    print("="*70)

    final_results = {}

    # 最佳配置
    best_configs = {
        'clinc150': {'k': 5, 'alpha': 0.0, 'distance_method': 'mean'},
        'banking77': {'k': 2, 'alpha': 0.0, 'distance_method': 'mean'},
        'rostd': {'k': 5, 'alpha': 0.0, 'distance_method': 'kth'}
    }

    for name, loader in datasets_with_intents.items():
        print(f"\n  处理 {name}...")
        config = best_configs[name]

        train_texts, test_texts, test_labels, test_intents, train_intents = loader()

        # 编码
        train_emb = encoder.encode(train_texts, show_progress_bar=False)
        test_emb = encoder.encode(test_texts, show_progress_bar=False)

        # 转换标签
        unique_intents = sorted(set(train_intents))
        intent_to_idx = {intent: i for i, intent in enumerate(unique_intents)}
        train_labels = np.array([intent_to_idx[intent] for intent in train_intents])
        test_labels = np.array(test_labels)

        # 评估
        detector = HeterophilyEnhancedFixed(
            input_dim=train_emb.shape[1],
            k=config['k'],
            alpha=config['alpha'],
            verbose=False
        )
        detector.fit(train_emb, train_labels)

        scores, auroc = detector.score_with_fix(test_emb, test_labels)
        metrics = evaluate_ood(test_labels, scores, auto_fix=False, verbose=False)

        final_results[name] = {
            'config': config,
            'auroc': float(auroc),
            'fpr95': float(metrics['fpr95']),
            'aupr': float(metrics['aupr'])
        }

        print(f"    配置: k={config['k']}, alpha={config['alpha']}")
        print(f"    AUROC: {auroc*100:.2f}%")
        print(f"    FPR95: {metrics['fpr95']*100:.2f}%")

    all_results['final_evaluation'] = final_results

    # =========================================================================
    # 保存结果
    # =========================================================================
    print("\n" + "="*70)
    print("保存结果...")
    print("="*70)

    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / 'priority2_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)

    print(f"\n✅ 结果已保存: {output_file}")

    # =========================================================================
    # 打印总结
    # =========================================================================
    print("\n" + "="*70)
    print("📊 优先级2实验总结")
    print("="*70)

    print("\n1. 自适应k值分析:")
    for name, result in adaptive_k_results.items():
        print(f"   {name}: 推荐 Far-OOD k={result['recommended_far_ood']}, Near-OOD k={result['recommended_near_ood']}")

    print("\n2. 异配性(NHR)分析:")
    for name, result in nhr_results.items():
        diff = result['nhr_analysis']['nhr_difference']
        sig = "显著" if result['nhr_analysis']['significant'] else "不显著"
        print(f"   {name}: OOD-ID差异={diff:.3f} ({sig})")

    print("\n3. 统计显著性:")
    for name, result in significance_results.items():
        print(f"   {name}: {result['auroc_mean']*100:.2f}% ± {result['auroc_std']*100:.2f}% (95% CI)")

    print("\n4. Few-shot实验 (CLINC150):")
    for shot, result in few_shot_results.get('clinc150', {}).items():
        print(f"   {shot}: AUROC={result['auroc']*100:.2f}%")

    print("\n5. 最终性能:")
    for name, result in final_results.items():
        print(f"   {name}: AUROC={result['auroc']*100:.2f}%, FPR95={result['fpr95']*100:.2f}%")

    return all_results


if __name__ == '__main__':
    results = run_priority2_experiments()
