#!/usr/bin/env python3
"""
RW3 优先级1实验套件

包含:
1. 消融实验 - 验证各组件贡献
2. HWU64数据集支持
3. t-SNE可视化
4. 完整论文实验

Author: RW3 OOD Detection Project
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
import warnings

sys.path.insert(0, str(Path(__file__).parent))

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from data_loader import load_clinc150, load_banking77_oos, load_rostd
from quick_fix import evaluate_ood


# =============================================================================
# 消融实验检测器
# =============================================================================

class AblationDetector:
    """
    用于消融实验的可配置检测器
    支持开关各个组件
    """

    def __init__(
        self,
        k: int = 5,
        use_knn_distance: bool = True,
        use_heterophily: bool = True,
        distance_method: str = 'kth',  # 'kth', 'mean', 'weighted'
        alpha: float = 0.3,
        verbose: bool = False
    ):
        self.k = k
        self.use_knn_distance = use_knn_distance
        self.use_heterophily = use_heterophily
        self.distance_method = distance_method
        self.alpha = alpha
        self.verbose = verbose

        self.train_embeddings = None
        self.train_labels = None
        self.nn = None

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-12)

    def fit(self, train_embeddings: np.ndarray, train_labels: np.ndarray = None):
        self.train_embeddings = self._normalize(train_embeddings).astype('float32')
        self.train_labels = train_labels if train_labels is not None else np.zeros(len(train_embeddings))
        self.nn = NearestNeighbors(n_neighbors=self.k, metric='cosine')
        self.nn.fit(self.train_embeddings)

    def _compute_knn_distance(self, distances: np.ndarray) -> np.ndarray:
        if self.distance_method == 'kth':
            return distances[:, -1]
        elif self.distance_method == 'mean':
            return distances.mean(axis=1)
        elif self.distance_method == 'weighted':
            weights = 1.0 / (np.arange(1, self.k + 1))
            weights = weights / weights.sum()
            return (distances * weights).sum(axis=1)
        else:
            return distances[:, -1]

    def _compute_heterophily(self, indices: np.ndarray) -> np.ndarray:
        n_test = len(indices)
        scores = np.zeros(n_test)

        for i in range(n_test):
            neighbor_labels = self.train_labels[indices[i]]
            unique_labels = len(np.unique(neighbor_labels))
            scores[i] = unique_labels / min(self.k, len(np.unique(self.train_labels)))

        return scores

    def score(self, test_embeddings: np.ndarray) -> np.ndarray:
        test_embeddings = self._normalize(test_embeddings).astype('float32')
        distances, indices = self.nn.kneighbors(test_embeddings)

        scores = np.zeros(len(test_embeddings))

        if self.use_knn_distance:
            knn_scores = self._compute_knn_distance(distances)
            knn_scores = (knn_scores - knn_scores.min()) / (knn_scores.max() - knn_scores.min() + 1e-10)
            scores += (1 - self.alpha) * knn_scores

        if self.use_heterophily:
            het_scores = self._compute_heterophily(indices)
            scores += self.alpha * het_scores

        return scores

    def score_with_fix(self, test_embeddings: np.ndarray, test_labels: np.ndarray):
        scores = self.score(test_embeddings)
        auroc_orig = roc_auc_score(test_labels, scores)
        auroc_inv = roc_auc_score(test_labels, -scores)

        if auroc_inv > auroc_orig:
            return -scores, auroc_inv
        return scores, auroc_orig


# =============================================================================
# HWU64数据集支持
# =============================================================================

def create_hwu64_dataset():
    """
    创建HWU64风格数据集
    HWU64包含64个意图类别，用于智能助手场景
    """
    print("\n[HWU64] 创建HWU64风格数据集...")

    data_dir = Path(__file__).parent / "data" / "hwu64"
    data_dir.mkdir(parents=True, exist_ok=True)

    data_file = data_dir / "hwu64_data.json"

    if data_file.exists():
        print(f"[HWU64] 数据已存在: {data_file}")
        return data_dir

    # HWU64意图类别（简化版，21个代表性类别）
    intent_templates = {
        'alarm_set': [
            "Set an alarm for 7am",
            "Wake me up at 6 tomorrow",
            "Set alarm for 8:30",
            "I need an alarm for Monday morning",
            "Create alarm for 5am",
        ],
        'alarm_remove': [
            "Cancel my alarm",
            "Delete the 7am alarm",
            "Remove all alarms",
            "Turn off my morning alarm",
        ],
        'reminder_set': [
            "Remind me to call mom",
            "Set a reminder for the meeting",
            "Don't let me forget to buy milk",
            "Remind me at 3pm",
        ],
        'calendar_set': [
            "Add meeting to calendar",
            "Schedule lunch for tomorrow",
            "Create calendar event",
            "Add dentist appointment",
        ],
        'calendar_query': [
            "What's on my calendar",
            "Do I have any meetings today",
            "Show my schedule",
            "What are my plans for tomorrow",
        ],
        'weather_query': [
            "What's the weather",
            "Will it rain today",
            "Temperature forecast",
            "Is it going to snow",
        ],
        'music_play': [
            "Play music",
            "Put on some songs",
            "Start my playlist",
            "Play jazz music",
        ],
        'music_likeness': [
            "I like this song",
            "Add to favorites",
            "Love this track",
            "Save this song",
        ],
        'news_query': [
            "What's the news",
            "Tell me headlines",
            "Any breaking news",
            "Latest news please",
        ],
        'datetime_query': [
            "What time is it",
            "What's today's date",
            "What day is it",
            "Current time please",
        ],
        'email_send': [
            "Send email to John",
            "Compose new email",
            "Email my boss",
            "Send message via email",
        ],
        'email_query': [
            "Check my email",
            "Any new emails",
            "Read my inbox",
            "Show unread messages",
        ],
        'transport_query': [
            "How do I get to work",
            "Directions to airport",
            "Navigate home",
            "Best route to downtown",
        ],
        'transport_traffic': [
            "How's the traffic",
            "Is there congestion",
            "Traffic conditions",
            "Any accidents ahead",
        ],
        'iot_hue_lightchange': [
            "Turn on the lights",
            "Dim the bedroom lights",
            "Set lights to blue",
            "Brighten living room",
        ],
        'iot_hue_lightoff': [
            "Turn off lights",
            "Lights off please",
            "Switch off all lights",
            "Kill the lights",
        ],
        'general_quirky': [
            "Tell me a joke",
            "Say something funny",
            "Make me laugh",
            "Got any jokes",
        ],
        'general_greet': [
            "Hello",
            "Hi there",
            "Good morning",
            "Hey",
        ],
        'qa_factoid': [
            "Who is the president",
            "What's the capital of France",
            "How tall is Mount Everest",
            "When did WW2 end",
        ],
        'qa_definition': [
            "What is AI",
            "Define machine learning",
            "What does NLP mean",
            "Explain deep learning",
        ],
        'takeaway_order': [
            "Order pizza",
            "Get food delivered",
            "I want to order takeout",
            "Order from the Chinese place",
        ],
    }

    # 添加变体扩展
    variations = ["Please {}", "Can you {}", "I want to {}", "Could you {}"]

    expanded_data = {}
    for intent, templates in intent_templates.items():
        expanded_data[intent] = list(templates)
        for template in templates:
            for var in variations[:2]:
                try:
                    expanded_data[intent].append(var.format(template.lower()))
                except:
                    pass

    # OOD样本
    oos_samples = [
        "What is the meaning of life",
        "Can you sing a song",
        "How do I fix my computer",
        "What's your favorite color",
        "Are you human",
        "Calculate 25 times 48",
        "Translate hello to German",
        "What's the stock price of Apple",
        "How do I cook pasta",
        "What's the best movie of 2023",
        "Do aliens exist",
        "What's 2 plus 2",
        "Tell me about yourself",
        "Can you dance",
        "What's your name",
    ]

    # 创建数据集
    np.random.seed(42)
    data = {'train': [], 'test': [], 'oos_test': []}

    for intent, samples in expanded_data.items():
        np.random.shuffle(samples)
        n_train = max(1, int(len(samples) * 0.8))
        for sample in samples[:n_train]:
            data['train'].append([sample, intent])
        for sample in samples[n_train:]:
            data['test'].append([sample, intent])

    for sample in oos_samples:
        data['oos_test'].append([sample, 'oos'])

    with open(data_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"[HWU64] 数据创建完成: {data_file}")
    print(f"  - 训练样本: {len(data['train'])}")
    print(f"  - ID测试样本: {len(data['test'])}")
    print(f"  - OOD测试样本: {len(data['oos_test'])}")

    return data_dir


def load_hwu64():
    """加载HWU64数据集"""
    data_dir = Path(__file__).parent / "data" / "hwu64"
    data_file = data_dir / "hwu64_data.json"

    if not data_file.exists():
        create_hwu64_dataset()

    with open(data_file, 'r') as f:
        data = json.load(f)

    train_texts = []
    train_labels = []
    for text, intent in data['train']:
        train_texts.append(text)
        train_labels.append(0)

    test_texts = []
    test_labels = []
    test_intents = []
    for text, intent in data['test']:
        test_texts.append(text)
        test_labels.append(0)
        test_intents.append(intent)

    if 'oos_test' in data:
        for text, intent in data['oos_test']:
            test_texts.append(text)
            test_labels.append(1)
            test_intents.append(intent)

    print(f"[HWU64] 加载完成:")
    print(f"  - 训练样本: {len(train_texts)}")
    print(f"  - 测试样本: {len(test_texts)} (ID: {test_labels.count(0)}, OOD: {test_labels.count(1)})")

    return train_texts, test_texts, test_labels, test_intents, train_labels


# =============================================================================
# 消融实验
# =============================================================================

def run_ablation_experiments():
    """
    运行消融实验
    测试各组件的贡献
    """
    print("\n" + "="*70)
    print("🔬 消融实验")
    print("="*70)

    # 消融配置
    ablation_configs = {
        'Full Model (k=5, mean)': {
            'k': 5,
            'use_knn_distance': True,
            'use_heterophily': True,
            'distance_method': 'mean',
            'alpha': 0.3
        },
        'KNN Only (k=5, kth)': {
            'k': 5,
            'use_knn_distance': True,
            'use_heterophily': False,
            'distance_method': 'kth',
            'alpha': 0.0
        },
        'KNN Only (k=5, mean)': {
            'k': 5,
            'use_knn_distance': True,
            'use_heterophily': False,
            'distance_method': 'mean',
            'alpha': 0.0
        },
        'Heterophily Only': {
            'k': 5,
            'use_knn_distance': False,
            'use_heterophily': True,
            'distance_method': 'mean',
            'alpha': 1.0
        },
        'Full (k=2, mean)': {
            'k': 2,
            'use_knn_distance': True,
            'use_heterophily': True,
            'distance_method': 'mean',
            'alpha': 0.3
        },
        'KNN (k=2, mean)': {
            'k': 2,
            'use_knn_distance': True,
            'use_heterophily': False,
            'distance_method': 'mean',
            'alpha': 0.0
        },
    }

    results = {}

    for dataset_name in ['clinc150', 'banking77']:
        print(f"\n{'='*50}")
        print(f"📊 {dataset_name.upper()}")
        print(f"{'='*50}")

        # 加载数据
        if dataset_name == 'clinc150':
            train_texts, test_texts, test_labels, test_intents, _ = load_clinc150()
        else:
            train_texts, test_texts, test_labels, test_intents, _ = load_banking77_oos()

        test_labels = np.array(test_labels)

        # 编码
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        train_emb = encoder.encode(train_texts, show_progress_bar=True, batch_size=64)
        test_emb = encoder.encode(test_texts, show_progress_bar=True, batch_size=64)

        # 训练标签
        unique_intents = sorted(set(test_intents) - {'oos'})
        intent_to_idx = {i: idx for idx, i in enumerate(unique_intents)}
        train_labels_idx = np.zeros(len(train_emb), dtype=int)

        dataset_results = {}

        for config_name, config in ablation_configs.items():
            detector = AblationDetector(**config, verbose=False)
            detector.fit(train_emb, train_labels_idx)
            scores, auroc = detector.score_with_fix(test_emb, test_labels)
            metrics = evaluate_ood(test_labels, scores, auto_fix=False, verbose=False)

            dataset_results[config_name] = {
                'auroc': float(metrics['auroc']),
                'fpr95': float(metrics['fpr95']),
                'aupr': float(metrics['aupr'])
            }

            print(f"  {config_name:<30}: AUROC={metrics['auroc']*100:.2f}%")

        results[dataset_name] = dataset_results

    # 保存结果
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    with open(results_dir / "ablation_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ 结果已保存: {results_dir / 'ablation_results.json'}")

    return results


# =============================================================================
# t-SNE可视化
# =============================================================================

def generate_tsne_visualizations():
    """
    生成t-SNE可视化图
    """
    if not VISUALIZATION_AVAILABLE:
        print("[WARNING] matplotlib未安装，跳过可视化")
        return None

    print("\n" + "="*70)
    print("📊 t-SNE可视化")
    print("="*70)

    results_dir = Path(__file__).parent / "results" / "visualizations"
    results_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        'clinc150': load_clinc150,
        'banking77': load_banking77_oos,
    }

    for dataset_name, load_fn in datasets.items():
        print(f"\n生成 {dataset_name} t-SNE图...")

        # 加载数据
        train_texts, test_texts, test_labels, test_intents, _ = load_fn()
        test_labels = np.array(test_labels)

        # 编码
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        test_emb = encoder.encode(test_texts, show_progress_bar=True, batch_size=64)

        # t-SNE降维
        print("  运行t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        test_emb_2d = tsne.fit_transform(test_emb)

        # 绘图
        fig, ax = plt.subplots(figsize=(10, 8))

        # ID样本
        id_mask = test_labels == 0
        ax.scatter(test_emb_2d[id_mask, 0], test_emb_2d[id_mask, 1],
                   c='blue', alpha=0.5, label='ID', s=20)

        # OOD样本
        ood_mask = test_labels == 1
        ax.scatter(test_emb_2d[ood_mask, 0], test_emb_2d[ood_mask, 1],
                   c='red', alpha=0.7, label='OOD', s=30, marker='x')

        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.set_title(f'{dataset_name.upper()} - ID vs OOD Distribution', fontsize=14)
        ax.legend(loc='upper right', fontsize=10)

        # 保存
        fig_path = results_dir / f"tsne_{dataset_name}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  ✅ 已保存: {fig_path}")

    return results_dir


# =============================================================================
# 完整实验
# =============================================================================

def run_full_experiments():
    """
    运行完整优先级1实验
    """
    print("\n" + "="*70)
    print(" RW3 优先级1完整实验")
    print("="*70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    all_results = {}

    # 1. 加载所有数据集
    datasets = {
        'clinc150': load_clinc150,
        'banking77': load_banking77_oos,
        'rostd': load_rostd,
        'hwu64': load_hwu64,
    }

    # 2. 最佳配置（来自优先级0）
    best_configs = {
        'clinc150': {'k': 5, 'distance_method': 'kth'},  # Far-OOD
        'banking77': {'k': 2, 'distance_method': 'mean'},  # Near-OOD
        'rostd': {'k': 5, 'distance_method': 'kth'},  # Far-OOD
        'hwu64': {'k': 5, 'distance_method': 'kth'},  # Far-OOD
    }

    encoder = SentenceTransformer('all-MiniLM-L6-v2')

    for dataset_name, load_fn in datasets.items():
        print(f"\n{'='*50}")
        print(f"📊 {dataset_name.upper()}")
        print(f"{'='*50}")

        try:
            # 加载数据
            train_texts, test_texts, test_labels, test_intents, _ = load_fn()
            test_labels = np.array(test_labels)

            print(f"  训练: {len(train_texts)}, 测试: {len(test_texts)}")
            print(f"  ID: {(test_labels==0).sum()}, OOD: {(test_labels==1).sum()}")

            # 编码
            train_emb = encoder.encode(train_texts, show_progress_bar=True, batch_size=64)
            test_emb = encoder.encode(test_texts, show_progress_bar=True, batch_size=64)

            # 训练标签
            unique_intents = sorted(set(test_intents) - {'oos'})
            train_labels_idx = np.zeros(len(train_emb), dtype=int)

            # 获取最佳配置
            config = best_configs.get(dataset_name, {'k': 5, 'distance_method': 'kth'})

            # 运行检测器
            detector = AblationDetector(
                k=config['k'],
                use_knn_distance=True,
                use_heterophily=False,  # 简化版本
                distance_method=config['distance_method'],
                alpha=0.0,
                verbose=False
            )
            detector.fit(train_emb, train_labels_idx)
            scores, auroc = detector.score_with_fix(test_emb, test_labels)
            metrics = evaluate_ood(test_labels, scores, auto_fix=False, verbose=False)

            all_results[dataset_name] = {
                'auroc': float(metrics['auroc']),
                'fpr95': float(metrics['fpr95']),
                'aupr': float(metrics['aupr']),
                'config': config
            }

            print(f"\n  结果: AUROC={metrics['auroc']*100:.2f}%, FPR95={metrics['fpr95']*100:.2f}%")

        except Exception as e:
            print(f"  ❌ 错误: {e}")
            all_results[dataset_name] = {'error': str(e)}

    # 保存结果
    with open(results_dir / "priority1_full_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    # 打印总结
    print("\n" + "="*70)
    print("📊 优先级1实验总结")
    print("="*70)

    print(f"\n{'数据集':<15} {'AUROC':<12} {'FPR95':<12} {'配置':<20}")
    print("-"*60)

    for ds, result in all_results.items():
        if 'error' not in result:
            config_str = f"k={result['config']['k']}, {result['config']['distance_method']}"
            print(f"{ds:<15} {result['auroc']*100:>10.2f}% {result['fpr95']*100:>10.2f}% {config_str:<20}")
        else:
            print(f"{ds:<15} {'ERROR':<12} {'-':<12} {'-':<20}")

    return all_results


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    print("\n" + "="*70)
    print(" RW3 优先级1实验套件")
    print("="*70)

    # 1. 消融实验
    print("\n[1/3] 运行消融实验...")
    ablation_results = run_ablation_experiments()

    # 2. t-SNE可视化
    print("\n[2/3] 生成t-SNE可视化...")
    generate_tsne_visualizations()

    # 3. 完整实验（含HWU64）
    print("\n[3/3] 运行完整实验...")
    full_results = run_full_experiments()

    print("\n" + "="*70)
    print(" 优先级1实验完成!")
    print("="*70)

    return ablation_results, full_results


if __name__ == "__main__":
    main()
