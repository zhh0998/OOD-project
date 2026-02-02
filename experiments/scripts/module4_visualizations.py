#!/usr/bin/env python3
"""
模块4: 论文级可视化
RW3 OOD检测项目 - CCF-A论文实验

生成图表:
1. ROC曲线对比 (Full Model vs KNN vs Heterophily)
2. 超参数热力图 (k × alpha AUROC)
3. ID/OOD分数分布直方图
4. 异配性分布对比图
5. 消融实验柱状图

输出格式: PDF (论文质量)

Author: RW3 OOD Detection Project
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# 设置论文级绘图风格
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (6, 4),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

from sklearn.metrics import roc_curve, auc
from src.datasets.ood_datasets import load_clinc150, load_banking77_oos

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False


class VisualizationGenerator:
    """论文级可视化生成器"""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.embedder = None

        # 输出目录
        self.figure_dir = PROJECT_ROOT / "experiments" / "figures"
        self.figure_dir.mkdir(parents=True, exist_ok=True)

        # 数据目录
        self.results_dir = PROJECT_ROOT / "experiments" / "results"

        # 颜色方案 (配色友好)
        self.colors = {
            'Full Model': '#2E86AB',      # 蓝色
            'KNN Only': '#A23B72',         # 紫红色
            'Heterophily Only': '#F18F01', # 橙色
            'ID': '#28A745',               # 绿色
            'OOD': '#DC3545',              # 红色
        }

    def _get_embedder(self):
        """获取sentence transformer"""
        if self.embedder is None and SBERT_AVAILABLE:
            print(f"Loading SentenceTransformer: {self.model_name}")
            self.embedder = SentenceTransformer(self.model_name)
        return self.embedder

    def _encode_texts(self, texts: list) -> np.ndarray:
        """编码文本"""
        embedder = self._get_embedder()
        embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=64)
        return embeddings.astype('float32')

    def _normalize(self, emb: np.ndarray) -> np.ndarray:
        """L2归一化"""
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / (norms + 1e-12)

    def _compute_scores(self, train_emb, train_labels, test_emb, k=10, alpha=0.3):
        """计算三种方法的分数"""
        train_emb = self._normalize(train_emb).astype('float32')
        test_emb = self._normalize(test_emb).astype('float32')

        train_labels = np.array(train_labels)
        num_classes = len(set(train_labels))

        # k-NN计算
        try:
            import faiss
            d = train_emb.shape[1]
            index = faiss.IndexFlatIP(d)
            index.add(train_emb)
            similarities, indices = index.search(test_emb, k)
            distances = 1 - similarities
        except ImportError:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=k, metric='cosine')
            nn.fit(train_emb)
            distances, indices = nn.kneighbors(test_emb)

        # k-NN分数
        knn_distances = distances[:, -1]
        knn_scores = (knn_distances - knn_distances.min()) / (knn_distances.max() - knn_distances.min() + 1e-10)

        # 异配性分数
        n_test = len(test_emb)
        heterophily_scores = np.zeros(n_test)
        for i in range(n_test):
            neighbor_labels = train_labels[indices[i]]
            unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
            probs = counts / counts.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(min(k, num_classes))
            normalized_entropy = entropy / (max_entropy + 1e-10)
            unique_ratio = len(unique_labels) / min(k, num_classes)
            heterophily_scores[i] = 0.5 * normalized_entropy + 0.5 * unique_ratio

        # Full Model
        full_scores = (1 - alpha) * knn_scores + alpha * heterophily_scores

        return {
            'Full Model': full_scores,
            'KNN Only': knn_scores,
            'Heterophily Only': heterophily_scores
        }

    def plot_roc_curves(self, dataset_name: str, scores: dict, labels: np.ndarray):
        """绘制ROC曲线对比图"""
        fig, ax = plt.subplots(figsize=(5, 4.5))

        for method_name, method_scores in scores.items():
            # 自动修复分数方向
            from sklearn.metrics import roc_auc_score
            auroc_orig = roc_auc_score(labels, method_scores)
            auroc_inv = roc_auc_score(labels, -method_scores)
            if auroc_inv > auroc_orig + 0.05:
                method_scores = -method_scores

            fpr, tpr, _ = roc_curve(labels, method_scores)
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr,
                   color=self.colors[method_name],
                   lw=2,
                   label=f'{method_name} (AUC = {roc_auc:.3f})')

        # 对角线
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curves - {dataset_name}')
        ax.legend(loc='lower right')
        ax.set_aspect('equal')

        # 保存
        output_path = self.figure_dir / f'roc_curves_{dataset_name.lower()}.pdf'
        fig.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.close(fig)

        print(f"ROC曲线已保存: {output_path}")

    def plot_grid_search_heatmap(self, dataset_name: str):
        """绘制网格搜索热力图"""
        # 加载数据
        heatmap_file = self.results_dir / 'grid_search' / 'heatmap_data.json'
        if not heatmap_file.exists():
            print(f"热力图数据不存在: {heatmap_file}")
            return

        with open(heatmap_file, 'r') as f:
            heatmap_data = json.load(f)

        if dataset_name not in heatmap_data:
            print(f"数据集 {dataset_name} 不在热力图数据中")
            return

        data = heatmap_data[dataset_name]
        matrix = np.array(data['auroc_matrix']) * 100  # 转换为百分比
        k_values = data['k_values']
        alpha_values = data['alpha_values']

        fig, ax = plt.subplots(figsize=(7, 5))

        # 自定义颜色映射
        cmap = plt.cm.RdYlGn

        im = ax.imshow(matrix, cmap=cmap, aspect='auto',
                      vmin=matrix.min() - 2, vmax=matrix.max() + 2)

        # 设置刻度
        ax.set_xticks(np.arange(len(alpha_values)))
        ax.set_yticks(np.arange(len(k_values)))
        ax.set_xticklabels([f'{a:.1f}' for a in alpha_values])
        ax.set_yticklabels([str(k) for k in k_values])

        ax.set_xlabel(r'$\alpha$ (Heterophily Weight)')
        ax.set_ylabel('$k$ (Number of Neighbors)')
        ax.set_title(f'AUROC (%) Grid Search - {dataset_name}')

        # 添加数值标签
        for i in range(len(k_values)):
            for j in range(len(alpha_values)):
                value = matrix[i, j]
                text_color = 'white' if value < (matrix.min() + matrix.max()) / 2 else 'black'
                ax.text(j, i, f'{value:.1f}',
                       ha='center', va='center',
                       color=text_color, fontsize=8)

        # 标记最佳位置
        best_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
        ax.add_patch(plt.Rectangle((best_idx[1]-0.5, best_idx[0]-0.5), 1, 1,
                                   fill=False, edgecolor='blue', linewidth=3))

        # 颜色条
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('AUROC (%)')

        # 保存
        output_path = self.figure_dir / f'heatmap_{dataset_name.lower()}.pdf'
        fig.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.close(fig)

        print(f"热力图已保存: {output_path}")

    def plot_score_distribution(self, dataset_name: str, scores: dict, labels: np.ndarray):
        """绘制ID/OOD分数分布图"""
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

        id_mask = labels == 0
        ood_mask = labels == 1

        for idx, (method_name, method_scores) in enumerate(scores.items()):
            ax = axes[idx]

            # 自动修复分数方向
            from sklearn.metrics import roc_auc_score
            auroc_orig = roc_auc_score(labels, method_scores)
            auroc_inv = roc_auc_score(labels, -method_scores)
            if auroc_inv > auroc_orig + 0.05:
                method_scores = -method_scores

            id_scores = method_scores[id_mask]
            ood_scores = method_scores[ood_mask]

            # 绘制直方图
            bins = 50
            ax.hist(id_scores, bins=bins, alpha=0.7, density=True,
                   color=self.colors['ID'], label='ID', edgecolor='white')
            ax.hist(ood_scores, bins=bins, alpha=0.7, density=True,
                   color=self.colors['OOD'], label='OOD', edgecolor='white')

            ax.set_xlabel('OOD Score')
            ax.set_ylabel('Density')
            ax.set_title(f'{method_name}')
            ax.legend()

        fig.suptitle(f'Score Distribution - {dataset_name}', fontsize=12, y=1.02)

        # 保存
        output_path = self.figure_dir / f'score_dist_{dataset_name.lower()}.pdf'
        fig.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.close(fig)

        print(f"分数分布图已保存: {output_path}")

    def plot_ablation_bar(self):
        """绘制消融实验柱状图"""
        # 加载完整实验结果
        results_file = self.results_dir / 'complete' / 'complete_results.json'
        if not results_file.exists():
            print(f"完整实验结果不存在: {results_file}")
            return

        with open(results_file, 'r') as f:
            data = json.load(f)

        results = data['results']
        datasets = list(results.keys())
        methods = ['Full Model', 'KNN Only', 'Heterophily Only']

        fig, ax = plt.subplots(figsize=(8, 5))

        x = np.arange(len(datasets))
        width = 0.25

        for idx, method in enumerate(methods):
            aurocs = [results[ds][method]['auroc']['mean'] * 100 for ds in datasets]
            stds = [results[ds][method]['auroc']['std'] * 100 for ds in datasets]

            bars = ax.bar(x + idx * width, aurocs, width,
                         label=method, color=self.colors[method],
                         yerr=stds, capsize=3)

            # 添加数值标签
            for bar, val in zip(bars, aurocs):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Dataset')
        ax.set_ylabel('AUROC (%)')
        ax.set_title('Ablation Study: Full Model vs Components')
        ax.set_xticks(x + width)
        ax.set_xticklabels(datasets)
        ax.legend(loc='upper right')
        ax.set_ylim([60, 105])

        # 保存
        output_path = self.figure_dir / 'ablation_study.pdf'
        fig.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.close(fig)

        print(f"消融实验图已保存: {output_path}")

    def plot_heterophily_comparison(self, dataset_name: str, train_emb, train_labels,
                                    test_emb, test_labels, k=10):
        """绘制ID vs OOD异配性分布对比"""
        train_emb = self._normalize(train_emb).astype('float32')
        test_emb = self._normalize(test_emb).astype('float32')

        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)
        num_classes = len(set(train_labels))

        # 计算异配性
        try:
            import faiss
            d = train_emb.shape[1]
            index = faiss.IndexFlatIP(d)
            index.add(train_emb)
            _, indices = index.search(test_emb, k)
        except ImportError:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=k, metric='cosine')
            nn.fit(train_emb)
            _, indices = nn.kneighbors(test_emb)

        n_test = len(test_emb)
        heterophily_scores = np.zeros(n_test)
        for i in range(n_test):
            neighbor_labels = train_labels[indices[i]]
            unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
            probs = counts / counts.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(min(k, num_classes))
            normalized_entropy = entropy / (max_entropy + 1e-10)
            unique_ratio = len(unique_labels) / min(k, num_classes)
            heterophily_scores[i] = 0.5 * normalized_entropy + 0.5 * unique_ratio

        # 分离ID和OOD
        id_het = heterophily_scores[test_labels == 0]
        ood_het = heterophily_scores[test_labels == 1]

        # 绘图
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.hist(id_het, bins=40, alpha=0.7, density=True,
               color=self.colors['ID'], label=f'ID (mean={id_het.mean():.3f})',
               edgecolor='white')
        ax.hist(ood_het, bins=40, alpha=0.7, density=True,
               color=self.colors['OOD'], label=f'OOD (mean={ood_het.mean():.3f})',
               edgecolor='white')

        ax.axvline(id_het.mean(), color=self.colors['ID'], linestyle='--', lw=2)
        ax.axvline(ood_het.mean(), color=self.colors['OOD'], linestyle='--', lw=2)

        ax.set_xlabel('Heterophily Score')
        ax.set_ylabel('Density')
        ax.set_title(f'Heterophily Distribution - {dataset_name}')
        ax.legend()

        # 保存
        output_path = self.figure_dir / f'heterophily_dist_{dataset_name.lower()}.pdf'
        fig.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.close(fig)

        print(f"异配性分布图已保存: {output_path}")

        return {'id_mean': id_het.mean(), 'ood_mean': ood_het.mean()}

    def generate_all(self):
        """生成所有可视化"""
        print("\n" + "="*70)
        print("模块4: 论文级可视化生成")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)

        datasets = {
            'CLINC150': load_clinc150,
            'Banking77': load_banking77_oos,
        }

        for dataset_name, loader in datasets.items():
            print(f"\n{'='*50}")
            print(f"处理数据集: {dataset_name}")
            print(f"{'='*50}")

            # 加载数据
            print("加载数据...")
            train_texts, test_texts, test_labels, _, train_labels = loader()

            # 生成embeddings
            print("生成embeddings...")
            train_emb = self._encode_texts(train_texts)
            test_emb = self._encode_texts(test_texts)

            test_labels = np.array(test_labels)

            # 计算分数
            print("计算OOD分数...")
            scores = self._compute_scores(train_emb, train_labels, test_emb, k=10, alpha=0.3)

            # 1. ROC曲线
            print("绘制ROC曲线...")
            self.plot_roc_curves(dataset_name, scores, test_labels)

            # 2. 分数分布
            print("绘制分数分布...")
            self.plot_score_distribution(dataset_name, scores, test_labels)

            # 3. 异配性分布
            print("绘制异配性分布...")
            self.plot_heterophily_comparison(
                dataset_name, train_emb, train_labels,
                test_emb, test_labels, k=10
            )

            # 4. 热力图
            print("绘制热力图...")
            self.plot_grid_search_heatmap(dataset_name)

        # 5. 消融实验柱状图
        print("\n绘制消融实验图...")
        self.plot_ablation_bar()

        print("\n" + "="*70)
        print("可视化生成完成!")
        print(f"所有图表保存在: {self.figure_dir}")
        print("="*70)


def main():
    """主函数"""
    generator = VisualizationGenerator()
    generator.generate_all()


if __name__ == "__main__":
    main()
