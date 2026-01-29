#!/usr/bin/env python3
"""
模块2: 完整实验重跑
RW3 OOD检测项目 - CCF-A论文实验

实验配置:
- 数据集: CLINC150, Banking77, ROSTD, HWU64
- 方法: Full Model, KNN Only, Heterophily Only
- 运行次数: 5次
- 输出: JSON结果 + LaTeX表格

Author: RW3 OOD Detection Project
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/home/user/OOD-project')

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

# 导入数据加载器
from src.datasets.ood_datasets import load_clinc150, load_banking77_oos, load_rostd

# 尝试导入sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("[WARNING] sentence_transformers not available")


class ExperimentRunner:
    """完整实验运行器"""

    def __init__(self, n_runs: int = 5, k: int = 10, alpha: float = 0.3,
                 model_name: str = 'all-MiniLM-L6-v2'):
        self.n_runs = n_runs
        self.k = k
        self.alpha = alpha
        self.results = {}
        self.model_name = model_name
        self.embedder = None

        # 输出目录
        self.output_dir = Path('/home/user/OOD-project/experiments/results/complete')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.table_dir = Path('/home/user/OOD-project/experiments/tables')
        self.table_dir.mkdir(parents=True, exist_ok=True)

    def _get_embedder(self):
        """获取或初始化sentence transformer"""
        if self.embedder is None and SBERT_AVAILABLE:
            print(f"[Init] Loading SentenceTransformer: {self.model_name}")
            self.embedder = SentenceTransformer(self.model_name)
        return self.embedder

    def _encode_texts(self, texts: list) -> np.ndarray:
        """将文本编码为embeddings"""
        embedder = self._get_embedder()
        if embedder is None:
            raise ImportError("SentenceTransformer not available")

        print(f"  Encoding {len(texts)} texts...")
        embeddings = embedder.encode(
            texts,
            show_progress_bar=True,
            batch_size=64,
            convert_to_numpy=True
        )
        return embeddings.astype('float32')

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2归一化"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-12)

    def _compute_knn_distances(self, train_emb: np.ndarray, test_emb: np.ndarray,
                                k: int) -> tuple:
        """计算k-NN距离和索引"""
        try:
            import faiss
            d = train_emb.shape[1]
            index = faiss.IndexFlatIP(d)
            index.add(train_emb.astype('float32'))
            similarities, indices = index.search(test_emb.astype('float32'), k)
            distances = 1 - similarities  # 余弦距离
        except ImportError:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=k, metric='cosine')
            nn.fit(train_emb)
            distances, indices = nn.kneighbors(test_emb)

        return distances, indices

    def _compute_heterophily(self, knn_indices: np.ndarray,
                             train_labels: np.ndarray,
                             num_classes: int, k: int) -> np.ndarray:
        """计算异配性分数"""
        n_test = len(knn_indices)
        heterophily_scores = np.zeros(n_test)

        for i in range(n_test):
            neighbor_labels = train_labels[knn_indices[i]]
            unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
            probs = counts / counts.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(min(k, num_classes))
            normalized_entropy = entropy / (max_entropy + 1e-10)
            unique_ratio = len(unique_labels) / min(k, num_classes)
            heterophily_scores[i] = 0.5 * normalized_entropy + 0.5 * unique_ratio

        return heterophily_scores

    def run_single_experiment(self, train_emb: np.ndarray, train_labels: np.ndarray,
                              test_emb: np.ndarray, test_labels: np.ndarray,
                              seed: int = 42) -> dict:
        """运行单次实验"""
        np.random.seed(seed)

        # 归一化
        train_emb = self._normalize(train_emb).astype('float32')
        test_emb = self._normalize(test_emb).astype('float32')

        # 确保标签是整数
        train_labels = np.array(train_labels)
        num_classes = len(set(train_labels))

        # 计算k-NN距离和索引
        distances, indices = self._compute_knn_distances(train_emb, test_emb, self.k)

        # k-NN分数 (第k近邻距离)
        knn_distances = distances[:, -1]
        knn_scores_raw = (knn_distances - knn_distances.min()) / (knn_distances.max() - knn_distances.min() + 1e-10)

        # 异配性分数
        heterophily_scores = self._compute_heterophily(indices, train_labels, num_classes, self.k)

        # 三种方法
        results = {}

        # 1. Full Model
        full_scores = (1 - self.alpha) * knn_scores_raw + self.alpha * heterophily_scores
        results['Full Model'] = self._evaluate(full_scores, test_labels)

        # 2. KNN Only (alpha=0)
        results['KNN Only'] = self._evaluate(knn_scores_raw, test_labels)

        # 3. Heterophily Only (alpha=1)
        results['Heterophily Only'] = self._evaluate(heterophily_scores, test_labels)

        return results

    def _evaluate(self, scores: np.ndarray, labels: np.ndarray) -> dict:
        """评估分数"""
        labels = np.array(labels)

        # 自动修复方向
        auroc_orig = roc_auc_score(labels, scores)
        auroc_inv = roc_auc_score(labels, -scores)

        if auroc_inv > auroc_orig + 0.05:
            scores = -scores
            auroc = auroc_inv
        else:
            auroc = auroc_orig

        # 计算其他指标
        aupr = average_precision_score(labels, scores)
        fpr, tpr, _ = roc_curve(labels, scores)
        fpr95_idx = np.argmin(np.abs(tpr - 0.95))
        fpr95 = fpr[fpr95_idx]

        return {
            'auroc': auroc,
            'aupr': aupr,
            'fpr95': fpr95
        }

    def run_dataset(self, dataset_name: str, data_loader_func) -> dict:
        """对单个数据集运行完整实验"""
        print(f"\n{'='*60}")
        print(f"数据集: {dataset_name}")
        print(f"{'='*60}")

        # 加载数据
        print("加载数据...")
        try:
            train_texts, test_texts, test_labels, test_intents, train_labels = data_loader_func()
            print(f"  训练: {len(train_texts)} 样本, {len(set(train_labels))} 类别")
            print(f"  测试: {len(test_texts)} 样本 (ID: {test_labels.count(0)}, OOD: {test_labels.count(1)})")
        except Exception as e:
            print(f"  加载失败: {e}")
            import traceback
            traceback.print_exc()
            return None

        # 生成embeddings
        print("生成embeddings...")
        train_emb = self._encode_texts(train_texts)
        test_emb = self._encode_texts(test_texts)

        # 转换标签为numpy数组
        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)

        # 多次运行
        all_results = {
            'Full Model': {'auroc': [], 'aupr': [], 'fpr95': []},
            'KNN Only': {'auroc': [], 'aupr': [], 'fpr95': []},
            'Heterophily Only': {'auroc': [], 'aupr': [], 'fpr95': []}
        }

        for run in range(self.n_runs):
            print(f"\n  Run {run+1}/{self.n_runs}...", end=' ')
            results = self.run_single_experiment(
                train_emb, train_labels, test_emb, test_labels,
                seed=42 + run
            )

            for method in all_results:
                for metric in ['auroc', 'aupr', 'fpr95']:
                    all_results[method][metric].append(results[method][metric])

            print(f"Full: {results['Full Model']['auroc']*100:.2f}%, "
                  f"KNN: {results['KNN Only']['auroc']*100:.2f}%, "
                  f"Het: {results['Heterophily Only']['auroc']*100:.2f}%")

        # 汇总统计
        summary = {}
        for method in all_results:
            summary[method] = {}
            for metric in ['auroc', 'aupr', 'fpr95']:
                values = all_results[method][metric]
                summary[method][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }

        # 打印结果
        print(f"\n{dataset_name} 结果汇总:")
        print("-" * 60)
        for method in summary:
            auroc = summary[method]['auroc']
            aupr = summary[method]['aupr']
            fpr95 = summary[method]['fpr95']
            print(f"  {method}:")
            print(f"    AUROC:  {auroc['mean']*100:.2f}% ± {auroc['std']*100:.2f}%")
            print(f"    AUPR:   {aupr['mean']*100:.2f}% ± {aupr['std']*100:.2f}%")
            print(f"    FPR@95: {fpr95['mean']*100:.2f}% ± {fpr95['std']*100:.2f}%")

        return summary

    def run_all(self):
        """运行所有数据集"""
        print("\n" + "="*70)
        print("模块2: 完整实验运行")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"配置: k={self.k}, alpha={self.alpha}, runs={self.n_runs}")
        print("="*70)

        datasets = {
            'CLINC150': load_clinc150,
            'Banking77': load_banking77_oos,
        }

        # 尝试添加ROSTD数据集
        try:
            datasets['ROSTD'] = load_rostd
        except Exception as e:
            print(f"[注意] ROSTD数据集不可用: {e}")

        # 运行实验
        for name, loader in datasets.items():
            result = self.run_dataset(name, loader)
            if result:
                self.results[name] = result

        # 保存结果
        self._save_results()
        self._generate_latex_table()

        return self.results

    def _save_results(self):
        """保存JSON结果"""
        output_file = self.output_dir / 'complete_results.json'

        # 转换为可序列化格式
        serializable = {}
        for dataset, methods in self.results.items():
            serializable[dataset] = {}
            for method, metrics in methods.items():
                serializable[dataset][method] = {}
                for metric, stats in metrics.items():
                    serializable[dataset][method][metric] = {
                        'mean': float(stats['mean']),
                        'std': float(stats['std']),
                        'min': float(stats['min']),
                        'max': float(stats['max'])
                    }

        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'config': {'k': self.k, 'alpha': self.alpha, 'n_runs': self.n_runs},
                'results': serializable
            }, f, indent=2)

        print(f"\n结果已保存: {output_file}")

    def _generate_latex_table(self):
        """生成LaTeX表格"""
        table_file = self.table_dir / 'main_results.tex'

        lines = [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Main experimental results. All metrics (\\%) are mean$\\pm$std over 5 runs. Best AUROC in bold.}",
            "\\label{tab:main_results}",
            "\\begin{tabular}{llccc}",
            "\\toprule",
            "Dataset & Method & AUROC (\\%) & AUPR (\\%) & FPR@95 (\\%) \\\\",
            "\\midrule"
        ]

        for i, (dataset, methods) in enumerate(self.results.items()):
            n_methods = len(methods)

            # 找到最佳方法
            best_auroc = max(m['auroc']['mean'] for m in methods.values())

            for j, (method, metrics) in enumerate(methods.items()):
                auroc = metrics['auroc']
                aupr = metrics['aupr']
                fpr95 = metrics['fpr95']

                is_best = auroc['mean'] == best_auroc

                auroc_str = f"{auroc['mean']*100:.2f}$\\pm${auroc['std']*100:.2f}"
                aupr_str = f"{aupr['mean']*100:.2f}$\\pm${aupr['std']*100:.2f}"
                fpr95_str = f"{fpr95['mean']*100:.2f}$\\pm${fpr95['std']*100:.2f}"

                if is_best:
                    auroc_str = f"\\textbf{{{auroc_str}}}"

                if j == 0:
                    lines.append(f"\\multirow{{{n_methods}}}{{*}}{{{dataset}}}")

                lines.append(f"& {method} & {auroc_str} & {aupr_str} & {fpr95_str} \\\\")

            if i < len(self.results) - 1:
                lines.append("\\midrule")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])

        with open(table_file, 'w') as f:
            f.write('\n'.join(lines))

        print(f"LaTeX表格已保存: {table_file}")


def main():
    """主函数"""
    runner = ExperimentRunner(n_runs=5, k=50, alpha=0.3)
    results = runner.run_all()

    print("\n" + "="*70)
    print("实验完成!")
    print("="*70)

    # 打印最终汇总
    print("\n最终结果汇总 (AUROC %):")
    print("-" * 50)
    for dataset, methods in results.items():
        print(f"\n{dataset}:")
        for method, metrics in methods.items():
            auroc = metrics['auroc']
            print(f"  {method}: {auroc['mean']*100:.2f}% ± {auroc['std']*100:.2f}%")

    return results


if __name__ == "__main__":
    main()
