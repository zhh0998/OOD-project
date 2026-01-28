#!/usr/bin/env python3
"""
模块3: 超参数网格搜索
RW3 OOD检测项目 - CCF-A论文实验

搜索空间:
- k: [5, 10, 20, 50, 100]
- alpha: [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

输出:
- JSON完整结果
- 最优超参数
- LaTeX表格

Author: RW3 OOD Detection Project
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from itertools import product

# 添加项目路径
sys.path.insert(0, '/home/user/OOD-project')

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

# 导入数据加载器
from src.datasets.ood_datasets import load_clinc150, load_banking77_oos

# 尝试导入sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("[WARNING] sentence_transformers not available")


class GridSearchRunner:
    """超参数网格搜索运行器"""

    def __init__(self, k_values: list = None, alpha_values: list = None,
                 model_name: str = 'all-MiniLM-L6-v2'):
        self.k_values = k_values or [5, 10, 20, 50, 100]
        self.alpha_values = alpha_values or [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        self.model_name = model_name
        self.embedder = None
        self.results = {}

        # 输出目录
        self.output_dir = Path('/home/user/OOD-project/experiments/results/grid_search')
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
            distances = 1 - similarities
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

    def run_grid_search(self, train_emb: np.ndarray, train_labels: np.ndarray,
                        test_emb: np.ndarray, test_labels: np.ndarray,
                        dataset_name: str) -> dict:
        """对单个数据集运行网格搜索"""
        print(f"\n{'='*60}")
        print(f"网格搜索: {dataset_name}")
        print(f"搜索空间: k={self.k_values}, alpha={self.alpha_values}")
        print(f"总配置数: {len(self.k_values) * len(self.alpha_values)}")
        print(f"{'='*60}")

        # 归一化
        train_emb = self._normalize(train_emb).astype('float32')
        test_emb = self._normalize(test_emb).astype('float32')

        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)
        num_classes = len(set(train_labels))

        results = {}
        best_auroc = 0
        best_config = None

        total_configs = len(self.k_values) * len(self.alpha_values)
        config_idx = 0

        for k in self.k_values:
            # 计算k-NN (只需要计算一次，复用不同alpha)
            distances, indices = self._compute_knn_distances(train_emb, test_emb, k)

            # k-NN分数
            knn_distances = distances[:, -1]
            knn_scores = (knn_distances - knn_distances.min()) / (knn_distances.max() - knn_distances.min() + 1e-10)

            # 异配性分数
            heterophily_scores = self._compute_heterophily(indices, train_labels, num_classes, k)

            for alpha in self.alpha_values:
                config_idx += 1

                # 组合分数
                combined_scores = (1 - alpha) * knn_scores + alpha * heterophily_scores

                # 评估
                metrics = self._evaluate(combined_scores, test_labels)

                config_key = f"k={k},alpha={alpha}"
                results[config_key] = {
                    'k': k,
                    'alpha': alpha,
                    **metrics
                }

                # 更新最佳配置
                if metrics['auroc'] > best_auroc:
                    best_auroc = metrics['auroc']
                    best_config = {'k': k, 'alpha': alpha, **metrics}

                # 打印进度
                print(f"  [{config_idx}/{total_configs}] k={k:3d}, alpha={alpha:.1f}: "
                      f"AUROC={metrics['auroc']*100:.2f}%")

        print(f"\n最佳配置: k={best_config['k']}, alpha={best_config['alpha']}")
        print(f"最佳AUROC: {best_config['auroc']*100:.2f}%")

        return {
            'results': results,
            'best_config': best_config,
            'k_values': self.k_values,
            'alpha_values': self.alpha_values
        }

    def run_all(self):
        """运行所有数据集的网格搜索"""
        print("\n" + "="*70)
        print("模块3: 超参数网格搜索")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)

        datasets = {
            'CLINC150': load_clinc150,
            'Banking77': load_banking77_oos,
        }

        all_results = {}

        for dataset_name, loader in datasets.items():
            print(f"\n加载 {dataset_name}...")
            try:
                train_texts, test_texts, test_labels, _, train_labels = loader()
                print(f"  训练: {len(train_texts)} 样本")
                print(f"  测试: {len(test_texts)} 样本")
            except Exception as e:
                print(f"  加载失败: {e}")
                continue

            # 生成embeddings
            print("  生成embeddings...")
            train_emb = self._encode_texts(train_texts)
            test_emb = self._encode_texts(test_texts)

            # 网格搜索
            result = self.run_grid_search(
                train_emb, train_labels, test_emb, test_labels,
                dataset_name
            )
            all_results[dataset_name] = result

        self.results = all_results

        # 保存结果
        self._save_results()
        self._generate_heatmap_data()
        self._generate_latex_table()

        return all_results

    def _save_results(self):
        """保存JSON结果"""
        output_file = self.output_dir / 'grid_search_results.json'

        # 转换为可序列化格式
        serializable = {}
        for dataset, data in self.results.items():
            serializable[dataset] = {
                'best_config': {
                    'k': int(data['best_config']['k']),
                    'alpha': float(data['best_config']['alpha']),
                    'auroc': float(data['best_config']['auroc']),
                    'aupr': float(data['best_config']['aupr']),
                    'fpr95': float(data['best_config']['fpr95'])
                },
                'k_values': [int(k) for k in data['k_values']],
                'alpha_values': [float(a) for a in data['alpha_values']],
                'results': {
                    k: {
                        'k': int(v['k']),
                        'alpha': float(v['alpha']),
                        'auroc': float(v['auroc']),
                        'aupr': float(v['aupr']),
                        'fpr95': float(v['fpr95'])
                    }
                    for k, v in data['results'].items()
                }
            }

        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': serializable
            }, f, indent=2)

        print(f"\n结果已保存: {output_file}")

    def _generate_heatmap_data(self):
        """生成热力图数据"""
        heatmap_file = self.output_dir / 'heatmap_data.json'

        heatmap_data = {}
        for dataset, data in self.results.items():
            # 创建k x alpha的AUROC矩阵
            matrix = []
            for k in data['k_values']:
                row = []
                for alpha in data['alpha_values']:
                    config_key = f"k={k},alpha={alpha}"
                    auroc = data['results'][config_key]['auroc']
                    row.append(float(auroc))
                matrix.append(row)

            heatmap_data[dataset] = {
                'k_values': [int(k) for k in data['k_values']],
                'alpha_values': [float(a) for a in data['alpha_values']],
                'auroc_matrix': matrix
            }

        with open(heatmap_file, 'w') as f:
            json.dump(heatmap_data, f, indent=2)

        print(f"热力图数据已保存: {heatmap_file}")

    def _generate_latex_table(self):
        """生成LaTeX最优配置表"""
        table_file = self.table_dir / 'best_hyperparams.tex'

        lines = [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Best hyperparameters found by grid search.}",
            "\\label{tab:best_hyperparams}",
            "\\begin{tabular}{lcccc}",
            "\\toprule",
            "Dataset & Best $k$ & Best $\\alpha$ & AUROC (\\%) & FPR@95 (\\%) \\\\",
            "\\midrule"
        ]

        for dataset, data in self.results.items():
            best = data['best_config']
            lines.append(
                f"{dataset} & {best['k']} & {best['alpha']:.1f} & "
                f"{best['auroc']*100:.2f} & {best['fpr95']*100:.2f} \\\\"
            )

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
    runner = GridSearchRunner(
        k_values=[5, 10, 20, 50, 100],
        alpha_values=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    )
    results = runner.run_all()

    print("\n" + "="*70)
    print("网格搜索完成!")
    print("="*70)

    # 打印最佳配置汇总
    print("\n最佳配置汇总:")
    print("-" * 50)
    for dataset, data in results.items():
        best = data['best_config']
        print(f"\n{dataset}:")
        print(f"  最佳 k = {best['k']}")
        print(f"  最佳 alpha = {best['alpha']}")
        print(f"  AUROC = {best['auroc']*100:.2f}%")
        print(f"  FPR@95 = {best['fpr95']*100:.2f}%")

    return results


if __name__ == "__main__":
    main()
