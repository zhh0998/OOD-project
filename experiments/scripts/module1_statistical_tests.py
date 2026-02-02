#!/usr/bin/env python3
"""
模块1: 统计显著性检验
RW3 OOD检测项目 - CCF-A论文实验

检验内容:
1. Paired t-test (Full Model vs KNN Only)
2. Cohen's d 效应量
3. Pearson/Spearman 相关性 (异配性 vs OOD标签)

输出:
- JSON完整结果
- LaTeX表格

Author: RW3 OOD Detection Project
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy import stats

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import roc_auc_score

# 导入数据加载器
from src.datasets.ood_datasets import load_clinc150, load_banking77_oos

# 尝试导入sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("[WARNING] sentence_transformers not available")


class StatisticalTestRunner:
    """统计显著性检验运行器"""

    def __init__(self, k: int = 50, alpha: float = 0.3, n_bootstrap: int = 1000,
                 model_name: str = 'all-MiniLM-L6-v2'):
        self.k = k
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.model_name = model_name
        self.embedder = None
        self.results = {}

        # 输出目录
        self.output_dir = PROJECT_ROOT / "experiments" / "results" / "statistical"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.table_dir = PROJECT_ROOT / "experiments" / "tables"
        self.table_dir.mkdir(parents=True, exist_ok=True)

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

    def _compute_knn_and_heterophily(self, train_emb, train_labels, test_emb, k):
        """计算k-NN距离和异配性分数"""
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

        return knn_scores, heterophily_scores

    def _bootstrap_auroc(self, labels, scores, n_bootstrap=1000):
        """Bootstrap AUROC采样"""
        aurocs = []
        n = len(labels)
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            try:
                auroc = roc_auc_score(labels[idx], scores[idx])
                aurocs.append(auroc)
            except:
                pass
        return np.array(aurocs)

    def _cohens_d(self, group1, group2):
        """计算Cohen's d效应量"""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        return (np.mean(group1) - np.mean(group2)) / pooled_std

    def run_statistical_tests(self, dataset_name, train_emb, train_labels, test_emb, test_labels):
        """运行统计检验"""
        print(f"\n{'='*60}")
        print(f"统计检验: {dataset_name}")
        print(f"{'='*60}")

        test_labels = np.array(test_labels)

        # 计算分数
        print("计算OOD分数...")
        knn_scores, heterophily_scores = self._compute_knn_and_heterophily(
            train_emb, train_labels, test_emb, self.k
        )

        # Full Model分数
        full_scores = (1 - self.alpha) * knn_scores + self.alpha * heterophily_scores

        # 自动修复分数方向
        for name, scores in [('full', full_scores), ('knn', knn_scores), ('het', heterophily_scores)]:
            auroc_orig = roc_auc_score(test_labels, scores)
            auroc_inv = roc_auc_score(test_labels, -scores)
            if auroc_inv > auroc_orig + 0.05:
                if name == 'full':
                    full_scores = -full_scores
                elif name == 'knn':
                    knn_scores = -knn_scores
                else:
                    heterophily_scores = -heterophily_scores

        # 1. Bootstrap AUROC
        print("Bootstrap AUROC采样...")
        full_aurocs = self._bootstrap_auroc(test_labels, full_scores, self.n_bootstrap)
        knn_aurocs = self._bootstrap_auroc(test_labels, knn_scores, self.n_bootstrap)
        het_aurocs = self._bootstrap_auroc(test_labels, heterophily_scores, self.n_bootstrap)

        # 2. Paired t-test
        print("Paired t-test...")
        t_stat, p_value = stats.ttest_rel(full_aurocs, knn_aurocs)

        # 3. Cohen's d
        cohens_d = self._cohens_d(full_aurocs, knn_aurocs)

        # 4. 相关性分析 (异配性 vs OOD标签)
        print("相关性分析...")
        pearson_r, pearson_p = stats.pearsonr(heterophily_scores, test_labels)
        spearman_r, spearman_p = stats.spearmanr(heterophily_scores, test_labels)

        # 5. ID vs OOD异配性均值
        id_mask = test_labels == 0
        ood_mask = test_labels == 1
        id_het_mean = heterophily_scores[id_mask].mean()
        ood_het_mean = heterophily_scores[ood_mask].mean()

        # 汇总结果
        results = {
            'auroc': {
                'full_model': {'mean': full_aurocs.mean(), 'std': full_aurocs.std()},
                'knn_only': {'mean': knn_aurocs.mean(), 'std': knn_aurocs.std()},
                'heterophily_only': {'mean': het_aurocs.mean(), 'std': het_aurocs.std()},
                'improvement': full_aurocs.mean() - knn_aurocs.mean()
            },
            'paired_ttest': {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            },
            'effect_size': {
                'cohens_d': cohens_d,
                'interpretation': 'Large' if abs(cohens_d) > 0.8 else ('Medium' if abs(cohens_d) > 0.5 else 'Small')
            },
            'correlation': {
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p
            },
            'heterophily_distribution': {
                'id_mean': id_het_mean,
                'ood_mean': ood_het_mean,
                'difference': ood_het_mean - id_het_mean
            }
        }

        # 打印结果
        print(f"\n结果汇总:")
        print(f"  Full Model AUROC: {full_aurocs.mean()*100:.2f}% ± {full_aurocs.std()*100:.2f}%")
        print(f"  KNN Only AUROC:   {knn_aurocs.mean()*100:.2f}% ± {knn_aurocs.std()*100:.2f}%")
        print(f"  Heterophily Only: {het_aurocs.mean()*100:.2f}% ± {het_aurocs.std()*100:.2f}%")
        print(f"  改进: {(full_aurocs.mean() - knn_aurocs.mean())*100:+.2f}%")
        print(f"\n  Paired t-test: t={t_stat:.3f}, p={p_value:.2e}")
        print(f"  Cohen's d: {cohens_d:.3f} ({results['effect_size']['interpretation']})")
        print(f"  Pearson r: {pearson_r:.3f} (p={pearson_p:.2e})")
        print(f"  Spearman r: {spearman_r:.3f} (p={spearman_p:.2e})")
        print(f"\n  ID异配性均值: {id_het_mean:.4f}")
        print(f"  OOD异配性均值: {ood_het_mean:.4f}")
        print(f"  差异: {ood_het_mean - id_het_mean:+.4f}")

        return results

    def run_all(self):
        """运行所有数据集的统计检验"""
        print("\n" + "="*70)
        print("模块1: 统计显著性检验")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"配置: k={self.k}, alpha={self.alpha}, bootstrap={self.n_bootstrap}")
        print("="*70)

        datasets = {
            'CLINC150': load_clinc150,
            'Banking77': load_banking77_oos,
        }

        all_results = {}

        for dataset_name, loader in datasets.items():
            print(f"\n加载 {dataset_name}...")
            train_texts, test_texts, test_labels, _, train_labels = loader()

            print("生成embeddings...")
            train_emb = self._encode_texts(train_texts)
            test_emb = self._encode_texts(test_texts)

            results = self.run_statistical_tests(
                dataset_name, train_emb, train_labels, test_emb, test_labels
            )
            all_results[dataset_name] = results

        self.results = all_results

        # 保存结果
        self._save_results()
        self._generate_latex_table()

        return all_results

    def _save_results(self):
        """保存JSON结果"""
        output_file = self.output_dir / 'statistical_analysis.json'

        # 转换为可序列化格式
        serializable = {}
        for dataset, data in self.results.items():
            serializable[dataset] = {
                'auroc': {
                    'full_model': {'mean': float(data['auroc']['full_model']['mean']),
                                  'std': float(data['auroc']['full_model']['std'])},
                    'knn_only': {'mean': float(data['auroc']['knn_only']['mean']),
                                'std': float(data['auroc']['knn_only']['std'])},
                    'heterophily_only': {'mean': float(data['auroc']['heterophily_only']['mean']),
                                        'std': float(data['auroc']['heterophily_only']['std'])},
                    'improvement': float(data['auroc']['improvement'])
                },
                'paired_ttest': {
                    't_statistic': float(data['paired_ttest']['t_statistic']),
                    'p_value': float(data['paired_ttest']['p_value']),
                    'significant': bool(data['paired_ttest']['significant'])
                },
                'effect_size': {
                    'cohens_d': float(data['effect_size']['cohens_d']),
                    'interpretation': data['effect_size']['interpretation']
                },
                'correlation': {
                    'pearson_r': float(data['correlation']['pearson_r']),
                    'pearson_p': float(data['correlation']['pearson_p']),
                    'spearman_r': float(data['correlation']['spearman_r']),
                    'spearman_p': float(data['correlation']['spearman_p'])
                },
                'heterophily_distribution': {
                    'id_mean': float(data['heterophily_distribution']['id_mean']),
                    'ood_mean': float(data['heterophily_distribution']['ood_mean']),
                    'difference': float(data['heterophily_distribution']['difference'])
                }
            }

        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'config': {'k': self.k, 'alpha': self.alpha, 'n_bootstrap': self.n_bootstrap},
                'results': serializable
            }, f, indent=2)

        print(f"\n结果已保存: {output_file}")

    def _generate_latex_table(self):
        """生成LaTeX表格"""
        table_file = self.table_dir / 'statistical_tests.tex'

        lines = [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Statistical significance tests. $^{***}$p<0.001, $^{**}$p<0.01, $^{*}$p<0.05.}",
            "\\label{tab:statistical_tests}",
            "\\begin{tabular}{lcccccc}",
            "\\toprule",
            "Dataset & Improvement & $t$-stat & $p$-value & Cohen's $d$ & Pearson $r$ \\\\",
            "\\midrule"
        ]

        for dataset, data in self.results.items():
            imp = data['auroc']['improvement'] * 100
            t_stat = data['paired_ttest']['t_statistic']
            p_val = data['paired_ttest']['p_value']
            cohens_d = data['effect_size']['cohens_d']
            pearson_r = data['correlation']['pearson_r']

            # 显著性标记
            if p_val < 0.001:
                sig = "$^{***}$"
            elif p_val < 0.01:
                sig = "$^{**}$"
            elif p_val < 0.05:
                sig = "$^{*}$"
            else:
                sig = ""

            lines.append(
                f"{dataset} & {imp:+.2f}\\%{sig} & {t_stat:.2f} & "
                f"{p_val:.2e} & {cohens_d:.2f} & {pearson_r:.2f} \\\\"
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
    runner = StatisticalTestRunner(k=10, alpha=0.3, n_bootstrap=1000)
    results = runner.run_all()

    print("\n" + "="*70)
    print("统计检验完成!")
    print("="*70)

    return results


if __name__ == "__main__":
    main()
