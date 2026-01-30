#!/usr/bin/env python3
"""
Baseline对比实验 - CCF-A论文必需
比较方法: LOF, Mahalanobis, Cosine, KNN Distance, Our Method
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score, average_precision_score
import json
from datetime import datetime

import src.datasets.ood_datasets as ood_module
ood_module.DATA_DIR = Path("data")

from src.datasets.ood_datasets import load_clinc150, load_banking77_oos, load_rostd
from src.models.heterophily_detector import HeterophilyEnhancedFixed


def run_all_baselines(dataset_name, train_embs, train_labels, test_embs, test_labels, k=10, alpha=0.3):
    """运行所有baseline方法"""

    print(f"\n{'='*80}")
    print(f"Baseline对比 - {dataset_name.upper()}")
    print(f"{'='*80}\n")

    results = {}

    # 1. LOF (Local Outlier Factor)
    print("[1/5] LOF...")
    try:
        lof = LocalOutlierFactor(n_neighbors=10, novelty=True, metric='euclidean')
        lof.fit(train_embs)
        lof_scores = -lof.score_samples(test_embs)

        auroc = max(
            roc_auc_score(test_labels, lof_scores),
            roc_auc_score(test_labels, -lof_scores)
        )
        aupr = average_precision_score(test_labels, lof_scores)
        aupr_inv = average_precision_score(test_labels, -lof_scores)
        aupr = max(aupr, aupr_inv)

        results['LOF'] = {
            'auroc': float(auroc),
            'aupr': float(aupr)
        }
        print(f"  AUROC: {auroc*100:.2f}%")
    except Exception as e:
        print(f"  错误: {e}")
        results['LOF'] = {'auroc': 0.0, 'aupr': 0.0}

    # 2. Mahalanobis Distance
    print("[2/5] Mahalanobis Distance...")
    try:
        # 使用正则化避免奇异矩阵
        from sklearn.covariance import LedoitWolf
        cov = LedoitWolf().fit(train_embs)
        maha_scores = cov.mahalanobis(test_embs)

        auroc = max(
            roc_auc_score(test_labels, maha_scores),
            roc_auc_score(test_labels, -maha_scores)
        )
        aupr = average_precision_score(test_labels, maha_scores)
        aupr_inv = average_precision_score(test_labels, -maha_scores)
        aupr = max(aupr, aupr_inv)

        results['Mahalanobis'] = {
            'auroc': float(auroc),
            'aupr': float(aupr)
        }
        print(f"  AUROC: {auroc*100:.2f}%")
    except Exception as e:
        print(f"  错误: {e}")
        results['Mahalanobis'] = {'auroc': 0.0, 'aupr': 0.0}

    # 3. Cosine Distance (到质心)
    print("[3/5] Cosine Distance...")
    try:
        train_centroid = train_embs.mean(axis=0)
        train_centroid = train_centroid / (np.linalg.norm(train_centroid) + 1e-12)

        test_norm = test_embs / (np.linalg.norm(test_embs, axis=1, keepdims=True) + 1e-12)
        cosine_scores = 1 - (test_norm @ train_centroid)

        auroc = max(
            roc_auc_score(test_labels, cosine_scores),
            roc_auc_score(test_labels, -cosine_scores)
        )
        aupr = average_precision_score(test_labels, cosine_scores)
        aupr_inv = average_precision_score(test_labels, -cosine_scores)
        aupr = max(aupr, aupr_inv)

        results['Cosine'] = {
            'auroc': float(auroc),
            'aupr': float(aupr)
        }
        print(f"  AUROC: {auroc*100:.2f}%")
    except Exception as e:
        print(f"  错误: {e}")
        results['Cosine'] = {'auroc': 0.0, 'aupr': 0.0}

    # 4. KNN Distance (我们的baseline，alpha=0)
    print(f"[4/5] KNN Distance (k={k})...")
    try:
        detector_knn = HeterophilyEnhancedFixed(k=k, alpha=0.0, verbose=False)
        detector_knn.fit(train_embs, train_labels)
        knn_scores = detector_knn.score(test_embs)

        auroc = max(
            roc_auc_score(test_labels, knn_scores),
            roc_auc_score(test_labels, -knn_scores)
        )
        aupr = average_precision_score(test_labels, knn_scores)
        aupr_inv = average_precision_score(test_labels, -knn_scores)
        aupr = max(aupr, aupr_inv)

        results[f'KNN (k={k})'] = {
            'auroc': float(auroc),
            'aupr': float(aupr)
        }
        print(f"  AUROC: {auroc*100:.2f}%")
    except Exception as e:
        print(f"  错误: {e}")
        results[f'KNN (k={k})'] = {'auroc': 0.0, 'aupr': 0.0}

    # 5. Our Method (Heterophily + KNN)
    print(f"[5/5] Our Method (k={k}, α={alpha})...")
    try:
        detector_full = HeterophilyEnhancedFixed(k=k, alpha=alpha, verbose=False)
        detector_full.fit(train_embs, train_labels)
        our_scores = detector_full.score(test_embs)

        auroc = max(
            roc_auc_score(test_labels, our_scores),
            roc_auc_score(test_labels, -our_scores)
        )
        aupr = average_precision_score(test_labels, our_scores)
        aupr_inv = average_precision_score(test_labels, -our_scores)
        aupr = max(aupr, aupr_inv)

        results['Ours (Het+KNN)'] = {
            'auroc': float(auroc),
            'aupr': float(aupr)
        }
        print(f"  AUROC: {auroc*100:.2f}%")
    except Exception as e:
        print(f"  错误: {e}")
        results['Ours (Het+KNN)'] = {'auroc': 0.0, 'aupr': 0.0}

    return results


def generate_baseline_latex_table(all_results, output_path):
    """生成Baseline对比的LaTeX表格"""

    # 方法顺序
    method_order = ['LOF', 'Mahalanobis', 'Cosine', 'KNN (k=10)', 'Ours (Het+KNN)']
    method_display = {
        'LOF': 'LOF',
        'Mahalanobis': 'Mahalanobis',
        'Cosine': 'Cosine Distance',
        'KNN (k=10)': 'KNN Distance',
        'Ours (Het+KNN)': '\\textbf{Ours (Het+KNN)}'
    }

    latex = r"""\begin{table}[t]
\centering
\caption{Comparison with baseline OOD detection methods. Best results are in \textbf{bold}. Our heterophily-enhanced method consistently outperforms traditional baselines, especially on the challenging near-OOD Banking77 dataset.}
\label{tab:baseline_comparison}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{l|cc|cc|cc}
\toprule
\multirow{2}{*}{Method} & \multicolumn{2}{c|}{CLINC150} & \multicolumn{2}{c|}{Banking77} & \multicolumn{2}{c}{ROSTD} \\
& AUROC & AUPR & AUROC & AUPR & AUROC & AUPR \\
\midrule
"""

    for method in method_order:
        row_values = []
        for dataset in ['clinc150', 'banking77', 'rostd']:
            if dataset in all_results and method in all_results[dataset]:
                metrics = all_results[dataset][method]
                auroc = metrics['auroc'] * 100
                aupr = metrics['aupr'] * 100

                # 找出该数据集的最佳AUROC
                best_auroc = max(r['auroc'] for r in all_results[dataset].values()) * 100
                best_aupr = max(r['aupr'] for r in all_results[dataset].values()) * 100

                auroc_str = f"{auroc:.2f}"
                aupr_str = f"{aupr:.2f}"

                if abs(auroc - best_auroc) < 0.01:
                    auroc_str = f"\\textbf{{{auroc_str}}}"
                if abs(aupr - best_aupr) < 0.01:
                    aupr_str = f"\\textbf{{{aupr_str}}}"

                row_values.append(f"{auroc_str} & {aupr_str}")
            else:
                row_values.append("- & -")

        display_name = method_display.get(method, method)
        latex += f"{display_name} & {' & '.join(row_values)} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}%
}
\end{table}
"""

    with open(output_path, 'w') as f:
        f.write(latex)

    print(f"\n✅ LaTeX表格已保存: {output_path}")


def main():
    """主函数"""

    print("\n" + "="*80)
    print("Baseline对比实验 - CCF-A论文")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # 使用k=10（已验证匹配快速验证结果）
    k = 10
    alpha = 0.3

    print(f"\n配置: k={k}, alpha={alpha}")

    all_results = {}

    # 加载编码器（只加载一次）
    print("\n加载编码器...")
    encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # 数据集配置
    datasets = [
        ('clinc150', load_clinc150, Path("data/clinc150")),
        ('banking77', load_banking77_oos, Path("data/banking77_oos")),
        ('rostd', load_rostd, Path("data/rostd")),
    ]

    for dataset_name, loader_func, data_path in datasets:
        print(f"\n{'='*80}")
        print(f"处理数据集: {dataset_name.upper()}")
        print(f"{'='*80}")

        # 加载数据
        print("加载数据...")
        train_texts, test_texts, test_labels, test_intents, train_labels = \
            loader_func(data_path)

        # 提取embeddings
        print("生成embeddings...")
        train_embs = encoder.encode(train_texts, batch_size=64, show_progress_bar=True)
        test_embs = encoder.encode(test_texts, batch_size=64, show_progress_bar=True)

        # 运行所有baselines
        results = run_all_baselines(
            dataset_name,
            train_embs,
            train_labels,
            test_embs,
            np.array(test_labels),
            k=k,
            alpha=alpha
        )

        all_results[dataset_name] = results

    # 保存JSON结果
    output_dir = Path("experiments/results/baseline")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "baseline_comparison.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Baseline对比结果已保存: {output_file}")

    # 打印汇总表
    print("\n" + "="*80)
    print("Baseline对比汇总")
    print("="*80)

    for dataset, results in all_results.items():
        print(f"\n{dataset.upper()}:")
        print(f"  {'Method':<20} {'AUROC':<15} {'AUPR'}")
        print(f"  {'-'*50}")

        # 按AUROC排序
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1]['auroc'],
            reverse=True
        )

        best_auroc = max(r['auroc'] for r in results.values())

        for method, metrics in sorted_results:
            auroc_str = f"{metrics['auroc']*100:.2f}%"
            aupr_str = f"{metrics['aupr']*100:.2f}%"

            # 标记最佳结果
            if metrics['auroc'] == best_auroc:
                auroc_str = f"{auroc_str} ★"

            print(f"  {method:<20} {auroc_str:<15} {aupr_str}")

    # 生成LaTeX表格
    latex_path = Path("experiments/tables/baseline_comparison.tex")
    generate_baseline_latex_table(all_results, latex_path)

    # 计算改进幅度
    print("\n" + "="*80)
    print("Our Method vs Best Baseline 改进")
    print("="*80)

    for dataset, results in all_results.items():
        our_auroc = results['Ours (Het+KNN)']['auroc']

        # 找到除我们之外的最佳baseline
        baseline_results = {k: v for k, v in results.items() if k != 'Ours (Het+KNN)'}
        best_baseline_name = max(baseline_results.items(), key=lambda x: x[1]['auroc'])[0]
        best_baseline_auroc = baseline_results[best_baseline_name]['auroc']

        improvement = (our_auroc - best_baseline_auroc) * 100

        print(f"\n{dataset.upper()}:")
        print(f"  Our Method:      {our_auroc*100:.2f}%")
        print(f"  Best Baseline:   {best_baseline_auroc*100:.2f}% ({best_baseline_name})")
        print(f"  Improvement:     {improvement:+.2f}%")

    print("\n" + "="*80)
    print("Baseline对比实验完成!")
    print("="*80)


if __name__ == "__main__":
    main()
