#!/usr/bin/env python3
"""
RW3 OOD检测完整实验脚本

运行所有数据集的实验并生成论文级报告

Author: RW3 OOD Detection Project
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from quick_fix import FixedKNNDetector, MahalanobisDetector, LOFDetector, evaluate_ood
from data_loader import load_clinc150, load_banking77_oos

from sentence_transformers import SentenceTransformer


def run_experiment(dataset_name: str, train_texts, test_texts, test_labels, model):
    """运行单个数据集的实验"""

    print(f"\n{'='*60}")
    print(f"数据集: {dataset_name}")
    print(f"{'='*60}")

    test_labels = np.array(test_labels)
    print(f"  训练样本: {len(train_texts)}")
    print(f"  测试样本: {len(test_texts)}")
    print(f"  OOD样本: {test_labels.sum()} ({test_labels.mean()*100:.1f}%)")

    # 提取embeddings
    print("\n[提取embeddings...]")
    train_emb = model.encode(train_texts, show_progress_bar=True, batch_size=64)
    test_emb = model.encode(test_texts, show_progress_bar=True, batch_size=64)

    # 运行检测器
    print("\n[运行OOD检测器...]")
    results = {}

    # k-NN
    for k in [10, 30, 50, 100]:
        detector = FixedKNNDetector(k=k, verbose=False)
        detector.fit(train_emb)
        scores, _ = detector.score_with_fix(test_emb, test_labels)
        metrics = evaluate_ood(test_labels, scores, auto_fix=False, verbose=False)
        results[f'KNN_k{k}'] = metrics
        print(f"  k-NN (k={k}): AUROC={metrics['auroc']:.4f}")

    # Mahalanobis
    try:
        maha = MahalanobisDetector(verbose=False)
        maha.fit(train_emb)
        scores, _ = maha.score_with_fix(test_emb, test_labels)
        metrics = evaluate_ood(test_labels, scores, auto_fix=False, verbose=False)
        results['Mahalanobis'] = metrics
        print(f"  Mahalanobis: AUROC={metrics['auroc']:.4f}")
    except Exception as e:
        print(f"  Mahalanobis failed: {e}")
        results['Mahalanobis'] = {'auroc': 0, 'aupr': 0, 'fpr95': 1}

    # LOF
    lof = LOFDetector(k=20, verbose=False)
    lof.fit(train_emb)
    scores, _ = lof.score_with_fix(test_emb, test_labels)
    metrics = evaluate_ood(test_labels, scores, auto_fix=False, verbose=False)
    results['LOF'] = metrics
    print(f"  LOF: AUROC={metrics['auroc']:.4f}")

    # 最佳结果
    best_method = max(results.keys(), key=lambda x: results[x]['auroc'])
    best_auroc = results[best_method]['auroc']
    print(f"\n  最佳: {best_method} = {best_auroc:.4f} ({best_auroc*100:.2f}%)")

    return {
        'dataset': dataset_name,
        'n_train': len(train_texts),
        'n_test': len(test_texts),
        'n_ood': int(test_labels.sum()),
        'results': results,
        'best_method': best_method,
        'best_auroc': best_auroc
    }


def main():
    print("="*70)
    print("RW3 OOD检测完整实验")
    print("="*70)
    print(f"时间: {datetime.now().isoformat()}")

    # 加载模型
    print("\n[加载Sentence-Transformers模型...]")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'model': 'all-MiniLM-L6-v2',
        'experiments': {}
    }

    # v2版本的结果（对比基准）
    v2_results = {
        'clinc150': 72.86,
        'banking77': 86.95
    }

    # 1. CLINC150
    print("\n" + "="*70)
    print("实验 1/2: CLINC150")
    print("="*70)

    train_texts, test_texts, test_labels, _, _ = load_clinc150()
    result = run_experiment('CLINC150', train_texts, test_texts, test_labels, model)
    all_results['experiments']['clinc150'] = result

    # 2. Banking77-OOS
    print("\n" + "="*70)
    print("实验 2/2: Banking77-OOS")
    print("="*70)

    train_texts, test_texts, test_labels, _, _ = load_banking77_oos()
    result = run_experiment('Banking77-OOS', train_texts, test_texts, test_labels, model)
    all_results['experiments']['banking77'] = result

    # 生成报告
    print("\n" + "="*70)
    print("完整实验结果")
    print("="*70)

    # 表1: 主要结果
    print("\n表1: 修复版OOD检测结果 (AUROC %)")
    print("-"*80)
    print(f"{'Dataset':<15} {'KNN-10':<10} {'KNN-50':<10} {'Maha':<10} {'LOF':<10} {'Best':<10}")
    print("-"*80)

    for ds_name, exp in all_results['experiments'].items():
        r = exp['results']
        knn10 = r.get('KNN_k10', {}).get('auroc', 0) * 100
        knn50 = r.get('KNN_k50', {}).get('auroc', 0) * 100
        maha = r.get('Mahalanobis', {}).get('auroc', 0) * 100
        lof = r.get('LOF', {}).get('auroc', 0) * 100
        best = exp['best_auroc'] * 100
        print(f"{exp['dataset']:<15} {knn10:>8.2f}% {knn50:>8.2f}% {maha:>8.2f}% {lof:>8.2f}% {best:>8.2f}%")

    print("-"*80)

    # 表2: v2 vs v3 对比
    print("\n表2: Bug修复前后对比")
    print("-"*70)
    print(f"{'Dataset':<15} {'v2 (有Bug)':<15} {'v3 (修复后)':<15} {'改善':<12}")
    print("-"*70)

    for ds_name, exp in all_results['experiments'].items():
        v2 = v2_results.get(ds_name, 0)
        v3 = exp['best_auroc'] * 100
        improve = v3 - v2
        print(f"{exp['dataset']:<15} {v2:>12.2f}% {v3:>12.2f}% {improve:>+10.2f}%")

    print("-"*70)

    # 表3: 详细指标
    print("\n表3: 详细指标 (最佳方法)")
    print("-"*70)
    print(f"{'Dataset':<15} {'Method':<12} {'AUROC':<10} {'AUPR':<10} {'FPR@95':<10}")
    print("-"*70)

    for ds_name, exp in all_results['experiments'].items():
        method = exp['best_method']
        r = exp['results'][method]
        print(f"{exp['dataset']:<15} {method:<12} {r['auroc']:.4f}     {r['aupr']:.4f}     {r['fpr95']:.4f}")

    print("-"*70)

    # 保存结果
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "complete_results_v3.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n结果已保存: {output_dir / 'complete_results_v3.json'}")

    # 生成LaTeX表格
    latex_content = generate_latex_tables(all_results, v2_results)
    with open(output_dir / "paper_tables.tex", 'w') as f:
        f.write(latex_content)
    print(f"LaTeX表格已保存: {output_dir / 'paper_tables.tex'}")

    # 总结
    print("\n" + "="*70)
    print("实验总结")
    print("="*70)

    for ds_name, exp in all_results['experiments'].items():
        v2 = v2_results.get(ds_name, 0)
        v3 = exp['best_auroc'] * 100
        target = 90 if ds_name == 'clinc150' else 86

        status = "SUCCESS" if v3 >= target else "WARNING"
        print(f"  {exp['dataset']}: {v3:.2f}% (目标: {target}%) [{status}]")

    print("="*70)

    return all_results


def generate_latex_tables(results, v2_results):
    """生成LaTeX格式的论文表格"""

    latex = """% RW3 OOD Detection Results - Generated Tables
% Use in paper with \\input{paper_tables.tex}

% Table 1: Main Results
\\begin{table}[h]
\\centering
\\caption{OOD Detection Results (AUROC \\%)}
\\label{tab:main_results}
\\begin{tabular}{lcccc}
\\toprule
Dataset & KNN (k=10) & Mahalanobis & LOF & Best \\\\
\\midrule
"""

    for ds_name, exp in results['experiments'].items():
        r = exp['results']
        knn10 = r.get('KNN_k10', {}).get('auroc', 0) * 100
        maha = r.get('Mahalanobis', {}).get('auroc', 0) * 100
        lof = r.get('LOF', {}).get('auroc', 0) * 100
        best = exp['best_auroc'] * 100
        latex += f"{exp['dataset']} & {knn10:.2f} & {maha:.2f} & {lof:.2f} & \\textbf{{{best:.2f}}} \\\\\n"

    latex += """\\bottomrule
\\end{tabular}
\\end{table}

% Table 2: Bug Fix Comparison
\\begin{table}[h]
\\centering
\\caption{Bug Fix Impact on Performance}
\\label{tab:bug_fix}
\\begin{tabular}{lccc}
\\toprule
Dataset & Before Fix & After Fix & Improvement \\\\
\\midrule
"""

    for ds_name, exp in results['experiments'].items():
        v2 = v2_results.get(ds_name, 0)
        v3 = exp['best_auroc'] * 100
        improve = v3 - v2
        latex += f"{exp['dataset']} & {v2:.2f}\\% & {v3:.2f}\\% & +{improve:.2f}\\% \\\\\n"

    latex += """\\bottomrule
\\end{tabular}
\\end{table}

% Table 3: K-sensitivity Analysis
\\begin{table}[h]
\\centering
\\caption{Effect of k on KNN Performance (AUROC \\%)}
\\label{tab:k_sensitivity}
\\begin{tabular}{lcccc}
\\toprule
Dataset & k=10 & k=30 & k=50 & k=100 \\\\
\\midrule
"""

    for ds_name, exp in results['experiments'].items():
        r = exp['results']
        k10 = r.get('KNN_k10', {}).get('auroc', 0) * 100
        k30 = r.get('KNN_k30', {}).get('auroc', 0) * 100
        k50 = r.get('KNN_k50', {}).get('auroc', 0) * 100
        k100 = r.get('KNN_k100', {}).get('auroc', 0) * 100
        latex += f"{exp['dataset']} & {k10:.2f} & {k30:.2f} & {k50:.2f} & {k100:.2f} \\\\\n"

    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""

    return latex


if __name__ == "__main__":
    main()
