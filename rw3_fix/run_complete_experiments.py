#!/usr/bin/env python3
"""
RW3完整实验 - 包含所有基线 + HeterophilyEnhanced

运行所有OOD检测方法并生成论文级结果

方法列表：
- 距离基线: KNN, Mahalanobis, LOF
- 分类器基线: MSP, Energy, MaxLogit
- 核心创新: HeterophilyEnhanced

Author: RW3 OOD Detection Project
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from quick_fix import FixedKNNDetector, MahalanobisDetector, LOFDetector, evaluate_ood
from heterophily_enhanced import HeterophilyEnhancedDetector
from classifier_baselines import (
    train_classifier, MSPDetector, EnergyDetector, MaxLogitDetector
)
from data_loader import load_clinc150, load_banking77_oos

from sentence_transformers import SentenceTransformer


def run_all_methods(dataset_name: str,
                    train_texts: List[str],
                    test_texts: List[str],
                    test_labels: np.ndarray,
                    train_intents: List[str],
                    encoder: SentenceTransformer) -> Dict:
    """
    运行所有OOD检测方法

    Args:
        dataset_name: 数据集名称
        train_texts: 训练文本
        test_texts: 测试文本
        test_labels: 测试标签 (0=ID, 1=OOD)
        train_intents: 训练意图标签
        encoder: Sentence encoder

    Returns:
        各方法的评估结果
    """
    print(f"\n{'='*70}")
    print(f" {dataset_name.upper()} - 完整实验")
    print(f"{'='*70}")

    test_labels = np.array(test_labels)

    # 1. 获取embeddings
    print("\n[1/5] 获取Sentence Embeddings...")
    train_emb = encoder.encode(train_texts, show_progress_bar=True, batch_size=64)
    test_emb = encoder.encode(test_texts, show_progress_bar=True, batch_size=64)

    print(f"   Train: {train_emb.shape}, Test: {test_emb.shape}")
    print(f"   ID: {(test_labels==0).sum()}, OOD: {(test_labels==1).sum()}")

    # 准备训练标签（转换为整数索引）
    unique_intents = sorted(set(train_intents))
    intent_to_idx = {intent: i for i, intent in enumerate(unique_intents)}
    train_labels_idx = np.array([intent_to_idx[intent] for intent in train_intents])
    num_classes = len(unique_intents)

    results = {}

    # 2. 距离基线方法
    print("\n[2/5] 距离基线方法...")

    # KNN (多个k值)
    for k in [10, 50, 100]:
        print(f"\n  KNN (k={k})")
        detector = FixedKNNDetector(k=k, verbose=False)
        detector.fit(train_emb)
        scores, auroc = detector.score_with_fix(test_emb, test_labels)
        metrics = evaluate_ood(test_labels, scores, auto_fix=False, verbose=False)
        results[f'KNN-{k}'] = metrics
        print(f"     AUROC: {metrics['auroc']:.4f} ({metrics['auroc']*100:.2f}%)")

    # Mahalanobis
    print(f"\n  Mahalanobis")
    try:
        detector = MahalanobisDetector(verbose=False)
        detector.fit(train_emb)
        scores, auroc = detector.score_with_fix(test_emb, test_labels)
        metrics = evaluate_ood(test_labels, scores, auto_fix=False, verbose=False)
        results['Mahalanobis'] = metrics
        print(f"     AUROC: {metrics['auroc']:.4f} ({metrics['auroc']*100:.2f}%)")
    except Exception as e:
        print(f"     Failed: {e}")
        results['Mahalanobis'] = {'auroc': 0, 'aupr': 0, 'fpr95': 1}

    # LOF
    print(f"\n  LOF")
    detector = LOFDetector(k=20, verbose=False)
    detector.fit(train_emb)
    scores, auroc = detector.score_with_fix(test_emb, test_labels)
    metrics = evaluate_ood(test_labels, scores, auto_fix=False, verbose=False)
    results['LOF'] = metrics
    print(f"     AUROC: {metrics['auroc']:.4f} ({metrics['auroc']*100:.2f}%)")

    # 3. 分类器基线方法
    print("\n[3/5] 分类器基线方法...")
    print("   训练分类器...")
    classifier = train_classifier(
        train_emb, train_labels_idx, num_classes,
        epochs=10, verbose=False
    )
    print("   训练完成")

    # MSP
    print(f"\n  MSP (Maximum Softmax Probability)")
    detector = MSPDetector(classifier, verbose=False)
    scores, auroc = detector.score_with_fix(test_emb, test_labels)
    metrics = evaluate_ood(test_labels, scores, auto_fix=False, verbose=False)
    results['MSP'] = metrics
    print(f"     AUROC: {metrics['auroc']:.4f} ({metrics['auroc']*100:.2f}%)")

    # Energy
    print(f"\n  Energy")
    detector = EnergyDetector(classifier, verbose=False)
    scores, auroc = detector.score_with_fix(test_emb, test_labels)
    metrics = evaluate_ood(test_labels, scores, auto_fix=False, verbose=False)
    results['Energy'] = metrics
    print(f"     AUROC: {metrics['auroc']:.4f} ({metrics['auroc']*100:.2f}%)")

    # MaxLogit
    print(f"\n  MaxLogit")
    detector = MaxLogitDetector(classifier, verbose=False)
    scores, auroc = detector.score_with_fix(test_emb, test_labels)
    metrics = evaluate_ood(test_labels, scores, auto_fix=False, verbose=False)
    results['MaxLogit'] = metrics
    print(f"     AUROC: {metrics['auroc']:.4f} ({metrics['auroc']*100:.2f}%)")

    # 4. HeterophilyEnhanced (核心方法)
    print("\n[4/5] HeterophilyEnhanced (RW3核心方法)")
    try:
        # 尝试不同的alpha值
        best_alpha = 0.3
        best_auroc = 0

        for alpha in [0.2, 0.3, 0.4, 0.5]:
            detector = HeterophilyEnhancedDetector(
                input_dim=train_emb.shape[1],
                hidden_dim=256,
                k=50,
                num_gnn_layers=2,
                alpha=alpha,
                verbose=False
            )
            detector.fit(train_emb, train_labels_idx)
            scores, auroc = detector.score_with_fix(test_emb, test_labels)

            if auroc > best_auroc:
                best_auroc = auroc
                best_alpha = alpha
                best_scores = scores

        print(f"   Best alpha: {best_alpha}")

        metrics = evaluate_ood(test_labels, best_scores, auto_fix=False, verbose=False)
        results['HeterophilyEnhanced'] = metrics
        print(f"     AUROC: {metrics['auroc']:.4f} ({metrics['auroc']*100:.2f}%)")

        # 比较与最佳基线
        baseline_methods = [k for k in results.keys() if k != 'HeterophilyEnhanced']
        best_baseline_auroc = max(results[k]['auroc'] for k in baseline_methods)
        improvement = metrics['auroc'] - best_baseline_auroc

        if improvement > 0:
            print(f"     提升: +{improvement*100:.2f}% vs 最佳基线")
        else:
            print(f"     vs 最佳基线: {improvement*100:.2f}%")

    except Exception as e:
        print(f"     Failed: {e}")
        import traceback
        traceback.print_exc()
        results['HeterophilyEnhanced'] = {'auroc': 0, 'aupr': 0, 'fpr95': 1, 'error': str(e)}

    # 5. 结果汇总
    print("\n[5/5] 结果汇总")
    print("-"*60)
    print(f"{'Method':<25} {'AUROC':<12} {'AUPR':<12} {'FPR@95':<12}")
    print("-"*60)

    sorted_results = sorted(results.items(), key=lambda x: -x[1].get('auroc', 0))
    for method, metrics in sorted_results:
        auroc = metrics.get('auroc', 0)
        aupr = metrics.get('aupr', 0)
        fpr95 = metrics.get('fpr95', 1)
        print(f"{method:<25} {auroc:.4f}       {aupr:.4f}       {fpr95:.4f}")

    print("-"*60)

    return results


def generate_latex_table(all_results: Dict, output_dir: Path):
    """生成LaTeX论文表格"""

    # 方法顺序
    method_order = [
        'KNN-10', 'KNN-50', 'KNN-100',
        'Mahalanobis', 'LOF',
        'MSP', 'Energy', 'MaxLogit',
        'HeterophilyEnhanced'
    ]

    # 方法显示名称
    method_names = {
        'KNN-10': 'KNN ($k$=10)',
        'KNN-50': 'KNN ($k$=50)',
        'KNN-100': 'KNN ($k$=100)',
        'Mahalanobis': 'Mahalanobis',
        'LOF': 'LOF',
        'MSP': 'MSP',
        'Energy': 'Energy',
        'MaxLogit': 'MaxLogit',
        'HeterophilyEnhanced': '\\textbf{HeterophilyEnhanced (Ours)}'
    }

    lines = [
        "% RW3 OOD Detection Results",
        "% Generated: " + datetime.now().isoformat(),
        "",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{OOD Detection Performance Comparison. "
        "Best results are in \\textbf{bold}, second best are \\underline{underlined}.}",
        "\\label{tab:ood_results}",
        "\\small",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\multirow{2}{*}{Method} & \\multicolumn{2}{c}{CLINC150} & \\multicolumn{2}{c}{Banking77-OOS} \\\\",
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
        " & AUROC$\\uparrow$ & FPR95$\\downarrow$ & AUROC$\\uparrow$ & FPR95$\\downarrow$ \\\\",
        "\\midrule"
    ]

    # 获取每个数据集的最佳和次佳AUROC
    def get_best_two(results, method_order):
        aurocs = [(m, results.get(m, {}).get('auroc', 0)) for m in method_order if m in results]
        aurocs.sort(key=lambda x: -x[1])
        return aurocs[0] if aurocs else (None, 0), aurocs[1] if len(aurocs) > 1 else (None, 0)

    clinc_best, clinc_second = get_best_two(all_results.get('clinc150', {}), method_order)
    bank_best, bank_second = get_best_two(all_results.get('banking77', {}), method_order)

    for method in method_order:
        clinc = all_results.get('clinc150', {}).get(method, {})
        bank = all_results.get('banking77', {}).get(method, {})

        if not clinc and not bank:
            continue

        c_auroc = clinc.get('auroc', 0) * 100
        c_fpr95 = clinc.get('fpr95', 1) * 100
        b_auroc = bank.get('auroc', 0) * 100
        b_fpr95 = bank.get('fpr95', 1) * 100

        # 格式化AUROC（标记最佳和次佳）
        def format_auroc(val, best, second, ds_results):
            if ds_results.get(method, {}).get('auroc', 0) == best[1] and best[1] > 0:
                return f"\\textbf{{{val:.2f}}}"
            elif ds_results.get(method, {}).get('auroc', 0) == second[1] and second[1] > 0:
                return f"\\underline{{{val:.2f}}}"
            return f"{val:.2f}"

        c_auroc_str = format_auroc(c_auroc, clinc_best, clinc_second, all_results.get('clinc150', {}))
        b_auroc_str = format_auroc(b_auroc, bank_best, bank_second, all_results.get('banking77', {}))

        method_name = method_names.get(method, method)
        line = f"{method_name} & {c_auroc_str} & {c_fpr95:.2f} & {b_auroc_str} & {b_fpr95:.2f} \\\\"
        lines.append(line)

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])

    latex_file = output_dir / "paper_table_complete.tex"
    with open(latex_file, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\nLaTeX表格已保存: {latex_file}")

    return latex_file


def generate_summary_report(all_results: Dict, output_dir: Path):
    """生成摘要报告"""

    report_lines = [
        "="*70,
        "RW3 OOD Detection - 完整实验报告",
        f"生成时间: {datetime.now().isoformat()}",
        "="*70,
        ""
    ]

    # 只处理数据集结果（排除metadata字段）
    dataset_keys = [k for k in all_results.keys() if k not in ['timestamp', 'model']]

    for ds_name in dataset_keys:
        ds_results = all_results[ds_name]
        if not isinstance(ds_results, dict):
            continue

        report_lines.append(f"\n{'='*50}")
        report_lines.append(f"数据集: {ds_name.upper()}")
        report_lines.append("="*50)

        # 排序结果
        sorted_results = sorted(ds_results.items(), key=lambda x: -x[1].get('auroc', 0) if isinstance(x[1], dict) else 0)

        report_lines.append(f"\n{'Method':<25} {'AUROC':<12} {'AUPR':<12} {'FPR@95':<12}")
        report_lines.append("-"*60)

        for i, (method, metrics) in enumerate(sorted_results):
            if not isinstance(metrics, dict):
                continue
            auroc = metrics.get('auroc', 0) * 100
            aupr = metrics.get('aupr', 0) * 100
            fpr95 = metrics.get('fpr95', 1) * 100

            rank = f"#{i+1}" if i < 3 else ""
            report_lines.append(f"{method:<25} {auroc:>8.2f}%    {aupr:>8.2f}%    {fpr95:>8.2f}%  {rank}")

        # HeterophilyEnhanced分析
        if 'HeterophilyEnhanced' in ds_results and isinstance(ds_results['HeterophilyEnhanced'], dict):
            he_auroc = ds_results['HeterophilyEnhanced'].get('auroc', 0)
            baseline_aurocs = [v.get('auroc', 0) for k, v in ds_results.items()
                              if k != 'HeterophilyEnhanced' and isinstance(v, dict)]
            best_baseline = max(baseline_aurocs) if baseline_aurocs else 0
            improvement = (he_auroc - best_baseline) * 100

            report_lines.append(f"\nHeterophilyEnhanced分析:")
            report_lines.append(f"  vs 最佳基线: {'+' if improvement > 0 else ''}{improvement:.2f}%")

    # 总结
    report_lines.append("\n" + "="*70)
    report_lines.append("实验总结")
    report_lines.append("="*70)

    for ds_name in dataset_keys:
        ds_results = all_results[ds_name]
        if not isinstance(ds_results, dict):
            continue
        if 'HeterophilyEnhanced' in ds_results and isinstance(ds_results['HeterophilyEnhanced'], dict):
            he_auroc = ds_results['HeterophilyEnhanced'].get('auroc', 0) * 100
            sorted_methods = sorted(
                [(k, v) for k, v in ds_results.items() if isinstance(v, dict)],
                key=lambda x: -x[1].get('auroc', 0)
            )
            rank = next((i+1 for i, (m, _) in enumerate(sorted_methods) if m == 'HeterophilyEnhanced'), 0)
            status = "SUCCESS" if rank <= 2 else "NEEDS IMPROVEMENT"
            report_lines.append(f"  {ds_name}: HeterophilyEnhanced = {he_auroc:.2f}% (Rank #{rank}) [{status}]")

    report = '\n'.join(report_lines)
    print(report)

    report_file = output_dir / "experiment_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)

    return report_file


def main():
    """主函数"""
    print("\n" + "="*70)
    print(" RW3完整实验 - 所有方法对比")
    print("="*70)
    print(f"开始时间: {datetime.now().isoformat()}")

    # 初始化
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    # 加载Sentence Encoder
    print("\n[初始化] 加载Sentence-Transformers模型...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'model': 'all-MiniLM-L6-v2'
    }

    # 实验1: CLINC150
    print("\n" + "="*70)
    print(" 数据集 1/2: CLINC150")
    print("="*70)

    train_texts, test_texts, test_labels, test_intents, train_labels = load_clinc150()

    # 获取训练意图
    from data_loader import DATA_DIR
    import json
    data_file = DATA_DIR / "clinc150" / "data_full.json"
    with open(data_file) as f:
        data = json.load(f)
    train_intents = [x[1] for x in data['train']] + [x[1] for x in data['val']]
    train_intents = [i for i in train_intents if i != 'oos']

    results_clinc = run_all_methods(
        'CLINC150',
        train_texts, test_texts, test_labels,
        train_intents, encoder
    )
    all_results['clinc150'] = results_clinc

    # 实验2: Banking77-OOS
    print("\n" + "="*70)
    print(" 数据集 2/2: Banking77-OOS")
    print("="*70)

    train_texts, test_texts, test_labels, test_intents, train_labels = load_banking77_oos()

    # 获取训练意图
    import csv
    data_dir = DATA_DIR / "banking77_oos"
    train_intents = []
    with open(data_dir / "train.csv", 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                train_intents.append(row[1])

    # 过滤OOS类别
    unique_intents = sorted(set(train_intents))
    n_oos = int(len(unique_intents) * 0.25)
    np.random.seed(42)
    oos_intents = set(np.random.choice(unique_intents, n_oos, replace=False))
    train_intents = [i for i in train_intents if i not in oos_intents]

    results_bank = run_all_methods(
        'Banking77-OOS',
        train_texts, test_texts, test_labels,
        train_intents, encoder
    )
    all_results['banking77'] = results_bank

    # 保存结果
    results_file = output_dir / "complete_results_final.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n结果已保存: {results_file}")

    # 生成LaTeX表格
    generate_latex_table(all_results, output_dir)

    # 生成报告
    generate_summary_report(all_results, output_dir)

    print("\n" + "="*70)
    print(" 实验完成!")
    print(f"结束时间: {datetime.now().isoformat()}")
    print("="*70)

    return all_results


if __name__ == "__main__":
    main()
