#!/usr/bin/env python3
"""
模块5: 失败案例分析
分析False Positives (ID误判为OOD) 和 False Negatives (OOD误判为ID)
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score
import json
from datetime import datetime

import src.datasets.ood_datasets as ood_module
ood_module.DATA_DIR = Path("data")

from src.datasets.ood_datasets import load_clinc150, load_banking77_oos, load_rostd
from src.models.heterophily_detector import HeterophilyEnhancedFixed


def analyze_failures(dataset_name, train_texts, train_labels, test_texts, test_labels,
                     test_intents, train_embs, test_embs, k=10, alpha=0.3):
    """分析失败案例"""

    print(f"\n{'='*80}")
    print(f"失败案例分析 - {dataset_name.upper()}")
    print(f"{'='*80}\n")

    # 创建检测器并训练
    detector = HeterophilyEnhancedFixed(k=k, alpha=alpha, verbose=False)
    detector.fit(train_embs, train_labels)

    # 获取OOD分数
    scores = detector.score(test_embs)

    # 获取异配性分数 (需要先获取KNN indices)
    _, knn_indices = detector._compute_knn_distances(test_embs)
    heterophily_scores = detector._compute_heterophily(test_embs, knn_indices)

    # 确定最佳方向
    auroc_pos = roc_auc_score(test_labels, scores)
    auroc_neg = roc_auc_score(test_labels, -scores)

    if auroc_neg > auroc_pos:
        scores = -scores

    # 使用percentile确定阈值（选择使得FPR=5%的阈值）
    id_scores = scores[np.array(test_labels) == 0]
    threshold = np.percentile(id_scores, 95)

    # 二分类预测
    predictions = (scores > threshold).astype(int)

    # 统计
    test_labels_arr = np.array(test_labels)
    tp = np.sum((predictions == 1) & (test_labels_arr == 1))
    tn = np.sum((predictions == 0) & (test_labels_arr == 0))
    fp = np.sum((predictions == 1) & (test_labels_arr == 0))
    fn = np.sum((predictions == 0) & (test_labels_arr == 1))

    print(f"阈值: {threshold:.4f} (FPR=5%)")
    print(f"混淆矩阵:")
    print(f"  TP (正确识别OOD): {tp}")
    print(f"  TN (正确识别ID):  {tn}")
    print(f"  FP (ID→OOD误判): {fp}")
    print(f"  FN (OOD→ID误判): {fn}")

    # 获取FP和FN索引
    fp_indices = np.where((predictions == 1) & (test_labels_arr == 0))[0]
    fn_indices = np.where((predictions == 0) & (test_labels_arr == 1))[0]

    results = {
        'dataset': dataset_name,
        'threshold': float(threshold),
        'confusion_matrix': {
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        },
        'metrics': {
            'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0,
            'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
            'fpr': float(fp / (fp + tn)) if (fp + tn) > 0 else 0,
            'fnr': float(fn / (fn + tp)) if (fn + tp) > 0 else 0
        },
        'fp_cases': [],
        'fn_cases': []
    }

    # 分析FP案例
    if len(fp_indices) > 0:
        print(f"\n{'='*60}")
        print(f"False Positives (ID误判为OOD): {len(fp_indices)}个")
        print(f"{'='*60}")

        fp_scores_val = scores[fp_indices]
        fp_het = heterophily_scores[fp_indices]

        print(f"\nFP统计:")
        print(f"  平均OOD分数: {fp_scores_val.mean():.4f}")
        print(f"  平均异配性:  {fp_het.mean():.4f}")

        # 按分数排序，显示最严重的误判
        sorted_fp = sorted(zip(fp_indices, fp_scores_val, fp_het),
                           key=lambda x: x[1], reverse=True)

        print(f"\n最严重的FP案例 (前5个):")
        for rank, (idx, score, het) in enumerate(sorted_fp[:5], 1):
            text = test_texts[idx][:100] + "..." if len(test_texts[idx]) > 100 else test_texts[idx]
            intent = test_intents[idx] if test_intents else "unknown"

            print(f"\n  [{rank}] 索引={idx}")
            print(f"      文本: {text}")
            print(f"      真实意图: {intent} (ID)")
            print(f"      OOD分数: {score:.4f}")
            print(f"      异配性: {het:.4f}")

            results['fp_cases'].append({
                'index': int(idx),
                'text': test_texts[idx],
                'true_label': 'ID',
                'true_intent': intent,
                'ood_score': float(score),
                'heterophily': float(het)
            })

        # 保存更多案例到结果
        for idx, score, het in sorted_fp[5:20]:
            intent = test_intents[idx] if test_intents else "unknown"
            results['fp_cases'].append({
                'index': int(idx),
                'text': test_texts[idx],
                'true_label': 'ID',
                'true_intent': intent,
                'ood_score': float(score),
                'heterophily': float(het)
            })

    # 分析FN案例
    if len(fn_indices) > 0:
        print(f"\n{'='*60}")
        print(f"False Negatives (OOD误判为ID): {len(fn_indices)}个")
        print(f"{'='*60}")

        fn_scores_val = scores[fn_indices]
        fn_het = heterophily_scores[fn_indices]

        print(f"\nFN统计:")
        print(f"  平均OOD分数: {fn_scores_val.mean():.4f}")
        print(f"  平均异配性:  {fn_het.mean():.4f}")

        # 按分数排序（分数越低越严重）
        sorted_fn = sorted(zip(fn_indices, fn_scores_val, fn_het),
                           key=lambda x: x[1])

        print(f"\n最严重的FN案例 (前5个):")
        for rank, (idx, score, het) in enumerate(sorted_fn[:5], 1):
            text = test_texts[idx][:100] + "..." if len(test_texts[idx]) > 100 else test_texts[idx]
            intent = test_intents[idx] if test_intents else "unknown"

            print(f"\n  [{rank}] 索引={idx}")
            print(f"      文本: {text}")
            print(f"      真实意图: {intent} (OOD)")
            print(f"      OOD分数: {score:.4f}")
            print(f"      异配性: {het:.4f}")

            results['fn_cases'].append({
                'index': int(idx),
                'text': test_texts[idx],
                'true_label': 'OOD',
                'true_intent': intent,
                'ood_score': float(score),
                'heterophily': float(het)
            })

        # 保存更多案例
        for idx, score, het in sorted_fn[5:20]:
            intent = test_intents[idx] if test_intents else "unknown"
            results['fn_cases'].append({
                'index': int(idx),
                'text': test_texts[idx],
                'true_label': 'OOD',
                'true_intent': intent,
                'ood_score': float(score),
                'heterophily': float(het)
            })

    # 分析FP和FN的异配性差异
    if len(fp_indices) > 0 and len(fn_indices) > 0:
        print(f"\n{'='*60}")
        print("异配性对比分析")
        print(f"{'='*60}")

        # 正确分类的样本
        correct_id = np.where((predictions == 0) & (test_labels_arr == 0))[0]
        correct_ood = np.where((predictions == 1) & (test_labels_arr == 1))[0]

        print(f"\n异配性分布:")
        print(f"  正确ID (TN):  均值={heterophily_scores[correct_id].mean():.4f}, 标准差={heterophily_scores[correct_id].std():.4f}")
        print(f"  错误ID→OOD (FP): 均值={heterophily_scores[fp_indices].mean():.4f}, 标准差={heterophily_scores[fp_indices].std():.4f}")
        print(f"  正确OOD (TP): 均值={heterophily_scores[correct_ood].mean():.4f}, 标准差={heterophily_scores[correct_ood].std():.4f}")
        print(f"  错误OOD→ID (FN): 均值={heterophily_scores[fn_indices].mean():.4f}, 标准差={heterophily_scores[fn_indices].std():.4f}")

        results['heterophily_analysis'] = {
            'correct_id_mean': float(heterophily_scores[correct_id].mean()),
            'correct_id_std': float(heterophily_scores[correct_id].std()),
            'fp_mean': float(heterophily_scores[fp_indices].mean()),
            'fp_std': float(heterophily_scores[fp_indices].std()),
            'correct_ood_mean': float(heterophily_scores[correct_ood].mean()),
            'correct_ood_std': float(heterophily_scores[correct_ood].std()),
            'fn_mean': float(heterophily_scores[fn_indices].mean()),
            'fn_std': float(heterophily_scores[fn_indices].std())
        }

    return results


def main():
    """主函数"""

    print("\n" + "="*80)
    print("模块5: 失败案例分析")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # 配置
    k = 10
    alpha = 0.3

    print(f"\n配置: k={k}, alpha={alpha}")

    all_results = {}

    # 加载编码器
    print("\n加载编码器...")
    encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # 数据集
    datasets = [
        ('clinc150', load_clinc150, Path("data/clinc150")),
        ('banking77', load_banking77_oos, Path("data/banking77_oos")),
    ]

    for dataset_name, loader_func, data_path in datasets:
        print(f"\n处理数据集: {dataset_name.upper()}")

        # 加载数据
        print("加载数据...")
        train_texts, test_texts, test_labels, test_intents, train_labels = \
            loader_func(data_path)

        # 提取embeddings
        print("生成embeddings...")
        train_embs = encoder.encode(train_texts, batch_size=64, show_progress_bar=True)
        test_embs = encoder.encode(test_texts, batch_size=64, show_progress_bar=True)

        # 分析失败案例
        results = analyze_failures(
            dataset_name,
            train_texts,
            train_labels,
            test_texts,
            test_labels,
            test_intents,
            train_embs,
            test_embs,
            k=k,
            alpha=alpha
        )

        all_results[dataset_name] = results

    # 保存结果
    output_dir = Path("experiments/results/failure_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "failure_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 失败案例分析结果已保存: {output_file}")

    # 生成论文用摘要
    print("\n" + "="*80)
    print("论文用摘要")
    print("="*80)

    for dataset, results in all_results.items():
        cm = results['confusion_matrix']
        metrics = results['metrics']

        print(f"\n{dataset.upper()}:")
        print(f"  - FP (ID→OOD): {cm['fp']} ({metrics['fpr']*100:.1f}% FPR)")
        print(f"  - FN (OOD→ID): {cm['fn']} ({metrics['fnr']*100:.1f}% FNR)")

        if 'heterophily_analysis' in results:
            ha = results['heterophily_analysis']
            print(f"  - FP平均异配性: {ha['fp_mean']:.4f} (正常ID: {ha['correct_id_mean']:.4f})")
            print(f"  - FN平均异配性: {ha['fn_mean']:.4f} (正常OOD: {ha['correct_ood_mean']:.4f})")

    print("\n" + "="*80)
    print("失败案例分析完成!")
    print("="*80)


if __name__ == "__main__":
    main()
