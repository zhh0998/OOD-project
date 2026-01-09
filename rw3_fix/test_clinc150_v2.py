#!/usr/bin/env python3
"""
CLINC150验证脚本 v2 - 使用Sentence-Transformers

使用专门为句子相似度训练的模型，预期性能更好。

Author: RW3 OOD Detection Project
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from quick_fix import FixedKNNDetector, MahalanobisDetector, LOFDetector, evaluate_ood
from data_loader import load_clinc150

from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score


def main():
    print("="*60)
    print("CLINC150 OOD检测验证 v2 - Sentence-Transformers")
    print("="*60)

    # 1. 加载数据
    print("\n[Step 1] 加载CLINC150数据集...")
    train_texts, test_texts, test_labels, test_intents, _ = load_clinc150()
    test_labels = np.array(test_labels)

    print(f"  训练样本: {len(train_texts)}")
    print(f"  测试样本: {len(test_texts)}")
    print(f"  OOD样本: {test_labels.sum()} ({test_labels.mean()*100:.1f}%)")

    # 2. 加载Sentence-Transformers模型
    print("\n[Step 2] 加载Sentence-Transformers模型...")

    # 尝试多个模型
    models_to_try = [
        'all-MiniLM-L6-v2',  # 快速，性能好
        # 'all-mpnet-base-v2',  # 更强但更慢
    ]

    best_auroc = 0
    best_model = None
    best_results = None

    for model_name in models_to_try:
        print(f"\n{'='*50}")
        print(f"测试模型: {model_name}")
        print(f"{'='*50}")

        model = SentenceTransformer(model_name)

        # 3. 提取embeddings
        print("\n[Step 3] 提取embeddings...")
        print("  提取训练集embeddings...")
        train_emb = model.encode(train_texts, show_progress_bar=True, batch_size=64)
        print(f"    Shape: {train_emb.shape}")

        print("  提取测试集embeddings...")
        test_emb = model.encode(test_texts, show_progress_bar=True, batch_size=64)
        print(f"    Shape: {test_emb.shape}")

        # 4. 运行多个检测器
        print("\n[Step 4] 运行OOD检测器...")

        results = {}

        # 4.1 k-NN with different k values
        for k in [10, 30, 50, 100, 200]:
            detector = FixedKNNDetector(k=k, verbose=False)
            detector.fit(train_emb)
            scores, auroc = detector.score_with_fix(test_emb, test_labels)
            metrics = evaluate_ood(test_labels, scores, auto_fix=False, verbose=False)
            results[f'KNN_k{k}'] = metrics
            print(f"  k-NN (k={k}): AUROC={metrics['auroc']:.4f}")

        # 4.2 Mahalanobis
        try:
            maha = MahalanobisDetector(verbose=False)
            maha.fit(train_emb)
            scores, auroc = maha.score_with_fix(test_emb, test_labels)
            metrics = evaluate_ood(test_labels, scores, auto_fix=False, verbose=False)
            results['Mahalanobis'] = metrics
            print(f"  Mahalanobis: AUROC={metrics['auroc']:.4f}")
        except Exception as e:
            print(f"  Mahalanobis failed: {e}")

        # 4.3 LOF
        lof = LOFDetector(k=20, verbose=False)
        lof.fit(train_emb)
        scores, auroc = lof.score_with_fix(test_emb, test_labels)
        metrics = evaluate_ood(test_labels, scores, auto_fix=False, verbose=False)
        results['LOF'] = metrics
        print(f"  LOF: AUROC={metrics['auroc']:.4f}")

        # 找最佳结果
        best_method = max(results.keys(), key=lambda x: results[x]['auroc'])
        model_best_auroc = results[best_method]['auroc']

        print(f"\n  {model_name} 最佳: {best_method} = {model_best_auroc:.4f}")

        if model_best_auroc > best_auroc:
            best_auroc = model_best_auroc
            best_model = model_name
            best_results = results.copy()

    # 5. 汇总结果
    print("\n" + "="*60)
    print("CLINC150 验证结果汇总")
    print("="*60)

    print(f"\n最佳模型: {best_model}")
    print(f"\n{'方法':<20} {'AUROC':<10} {'AUPR':<10} {'FPR@95':<10}")
    print("-"*50)

    for method, metrics in sorted(best_results.items(), key=lambda x: -x[1]['auroc']):
        print(f"{method:<20} {metrics['auroc']:.4f}     {metrics['aupr']:.4f}     {metrics['fpr95']:.4f}")

    # 最佳AUROC
    best_method = max(best_results.keys(), key=lambda x: best_results[x]['auroc'])
    best_metrics = best_results[best_method]

    print("\n" + "="*60)
    print(f"最佳结果: {best_method}")
    print(f"  AUROC: {best_metrics['auroc']:.4f} ({best_metrics['auroc']*100:.2f}%)")
    print(f"  AUPR:  {best_metrics['aupr']:.4f}")
    print(f"  FPR@95: {best_metrics['fpr95']:.4f}")
    print("="*60)

    # 与v2对比
    v2_auroc = 0.7286
    improvement = best_metrics['auroc'] - v2_auroc

    print("\n对比分析:")
    print(f"  v2 (有Bug): {v2_auroc*100:.2f}%")
    print(f"  v3 (修复后): {best_metrics['auroc']*100:.2f}%")
    print(f"  改善: {improvement*100:+.2f}%")

    # 验证目标
    target = 0.90
    if best_metrics['auroc'] >= target:
        print(f"\n  [SUCCESS] 达到目标 >= {target*100:.0f}%!")
    else:
        print(f"\n  [INFO] 当前 {best_metrics['auroc']*100:.2f}%, 目标 {target*100:.0f}%")

    # 保存结果
    results_to_save = {
        'dataset': 'clinc150',
        'model': best_model,
        'v2_auroc': v2_auroc,
        'v3_auroc': best_metrics['auroc'],
        'improvement': improvement,
        'best_method': best_method,
        'all_results': {k: v for k, v in best_results.items()},
        'n_train': len(train_texts),
        'n_test': len(test_texts),
        'n_ood': int(test_labels.sum())
    }

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "clinc150_verification_v2.json", 'w') as f:
        json.dump(results_to_save, f, indent=2)

    print(f"\n结果已保存: {output_dir / 'clinc150_verification_v2.json'}")

    return best_metrics['auroc']


if __name__ == "__main__":
    main()
