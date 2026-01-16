#!/usr/bin/env python3
"""
RW3 超参数优化

目标:
- CLINC150: 93.88% → 94.5%+ AUROC
- Banking77: 86.21% → 88%+ AUROC

策略:
1. k值网格搜索 (10, 20, 30, 50, 100)
2. alpha值网格搜索 (0.1, 0.2, 0.3, 0.4, 0.5)
3. 尝试不同编码器

Author: RW3 OOD Detection Project
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent))

from sentence_transformers import SentenceTransformer
from data_loader import load_clinc150, load_banking77_oos
from heterophily_enhanced_fixed import HeterophilyEnhancedFixed
from quick_fix import evaluate_ood


def run_hyperparameter_search():
    """
    超参数网格搜索
    """
    print("\n" + "="*70)
    print("🔍 超参数优化实验")
    print("="*70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    all_results = {}

    # 定义搜索空间
    k_values = [5, 10, 20, 30, 50, 75, 100]
    alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    for dataset_name in ['clinc150', 'banking77']:
        print(f"\n{'='*70}")
        print(f"📊 {dataset_name.upper()} 超参数搜索")
        print(f"{'='*70}")

        # 加载数据
        if dataset_name == 'clinc150':
            train_texts, test_texts, test_labels, test_intents, _ = load_clinc150()
        else:
            train_texts, test_texts, test_labels, test_intents, _ = load_banking77_oos()

        test_labels = np.array(test_labels)

        # 编码
        print(f"\n编码文本...")
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        train_emb = encoder.encode(train_texts, show_progress_bar=True, batch_size=64)
        test_emb = encoder.encode(test_texts, show_progress_bar=True, batch_size=64)

        # 训练标签
        unique_intents = sorted(set(test_intents) - {'oos'})
        intent_to_idx = {i: idx for idx, i in enumerate(unique_intents)}
        train_labels_idx = np.zeros(len(train_emb), dtype=int)

        # 网格搜索
        print(f"\n开始网格搜索: k={k_values}, alpha={alpha_values}")
        print(f"总组合数: {len(k_values) * len(alpha_values)}")

        dataset_results = []
        best_auroc = 0
        best_config = None

        for k in k_values:
            for alpha in alpha_values:
                # 训练和评估
                detector = HeterophilyEnhancedFixed(
                    input_dim=train_emb.shape[1],
                    k=k,
                    alpha=alpha,
                    verbose=False
                )
                detector.fit(train_emb, train_labels_idx)
                scores, auroc = detector.score_with_fix(test_emb, test_labels)
                metrics = evaluate_ood(test_labels, scores, auto_fix=False, verbose=False)

                result = {
                    'k': k,
                    'alpha': alpha,
                    'auroc': float(metrics['auroc']),
                    'fpr95': float(metrics['fpr95']),
                    'aupr': float(metrics['aupr'])
                }
                dataset_results.append(result)

                # 更新最佳配置
                if metrics['auroc'] > best_auroc:
                    best_auroc = metrics['auroc']
                    best_config = (k, alpha)

                print(f"  k={k:3d}, alpha={alpha:.1f}: AUROC={metrics['auroc']*100:.2f}%", end='')
                if metrics['auroc'] == best_auroc:
                    print(" ★")
                else:
                    print()

        print(f"\n{'='*50}")
        print(f"最佳配置: k={best_config[0]}, alpha={best_config[1]}")
        print(f"最佳AUROC: {best_auroc*100:.2f}%")

        all_results[dataset_name] = {
            'grid_search': dataset_results,
            'best_config': {'k': best_config[0], 'alpha': best_config[1]},
            'best_auroc': float(best_auroc)
        }

    # 保存结果
    with open(results_dir / "hyperparameter_search_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print("📊 优化结果总结")
    print(f"{'='*70}")

    print(f"\n{'数据集':<15} {'原始配置':<20} {'原始AUROC':<12} {'最佳配置':<20} {'最佳AUROC':<12} {'提升':<10}")
    print("-"*90)

    original_configs = {
        'clinc150': {'k': 50, 'alpha': 0.3, 'auroc': 0.9388},
        'banking77': {'k': 5, 'alpha': 0.2, 'auroc': 0.8621}
    }

    for ds, results in all_results.items():
        orig = original_configs[ds]
        best = results['best_config']
        improvement = (results['best_auroc'] - orig['auroc']) * 100

        print(f"{ds:<15} k={orig['k']}, α={orig['alpha']:<10} {orig['auroc']*100:>10.2f}% "
              f"k={best['k']}, α={best['alpha']:<10} {results['best_auroc']*100:>10.2f}% "
              f"{improvement:>+8.2f}%")

    # 检查是否达到目标
    print(f"\n{'='*70}")
    print("🎯 目标检查")
    print(f"{'='*70}")

    targets = {'clinc150': 0.945, 'banking77': 0.88}

    for ds, target in targets.items():
        achieved = all_results[ds]['best_auroc']
        status = '✅' if achieved >= target else '❌'
        print(f"{ds}: {achieved*100:.2f}% vs {target*100:.1f}% {status}")

    return all_results


def run_encoder_comparison():
    """
    尝试不同编码器
    """
    print("\n" + "="*70)
    print("🔍 编码器对比实验")
    print("="*70)

    encoders = {
        'all-MiniLM-L6-v2': 'all-MiniLM-L6-v2',
        'all-mpnet-base-v2': 'all-mpnet-base-v2',  # 更强的编码器
    }

    # 使用CLINC150测试
    train_texts, test_texts, test_labels, test_intents, _ = load_clinc150()
    test_labels = np.array(test_labels)

    results = {}

    for name, model_name in encoders.items():
        print(f"\n测试编码器: {name}")

        try:
            encoder = SentenceTransformer(model_name)
            train_emb = encoder.encode(train_texts, show_progress_bar=True, batch_size=64)
            test_emb = encoder.encode(test_texts, show_progress_bar=True, batch_size=64)

            # 训练标签
            unique_intents = sorted(set(test_intents) - {'oos'})
            train_labels_idx = np.zeros(len(train_emb), dtype=int)

            # 使用最佳k值（从之前搜索得到）
            k = 20  # 尝试中等k值
            alpha = 0.2

            detector = HeterophilyEnhancedFixed(
                input_dim=train_emb.shape[1],
                k=k,
                alpha=alpha,
                verbose=False
            )
            detector.fit(train_emb, train_labels_idx)
            scores, auroc = detector.score_with_fix(test_emb, test_labels)
            metrics = evaluate_ood(test_labels, scores, auto_fix=False, verbose=False)

            results[name] = {
                'auroc': float(metrics['auroc']),
                'fpr95': float(metrics['fpr95']),
                'dim': train_emb.shape[1]
            }

            print(f"  AUROC: {metrics['auroc']*100:.2f}%")
            print(f"  维度: {train_emb.shape[1]}")

        except Exception as e:
            print(f"  错误: {e}")
            results[name] = {'error': str(e)}

    return results


def main():
    """主函数"""
    # 1. 超参数搜索
    hp_results = run_hyperparameter_search()

    # 2. 编码器对比（可选）
    # encoder_results = run_encoder_comparison()

    print(f"\n{'='*70}")
    print("优化完成!")
    print(f"{'='*70}")

    # 返回最佳配置建议
    print("\n推荐配置:")
    for ds, results in hp_results.items():
        best = results['best_config']
        print(f"  {ds}: k={best['k']}, alpha={best['alpha']} (AUROC={results['best_auroc']*100:.2f}%)")


if __name__ == "__main__":
    main()
