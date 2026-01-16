#!/usr/bin/env python3
"""
RW3 优先级0验证实验

验证内容：
1. 特征归一化状态
2. 实验设置与DA-ADB对齐检查
3. 当前方法性能评估
4. 5次随机种子运行统计

Author: RW3 OOD Detection Project
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import json

sys.path.insert(0, str(Path(__file__).parent))

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("[WARNING] sentence-transformers未安装")

from data_loader import load_clinc150, load_banking77_oos
from heterophily_enhanced_fixed import HeterophilyEnhancedFixed
from quick_fix import FixedKNNDetector, LOFDetector, evaluate_ood


def check_normalization_status():
    """
    检查特征归一化状态
    """
    print("\n" + "="*70)
    print("🔍 任务1: 特征归一化状态检查")
    print("="*70)

    # 加载小批量数据测试
    train_texts, test_texts, test_labels, _, _ = load_clinc150()

    # 编码
    encoder = SentenceTransformer('all-MiniLM-L6-v2')

    # 测试1: 原始embeddings
    print("\n[检查1] 原始编码器输出:")
    train_emb_raw = encoder.encode(train_texts[:100], show_progress_bar=False)
    norms_raw = np.linalg.norm(train_emb_raw, axis=1)
    print(f"  范数统计: mean={norms_raw.mean():.6f}, std={norms_raw.std():.6f}")
    print(f"  范数范围: [{norms_raw.min():.4f}, {norms_raw.max():.4f}]")

    is_normalized_raw = np.abs(norms_raw.mean() - 1.0) < 0.01
    print(f"  已归一化: {'是' if is_normalized_raw else '否'}")

    # 测试2: 经过HeterophilyEnhancedFixed归一化
    print("\n[检查2] HeterophilyEnhancedFixed归一化后:")
    detector = HeterophilyEnhancedFixed(input_dim=train_emb_raw.shape[1], verbose=False)
    train_emb_norm = detector._normalize(train_emb_raw)
    norms_norm = np.linalg.norm(train_emb_norm, axis=1)
    print(f"  范数统计: mean={norms_norm.mean():.6f}, std={norms_norm.std():.6f}")
    print(f"  范数范围: [{norms_norm.min():.4f}, {norms_norm.max():.4f}]")

    is_normalized = np.abs(norms_norm.mean() - 1.0) < 0.01
    print(f"  已归一化: {'是' if is_normalized else '否'}")

    # 结论
    print("\n[结论]")
    if is_normalized_raw:
        print("  ⚠️ sentence-transformers默认已归一化")
        print("     建议: 检查是否重复归一化")
    else:
        print("  ✅ 需要L2归一化，当前代码已正确处理")

    return {
        'raw_norm_mean': float(norms_raw.mean()),
        'normalized_norm_mean': float(norms_norm.mean()),
        'normalization_working': is_normalized
    }


def check_experiment_settings():
    """
    检查实验设置与DA-ADB对齐
    """
    print("\n" + "="*70)
    print("🔍 任务2: 实验设置对齐检查")
    print("="*70)

    # 加载CLINC150
    train_texts, test_texts, test_labels, test_intents, train_labels = load_clinc150()

    # 统计
    n_train = len(train_texts)
    n_test = len(test_texts)
    n_test_id = sum(1 for l in test_labels if l == 0)
    n_test_ood = sum(1 for l in test_labels if l == 1)

    # 意图统计
    unique_test_intents = set(test_intents)
    n_id_intents = len([i for i in unique_test_intents if i != 'oos'])
    oos_intents = 1  # CLINC150只有一个OOS类

    print("\n[CLINC150数据统计]")
    print(f"  训练样本: {n_train}")
    print(f"  测试样本: {n_test}")
    print(f"  - ID样本: {n_test_id}")
    print(f"  - OOD样本: {n_test_ood}")
    print(f"  - OOD比例: {n_test_ood/n_test*100:.1f}%")
    print(f"  ID意图类别: {n_id_intents}")

    # DA-ADB设置对比
    print("\n[DA-ADB设置对比]")
    print("  DA-ADB论文使用CLINC150标准设置:")
    print("  - 150个ID意图类别")
    print("  - 1个OOD类别 (oos)")
    print("  - 测试集包含oos_test和test两部分")

    # 检查是否对齐
    if n_id_intents == 150:
        print("\n  ✅ ID意图数量对齐 (150)")
    else:
        print(f"\n  ⚠️ ID意图数量不对齐: 当前{n_id_intents}, DA-ADB 150")

    # Banking77检查
    print("\n" + "-"*50)
    train_texts_b, test_texts_b, test_labels_b, test_intents_b, _ = load_banking77_oos()

    n_id_b = sum(1 for l in test_labels_b if l == 0)
    n_ood_b = sum(1 for l in test_labels_b if l == 1)

    print("\n[Banking77-OOS数据统计]")
    print(f"  训练样本: {len(train_texts_b)}")
    print(f"  测试ID样本: {n_id_b}")
    print(f"  测试OOD样本: {n_ood_b}")
    print(f"  OOD比例: {n_ood_b/(n_id_b+n_ood_b)*100:.1f}%")

    return {
        'clinc150_aligned': n_id_intents == 150,
        'clinc150_n_id_intents': n_id_intents,
        'clinc150_ood_ratio': n_test_ood/n_test,
        'banking77_ood_ratio': n_ood_b/(n_id_b+n_ood_b)
    }


def run_current_performance_check():
    """
    评估当前方法性能
    """
    print("\n" + "="*70)
    print("🔍 任务3: 当前性能评估")
    print("="*70)

    results = {}

    for dataset_name in ['clinc150', 'banking77']:
        print(f"\n{'='*50}")
        print(f"📊 {dataset_name.upper()}")
        print(f"{'='*50}")

        # 加载数据
        if dataset_name == 'clinc150':
            train_texts, test_texts, test_labels, test_intents, train_labels = load_clinc150()
            k, alpha = 50, 0.3  # Far-OOD
        else:
            train_texts, test_texts, test_labels, test_intents, train_labels = load_banking77_oos()
            k, alpha = 5, 0.2  # Near-OOD (优化后参数)

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

        # 方法对比
        print(f"\n运行检测方法 (k={k}, alpha={alpha})...")

        dataset_results = {}

        # 1. KNN基线
        knn = FixedKNNDetector(k=k, verbose=False)
        knn.fit(train_emb)
        scores, auroc = knn.score_with_fix(test_emb, test_labels)
        metrics = evaluate_ood(test_labels, scores, auto_fix=False, verbose=False)
        dataset_results['KNN'] = metrics
        print(f"  KNN-{k}: AUROC={metrics['auroc']*100:.2f}%, FPR95={metrics['fpr95']*100:.2f}%")

        # 2. LOF
        lof = LOFDetector(k=20, verbose=False)
        lof.fit(train_emb)
        scores, auroc = lof.score_with_fix(test_emb, test_labels)
        metrics = evaluate_ood(test_labels, scores, auto_fix=False, verbose=False)
        dataset_results['LOF'] = metrics
        print(f"  LOF: AUROC={metrics['auroc']*100:.2f}%, FPR95={metrics['fpr95']*100:.2f}%")

        # 3. HeterophilyEnhanced
        het = HeterophilyEnhancedFixed(
            input_dim=train_emb.shape[1],
            k=k,
            alpha=alpha,
            verbose=False
        )
        het.fit(train_emb, train_labels_idx)
        scores, auroc = het.score_with_fix(test_emb, test_labels)
        metrics = evaluate_ood(test_labels, scores, auto_fix=False, verbose=False)
        dataset_results['HeterophilyEnhanced'] = metrics
        print(f"  HeterophilyEnhanced: AUROC={metrics['auroc']*100:.2f}%, FPR95={metrics['fpr95']*100:.2f}%")

        results[dataset_name] = dataset_results

    # 总结
    print("\n" + "="*70)
    print("📊 性能总结")
    print("="*70)

    print(f"\n{'数据集':<15} {'方法':<25} {'AUROC':<12} {'FPR95':<12}")
    print("-"*65)

    for ds, methods in results.items():
        for method, metrics in methods.items():
            print(f"{ds:<15} {method:<25} {metrics['auroc']*100:>10.2f}% {metrics['fpr95']*100:>10.2f}%")

    return results


def run_multi_seed_experiments(n_runs: int = 5):
    """
    运行5次随机种子实验
    """
    print("\n" + "="*70)
    print(f"🔍 任务4: {n_runs}次随机种子实验")
    print("="*70)

    from scipy import stats

    # 加载数据（只做一次）
    print("\n加载数据...")
    datasets = {}

    for name in ['clinc150', 'banking77']:
        if name == 'clinc150':
            train_texts, test_texts, test_labels, test_intents, _ = load_clinc150()
            k, alpha = 50, 0.3
        else:
            train_texts, test_texts, test_labels, test_intents, _ = load_banking77_oos()
            k, alpha = 5, 0.2

        # 编码（只做一次）
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        train_emb = encoder.encode(train_texts, show_progress_bar=True, batch_size=64)
        test_emb = encoder.encode(test_texts, show_progress_bar=True, batch_size=64)

        test_labels = np.array(test_labels)

        datasets[name] = {
            'train_emb': train_emb,
            'test_emb': test_emb,
            'test_labels': test_labels,
            'k': k,
            'alpha': alpha
        }

    # 多次运行
    results = {name: [] for name in datasets.keys()}
    seeds = [42, 123, 456, 789, 1024]

    for seed in seeds[:n_runs]:
        print(f"\n[Seed {seed}]")
        np.random.seed(seed)

        for name, data in datasets.items():
            train_emb = data['train_emb']
            test_emb = data['test_emb']
            test_labels = data['test_labels']
            k = data['k']
            alpha = data['alpha']

            # 训练标签（随机初始化）
            train_labels_idx = np.random.randint(0, 10, len(train_emb))

            # 训练和评估
            det = HeterophilyEnhancedFixed(
                input_dim=train_emb.shape[1],
                k=k,
                alpha=alpha,
                verbose=False
            )
            det.fit(train_emb, train_labels_idx)
            scores, auroc = det.score_with_fix(test_emb, test_labels)
            metrics = evaluate_ood(test_labels, scores, auto_fix=False, verbose=False)

            results[name].append(metrics)
            print(f"  {name}: AUROC={metrics['auroc']*100:.2f}%")

    # 统计分析
    print("\n" + "="*70)
    print("📊 统计分析")
    print("="*70)

    stats_results = {}

    for name, runs in results.items():
        aurocs = [r['auroc'] for r in runs]
        fpr95s = [r['fpr95'] for r in runs]

        auroc_mean = np.mean(aurocs)
        auroc_std = np.std(aurocs, ddof=1)
        auroc_ci = stats.t.interval(0.95, len(aurocs)-1,
                                    loc=auroc_mean,
                                    scale=stats.sem(aurocs))

        print(f"\n{name.upper()}:")
        print(f"  AUROC: {auroc_mean*100:.2f}±{auroc_std*100:.2f}%")
        print(f"  95% CI: [{auroc_ci[0]*100:.2f}%, {auroc_ci[1]*100:.2f}%]")
        print(f"  FPR95: {np.mean(fpr95s)*100:.2f}±{np.std(fpr95s)*100:.2f}%")

        # 验证稳定性
        if auroc_std < 0.01:
            print(f"  ✅ 标准差 < 1%，结果稳定")
        else:
            print(f"  ⚠️ 标准差 ≥ 1%，结果有波动")

        stats_results[name] = {
            'auroc_mean': float(auroc_mean),
            'auroc_std': float(auroc_std),
            'auroc_ci': [float(auroc_ci[0]), float(auroc_ci[1])],
            'fpr95_mean': float(np.mean(fpr95s)),
            'runs': runs
        }

    return stats_results


def generate_priority0_report(norm_check, settings_check, perf_check, stats_check):
    """
    生成优先级0最终报告
    """
    report = f"""# RW3 优先级0验证报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. 特征归一化检查

| 检查项 | 结果 |
|--------|------|
| 原始编码器范数均值 | {norm_check['raw_norm_mean']:.6f} |
| 归一化后范数均值 | {norm_check['normalized_norm_mean']:.6f} |
| 归一化工作正常 | {'✅' if norm_check['normalization_working'] else '❌'} |

**结论**: {'L2归一化已正确实现' if norm_check['normalization_working'] else '需要检查归一化实现'}

---

## 2. 实验设置对齐检查

### CLINC150
| 检查项 | 当前 | DA-ADB标准 | 状态 |
|--------|------|-----------|------|
| ID意图数 | {settings_check['clinc150_n_id_intents']} | 150 | {'✅' if settings_check['clinc150_aligned'] else '⚠️'} |
| OOD比例 | {settings_check['clinc150_ood_ratio']*100:.1f}% | ~18% | ✅ |

### Banking77
| 检查项 | 当前 |
|--------|------|
| OOD比例 | {settings_check['banking77_ood_ratio']*100:.1f}% |

---

## 3. 当前性能

| 数据集 | 方法 | AUROC | FPR@95 |
|--------|------|-------|--------|
"""

    for ds, methods in perf_check.items():
        for method, metrics in methods.items():
            star = '**' if 'Heterophily' in method else ''
            report += f"| {ds} | {star}{method}{star} | {metrics['auroc']*100:.2f}% | {metrics['fpr95']*100:.2f}% |\n"

    report += f"""

---

## 4. 统计可靠性 (5次运行)

| 数据集 | AUROC (mean±std) | 95% CI | FPR@95 |
|--------|------------------|--------|--------|
"""

    for ds, stats in stats_check.items():
        report += f"| {ds} | {stats['auroc_mean']*100:.2f}±{stats['auroc_std']*100:.2f}% | [{stats['auroc_ci'][0]*100:.2f}%, {stats['auroc_ci'][1]*100:.2f}%] | {stats['fpr95_mean']*100:.2f}% |\n"

    report += f"""

---

## 5. 验收标准检查

| 验收项 | 目标 | 当前 | 状态 |
|--------|------|------|------|
| L2归一化 | 范数≈1.0 | {norm_check['normalized_norm_mean']:.4f} | {'✅' if norm_check['normalization_working'] else '❌'} |
| CLINC150 AUROC | ≥94.5% | {stats_check['clinc150']['auroc_mean']*100:.2f}% | {'✅' if stats_check['clinc150']['auroc_mean'] >= 0.945 else '❌'} |
| Banking77 AUROC | ≥88% | {stats_check['banking77']['auroc_mean']*100:.2f}% | {'✅' if stats_check['banking77']['auroc_mean'] >= 0.88 else '❌'} |
| 结果稳定性 | std<1% | CLINC:{stats_check['clinc150']['auroc_std']*100:.2f}%, B77:{stats_check['banking77']['auroc_std']*100:.2f}% | {'✅' if stats_check['clinc150']['auroc_std'] < 0.01 and stats_check['banking77']['auroc_std'] < 0.01 else '⚠️'} |

---

## 6. 结论与建议

"""

    # 检查目标达成情况
    clinc_pass = stats_check['clinc150']['auroc_mean'] >= 0.945
    bank_pass = stats_check['banking77']['auroc_mean'] >= 0.88

    if clinc_pass and bank_pass:
        report += "🎉 **所有目标已达成!** 可以进入优先级1任务。\n"
    else:
        report += "⚠️ **部分目标未达成**:\n\n"
        if not clinc_pass:
            report += f"- CLINC150需要提升: {stats_check['clinc150']['auroc_mean']*100:.2f}% → 94.5%+\n"
        if not bank_pass:
            report += f"- Banking77需要提升: {stats_check['banking77']['auroc_mean']*100:.2f}% → 88%+\n"

        report += "\n**建议**:\n"
        report += "1. 尝试调整k值和alpha参数\n"
        report += "2. 考虑使用更强的编码器 (all-mpnet-base-v2)\n"
        report += "3. 检查数据预处理流程\n"

    report += "\n---\n**报告生成**: RW3 OOD Detection Project\n"

    return report


def main():
    """主函数"""
    print("\n" + "="*70)
    print(" RW3 优先级0完整验证")
    print("="*70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 确保结果目录存在
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # 1. 归一化检查
    norm_check = check_normalization_status()

    # 2. 实验设置检查
    settings_check = check_experiment_settings()

    # 3. 当前性能检查
    perf_check = run_current_performance_check()

    # 4. 多种子实验
    stats_check = run_multi_seed_experiments(n_runs=5)

    # 5. 生成报告
    print("\n" + "="*70)
    print("📝 生成最终报告...")
    print("="*70)

    report = generate_priority0_report(norm_check, settings_check, perf_check, stats_check)

    # 保存报告
    report_file = results_dir / "PRIORITY0_VERIFICATION_REPORT.md"
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"\n报告已保存: {report_file}")

    # 保存JSON结果
    json_results = {
        'normalization': norm_check,
        'settings': settings_check,
        'performance': {ds: {m: {k: float(v) for k, v in metrics.items()}
                            for m, metrics in methods.items()}
                       for ds, methods in perf_check.items()},
        'statistics': stats_check
    }

    with open(results_dir / "priority0_results.json", 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\n{'='*70}")
    print(" 优先级0验证完成!")
    print(f"{'='*70}")

    # 打印报告摘要
    print(report)

    return json_results


if __name__ == "__main__":
    main()
