#!/usr/bin/env python3
"""Evaluation and statistical analysis script"""
import os, json, argparse
import numpy as np
from scipy import stats


def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['baseline', 'dygprompt', 'tpnet', 'ssm_memory_llm'])
    parser.add_argument('--checkpoint_dir', default='checkpoints/')
    parser.add_argument('--output_dir', default='results/')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = {}

    for model in args.models:
        path = os.path.join(args.checkpoint_dir, model, 'results.json')
        if os.path.exists(path):
            with open(path) as f:
                results[model] = json.load(f)
            print(f"✅ Loaded {model}: MRR={results[model]['test_mrr_mean']:.4f}±{results[model]['test_mrr_std']:.4f}")

    if not results:
        print("❌ No results found!")
        return

    baseline_name = 'baseline' if 'baseline' in results else list(results.keys())[0]
    baseline = results[baseline_name]
    baseline_mrrs = [r['test_mrr'] for r in baseline['individual_runs']]

    summary = {'baseline': baseline_name, 'models': {}}
    stat_analysis = {}

    print("\n" + "=" * 60)
    print("Statistical Analysis")
    print("=" * 60)

    for model, data in results.items():
        model_mrrs = [r['test_mrr'] for r in data['individual_runs']]
        summary['models'][model] = {'mrr_mean': data['test_mrr_mean'], 'mrr_std': data['test_mrr_std']}

        if model != baseline_name:
            improvement = (np.mean(model_mrrs) - np.mean(baseline_mrrs)) / np.mean(baseline_mrrs)
            d = cohens_d(model_mrrs, baseline_mrrs)
            t_stat, p_value = stats.ttest_ind(model_mrrs, baseline_mrrs)

            stat_analysis[model] = {
                'baseline_mrr_mean': float(np.mean(baseline_mrrs)),
                'model_mrr_mean': float(np.mean(model_mrrs)),
                'improvement_relative': float(improvement),
                'cohen_d': float(d),
                'p_value': float(p_value)
            }
            summary['models'][model]['mrr_improvement'] = float(improvement)

            status_mrr = "✅" if improvement >= 0.03 else "❌"
            status_d = "✅" if abs(d) >= 0.45 else "❌"
            status_p = "✅" if p_value < 0.05 else "❌"

            print(f"\n{model} vs {baseline_name}:")
            print(f"  MRR: {np.mean(model_mrrs):.4f} vs {np.mean(baseline_mrrs):.4f}")
            print(f"  Improvement: {improvement*100:.2f}% {status_mrr}")
            print(f"  Cohen's d: {d:.3f} {status_d}")
            print(f"  p-value: {p_value:.4f} {status_p}")

    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, 'statistical_analysis.json'), 'w') as f:
        json.dump(stat_analysis, f, indent=2)

    print(f"\n✅ Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
