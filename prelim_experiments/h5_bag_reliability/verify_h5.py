#!/usr/bin/env python3
"""
Hypothesis 5 Verification: Bag Size vs Label Reliability

H5: Bags with size ≥ 3 have significantly higher label reliability
    than single-instance bags (Cohen's d = 0.5-0.8, based on NYT-H)

Methodology:
1. Load NYT-H dataset with human annotations
2. Group bags by size (1, 2, 3+)
3. Compare reliability (human "yes" rate) across groups
4. Compute Cohen's d effect size
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.nyth_loader import NYTHDataset, load_nyth, NYTHBag
from src.utils.statistics import (
    compute_cohens_d,
    independent_t_test,
    HypothesisTestResult
)
from src.utils.visualization import (
    plot_grouped_boxplot,
    plot_fitted_line,
    set_publication_style
)


def run_bag_reliability_experiment(
    data_dir: str = './nyth',
    output_dir: str = './results/h5',
    seed: int = 42
) -> Dict:
    """
    Run the complete H5 verification experiment

    Args:
        data_dir: Path to NYT-H data
        output_dir: Output directory
        seed: Random seed

    Returns:
        Experiment results
    """
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)

    print("=" * 60)
    print("Hypothesis 5 Verification: Bag Size vs Label Reliability")
    print("=" * 60)

    # Step 1: Load NYT-H dataset
    print("\n[Step 1] Loading NYT-H dataset...")
    dataset = NYTHDataset(data_dir)
    dataset.load()

    stats = dataset.get_statistics()
    print(f"  - Total sentences: {stats['n_sentences']}")
    print(f"  - Total bags: {stats['n_bags']}")
    print(f"  - Average bag size: {stats['avg_bag_size']:.2f}")
    print(f"  - Human 'yes' rate: {stats['yes_rate']:.4f}")
    print(f"  - Human 'no' rate: {stats['no_rate']:.4f}")

    # Step 2: Group bags by size
    print("\n[Step 2] Grouping bags by size...")
    bags_by_size = dataset.get_bags_by_size()

    size_1_bags = bags_by_size.get(1, [])
    size_2_bags = bags_by_size.get(2, [])
    size_3plus_bags = []
    for size, bags in bags_by_size.items():
        if size >= 3:
            size_3plus_bags.extend(bags)

    print(f"  Size=1 bags: {len(size_1_bags)}")
    print(f"  Size=2 bags: {len(size_2_bags)}")
    print(f"  Size≥3 bags: {len(size_3plus_bags)}")

    # Step 3: Compute reliability for each group
    print("\n[Step 3] Computing reliability metrics...")

    def bag_reliability(bag: NYTHBag) -> int:
        """Binary reliability: 1 if bag is correct, 0 otherwise"""
        return 1 if bag.human_label == 'yes' else 0

    reliability_1 = [bag_reliability(b) for b in size_1_bags]
    reliability_2 = [bag_reliability(b) for b in size_2_bags]
    reliability_3plus = [bag_reliability(b) for b in size_3plus_bags]

    mean_rel_1 = np.mean(reliability_1) if reliability_1 else 0
    mean_rel_2 = np.mean(reliability_2) if reliability_2 else 0
    mean_rel_3plus = np.mean(reliability_3plus) if reliability_3plus else 0

    print(f"  Size=1 reliability: {mean_rel_1:.4f}")
    print(f"  Size=2 reliability: {mean_rel_2:.4f}")
    print(f"  Size≥3 reliability: {mean_rel_3plus:.4f}")

    # Step 4: Statistical tests
    print("\n[Step 4] Computing effect sizes...")

    # Cohen's d between size=1 and size≥3
    cohens_result = compute_cohens_d(reliability_3plus, reliability_1)
    print(f"  Cohen's d (3+ vs 1): {cohens_result['d']:.4f}")
    print(f"  Interpretation: {cohens_result['interpretation']}")

    # T-test
    t_test = independent_t_test(reliability_3plus, reliability_1)
    print(f"  t-statistic: {t_test['t_statistic']:.4f}")
    print(f"  p-value: {t_test['p_value']:.6f}")

    # Also compare 2 vs 1
    cohens_2v1 = compute_cohens_d(reliability_2, reliability_1)
    print(f"\n  Cohen's d (2 vs 1): {cohens_2v1['d']:.4f}")

    # Step 5: Trend analysis (reliability vs bag size)
    print("\n[Step 5] Trend analysis...")

    size_reliability = []
    for size in sorted(bags_by_size.keys()):
        bags = bags_by_size[size]
        reliabilities = [bag_reliability(b) for b in bags]
        mean_rel = np.mean(reliabilities) if reliabilities else 0
        size_reliability.append({
            'size': size,
            'mean_reliability': mean_rel,
            'n_bags': len(bags)
        })
        if size <= 5:
            print(f"  Size {size}: reliability = {mean_rel:.4f} (n={len(bags)})")

    # Correlation between size and reliability
    sizes = [sr['size'] for sr in size_reliability if sr['n_bags'] >= 10]
    reliabilities = [sr['mean_reliability'] for sr in size_reliability if sr['n_bags'] >= 10]

    from src.utils.statistics import compute_pearson_correlation
    if len(sizes) > 2:
        correlation = compute_pearson_correlation(sizes, reliabilities)
        print(f"\n  Pearson r (size vs reliability): {correlation['r']:.4f}")
    else:
        correlation = {'r': 0, 'p_value': 1}

    # Step 6: Relation-level analysis
    print("\n[Step 6] Relation-level analysis...")

    relation_analysis = defaultdict(lambda: {'size_1': [], 'size_3plus': []})

    for bag in size_1_bags:
        relation_analysis[bag.relation]['size_1'].append(bag_reliability(bag))
    for bag in size_3plus_bags:
        relation_analysis[bag.relation]['size_3plus'].append(bag_reliability(bag))

    # Find relations with largest reliability difference
    relation_effects = []
    for relation, data in relation_analysis.items():
        if len(data['size_1']) >= 10 and len(data['size_3plus']) >= 10:
            d = compute_cohens_d(data['size_3plus'], data['size_1'])
            relation_effects.append({
                'relation': relation,
                'cohens_d': d['d'],
                'reliability_diff': np.mean(data['size_3plus']) - np.mean(data['size_1'])
            })

    relation_effects.sort(key=lambda x: -x['cohens_d'])
    print("  Top 5 relations with largest bag size effect:")
    for re in relation_effects[:5]:
        print(f"    {re['relation']}: d = {re['cohens_d']:.4f}")

    # Step 7: Hypothesis test
    # Target: Cohen's d between 0.5 and 0.8 (medium effect)
    hypothesis_passed = 0.5 <= abs(cohens_result['d']) <= 0.8

    result = HypothesisTestResult(
        hypothesis_id="H5",
        description="Bag size ≥3 has higher reliability than single-instance bags",
        passed=hypothesis_passed,
        statistics={
            'cohens_d': cohens_result['d'],
            't_statistic': t_test['t_statistic'],
            'p_value': t_test['p_value'],
            'reliability_size_1': mean_rel_1,
            'reliability_size_2': mean_rel_2,
            'reliability_size_3plus': mean_rel_3plus,
            'n_size_1': len(size_1_bags),
            'n_size_3plus': len(size_3plus_bags)
        },
        threshold={
            'cohens_d': '0.5 - 0.8 (medium effect)'
        },
        details=f"Based on {len(size_1_bags) + len(size_3plus_bags)} bags from NYT-H"
    )

    print("\n" + "=" * 60)
    print(result.summary())
    print("=" * 60)

    # Step 8: Visualization
    print("\n[Step 8] Generating visualizations...")

    # Boxplot by bag size
    fig1 = plot_grouped_boxplot(
        [reliability_1, reliability_2, reliability_3plus],
        group_labels=['Size=1', 'Size=2', 'Size≥3'],
        ylabel='Label Reliability (Human "Yes" Rate)',
        title='H5: Bag Size vs Label Reliability (NYT-H)',
        save_path=os.path.join(output_dir, 'figures', 'h5_boxplot.png')
    )

    # Line plot of reliability trend
    if len(sizes) > 2:
        fig2 = plot_fitted_line(
            sizes, reliabilities,
            xlabel='Bag Size',
            ylabel='Mean Reliability',
            title='H5: Label Reliability Trend by Bag Size',
            save_path=os.path.join(output_dir, 'figures', 'h5_trend.png')
        )

    # Step 9: Save results
    print("\n[Step 9] Saving results...")

    output_data = {
        'hypothesis': 'H5',
        'description': 'Bag Size vs Label Reliability',
        'passed': hypothesis_passed,
        'statistics': {
            'cohens_d': cohens_result['d'],
            't_statistic': t_test['t_statistic'],
            'p_value': t_test['p_value'],
            'pearson_r': correlation['r']
        },
        'group_analysis': {
            'size_1': {
                'count': len(size_1_bags),
                'reliability': mean_rel_1
            },
            'size_2': {
                'count': len(size_2_bags),
                'reliability': mean_rel_2
            },
            'size_3plus': {
                'count': len(size_3plus_bags),
                'reliability': mean_rel_3plus
            }
        },
        'relation_analysis': relation_effects[:10],
        'size_trend': size_reliability,
        'threshold': {
            'cohens_d': [0.5, 0.8]
        }
    }

    with open(os.path.join(output_dir, 'h5_results.json'), 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"  Results saved to: {output_dir}/h5_results.json")
    print(f"  Figures saved to: {output_dir}/figures/")

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Verify Hypothesis 5: Bag Size vs Label Reliability"
    )
    parser.add_argument(
        '--data_dir', type=str, default='./nyth',
        help='Path to NYT-H data directory'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./results/h5',
        help='Output directory for results'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    results = run_bag_reliability_experiment(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed
    )

    sys.exit(0 if results['passed'] else 1)


if __name__ == '__main__':
    main()
