#!/usr/bin/env python3
"""
H5 Fixed: Bag Size vs Label Reliability

Fix: Use proportion of 'yes' labels as reliability, not label consistency.

NYT-H bag_label meanings:
- 'yes': relation is correct (human verified)
- 'no': relation is incorrect
- 'unk': uncertain
"""
import json
import numpy as np
from scipy import stats
from collections import defaultdict
import os

def run_h5_fixed():
    print("=" * 60)
    print("H5 FIXED: Bag Size → Label Reliability")
    print("Using 'yes' label proportion as reliability metric")
    print("=" * 60)

    # Load NYT-H data
    data_path = '/home/user/OOD-project/data/nyth/data/test.json'
    print(f"\nLoading data from: {data_path}")

    with open(data_path) as f:
        data = [json.loads(line) for line in f]

    print(f"Total instances: {len(data)}")

    # Group by bag_id
    bags = defaultdict(list)
    for sample in data:
        bags[sample['bag_id']].append(sample)

    print(f"Total bags: {len(bags)}")

    # Analyze bag_label distribution
    all_labels = [s['bag_label'] for s in data]
    label_counts = defaultdict(int)
    for label in all_labels:
        label_counts[label] += 1

    print(f"\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} ({count/len(data)*100:.1f}%)")

    # Calculate reliability per bag
    # Reliability = proportion of 'yes' labels in the bag
    results = []
    for bag_id, samples in bags.items():
        bag_size = len(samples)
        yes_count = sum(1 for s in samples if s['bag_label'] == 'yes')
        no_count = sum(1 for s in samples if s['bag_label'] == 'no')
        unk_count = sum(1 for s in samples if s['bag_label'] == 'unk')

        # Reliability = yes / (yes + no), excluding 'unk'
        labeled_count = yes_count + no_count
        if labeled_count > 0:
            reliability = yes_count / labeled_count
        else:
            reliability = 0.5  # Default for all 'unk'

        results.append({
            'bag_id': bag_id,
            'bag_size': bag_size,
            'yes_count': yes_count,
            'no_count': no_count,
            'unk_count': unk_count,
            'reliability': reliability
        })

    # Group statistics by bag size
    print("\n" + "=" * 60)
    print("Bag Size vs Reliability Statistics")
    print("=" * 60)

    size_stats = defaultdict(list)
    for r in results:
        size = r['bag_size']
        if size >= 5:
            size_key = '5+'
        else:
            size_key = str(size)
        size_stats[size_key].append(r['reliability'])

    for size_key in ['1', '2', '3', '4', '5+']:
        if size_key in size_stats:
            values = size_stats[size_key]
            print(f"Size {size_key}: {len(values):4d} bags, "
                  f"mean={np.mean(values):.4f}, "
                  f"std={np.std(values):.4f}, "
                  f"median={np.median(values):.4f}")

    # Cohen's d: Size=1 vs Size≥3
    size_1 = [r['reliability'] for r in results if r['bag_size'] == 1]
    size_3plus = [r['reliability'] for r in results if r['bag_size'] >= 3]

    print(f"\n" + "=" * 60)
    print("Cohen's d Analysis: Size=1 vs Size≥3")
    print("=" * 60)

    n1, n2 = len(size_1), len(size_3plus)
    mean_1, mean_3 = np.mean(size_1), np.mean(size_3plus)
    std_1, std_3 = np.std(size_1, ddof=1), np.std(size_3plus, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std_1**2 + (n2 - 1) * std_3**2) / (n1 + n2 - 2))

    # Cohen's d (positive if larger bags more reliable)
    cohens_d = (mean_3 - mean_1) / pooled_std if pooled_std > 0 else 0

    # T-test
    t_stat, p_value = stats.ttest_ind(size_3plus, size_1)

    print(f"Size=1:  n={n1:4d}, mean={mean_1:.4f}, std={std_1:.4f}")
    print(f"Size≥3: n={n2:4d}, mean={mean_3:.4f}, std={std_3:.4f}")
    print(f"\nCohen's d: {cohens_d:.4f}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.6f}")

    # Interpretation
    print(f"\n" + "=" * 60)
    print("Interpretation")
    print("=" * 60)

    if cohens_d > 0:
        print(f"Direction: Larger bags have HIGHER reliability (as expected)")
    else:
        print(f"Direction: Larger bags have LOWER reliability (unexpected)")

    if 0.5 <= abs(cohens_d) <= 0.8:
        verdict = "PASSED"
        print(f"Effect size: Medium ({abs(cohens_d):.2f} in range [0.5, 0.8])")
    elif abs(cohens_d) < 0.5:
        verdict = "FAILED (effect too small)"
        print(f"Effect size: Small ({abs(cohens_d):.2f} < 0.5)")
    else:
        verdict = "FAILED (effect too large)"
        print(f"Effect size: Large ({abs(cohens_d):.2f} > 0.8)")

    passed = 0.5 <= abs(cohens_d) <= 0.8
    print(f"\nVerdict: {verdict}")

    # Save results
    result = {
        'hypothesis': 'H5',
        'description': 'Bag Size → Label Reliability (FIXED)',
        'fix_applied': 'Use yes/(yes+no) as reliability instead of label consistency',
        'data_source': 'REAL NYT-H',
        'total_bags': len(bags),
        'size_1_count': n1,
        'size_3plus_count': n2,
        'mean_reliability_size_1': float(mean_1),
        'mean_reliability_size_3plus': float(mean_3),
        'cohens_d': float(cohens_d),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'threshold': '0.5-0.8',
        'passed': bool(passed),
        'verdict': verdict
    }

    output_dir = '/home/user/OOD-project/results/h5_fixed'
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'h5_results.json'), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to: {output_dir}/h5_results.json")

    return result

if __name__ == '__main__':
    run_h5_fixed()
