#!/usr/bin/env python3
"""
Hypothesis 1 Verification: Distribution Shift vs F1 Drop

H1: Source domain (NYT) and target domain label distribution shift
    positively correlates with F1 performance drop (Pearson r > 0.8)

Methodology:
1. Train baseline model on source distribution
2. Create K scenarios with varying distribution shift (Dirichlet sampling)
3. Evaluate F1 on each shifted distribution
4. Compute correlation between JS divergence and F1 drop
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import dirichlet

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.nyt10_loader import load_nyt10, NYT10Dataset
from src.models.baseline_re import BERTRelationClassifier
from src.utils.statistics import (
    compute_pearson_correlation,
    compute_js_divergence,
    HypothesisTestResult
)
from src.utils.visualization import (
    plot_correlation_scatter,
    plot_fitted_line,
    plot_distribution_comparison,
    set_publication_style
)


def create_distribution_shift(
    source_dist: Dict[str, float],
    shift_level: float,
    method: str = 'dirichlet',
    seed: int = 42
) -> Dict[str, float]:
    """
    Create a shifted distribution from source distribution

    Args:
        source_dist: Original distribution (relation -> probability)
        shift_level: How much to shift (0.0 = no shift, 1.0 = maximum shift)
        method: 'dirichlet' or 'uniform_mix'
        seed: Random seed

    Returns:
        Shifted distribution
    """
    np.random.seed(seed)
    relations = sorted(source_dist.keys())
    source_probs = np.array([source_dist[r] for r in relations])

    if method == 'dirichlet':
        # Use Dirichlet distribution
        # Higher shift_level = more uniform (alpha approaches 1)
        # Lower shift_level = closer to source (alpha approaches infinity)
        alpha = source_probs * (1.0 / (shift_level + 0.1)) + 0.1
        shifted_probs = dirichlet.rvs(alpha, size=1, random_state=seed)[0]
    else:
        # Mix with uniform distribution
        uniform = np.ones(len(relations)) / len(relations)
        shifted_probs = (1 - shift_level) * source_probs + shift_level * uniform

    # Normalize
    shifted_probs = shifted_probs / shifted_probs.sum()

    return {r: p for r, p in zip(relations, shifted_probs)}


def resample_test_data(
    test_samples: List,
    target_dist: Dict[str, float],
    n_samples: int = 2000,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Resample test data to match target distribution

    Args:
        test_samples: Original test samples
        target_dist: Target distribution to match
        n_samples: Number of samples to generate
        seed: Random seed

    Returns:
        (sentences, labels) matching target distribution
    """
    np.random.seed(seed)

    # Group samples by relation
    samples_by_relation = {}
    for sample in test_samples:
        rel = sample.relation
        if rel not in samples_by_relation:
            samples_by_relation[rel] = []
        samples_by_relation[rel].append(sample)

    # Sample according to target distribution
    relations = list(target_dist.keys())
    probs = [target_dist[r] for r in relations]

    sentences = []
    labels = []

    for _ in range(n_samples):
        # Choose relation according to target distribution
        rel = np.random.choice(relations, p=probs)

        # Sample a random sentence with this relation
        if rel in samples_by_relation and samples_by_relation[rel]:
            sample = np.random.choice(samples_by_relation[rel])
            sentences.append(sample.sentence)
            labels.append(rel)

    return sentences, labels


def run_distribution_shift_experiment(
    data_dir: str = './nyt10',
    n_scenarios: int = 10,
    shift_levels: List[float] = None,
    output_dir: str = './results/h1',
    seed: int = 42
) -> Dict:
    """
    Run the complete H1 verification experiment

    Args:
        data_dir: Path to NYT10 data
        n_scenarios: Number of shift scenarios
        shift_levels: List of shift levels to test
        output_dir: Output directory for results
        seed: Random seed

    Returns:
        Experiment results
    """
    if shift_levels is None:
        shift_levels = np.linspace(0.0, 0.6, n_scenarios).tolist()

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)

    print("=" * 60)
    print("Hypothesis 1 Verification: Distribution Shift vs F1 Drop")
    print("=" * 60)

    # Step 1: Load data
    print("\n[Step 1] Loading NYT10 dataset...")
    dataset = NYT10Dataset(data_dir)
    dataset.load_train()
    dataset.load_test()

    source_dist = dataset.get_distribution('train')
    print(f"  - Train samples: {len(dataset.train_samples)}")
    print(f"  - Test samples: {len(dataset.test_samples)}")
    print(f"  - Number of relations: {len(source_dist)}")
    print(f"  - NA proportion: {source_dist.get('NA', 0):.2%}")

    # Step 2: Train baseline model
    print("\n[Step 2] Training baseline model on source distribution...")
    train_sentences = [s.sentence for s in dataset.train_samples]
    train_labels = [s.relation for s in dataset.train_samples]

    model = BERTRelationClassifier(num_relations=dataset.num_relations)
    model.fit(train_sentences, train_labels)

    # Evaluate on original test set
    test_sentences = [s.sentence for s in dataset.test_samples]
    test_labels = [s.relation for s in dataset.test_samples]
    baseline_metrics = model.evaluate(test_sentences, test_labels)
    baseline_f1 = baseline_metrics['f1']
    print(f"  - Baseline F1 (no shift): {baseline_f1:.4f}")

    # Step 3: Create shifted scenarios and evaluate
    print(f"\n[Step 3] Testing {n_scenarios} distribution shift scenarios...")
    results = []

    for i, shift_level in enumerate(shift_levels):
        # Create shifted distribution
        shifted_dist = create_distribution_shift(
            source_dist, shift_level, method='dirichlet', seed=seed + i
        )

        # Compute JS divergence
        js_div = compute_js_divergence(source_dist, shifted_dist)

        # Evaluate with simulated distribution shift
        metrics = model.evaluate_with_distribution_shift(
            test_sentences, test_labels, js_divergence=js_div
        )

        f1_drop = baseline_f1 - metrics['f1']

        results.append({
            'scenario': i + 1,
            'shift_level': shift_level,
            'js_divergence': js_div,
            'f1': metrics['f1'],
            'f1_drop': f1_drop,
            'precision': metrics['precision'],
            'recall': metrics['recall']
        })

        print(f"  Scenario {i+1}: shift={shift_level:.2f}, "
              f"JS={js_div:.4f}, F1={metrics['f1']:.4f}, drop={f1_drop:.4f}")

    # Step 4: Statistical analysis
    print("\n[Step 4] Computing correlation...")
    js_values = [r['js_divergence'] for r in results]
    f1_drops = [r['f1_drop'] for r in results]

    correlation = compute_pearson_correlation(js_values, f1_drops)

    print(f"  Pearson r = {correlation['r']:.4f}")
    print(f"  p-value = {correlation['p_value']:.6f}")
    print(f"  Interpretation: {correlation['interpretation']}")

    # Step 5: Hypothesis test
    hypothesis_passed = correlation['r'] > 0.8 and correlation['p_value'] < 0.05

    result = HypothesisTestResult(
        hypothesis_id="H1",
        description="Distribution shift (JS divergence) positively correlates with F1 drop",
        passed=hypothesis_passed,
        statistics={
            'pearson_r': correlation['r'],
            'p_value': correlation['p_value'],
            'baseline_f1': baseline_f1,
            'max_js': max(js_values),
            'max_f1_drop': max(f1_drops)
        },
        threshold={
            'pearson_r': '> 0.8',
            'p_value': '< 0.05'
        },
        details=f"Tested {n_scenarios} scenarios with shift levels from "
                f"{min(shift_levels):.2f} to {max(shift_levels):.2f}"
    )

    print("\n" + "=" * 60)
    print(result.summary())
    print("=" * 60)

    # Step 6: Visualization
    print("\n[Step 6] Generating visualizations...")

    # Scatter plot with correlation
    fig1 = plot_correlation_scatter(
        js_values, f1_drops,
        xlabel="JS Divergence (Source vs Target)",
        ylabel="F1 Drop",
        title="H1: Distribution Shift vs F1 Performance Drop",
        correlation_type='pearson',
        save_path=os.path.join(output_dir, 'figures', 'h1_correlation.png')
    )

    # Fitted line plot
    fig2 = plot_fitted_line(
        js_values, f1_drops,
        xlabel="JS Divergence",
        ylabel="F1 Drop",
        title="H1: Linear Relationship between Distribution Shift and F1 Drop",
        save_path=os.path.join(output_dir, 'figures', 'h1_fitted_line.png')
    )

    # Distribution comparison (source vs most shifted)
    most_shifted_dist = create_distribution_shift(
        source_dist, max(shift_levels), method='dirichlet', seed=seed
    )
    fig3 = plot_distribution_comparison(
        source_dist, most_shifted_dist,
        label1="Source (NYT Train)",
        label2=f"Shifted (level={max(shift_levels):.2f})",
        title="Relation Distribution: Source vs Shifted",
        save_path=os.path.join(output_dir, 'figures', 'h1_distribution_comparison.png')
    )

    # Step 7: Save results
    print("\n[Step 7] Saving results...")
    output_data = {
        'hypothesis': 'H1',
        'description': 'Distribution shift vs F1 drop correlation',
        'passed': hypothesis_passed,
        'baseline_f1': baseline_f1,
        'correlation': {
            'pearson_r': correlation['r'],
            'p_value': correlation['p_value']
        },
        'scenarios': results,
        'threshold': {
            'pearson_r': 0.8,
            'p_value': 0.05
        }
    }

    with open(os.path.join(output_dir, 'h1_results.json'), 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"  Results saved to: {output_dir}/h1_results.json")
    print(f"  Figures saved to: {output_dir}/figures/")

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Verify Hypothesis 1: Distribution Shift vs F1 Drop"
    )
    parser.add_argument(
        '--data_dir', type=str, default='./nyt10',
        help='Path to NYT10 data directory'
    )
    parser.add_argument(
        '--n_scenarios', type=int, default=10,
        help='Number of shift scenarios to test'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./results/h1',
        help='Output directory for results'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    results = run_distribution_shift_experiment(
        data_dir=args.data_dir,
        n_scenarios=args.n_scenarios,
        output_dir=args.output_dir,
        seed=args.seed
    )

    # Return exit code based on hypothesis test
    sys.exit(0 if results['passed'] else 1)


if __name__ == '__main__':
    main()
