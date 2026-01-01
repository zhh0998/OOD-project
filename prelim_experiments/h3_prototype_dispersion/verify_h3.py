#!/usr/bin/env python3
"""
Hypothesis 3 Verification: Prototype Dispersion Index vs Noise Rate

H3: Prototype Dispersion Index (PDI = trace(Σ)/d) positively correlates
    with noise rate in the training data (Pearson r > 0.5)

Methodology:
1. Train Gaussian prototype on clean data
2. Inject varying levels of symmetric noise
3. Measure PDI for each noise level
4. Compute correlation between noise rate and PDI
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.nyt10_loader import load_nyt10, NYT10Dataset
from src.models.gaussian_prototype import (
    GaussianPrototype,
    inject_symmetric_noise,
    verify_pdi_noise_correlation
)
from src.utils.statistics import (
    compute_pearson_correlation,
    HypothesisTestResult
)
from src.utils.visualization import (
    plot_fitted_line,
    plot_grouped_boxplot,
    set_publication_style
)


def run_pdi_noise_experiment(
    data_dir: str = './nyt10',
    noise_rates: List[float] = None,
    n_runs: int = 5,
    output_dir: str = './results/h3',
    seed: int = 42
) -> Dict:
    """
    Run the complete H3 verification experiment

    Args:
        data_dir: Path to NYT10 data
        noise_rates: List of noise rates to test
        n_runs: Number of experiment runs
        output_dir: Output directory
        seed: Random seed

    Returns:
        Experiment results
    """
    if noise_rates is None:
        noise_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)

    print("=" * 60)
    print("Hypothesis 3 Verification: PDI vs Noise Rate")
    print("=" * 60)

    # Step 1: Load data
    print("\n[Step 1] Loading NYT10 dataset...")
    dataset = NYT10Dataset(data_dir)
    dataset.load_train()

    sentences = [s.sentence for s in dataset.train_samples]
    labels = [s.relation for s in dataset.train_samples]

    print(f"  - Training samples: {len(sentences)}")
    print(f"  - Number of relations: {len(set(labels))}")

    # Step 2: Run experiments across noise levels
    print(f"\n[Step 2] Testing {len(noise_rates)} noise levels...")

    all_results = []
    all_noise = []
    all_pdi = []

    for run in range(n_runs):
        print(f"\n  Run {run + 1}/{n_runs}:")
        run_noise = []
        run_pdi = []

        for noise_rate in noise_rates:
            # Inject noise
            noisy_labels = inject_symmetric_noise(
                labels, noise_rate, seed=seed + run * 100 + int(noise_rate * 100)
            )

            # Train Gaussian prototype
            model = GaussianPrototype(embedding_dim=min(256, len(sentences)))
            model.fit(sentences, noisy_labels)

            # Compute PDI for all relations
            pdi_scores = model.compute_pdi()
            avg_pdi = np.mean(list(pdi_scores.values()))
            std_pdi = np.std(list(pdi_scores.values()))

            run_noise.append(noise_rate)
            run_pdi.append(avg_pdi)
            all_noise.append(noise_rate)
            all_pdi.append(avg_pdi)

            print(f"    Noise={noise_rate:.1f}: PDI={avg_pdi:.4f} (±{std_pdi:.4f})")

        all_results.append({
            'run': run + 1,
            'noise_rates': run_noise,
            'pdi_values': run_pdi
        })

    # Step 3: Statistical analysis
    print("\n[Step 3] Computing correlation...")

    correlation = compute_pearson_correlation(all_noise, all_pdi)
    print(f"  Pearson r = {correlation['r']:.4f}")
    print(f"  p-value = {correlation['p_value']:.6f}")
    print(f"  Interpretation: {correlation['interpretation']}")

    # Step 4: Group analysis (low noise vs high noise PDI)
    print("\n[Step 4] Group analysis...")
    low_noise_pdi = [all_pdi[i] for i in range(len(all_noise))
                     if all_noise[i] <= 0.2]
    high_noise_pdi = [all_pdi[i] for i in range(len(all_noise))
                      if all_noise[i] >= 0.3]

    from src.utils.statistics import compute_cohens_d
    group_diff = compute_cohens_d(high_noise_pdi, low_noise_pdi)

    print(f"  Low noise (≤0.2) mean PDI: {np.mean(low_noise_pdi):.4f}")
    print(f"  High noise (≥0.3) mean PDI: {np.mean(high_noise_pdi):.4f}")
    print(f"  Cohen's d: {group_diff['d']:.4f}")

    # Step 5: Hypothesis test
    hypothesis_passed = correlation['r'] > 0.5 and correlation['p_value'] < 0.05

    result = HypothesisTestResult(
        hypothesis_id="H3",
        description="Prototype Dispersion Index (PDI) correlates with noise rate",
        passed=hypothesis_passed,
        statistics={
            'pearson_r': correlation['r'],
            'p_value': correlation['p_value'],
            'cohens_d': group_diff['d'],
            'low_noise_mean_pdi': np.mean(low_noise_pdi),
            'high_noise_mean_pdi': np.mean(high_noise_pdi),
            'n_samples': len(all_noise)
        },
        threshold={
            'pearson_r': '> 0.5',
            'p_value': '< 0.05'
        },
        details=f"Tested {n_runs} runs across {len(noise_rates)} noise levels"
    )

    print("\n" + "=" * 60)
    print(result.summary())
    print("=" * 60)

    # Step 6: Visualization
    print("\n[Step 6] Generating visualizations...")

    # Fitted line plot
    fig1 = plot_fitted_line(
        all_noise, all_pdi,
        xlabel="Noise Rate",
        ylabel="Prototype Dispersion Index (PDI)",
        title="H3: Noise Rate vs Prototype Dispersion Index",
        save_path=os.path.join(output_dir, 'figures', 'h3_fitted_line.png')
    )

    # Group boxplot
    fig2 = plot_grouped_boxplot(
        [low_noise_pdi, high_noise_pdi],
        group_labels=['Low Noise (≤0.2)', 'High Noise (≥0.3)'],
        ylabel='PDI (trace(Σ)/d)',
        title='H3: PDI by Noise Level Group',
        save_path=os.path.join(output_dir, 'figures', 'h3_boxplot.png')
    )

    # Per-relation analysis
    print("\n[Step 7] Per-relation analysis...")

    # Train on clean data
    model_clean = GaussianPrototype()
    model_clean.fit(sentences, labels)
    pdi_clean = model_clean.compute_pdi()

    # Train on noisy data (0.3 noise)
    noisy_labels_30 = inject_symmetric_noise(labels, 0.3, seed=seed)
    model_noisy = GaussianPrototype()
    model_noisy.fit(sentences, noisy_labels_30)
    pdi_noisy = model_noisy.compute_pdi()

    # Find relations most affected by noise
    pdi_increase = {}
    for rel_id in pdi_clean:
        if rel_id in pdi_noisy:
            pdi_increase[rel_id] = pdi_noisy[rel_id] - pdi_clean[rel_id]

    sorted_relations = sorted(pdi_increase.items(), key=lambda x: -x[1])
    print("  Top 5 relations most affected by noise:")
    for rel_id, increase in sorted_relations[:5]:
        print(f"    Relation {rel_id}: PDI increase = {increase:.4f}")

    # Step 8: Save results
    print("\n[Step 8] Saving results...")
    output_data = {
        'hypothesis': 'H3',
        'description': 'PDI vs Noise Rate correlation',
        'passed': hypothesis_passed,
        'correlation': {
            'pearson_r': correlation['r'],
            'p_value': correlation['p_value']
        },
        'group_analysis': {
            'low_noise_mean_pdi': float(np.mean(low_noise_pdi)),
            'high_noise_mean_pdi': float(np.mean(high_noise_pdi)),
            'cohens_d': group_diff['d']
        },
        'runs': all_results,
        'per_relation_analysis': {
            str(k): float(v) for k, v in sorted_relations[:10]
        },
        'threshold': {
            'pearson_r': 0.5,
            'p_value': 0.05
        }
    }

    with open(os.path.join(output_dir, 'h3_results.json'), 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"  Results saved to: {output_dir}/h3_results.json")
    print(f"  Figures saved to: {output_dir}/figures/")

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Verify Hypothesis 3: PDI vs Noise Rate"
    )
    parser.add_argument(
        '--data_dir', type=str, default='./nyt10',
        help='Path to NYT10 data directory'
    )
    parser.add_argument(
        '--noise_rates', type=str, default='0.0,0.1,0.2,0.3,0.4,0.5',
        help='Comma-separated noise rates to test'
    )
    parser.add_argument(
        '--n_runs', type=int, default=5,
        help='Number of experiment runs'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./results/h3',
        help='Output directory for results'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )

    args = parser.parse_args()
    noise_rates = [float(x) for x in args.noise_rates.split(',')]

    results = run_pdi_noise_experiment(
        data_dir=args.data_dir,
        noise_rates=noise_rates,
        n_runs=args.n_runs,
        output_dir=args.output_dir,
        seed=args.seed
    )

    sys.exit(0 if results['passed'] else 1)


if __name__ == '__main__':
    main()
