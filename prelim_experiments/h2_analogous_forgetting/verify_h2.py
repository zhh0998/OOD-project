#!/usr/bin/env python3
"""
Hypothesis 2 Verification: Analogous Relation Similarity vs Forgetting Rate

H2: Analogous Relation Similarity (ARS) positively correlates with
    catastrophic forgetting rate in continual learning
    (Spearman rho > 0.5, Cohen's d > 0.5)

Methodology:
1. Compute ARS matrix for all relation pairs (using LLM or semantic groups)
2. Run continual learning simulation with baseline prototype network
3. Compute forgetting rate for each task
4. Measure correlation between average ARS and forgetting rate
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import spearmanr

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.fewrel_loader import FewRelDataset, load_fewrel
from src.models.prototype_network import (
    PrototypeNetwork,
    ContinualLearningSimulator,
    compute_ars_matrix,
    verify_ars_forgetting_correlation
)
from src.utils.statistics import (
    compute_spearman_correlation,
    compute_cohens_d,
    HypothesisTestResult
)
from src.utils.visualization import (
    plot_correlation_scatter,
    plot_grouped_boxplot,
    set_publication_style
)


def compute_ars_with_llm(
    rel1: str,
    rel2: str,
    relation_descriptions: Dict[str, str],
    use_cache: bool = True,
    cache_path: str = './ars_cache.json'
) -> float:
    """
    Compute Analogous Relation Similarity using LLM (simulated)

    In production, this would call OpenAI API with a prompt like:
    "Rate the semantic similarity between '{rel1}' and '{rel2}' from 0 to 1"

    For this experiment, we use pre-defined semantic groupings.

    Args:
        rel1, rel2: Relation identifiers
        relation_descriptions: Mapping from relation ID to description
        use_cache: Whether to use cached results
        cache_path: Path to cache file

    Returns:
        ARS score between 0 and 1
    """
    # Check cache
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            cache = json.load(f)
        key = f"{min(rel1, rel2)}_{max(rel1, rel2)}"
        if key in cache:
            return cache[key]

    # Semantic similarity based on description overlap
    desc1 = relation_descriptions.get(rel1, '').lower().split()
    desc2 = relation_descriptions.get(rel2, '').lower().split()

    if not desc1 or not desc2:
        return np.random.uniform(0, 0.3)

    # Jaccard similarity
    set1, set2 = set(desc1), set(desc2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    jaccard = intersection / union if union > 0 else 0

    # Add semantic group bonus
    semantic_groups = [
        {'country', 'nation', 'citizenship', 'origin'},
        {'birth', 'death', 'place', 'residence', 'location'},
        {'father', 'mother', 'child', 'parent', 'spouse', 'family'},
        {'author', 'director', 'creator', 'composer', 'writer'},
        {'company', 'employer', 'founder', 'owner', 'subsidiary'},
        {'political', 'party', 'position', 'government'},
    ]

    for group in semantic_groups:
        if (set1 & group) and (set2 & group):
            jaccard += 0.3

    ars = min(jaccard, 1.0)

    # Add noise
    ars = np.clip(ars + np.random.uniform(-0.1, 0.1), 0, 1)

    return ars


def run_ars_forgetting_experiment(
    data_dir: str = './fewrel',
    n_tasks: int = 10,
    relations_per_task: int = 8,
    n_runs: int = 5,
    output_dir: str = './results/h2',
    seed: int = 42
) -> Dict:
    """
    Run the complete H2 verification experiment

    Args:
        data_dir: Path to FewRel data
        n_tasks: Number of continual learning tasks
        relations_per_task: Relations per task
        n_runs: Number of experiment runs
        output_dir: Output directory
        seed: Random seed

    Returns:
        Experiment results
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)

    print("=" * 60)
    print("Hypothesis 2 Verification: ARS vs Forgetting Rate")
    print("=" * 60)

    # Step 1: Load dataset and relations
    print("\n[Step 1] Loading FewRel dataset...")
    dataset = FewRelDataset(data_dir)
    dataset.load_train()

    relations = dataset.relations[:n_tasks * relations_per_task]
    print(f"  - Number of relations: {len(relations)}")
    print(f"  - Tasks: {n_tasks} x {relations_per_task} relations")

    # Step 2: Compute ARS matrix
    print("\n[Step 2] Computing Analogous Relation Similarity (ARS) matrix...")
    ars_matrix = {}

    for i, rel1 in enumerate(relations):
        for rel2 in relations[i+1:]:
            ars = compute_ars_with_llm(
                rel1, rel2,
                dataset.RELATION_DESCRIPTIONS
            )
            key = (rel1, rel2) if rel1 < rel2 else (rel2, rel1)
            ars_matrix[key] = ars

    avg_ars = np.mean(list(ars_matrix.values()))
    print(f"  - Computed {len(ars_matrix)} ARS pairs")
    print(f"  - Average ARS: {avg_ars:.4f}")

    # Step 3: Run continual learning simulation
    print(f"\n[Step 3] Running {n_runs} continual learning simulations...")

    all_ars_values = []
    all_fr_values = []
    run_results = []

    for run in range(n_runs):
        print(f"\n  Run {run + 1}/{n_runs}:")

        simulator = ContinualLearningSimulator(
            n_tasks=n_tasks,
            relations_per_task=relations_per_task,
            seed=seed + run
        )

        # Generate data
        data = simulator.generate_synthetic_data(relations)

        # Split into tasks
        tasks = simulator.split_relations_into_tasks(relations)

        # Run continual learning
        accuracy_matrix = simulator.run_continual_learning(data, tasks)

        # Compute forgetting rates
        forgetting_rates = simulator.compute_forgetting_rates()

        # Collect ARS vs FR pairs for this run
        run_ars = []
        run_fr = []

        for task_id, fr in forgetting_rates.items():
            avg_task_ars = simulator.compute_ars_with_future_tasks(task_id, ars_matrix)
            run_ars.append(avg_task_ars)
            run_fr.append(fr)
            all_ars_values.append(avg_task_ars)
            all_fr_values.append(fr)

        # Compute run-level correlation
        if len(run_ars) > 2:
            rho, p = spearmanr(run_ars, run_fr)
            print(f"    Spearman rho = {rho:.4f}, p = {p:.4f}")

        run_results.append({
            'run': run + 1,
            'ars_values': run_ars,
            'fr_values': run_fr,
            'accuracy_matrix': accuracy_matrix.tolist()
        })

    # Step 4: Statistical analysis
    print("\n[Step 4] Computing overall correlation...")

    correlation = compute_spearman_correlation(all_ars_values, all_fr_values)
    print(f"  Spearman rho = {correlation['rho']:.4f}")
    print(f"  p-value = {correlation['p_value']:.6f}")

    # Compute Cohen's d (high ARS vs low ARS)
    median_ars = np.median(all_ars_values)
    high_ars_fr = [all_fr_values[i] for i in range(len(all_ars_values))
                   if all_ars_values[i] >= median_ars]
    low_ars_fr = [all_fr_values[i] for i in range(len(all_ars_values))
                  if all_ars_values[i] < median_ars]

    cohens_d_result = compute_cohens_d(high_ars_fr, low_ars_fr)
    print(f"  Cohen's d = {cohens_d_result['d']:.4f}")
    print(f"  High ARS group mean FR: {np.mean(high_ars_fr):.4f}")
    print(f"  Low ARS group mean FR: {np.mean(low_ars_fr):.4f}")

    # Step 5: Hypothesis test
    hypothesis_passed = (
        correlation['rho'] > 0.5 and
        abs(cohens_d_result['d']) > 0.5 and
        correlation['p_value'] < 0.05
    )

    result = HypothesisTestResult(
        hypothesis_id="H2",
        description="Analogous Relation Similarity correlates with Forgetting Rate",
        passed=hypothesis_passed,
        statistics={
            'spearman_rho': correlation['rho'],
            'p_value': correlation['p_value'],
            'cohens_d': cohens_d_result['d'],
            'high_ars_mean_fr': np.mean(high_ars_fr),
            'low_ars_mean_fr': np.mean(low_ars_fr),
            'n_samples': len(all_ars_values)
        },
        threshold={
            'spearman_rho': '> 0.5',
            'cohens_d': '> 0.5',
            'p_value': '< 0.05'
        },
        details=f"Tested {n_runs} runs with {n_tasks} tasks each"
    )

    print("\n" + "=" * 60)
    print(result.summary())
    print("=" * 60)

    # Step 6: Visualization
    print("\n[Step 6] Generating visualizations...")

    # Scatter plot with correlation
    fig1 = plot_correlation_scatter(
        all_ars_values, all_fr_values,
        xlabel="Average ARS with Future Tasks",
        ylabel="Forgetting Rate",
        title="H2: Analogous Relation Similarity vs Forgetting Rate",
        correlation_type='spearman',
        save_path=os.path.join(output_dir, 'figures', 'h2_correlation.png')
    )

    # Grouped boxplot (high vs low ARS)
    fig2 = plot_grouped_boxplot(
        [low_ars_fr, high_ars_fr],
        group_labels=['Low ARS', 'High ARS'],
        ylabel='Forgetting Rate',
        title='H2: Forgetting Rate by ARS Group',
        save_path=os.path.join(output_dir, 'figures', 'h2_boxplot.png')
    )

    # Step 7: Save results
    print("\n[Step 7] Saving results...")
    output_data = {
        'hypothesis': 'H2',
        'description': 'ARS vs Forgetting Rate correlation',
        'passed': hypothesis_passed,
        'correlation': {
            'spearman_rho': correlation['rho'],
            'p_value': correlation['p_value'],
            'cohens_d': cohens_d_result['d']
        },
        'group_analysis': {
            'high_ars_mean_fr': float(np.mean(high_ars_fr)),
            'low_ars_mean_fr': float(np.mean(low_ars_fr)),
            'median_ars': float(median_ars)
        },
        'runs': run_results,
        'threshold': {
            'spearman_rho': 0.5,
            'cohens_d': 0.5,
            'p_value': 0.05
        }
    }

    with open(os.path.join(output_dir, 'h2_results.json'), 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"  Results saved to: {output_dir}/h2_results.json")
    print(f"  Figures saved to: {output_dir}/figures/")

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Verify Hypothesis 2: ARS vs Forgetting Rate"
    )
    parser.add_argument(
        '--data_dir', type=str, default='./fewrel',
        help='Path to FewRel data directory'
    )
    parser.add_argument(
        '--n_tasks', type=int, default=10,
        help='Number of continual learning tasks'
    )
    parser.add_argument(
        '--relations_per_task', type=int, default=8,
        help='Number of relations per task'
    )
    parser.add_argument(
        '--n_runs', type=int, default=5,
        help='Number of experiment runs'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./results/h2',
        help='Output directory for results'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    results = run_ars_forgetting_experiment(
        data_dir=args.data_dir,
        n_tasks=args.n_tasks,
        relations_per_task=args.relations_per_task,
        n_runs=args.n_runs,
        output_dir=args.output_dir,
        seed=args.seed
    )

    sys.exit(0 if results['passed'] else 1)


if __name__ == '__main__':
    main()
