#!/usr/bin/env python3
"""
Evaluation Script for RW2 Temporal Network Embedding.

Evaluates trained models and computes comprehensive metrics including:
- MRR, Hits@K
- Cohen's d effect size
- Statistical significance tests

Usage:
    python evaluate.py --model ssm_memory_llm --checkpoint checkpoints/best.pth
    python evaluate.py --compare baseline ssm_memory_llm tpnet dygprompt

Author: RW2 Temporal Network Embedding Project
"""

import argparse
import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.data_loader import RealDataLoader
from models.base_model import TempMemLLM
from models.ssm_memory_llm import SSMMemoryLLM
from models.tpnet_llm import TPNetLLM
from models.dygprompt import StandaloneDyGPrompt
from utils.metrics import (
    compute_metrics, compute_cohen_d, independent_t_test,
    StatisticalAnalysis
)
from utils.negative_sampling import create_negative_sampler


def load_model(model_name: str, checkpoint_path: str, num_nodes: int, device: torch.device):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint.get('config', {})

    if model_name == 'ssm_memory_llm':
        model = SSMMemoryLLM(num_nodes=num_nodes, **config)
    elif model_name == 'tpnet':
        model = TPNetLLM(num_nodes=num_nodes, **config)
    elif model_name == 'dygprompt':
        model = StandaloneDyGPrompt(num_nodes=num_nodes, **config)
    else:
        model = TempMemLLM(num_nodes=num_nodes, **config)

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, config


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataset,
    neg_sampler,
    device: torch.device,
    num_samples: int = 2000
) -> Dict[str, float]:
    """Evaluate a single model."""
    model.eval()

    eval_edges = getattr(dataset, 'eval_edges', dataset.edges)

    if len(eval_edges) > num_samples:
        indices = np.random.choice(len(eval_edges), num_samples, replace=False)
        eval_edges = [eval_edges[i] for i in indices]

    all_ranks = []

    for edge in tqdm(eval_edges, desc="Evaluating", leave=False):
        src = torch.tensor([edge.src], device=device)
        dst = torch.tensor([edge.dst], device=device)
        timestamp = torch.tensor([edge.timestamp], device=device)

        src_neighbors, src_times = dataset.get_temporal_neighbors(edge.src, edge.timestamp)
        dst_neighbors, dst_times = dataset.get_temporal_neighbors(edge.dst, edge.timestamp)

        max_len = 64
        def pad(seq, val=0):
            if len(seq) >= max_len:
                return seq[-max_len:]
            return [val] * (max_len - len(seq)) + seq

        src_neighbor_seq = torch.tensor([pad(src_neighbors)], device=device)
        src_time_seq = torch.tensor([pad(src_times, 0.0)], device=device)
        dst_neighbor_seq = torch.tensor([pad(dst_neighbors)], device=device)
        dst_time_seq = torch.tensor([pad(dst_times, 0.0)], device=device)

        neg_dst = neg_sampler.sample(edge.src, edge.dst, edge.timestamp)
        neg_dst = torch.tensor(neg_dst[:100].reshape(1, -1), device=device)

        output = model(
            src, dst, timestamp,
            src_neighbor_seq, src_time_seq,
            dst_neighbor_seq, dst_time_seq,
            neg_dst=neg_dst
        )

        pos_score = output['pos_score'].item()
        neg_scores = output['neg_score'].squeeze(0).cpu().numpy()
        rank = (neg_scores > pos_score).sum() + 1
        all_ranks.append(rank)

    ranks = np.array(all_ranks)
    return compute_metrics(ranks, ks=[1, 3, 10, 50])


def run_multiple_evaluations(
    model_name: str,
    checkpoint_path: str,
    dataset_name: str,
    num_runs: int = 5,
    device: torch.device = None
) -> List[Dict[str, float]]:
    """Run multiple evaluations with different seeds."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    data_loader = RealDataLoader(dataset_name=dataset_name)
    data = data_loader.get_data()
    _, _, test_dataset = data_loader.get_temporal_split()

    # Load model
    model, config = load_model(model_name, checkpoint_path, data.num_nodes, device)

    results = []
    for run in range(num_runs):
        print(f"  Run {run + 1}/{num_runs}")

        # Create negative sampler with different seed
        neg_sampler = create_negative_sampler(
            src_nodes=data.src,
            dst_nodes=data.dst,
            timestamps=data.timestamps,
            num_nodes=data.num_nodes,
            strategy='temporal',
            seed=42 + run
        )

        metrics = evaluate_model(model, test_dataset, neg_sampler, device)
        results.append(metrics)

    return results


def compare_models(
    model_results: Dict[str, List[Dict[str, float]]],
    baseline_name: str = 'baseline'
) -> Dict:
    """Compare multiple models and compute statistical analysis."""
    analyzer = StatisticalAnalysis(num_runs=5)

    # Add results to analyzer
    for model_name, runs in model_results.items():
        for run_id, metrics in enumerate(runs):
            analyzer.add_result(model_name, metrics, run_id)

    # Compute comparisons
    comparisons = {}
    for model_name in model_results:
        if model_name == baseline_name:
            continue

        # Get MRR scores
        baseline_mrr = [r['mrr'] for r in model_results[baseline_name]]
        model_mrr = [r['mrr'] for r in model_results[model_name]]

        # Cohen's d
        cohen_d, d_interp = compute_cohen_d(baseline_mrr, model_mrr)

        # T-test
        t_result = independent_t_test(baseline_mrr, model_mrr)

        # Improvement
        baseline_mean = np.mean(baseline_mrr)
        model_mean = np.mean(model_mrr)
        improvement = ((model_mean - baseline_mean) / baseline_mean) * 100

        comparisons[model_name] = {
            'baseline_mrr_mean': float(baseline_mean),
            'baseline_mrr_std': float(np.std(baseline_mrr, ddof=1)),
            'model_mrr_mean': float(model_mean),
            'model_mrr_std': float(np.std(model_mrr, ddof=1)),
            'improvement_pct': float(improvement),
            'cohen_d': float(cohen_d),
            'cohen_d_interpretation': d_interp,
            't_statistic': float(t_result.t_statistic),
            'p_value': float(t_result.p_value),
            'significant': t_result.significant
        }

    return {
        'model_results': {
            name: {
                'mrr_mean': float(np.mean([r['mrr'] for r in runs])),
                'mrr_std': float(np.std([r['mrr'] for r in runs], ddof=1)),
                'hits@10_mean': float(np.mean([r['hits@10'] for r in runs])),
                'hits@10_std': float(np.std([r['hits@10'] for r in runs], ddof=1)),
            }
            for name, runs in model_results.items()
        },
        'comparisons': comparisons
    }


def generate_comparison_report(
    comparison_results: Dict,
    output_path: str
):
    """Generate markdown comparison report."""
    lines = []
    lines.append("# RW2 Temporal Network Embedding - Evaluation Report\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Summary table
    lines.append("## Model Performance Summary\n")
    lines.append("| Model | MRR | Hits@10 |")
    lines.append("|-------|-----|---------|")

    for model, stats in comparison_results['model_results'].items():
        lines.append(
            f"| {model} | {stats['mrr_mean']:.4f} +/- {stats['mrr_std']:.4f} | "
            f"{stats['hits@10_mean']:.4f} +/- {stats['hits@10_std']:.4f} |"
        )

    # Comparisons
    lines.append("\n## Statistical Comparisons vs Baseline\n")
    lines.append("| Model | Improvement | Cohen's d | p-value | Significant |")
    lines.append("|-------|-------------|-----------|---------|-------------|")

    for model, comp in comparison_results['comparisons'].items():
        sig = "Yes" if comp['significant'] else "No"
        lines.append(
            f"| {model} | {comp['improvement_pct']:+.2f}% | "
            f"{comp['cohen_d']:.3f} ({comp['cohen_d_interpretation']}) | "
            f"{comp['p_value']:.4f} | {sig} |"
        )

    # Save report
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate temporal network models")

    parser.add_argument('--model', '-m', type=str, default='ssm_memory_llm')
    parser.add_argument('--checkpoint', '-c', type=str, default=None)
    parser.add_argument('--dataset', '-d', type=str, default='tgbl-wiki')
    parser.add_argument('--num_runs', '-n', type=int, default=5)
    parser.add_argument('--compare', nargs='+', type=str, default=None)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--output', '-o', type=str, default='./reports')
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    device = torch.device(
        f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    )

    os.makedirs(args.output, exist_ok=True)

    if args.compare:
        # Compare multiple models
        print(f"Comparing models: {args.compare}")
        model_results = {}

        for model_name in args.compare:
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f"{model_name}_{args.dataset}_best.pth"
            )

            if not os.path.exists(checkpoint_path):
                print(f"Warning: Checkpoint not found for {model_name}, skipping...")
                continue

            print(f"\nEvaluating {model_name}...")
            results = run_multiple_evaluations(
                model_name, checkpoint_path, args.dataset,
                num_runs=args.num_runs, device=device
            )
            model_results[model_name] = results

        # Compare and generate report
        comparison = compare_models(model_results, baseline_name=args.compare[0])

        # Save results
        results_path = os.path.join(args.output, f"comparison_{args.dataset}.json")
        with open(results_path, 'w') as f:
            json.dump(comparison, f, indent=2)

        # Generate report
        report_path = os.path.join(args.output, f"comparison_{args.dataset}.md")
        generate_comparison_report(comparison, report_path)

    else:
        # Single model evaluation
        if args.checkpoint is None:
            args.checkpoint = os.path.join(
                args.checkpoint_dir,
                f"{args.model}_{args.dataset}_best.pth"
            )

        print(f"Evaluating {args.model} on {args.dataset}")
        results = run_multiple_evaluations(
            args.model, args.checkpoint, args.dataset,
            num_runs=args.num_runs, device=device
        )

        # Print results
        mrr_values = [r['mrr'] for r in results]
        print(f"\nMRR: {np.mean(mrr_values):.4f} +/- {np.std(mrr_values, ddof=1):.4f}")

        hits10_values = [r['hits@10'] for r in results]
        print(f"Hits@10: {np.mean(hits10_values):.4f} +/- {np.std(hits10_values, ddof=1):.4f}")


if __name__ == '__main__':
    main()
