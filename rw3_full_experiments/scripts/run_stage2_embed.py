#!/usr/bin/env python3
"""
Stage 2: Embedding Extraction and Anisotropy Diagnosis
Extracts embeddings for all datasets with all models.
Computes I-STAR compliant anisotropy metrics.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
from pathlib import Path
from src.data.data_loader import DatasetLoader
from src.embeddings.embedding_extractor import EmbeddingExtractor
from src.embeddings.anisotropy_metrics import compute_anisotropy_metrics, diagnose_anisotropy


def main():
    print("=" * 60)
    print("STAGE 2: EMBEDDING EXTRACTION & ANISOTROPY DIAGNOSIS")
    print("=" * 60)

    # Load processed datasets
    data_dir = Path("data/processed")
    datasets_to_process = [
        'clinc150',
        'banking77_alpha',
        'banking77_seed42',
        'hwu64',
        'massive',
        'newsgroups',
        'nq_open'
    ]

    # Models to use
    models = ['minilm', 'bge', 'e5', 'mpnet']

    # Initialize extractor
    extractor = EmbeddingExtractor(cache_dir="cache/embeddings")

    # Results storage
    anisotropy_results = {}
    embedding_stats = []

    for dataset_name in datasets_to_process:
        dataset_path = data_dir / dataset_name / "data.json"

        if not dataset_path.exists():
            print(f"\nSkipping {dataset_name} (not found)")
            continue

        print(f"\n{'=' * 50}")
        print(f"Processing: {dataset_name}")
        print(f"{'=' * 50}")

        # Load dataset
        with open(dataset_path) as f:
            data = json.load(f)

        anisotropy_results[dataset_name] = {}

        for model_key in models:
            print(f"\n--- Model: {model_key} ---")

            # Extract embeddings
            embeddings = extractor.extract_dataset(
                data, model_key, dataset_name,
                use_cache=True, batch_size=32
            )

            # Compute anisotropy metrics on train_id
            if 'train_id' in embeddings:
                train_emb = embeddings['train_id']
                metrics = compute_anisotropy_metrics(train_emb)
                anisotropy_results[dataset_name][model_key] = metrics

                # Store for table
                embedding_stats.append({
                    'dataset': dataset_name,
                    'model': model_key,
                    'n_samples': len(train_emb),
                    'dim': train_emb.shape[1],
                    'effective_dim': metrics['effective_dim'],
                    'effective_dim_ratio': metrics['effective_dim'] / train_emb.shape[1],
                    'top1_variance_ratio': metrics['top1_variance_ratio'],
                    'topk_cumulative_5': metrics['topk_cumulative_5'],
                    'avg_cosine': metrics['avg_cosine']
                })

                print(f"  Effective dim: {metrics['effective_dim']:.2f} / {train_emb.shape[1]} "
                      f"({metrics['effective_dim'] / train_emb.shape[1]:.1%})")
                print(f"  Top-1 variance: {metrics['top1_variance_ratio']:.3f}")
                print(f"  Avg cosine: {metrics['avg_cosine']:.3f} (I-STAR cautions)")

    # Create anisotropy table
    print("\n" + "=" * 60)
    print("ANISOTROPY DIAGNOSTIC TABLE")
    print("=" * 60)

    df = pd.DataFrame(embedding_stats)

    # Pivot table: datasets x models for effective_dim
    if len(df) > 0:
        pivot = df.pivot(index='dataset', columns='model', values='effective_dim')
        print("\nEffective Dimension (I-STAR primary metric):")
        print(pivot.to_string())

        # Save tables
        results_dir = Path("results/tables")
        results_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(results_dir / "anisotropy_table.csv", index=False)
        pivot.to_csv(results_dir / "anisotropy_pivot.csv")

        # Save full results as JSON
        with open(results_dir / "anisotropy_results.json", 'w') as f:
            # Convert numpy types for JSON serialization
            def convert(obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                return obj

            json.dump(convert(anisotropy_results), f, indent=2)

        print(f"\nSaved: {results_dir}/anisotropy_table.csv")
        print(f"Saved: {results_dir}/anisotropy_pivot.csv")
        print(f"Saved: {results_dir}/anisotropy_results.json")

    # Generate diagnostic reports
    print("\n" + "=" * 60)
    print("DIAGNOSTIC REPORTS")
    print("=" * 60)

    report_text = "# Anisotropy Diagnostic Reports\n\n"

    for dataset_name, models_data in anisotropy_results.items():
        report_text += f"## {dataset_name}\n\n"
        for model_key, metrics in models_data.items():
            report_text += f"### {model_key}\n"
            report_text += f"- Effective dimension: {metrics['effective_dim']:.2f}\n"
            report_text += f"- Top-1 variance ratio: {metrics['top1_variance_ratio']:.3f}\n"
            report_text += f"- Top-5 cumulative: {metrics['topk_cumulative_5']:.3f}\n"
            report_text += f"- Avg cosine (caution): {metrics['avg_cosine']:.3f}\n\n"

    with open(results_dir / "anisotropy_report.md", 'w') as f:
        f.write(report_text)

    print(f"Saved: {results_dir}/anisotropy_report.md")

    # Checkpoint
    print("\n" + "=" * 60)
    print("STAGE 2 CHECKPOINT")
    print("=" * 60)
    print(f"Datasets processed: {len(anisotropy_results)}")
    print(f"Models processed: {len(models)}")
    print(f"Total embeddings cached: {len(embedding_stats)}")
    print("\n✓ Stage 2 complete. Ready for Stage 3.")


if __name__ == "__main__":
    main()
