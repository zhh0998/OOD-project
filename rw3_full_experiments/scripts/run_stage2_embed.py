#!/usr/bin/env python3
"""
Stage 2: Embedding Extraction and Anisotropy Diagnosis (v4.1)

Features:
- Checkpoint/resume support: skips already cached embeddings
- Full dataset coverage including all Banking77 seeds
- Audit logging for leakage prevention
- I-STAR compliant anisotropy metrics
- Manifest update for pipeline tracking

IRON RULE: All PCA/whitening fit on ID-train ONLY (logged in audit)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

from src.embeddings.embedding_extractor import EmbeddingExtractor
from src.embeddings.anisotropy_metrics import compute_anisotropy_metrics, diagnose_anisotropy


# =============================================================================
# Configuration
# =============================================================================

# Full dataset list (v4.1)
DATASETS = [
    'clinc150',
    'banking77_alpha',
    'banking77_seed42',
    'banking77_seed123',
    'banking77_seed456',
    'banking77_seed789',
    'banking77_seed2024',
    'hwu64',
    'massive',
    'newsgroups',
    'nq_open'
]

# All embedding models
MODELS = ['minilm', 'bge', 'e5', 'mpnet']

# Required splits per dataset
REQUIRED_SPLITS = ['train_id', 'cal_id', 'test_id', 'test_ood']


# =============================================================================
# Audit Logger
# =============================================================================

class AuditLogger:
    """Leakage prevention audit logger."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.entries = []
        self.start_time = datetime.now()

    def log(self, operation: str, fit_split: str, dataset: str, model: str,
            details: str = ""):
        """Log an operation with fit split information."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'fit_split': fit_split,
            'dataset': dataset,
            'model': model,
            'details': details,
            'compliant': fit_split in ['id_train_only', 'id_cal_only', 'N/A']
        }
        self.entries.append(entry)

        # Print audit line
        status = "✓" if entry['compliant'] else "✗ LEAKAGE"
        print(f"  [AUDIT] {operation}: fit={fit_split} {status}")

        if not entry['compliant']:
            print(f"  ⚠️  WARNING: Potential data leakage detected!")

    def save(self):
        """Save audit log to file."""
        log_data = {
            'version': '4.1',
            'stage': 'stage2_embedding',
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_operations': len(self.entries),
            'compliant_operations': sum(1 for e in self.entries if e['compliant']),
            'entries': self.entries
        }

        with open(self.log_path, 'w') as f:
            json.dump(log_data, f, indent=2)

        print(f"\n[AUDIT] Log saved to {self.log_path}")
        return log_data


# =============================================================================
# Checkpoint Manager
# =============================================================================

class CheckpointManager:
    """Manages Stage 2 checkpointing and resume."""

    def __init__(self, cache_dir: Path, results_dir: Path):
        self.cache_dir = cache_dir
        self.results_dir = results_dir
        self.checkpoint_path = results_dir / "stage2_checkpoint.json"

    def check_cached(self, dataset: str, model: str) -> Tuple[bool, List[str]]:
        """
        Check if embeddings are cached for a dataset-model pair.

        Returns:
            (is_complete, missing_splits)
        """
        model_dir = self.cache_dir / dataset / model

        if not model_dir.exists():
            return False, REQUIRED_SPLITS.copy()

        cached_splits = []
        missing_splits = []

        for split in REQUIRED_SPLITS:
            cache_file = model_dir / f"{split}.npy"
            if cache_file.exists():
                cached_splits.append(split)
            else:
                missing_splits.append(split)

        is_complete = len(missing_splits) == 0
        return is_complete, missing_splits

    def get_progress(self) -> Dict:
        """Get overall Stage 2 progress."""
        total = len(DATASETS) * len(MODELS)
        completed = 0
        partial = 0
        pending = 0
        details = {}

        for dataset in DATASETS:
            details[dataset] = {}
            for model in MODELS:
                is_complete, missing = self.check_cached(dataset, model)
                if is_complete:
                    completed += 1
                    details[dataset][model] = 'complete'
                elif len(missing) < len(REQUIRED_SPLITS):
                    partial += 1
                    details[dataset][model] = f'partial ({len(REQUIRED_SPLITS) - len(missing)}/{len(REQUIRED_SPLITS)})'
                else:
                    pending += 1
                    details[dataset][model] = 'pending'

        return {
            'total': total,
            'completed': completed,
            'partial': partial,
            'pending': pending,
            'completion_pct': completed / total * 100,
            'details': details
        }

    def save_checkpoint(self, progress: Dict, anisotropy_results: Dict):
        """Save checkpoint data."""
        checkpoint = {
            'version': '4.1',
            'timestamp': datetime.now().isoformat(),
            'stage': 'stage2_embedding',
            'progress': progress,
            'anisotropy_computed': list(anisotropy_results.keys())
        }

        with open(self.checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        print(f"\n[CHECKPOINT] Saved to {self.checkpoint_path}")


# =============================================================================
# Main Execution
# =============================================================================

def update_manifest(results_dir: Path, stage2_complete: bool, stats: Dict):
    """Update the pipeline manifest."""
    manifest_path = results_dir / "manifest.json"

    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {
            'version': '4.1',
            'created': datetime.now().strftime('%Y-%m-%d'),
            'completed': [],
            'failed': [],
            'remaining': []
        }

    # Update Stage 2 status
    if stage2_complete:
        if 'stage2_embedding' not in manifest['completed']:
            manifest['completed'].append('stage2_embedding')
        if 'stage2_embedding' in manifest.get('remaining', []):
            manifest['remaining'].remove('stage2_embedding')

    # Add stats
    manifest['stage2_stats'] = stats
    manifest['last_updated'] = datetime.now().isoformat()

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"[MANIFEST] Updated {manifest_path}")


def main():
    print("=" * 70)
    print("STAGE 2: EMBEDDING EXTRACTION & ANISOTROPY DIAGNOSIS (v4.1)")
    print("=" * 70)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Datasets: {len(DATASETS)} | Models: {len(MODELS)}")
    print(f"Total combinations: {len(DATASETS) * len(MODELS)}")
    print()

    # Setup paths
    base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(base_dir)

    data_dir = Path("data/processed")
    cache_dir = Path("cache/embeddings")
    results_dir = Path("results")
    tables_dir = results_dir / "tables"
    logs_dir = results_dir / "logs"

    # Create directories
    for d in [cache_dir, results_dir, tables_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Initialize managers
    checkpoint_mgr = CheckpointManager(cache_dir, results_dir)
    audit_logger = AuditLogger(logs_dir / "stage2_audit.json")

    # Check initial progress
    initial_progress = checkpoint_mgr.get_progress()
    print(f"[RESUME] Initial state: {initial_progress['completed']}/{initial_progress['total']} complete "
          f"({initial_progress['completion_pct']:.1f}%)")
    print(f"         Partial: {initial_progress['partial']} | Pending: {initial_progress['pending']}")
    print()

    # Initialize extractor
    extractor = EmbeddingExtractor(cache_dir=str(cache_dir))

    # Results storage
    anisotropy_results = {}
    embedding_stats = []
    processed_count = 0
    skipped_count = 0
    error_count = 0

    # Process each dataset-model combination
    for dataset_name in DATASETS:
        dataset_path = data_dir / dataset_name / "data.json"

        if not dataset_path.exists():
            print(f"\n[SKIP] {dataset_name}: data.json not found")
            continue

        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 60}")

        # Load dataset
        with open(dataset_path) as f:
            data = json.load(f)

        anisotropy_results[dataset_name] = {}

        for model_key in MODELS:
            # Check if already cached
            is_complete, missing_splits = checkpoint_mgr.check_cached(dataset_name, model_key)

            if is_complete:
                print(f"\n[CACHED] {model_key}: All splits cached, loading for anisotropy...")
                skipped_count += 1

                # Load cached embeddings for anisotropy computation
                train_path = cache_dir / dataset_name / model_key / "train_id.npy"
                if train_path.exists():
                    train_emb = np.load(train_path)

                    # Audit: anisotropy computed on train_id only
                    audit_logger.log(
                        operation="anisotropy_metrics",
                        fit_split="id_train_only",
                        dataset=dataset_name,
                        model=model_key,
                        details=f"Loaded from cache, shape={train_emb.shape}"
                    )

                    metrics = compute_anisotropy_metrics(train_emb)
                    anisotropy_results[dataset_name][model_key] = metrics

                    embedding_stats.append({
                        'dataset': dataset_name,
                        'model': model_key,
                        'n_samples': len(train_emb),
                        'dim': train_emb.shape[1],
                        'effective_dim': metrics['effective_dim'],
                        'effective_dim_ratio': metrics['effective_dim'] / train_emb.shape[1],
                        'top1_variance_ratio': metrics['top1_variance_ratio'],
                        'topk_cumulative_5': metrics['topk_cumulative_5'],
                        'avg_cosine': metrics['avg_cosine'],
                        'source': 'cached'
                    })

                    print(f"         Effective dim: {metrics['effective_dim']:.2f}/{train_emb.shape[1]} "
                          f"({metrics['effective_dim']/train_emb.shape[1]:.1%})")
            else:
                print(f"\n[EXTRACT] {model_key}: Missing {missing_splits}")

                try:
                    # Extract embeddings (extractor handles partial caching)
                    embeddings = extractor.extract_dataset(
                        data, model_key, dataset_name,
                        use_cache=True, batch_size=32
                    )
                    processed_count += 1

                    # Compute anisotropy on train_id ONLY
                    if 'train_id' in embeddings:
                        train_emb = embeddings['train_id']

                        # Audit: anisotropy computed on train_id only
                        audit_logger.log(
                            operation="anisotropy_metrics",
                            fit_split="id_train_only",
                            dataset=dataset_name,
                            model=model_key,
                            details=f"Fresh extraction, shape={train_emb.shape}"
                        )

                        metrics = compute_anisotropy_metrics(train_emb)
                        anisotropy_results[dataset_name][model_key] = metrics

                        embedding_stats.append({
                            'dataset': dataset_name,
                            'model': model_key,
                            'n_samples': len(train_emb),
                            'dim': train_emb.shape[1],
                            'effective_dim': metrics['effective_dim'],
                            'effective_dim_ratio': metrics['effective_dim'] / train_emb.shape[1],
                            'top1_variance_ratio': metrics['top1_variance_ratio'],
                            'topk_cumulative_5': metrics['topk_cumulative_5'],
                            'avg_cosine': metrics['avg_cosine'],
                            'source': 'extracted'
                        })

                        print(f"  Effective dim: {metrics['effective_dim']:.2f}/{train_emb.shape[1]} "
                              f"({metrics['effective_dim']/train_emb.shape[1]:.1%})")
                        print(f"  Top-1 variance: {metrics['top1_variance_ratio']:.3f}")

                except Exception as e:
                    print(f"  [ERROR] {model_key}: {str(e)}")
                    error_count += 1
                    audit_logger.log(
                        operation="embedding_extraction",
                        fit_split="N/A",
                        dataset=dataset_name,
                        model=model_key,
                        details=f"ERROR: {str(e)}"
                    )

    # ==========================================================================
    # Generate Anisotropy Table
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ANISOTROPY DIAGNOSTIC TABLE (I-STAR Compliant)")
    print("=" * 70)

    df = pd.DataFrame(embedding_stats)

    if len(df) > 0:
        # Pivot table: datasets x models for effective_dim
        pivot_eff = df.pivot(index='dataset', columns='model', values='effective_dim')
        pivot_ratio = df.pivot(index='dataset', columns='model', values='effective_dim_ratio')
        pivot_top1 = df.pivot(index='dataset', columns='model', values='top1_variance_ratio')
        pivot_cos = df.pivot(index='dataset', columns='model', values='avg_cosine')

        print("\nEffective Dimension (I-STAR primary metric):")
        print(pivot_eff.round(2).to_string())

        print("\nTop-1 Variance Ratio:")
        print(pivot_top1.round(3).to_string())

        # Save tables
        df.to_csv(tables_dir / "anisotropy_table.csv", index=False)
        pivot_eff.to_csv(tables_dir / "anisotropy_pivot_effdim.csv")
        pivot_ratio.to_csv(tables_dir / "anisotropy_pivot_ratio.csv")
        pivot_top1.to_csv(tables_dir / "anisotropy_pivot_top1var.csv")
        pivot_cos.to_csv(tables_dir / "anisotropy_pivot_avgcos.csv")

        # Save full results as JSON
        def convert_numpy(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj

        with open(tables_dir / "anisotropy_results.json", 'w') as f:
            json.dump(convert_numpy(anisotropy_results), f, indent=2)

        print(f"\nSaved: {tables_dir}/anisotropy_table.csv")
        print(f"Saved: {tables_dir}/anisotropy_pivot_*.csv (4 files)")
        print(f"Saved: {tables_dir}/anisotropy_results.json")

    # ==========================================================================
    # Final Progress & Checkpoint
    # ==========================================================================
    final_progress = checkpoint_mgr.get_progress()
    stage2_complete = final_progress['completed'] == final_progress['total']

    # Save checkpoint
    checkpoint_mgr.save_checkpoint(final_progress, anisotropy_results)

    # Save audit log
    audit_log = audit_logger.save()

    # Update manifest
    stats = {
        'datasets': len(DATASETS),
        'models': len(MODELS),
        'total_combinations': final_progress['total'],
        'completed': final_progress['completed'],
        'processed_this_run': processed_count,
        'skipped_cached': skipped_count,
        'errors': error_count,
        'audit_compliant': audit_log['compliant_operations'] == audit_log['total_operations']
    }
    update_manifest(results_dir, stage2_complete, stats)

    # ==========================================================================
    # Stage 2 Checkpoint Report
    # ==========================================================================
    print("\n" + "=" * 70)
    print("STAGE 2 CHECKPOINT REPORT")
    print("=" * 70)

    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 2: EMBEDDING EXTRACTION & ANISOTROPY                         │
├─────────────────────────────────────────────────────────────────────┤
│ Status: {'✓ COMPLETE' if stage2_complete else '○ IN PROGRESS'}                                                   │
├─────────────────────────────────────────────────────────────────────┤
│ Progress:                                                           │
│   • Total combinations: {final_progress['total']:3d} (datasets × models)                    │
│   • Completed:          {final_progress['completed']:3d} ({final_progress['completion_pct']:.1f}%)                                    │
│   • Partial:            {final_progress['partial']:3d}                                           │
│   • Pending:            {final_progress['pending']:3d}                                           │
├─────────────────────────────────────────────────────────────────────┤
│ This Run:                                                           │
│   • Newly processed:    {processed_count:3d}                                           │
│   • Skipped (cached):   {skipped_count:3d}                                           │
│   • Errors:             {error_count:3d}                                           │
├─────────────────────────────────────────────────────────────────────┤
│ Audit Compliance:                                                   │
│   • Operations logged:  {audit_log['total_operations']:3d}                                           │
│   • Compliant:          {audit_log['compliant_operations']:3d}                                           │
│   • LEAKAGE DETECTED:   {'NO ✓' if audit_log['compliant_operations'] == audit_log['total_operations'] else 'YES ⚠️'}                                         │
└─────────────────────────────────────────────────────────────────────┘
""")

    # Print coverage matrix
    print("\nCoverage Matrix:")
    print("─" * 70)
    header = "Dataset".ljust(25) + " | " + " | ".join(m.ljust(8) for m in MODELS)
    print(header)
    print("─" * 70)

    for dataset in DATASETS:
        row = dataset.ljust(25) + " | "
        for model in MODELS:
            status = final_progress['details'].get(dataset, {}).get(model, 'N/A')
            if status == 'complete':
                row += "✓".ljust(8) + " | "
            elif 'partial' in status:
                row += "◐".ljust(8) + " | "
            else:
                row += "○".ljust(8) + " | "
        print(row)

    print("─" * 70)
    print("Legend: ✓ = complete | ◐ = partial | ○ = pending")

    if stage2_complete:
        print("\n✓ Stage 2 COMPLETE. Ready for Stage 3 (Experiment Matrix).")
    else:
        print(f"\n○ Stage 2 IN PROGRESS. Run again to continue ({final_progress['pending']} remaining).")

    print(f"\nEnd time: {datetime.now().isoformat()}")

    return stage2_complete


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
