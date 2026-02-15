#!/usr/bin/env python3
"""
Stage 2: Targeted Embedding Extraction (v4.1)

Purpose: Extract embeddings ONLY for specified dataset-model combinations.
This script is designed for incremental progress on Stage 2.

Usage:
    python run_stage2_targeted.py --dataset clinc150 --models bge e5
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import time

from src.embeddings.embedding_extractor import EmbeddingExtractor


def main():
    # Configuration - ONLY these 2 combinations
    DATASET = 'clinc150'
    MODELS = ['bge', 'e5']
    SPLITS = ['train_id', 'cal_id', 'test_id', 'test_ood']

    print("=" * 70)
    print("STAGE 2: TARGETED EMBEDDING EXTRACTION (v4.1)")
    print("=" * 70)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Dataset: {DATASET}")
    print(f"Models: {MODELS}")
    print()

    # Setup paths
    base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(base_dir)

    data_dir = Path("data/processed")
    cache_dir = Path("cache/embeddings")
    audit_dir = Path("results/audit")

    # Create directories
    cache_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    data_path = data_dir / DATASET / "data.json"
    print(f"Loading data from {data_path}...")
    with open(data_path) as f:
        data = json.load(f)

    # Print dataset stats
    print(f"\nDataset splits:")
    for split in SPLITS:
        if split in data:
            print(f"  {split}: {len(data[split]['texts'])} samples")
    print()

    # Initialize extractor
    extractor = EmbeddingExtractor(cache_dir=str(cache_dir))

    # Results storage
    results = []
    audit_entries = []

    for model_key in MODELS:
        print("=" * 60)
        print(f"Model: {model_key}")
        print("=" * 60)

        model_cache_dir = cache_dir / DATASET / model_key

        # Check if already exists
        existing = [s for s in SPLITS if (model_cache_dir / f"{s}.npy").exists()]
        if len(existing) == len(SPLITS):
            print(f"[SKIP] All splits already cached for {model_key}")
            continue

        # Start timing
        start_time = time.time()

        # Extract embeddings
        try:
            embeddings = extractor.extract_dataset(
                data, model_key, DATASET,
                use_cache=True, batch_size=32
            )

            elapsed = time.time() - start_time

            # Get stats
            train_emb = embeddings.get('train_id')
            if train_emb is not None:
                n_samples = sum(len(embeddings[s]) for s in embeddings)
                dim = train_emb.shape[1]

                result = {
                    'dataset': DATASET,
                    'model': model_key,
                    'dim': dim,
                    'splits': {s: len(embeddings[s]) for s in embeddings},
                    'total_samples': n_samples,
                    'elapsed_sec': round(elapsed, 2)
                }
                results.append(result)

                print(f"\n[SUCCESS] {model_key}:")
                print(f"  Dimension: {dim}")
                print(f"  Total samples: {n_samples}")
                print(f"  Elapsed: {elapsed:.2f}s")

                # Audit entry
                audit_entries.append({
                    'timestamp': datetime.now().isoformat(),
                    'operation': 'embedding_extraction',
                    'dataset': DATASET,
                    'model': model_key,
                    'fit_split': 'N/A',
                    'details': f"dim={dim}, samples={n_samples}, time={elapsed:.2f}s",
                    'compliant': True
                })

        except Exception as e:
            print(f"[ERROR] {model_key}: {e}")
            audit_entries.append({
                'timestamp': datetime.now().isoformat(),
                'operation': 'embedding_extraction',
                'dataset': DATASET,
                'model': model_key,
                'fit_split': 'N/A',
                'details': f"ERROR: {str(e)}",
                'compliant': False
            })

    # Save audit log
    audit_log = {
        'version': '4.1',
        'stage': 'stage2_targeted',
        'timestamp': datetime.now().isoformat(),
        'target_dataset': DATASET,
        'target_models': MODELS,
        'entries': audit_entries,
        'total_operations': len(audit_entries),
        'compliant_operations': sum(1 for e in audit_entries if e['compliant'])
    }

    audit_path = audit_dir / f"stage2_{DATASET}_{'_'.join(MODELS)}.json"
    with open(audit_path, 'w') as f:
        json.dump(audit_log, f, indent=2)
    print(f"\n[AUDIT] Saved to {audit_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for r in results:
        print(f"\n{r['dataset']} × {r['model']}:")
        print(f"  Embedding dimension: {r['dim']}")
        print(f"  Total samples: {r['total_samples']}")
        for split, count in r['splits'].items():
            print(f"    {split}: {count}")
        print(f"  Time: {r['elapsed_sec']}s")

    # Verify cached files
    print("\n" + "=" * 70)
    print("CACHED FILES")
    print("=" * 70)

    for model_key in MODELS:
        model_dir = cache_dir / DATASET / model_key
        if model_dir.exists():
            print(f"\n{model_dir}/")
            for f in sorted(model_dir.iterdir()):
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  {f.name}: {size_mb:.2f} MB")

    print(f"\nEnd time: {datetime.now().isoformat()}")

    return len(results) == len(MODELS)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
