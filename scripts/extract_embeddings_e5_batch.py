#!/usr/bin/env python3
"""
RW3 v4.1 Stage 2 - E5 Embedding Extraction Script
Extracts embeddings for e5 model (intfloat/e5-large-v2) across 4 datasets.
Key: e5 requires "query: " prefix for all texts.
Loads model once and processes all datasets sequentially.
"""

import json
import os
import time
import numpy as np
from datetime import datetime
from pathlib import Path

# E5 model configuration
MODEL_KEY = "e5"
MODEL_NAME = "intfloat/e5-large-v2"
E5_PREFIX = "query: "

DATASETS = ["clinc150", "hwu64", "massive", "newsgroups"]
SPLITS = ["train_id", "cal_id", "test_id", "test_ood"]

def load_data(dataset_name):
    """Load dataset from processed data directory."""
    data_path = Path(f"rw3_full_experiments/data/processed/{dataset_name}/data.json")
    with open(data_path) as f:
        return json.load(f)

def get_texts(data, split):
    """Extract texts from a split."""
    return data[split]["texts"]

def save_embeddings(embeddings, dataset, cache_dir):
    """Save embeddings to cache directory."""
    output_dir = cache_dir / dataset / MODEL_KEY
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def check_complete(cache_dir, dataset):
    """Check if all 4 splits already exist for a dataset."""
    for split in SPLITS:
        path = cache_dir / dataset / MODEL_KEY / f"{split}.npy"
        if not path.exists():
            return False
    return True

def write_audit_log(audit_dir, dataset, results):
    """Write audit log for the extraction."""
    audit_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = audit_dir / f"embedding_{dataset}_{MODEL_KEY}_{timestamp}.json"

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset,
        "model_key": MODEL_KEY,
        "model_name": MODEL_NAME,
        "prefix_used": E5_PREFIX,
        "results": results
    }

    with open(log_path, "w") as f:
        json.dump(log_entry, f, indent=2)

    return log_path

def main():
    """Main entry point."""
    from sentence_transformers import SentenceTransformer

    cache_dir = Path("cache/embeddings")
    audit_dir = Path("results/audit")

    print("=" * 70)
    print("RW3 v4.1 Stage 2 - E5 Embedding Extraction")
    print("=" * 70)
    print(f"Model: {MODEL_KEY} ({MODEL_NAME})")
    print(f"Prefix: '{E5_PREFIX}'")
    print(f"Datasets: {DATASETS}")
    print("=" * 70)

    # Check which datasets need processing
    to_process = []
    for dataset in DATASETS:
        if check_complete(cache_dir, dataset):
            print(f"  ✓ {dataset}: already complete (4/4 splits)")
        else:
            print(f"  → {dataset}: needs processing")
            to_process.append(dataset)

    if not to_process:
        print("\nAll datasets already complete. Nothing to do.")
        return {}

    print(f"\n{'='*70}")
    print(f"Loading model: {MODEL_NAME}")
    print(f"{'='*70}")

    model_load_start = time.time()
    model = SentenceTransformer(MODEL_NAME)
    model_load_time = time.time() - model_load_start
    print(f"Model loaded in {model_load_time:.2f}s")

    all_results = {}
    overall_start = time.time()

    for dataset in to_process:
        print(f"\n{'─'*60}")
        print(f"Processing: {dataset} × {MODEL_KEY}")
        print(f"{'─'*60}")

        # Load data
        print(f"  Loading data from rw3_full_experiments/data/processed/{dataset}/data.json")
        data = load_data(dataset)

        results = {
            "splits": {},
            "total_time": 0,
            "model_load_time": model_load_time
        }

        dataset_start = time.time()
        output_dir = cache_dir / dataset / MODEL_KEY
        output_dir.mkdir(parents=True, exist_ok=True)

        for split in SPLITS:
            split_path = output_dir / f"{split}.npy"
            if split_path.exists():
                print(f"  {split}: already exists, skipping")
                continue

            print(f"  {split}:", end=" ", flush=True)
            texts = get_texts(data, split)

            # Apply e5 prefix
            encode_texts = [E5_PREFIX + t for t in texts]

            start_time = time.time()
            embeddings = model.encode(
                encode_texts,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=32
            )
            elapsed = time.time() - start_time

            # Save embeddings
            np.save(split_path, embeddings)

            results["splits"][split] = {
                "samples": len(texts),
                "dimensions": embeddings.shape[1],
                "time_seconds": round(elapsed, 2),
                "output_path": str(split_path)
            }

            print(f"{len(texts)} samples, {embeddings.shape[1]}D, {elapsed:.2f}s")

        results["total_time"] = round(time.time() - dataset_start, 2)
        if results["splits"]:
            results["embedding_dim"] = list(results["splits"].values())[0]["dimensions"]

            # Write audit log
            log_path = write_audit_log(audit_dir, dataset, results)
            print(f"  Audit log: {log_path}")

            all_results[f"{dataset}_{MODEL_KEY}"] = results

    overall_time = time.time() - overall_start

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY - RW3 v4.1 Stage 2 E5 Batch")
    print("=" * 70)

    total_samples_all = 0
    for combo, result in all_results.items():
        print(f"\n{combo}:")
        print(f"  Embedding dimension: {result.get('embedding_dim', 'N/A')}")
        print(f"  Total time: {result['total_time']}s")
        total_samples = 0
        for split, info in result["splits"].items():
            print(f"    {split}: {info['samples']} samples, {info['dimensions']}D, {info['time_seconds']}s")
            total_samples += info['samples']
        print(f"  Total samples: {total_samples}")
        total_samples_all += total_samples

    print(f"\n{'='*70}")
    print(f"Model load time: {model_load_time:.2f}s")
    print(f"Overall extraction time: {overall_time:.2f}s ({overall_time/60:.1f} min)")
    print(f"Total samples processed: {total_samples_all}")
    print(f"Combinations processed: {len(all_results)}")
    print(f"{'='*70}")

    return all_results

if __name__ == "__main__":
    main()
