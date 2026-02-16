#!/usr/bin/env python3
"""
RW3 v4.1 Stage 2 - Embedding Extraction Script (Batch 4)
Extracts embeddings for 8 combinations (clinc150×3 + hwu64×2 + massive×2 + newsgroups×1).
Groups by model to avoid repeated model loading.
"""

import json
import os
import time
import numpy as np
from datetime import datetime
from pathlib import Path

# Model configurations
MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge": "BAAI/bge-base-en-v1.5",
    "e5": "intfloat/e5-large-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2"
}

# Prefix required by certain models (e5 requires "query: " prefix)
MODEL_PREFIX = {
    "e5": "query: "
}

SPLITS = ["train_id", "cal_id", "test_id", "test_ood"]

def load_data(dataset_name):
    """Load dataset from processed data directory."""
    data_path = Path(f"rw3_full_experiments/data/processed/{dataset_name}/data.json")
    with open(data_path) as f:
        return json.load(f)

def get_texts(data, split):
    """Extract texts from a split."""
    return data[split]["texts"]

def save_embeddings(embeddings, dataset, model_key, split, cache_dir):
    """Save embeddings to cache directory."""
    output_dir = cache_dir / dataset / model_key
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split}.npy"
    np.save(output_path, embeddings)
    return output_path

def check_complete(cache_dir, dataset, model_key):
    """Check if all 4 splits already exist for a combination."""
    for split in SPLITS:
        path = cache_dir / dataset / model_key / f"{split}.npy"
        if not path.exists():
            return False
    return True

def write_audit_log(audit_dir, dataset, model_key, results):
    """Write audit log for the extraction."""
    audit_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = audit_dir / f"embedding_{dataset}_{model_key}_{timestamp}.json"

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset,
        "model_key": model_key,
        "model_name": MODELS[model_key],
        "results": results
    }

    with open(log_path, "w") as f:
        json.dump(log_entry, f, indent=2)

    return log_path

def process_datasets_with_model(model_key, datasets, cache_dir, audit_dir):
    """Process multiple datasets with the same model (load once)."""
    from sentence_transformers import SentenceTransformer

    model_name = MODELS[model_key]
    prefix = MODEL_PREFIX.get(model_key)

    print(f"\n{'='*70}")
    print(f"Loading model: {model_key} ({model_name})")
    print(f"Datasets to process: {datasets}")
    print(f"{'='*70}")

    model = SentenceTransformer(model_name)

    all_results = {}

    for dataset in datasets:
        print(f"\n{'─'*60}")
        print(f"Processing: {dataset} × {model_key}")
        print(f"{'─'*60}")

        # Check if already complete
        if check_complete(cache_dir, dataset, model_key):
            print(f"  ✓ Already complete (4/4 splits), skipping.")
            continue

        # Load data
        print(f"  Loading data...")
        data = load_data(dataset)

        results = {
            "splits": {},
            "total_time": 0
        }

        total_start = time.time()

        for split in SPLITS:
            split_path = cache_dir / dataset / model_key / f"{split}.npy"
            if split_path.exists():
                print(f"  {split}: already exists, skipping")
                continue

            print(f"  {split}:", end=" ", flush=True)
            texts = get_texts(data, split)

            # Apply model-specific prefix
            encode_texts = [prefix + t for t in texts] if prefix else texts

            start_time = time.time()
            embeddings = model.encode(
                encode_texts,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            elapsed = time.time() - start_time

            # Save embeddings
            output_path = save_embeddings(embeddings, dataset, model_key, split, cache_dir)

            results["splits"][split] = {
                "samples": len(texts),
                "dimensions": embeddings.shape[1],
                "time_seconds": round(elapsed, 2),
                "output_path": str(output_path)
            }

            print(f"{len(texts)} samples, {embeddings.shape[1]}D, {elapsed:.2f}s")

        results["total_time"] = round(time.time() - total_start, 2)
        if results["splits"]:
            results["embedding_dim"] = list(results["splits"].values())[0]["dimensions"]

            # Write audit log
            log_path = write_audit_log(audit_dir, dataset, model_key, results)
            print(f"  Audit log: {log_path}")

            all_results[f"{dataset}_{model_key}"] = results

    return all_results

def main():
    """Main entry point."""
    cache_dir = Path("cache/embeddings")
    audit_dir = Path("results/audit")

    # Process in model-grouped order to minimize model reloading
    # Order: minilm → bge → e5 → mpnet

    all_results = {}
    overall_start = time.time()

    # 1. minilm: clinc150
    results = process_datasets_with_model("minilm", ["clinc150"], cache_dir, audit_dir)
    all_results.update(results)

    # 2. bge: clinc150, newsgroups
    results = process_datasets_with_model("bge", ["clinc150", "newsgroups"], cache_dir, audit_dir)
    all_results.update(results)

    # 3. e5: clinc150, hwu64, massive
    results = process_datasets_with_model("e5", ["clinc150", "hwu64", "massive"], cache_dir, audit_dir)
    all_results.update(results)

    # 4. mpnet: hwu64, massive
    results = process_datasets_with_model("mpnet", ["hwu64", "massive"], cache_dir, audit_dir)
    all_results.update(results)

    overall_time = time.time() - overall_start

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY - RW3 v4.1 Stage 2 Batch 4")
    print("="*70)

    for combo, result in all_results.items():
        print(f"\n{combo}:")
        print(f"  Embedding dimension: {result.get('embedding_dim', 'N/A')}")
        print(f"  Total time: {result['total_time']}s")
        total_samples = 0
        for split, info in result["splits"].items():
            print(f"    {split}: {info['samples']} samples, {info['dimensions']}D, {info['time_seconds']}s")
            total_samples += info['samples']
        print(f"  Total samples: {total_samples}")

    print(f"\n{'='*70}")
    print(f"Overall time: {overall_time:.2f}s ({overall_time/60:.1f} min)")
    print(f"Combinations processed: {len(all_results)}")
    print(f"{'='*70}")

    return all_results

if __name__ == "__main__":
    main()
