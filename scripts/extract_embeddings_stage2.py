#!/usr/bin/env python3
"""
RW3 v4.1 Stage 2 - Embedding Extraction Script
Extracts embeddings for specified dataset × model combinations.
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

def extract_embeddings(texts, model_name, model_key):
    """Extract embeddings using sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    print(f"  Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"  Encoding {len(texts)} texts...")
    start_time = time.time()

    # For BGE models, add instruction prefix
    if "bge" in model_key:
        # BGE recommends adding query prefix for retrieval tasks
        # For embedding tasks, we use texts directly
        pass

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # L2 normalize
    )

    elapsed = time.time() - start_time
    return embeddings, elapsed

def save_embeddings(embeddings, dataset, model_key, split, cache_dir):
    """Save embeddings to cache directory."""
    output_dir = cache_dir / dataset / model_key
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split}.npy"
    np.save(output_path, embeddings)
    return output_path

def check_existing(cache_dir, dataset, model_key, split):
    """Check if embeddings already exist."""
    path = cache_dir / dataset / model_key / f"{split}.npy"
    return path.exists()

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

def process_combination(dataset, model_key, cache_dir, audit_dir):
    """Process one dataset × model combination."""
    model_name = MODELS[model_key]
    print(f"\n{'='*60}")
    print(f"Processing: {dataset} × {model_key}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    # Check for existing embeddings
    existing = []
    to_process = []
    for split in SPLITS:
        if check_existing(cache_dir, dataset, model_key, split):
            existing.append(split)
        else:
            to_process.append(split)

    if existing:
        print(f"  Skipping existing: {existing}")

    if not to_process:
        print(f"  All splits already exist, skipping.")
        return None

    # Load data
    print(f"  Loading data...")
    data = load_data(dataset)

    # Load model once
    from sentence_transformers import SentenceTransformer
    print(f"  Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    results = {
        "splits": {},
        "total_time": 0
    }

    total_start = time.time()

    for split in to_process:
        print(f"\n  Processing split: {split}")
        texts = get_texts(data, split)

        # Apply model-specific prefix (e.g., e5 requires "query: " prefix)
        prefix = MODEL_PREFIX.get(model_key)
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

        print(f"    Samples: {len(texts)}")
        print(f"    Dimensions: {embeddings.shape[1]}")
        print(f"    Time: {elapsed:.2f}s")
        print(f"    Saved: {output_path}")

    results["total_time"] = round(time.time() - total_start, 2)
    results["embedding_dim"] = embeddings.shape[1]

    # Write audit log
    log_path = write_audit_log(audit_dir, dataset, model_key, results)
    print(f"\n  Audit log: {log_path}")

    return results

def main():
    """Main entry point."""
    # Define combinations to process (batch 3: banking77_alpha e5/mpnet + hwu64 minilm/bge)
    combinations = [
        ("banking77_alpha", "e5"),
        ("banking77_alpha", "mpnet"),
        ("hwu64", "minilm"),
        ("hwu64", "bge"),
    ]

    cache_dir = Path("cache/embeddings")
    audit_dir = Path("results/audit")

    all_results = {}

    for dataset, model_key in combinations:
        result = process_combination(dataset, model_key, cache_dir, audit_dir)
        if result:
            all_results[f"{dataset}_{model_key}"] = result

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for combo, result in all_results.items():
        print(f"\n{combo}:")
        print(f"  Embedding dimension: {result.get('embedding_dim', 'N/A')}")
        print(f"  Total time: {result['total_time']}s")
        for split, info in result["splits"].items():
            print(f"  {split}: {info['samples']} samples, {info['dimensions']}D, {info['time_seconds']}s")

if __name__ == "__main__":
    main()
