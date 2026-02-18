#!/usr/bin/env python3
"""
RW3 v4.1 Stage 2 - Embedding Extraction Batch 5
Extracts embeddings for all remaining dataset × model combinations.
Groups by model to minimize model loading overhead.
"""

import json
import os
import sys
import time
import shutil
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

# Prefix required by certain models
MODEL_PREFIX = {
    "e5": "query: "
}

SPLITS = ["train_id", "cal_id", "test_id", "test_ood"]

CACHE_DIR = Path("cache/embeddings")
AUDIT_DIR = Path("results/audit")

# All 14 combinations grouped by model (model loaded once per group)
GROUPS = [
    ("mpnet", [
        "hwu64",
        "massive",
        "newsgroups",
    ]),
    ("e5", [
        "clinc150",
        "hwu64",
        "massive",
        "newsgroups",
    ]),
    ("minilm", [
        "clinc150",
        "nq_open",
    ]),
    ("bge", [
        "clinc150",
        "newsgroups",  # needs rm first if incomplete
        "nq_open",
    ]),
    # Tail combinations (model already cached from earlier groups)
    ("e5", [
        "nq_open",
    ]),
    ("mpnet", [
        "nq_open",
    ]),
]


def is_complete(dataset, model_key):
    """Check if all 4 split files exist."""
    d = CACHE_DIR / dataset / model_key
    return all((d / f"{s}.npy").exists() for s in SPLITS)


def load_data(dataset_name):
    data_path = Path(f"rw3_full_experiments/data/processed/{dataset_name}/data.json")
    with open(data_path) as f:
        return json.load(f)


def write_audit_log(dataset, model_key, results):
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = AUDIT_DIR / f"embedding_{dataset}_{model_key}_{ts}.json"
    entry = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset,
        "model_key": model_key,
        "model_name": MODELS[model_key],
        "results": results,
    }
    with open(log_path, "w") as f:
        json.dump(entry, f, indent=2)
    return log_path


def process_group(model_key, datasets):
    """Process multiple datasets with the same model (loaded once)."""
    from sentence_transformers import SentenceTransformer

    model_name = MODELS[model_key]

    # Filter to only datasets that need processing
    todo = []
    for ds in datasets:
        if is_complete(ds, model_key):
            print(f"  [SKIP] {ds} × {model_key} — all 4 splits exist")
        else:
            todo.append(ds)

    if not todo:
        return {}

    print(f"\n{'#'*60}")
    print(f"  Loading model: {model_key} ({model_name})")
    print(f"{'#'*60}")
    model = SentenceTransformer(model_name)

    prefix = MODEL_PREFIX.get(model_key)
    group_results = {}

    for ds in todo:
        print(f"\n{'='*60}")
        print(f"  {ds} × {model_key}")
        print(f"{'='*60}")

        data = load_data(ds)
        results = {"splits": {}, "total_time": 0}
        total_start = time.time()

        for split in SPLITS:
            # Skip individual splits that already exist
            npy_path = CACHE_DIR / ds / model_key / f"{split}.npy"
            if npy_path.exists():
                print(f"    {split}: exists, skipping")
                continue

            texts = data[split]["texts"]
            encode_texts = [prefix + t for t in texts] if prefix else texts

            t0 = time.time()
            embeddings = model.encode(
                encode_texts,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            elapsed = time.time() - t0

            # Save
            out_dir = CACHE_DIR / ds / model_key
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(npy_path, embeddings)

            results["splits"][split] = {
                "samples": len(texts),
                "dimensions": int(embeddings.shape[1]),
                "time_seconds": round(elapsed, 2),
                "output_path": str(npy_path),
            }
            print(f"    {split}: {len(texts)} samples, {embeddings.shape[1]}D, {elapsed:.2f}s → {npy_path}")

        results["total_time"] = round(time.time() - total_start, 2)
        if results["splits"]:
            first = next(iter(results["splits"].values()))
            results["embedding_dim"] = first["dimensions"]
        else:
            results["embedding_dim"] = "N/A"

        log_path = write_audit_log(ds, model_key, results)
        print(f"    Audit: {log_path}")
        group_results[f"{ds}_{model_key}"] = results

    # Free model memory
    del model
    import gc; gc.collect()

    return group_results


def main():
    # Special handling: rm newsgroups/bge if incomplete
    ng_bge = CACHE_DIR / "newsgroups" / "bge"
    if ng_bge.exists() and not is_complete("newsgroups", "bge"):
        print(f"[CLEANUP] Removing incomplete newsgroups/bge: {ng_bge}")
        shutil.rmtree(ng_bge)

    all_results = {}
    grand_start = time.time()

    for model_key, datasets in GROUPS:
        print(f"\n\n{'*'*60}")
        print(f"  MODEL GROUP: {model_key}")
        print(f"{'*'*60}")
        group_results = process_group(model_key, datasets)
        all_results.update(group_results)

    grand_elapsed = time.time() - grand_start

    # Final summary
    print(f"\n\n{'='*60}")
    print(f"  FINAL SUMMARY  (total: {grand_elapsed:.1f}s)")
    print(f"{'='*60}")

    for combo, result in all_results.items():
        print(f"\n  {combo}:")
        print(f"    dim={result.get('embedding_dim', 'N/A')}, total={result['total_time']}s")
        for split, info in result["splits"].items():
            print(f"      {split}: {info['samples']} samples, {info['dimensions']}D, {info['time_seconds']}s")

    # Verification: check all 14 expected combos
    expected = [
        ("hwu64", "mpnet"), ("massive", "mpnet"), ("newsgroups", "mpnet"),
        ("clinc150", "e5"), ("hwu64", "e5"), ("massive", "e5"), ("newsgroups", "e5"),
        ("clinc150", "minilm"), ("nq_open", "minilm"),
        ("clinc150", "bge"), ("newsgroups", "bge"), ("nq_open", "bge"),
        ("nq_open", "e5"), ("nq_open", "mpnet"),
    ]
    print(f"\n{'='*60}")
    print("  VERIFICATION")
    print(f"{'='*60}")
    all_ok = True
    for ds, mk in expected:
        ok = is_complete(ds, mk)
        status = "OK" if ok else "MISSING"
        if not ok:
            all_ok = False
        print(f"  [{status}] {ds} × {mk}")

    if all_ok:
        print("\n  All 14 combinations complete!")
    else:
        print("\n  WARNING: Some combinations are incomplete!")
        sys.exit(1)


if __name__ == "__main__":
    main()
