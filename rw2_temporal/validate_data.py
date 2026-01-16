#!/usr/bin/env python3
"""
Data Validation Script for RW2 Temporal Network Embedding.

This script validates that we're using REAL TGB datasets, not simulated data.
SIMULATED DATA IS STRICTLY PROHIBITED for valid experiments.

Usage:
    python validate_data.py
    python validate_data.py --dataset tgbl-wiki
    python validate_data.py --all

Author: RW2 Temporal Network Embedding Project
"""

import argparse
import sys
import os
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.data_loader import RealDataLoader


def validate_single_dataset(dataset_name: str, verbose: bool = True) -> bool:
    """
    Validate a single dataset.

    Returns:
        True if validation passed, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Validating Dataset: {dataset_name}")
    print(f"{'='*60}")

    try:
        loader = RealDataLoader(dataset_name=dataset_name)
        data = loader.get_data()

        if verbose:
            print(f"\nDataset Statistics:")
            print(f"  Nodes: {data.num_nodes:,}")
            print(f"  Edges: {data.num_edges:,}")
            print(f"  Timestamp range: {data.timestamps.min():.2f} - {data.timestamps.max():.2f}")
            print(f"  Timestamp std: {np.std(data.timestamps):.2f}")
            print(f"  Unique timestamps: {len(np.unique(data.timestamps)):,}")

            # Additional statistics
            unique_src = len(np.unique(data.src))
            unique_dst = len(np.unique(data.dst))
            print(f"  Unique source nodes: {unique_src:,}")
            print(f"  Unique destination nodes: {unique_dst:,}")

            # Edge density
            density = data.num_edges / (data.num_nodes * data.num_nodes)
            print(f"  Edge density: {density:.6f}")

        print(f"\n[OK] Dataset '{dataset_name}' validation PASSED")
        return True

    except Exception as e:
        print(f"\n[FAILED] Dataset '{dataset_name}' validation FAILED")
        print(f"Error: {e}")
        return False


def validate_all_datasets() -> dict:
    """
    Validate all TGB datasets.

    Returns:
        Dictionary with validation results
    """
    datasets = ['tgbl-wiki', 'tgbl-review', 'tgbl-coin']
    results = {}

    print("\n" + "=" * 60)
    print("RW2 Temporal Network Embedding - Data Validation")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Datasets to validate: {', '.join(datasets)}")

    for dataset in datasets:
        results[dataset] = validate_single_dataset(dataset)

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for dataset, status in results.items():
        status_str = "[OK] PASSED" if status else "[X] FAILED"
        print(f"  {dataset}: {status_str}")

    print(f"\nTotal: {passed}/{total} datasets validated")

    if passed == total:
        print("\n[SUCCESS] All datasets validated. Ready for experiments!")
    else:
        print("\n[WARNING] Some datasets failed validation.")
        print("Please check your data sources and try again.")
        print("Remember: SIMULATED DATA IS NOT ALLOWED for valid experiments!")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate TGB datasets for RW2 experiments"
    )
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default=None,
        help="Dataset name to validate (e.g., tgbl-wiki)"
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help="Validate all datasets"
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help="Quiet mode (less verbose output)"
    )

    args = parser.parse_args()

    if args.all or args.dataset is None:
        results = validate_all_datasets()
        # Exit with error code if any validation failed
        if not all(results.values()):
            sys.exit(1)
    else:
        if not validate_single_dataset(args.dataset, verbose=not args.quiet):
            sys.exit(1)

    print("\nData validation complete!")


if __name__ == '__main__':
    main()
