#!/usr/bin/env python3
"""
Stage 1: Data Preparation Script
Loads and processes all 6 datasets with unified split format.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path
from src.data.data_loader import DatasetLoader


def main():
    print("=" * 60)
    print("STAGE 1: DATA PREPARATION")
    print("=" * 60)

    # Initialize loader
    loader = DatasetLoader(
        output_dir="data/processed",
        cache_dir="cache/datasets"
    )

    # Banking77 random seeds (≥5 for robustness)
    banking77_seeds = [42, 123, 456, 789, 2024]

    # Load all datasets
    print("\nLoading all datasets...")
    datasets = {}
    cards = []

    # D1: CLINC150
    print("\n" + "-" * 40)
    print("D1: CLINC150")
    print("-" * 40)
    try:
        data, card = loader.load_clinc150()
        datasets['clinc150'] = data
        cards.append(card)
        loader.save_dataset(data, card, 'clinc150')
        print(card.to_markdown())
    except Exception as e:
        print(f"ERROR loading CLINC150: {e}")

    # D2: Banking77 (alphabetical + random seeds)
    print("\n" + "-" * 40)
    print("D2: Banking77 (alphabetical)")
    print("-" * 40)
    try:
        data, card = loader.load_banking77(partition='alphabetical')
        datasets['banking77_alpha'] = data
        cards.append(card)
        loader.save_dataset(data, card, 'banking77_alpha')
        print(card.to_markdown())
    except Exception as e:
        print(f"ERROR loading Banking77 alphabetical: {e}")

    for seed in banking77_seeds:
        print(f"\nD2: Banking77 (random seed={seed})")
        print("-" * 40)
        try:
            data, card = loader.load_banking77(partition='random', seed=seed)
            datasets[f'banking77_seed{seed}'] = data
            cards.append(card)
            loader.save_dataset(data, card, f'banking77_seed{seed}')
            print(card.to_markdown())
        except Exception as e:
            print(f"ERROR loading Banking77 seed {seed}: {e}")

    # D3: HWU64
    print("\n" + "-" * 40)
    print("D3: HWU64")
    print("-" * 40)
    try:
        data, card = loader.load_hwu64()
        datasets['hwu64'] = data
        cards.append(card)
        loader.save_dataset(data, card, 'hwu64')
        print(card.to_markdown())
    except Exception as e:
        print(f"ERROR loading HWU64: {e}")

    # D4: MASSIVE
    print("\n" + "-" * 40)
    print("D4: MASSIVE (en-US)")
    print("-" * 40)
    try:
        data, card = loader.load_massive()
        datasets['massive'] = data
        cards.append(card)
        loader.save_dataset(data, card, 'massive')
        print(card.to_markdown())
    except Exception as e:
        print(f"ERROR loading MASSIVE: {e}")

    # D5: 20Newsgroups
    print("\n" + "-" * 40)
    print("D5: 20Newsgroups")
    print("-" * 40)
    try:
        data, card = loader.load_20newsgroups()
        datasets['newsgroups'] = data
        cards.append(card)
        loader.save_dataset(data, card, 'newsgroups')
        print(card.to_markdown())
    except Exception as e:
        print(f"ERROR loading 20Newsgroups: {e}")

    # D6: NQ-Open
    print("\n" + "-" * 40)
    print("D6: NQ-Open")
    print("-" * 40)
    try:
        data, card = loader.load_nq_open()
        datasets['nq_open'] = data
        cards.append(card)
        loader.save_dataset(data, card, 'nq_open')
        print(card.to_markdown())
    except Exception as e:
        print(f"ERROR loading NQ-Open: {e}")

    # Generate combined data cards
    print("\n" + "=" * 60)
    print("STAGE 1 CHECKPOINT")
    print("=" * 60)

    cards_md = "# Data Cards\n\n"
    for card in cards:
        cards_md += card.to_markdown() + "\n"

    with open("data/processed/data_cards.md", 'w') as f:
        f.write(cards_md)

    print(f"\nDatasets loaded: {len(datasets)}")
    print(f"Data cards saved: data/processed/data_cards.md")

    # Summary table
    print("\n## Dataset Summary")
    print("-" * 80)
    print(f"{'Dataset':<25} {'Train':<10} {'Cal':<10} {'Test ID':<10} {'Test OOD':<10}")
    print("-" * 80)

    for name, data in datasets.items():
        train_n = len(data.get('train_id', {}).get('texts', []))
        cal_n = len(data.get('cal_id', {}).get('texts', []))
        test_id_n = len(data.get('test_id', {}).get('texts', []))
        test_ood_n = len(data.get('test_ood', {}).get('texts', []))
        print(f"{name:<25} {train_n:<10} {cal_n:<10} {test_id_n:<10} {test_ood_n:<10}")

    print("-" * 80)
    print("\n✓ Stage 1 complete. Ready for Stage 2.")

    return datasets


if __name__ == "__main__":
    main()
