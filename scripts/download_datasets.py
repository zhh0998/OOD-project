#!/usr/bin/env python3
"""
Download Real Datasets for RW1 Preliminary Experiments

Datasets:
1. NYT10 - OpenNRE benchmark
2. FewRel - Few-shot relation extraction
3. Re-DocRED - Document-level RE with revised annotations
4. NYT-H - NYT with human labels
"""

import os
import sys
import json
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from typing import Optional
import ssl

# Disable SSL verification for some sources (use with caution)
ssl._create_default_https_context = ssl._create_unverified_context


class DatasetDownloader:
    """Download datasets from official sources"""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, url: str, save_path: Path, desc: str = "") -> bool:
        """Download a file with progress indicator"""
        try:
            print(f"  Downloading: {desc or url}")
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Add headers to avoid 403
            request = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )

            with urllib.request.urlopen(request, timeout=60) as response:
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                block_size = 8192

                with open(save_path, 'wb') as f:
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        downloaded += len(buffer)
                        f.write(buffer)

                        if total_size > 0:
                            percent = downloaded * 100 / total_size
                            sys.stdout.write(f"\r  Progress: {percent:.1f}%")
                            sys.stdout.flush()

                print(f"\r  Downloaded: {save_path.name} ({downloaded / 1024 / 1024:.1f} MB)")
                return True

        except Exception as e:
            print(f"\n  Error downloading {url}: {e}")
            return False

    def download_nyt10(self) -> bool:
        """
        Download NYT10 dataset from OpenNRE

        Source: https://github.com/thunlp/OpenNRE
        Files: nyt10_train.txt, nyt10_test.txt, nyt10_rel2id.json
        """
        print("\n" + "="*60)
        print("Downloading NYT10 Dataset")
        print("="*60)

        nyt10_dir = self.data_dir / "nyt10"
        nyt10_dir.mkdir(parents=True, exist_ok=True)

        # Try multiple sources
        sources = [
            # OpenNRE GitHub raw files
            {
                "base": "https://raw.githubusercontent.com/thunlp/OpenNRE/master/benchmark/nyt10/",
                "files": ["nyt10_rel2id.json"]
            },
            # Tsinghua Cloud (main data)
            {
                "url": "https://cloud.tsinghua.edu.cn/f/09265750ae6340429827/?dl=1",
                "name": "nyt10.zip"
            },
            # Alternative: THUNLP OpenNRE data release
            {
                "url": "https://github.com/thunlp/OpenNRE/raw/master/benchmark/download_nyt10.sh",
                "type": "script"
            }
        ]

        success = False

        # First, try to download rel2id which is usually available
        rel2id_url = "https://raw.githubusercontent.com/thunlp/OpenNRE/master/benchmark/nyt10/nyt10_rel2id.json"
        rel2id_path = nyt10_dir / "nyt10_rel2id.json"

        if self.download_file(rel2id_url, rel2id_path, "nyt10_rel2id.json"):
            print("  ✓ Downloaded rel2id.json")

        # Try Tsinghua cloud
        zip_path = nyt10_dir / "nyt10.zip"
        tsinghua_url = "https://cloud.tsinghua.edu.cn/f/09265750ae6340429827/?dl=1"

        if self.download_file(tsinghua_url, zip_path, "nyt10.zip from Tsinghua Cloud"):
            try:
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(nyt10_dir)
                print("  ✓ Extracted nyt10.zip")
                zip_path.unlink()  # Remove zip after extraction
                success = True
            except Exception as e:
                print(f"  Error extracting: {e}")

        # If Tsinghua fails, try HuggingFace
        if not success:
            print("  Trying HuggingFace datasets...")
            try:
                from datasets import load_dataset
                dataset = load_dataset("thunlp/nyt10")

                # Save to local files
                for split in ['train', 'test']:
                    if split in dataset:
                        save_path = nyt10_dir / f"nyt10_{split}.txt"
                        with open(save_path, 'w') as f:
                            for item in dataset[split]:
                                f.write(json.dumps(item) + '\n')
                        print(f"  ✓ Saved {split} split")
                success = True
            except Exception as e:
                print(f"  HuggingFace failed: {e}")

        # Verify
        if success or (nyt10_dir / "nyt10_train.txt").exists():
            print("  ✓ NYT10 dataset ready")
            return True
        else:
            print("  ✗ NYT10 download failed - will use existing data in ./nyt10")
            return False

    def download_fewrel(self) -> bool:
        """
        Download FewRel dataset

        Source: https://github.com/thunlp/FewRel
        Files: train_wiki.json, val_wiki.json
        """
        print("\n" + "="*60)
        print("Downloading FewRel Dataset")
        print("="*60)

        fewrel_dir = self.data_dir / "fewrel"
        fewrel_dir.mkdir(parents=True, exist_ok=True)

        # Direct GitHub URLs
        files = {
            "train_wiki.json": "https://raw.githubusercontent.com/thunlp/FewRel/master/data/train_wiki.json",
            "val_wiki.json": "https://raw.githubusercontent.com/thunlp/FewRel/master/data/val_wiki.json",
            "pid2name.json": "https://raw.githubusercontent.com/thunlp/FewRel/master/data/pid2name.json"
        }

        success_count = 0
        for filename, url in files.items():
            save_path = fewrel_dir / filename
            if self.download_file(url, save_path, filename):
                success_count += 1

        # If direct download fails, try HuggingFace
        if success_count < 2:
            print("  Trying HuggingFace datasets...")
            try:
                from datasets import load_dataset
                dataset = load_dataset("few_rel", "default")

                # Convert to FewRel format
                train_data = {}
                for item in dataset['train']:
                    rel = item['relation']
                    if rel not in train_data:
                        train_data[rel] = []
                    train_data[rel].append({
                        'tokens': item['tokens'],
                        'h': item['head'],
                        't': item['tail']
                    })

                with open(fewrel_dir / "train_wiki.json", 'w') as f:
                    json.dump(train_data, f)
                print("  ✓ Saved train_wiki.json from HuggingFace")
                success_count = 2
            except Exception as e:
                print(f"  HuggingFace failed: {e}")

        if success_count >= 2:
            print("  ✓ FewRel dataset ready")
            return True
        else:
            print("  ✗ FewRel download failed")
            return False

    def download_redocred(self) -> bool:
        """
        Download Re-DocRED dataset

        Source: https://github.com/tonytan48/Re-DocRED
        Files: train_revised.json, dev_revised.json, test_revised.json
        """
        print("\n" + "="*60)
        print("Downloading Re-DocRED Dataset")
        print("="*60)

        redocred_dir = self.data_dir / "redocred"
        redocred_dir.mkdir(parents=True, exist_ok=True)

        # GitHub raw URLs
        base_url = "https://raw.githubusercontent.com/tonytan48/Re-DocRED/main/data/"
        files = [
            "train_revised.json",
            "dev_revised.json",
            "test_revised.json"
        ]

        success_count = 0
        for filename in files:
            url = base_url + filename
            save_path = redocred_dir / filename
            if self.download_file(url, save_path, filename):
                success_count += 1

        # Also try original DocRED if Re-DocRED fails
        if success_count == 0:
            print("  Trying original DocRED from HuggingFace...")
            try:
                from datasets import load_dataset
                dataset = load_dataset("docred")

                for split in ['train', 'validation']:
                    out_name = f"{split}.json" if split != 'validation' else "dev.json"
                    data = [item for item in dataset[split]]
                    with open(redocred_dir / out_name, 'w') as f:
                        json.dump(data, f)
                    print(f"  ✓ Saved {out_name}")
                success_count = 2
            except Exception as e:
                print(f"  HuggingFace failed: {e}")

        if success_count >= 2:
            print("  ✓ Re-DocRED dataset ready")
            return True
        else:
            print("  ✗ Re-DocRED download failed")
            return False

    def download_nyth(self) -> bool:
        """
        Download NYT-H dataset (NYT with human labels)

        Source: https://github.com/Spico197/NYT-H
        File: test_nyt_human.json
        """
        print("\n" + "="*60)
        print("Downloading NYT-H Dataset")
        print("="*60)

        nyth_dir = self.data_dir / "nyth"
        nyth_dir.mkdir(parents=True, exist_ok=True)

        # GitHub raw URLs
        base_url = "https://raw.githubusercontent.com/Spico197/NYT-H/main/data/"
        files = [
            "test_nyt_human.json",
            "human_anno.json"
        ]

        success = False
        for filename in files:
            url = base_url + filename
            save_path = nyth_dir / filename
            if self.download_file(url, save_path, filename):
                success = True
                break

        # Alternative URL structure
        if not success:
            alt_url = "https://raw.githubusercontent.com/Spico197/NYT-H/master/data/test_nyt_human.json"
            if self.download_file(alt_url, nyth_dir / "test_nyt_human.json", "test_nyt_human.json"):
                success = True

        if success:
            print("  ✓ NYT-H dataset ready")
            return True
        else:
            print("  ✗ NYT-H download failed")
            return False

    def download_all(self) -> dict:
        """Download all datasets and return status"""
        results = {
            'nyt10': self.download_nyt10(),
            'fewrel': self.download_fewrel(),
            'redocred': self.download_redocred(),
            'nyth': self.download_nyth()
        }

        print("\n" + "="*60)
        print("Download Summary")
        print("="*60)
        for name, success in results.items():
            status = "✓" if success else "✗"
            print(f"  {status} {name}")

        return results


def verify_datasets(data_dir: str = "./data") -> dict:
    """Verify downloaded datasets"""
    print("\n" + "="*60)
    print("Verifying Datasets")
    print("="*60)

    data_path = Path(data_dir)
    results = {}

    # Check NYT10
    nyt10_dir = data_path / "nyt10"
    if nyt10_dir.exists():
        train_file = nyt10_dir / "nyt10_train.txt"
        if train_file.exists():
            with open(train_file, 'r') as f:
                count = sum(1 for _ in f)
            print(f"  NYT10: {count} training samples")
            results['nyt10'] = {'train_samples': count, 'status': 'ok'}
        else:
            print("  NYT10: train file not found")
            results['nyt10'] = {'status': 'missing'}

    # Check FewRel
    fewrel_dir = data_path / "fewrel"
    if fewrel_dir.exists():
        train_file = fewrel_dir / "train_wiki.json"
        if train_file.exists():
            with open(train_file, 'r') as f:
                data = json.load(f)
            n_relations = len(data)
            n_samples = sum(len(v) for v in data.values())
            print(f"  FewRel: {n_relations} relations, {n_samples} samples")
            results['fewrel'] = {'relations': n_relations, 'samples': n_samples, 'status': 'ok'}
        else:
            print("  FewRel: train file not found")
            results['fewrel'] = {'status': 'missing'}

    # Check Re-DocRED
    redocred_dir = data_path / "redocred"
    if redocred_dir.exists():
        dev_file = redocred_dir / "dev_revised.json"
        if not dev_file.exists():
            dev_file = redocred_dir / "dev.json"
        if dev_file.exists():
            with open(dev_file, 'r') as f:
                data = json.load(f)
            n_docs = len(data)
            print(f"  Re-DocRED: {n_docs} dev documents")
            results['redocred'] = {'documents': n_docs, 'status': 'ok'}
        else:
            print("  Re-DocRED: dev file not found")
            results['redocred'] = {'status': 'missing'}

    # Check NYT-H
    nyth_dir = data_path / "nyth"
    if nyth_dir.exists():
        for fname in ["test_nyt_human.json", "human_anno.json", "nyth.json"]:
            test_file = nyth_dir / fname
            if test_file.exists():
                with open(test_file, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    n_samples = len(data)
                else:
                    n_samples = sum(len(v) if isinstance(v, list) else 1 for v in data.values())
                print(f"  NYT-H: {n_samples} samples")
                results['nyth'] = {'samples': n_samples, 'status': 'ok'}
                break
        else:
            print("  NYT-H: data file not found")
            results['nyth'] = {'status': 'missing'}

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download datasets for RW1 experiments")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--verify_only", action="store_true", help="Only verify existing data")
    parser.add_argument("--dataset", type=str, choices=['nyt10', 'fewrel', 'redocred', 'nyth', 'all'],
                        default='all', help="Which dataset to download")

    args = parser.parse_args()

    if args.verify_only:
        verify_datasets(args.data_dir)
    else:
        downloader = DatasetDownloader(args.data_dir)

        if args.dataset == 'all':
            downloader.download_all()
        elif args.dataset == 'nyt10':
            downloader.download_nyt10()
        elif args.dataset == 'fewrel':
            downloader.download_fewrel()
        elif args.dataset == 'redocred':
            downloader.download_redocred()
        elif args.dataset == 'nyth':
            downloader.download_nyth()

        # Verify after download
        verify_datasets(args.data_dir)
