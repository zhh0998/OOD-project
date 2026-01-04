#!/usr/bin/env python3
"""
Real Data Verification Script
Validates that all datasets are real (not LFS pointers or synthetic).
"""
import os
import sys
import json

def verify_not_lfs_pointer(filepath: str) -> bool:
    """Check if file is not a Git LFS pointer."""
    with open(filepath, 'rb') as f:
        first_bytes = f.read(100)
        if b'git-lfs' in first_bytes or b'version https' in first_bytes:
            return False
    return True

def verify_file_size(filepath: str, min_size_bytes: int) -> bool:
    """Check if file meets minimum size requirement."""
    return os.path.getsize(filepath) >= min_size_bytes

def verify_nyt10(data_dir: str) -> bool:
    """Verify NYT10 dataset."""
    print("=" * 60)
    print("VERIFYING NYT10")
    print("=" * 60)

    train_path = os.path.join(data_dir, 'nyt10_train.txt')

    if not os.path.exists(train_path):
        print(f"❌ FATAL: {train_path} not found!")
        return False

    if not verify_not_lfs_pointer(train_path):
        print(f"❌ FATAL: {train_path} is LFS pointer!")
        return False

    file_size = os.path.getsize(train_path)
    if not verify_file_size(train_path, 100 * 1024 * 1024):  # 100MB
        print(f"❌ FATAL: {train_path} too small ({file_size} bytes)!")
        return False

    # Count samples
    with open(train_path, 'r') as f:
        num_lines = sum(1 for _ in f)

    if num_lines < 500000:
        print(f"❌ FATAL: Only {num_lines} samples, expected >= 500,000!")
        return False

    print(f"✅ NYT10 verified: {num_lines} samples, {file_size/1024/1024:.1f}MB")
    return True

def verify_nyth(data_dir: str) -> bool:
    """Verify NYT-H dataset."""
    print("=" * 60)
    print("VERIFYING NYT-H")
    print("=" * 60)

    # Check for the extracted data directory
    test_path = os.path.join(data_dir, 'data', 'test.json')

    if not os.path.exists(test_path):
        print(f"❌ FATAL: {test_path} not found!")
        return False

    # Load and verify
    with open(test_path, 'r') as f:
        data = [json.loads(line) for line in f]

    if len(data) < 9000:
        print(f"❌ FATAL: Only {len(data)} samples, expected >= 9,000!")
        return False

    # Check for human annotation field
    sample = data[0]
    if 'bag_label' not in sample:
        print("❌ FATAL: Missing 'bag_label' field!")
        return False

    print(f"✅ NYT-H verified: {len(data)} samples with human annotations")
    return True

def verify_fewrel(data_dir: str) -> bool:
    """Verify FewRel dataset."""
    print("=" * 60)
    print("VERIFYING FewRel")
    print("=" * 60)

    train_path = os.path.join(data_dir, 'train_wiki.json')

    if not os.path.exists(train_path):
        print(f"❌ FATAL: {train_path} not found!")
        return False

    with open(train_path, 'r') as f:
        data = json.load(f)

    if len(data) < 60:
        print(f"❌ FATAL: Only {len(data)} relations, expected >= 60!")
        return False

    total_samples = sum(len(v) for v in data.values())
    if total_samples < 40000:
        print(f"❌ FATAL: Only {total_samples} samples, expected >= 40,000!")
        return False

    print(f"✅ FewRel verified: {len(data)} relations, {total_samples} samples")
    return True

def verify_docred(data_dir: str) -> bool:
    """Verify DocRED dataset."""
    print("=" * 60)
    print("VERIFYING Re-DocRED")
    print("=" * 60)

    dev_path = os.path.join(data_dir, 'dev_revised.json')

    if not os.path.exists(dev_path):
        print(f"❌ FATAL: {dev_path} not found!")
        return False

    with open(dev_path, 'r') as f:
        data = json.load(f)

    if len(data) < 500:
        print(f"❌ FATAL: Only {len(data)} documents, expected >= 500!")
        return False

    print(f"✅ Re-DocRED verified: {len(data)} documents")
    return True

def main():
    """Run all verifications."""
    base_dir = '/home/user/OOD-project'

    all_passed = True

    # Verify NYT10
    if not verify_nyt10(os.path.join(base_dir, 'data/nyt10_real')):
        all_passed = False

    # Verify NYT-H
    if not verify_nyth(os.path.join(base_dir, 'data/nyth')):
        all_passed = False

    # Verify FewRel
    if not verify_fewrel(os.path.join(base_dir, 'data/fewrel')):
        all_passed = False

    # Verify DocRED
    if not verify_docred(os.path.join(base_dir, 'data/redocred')):
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL DATASETS VERIFIED AS REAL DATA")
    else:
        print("❌ SOME DATASETS FAILED VERIFICATION")
        sys.exit(1)
    print("=" * 60)

if __name__ == '__main__':
    main()
