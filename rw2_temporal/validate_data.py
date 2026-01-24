#!/usr/bin/env python3
"""Data validation script for RW2 experiments"""
import sys
from data.data_loader import validate_dataset

def main():
    print("\n" + "=" * 60)
    print("RW2 Data Validation")
    print("=" * 60)

    dataset = sys.argv[1] if len(sys.argv) > 1 else 'tgbl-wiki'
    success, stats = validate_dataset(dataset)

    if success:
        print("✅ 数据验证通过，可以开始实验！")
        return 0
    else:
        print("❌ 数据验证失败！")
        return 1

if __name__ == '__main__':
    sys.exit(main())
