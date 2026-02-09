#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_mock_data.py — 模拟数据生成器

生成可配置规模的完整模拟数据集，用于系统测试。
生成的数据格式与真实数据完全一致。
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def generate_stock_codes(n_stocks):
    """
    生成模拟股票代码列表
    前一半为深市(000xxx.SZ)，后一半为沪市(600xxx.SH)
    """
    codes = []
    n_sz = n_stocks // 2
    n_sh = n_stocks - n_sz
    for i in range(n_sz):
        codes.append(f"{i + 1:06d}.SZ")
    for i in range(n_sh):
        codes.append(f"{600000 + i:06d}.SH")
    return codes


def generate_trading_dates(n_dates, start_date='2024-01-02'):
    """
    生成交易日序列（跳过周末）
    返回 YYYY.MM.DD 格式的字符串列表
    """
    dates = pd.bdate_range(start=start_date, periods=n_dates)
    return [d.strftime('%Y.%m.%d') for d in dates]


def generate_concept_matrix(n_stocks, n_concepts, sparsity=0.99):
    """
    生成股票×概念关联矩阵
    sparsity: 稀疏度（0.99 表示 99% 为 0）
    """
    matrix = np.zeros((n_stocks, n_concepts), dtype=np.float64)
    n_nonzero = int(n_stocks * n_concepts * (1 - sparsity))
    # 随机填充非零元素
    rows = np.random.randint(0, n_stocks, size=n_nonzero)
    cols = np.random.randint(0, n_concepts, size=n_nonzero)
    values = np.random.uniform(0.1, 1.0, size=n_nonzero)
    matrix[rows, cols] = values
    return matrix


def generate_factor_csvs(output_folder, stock_codes, trading_dates, n_factors):
    """
    生成因子CSV文件
    每个因子一个文件，格式: TradingDate, SecurityID, Value
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    from tqdm import tqdm
    for f_idx in tqdm(range(n_factors), desc='生成因子CSV'):
        factor_name = f"factor_{f_idx + 1:03d}"
        rows = []
        for date in trading_dates:
            # 每天随机选择 80%-100% 的股票有数据（模拟缺失）
            n_available = np.random.randint(int(len(stock_codes) * 0.8), len(stock_codes) + 1)
            selected = np.random.choice(stock_codes, size=n_available, replace=False)
            values = np.random.randn(n_available) * 0.5 + np.random.randn() * 0.1
            for stock, val in zip(selected, values):
                rows.append((date, stock, round(val, 6)))

        df = pd.DataFrame(rows, columns=['TradingDate', 'SecurityID', 'Value'])
        df.to_csv(output_folder / f"{factor_name}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description='生成模拟数据集')
    parser.add_argument('--n_stocks', type=int, default=200, help='股票数量 (默认: 200)')
    parser.add_argument('--n_dates', type=int, default=60, help='交易日数量 (默认: 60)')
    parser.add_argument('--n_factors', type=int, default=20, help='因子数量 (默认: 20)')
    parser.add_argument('--n_concepts', type=int, default=500, help='概念数量 (默认: 500)')
    parser.add_argument('--sparsity', type=float, default=0.99, help='概念矩阵稀疏度 (默认: 0.99)')
    parser.add_argument('--output_dir', type=str, default='mock_data', help='输出目录 (默认: mock_data)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子 (默认: 42)')
    args = parser.parse_args()

    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  模拟数据生成器")
    print("=" * 60)
    print(f"  股票数量:   {args.n_stocks}")
    print(f"  交易日数量: {args.n_dates}")
    print(f"  因子数量:   {args.n_factors}")
    print(f"  概念数量:   {args.n_concepts}")
    print(f"  稀疏度:     {args.sparsity}")
    print(f"  输出目录:   {output_dir}")
    print("=" * 60)

    # 1. 生成基础数据
    print("\n[1/4] 生成股票代码...")
    stock_codes = generate_stock_codes(args.n_stocks)
    print(f"  生成 {len(stock_codes)} 个股票代码")
    print(f"  示例: {stock_codes[:3]} ... {stock_codes[-3:]}")

    print("\n[2/4] 生成交易日期...")
    trading_dates = generate_trading_dates(args.n_dates)
    print(f"  生成 {len(trading_dates)} 个交易日")
    print(f"  范围: {trading_dates[0]} ~ {trading_dates[-1]}")

    # 2. 生成概念矩阵
    print("\n[3/4] 生成概念关联矩阵...")
    concept_matrix = generate_concept_matrix(args.n_stocks, args.n_concepts, args.sparsity)
    npy_path = output_dir / 'concept_matrix.npy'
    np.save(npy_path, concept_matrix)
    n_nonzero = np.count_nonzero(concept_matrix)
    total = concept_matrix.size
    actual_sparsity = 1 - n_nonzero / total
    print(f"  矩阵形状: {concept_matrix.shape}")
    print(f"  非零元素: {n_nonzero} / {total} (稀疏度: {actual_sparsity:.4f})")
    print(f"  保存至: {npy_path}")

    # 同时保存股票代码列表（供 concept_embedding 使用）
    codes_path = output_dir / 'stock_codes.txt'
    with open(codes_path, 'w') as f:
        for code in stock_codes:
            f.write(code + '\n')
    print(f"  股票代码列表保存至: {codes_path}")

    # 3. 生成因子CSV
    print(f"\n[4/4] 生成 {args.n_factors} 个因子CSV...")
    factor_dir = output_dir / 'factors'
    generate_factor_csvs(factor_dir, stock_codes, trading_dates, args.n_factors)
    print(f"  因子文件保存至: {factor_dir}")

    # 4. 验证
    print("\n" + "=" * 60)
    print("  验证生成的数据")
    print("=" * 60)

    # 验证 npy
    loaded = np.load(npy_path)
    print(f"\n  概念矩阵: shape={loaded.shape}, dtype={loaded.dtype}")
    assert loaded.shape == (args.n_stocks, args.n_concepts)
    assert loaded.dtype == np.float64

    # 验证因子CSV
    sample_csv = sorted(factor_dir.glob('*.csv'))[0]
    df_sample = pd.read_csv(sample_csv)
    print(f"\n  示例因子文件: {sample_csv.name}")
    print(f"  列名: {list(df_sample.columns)}")
    print(f"  行数: {len(df_sample)}")
    print(f"  前3行:")
    print(df_sample.head(3).to_string(index=False))

    print("\n" + "=" * 60)
    print("  ✓ 模拟数据生成完成！")
    print(f"  输出目录: {output_dir.resolve()}")
    print("=" * 60)


if __name__ == '__main__':
    main()
