#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_mock_data.py — 模拟数据生成器

生成可配置规模的完整模拟数据集，用于系统测试。
生成的数据格式与真实数据完全一致。

命令行参数（全部可选，有默认值）：
  --n_stocks    股票数量      默认 100
  --n_dates     交易日天数    默认 60
  --n_factors   因子文件数量  默认 10
  --n_concepts  概念维度数    默认 500
  --seed        随机种子      默认 42
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def generate_stock_codes(n_stocks):
    """
    生成模拟股票代码列表
    前一半为深市(000001.SZ ~ 000050.SZ)，后一半为沪市(600000.SH ~ 600049.SH)
    """
    codes = []
    n_sz = n_stocks // 2
    n_sh = n_stocks - n_sz
    for i in range(n_sz):
        codes.append(f"{i + 1:06d}.SZ")
    for i in range(n_sh):
        codes.append(f"{600000 + i:06d}.SH")
    return codes


def generate_trading_dates(n_dates, start_date='2024-06-03'):
    """
    生成交易日序列（只要工作日，跳过周末）
    返回 YYYY.MM.DD 格式的字符串列表
    """
    dates = pd.bdate_range(start=start_date, periods=n_dates)
    return [d.strftime('%Y.%m.%d') for d in dates]


def generate_concept_matrix(n_stocks, n_concepts, rng):
    """
    生成股票×概念关联矩阵
    - 每行随机 20~80 个非零元素
    - 非零值用 np.random.exponential(scale=5.0)
    - dtype: float64
    """
    matrix = np.zeros((n_stocks, n_concepts), dtype=np.float64)
    for i in range(n_stocks):
        n_nonzero = rng.integers(20, 81)  # 20~80 个非零元素
        cols = rng.choice(n_concepts, size=min(n_nonzero, n_concepts), replace=False)
        values = rng.exponential(scale=5.0, size=len(cols))
        matrix[i, cols] = values
    return matrix


def generate_factor_csvs(output_folder, stock_codes, trading_dates, n_factors, rng):
    """
    生成因子CSV文件
    每个因子一个文件，格式: TradingDate, SecurityID, Value
    - Value: np.random.exponential(0.12)
    - 每天随机 2%~5% 的股票缺失（模拟停牌）
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    n_stocks = len(stock_codes)

    for f_idx in tqdm(range(n_factors), desc='生成因子CSV'):
        factor_name = f"Factor_{f_idx + 1:03d}"
        rows_list = []
        for date in trading_dates:
            # 每天随机 2%~5% 的股票缺失
            miss_rate = rng.uniform(0.02, 0.05)
            n_missing = max(1, int(n_stocks * miss_rate))
            n_available = n_stocks - n_missing
            selected = rng.choice(stock_codes, size=n_available, replace=False)
            values = rng.exponential(scale=0.12, size=n_available)
            for stock, val in zip(selected, values):
                rows_list.append((date, stock, val))

        df = pd.DataFrame(rows_list, columns=['TradingDate', 'SecurityID', 'Value'])
        # 确保无重复行（同一天+同一股票只出现一次）
        df = df.drop_duplicates(subset=['TradingDate', 'SecurityID'], keep='last')
        df.to_csv(output_folder / f"{factor_name}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description='生成模拟数据集')
    parser.add_argument('--n_stocks', type=int, default=100, help='股票数量 (默认: 100)')
    parser.add_argument('--n_dates', type=int, default=60, help='交易日数量 (默认: 60)')
    parser.add_argument('--n_factors', type=int, default=10, help='因子数量 (默认: 10)')
    parser.add_argument('--n_concepts', type=int, default=500, help='概念数量 (默认: 500)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子 (默认: 42)')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print("=" * 60)
    print("  模拟数据生成器")
    print("=" * 60)
    print(f"  股票数量:   {args.n_stocks}")
    print(f"  交易日数量: {args.n_dates}")
    print(f"  因子数量:   {args.n_factors}")
    print(f"  概念数量:   {args.n_concepts}")
    print("=" * 60)

    # 1. 生成股票代码
    stock_codes = generate_stock_codes(args.n_stocks)

    # 2. 生成交易日期
    trading_dates = generate_trading_dates(args.n_dates)

    # 3. 生成概念关联矩阵 → ashare_stock2concept.npy（当前目录）
    print("\n[1/3] 生成概念关联矩阵...")
    concept_matrix = generate_concept_matrix(args.n_stocks, args.n_concepts, rng)
    npy_path = Path('ashare_stock2concept.npy')
    np.save(npy_path, concept_matrix)

    # 同时保存股票代码列表（供 concept_embedding 使用）
    codes_path = Path('stock_codes.txt')
    with open(codes_path, 'w') as f:
        for code in stock_codes:
            f.write(code + '\n')

    # 4. 生成因子CSV → stock_transaction_features1/
    print(f"\n[2/3] 生成 {args.n_factors} 个因子CSV...")
    factor_dir = Path('stock_transaction_features1')
    generate_factor_csvs(factor_dir, stock_codes, trading_dates, args.n_factors, rng)

    # 5. 验证
    print("\n[3/3] 验证生成的数据...")
    loaded = np.load(npy_path)
    assert loaded.dtype == np.float64
    assert loaded.shape == (args.n_stocks, args.n_concepts)

    factor_files = sorted(factor_dir.glob('*.csv'))
    sample_df = pd.read_csv(factor_files[0])
    assert list(sample_df.columns) == ['TradingDate', 'SecurityID', 'Value']
    sample_date = sample_df['TradingDate'].iloc[0]
    assert sample_date.count('.') == 2, f"日期格式错误: {sample_date}"
    avg_rows = sum(len(pd.read_csv(f)) for f in factor_files) // len(factor_files)

    # 汇总
    print(f"\n✓ 生成 ashare_stock2concept.npy: shape {loaded.shape}")
    print(f"✓ 生成 {len(factor_files)} 个因子CSV到 stock_transaction_features1/")
    print(f"  每个文件约 {avg_rows} 行, 3列: TradingDate, SecurityID, Value")
    print(f"  日期范围: {trading_dates[0]} ~ {trading_dates[-1]}")
    print(f"  股票数: {args.n_stocks}")
    print(f"✓ 模拟数据准备完毕")


if __name__ == '__main__':
    main()
