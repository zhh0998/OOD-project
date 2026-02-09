#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quick_4emb_fastest.py — 快速替代版: PCA动态嵌入

用PCA替代LSTM生成动态嵌入，不需要GPU。
特性:
  - 多进程并行读取因子CSV
  - groupby矩阵构建
  - 缓存机制（避免重复加载）
  - PCA(n_components=min(target_dim, n_factors)) — 自适应维度
"""

import argparse
import gc
import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from config import (
    load_factors_fast, ensure_output_dir,
    DEFAULT_EMBED_DIM, DEFAULT_WINDOW
)


# ============================================================
# 缓存机制
# ============================================================
def get_cache_key(folder):
    """基于文件夹内容生成缓存键"""
    files = sorted(Path(folder).glob('*.csv'))
    content = '|'.join(f"{f.name}:{f.stat().st_size}:{f.stat().st_mtime_ns}" for f in files)
    return hashlib.md5(content.encode()).hexdigest()[:12]


def load_cached(cache_path):
    """尝试从缓存加载"""
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            print(f"[缓存] 从缓存加载: {cache_path.name}")
            return data
        except Exception:
            pass
    return None


def save_cache(cache_path, data):
    """保存到缓存"""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[缓存] 已保存缓存: {cache_path.name}")
    except Exception as e:
        print(f"[缓存] 保存缓存失败: {e}")


# ============================================================
# PCA 动态嵌入
# ============================================================
def pca_embedding_per_date(data_dict, factor_names, all_dates, embed_dim):
    """
    对每个交易日的因子矩阵做 PCA 降维

    data_dict: {date: DataFrame(index=stock, columns=factors)}
    返回: DataFrame(TradingDate, SecurityID, emb_0, ..., emb_{dim-1})
    """
    # 自适应维度: 不能超过因子数
    actual_dim = min(embed_dim, len(factor_names))
    if actual_dim < embed_dim:
        print(f"[PCA] 目标维度 {embed_dim} > 因子数 {len(factor_names)}，自适应调整为 {actual_dim}")

    sorted_dates = sorted(all_dates)
    results = []

    # 全局 Scaler 和 PCA（用所有数据拟合）
    print("[PCA] 收集全局数据拟合 PCA...")
    all_matrices = []
    for date in sorted_dates:
        if date in data_dict:
            mat = data_dict[date].reindex(columns=factor_names).values
            mat = np.nan_to_num(mat, nan=0.0)
            all_matrices.append(mat)

    if len(all_matrices) == 0:
        raise ValueError("[错误] 无可用数据")

    global_data = np.vstack(all_matrices)
    scaler = StandardScaler()
    scaler.fit(global_data)

    pca = PCA(n_components=actual_dim, random_state=42)
    pca.fit(scaler.transform(global_data))
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"[PCA] 维度: {actual_dim}, 累计方差解释率: {explained_var:.4f}")

    del global_data, all_matrices
    gc.collect()

    # 对每天的数据做 PCA 变换
    for date in tqdm(sorted_dates, desc='PCA嵌入'):
        if date not in data_dict:
            continue
        df = data_dict[date]
        mat = df.reindex(columns=factor_names).values
        mat = np.nan_to_num(mat, nan=0.0)
        mat_scaled = scaler.transform(mat)
        emb = pca.transform(mat_scaled)

        date_str = date.strftime('%Y.%m.%d') if hasattr(date, 'strftime') else str(date)
        for i, stock in enumerate(df.index):
            row = [date_str, stock] + emb[i].tolist()
            results.append(row)

    emb_cols = [f"emb_{i}" for i in range(actual_dim)]
    result_df = pd.DataFrame(results, columns=['TradingDate', 'SecurityID'] + emb_cols)
    return result_df, actual_dim


def main():
    parser = argparse.ArgumentParser(description='快速替代版: PCA动态嵌入（无需GPU）')
    parser.add_argument('--input', '-i', type=str, default='stock_transaction_features1',
                        help='因子CSV文件夹路径 (默认: stock_transaction_features1)')
    parser.add_argument('--output', '-o', type=str, default='outputs',
                        help='输出目录 (默认: outputs)')
    parser.add_argument('--embed_dim', type=int, default=4,
                        help='嵌入维度 (默认: 4)')
    parser.add_argument('--no_cache', action='store_true',
                        help='禁用缓存')
    args = parser.parse_args()

    print("=" * 60)
    print("  快速替代版: PCA 动态嵌入")
    print("=" * 60)
    print(f"  输入文件夹: {args.input}")
    print(f"  输出目录:   {args.output}")
    print(f"  嵌入维度:   {args.embed_dim}")
    print("=" * 60)

    # 尝试从缓存加载
    cache_dir = Path(args.output) / '.cache'
    cache_key = get_cache_key(args.input)
    cache_path = cache_dir / f"factors_{cache_key}.pkl"

    cached = None if args.no_cache else load_cached(cache_path)

    if cached is not None:
        data_dict, factor_names, all_stocks, all_dates = cached
        print(f"[数据] {len(all_stocks)} 只股票, {len(all_dates)} 个交易日, {len(factor_names)} 个因子")
    else:
        print("\n[1/3] 加载因子数据...")
        data_dict, factor_names, all_stocks, all_dates = load_factors_fast(args.input)
        if not args.no_cache:
            save_cache(cache_path, (data_dict, factor_names, all_stocks, all_dates))

    # PCA 嵌入
    print("\n[2/3] 计算 PCA 嵌入...")
    result_df, actual_dim = pca_embedding_per_date(data_dict, factor_names, all_dates, args.embed_dim)

    del data_dict
    gc.collect()

    # 保存
    print("\n[3/3] 保存结果...")
    ensure_output_dir(args.output)
    output_path = Path(args.output) / f'embeddings_{actual_dim}d.csv'
    result_df.to_csv(output_path, index=False)

    print(f"\n[预览] 前5行:")
    print(result_df.head().to_string(index=False))

    n_unique_dates = result_df['TradingDate'].nunique()
    n_unique_stocks = result_df['SecurityID'].nunique()
    print(f"\n[统计] {n_unique_dates} 个交易日, {n_unique_stocks} 只股票, {len(result_df)} 条记录")

    print(f"\n{'=' * 60}")
    print(f"  ✓ PCA 动态嵌入生成完成")
    print(f"  输出: {output_path}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
