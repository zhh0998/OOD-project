#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_data.py — 客户数据格式转换工具

将客户的原始数据转换为系统要求的标准格式：
- 日期: YYYY.MM.DD（点号分隔）
- 股票代码: XXXXXX.SZ / XXXXXX.SH
- 列名: TradingDate, SecurityID, Value

支持的输入格式:
- 日期: YYYY-MM-DD, YYYYMMDD, YYYY/MM/DD, YYYY.MM.DD
- 股票代码: 000001, 000001.SZ, SZ000001, 000001.XSHE
- 列名: 自动检测（支持中英文）
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import (
    detect_date_col, detect_stock_col, detect_value_col,
    normalize_stock_codes_series, parse_dates_series,
    detect_code_format
)


def preview_file(filepath):
    """预览文件格式，显示检测到的列和格式"""
    df = pd.read_csv(filepath, nrows=5)
    print(f"\n  文件: {Path(filepath).name}")
    print(f"  列名: {list(df.columns)}")
    print(f"  行数（预览）: {len(df)}")

    try:
        date_col = detect_date_col(df)
        stock_col = detect_stock_col(df)
        value_col = detect_value_col(df, date_col, stock_col)

        print(f"  检测到的列映射:")
        print(f"    日期列:     {date_col} → TradingDate")
        print(f"    股票代码列: {stock_col} → SecurityID")
        print(f"    值列:       {value_col} → Value")

        # 检测日期格式
        sample_date = str(df[date_col].iloc[0])
        print(f"    日期示例:   {sample_date}")

        # 检测股票代码格式
        code_fmt = detect_code_format(df[stock_col])
        print(f"    代码格式:   {code_fmt}")
        print(f"    代码示例:   {df[stock_col].iloc[0]}")

        return date_col, stock_col, value_col
    except ValueError as e:
        print(f"  [错误] {e}")
        return None, None, None


def convert_file(filepath, output_path, date_col, stock_col, value_col):
    """转换单个CSV文件"""
    df = pd.read_csv(filepath, low_memory=False)

    # 提取需要的列
    result = pd.DataFrame()

    # 转换日期
    dates = parse_dates_series(df[date_col])
    result['TradingDate'] = dates.dt.strftime('%Y.%m.%d')

    # 转换股票代码
    result['SecurityID'] = normalize_stock_codes_series(df[stock_col])

    # 转换值
    result['Value'] = pd.to_numeric(df[value_col], errors='coerce')

    # 去除无效行
    n_before = len(result)
    result = result.dropna()
    n_dropped = n_before - len(result)
    if n_dropped > 0:
        print(f"  [提示] 丢弃 {n_dropped} 行无效数据")

    # 保存
    result.to_csv(output_path, index=False)
    return len(result)


def main():
    parser = argparse.ArgumentParser(description='客户数据格式转换工具')
    parser.add_argument('--input', '-i', required=True,
                        help='输入文件或文件夹路径')
    parser.add_argument('--output', '-o', required=True,
                        help='输出文件或文件夹路径')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='跳过确认提示，直接转换')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"[错误] 输入路径不存在: {input_path}")
        sys.exit(1)

    # 收集要处理的文件
    if input_path.is_file():
        files = [input_path]
        output_path.parent.mkdir(parents=True, exist_ok=True)
    elif input_path.is_dir():
        files = sorted(input_path.glob('*.csv'))
        if len(files) == 0:
            print(f"[错误] 文件夹中没有CSV文件: {input_path}")
            sys.exit(1)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        print(f"[错误] 输入路径既不是文件也不是文件夹: {input_path}")
        sys.exit(1)

    print("=" * 60)
    print("  客户数据格式转换工具")
    print("=" * 60)
    print(f"  输入: {input_path}")
    print(f"  输出: {output_path}")
    print(f"  文件数量: {len(files)}")

    # 预览第一个文件
    print("\n--- 格式预览 ---")
    date_col, stock_col, value_col = preview_file(files[0])
    if date_col is None:
        sys.exit(1)

    if len(files) > 1:
        print(f"\n  (其余 {len(files) - 1} 个文件将使用相同的列映射)")

    # 确认
    if not args.yes:
        resp = input("\n是否开始转换? [y/N] ").strip().lower()
        if resp not in ('y', 'yes'):
            print("已取消。")
            sys.exit(0)

    # 开始转换
    print("\n--- 开始转换 ---")
    total_rows = 0
    for f in tqdm(files, desc='转换文件'):
        if input_path.is_file():
            out = output_path
        else:
            out = output_path / f.name
        try:
            n = convert_file(f, out, date_col, stock_col, value_col)
            total_rows += n
        except Exception as e:
            print(f"\n  [错误] 转换 {f.name} 失败: {e}")
            continue

    print(f"\n--- 转换完成 ---")
    print(f"  转换文件数: {len(files)}")
    print(f"  总行数:     {total_rows}")
    print(f"  输出目录:   {output_path.resolve()}")

    # 验证第一个输出文件
    if input_path.is_dir():
        sample_out = sorted(output_path.glob('*.csv'))[0]
    else:
        sample_out = output_path
    df = pd.read_csv(sample_out, nrows=5)
    print(f"\n  输出样例 ({sample_out.name}):")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
