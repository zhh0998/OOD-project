#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all.py — 一键运行脚本

按顺序执行所有模块，每步检查输出是否成功。

使用:
  python run_all.py                    # 跑全部（需要预先准备数据）
  python run_all.py --mock             # 先生成模拟数据再跑
  python run_all.py --skip-lstm        # 跳过 LSTM 模块（只跑 PCA 快速版）
  python run_all.py --mock --skip-lstm # 模拟数据 + 只跑 PCA
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_step(name, cmd, check_outputs=None):
    """
    运行一个步骤
    name: 步骤名称
    cmd: 命令列表
    check_outputs: 需要检查的输出文件列表
    返回: True 成功, False 失败
    """
    print(f"\n{'=' * 60}")
    print(f"  ▶ {name}")
    print(f"  命令: {' '.join(cmd)}")
    print(f"{'=' * 60}")

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).parent),
            capture_output=False,
            text=True,
            timeout=3600  # 最长1小时
        )
        elapsed = time.time() - start

        if result.returncode != 0:
            print(f"\n  ✗ {name} 失败 (返回码: {result.returncode})")
            print(f"  耗时: {elapsed:.1f}s")
            return False

        # 检查输出文件
        if check_outputs:
            for f in check_outputs:
                fp = Path(__file__).parent / f
                if not fp.exists():
                    print(f"  ✗ 输出文件缺失: {f}")
                    return False
                size = fp.stat().st_size
                print(f"  ✓ 输出: {f} ({size:,} bytes)")

        print(f"\n  ✓ {name} 完成 (耗时: {elapsed:.1f}s)")
        return True

    except subprocess.TimeoutExpired:
        print(f"\n  ✗ {name} 超时 (>3600s)")
        return False
    except Exception as e:
        print(f"\n  ✗ {name} 异常: {e}")
        return False


def preview_csv(filepath, n=5):
    """预览CSV前几行"""
    import pandas as pd
    fp = Path(__file__).parent / filepath
    if fp.exists():
        df = pd.read_csv(fp, nrows=n)
        print(f"\n  预览 {filepath} (前{n}行):")
        print(df.to_string(index=False))
    else:
        print(f"  [跳过预览] 文件不存在: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='一键运行所有模块')
    parser.add_argument('--mock', action='store_true',
                        help='先生成模拟数据再运行')
    parser.add_argument('--skip-lstm', action='store_true',
                        help='跳过 LSTM 模块（只跑 PCA 快速版）')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='输出目录 (默认: outputs)')
    parser.add_argument('--dim', type=int, default=8,
                        help='静态嵌入维度 (默认: 8)')
    parser.add_argument('--pca_dim', type=int, default=4,
                        help='PCA嵌入维度 (默认: 4)')
    parser.add_argument('--lstm_dim', type=int, default=16,
                        help='LSTM嵌入维度 (默认: 16)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='训练轮数 (默认: 30)')
    parser.add_argument('--window', type=int, default=20,
                        help='时间窗口 (默认: 20)')
    parser.add_argument('--n_stocks', type=int, default=100,
                        help='模拟股票数 (默认: 100)')
    parser.add_argument('--n_dates', type=int, default=60,
                        help='模拟交易日数 (默认: 60)')
    parser.add_argument('--n_factors', type=int, default=10,
                        help='模拟因子数 (默认: 10)')
    parser.add_argument('--n_concepts', type=int, default=500,
                        help='模拟概念数 (默认: 500)')
    args = parser.parse_args()

    python = sys.executable
    output_dir = args.output_dir
    # 新的数据路径约定
    npy_path = 'ashare_stock2concept.npy'
    codes_path = 'stock_codes.txt'
    factor_dir = 'stock_transaction_features1'
    results = {}

    print("\n" + "#" * 60)
    print("#  股票多层次嵌入降维系统 — 一键运行")
    print("#" * 60)

    # ========================================
    # 步骤0: 生成模拟数据（可选）
    # ========================================
    if args.mock:
        ok = run_step(
            "步骤0: 生成模拟数据",
            [python, 'generate_mock_data.py',
             '--n_stocks', str(args.n_stocks),
             '--n_dates', str(args.n_dates),
             '--n_factors', str(args.n_factors),
             '--n_concepts', str(args.n_concepts)],
            check_outputs=[npy_path, codes_path]
        )
        results['生成模拟数据'] = ok
        if not ok:
            print("\n[终止] 模拟数据生成失败，无法继续")
            sys.exit(1)

    # 检查数据是否存在
    npy_full = Path(__file__).parent / npy_path
    factor_full = Path(__file__).parent / factor_dir
    if not npy_full.exists():
        print(f"\n[错误] 概念矩阵文件不存在: {npy_path}")
        print(f"  请先准备数据，或使用 --mock 生成模拟数据")
        sys.exit(1)
    if not factor_full.exists():
        print(f"\n[错误] 因子文件夹不存在: {factor_dir}")
        print(f"  请先准备数据，或使用 --mock 生成模拟数据")
        sys.exit(1)

    # ========================================
    # 步骤1: 静态概念嵌入
    # ========================================
    ok = run_step(
        "步骤1: 静态概念嵌入 (SVD)",
        [python, 'concept_embedding.py',
         '--input', npy_path,
         '--codes', codes_path,
         '--method', 'svd',
         '--dim', str(args.dim),
         '--output', './'],
        check_outputs=[
            f'embeddings_{args.dim}d.npy',
            f'embeddings_{args.dim}d_with_codes.csv',
        ]
    )
    results['静态概念嵌入'] = ok

    # ========================================
    # 步骤2: 动态时序嵌入 (LSTM)
    # ========================================
    if not args.skip_lstm:
        ok = run_step(
            "步骤2: 动态时序嵌入 (LSTM)",
            [python, 'dynamic_temporal_embedding.py',
             '--input', factor_dir,
             '--output', output_dir,
             '--window', str(args.window),
             '--embed_dim', str(args.lstm_dim),
             '--epochs', str(args.epochs)],
            check_outputs=[f'{output_dir}/embeddings_lstm.csv']
        )
        results['动态时序嵌入(LSTM)'] = ok
    else:
        print("\n[跳过] 步骤2: 动态时序嵌入 (LSTM) — 使用 --skip-lstm")
        results['动态时序嵌入(LSTM)'] = '跳过'

    # ========================================
    # 步骤3: 快速PCA嵌入
    # ========================================
    ok = run_step(
        "步骤3: 快速PCA嵌入",
        [python, 'quick_4emb_fastest.py',
         '--input', factor_dir,
         '--output', output_dir,
         '--embed_dim', str(args.pca_dim)],
        check_outputs=[f'{output_dir}/embeddings_{args.pca_dim}d.csv']
    )
    results['快速PCA嵌入'] = ok

    # ========================================
    # 步骤4: 动静融合嵌入
    # ========================================
    static_csv = f'embeddings_{args.dim}d_with_codes.csv'
    if results.get('静态概念嵌入'):
        ok = run_step(
            "步骤4: 动静融合嵌入",
            [python, 'hybrid_embedding.py',
             '--static', static_csv,
             '--factors', factor_dir,
             '--output', output_dir,
             '--window', str(args.window),
             '--embed_dim', str(args.lstm_dim),
             '--epochs', str(args.epochs)],
            check_outputs=[f'{output_dir}/hybrid_embeddings.csv']
        )
        results['动静融合嵌入'] = ok
    else:
        print("\n[跳过] 步骤4: 动静融合嵌入 — 依赖静态嵌入失败")
        results['动静融合嵌入'] = '跳过(依赖失败)'

    # ========================================
    # 最终报告
    # ========================================
    print("\n\n" + "#" * 60)
    print("#  运行结果汇总")
    print("#" * 60)

    all_ok = True
    for step, status in results.items():
        if status is True:
            icon = "✓"
        elif status is False:
            icon = "✗"
            all_ok = False
        else:
            icon = "–"
        print(f"  {icon} {step}: {status}")

    # 显示输出文件
    output_path = Path(__file__).parent / output_dir
    cwd = Path(__file__).parent
    print(f"\n  输出文件:")
    all_output_files = []
    # 当前目录的输出
    for pattern in [f'embeddings_{args.dim}d.npy', f'embeddings_{args.dim}d_with_codes.csv']:
        fp = cwd / pattern
        if fp.exists():
            all_output_files.append(fp)
    # outputs 目录的输出
    if output_path.exists():
        all_output_files.extend(sorted(output_path.glob('*.csv')))
        all_output_files.extend(sorted(output_path.glob('*.npy')))

    for f in all_output_files:
        size = f.stat().st_size
        rel = f.relative_to(cwd)
        print(f"    {str(rel):40s} {size:>10,} bytes")

    # 预览所有输出CSV
    print("\n" + "-" * 60)
    print("  输出文件预览")
    print("-" * 60)

    for csv_rel in [f'embeddings_{args.dim}d_with_codes.csv',
                    f'{output_dir}/embeddings_{args.pca_dim}d.csv',
                    f'{output_dir}/embeddings_lstm.csv',
                    f'{output_dir}/hybrid_embeddings.csv']:
        fp = cwd / csv_rel
        if fp.exists():
            preview_csv(csv_rel)

    print(f"\n{'#' * 60}")
    if all_ok:
        print("#  ✓ 所有步骤完成！")
    else:
        print("#  ✗ 部分步骤失败，请检查上面的输出")
    print(f"{'#' * 60}")

    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
