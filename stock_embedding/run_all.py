#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all.py — 一键运行脚本

按顺序执行所有模块，每步检查输出是否成功。

使用:
  python run_all.py                    # 跑全部（需要预先准备数据）
  python run_all.py --mock             # 先生成模拟数据再跑
  python run_all.py --skip-lstm        # 跳过 LSTM，只跑 PCA 版
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
    parser.add_argument('--data_dir', type=str, default='mock_data',
                        help='数据目录 (默认: mock_data)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='输出目录 (默认: outputs)')
    parser.add_argument('--dim', type=int, default=16,
                        help='嵌入维度 (默认: 16)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='训练轮数 (默认: 30)')
    parser.add_argument('--window', type=int, default=20,
                        help='时间窗口 (默认: 20)')
    parser.add_argument('--n_stocks', type=int, default=200,
                        help='模拟股票数 (默认: 200)')
    parser.add_argument('--n_dates', type=int, default=60,
                        help='模拟交易日数 (默认: 60)')
    parser.add_argument('--n_factors', type=int, default=20,
                        help='模拟因子数 (默认: 20)')
    parser.add_argument('--n_concepts', type=int, default=500,
                        help='模拟概念数 (默认: 500)')
    args = parser.parse_args()

    python = sys.executable
    data_dir = args.data_dir
    output_dir = args.output_dir
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
             '--output_dir', data_dir,
             '--n_stocks', str(args.n_stocks),
             '--n_dates', str(args.n_dates),
             '--n_factors', str(args.n_factors),
             '--n_concepts', str(args.n_concepts)],
            check_outputs=[
                f'{data_dir}/concept_matrix.npy',
                f'{data_dir}/stock_codes.txt',
            ]
        )
        results['生成模拟数据'] = ok
        if not ok:
            print("\n[终止] 模拟数据生成失败，无法继续")
            sys.exit(1)

    # 检查数据是否存在
    data_path = Path(__file__).parent / data_dir
    if not data_path.exists():
        print(f"\n[错误] 数据目录不存在: {data_dir}")
        print(f"  请先准备数据，或使用 --mock 生成模拟数据")
        sys.exit(1)

    # ========================================
    # 步骤1: 静态概念嵌入
    # ========================================
    ok = run_step(
        "步骤1: 静态概念嵌入 (SVD)",
        [python, 'concept_embedding.py',
         '--input', f'{data_dir}/concept_matrix.npy',
         '--codes', f'{data_dir}/stock_codes.txt',
         '--method', 'svd',
         '--dim', str(args.dim),
         '--output', output_dir],
        check_outputs=[
            f'{output_dir}/embeddings_{args.dim}d.npy',
            f'{output_dir}/embeddings_{args.dim}d_with_codes.csv',
        ]
    )
    results['静态概念嵌入'] = ok

    # ========================================
    # 步骤2: 动态时序嵌入
    # ========================================
    if not args.skip_lstm:
        ok = run_step(
            "步骤2: 动态时序嵌入 (LSTM)",
            [python, 'dynamic_temporal_embedding.py',
             '--input', f'{data_dir}/factors',
             '--output', output_dir,
             '--window', str(args.window),
             '--embed_dim', str(args.dim),
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
         '--input', f'{data_dir}/factors',
         '--output', output_dir,
         '--embed_dim', str(args.dim)],
        check_outputs=[f'{output_dir}/embeddings_pca.csv']
    )
    results['快速PCA嵌入'] = ok

    # ========================================
    # 步骤4: 动静融合嵌入
    # ========================================
    static_csv = f'{output_dir}/embeddings_{args.dim}d_with_codes.csv'
    if results.get('静态概念嵌入'):
        ok = run_step(
            "步骤4: 动静融合嵌入",
            [python, 'hybrid_embedding.py',
             '--static', static_csv,
             '--factors', f'{data_dir}/factors',
             '--output', output_dir,
             '--window', str(args.window),
             '--embed_dim', str(args.dim),
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
    if output_path.exists():
        print(f"\n  输出目录: {output_path}")
        csv_files = sorted(output_path.glob('*.csv'))
        npy_files = sorted(output_path.glob('*.npy'))
        for f in csv_files + npy_files:
            size = f.stat().st_size
            print(f"    {f.name:40s} {size:>10,} bytes")

    # 预览所有输出CSV
    print("\n" + "-" * 60)
    print("  输出文件预览")
    print("-" * 60)

    import pandas as pd
    for csv_name in ['embeddings_lstm.csv', 'embeddings_pca.csv', 'hybrid_embeddings.csv',
                     f'embeddings_{args.dim}d_with_codes.csv']:
        csv_path = output_path / csv_name
        if csv_path.exists():
            preview_csv(f'{output_dir}/{csv_name}')

    print(f"\n{'#' * 60}")
    if all_ok:
        print("#  ✓ 所有步骤完成！")
    else:
        print("#  ✗ 部分步骤失败，请检查上面的输出")
    print(f"{'#' * 60}")

    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
