#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
concept_embedding.py — 模块1: 静态概念嵌入

从股票×概念关联矩阵生成低维嵌入向量。
支持方法:
  - svd: TruncatedSVD（适合稀疏数据，默认）
  - node2vec: GNN Node2Vec（需要安装 node2vec 包）

输入: .npy 文件 (shape: n_stocks × n_concepts, float64, 稀疏)
输出: embeddings_{dim}d.npy + embeddings_{dim}d_with_codes.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from config import ensure_output_dir


def load_concept_matrix(input_path):
    """加载概念关联矩阵"""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(
            f"[错误] 概念矩阵文件不存在: {path}\n"
            f"  请检查路径是否正确，或先运行 generate_mock_data.py 生成模拟数据"
        )
    matrix = np.load(path)
    print(f"[概念矩阵] shape={matrix.shape}, dtype={matrix.dtype}")
    n_nonzero = np.count_nonzero(matrix)
    sparsity = 1 - n_nonzero / matrix.size
    print(f"[概念矩阵] 非零元素: {n_nonzero}, 稀疏度: {sparsity:.4f}")
    return matrix


def load_stock_codes(codes_path, n_stocks):
    """
    加载股票代码列表
    如果文件不存在，生成默认编号
    """
    path = Path(codes_path) if codes_path else None
    if path and path.exists():
        with open(path) as f:
            codes = [line.strip() for line in f if line.strip()]
        if len(codes) != n_stocks:
            print(f"[警告] 股票代码数({len(codes)})与矩阵行数({n_stocks})不一致，使用默认编号")
            codes = [f"STOCK_{i:04d}" for i in range(n_stocks)]
    else:
        print(f"[提示] 未提供股票代码文件，使用默认编号")
        codes = [f"STOCK_{i:04d}" for i in range(n_stocks)]
    return codes


def embed_svd(matrix, dim, random_state=42):
    """TruncatedSVD 降维"""
    # 确保维度不超过特征数和样本数的最小值
    max_dim = min(matrix.shape) - 1
    if dim > max_dim:
        print(f"[警告] 目标维度 {dim} > 最大可用维度 {max_dim}，自动调整为 {max_dim}")
        dim = max_dim

    svd = TruncatedSVD(n_components=dim, random_state=random_state)
    embeddings = svd.fit_transform(matrix)

    explained_var = svd.explained_variance_ratio_.sum()
    print(f"[SVD] 维度: {dim}, 累计方差解释率: {explained_var:.4f}")

    # L2 归一化
    embeddings = normalize(embeddings, norm='l2')
    return embeddings


def embed_node2vec(matrix, dim, walk_length=30, num_walks=200, p=1, q=1):
    """Node2Vec 嵌入（将概念关联矩阵视为二部图）"""
    try:
        import networkx as nx
        from node2vec import Node2Vec
    except ImportError:
        raise ImportError(
            "[错误] Node2Vec 方法需要安装额外包:\n"
            "  pip install node2vec networkx\n"
            "  如果不需要 GNN 方法，请使用 --method svd"
        )

    n_stocks, n_concepts = matrix.shape
    print(f"[Node2Vec] 构建二部图: {n_stocks} 股票节点 + {n_concepts} 概念节点")

    # 构建二部图
    G = nx.Graph()
    stock_nodes = [f"S_{i}" for i in range(n_stocks)]
    concept_nodes = [f"C_{j}" for j in range(n_concepts)]
    G.add_nodes_from(stock_nodes)
    G.add_nodes_from(concept_nodes)

    # 添加边（非零元素）
    rows, cols = np.nonzero(matrix)
    for r, c in zip(rows, cols):
        G.add_edge(f"S_{r}", f"C_{c}", weight=matrix[r, c])

    print(f"[Node2Vec] 图: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

    # 训练 Node2Vec
    model = Node2Vec(G, dimensions=dim, walk_length=walk_length,
                     num_walks=num_walks, p=p, q=q, workers=4, quiet=True)
    fitted = model.fit(window=10, min_count=1, batch_words=4)

    # 提取股票节点的嵌入
    embeddings = np.zeros((n_stocks, dim), dtype=np.float32)
    for i in range(n_stocks):
        node = f"S_{i}"
        if node in fitted.wv:
            embeddings[i] = fitted.wv[node]

    # L2 归一化
    embeddings = normalize(embeddings, norm='l2')
    return embeddings


def main():
    parser = argparse.ArgumentParser(description='模块1: 静态概念嵌入')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='概念关联矩阵文件路径 (.npy)')
    parser.add_argument('--codes', type=str, default=None,
                        help='股票代码文件路径（每行一个代码）')
    parser.add_argument('--method', type=str, default='svd',
                        choices=['svd', 'node2vec'],
                        help='嵌入方法 (默认: svd)')
    parser.add_argument('--dim', type=int, default=16,
                        help='嵌入维度 (默认: 16)')
    parser.add_argument('--output', '-o', type=str, default='outputs',
                        help='输出目录 (默认: outputs)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    args = parser.parse_args()

    print("=" * 60)
    print("  模块1: 静态概念嵌入")
    print("=" * 60)
    print(f"  方法:   {args.method}")
    print(f"  维度:   {args.dim}")
    print(f"  输入:   {args.input}")
    print(f"  输出:   {args.output}")
    print("=" * 60)

    # 加载数据
    matrix = load_concept_matrix(args.input)
    n_stocks = matrix.shape[0]
    stock_codes = load_stock_codes(args.codes, n_stocks)

    # 生成嵌入
    print(f"\n[计算] 使用 {args.method.upper()} 方法生成 {args.dim} 维嵌入...")
    if args.method == 'svd':
        embeddings = embed_svd(matrix, args.dim, random_state=args.seed)
    elif args.method == 'node2vec':
        embeddings = embed_node2vec(matrix, args.dim)
    else:
        print(f"[错误] 不支持的方法: {args.method}")
        sys.exit(1)

    print(f"[结果] 嵌入形状: {embeddings.shape}")

    # 保存结果
    ensure_output_dir(args.output)

    # 保存 npy
    npy_path = Path(args.output) / f"embeddings_{args.dim}d.npy"
    np.save(npy_path, embeddings)
    print(f"[保存] {npy_path}")

    # 保存 CSV（带股票代码）
    csv_path = Path(args.output) / f"embeddings_{args.dim}d_with_codes.csv"
    emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    df = pd.DataFrame(embeddings, columns=emb_cols)
    df.insert(0, 'SecurityID', stock_codes)
    df.to_csv(csv_path, index=False)
    print(f"[保存] {csv_path}")

    # 显示前几行
    print(f"\n[预览] 前5行:")
    print(df.head().to_string(index=False))

    print(f"\n{'=' * 60}")
    print(f"  ✓ 静态概念嵌入生成完成")
    print(f"  输出文件:")
    print(f"    {npy_path}")
    print(f"    {csv_path}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
