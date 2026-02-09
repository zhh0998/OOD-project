#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hybrid_embedding.py — 模块3: 动静融合嵌入

将静态概念嵌入与动态因子序列融合，生成统一的混合嵌入。
方法: LSTM编码动态序列 + concat静态嵌入 → 自编码器融合降维

输入:
  - 静态嵌入CSV (来自 concept_embedding.py 的输出)
  - 因子CSV文件夹 (与模块2相同的输入)
输出: outputs/hybrid_embeddings.csv
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config import (
    get_device, load_factors_fast, build_factor_tensor,
    normalize_stock_code, ensure_output_dir,
    DEFAULT_EMBED_DIM, DEFAULT_WINDOW, DEFAULT_HIDDEN_SIZE,
    DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LR
)


# ============================================================
# 模型定义
# ============================================================
class HybridEncoder(nn.Module):
    """
    混合嵌入编码器:
    1. LSTM 编码动态时序 → 动态向量
    2. concat 静态嵌入
    3. 自编码器融合 → 最终嵌入
    """

    def __init__(self, n_factors, static_dim, hidden_size, embed_dim, num_layers=2, dropout=0.1):
        super().__init__()
        # 动态序列编码
        self.lstm = nn.LSTM(
            input_size=n_factors,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        # 时间注意力
        self.attn = nn.Linear(hidden_size, 1)

        # 融合层的输入维度 = LSTM输出 + 静态嵌入维度
        fusion_input_dim = hidden_size + static_dim

        # 自编码器融合
        self.encoder = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_input_dim // 2, embed_dim),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, fusion_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_input_dim // 2, fusion_input_dim),
        )

    def encode(self, x_dynamic, x_static):
        """编码: 动态序列 + 静态嵌入 → 融合嵌入"""
        # LSTM 编码动态序列
        lstm_out, _ = self.lstm(x_dynamic)  # (batch, window, hidden)
        # 时间注意力
        scores = self.attn(lstm_out).squeeze(-1)  # (batch, window)
        weights = torch.softmax(scores, dim=-1)
        dynamic_repr = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)  # (batch, hidden)

        # 拼接动态 + 静态
        fused = torch.cat([dynamic_repr, x_static], dim=-1)  # (batch, hidden + static_dim)

        # 自编码器压缩
        embedding = self.encoder(fused)
        return embedding, fused

    def decode(self, embedding):
        """解码"""
        return self.decoder(embedding)

    def forward(self, x_dynamic, x_static):
        embedding, fused = self.encode(x_dynamic, x_static)
        reconstructed = self.decode(embedding)
        return embedding, fused, reconstructed


def load_static_embeddings(csv_path):
    """
    加载静态嵌入CSV，自动检测嵌入维度和股票代码格式
    返回: {stock_code: np.array} 字典, 嵌入维度
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"[错误] 静态嵌入文件不存在: {path}\n"
            f"  请先运行 concept_embedding.py 生成静态嵌入"
        )

    df = pd.read_csv(path)
    print(f"[静态嵌入] 文件: {path.name}, shape: {df.shape}")

    # 找到股票代码列（第一列通常是代码）
    code_col = df.columns[0]
    emb_cols = [c for c in df.columns if c.startswith('emb_')]
    if not emb_cols:
        # 如果没有 emb_ 前缀的列，取除代码列外的所有数值列
        emb_cols = [c for c in df.columns[1:] if pd.api.types.is_numeric_dtype(df[c])]

    if not emb_cols:
        raise ValueError(f"[错误] 无法识别嵌入列。列名: {list(df.columns)}")

    static_dim = len(emb_cols)
    print(f"[静态嵌入] 代码列: {code_col}, 嵌入维度: {static_dim}")

    # 统一股票代码格式
    static_dict = {}
    for _, row in df.iterrows():
        code = normalize_stock_code(row[code_col])
        emb = row[emb_cols].values.astype(np.float32)
        static_dict[code] = emb

    print(f"[静态嵌入] 加载 {len(static_dict)} 只股票的静态嵌入")
    return static_dict, static_dim


def train_hybrid_model(model, X_dynamic, X_static, device, epochs, batch_size, lr):
    """训练混合嵌入模型"""
    model.to(device)
    model.train()

    dataset = TensorDataset(
        torch.FloatTensor(X_dynamic),
        torch.FloatTensor(X_static)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"[训练] 开始训练: {epochs} 轮, batch_size={batch_size}, 样本数={len(X_dynamic)}")

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for batch_dyn, batch_sta in loader:
            batch_dyn = batch_dyn.to(device)
            batch_sta = batch_sta.to(device)

            embedding, fused, reconstructed = model(batch_dyn, batch_sta)

            # 重构损失
            recon_loss = nn.functional.mse_loss(reconstructed, fused)
            # 方差损失（鼓励嵌入有区分度）
            var_loss = -embedding.var(dim=0).mean()
            # 总损失
            loss = recon_loss + 0.1 * var_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:>4d}/{epochs}: loss={avg_loss:.6f}")

    return model


def main():
    parser = argparse.ArgumentParser(description='模块3: 动静融合嵌入')
    parser.add_argument('--static', '-s', type=str, default='embeddings_8d_with_codes.csv',
                        help='静态嵌入CSV路径 (默认: embeddings_8d_with_codes.csv)')
    parser.add_argument('--factors', '-f', type=str, default='stock_transaction_features1',
                        help='因子CSV文件夹路径 (默认: stock_transaction_features1)')
    parser.add_argument('--output', '-o', type=str, default='outputs',
                        help='输出目录 (默认: outputs)')
    parser.add_argument('--window', type=int, default=DEFAULT_WINDOW,
                        help=f'时间窗口天数 (默认: {DEFAULT_WINDOW})')
    parser.add_argument('--embed_dim', type=int, default=DEFAULT_EMBED_DIM,
                        help=f'最终嵌入维度 (默认: {DEFAULT_EMBED_DIM})')
    parser.add_argument('--hidden_size', type=int, default=DEFAULT_HIDDEN_SIZE,
                        help=f'LSTM隐藏层大小 (默认: {DEFAULT_HIDDEN_SIZE})')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help=f'训练轮数 (默认: {DEFAULT_EPOCHS})')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'批大小 (默认: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--lr', type=float, default=DEFAULT_LR,
                        help=f'学习率 (默认: {DEFAULT_LR})')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    args = parser.parse_args()

    print("=" * 60)
    print("  模块3: 动静融合嵌入")
    print("=" * 60)
    print(f"  静态嵌入: {args.static}")
    print(f"  因子文件: {args.factors}")
    print(f"  嵌入维度: {args.embed_dim}")
    print(f"  时间窗口: {args.window}")
    print("=" * 60)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()

    # 加载静态嵌入
    print("\n[1/5] 加载静态嵌入...")
    static_dict, static_dim = load_static_embeddings(args.static)

    # 加载因子数据
    print("\n[2/5] 加载因子数据...")
    data_dict, factor_names, all_stocks, all_dates = load_factors_fast(args.factors)
    n_factors = len(factor_names)

    # 找到静态嵌入和因子数据的共同股票
    factor_stocks = set(all_stocks)
    static_stocks = set(static_dict.keys())
    common_stocks = sorted(factor_stocks & static_stocks)

    print(f"\n[股票匹配]")
    print(f"  因子数据股票:  {len(factor_stocks)}")
    print(f"  静态嵌入股票:  {len(static_stocks)}")
    print(f"  共同股票:      {len(common_stocks)}")

    if len(common_stocks) == 0:
        # 尝试显示两边的代码格式，帮助调试
        f_sample = list(factor_stocks)[:5]
        s_sample = list(static_stocks)[:5]
        raise ValueError(
            f"[错误] 静态嵌入和因子数据没有共同股票！\n"
            f"  因子数据代码示例: {f_sample}\n"
            f"  静态嵌入代码示例: {s_sample}\n"
            f"  请检查股票代码格式是否一致，或使用 convert_data.py 统一格式"
        )

    if len(common_stocks) < len(factor_stocks) * 0.5:
        print(f"  [警告] 共同股票不到因子数据的50%，请检查股票代码格式")

    # 构建张量（只用共同股票）
    print("\n[3/5] 构建滑动窗口张量...")
    # 过滤 data_dict 只保留共同股票
    common_set = set(common_stocks)
    filtered_dict = {}
    for date, df in data_dict.items():
        filtered = df[df.index.isin(common_set)]
        if len(filtered) > 0:
            filtered_dict[date] = filtered
    data_dict = filtered_dict

    X_dynamic, labels = build_factor_tensor(data_dict, common_stocks, all_dates, factor_names, args.window)
    print(f"  动态张量: {X_dynamic.shape}")

    del data_dict
    gc.collect()

    # 构建对应的静态嵌入矩阵
    print("\n[4/5] 构建静态嵌入矩阵...")
    X_static = np.zeros((len(labels), static_dim), dtype=np.float32)
    for i, (_, stock) in enumerate(labels):
        if stock in static_dict:
            X_static[i] = static_dict[stock]
    print(f"  静态矩阵: {X_static.shape}")

    # 训练
    print("\n[5/5] 训练融合模型...")
    model = HybridEncoder(
        n_factors=n_factors,
        static_dim=static_dim,
        hidden_size=args.hidden_size,
        embed_dim=args.embed_dim,
        num_layers=2,
        dropout=0.1
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {param_count:,}")
    print(f"  融合输入维度: {args.hidden_size} (动态) + {static_dim} (静态) = {args.hidden_size + static_dim}")

    model = train_hybrid_model(model, X_dynamic, X_static, device, args.epochs, args.batch_size, args.lr)

    # 生成嵌入
    print("\n[生成] 生成混合嵌入向量...")
    model.eval()
    all_embeddings = []
    dataset = TensorDataset(torch.FloatTensor(X_dynamic), torch.FloatTensor(X_static))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    with torch.no_grad():
        for batch_dyn, batch_sta in tqdm(loader, desc='生成嵌入', leave=False):
            batch_dyn = batch_dyn.to(device)
            batch_sta = batch_sta.to(device)
            emb, _, _ = model(batch_dyn, batch_sta)
            all_embeddings.append(emb.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"  嵌入形状: {embeddings.shape}")

    # 构建输出
    dates_str = [d.strftime('%Y.%m.%d') if hasattr(d, 'strftime') else str(d) for d, _ in labels]
    stocks = [s for _, s in labels]

    emb_cols = [f"emb_{i}" for i in range(args.embed_dim)]
    df = pd.DataFrame(embeddings, columns=emb_cols)
    df.insert(0, 'SecurityID', stocks)
    df.insert(0, 'TradingDate', dates_str)

    ensure_output_dir(args.output)
    output_path = Path(args.output) / 'hybrid_embeddings.csv'
    df.to_csv(output_path, index=False)

    print(f"\n[预览] 前5行:")
    print(df.head().to_string(index=False))

    n_unique_dates = df['TradingDate'].nunique()
    n_unique_stocks = df['SecurityID'].nunique()
    print(f"\n[统计] {n_unique_dates} 个交易日, {n_unique_stocks} 只股票, {len(df)} 条记录")

    print(f"\n{'=' * 60}")
    print(f"  ✓ 动静融合嵌入生成完成")
    print(f"  输出: {output_path}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
