#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dynamic_temporal_embedding.py — 模块2: 动态时序嵌入 (LSTM + 时间注意力)

使用 LSTM + 时间注意力机制，对每只股票每天生成动态嵌入。
训练目标: 自监督（最大化 embedding 方差，使不同股票的嵌入具有区分度）

输入: 因子CSV文件夹（每个CSV含 TradingDate, SecurityID, Value 三列）
输出: outputs/embeddings_lstm.csv (TradingDate, SecurityID, emb_0, ..., emb_{dim-1})
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
    ensure_output_dir, DEFAULT_EMBED_DIM, DEFAULT_WINDOW,
    DEFAULT_HIDDEN_SIZE, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LR
)


# ============================================================
# 模型定义: LSTM + 时间注意力
# ============================================================
class TemporalAttention(nn.Module):
    """时间注意力层：对 LSTM 各时间步输出加权"""

    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        """
        lstm_output: (batch, seq_len, hidden_size)
        返回: (batch, hidden_size) — 加权后的表示
        """
        scores = self.attn(lstm_output).squeeze(-1)  # (batch, seq_len)
        weights = torch.softmax(scores, dim=-1)  # (batch, seq_len)
        weighted = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)  # (batch, hidden_size)
        return weighted


class LSTMEncoder(nn.Module):
    """LSTM 时序编码器 + 时间注意力 → 嵌入向量"""

    def __init__(self, input_dim, hidden_size, embed_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = TemporalAttention(hidden_size)
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, embed_dim),
            nn.Tanh()
        )

    def forward(self, x):
        """
        x: (batch, window, n_factors)
        返回: (batch, embed_dim)
        """
        lstm_out, _ = self.lstm(x)  # (batch, window, hidden_size)
        attended = self.attention(lstm_out)  # (batch, hidden_size)
        embedding = self.projection(attended)  # (batch, embed_dim)
        return embedding


# ============================================================
# 自监督损失: 最大化 embedding 方差
# ============================================================
def variance_loss(embeddings):
    """
    自监督损失: 鼓励 embedding 在各维度上有较大方差
    即鼓励不同样本的 embedding 有区分度
    """
    # 各维度方差
    var = embeddings.var(dim=0)  # (embed_dim,)
    # 负方差作为损失（最大化方差 = 最小化负方差）
    loss = -var.mean()
    # 加入正则项: 鼓励各维度去相关
    if embeddings.shape[0] > 1:
        centered = embeddings - embeddings.mean(dim=0)
        cov = (centered.T @ centered) / (embeddings.shape[0] - 1)
        # 去对角线，最小化协方差
        off_diag = cov - torch.diag(cov.diag())
        loss += 0.01 * off_diag.pow(2).mean()
    return loss


# ============================================================
# 训练过程
# ============================================================
def train_model(model, X, device, epochs, batch_size, lr):
    """训练 LSTM 编码器"""
    model.to(device)
    model.train()

    # 创建 DataLoader
    X_tensor = torch.FloatTensor(X).to(device)
    dataset = TensorDataset(X_tensor)

    # 若数据量太大，分块到 CPU 再按 batch 送 GPU
    use_cpu_data = (X.nbytes > 1e9)  # 超过 1GB
    if use_cpu_data:
        X_tensor = torch.FloatTensor(X)
        dataset = TensorDataset(X_tensor)
        print(f"[训练] 数据量较大({X.nbytes / 1e9:.1f}GB)，使用分批传输模式")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"[训练] 开始训练: {epochs} 轮, batch_size={batch_size}, 样本数={len(X)}")

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for (batch_x,) in loader:
            if use_cpu_data:
                batch_x = batch_x.to(device)
            embeddings = model(batch_x)
            loss = variance_loss(embeddings)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:>4d}/{epochs}: loss={avg_loss:.6f}, lr={scheduler.get_last_lr()[0]:.6f}")

    return model


def generate_embeddings(model, X, labels, device, batch_size):
    """用训练好的模型生成所有样本的嵌入"""
    model.eval()
    all_embeddings = []

    X_tensor = torch.FloatTensor(X)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for (batch_x,) in tqdm(loader, desc='生成嵌入', leave=False):
            batch_x = batch_x.to(device)
            emb = model(batch_x)
            all_embeddings.append(emb.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    return embeddings


def main():
    parser = argparse.ArgumentParser(description='模块2: 动态时序嵌入 (LSTM)')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='因子CSV文件夹路径')
    parser.add_argument('--output', '-o', type=str, default='outputs',
                        help='输出目录 (默认: outputs)')
    parser.add_argument('--window', type=int, default=DEFAULT_WINDOW,
                        help=f'时间窗口天数 (默认: {DEFAULT_WINDOW})')
    parser.add_argument('--embed_dim', type=int, default=DEFAULT_EMBED_DIM,
                        help=f'嵌入维度 (默认: {DEFAULT_EMBED_DIM})')
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
    print("  模块2: 动态时序嵌入 (LSTM + 时间注意力)")
    print("=" * 60)
    print(f"  输入文件夹: {args.input}")
    print(f"  输出目录:   {args.output}")
    print(f"  时间窗口:   {args.window}")
    print(f"  嵌入维度:   {args.embed_dim}")
    print(f"  隐藏层大小: {args.hidden_size}")
    print(f"  训练轮数:   {args.epochs}")
    print("=" * 60)

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 检测设备
    device = get_device()

    # 加载因子数据
    print("\n[1/4] 加载因子数据...")
    data_dict, factor_names, all_stocks, all_dates = load_factors_fast(args.input)
    n_factors = len(factor_names)
    print(f"  因子数: {n_factors}")

    # 构建张量
    print("\n[2/4] 构建滑动窗口张量...")
    X, labels = build_factor_tensor(data_dict, all_stocks, all_dates, factor_names, args.window)
    print(f"  样本数: {X.shape[0]}, 窗口: {X.shape[1]}, 因子: {X.shape[2]}")

    # 释放原始数据
    del data_dict
    gc.collect()

    # 创建模型
    print("\n[3/4] 训练 LSTM 编码器...")
    model = LSTMEncoder(
        input_dim=n_factors,
        hidden_size=args.hidden_size,
        embed_dim=args.embed_dim,
        num_layers=2,
        dropout=0.1
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {param_count:,}")

    model = train_model(model, X, device, args.epochs, args.batch_size, args.lr)

    # 生成嵌入
    print("\n[4/4] 生成嵌入向量...")
    embeddings = generate_embeddings(model, X, labels, device, args.batch_size)
    print(f"  嵌入形状: {embeddings.shape}")

    # 构建输出 DataFrame
    dates_str = [d.strftime('%Y.%m.%d') if hasattr(d, 'strftime') else str(d) for d, _ in labels]
    stocks = [s for _, s in labels]

    emb_cols = [f"emb_{i}" for i in range(args.embed_dim)]
    df = pd.DataFrame(embeddings, columns=emb_cols)
    df.insert(0, 'SecurityID', stocks)
    df.insert(0, 'TradingDate', dates_str)

    # 保存
    ensure_output_dir(args.output)
    output_path = Path(args.output) / 'embeddings_lstm.csv'
    df.to_csv(output_path, index=False)
    print(f"\n[保存] {output_path}")

    # 显示前几行
    print(f"\n[预览] 前5行:")
    print(df.head().to_string(index=False))

    # 统计
    n_unique_dates = df['TradingDate'].nunique()
    n_unique_stocks = df['SecurityID'].nunique()
    print(f"\n[统计] {n_unique_dates} 个交易日, {n_unique_stocks} 只股票, {len(df)} 条记录")

    print(f"\n{'=' * 60}")
    print(f"  ✓ 动态时序嵌入生成完成")
    print(f"  输出: {output_path}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
