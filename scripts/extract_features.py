#!/usr/bin/env python3
"""
从微调RoBERTa模型提取L2归一化特征

特征用于下游OOD检测任务（k-NN + 异配性）

使用示例:
    # 从微调模型提取特征
    python scripts/extract_features.py \
        --model_path finetuned_models/clinc150_native_oos_seed42_final \
        --dataset clinc150 \
        --protocol native_oos \
        --output features/clinc150_native_oos_seed42.npz

    # 使用预训练模型（不微调）
    python scripts/extract_features.py \
        --model_path roberta-base \
        --dataset clinc150 \
        --protocol native_oos \
        --output features/clinc150_native_oos_pretrained.npz
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import RobertaModel, RobertaTokenizer


def extract_and_normalize(model, tokenizer, texts, batch_size=32, max_length=128, device='cuda'):
    """
    提取并L2归一化特征

    Args:
        model: RoBERTa模型
        tokenizer: tokenizer
        texts: 文本列表
        batch_size: 批次大小
        max_length: 最大序列长度
        device: 设备

    Returns:
        features: L2归一化的特征矩阵 [n_samples, hidden_dim]
    """
    model.to(device)
    model.eval()

    all_features = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting"):
            batch_texts = texts[i:i+batch_size]

            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)

            # 使用pooler_output ([CLS] token经过线性层和tanh)
            features = outputs.pooler_output  # [batch, 768]

            # L2归一化
            features = F.normalize(features, p=2, dim=1)

            all_features.append(features.cpu().numpy())

    features = np.vstack(all_features)

    # 验证L2范数
    norms = np.linalg.norm(features, axis=1)
    print(f"  L2范数验证: mean={norms.mean():.6f}, std={norms.std():.8f}")
    assert np.allclose(norms, 1.0, atol=1e-5), "L2归一化失败！"

    return features


def main():
    parser = argparse.ArgumentParser(description='特征提取脚本')
    parser.add_argument('--model_path', type=str, required=True,
                       help='微调模型路径或预训练模型名称')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['clinc150', 'banking77', 'rostd'],
                       help='数据集名称')
    parser.add_argument('--protocol', type=str, required=True,
                       choices=['native_oos', 'class_holdout'],
                       help='实验协议')
    parser.add_argument('--kir', type=float, default=None,
                       help='Known Intent Ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--output', type=str, required=True,
                       help='输出文件路径 (.npz)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--max_length', type=int, default=128,
                       help='最大序列长度')
    parser.add_argument('--data_root', type=str, default='data',
                       help='数据根目录')

    args = parser.parse_args()

    print("="*80)
    print("特征提取")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")

    # 加载数据
    print("\n加载数据...")
    from src.datasets.unified_loader import load_ood_dataset

    load_kwargs = {
        'dataset': args.dataset,
        'protocol': args.protocol,
        'seed': args.seed,
        'data_root': args.data_root
    }

    if args.protocol == 'class_holdout' and args.dataset == 'clinc150':
        if args.kir is not None:
            load_kwargs['known_intent_ratio'] = args.kir

    data = load_ood_dataset(**load_kwargs)

    print(f"  训练样本: {len(data['train_texts'])}")
    print(f"  测试样本: {len(data['test_texts'])}")

    # 加载模型
    print(f"\n加载模型: {args.model_path}")
    model = RobertaModel.from_pretrained(args.model_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.model_path)

    # 提取训练特征
    print("\n提取训练特征...")
    train_features = extract_and_normalize(
        model, tokenizer,
        data['train_texts'],
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device
    )
    print(f"  训练特征形状: {train_features.shape}")

    # 提取测试特征
    print("\n提取测试特征...")
    test_features = extract_and_normalize(
        model, tokenizer,
        data['test_texts'],
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device
    )
    print(f"  测试特征形状: {test_features.shape}")

    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        args.output,
        train_features=train_features,
        test_features=test_features,
        train_labels=np.array(data['train_labels']),
        test_labels=np.array(data['test_labels']),
        metadata=data['metadata']
    )

    print(f"\n✅ 特征已保存: {args.output}")
    print(f"  训练特征: {train_features.shape}")
    print(f"  测试特征: {test_features.shape}")
    print("="*80)


if __name__ == "__main__":
    main()
