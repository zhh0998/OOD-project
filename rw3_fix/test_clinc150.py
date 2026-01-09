#!/usr/bin/env python3
"""
CLINC150快速验证脚本 - 验证Bug修复效果

预期结果: AUROC >= 90% (修复Bug后)
对比: v2版本 72.86% -> v3版本 90%+

Author: RW3 OOD Detection Project
"""

import sys
import json
import numpy as np
from pathlib import Path

# 添加当前目录
sys.path.insert(0, str(Path(__file__).parent))

from quick_fix import FixedKNNDetector, evaluate_ood
from data_loader import load_clinc150

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score


def main():
    print("="*60)
    print("CLINC150 OOD检测验证 - Bug修复版")
    print("="*60)

    # 1. 加载数据
    print("\n[Step 1] 加载CLINC150数据集...")
    train_texts, test_texts, test_labels, test_intents, _ = load_clinc150()
    test_labels = np.array(test_labels)

    print(f"  训练样本: {len(train_texts)}")
    print(f"  测试样本: {len(test_texts)}")
    print(f"  OOD样本: {test_labels.sum()} ({test_labels.mean()*100:.1f}%)")

    # 2. 加载模型
    print("\n[Step 2] 加载RoBERTa模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  设备: {device}")

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModel.from_pretrained("roberta-base")
    model.to(device)
    model.eval()

    # 3. 提取embeddings
    print("\n[Step 3] 提取embeddings...")

    def get_embeddings(texts, batch_size=32):
        embeddings = []
        n_batches = (len(texts) + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                inputs = tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=128, return_tensors='pt'
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                outputs = model(**inputs)
                batch_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_emb)

                if (i // batch_size + 1) % 20 == 0:
                    print(f"    进度: {i // batch_size + 1}/{n_batches}")

        return np.vstack(embeddings)

    print("  提取训练集embeddings...")
    train_emb = get_embeddings(train_texts)
    print(f"    Shape: {train_emb.shape}")

    print("  提取测试集embeddings...")
    test_emb = get_embeddings(test_texts)
    print(f"    Shape: {test_emb.shape}")

    # 4. 运行修复版k-NN检测器
    print("\n[Step 4] 运行修复版k-NN检测器...")
    print("  (包含3个Bug修复: 归一化、k-th距离、方向修正)")

    detector = FixedKNNDetector(k=50, verbose=True)
    detector.fit(train_emb)
    scores, auroc = detector.score_with_fix(test_emb, test_labels)

    # 5. 评估
    print("\n[Step 5] 评估结果...")
    metrics = evaluate_ood(test_labels, scores, auto_fix=False, verbose=False)

    print("\n" + "="*60)
    print("CLINC150 验证结果")
    print("="*60)
    print(f"  AUROC: {metrics['auroc']:.4f} ({metrics['auroc']*100:.2f}%)")
    print(f"  AUPR:  {metrics['aupr']:.4f}")
    print(f"  FPR@95%TPR: {metrics['fpr95']:.4f}")
    print("="*60)

    # 与v2对比
    v2_auroc = 0.7286
    improvement = metrics['auroc'] - v2_auroc

    print("\n对比分析:")
    print(f"  v2 (有Bug): {v2_auroc*100:.2f}%")
    print(f"  v3 (修复后): {metrics['auroc']*100:.2f}%")
    print(f"  改善: {improvement*100:+.2f}%")

    # 验证目标
    target = 0.90
    if metrics['auroc'] >= target:
        print(f"\n  [SUCCESS] 达到目标 >= {target*100:.0f}%!")
    else:
        print(f"\n  [WARNING] 未达目标 {target*100:.0f}%, 当前 {metrics['auroc']*100:.2f}%")

    # 保存结果
    results = {
        'dataset': 'clinc150',
        'v2_auroc': v2_auroc,
        'v3_auroc': metrics['auroc'],
        'improvement': improvement,
        'metrics': metrics,
        'n_train': len(train_texts),
        'n_test': len(test_texts),
        'n_ood': int(test_labels.sum())
    }

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "clinc150_verification.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存: {output_dir / 'clinc150_verification.json'}")

    return metrics['auroc']


if __name__ == "__main__":
    main()
