#!/usr/bin/env python3
"""
快速测试 - 验证HeterophilyEnhanced v2是否工作

预计运行时间：5-10分钟
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("\n🚀 快速测试 HeterophilyEnhanced v2")
print("="*70)
sys.stdout.flush()

# Step 1: 导入检查
print("\n1️⃣ 检查依赖...")
sys.stdout.flush()

try:
    import torch
    print(f"   ✅ PyTorch: {torch.__version__}")
except ImportError as e:
    print(f"   ❌ PyTorch: {e}")
    sys.exit(1)
sys.stdout.flush()

try:
    from torch_geometric.nn import GATv2Conv
    print(f"   ✅ PyTorch Geometric: GATv2Conv可用")
except ImportError as e:
    print(f"   ❌ PyTorch Geometric: {e}")
    sys.exit(1)
sys.stdout.flush()

try:
    from sentence_transformers import SentenceTransformer
    print(f"   ✅ Sentence Transformers: 可用")
except ImportError as e:
    print(f"   ❌ Sentence Transformers: {e}")
    sys.exit(1)
sys.stdout.flush()

# Step 2: 加载数据（小样本）
print("\n2️⃣ 加载测试数据...")
sys.stdout.flush()
from data_loader import load_clinc150

train_texts, test_texts, test_labels, test_intents, train_labels = load_clinc150()

# 只用前1000个训练样本
train_texts = train_texts[:1000]
train_labels = train_labels[:1000]

# 测试样本：取前250 ID + 后250 OOD（确保有OOD样本）
import numpy as np
test_arr = np.array(test_labels)
id_idx = np.where(test_arr == 0)[0][:250]
ood_idx = np.where(test_arr == 1)[0][:250]
keep_idx = np.concatenate([id_idx, ood_idx])
test_texts = [test_texts[i] for i in keep_idx]
test_labels = [test_labels[i] for i in keep_idx]

print(f"   训练集: {len(train_texts)} 样本")
print(f"   测试集: {len(test_texts)} 样本")
sys.stdout.flush()

# Step 3: 获取embeddings
print("\n3️⃣ 获取embeddings...")
sys.stdout.flush()
encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
train_emb = encoder.encode(train_texts, show_progress_bar=False)
test_emb = encoder.encode(test_texts, show_progress_bar=False)
print(f"   ✅ Embedding shape: {train_emb.shape}")
sys.stdout.flush()

# Step 4: 测试HeterophilyEnhanced v2
print("\n4️⃣ 测试HeterophilyEnhanced v2...")
sys.stdout.flush()

try:
    from heterophily_enhanced_v2 import HeterophilyEnhancedV2

    detector = HeterophilyEnhancedV2(
        input_dim=train_emb.shape[1],
        hidden_dim=128,
        output_dim=64,
        k=30,
        num_layers=2,
        alpha=0.3
    )

    print("   训练中（5 epochs）...")
    sys.stdout.flush()
    detector.fit(train_emb, train_labels, epochs=5, verbose=True)

    print("   评估中...")
    sys.stdout.flush()
    scores = detector.score(test_emb)

    from quick_fix import evaluate_ood
    metrics = evaluate_ood(test_labels, scores)

    print(f"\n   ✅ AUROC: {metrics['auroc']:.2f}%")
    print(f"   FPR@95: {metrics['fpr95']:.2f}%")
    sys.stdout.flush()

    print("\n🎉 HeterophilyEnhanced v2 工作正常!")
    sys.stdout.flush()

except Exception as e:
    print(f"\n   ❌ 失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: 对比KNN基线
print("\n5️⃣ 对比KNN-10基线...")
sys.stdout.flush()
from quick_fix import FixedKNNDetector

knn = FixedKNNDetector(k=10)
knn.fit(train_emb)
knn_scores, _ = knn.score_with_fix(test_emb, test_labels)
knn_metrics = evaluate_ood(test_labels, knn_scores)

print(f"   KNN-10 AUROC: {knn_metrics['auroc']:.2f}%")
sys.stdout.flush()

improvement = metrics['auroc'] - knn_metrics['auroc']
print(f"\n📊 改进: {improvement:+.2f}%")
sys.stdout.flush()

if improvement > 0:
    print("   ✅ HeterophilyEnhanced超过KNN!")
else:
    print("   ⚠️ 需要进一步优化")

print("\n" + "="*70)
print("✅ 快速测试完成!")
sys.stdout.flush()
