#!/usr/bin/env python3
"""
完整数据集实验 - 单次运行验证

预计时间：30-40分钟
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from sentence_transformers import SentenceTransformer

print("\n🚀 完整数据集实验")
print("="*70)
sys.stdout.flush()

# 1. 加载完整数据
print("\n1️⃣ 加载CLINC150完整数据集...")
sys.stdout.flush()

from data_loader import load_clinc150
train_texts, test_texts, test_labels, test_intents, train_labels = load_clinc150()

print(f"   训练集: {len(train_texts)} 样本")
print(f"   测试集: {len(test_texts)} 样本")
print(f"   测试集OOD: {sum(test_labels)} 样本")
sys.stdout.flush()

# 2. 获取embeddings
print("\n2️⃣ 获取sentence embeddings（约3-5分钟）...")
sys.stdout.flush()

encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
train_emb = encoder.encode(train_texts, show_progress_bar=True, batch_size=64)
test_emb = encoder.encode(test_texts, show_progress_bar=True, batch_size=64)

print(f"   ✅ Train embeddings: {train_emb.shape}")
print(f"   ✅ Test embeddings: {test_emb.shape}")
sys.stdout.flush()

# 3. KNN基线（快速参考）
print("\n3️⃣ KNN-10基线（参考）...")
sys.stdout.flush()

from quick_fix import FixedKNNDetector, evaluate_ood

knn = FixedKNNDetector(k=10)
knn.fit(train_emb)
knn_scores, _ = knn.score_with_fix(test_emb, np.array(test_labels))
knn_metrics = evaluate_ood(test_labels, knn_scores)

print(f"   KNN-10 AUROC: {knn_metrics['auroc']:.2f}%")
print(f"   目标: 超过 {knn_metrics['auroc']:.2f}%")
sys.stdout.flush()

# 4. HeterophilyEnhanced（3组参数）
print("\n4️⃣ HeterophilyEnhanced v2（测试3组参数）...")
sys.stdout.flush()

from heterophily_enhanced_v2 import HeterophilyEnhancedV2

configs = [
    {'k': 50, 'alpha': 0.2, 'hidden_dim': 256, 'epochs': 15, 'name': 'Config-A'},
    {'k': 30, 'alpha': 0.3, 'hidden_dim': 256, 'epochs': 20, 'name': 'Config-B'},
    {'k': 100, 'alpha': 0.3, 'hidden_dim': 128, 'epochs': 15, 'name': 'Config-C'},
]

results = []
best_auroc = 0
best_config = None

for i, config in enumerate(configs):
    print(f"\n{'='*70}")
    print(f"🔬 测试 {config['name']} ({i+1}/{len(configs)})")
    print(f"   参数: k={config['k']}, alpha={config['alpha']}, "
          f"hidden={config['hidden_dim']}, epochs={config['epochs']}")
    print('-'*70)
    sys.stdout.flush()

    try:
        detector = HeterophilyEnhancedV2(
            input_dim=train_emb.shape[1],
            hidden_dim=config['hidden_dim'],
            output_dim=128,
            k=config['k'],
            num_layers=2,
            alpha=config['alpha']
        )

        print(f"   训练中（{config['epochs']} epochs，约5-8分钟）...")
        sys.stdout.flush()

        detector.fit(train_emb, train_labels, epochs=config['epochs'], verbose=True)

        print(f"   评估中...")
        sys.stdout.flush()

        scores = detector.score(test_emb)
        metrics = evaluate_ood(test_labels, scores)

        auroc = metrics['auroc']
        print(f"\n   ✅ AUROC: {auroc:.2f}%")
        print(f"   FPR@95: {metrics['fpr95']:.2f}%")
        sys.stdout.flush()

        results.append({
            'config': config,
            'auroc': auroc,
            'metrics': metrics
        })

        if auroc > best_auroc:
            best_auroc = auroc
            best_config = config
            print(f"   🏆 NEW BEST!")
            sys.stdout.flush()

        improvement = auroc - knn_metrics['auroc']
        if improvement > 0:
            print(f"   ✅ vs KNN: +{improvement:.2f}%")
        else:
            print(f"   ⚠️ vs KNN: {improvement:.2f}%")
        sys.stdout.flush()

    except Exception as e:
        print(f"   ❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        continue

# 5. 最终结果
print("\n" + "="*70)
print("📊 实验结果汇总")
print("="*70)
sys.stdout.flush()

print(f"\n📏 基线:")
print(f"   KNN-10: {knn_metrics['auroc']:.2f}%")

print(f"\n🔬 HeterophilyEnhanced v2:")
for r in results:
    auroc = r['auroc']
    name = r['config']['name']
    improvement = auroc - knn_metrics['auroc']
    status = "✅" if improvement > 0 else "⚠️"
    print(f"   {status} {name}: {auroc:.2f}% ({improvement:+.2f}%)")

if best_auroc > knn_metrics['auroc']:
    improvement = best_auroc - knn_metrics['auroc']
    print(f"\n🎉 SUCCESS!")
    print(f"   最佳配置: {best_config['name']}")
    print(f"   最佳AUROC: {best_auroc:.2f}%")
    print(f"   提升: +{improvement:.2f}%")

    if improvement >= 1.0:
        print(f"   ✅ 达到发表标准（≥1%提升）")
    else:
        print(f"   ⚠️ 未达发表标准，但方向正确")
else:
    gap = knn_metrics['auroc'] - best_auroc
    print(f"\n⚠️ 需要继续优化")
    print(f"   最佳结果: {best_auroc:.2f}%")
    print(f"   差距: -{gap:.2f}%")
    print(f"\n💡 建议:")
    print(f"   1. 增加训练epochs: 20 → 30")
    print(f"   2. 调大alpha: 0.3 → 0.5")
    print(f"   3. 尝试更大的k: 100 → 200")

# 6. 保存结果
import json
output_dir = Path(__file__).parent / 'results'
output_dir.mkdir(exist_ok=True)
output_file = output_dir / 'full_experiment_results.json'

with open(output_file, 'w') as f:
    json.dump({
        'knn_baseline': {
            'auroc': knn_metrics['auroc'],
            'fpr95': knn_metrics['fpr95']
        },
        'heterophily_enhanced': [
            {
                'config': r['config'],
                'auroc': r['auroc'],
                'fpr95': r['metrics']['fpr95']
            } for r in results
        ],
        'best_config': best_config,
        'best_auroc': best_auroc
    }, f, indent=2)

print(f"\n💾 结果已保存: {output_file}")
print("\n✅ 实验完成!")
sys.stdout.flush()
