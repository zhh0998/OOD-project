#!/usr/bin/env python3
"""
MLE能量 vs Softmax 对比验证
假设：MLE能量在OOD检测上比传统Softmax更有效
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("MLE能量 vs Softmax 对比验证")
print("假设: Energy score 比 Max Softmax Probability 更有效")
print("=" * 70)

# ============================================================
# A. 创建模拟数据
# ============================================================
print("\n[A] 创建模拟OOD检测数据...")

np.random.seed(42)

def create_classification_data(n_train=1000, n_id_test=300, n_ood_test=200, n_classes=10):
    """
    创建分类数据
    - 训练集：ID数据
    - ID测试集：与训练集同分布
    - OOD测试集：不同分布
    """
    # ID数据：10个类别的文本
    id_templates = {
        0: ["book flight to {}", "reserve plane ticket {}", "fly to {}"],
        1: ["weather forecast {}", "temperature in {}", "climate {}"],
        2: ["play {} music", "listen to {} song", "music by {}"],
        3: ["set alarm {}", "reminder at {}", "wake up {}"],
        4: ["order {} food", "delivery from {}", "buy {} meal"],
        5: ["call {}", "phone {}", "dial {}"],
        6: ["send message to {}", "text {}", "sms to {}"],
        7: ["navigate to {}", "directions {}", "route to {}"],
        8: ["search for {}", "find {}", "look up {}"],
        9: ["translate {}", "convert {} language", "{} in spanish"]
    }

    ood_templates = [
        "what is quantum physics {}",
        "explain blockchain {}",
        "history of {} war",
        "recipe for {} cake",
        "how to fix {} car",
        "medical symptoms of {}",
        "stock price of {}",
        "movie review {}",
    ]

    entities = ["Paris", "Tokyo", "pizza", "jazz", "noon", "John", "home", "news", "python", "love"]

    # 生成训练数据
    train_texts = []
    train_labels = []
    for _ in range(n_train):
        c = np.random.randint(0, n_classes)
        template = np.random.choice(id_templates[c])
        text = template.format(np.random.choice(entities))
        train_texts.append(text)
        train_labels.append(c)

    # 生成ID测试数据
    id_test_texts = []
    id_test_labels = []
    for _ in range(n_id_test):
        c = np.random.randint(0, n_classes)
        template = np.random.choice(id_templates[c])
        text = template.format(np.random.choice(entities))
        id_test_texts.append(text)
        id_test_labels.append(c)

    # 生成OOD测试数据
    ood_test_texts = []
    for _ in range(n_ood_test):
        template = np.random.choice(ood_templates)
        text = template.format(np.random.choice(entities))
        ood_test_texts.append(text)

    return train_texts, train_labels, id_test_texts, id_test_labels, ood_test_texts

train_texts, train_labels, id_test_texts, id_test_labels, ood_test_texts = create_classification_data()

print(f"    训练集: {len(train_texts)} 样本, {len(set(train_labels))} 类")
print(f"    ID测试集: {len(id_test_texts)} 样本")
print(f"    OOD测试集: {len(ood_test_texts)} 样本")

# ============================================================
# B. 训练分类器
# ============================================================
print("\n[B] 训练分类器...")

# TF-IDF特征提取
vectorizer = TfidfVectorizer(max_features=500)
X_train = vectorizer.fit_transform(train_texts).toarray()
X_id_test = vectorizer.transform(id_test_texts).toarray()
X_ood_test = vectorizer.transform(ood_test_texts).toarray()

# 训练Logistic Regression（模拟神经网络的最后一层）
clf = LogisticRegression(max_iter=1000, multi_class='multinomial')
clf.fit(X_train, train_labels)

train_acc = clf.score(X_train, train_labels)
id_test_acc = clf.score(X_id_test, id_test_labels)
print(f"    训练准确率: {train_acc*100:.1f}%")
print(f"    ID测试准确率: {id_test_acc*100:.1f}%")

# ============================================================
# C. 计算OOD检测分数
# ============================================================
print("\n[C] 计算OOD检测分数...")

def compute_msp(probs):
    """Max Softmax Probability"""
    return np.max(probs, axis=1)

def compute_energy(logits, T=1.0):
    """
    Energy Score
    E(x) = -T * log(sum(exp(f_i(x)/T)))
    较低的能量表示更可能是ID
    """
    # 为了与MSP方向一致（越高越可能是ID），返回负能量
    return T * np.log(np.sum(np.exp(logits / T), axis=1))

# 获取logits和概率
# Logistic Regression的decision_function返回的就是logits
id_logits = clf.decision_function(X_id_test)
ood_logits = clf.decision_function(X_ood_test)

id_probs = clf.predict_proba(X_id_test)
ood_probs = clf.predict_proba(X_ood_test)

# 计算MSP分数
id_msp = compute_msp(id_probs)
ood_msp = compute_msp(ood_probs)

# 计算Energy分数
id_energy = compute_energy(id_logits)
ood_energy = compute_energy(ood_logits)

print(f"    ID样本 MSP: {id_msp.mean():.4f} ± {id_msp.std():.4f}")
print(f"    OOD样本 MSP: {ood_msp.mean():.4f} ± {ood_msp.std():.4f}")
print(f"    ID样本 Energy: {id_energy.mean():.4f} ± {id_energy.std():.4f}")
print(f"    OOD样本 Energy: {ood_energy.mean():.4f} ± {ood_energy.std():.4f}")

# ============================================================
# D. 计算OOD检测指标
# ============================================================
print("\n[D] 计算OOD检测指标...")

def compute_ood_metrics(id_scores, ood_scores):
    """
    计算OOD检测指标
    - AUROC: Area Under ROC Curve
    - FPR@95: False Positive Rate when TPR=95%
    """
    # 标签：ID=0, OOD=1
    # 分数：越低越可能是OOD（对于MSP和Energy）
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])

    # AUROC（分数越高越是ID，所以OOD检测用负分数）
    auroc = roc_auc_score(labels, -scores)

    # FPR@95
    fpr, tpr, thresholds = roc_curve(labels, -scores)
    # 找到TPR>=95%时的最小FPR
    idx = np.argmax(tpr >= 0.95)
    fpr_at_95 = fpr[idx]

    return auroc, fpr_at_95

# MSP指标
msp_auroc, msp_fpr95 = compute_ood_metrics(id_msp, ood_msp)
print(f"\n    MSP方法:")
print(f"      AUROC: {msp_auroc*100:.2f}%")
print(f"      FPR@95: {msp_fpr95*100:.2f}%")

# Energy指标
energy_auroc, energy_fpr95 = compute_ood_metrics(id_energy, ood_energy)
print(f"\n    Energy方法:")
print(f"      AUROC: {energy_auroc*100:.2f}%")
print(f"      FPR@95: {energy_fpr95*100:.2f}%")

# 计算提升
auroc_improvement = (energy_auroc - msp_auroc) * 100
fpr_improvement = (msp_fpr95 - energy_fpr95) * 100

print(f"\n    Energy相对MSP的提升:")
print(f"      AUROC: {auroc_improvement:+.2f}%")
print(f"      FPR@95降低: {fpr_improvement:+.2f}%")

# ============================================================
# E. 不同温度参数的Energy
# ============================================================
print("\n[E] 测试不同温度参数...")

temperatures = [0.5, 1.0, 2.0, 5.0, 10.0]
energy_results = []

for T in temperatures:
    id_e = compute_energy(id_logits, T)
    ood_e = compute_energy(ood_logits, T)
    auroc, fpr95 = compute_ood_metrics(id_e, ood_e)
    energy_results.append({
        'T': T,
        'auroc': auroc,
        'fpr95': fpr95
    })
    print(f"    T={T}: AUROC={auroc*100:.2f}%, FPR@95={fpr95*100:.2f}%")

best_T = max(energy_results, key=lambda x: x['auroc'])
print(f"\n    最佳温度: T={best_T['T']}, AUROC={best_T['auroc']*100:.2f}%")

# ============================================================
# F. 可视化
# ============================================================
print("\n[F] 生成可视化...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 子图1: MSP分布
ax1 = axes[0, 0]
ax1.hist(id_msp, bins=30, alpha=0.6, label='ID', color='blue', density=True)
ax1.hist(ood_msp, bins=30, alpha=0.6, label='OOD', color='red', density=True)
ax1.axvline(x=id_msp.mean(), color='blue', linestyle='--', linewidth=2)
ax1.axvline(x=ood_msp.mean(), color='red', linestyle='--', linewidth=2)
ax1.set_xlabel('Max Softmax Probability', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title(f'MSP Distribution\n(AUROC={msp_auroc*100:.1f}%)', fontsize=13)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 子图2: Energy分布
ax2 = axes[0, 1]
ax2.hist(id_energy, bins=30, alpha=0.6, label='ID', color='blue', density=True)
ax2.hist(ood_energy, bins=30, alpha=0.6, label='OOD', color='red', density=True)
ax2.axvline(x=id_energy.mean(), color='blue', linestyle='--', linewidth=2)
ax2.axvline(x=ood_energy.mean(), color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Energy Score', fontsize=12)
ax2.set_ylabel('Density', fontsize=12)
ax2.set_title(f'Energy Distribution\n(AUROC={energy_auroc*100:.1f}%)', fontsize=13)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 子图3: ROC曲线对比
ax3 = axes[1, 0]
# MSP ROC
labels = np.concatenate([np.zeros(len(id_msp)), np.ones(len(ood_msp))])
msp_scores = np.concatenate([id_msp, ood_msp])
energy_scores = np.concatenate([id_energy, ood_energy])

fpr_msp, tpr_msp, _ = roc_curve(labels, -msp_scores)
fpr_energy, tpr_energy, _ = roc_curve(labels, -energy_scores)

ax3.plot(fpr_msp, tpr_msp, 'b-', linewidth=2, label=f'MSP (AUROC={msp_auroc*100:.1f}%)')
ax3.plot(fpr_energy, tpr_energy, 'r-', linewidth=2, label=f'Energy (AUROC={energy_auroc*100:.1f}%)')
ax3.plot([0, 1], [0, 1], 'k--', linewidth=1)
ax3.axhline(y=0.95, color='gray', linestyle=':', linewidth=1, label='TPR=95%')
ax3.set_xlabel('False Positive Rate', fontsize=12)
ax3.set_ylabel('True Positive Rate', fontsize=12)
ax3.set_title('ROC Curve Comparison', fontsize=13)
ax3.legend(loc='lower right')
ax3.grid(True, alpha=0.3)

# 子图4: 温度参数敏感性
ax4 = axes[1, 1]
temps = [r['T'] for r in energy_results]
aurocs = [r['auroc']*100 for r in energy_results]
ax4.plot(temps, aurocs, 'bo-', linewidth=2, markersize=8)
ax4.axhline(y=msp_auroc*100, color='red', linestyle='--', linewidth=2, label=f'MSP Baseline ({msp_auroc*100:.1f}%)')
ax4.set_xlabel('Temperature T', fontsize=12)
ax4.set_ylabel('AUROC (%)', fontsize=12)
ax4.set_title('Energy Score: Temperature Sensitivity', fontsize=13)
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xscale('log')

plt.tight_layout()
plt.savefig('mle_energy_validation.png', dpi=150, bbox_inches='tight')
print("    图表已保存: mle_energy_validation.png")

# ============================================================
# G. 结论
# ============================================================
print("\n" + "=" * 70)
print("MLE能量 vs Softmax 验证结果")
print("=" * 70)

print("\n性能对比:")
print("-" * 50)
print(f"{'方法':<15} {'AUROC':>12} {'FPR@95':>12}")
print("-" * 50)
print(f"{'MSP':<15} {msp_auroc*100:>11.2f}% {msp_fpr95*100:>11.2f}%")
print(f"{'Energy':<15} {energy_auroc*100:>11.2f}% {energy_fpr95*100:>11.2f}%")
print(f"{'Energy (best T)':<15} {best_T['auroc']*100:>11.2f}% {best_T['fpr95']*100:>11.2f}%")
print("-" * 50)
print(f"{'提升':<15} {auroc_improvement:>+11.2f}% {fpr_improvement:>+11.2f}%")
print("-" * 50)

print("\n" + "-" * 70)

# 判断标准
threshold = 2.0  # AUROC提升>2%

if auroc_improvement > threshold:
    print("\n✅ MLE能量理论优势验证通过！")
    print()
    print(f"   • Energy AUROC比MSP提升: {auroc_improvement:+.2f}% > {threshold}%")
    print(f"   • FPR@95降低: {fpr_improvement:+.2f}%")
    print()
    print("   结论: Energy score确实优于传统MSP")
    print("   建议: RW3可以采用Energy-based方法")
    result = "PASS"
elif auroc_improvement > 0:
    print("\n⚠️ MLE能量有一定优势，但不显著")
    print()
    print(f"   • Energy AUROC提升: {auroc_improvement:+.2f}% < {threshold}%")
    print()
    print("   结论: Energy略优于MSP，但提升有限")
    print("   建议: 可以尝试，但需要结合其他方法")
    result = "PARTIAL"
else:
    print("\n❌ MLE能量理论优势验证失败")
    print()
    print(f"   • Energy AUROC提升: {auroc_improvement:+.2f}% (无提升或下降)")
    print()
    print("   结论: 在当前设置下Energy不优于MSP")
    print("   建议: 需要重新审视Energy的适用条件")
    result = "FAIL"

print("\n" + "=" * 70)

# 保存结果
with open('mle_energy_result.txt', 'w') as f:
    f.write(f"MLE Energy vs MSP Result: {result}\n")
    f.write(f"MSP AUROC: {msp_auroc:.4f}\n")
    f.write(f"Energy AUROC: {energy_auroc:.4f}\n")
    f.write(f"AUROC Improvement: {auroc_improvement:.4f}%\n")
    f.write(f"Best Temperature: {best_T['T']}\n")
