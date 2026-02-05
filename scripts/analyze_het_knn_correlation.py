#!/usr/bin/env python3
"""
Phase 0.5: 分析 Heterophily 与 KNN Distance 的相关性
判断残差（residual）方法是否值得追求

Decision Criteria:
  PURSUE_RESIDUAL            — 残差相关性低 + 独立判别力 → Phase 1
  PURSUE_RESIDUAL_CAUTIOUS   — 残差略有提升 → 谨慎尝试 Phase 1
  HIGH_CORRELATION_TRY_MULTISCALE — 高度相关 → Phase 2
  RECONSIDER_APPROACH        — 都不行 → Phase 3

Author: RW3 OOD Detection Project
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.heterophily_detector import HeterophilyEnhancedFixed


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_features(dataset_name: str) -> dict:
    """加载预提取的 finetuned RoBERTa embeddings"""
    feature_map = {
        "clinc150": PROJECT_ROOT / "features" / "clinc150_native_oos_finetuned_seed42.npz",
        "banking77": PROJECT_ROOT / "features" / "banking77_finetuned_seed42.npz",
    }
    npz_path = feature_map[dataset_name]
    data = np.load(str(npz_path), allow_pickle=True)
    return {
        "train_features": data["train_features"],
        "test_features": data["test_features"],
        "train_labels": data["train_labels"],
        "test_labels": data["test_labels"],
    }


# ---------------------------------------------------------------------------
# 分数分离：纯 KNN vs 纯 Heterophily
# ---------------------------------------------------------------------------

def compute_scores(detector: HeterophilyEnhancedFixed, test_emb: np.ndarray):
    """
    用 alpha 切换的方式分别获取 KNN 和 Heterophily 分数。
    score() 内部的 min-max 归一化只依赖当前 test_emb，所以两次调用
    对同一 test_emb 得到的 knn_scores 归一化是一致的。
    """
    original_alpha = detector.alpha

    detector.alpha = 0.0
    knn_scores = detector.score(test_emb)

    detector.alpha = 1.0
    het_scores = detector.score(test_emb)

    detector.alpha = original_alpha
    return knn_scores, het_scores


# ---------------------------------------------------------------------------
# 核心分析
# ---------------------------------------------------------------------------

def analyze_correlation(dataset_name: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  Analyzing: {dataset_name}")
    print(f"{'='*60}\n")

    # 1. 加载数据
    data = load_features(dataset_name)
    train_emb = data["train_features"]
    train_labels = data["train_labels"]
    test_emb = data["test_features"]
    test_labels = data["test_labels"]
    input_dim = train_emb.shape[1]

    # 2. 初始化 detector（verbose=False 避免刷屏）
    detector = HeterophilyEnhancedFixed(
        input_dim=input_dim, k=10, alpha=0.3, verbose=False
    )
    detector.fit(train_emb, train_labels)

    # 3. 分别获取 KNN / Het 分数
    knn_scores, het_scores = compute_scores(detector, test_emb)

    id_mask = test_labels == 0
    ood_mask = test_labels == 1

    results = {
        "dataset": dataset_name,
        "embedding_dim": input_dim,
        "n_id_test": int(id_mask.sum()),
        "n_ood_test": int(ood_mask.sum()),
        "correlation": {},
        "auroc": {},
        "residual": {},
        "decision": None,
        "decision_reason": None,
    }

    # ------------------------------------------------------------------
    # 4. 全局相关性
    # ------------------------------------------------------------------
    pearson_r, pearson_p = pearsonr(knn_scores, het_scores)
    spearman_r, spearman_p = spearmanr(knn_scores, het_scores)

    results["correlation"]["pearson_r"] = round(float(pearson_r), 4)
    results["correlation"]["pearson_p"] = float(pearson_p)
    results["correlation"]["spearman_r"] = round(float(spearman_r), 4)
    results["correlation"]["spearman_p"] = float(spearman_p)

    print(f"Overall Correlation:")
    print(f"  Pearson r:  {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"  Spearman r: {spearman_r:.4f} (p={spearman_p:.2e})")

    # 5. ID / OOD 各自的相关性
    for tag, mask in [("id", id_mask), ("ood", ood_mask)]:
        if mask.sum() > 10:
            r, _ = pearsonr(knn_scores[mask], het_scores[mask])
            results["correlation"][f"{tag}_pearson_r"] = round(float(r), 4)
            print(f"  {tag.upper()} Pearson r: {r:.4f}")

    # ------------------------------------------------------------------
    # 6. 原始 AUROC
    # ------------------------------------------------------------------
    knn_auroc = roc_auc_score(test_labels, knn_scores)
    het_auroc = roc_auc_score(test_labels, het_scores)

    results["auroc"]["knn"] = round(float(knn_auroc), 4)
    results["auroc"]["het"] = round(float(het_auroc), 4)

    print(f"\nOriginal AUROC:")
    print(f"  KNN:  {knn_auroc:.4f}")
    print(f"  Het:  {het_auroc:.4f}")

    # ------------------------------------------------------------------
    # 7. Isotonic regression: het ~ knn  (拟合在 ID 上)
    # ------------------------------------------------------------------
    print(f"\nFitting Isotonic Regression on ID samples...")
    iso_reg = IsotonicRegression(out_of_bounds="clip")
    iso_reg.fit(knn_scores[id_mask], het_scores[id_mask])

    # 8. Residual = actual het − expected het
    het_expected = iso_reg.predict(knn_scores)
    het_residual = het_scores - het_expected

    residual_het_corr, _ = pearsonr(het_residual, het_scores)
    residual_knn_corr, _ = pearsonr(het_residual, knn_scores)

    results["residual"]["het_corr_with_original"] = round(float(residual_het_corr), 4)
    results["residual"]["knn_corr_with_residual"] = round(float(residual_knn_corr), 4)

    print(f"\nResidual Analysis:")
    print(f"  Residual-Het correlation:  {residual_het_corr:.4f}")
    print(f"  Residual-KNN correlation:  {residual_knn_corr:.4f}")

    # 9. Residual 的独立判别力
    residual_auroc = roc_auc_score(test_labels, het_residual)
    results["residual"]["auroc"] = round(float(residual_auroc), 4)
    print(f"  Residual AUROC: {residual_auroc:.4f}")

    # ------------------------------------------------------------------
    # 10. 融合对比
    # ------------------------------------------------------------------
    # 简单线性融合
    simple_fusion = 0.7 * knn_scores + 0.3 * het_scores
    simple_auroc = roc_auc_score(test_labels, simple_fusion)

    # 残差融合
    residual_fusion = knn_scores + 0.3 * het_residual
    rf_min, rf_max = residual_fusion.min(), residual_fusion.max()
    residual_fusion_norm = (residual_fusion - rf_min) / (rf_max - rf_min + 1e-10)
    residual_auroc_fused = roc_auc_score(test_labels, residual_fusion_norm)

    results["auroc"]["simple_fusion"] = round(float(simple_auroc), 4)
    results["auroc"]["residual_fusion"] = round(float(residual_auroc_fused), 4)

    improvement_simple = simple_auroc - knn_auroc
    improvement_residual = residual_auroc_fused - knn_auroc

    print(f"\nFusion Comparison:")
    print(f"  Simple fusion (0.7*KNN + 0.3*Het): {simple_auroc:.4f}")
    print(f"  Residual fusion (KNN + 0.3*Res):   {residual_auroc_fused:.4f}")
    print(f"  Baseline (KNN only):                {knn_auroc:.4f}")
    print(f"\n  Improvement over KNN:")
    print(f"    Simple:   {improvement_simple:+.4f}")
    print(f"    Residual: {improvement_residual:+.4f}")

    # ------------------------------------------------------------------
    # 11. Decision
    # ------------------------------------------------------------------
    if abs(residual_knn_corr) < 0.3 and residual_auroc > 0.6:
        decision = "PURSUE_RESIDUAL"
        reason = (
            f"Residual has low correlation with KNN ({residual_knn_corr:.3f}) "
            f"and decent discrimination ({residual_auroc:.3f})"
        )
    elif improvement_residual > 0.01:
        decision = "PURSUE_RESIDUAL_CAUTIOUS"
        reason = f"Residual shows {improvement_residual:.4f} improvement, worth exploring"
    elif abs(pearson_r) > 0.7:
        decision = "HIGH_CORRELATION_TRY_MULTISCALE"
        reason = f"Very high correlation ({pearson_r:.3f}), consider multi-scale instead"
    else:
        decision = "RECONSIDER_APPROACH"
        reason = "Residual doesn't show clear advantage"

    results["decision"] = decision
    results["decision_reason"] = reason

    print(f"\n{'='*60}")
    print(f"  DECISION: {decision}")
    print(f"  REASON:   {reason}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 12. 可视化
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (0,0) Scatter: Het vs KNN
    ax = axes[0, 0]
    ax.scatter(knn_scores[id_mask], het_scores[id_mask],
               alpha=0.3, s=10, label="ID", c="blue")
    ax.scatter(knn_scores[ood_mask], het_scores[ood_mask],
               alpha=0.5, s=20, label="OOD", c="red", marker="x")
    # Plot expected line (sorted for smooth curve)
    sort_idx = np.argsort(knn_scores[id_mask])
    ax.plot(knn_scores[id_mask][sort_idx], het_expected[id_mask][sort_idx],
            "g-", linewidth=2, label="Expected (ID fit)")
    ax.set_xlabel("KNN Score")
    ax.set_ylabel("Heterophily Score")
    ax.set_title(f"Het vs KNN (r={pearson_r:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (0,1) Scatter: Residual vs KNN
    ax = axes[0, 1]
    ax.scatter(knn_scores[id_mask], het_residual[id_mask],
               alpha=0.3, s=10, label="ID", c="blue")
    ax.scatter(knn_scores[ood_mask], het_residual[ood_mask],
               alpha=0.5, s=20, label="OOD", c="red", marker="x")
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("KNN Score")
    ax.set_ylabel("Residual Heterophily")
    ax.set_title(f"Residual vs KNN (r={residual_knn_corr:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,0) Distribution: Het — ID vs OOD
    ax = axes[1, 0]
    ax.hist(het_scores[id_mask], bins=30, alpha=0.6, label="ID", color="blue")
    ax.hist(het_scores[ood_mask], bins=30, alpha=0.6, label="OOD", color="red")
    ax.set_xlabel("Heterophily Score")
    ax.set_ylabel("Count")
    ax.set_title("Het Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,1) Distribution: Residual — ID vs OOD
    ax = axes[1, 1]
    ax.hist(het_residual[id_mask], bins=30, alpha=0.6, label="ID", color="blue")
    ax.hist(het_residual[ood_mask], bins=30, alpha=0.6, label="OOD", color="red")
    ax.set_xlabel("Residual Heterophily")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual Distribution (AUROC={residual_auroc:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"{dataset_name} — Het-KNN Correlation Analysis", fontsize=14, y=1.02)
    plt.tight_layout()

    save_dir = PROJECT_ROOT / "results" / "phase0"
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / f"correlation_analysis_{dataset_name}.pdf",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {save_dir / f'correlation_analysis_{dataset_name}.pdf'}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    all_results = {}

    for dataset in ["clinc150", "banking77"]:
        results = analyze_correlation(dataset)
        all_results[dataset] = results

    # 保存 JSON
    output_path = PROJECT_ROOT / "results" / "phase0" / "correlation_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")

    # 总结
    print(f"\n{'='*60}")
    print("FINAL RECOMMENDATIONS:")
    print(f"{'='*60}")
    for dataset, res in all_results.items():
        print(f"\n  {dataset}:")
        print(f"    Pearson r:       {res['correlation']['pearson_r']:.4f}")
        print(f"    Residual AUROC:  {res['residual']['auroc']:.4f}")
        print(f"    Decision:        {res['decision']}")
        print(f"    Reason:          {res['decision_reason']}")
