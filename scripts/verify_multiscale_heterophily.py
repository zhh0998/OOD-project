#!/usr/bin/env python3
"""
Phase 2: 验证多尺度 heterophily 方法
不同的 k 值可能捕获不同层次的结构信息

Decision Criteria:
  SUCCESS  — fusion improves > 1% over best KNN → 继续开发
  MARGINAL — 0.5-1% improvement → 谨慎评估
  FAILED   — no improvement → 转向 Phase 3 (Near-OOD analysis)

Author: RW3 OOD Detection Project
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.heterophily_detector import HeterophilyEnhancedFixed


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_features(dataset_name: str) -> dict:
    feature_map = {
        "clinc150": PROJECT_ROOT / "features" / "clinc150_native_oos_finetuned_seed42.npz",
        "banking77": PROJECT_ROOT / "features" / "banking77_finetuned_seed42.npz",
    }
    data = np.load(str(feature_map[dataset_name]), allow_pickle=True)
    return {
        "train_features": data["train_features"],
        "test_features": data["test_features"],
        "train_labels": data["train_labels"],
        "test_labels": data["test_labels"],
    }


# ---------------------------------------------------------------------------
# 核心验证
# ---------------------------------------------------------------------------

def verify_multiscale(dataset_name: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  Multi-scale verification: {dataset_name}")
    print(f"{'='*60}\n")

    data = load_features(dataset_name)
    train_emb = data["train_features"]
    train_labels = data["train_labels"]
    test_emb = data["test_features"]
    test_labels = data["test_labels"]
    input_dim = train_emb.shape[1]

    results = {"dataset": dataset_name, "scales": {}}

    # ------------------------------------------------------------------
    # 1. 不同 k 值的 KNN / Het 单独表现
    # ------------------------------------------------------------------
    k_values = [3, 5, 10, 20, 50]

    print("Testing different k values...")
    scores_dict = {}
    aurocs = {}

    for k in k_values:
        # 纯 KNN (alpha=0)
        det_knn = HeterophilyEnhancedFixed(
            input_dim=input_dim, k=k, alpha=0.0, verbose=False
        )
        det_knn.fit(train_emb, train_labels)
        knn_scores = det_knn.score(test_emb)
        knn_auroc = roc_auc_score(test_labels, knn_scores)

        # 纯 Heterophily (alpha=1)
        det_het = HeterophilyEnhancedFixed(
            input_dim=input_dim, k=k, alpha=1.0, verbose=False
        )
        det_het.fit(train_emb, train_labels)
        het_scores = det_het.score(test_emb)
        het_auroc = roc_auc_score(test_labels, het_scores)

        scores_dict[f"knn_k{k}"] = knn_scores
        scores_dict[f"het_k{k}"] = het_scores

        aurocs[k] = {"knn": round(float(knn_auroc), 4), "het": round(float(het_auroc), 4)}
        print(f"  k={k:2d}: KNN={knn_auroc:.4f}, Het={het_auroc:.4f}")

        results["scales"][f"k{k}"] = aurocs[k]

    # ------------------------------------------------------------------
    # 2. 多尺度融合策略
    # ------------------------------------------------------------------
    print("\nTrying multi-scale fusion...")

    # 策略 1: 简单平均 (all het scores across k)
    multiscale_avg = np.mean(
        [scores_dict[f"het_k{k}"] for k in k_values], axis=0
    )
    avg_auroc = roc_auc_score(test_labels, multiscale_avg)

    # 策略 2: AUROC 加权平均
    weights = np.array([aurocs[k]["het"] for k in k_values])
    weights = weights / weights.sum()
    multiscale_weighted = np.sum(
        [w * scores_dict[f"het_k{k}"] for w, k in zip(weights, k_values)],
        axis=0,
    )
    weighted_auroc = roc_auc_score(test_labels, multiscale_weighted)

    # 策略 3: 与最佳 KNN 融合
    best_k = max(k_values, key=lambda k: aurocs[k]["knn"])
    best_knn_scores = scores_dict[f"knn_k{best_k}"]
    best_knn_auroc = aurocs[best_k]["knn"]

    fusion_score = 0.7 * best_knn_scores + 0.3 * multiscale_weighted
    fusion_auroc = roc_auc_score(test_labels, fusion_score)

    results["multiscale"] = {
        "avg_het_auroc": round(float(avg_auroc), 4),
        "weighted_het_auroc": round(float(weighted_auroc), 4),
        "best_knn_k": int(best_k),
        "best_knn_auroc": round(float(best_knn_auroc), 4),
        "fusion_auroc": round(float(fusion_auroc), 4),
    }

    print(f"\nMulti-scale results:")
    print(f"  Avg het:           {avg_auroc:.4f}")
    print(f"  Weighted het:      {weighted_auroc:.4f}")
    print(f"  Best KNN (k={best_k}):  {best_knn_auroc:.4f}")
    print(f"  Fusion (0.7*KNN + 0.3*wHet): {fusion_auroc:.4f}")

    # ------------------------------------------------------------------
    # 3. Decision
    # ------------------------------------------------------------------
    improvement = fusion_auroc - best_knn_auroc

    if improvement > 0.01:
        decision = "SUCCESS"
        reason = f"Multi-scale fusion improves by {improvement:+.4f}"
    elif improvement > 0.005:
        decision = "MARGINAL"
        reason = f"Small improvement ({improvement:+.4f})"
    else:
        decision = "FAILED"
        reason = f"No improvement ({improvement:+.4f})"

    results["decision"] = decision
    results["decision_reason"] = reason
    results["improvement_over_best_knn"] = round(float(improvement), 4)

    print(f"\n{'='*60}")
    print(f"  DECISION: {decision}")
    print(f"  REASON:   {reason}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 4. 可视化
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (左) k 值敏感性
    ax = axes[0]
    k_list = sorted(aurocs.keys())
    knn_list = [aurocs[k]["knn"] for k in k_list]
    het_list = [aurocs[k]["het"] for k in k_list]

    ax.plot(k_list, knn_list, "o-", linewidth=2, label="KNN", markersize=8)
    ax.plot(k_list, het_list, "s-", linewidth=2, label="Het", markersize=8)
    ax.axhline(y=fusion_auroc, color="r", linestyle="--",
               label=f"Multi-scale Fusion ({fusion_auroc:.3f})")
    ax.axhline(y=best_knn_auroc, color="gray", linestyle=":",
               label=f"Best KNN k={best_k} ({best_knn_auroc:.3f})")
    ax.set_xlabel("k value")
    ax.set_ylabel("AUROC")
    ax.set_title(f"{dataset_name} — k Sensitivity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    # (右) het 分数跨 k 值的相关性矩阵
    ax = axes[1]
    het_stack = [scores_dict[f"het_k{k}"] for k in k_values]
    corr_matrix = np.corrcoef(het_stack)
    im = ax.imshow(corr_matrix, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(k_values)))
    ax.set_yticks(range(len(k_values)))
    ax.set_xticklabels([f"k={k}" for k in k_values])
    ax.set_yticklabels([f"k={k}" for k in k_values])
    ax.set_title("Het Score Correlation Across k")
    plt.colorbar(im, ax=ax)

    for i in range(len(k_values)):
        for j in range(len(k_values)):
            ax.text(j, i, f"{corr_matrix[i, j]:.2f}",
                    ha="center", va="center", color="black", fontsize=9)

    plt.tight_layout()

    save_dir = PROJECT_ROOT / "results" / "phase2"
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / f"multiscale_analysis_{dataset_name}.pdf",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {save_dir / f'multiscale_analysis_{dataset_name}.pdf'}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    all_results = {}

    for dataset in ["clinc150", "banking77"]:
        results = verify_multiscale(dataset)
        all_results[dataset] = results

    # 保存 JSON
    output_path = PROJECT_ROOT / "results" / "phase2" / "multiscale_verification.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")

    # 总决策
    print(f"\n{'='*60}")
    print("FINAL DECISION:")
    print(f"{'='*60}")
    for dataset, res in all_results.items():
        print(f"\n  {dataset}:")
        print(f"    Best KNN (k={res['multiscale']['best_knn_k']}): {res['multiscale']['best_knn_auroc']:.4f}")
        print(f"    Fusion:  {res['multiscale']['fusion_auroc']:.4f}  ({res['improvement_over_best_knn']:+.4f})")
        print(f"    Decision: {res['decision']} — {res['decision_reason']}")
