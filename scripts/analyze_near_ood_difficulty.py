#!/usr/bin/env python3
"""
Phase 3: 深度分析 Near-OOD 检测的困难性
重新定位 paper 为: "Understanding Near-OOD Challenges in Text"

前置条件:
  Phase 0   — heterophily 对 near-OOD 无特殊价值
  Phase 0.5 — 残差方法无独立判别力
  Phase 2   — 多尺度融合无法超越纯 KNN

本脚本回答:
  1. 有多少 OOD 样本在语义上极度接近 ID？
  2. 这些 "very near" 样本有多难检测？
  3. embedding 空间中 near-OOD 与 far-OOD 的结构差异

Author: RW3 OOD Detection Project
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
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
# 核心分析
# ---------------------------------------------------------------------------

def analyze_near_ood(dataset_name: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  Near-OOD Analysis: {dataset_name}")
    print(f"{'='*60}\n")

    data = load_features(dataset_name)
    train_emb = data["train_features"]
    train_labels = data["train_labels"]
    test_emb = data["test_features"]
    test_labels = data["test_labels"]
    input_dim = train_emb.shape[1]

    id_mask = test_labels == 0
    ood_mask = test_labels == 1
    id_test = test_emb[id_mask]
    ood_test = test_emb[ood_mask]

    results = {
        "dataset": dataset_name,
        "embedding_dim": input_dim,
        "n_id_test": int(id_mask.sum()),
        "n_ood_test": int(ood_mask.sum()),
    }

    # ==================================================================
    # 1. Semantic Similarity Analysis
    #    Features are L2-normalised → cosine similarity = dot product
    # ==================================================================
    print("1. Semantic Similarity Analysis...")

    # Compute max cosine similarity of each OOD sample to training set.
    # (n_ood, dim) @ (dim, n_train) → (n_ood, n_train)
    # Process in batches to limit memory.
    batch_size = 256
    max_sims = np.empty(len(ood_test), dtype=np.float32)
    for start in range(0, len(ood_test), batch_size):
        end = min(start + batch_size, len(ood_test))
        sims = ood_test[start:end] @ train_emb.T  # cosine sim (L2-normed)
        max_sims[start:end] = sims.max(axis=1)

    results["semantic_similarity"] = {
        "mean": round(float(max_sims.mean()), 4),
        "std": round(float(max_sims.std()), 4),
        "min": round(float(max_sims.min()), 4),
        "max": round(float(max_sims.max()), 4),
        "percentiles": {
            str(p): round(float(np.percentile(max_sims, p)), 4)
            for p in [25, 50, 75, 90]
        },
    }

    print(f"  OOD-to-ID max similarity:")
    print(f"    Mean: {max_sims.mean():.4f} +/- {max_sims.std():.4f}")
    print(f"    Range: [{max_sims.min():.4f}, {max_sims.max():.4f}]")
    print(f"    90th percentile: {np.percentile(max_sims, 90):.4f}")

    # ==================================================================
    # 2. Near-OOD Breakdown by similarity bands
    # ==================================================================
    print("\n2. Near-OOD Breakdown...")

    very_near = max_sims > 0.9
    near = (max_sims > 0.8) & (max_sims <= 0.9)
    medium = (max_sims > 0.7) & (max_sims <= 0.8)
    far = max_sims <= 0.7

    print(f"  OOD distribution by similarity:")
    for label, mask in [("Very Near (>0.9)", very_near),
                        ("Near (0.8-0.9)", near),
                        ("Medium (0.7-0.8)", medium),
                        ("Far (<0.7)", far)]:
        print(f"    {label:20s}: {mask.sum():5d} ({mask.mean()*100:5.1f}%)")

    results["ood_breakdown"] = {
        "very_near_count": int(very_near.sum()),
        "very_near_pct": round(float(very_near.mean() * 100), 1),
        "near_count": int(near.sum()),
        "near_pct": round(float(near.mean() * 100), 1),
        "medium_count": int(medium.sum()),
        "medium_pct": round(float(medium.mean() * 100), 1),
        "far_count": int(far.sum()),
        "far_pct": round(float(far.mean() * 100), 1),
    }

    # ==================================================================
    # 3. Detection Difficulty per group
    #    IMPORTANT: score the full test_emb once so min-max normalisation
    #    is consistent, then split by group.
    # ==================================================================
    print("\n3. Detection Difficulty by Similarity...")

    detector = HeterophilyEnhancedFixed(
        input_dim=input_dim, k=10, alpha=0.0, verbose=False
    )
    detector.fit(train_emb, train_labels)

    all_scores = detector.score(test_emb)       # single call → consistent normalisation
    id_scores = all_scores[id_mask]
    ood_scores = all_scores[ood_mask]

    difficulty = {}
    for group_name, group_mask in [("very_near", very_near),
                                    ("near", near),
                                    ("medium", medium),
                                    ("far", far)]:
        n_group = group_mask.sum()
        if n_group > 10:
            group_labels = np.concatenate([
                np.zeros(len(id_scores)),
                np.ones(n_group),
            ])
            group_scores = np.concatenate([id_scores, ood_scores[group_mask]])
            auroc = roc_auc_score(group_labels, group_scores)
            difficulty[group_name] = round(float(auroc), 4)
            print(f"  {group_name:15s}: AUROC = {auroc:.4f}  (n={n_group})")
        else:
            difficulty[group_name] = None
            print(f"  {group_name:15s}: too few samples (n={n_group})")

    results["detection_difficulty"] = difficulty

    # ==================================================================
    # 4. t-SNE visualisation
    # ==================================================================
    print("\n4. Generating t-SNE visualization...")
    np.random.seed(42)

    n_train_sample = min(1000, len(train_emb))
    n_id_sample = min(200, len(id_test))
    n_very_near_sample = min(50, int(very_near.sum()))
    n_far_sample = min(50, int(far.sum()))

    train_idx = np.random.choice(len(train_emb), n_train_sample, replace=False)

    parts = [train_emb[train_idx], id_test[:n_id_sample]]
    if n_very_near_sample > 0:
        parts.append(ood_test[very_near][:n_very_near_sample])
    if n_far_sample > 0:
        parts.append(ood_test[far][:n_far_sample])

    vis_emb = np.vstack(parts)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb_2d = tsne.fit_transform(vis_emb)

    # ------------------------------------------------------------------
    # Build figure (1x3)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: similarity histogram
    ax = axes[0]
    ax.hist(max_sims, bins=50, alpha=0.7, edgecolor="black")
    ax.axvline(x=0.9, color="r", linestyle="--", label="Very Near (0.9)")
    ax.axvline(x=0.8, color="orange", linestyle="--", label="Near (0.8)")
    ax.axvline(x=0.7, color="gold", linestyle="--", label="Medium (0.7)")
    ax.set_xlabel("Max Cosine Similarity to ID")
    ax.set_ylabel("Count")
    ax.set_title("OOD Semantic Similarity Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: AUROC bar chart
    ax = axes[1]
    groups = [k for k, v in difficulty.items() if v is not None]
    auroc_vals = [difficulty[k] for k in groups]
    color_map = {"very_near": "red", "near": "orange", "medium": "gold", "far": "green"}
    colors = [color_map.get(g, "gray") for g in groups]

    bars = ax.bar(groups, auroc_vals, color=colors, alpha=0.7, edgecolor="black")
    ax.axhline(y=0.5, color="k", linestyle="--", alpha=0.5, label="Random")
    ax.set_ylabel("AUROC")
    ax.set_title("Detection Difficulty by OOD Type")
    ax.set_ylim([0, 1.08])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, auroc_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold")

    # Plot 3: t-SNE
    ax = axes[2]
    cursor = 0
    ax.scatter(emb_2d[cursor:cursor + n_train_sample, 0],
               emb_2d[cursor:cursor + n_train_sample, 1],
               c="lightblue", s=10, alpha=0.3, label="Train (ID)")
    cursor += n_train_sample

    ax.scatter(emb_2d[cursor:cursor + n_id_sample, 0],
               emb_2d[cursor:cursor + n_id_sample, 1],
               c="blue", s=30, alpha=0.6, label="Test (ID)", marker="o")
    cursor += n_id_sample

    if n_very_near_sample > 0:
        ax.scatter(emb_2d[cursor:cursor + n_very_near_sample, 0],
                   emb_2d[cursor:cursor + n_very_near_sample, 1],
                   c="red", s=50, alpha=0.8, label="Very Near OOD", marker="x")
        cursor += n_very_near_sample

    if n_far_sample > 0:
        ax.scatter(emb_2d[cursor:cursor + n_far_sample, 0],
                   emb_2d[cursor:cursor + n_far_sample, 1],
                   c="darkred", s=50, alpha=0.8, label="Far OOD", marker="^")

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("Embedding Space Visualization")
    ax.legend(fontsize=8)

    plt.tight_layout()

    save_dir = PROJECT_ROOT / "results" / "phase3"
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / f"near_ood_analysis_{dataset_name}.pdf",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {save_dir / f'near_ood_analysis_{dataset_name}.pdf'}")

    # ==================================================================
    # 5. Key insight
    # ==================================================================
    very_near_pct = very_near.mean() * 100
    near_auroc = difficulty.get("very_near") or 0.5

    if very_near_pct > 30 and (near_auroc is not None and near_auroc < 0.7):
        insight = (
            f"HIGH NEAR-OOD PREVALENCE: {very_near_pct:.1f}% of OOD samples are "
            f"very similar to ID (>0.9), and they are HARD to detect "
            f"(AUROC={near_auroc:.3f})"
        )
    elif very_near_pct > 20:
        insight = f"MODERATE NEAR-OOD CHALLENGE: {very_near_pct:.1f}% very similar samples"
    else:
        insight = "PRIMARILY FAR-OOD: Most OOD samples are distinguishable"

    results["key_insight"] = insight

    print(f"\n{'='*60}")
    print("KEY FINDINGS:")
    print(f"{'='*60}")
    print(f"\n{insight}\n")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    all_results = {}

    for dataset in ["clinc150", "banking77"]:
        results = analyze_near_ood(dataset)
        all_results[dataset] = results

    # Save JSON
    output_path = PROJECT_ROOT / "results" / "phase3" / "near_ood_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")

    # Paper repositioning
    print(f"\n{'='*60}")
    print("PAPER REPOSITIONING:")
    print(f"{'='*60}")

    print("""
New Paper Focus: "Understanding and Addressing Near-OOD Detection
                  in Text Classification"

Key Contributions:
  1. Empirical Analysis: Systematic study of heterophily signal and its
     correlation with KNN distance (Phase 0 + 0.5)

  2. Negative Results with Rigor:
     - Residual de-correlation: removes signal, not noise (Phase 0.5)
     - Multi-scale fusion: different k values are redundant (Phase 2)
     - Gated fusion: heterophily adds no orthogonal information

  3. Near-OOD Characterisation: Quantified the prevalence and detection
     difficulty of "very near" OOD samples (cosine sim > 0.9 to ID)

  4. Failure Taxonomy: When and why distance-based OOD methods fail
     in NLU — samples that are semantically adjacent to ID intents

  5. Future Directions:
     - Contrastive fine-tuning to push near-OOD away from ID clusters
     - Class-conditional density estimation (not global distance)
     - Hybrid approaches combining logit-based and distance-based signals

Target Venue: EMNLP Findings or ACL Findings
(Honest negative-result + analysis papers are valued!)
    """)
