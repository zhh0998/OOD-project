"""
Phase 0: 验证 heterophily 在 near-OOD vs far-OOD 上的不同表现

假设：heterophily（邻域标签熵）对 near-OOD 的判别贡献 > far-OOD
- Near-OOD: KNN距离小（样本靠近ID边界），het_relative_gain > 50%
- Far-OOD:  KNN距离大（样本远离ID分布），het_relative_gain < 30%

如果成立 → 继续实现门控融合（gated fusion）
如果不成立 → 分析原因后再决定

Author: RW3 OOD Detection Project
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.heterophily_detector import HeterophilyEnhancedFixed


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_features(dataset_name: str):
    """
    加载预提取的 finetuned RoBERTa embeddings（768维）。
    如果 npz 不存在，则回退到 SentenceTransformer（384维）。

    Returns:
        dict with keys: train_emb, test_emb, train_labels, test_labels, dim
    """
    feature_map = {
        "clinc150": PROJECT_ROOT / "features" / "clinc150_native_oos_finetuned_seed42.npz",
        "banking77": PROJECT_ROOT / "features" / "banking77_finetuned_seed42.npz",
    }

    npz_path = feature_map.get(dataset_name)

    if npz_path is not None and npz_path.exists():
        print(f"[load] 使用预提取特征: {npz_path.name}")
        data = np.load(str(npz_path), allow_pickle=True)
        return {
            "train_emb": data["train_features"],
            "test_emb": data["test_features"],
            "train_labels": data["train_labels"],
            "test_labels": data["test_labels"],
            "dim": data["train_features"].shape[1],
        }

    # 回退: 用 SentenceTransformer 从文本生成 embeddings
    print(f"[load] npz 不存在，回退到 SentenceTransformer ...")
    from sentence_transformers import SentenceTransformer
    from src.utils.data_loader import load_clinc150, load_banking77_oos

    if dataset_name == "clinc150":
        train_texts, test_texts, test_labels, _, train_labels = load_clinc150()
    else:
        train_texts, test_texts, test_labels, _, train_labels = load_banking77_oos()

    model = SentenceTransformer("all-MiniLM-L6-v2")
    train_emb = model.encode(train_texts, show_progress_bar=True, batch_size=256)
    test_emb = model.encode(test_texts, show_progress_bar=True, batch_size=256)

    return {
        "train_emb": train_emb.astype(np.float32),
        "test_emb": test_emb.astype(np.float32),
        "train_labels": np.array(train_labels),
        "test_labels": np.array(test_labels),
        "dim": train_emb.shape[1],
    }


# ---------------------------------------------------------------------------
# 核心分析
# ---------------------------------------------------------------------------

def near_far_breakdown(dataset_name: str, k: int = 10):
    """
    验证 heterophily 在 near-OOD vs far-OOD 的不同表现。

    步骤：
      1. alpha=0 得到纯 KNN 分数
      2. alpha=1 得到纯 heterophily 分数
      3. 按 KNN 分数将 OOD 样本分成 near(30%) / middle(40%) / far(30%)
      4. 对每组分别计算 AUROC 和 relative gain
    """

    feat = load_features(dataset_name)
    train_emb = feat["train_emb"]
    test_emb = feat["test_emb"]
    train_labels = feat["train_labels"]
    test_labels = feat["test_labels"]
    dim = feat["dim"]

    # --- 纯 KNN 分数 (alpha=0) ---
    det_knn = HeterophilyEnhancedFixed(input_dim=dim, k=k, alpha=0.0, verbose=False)
    det_knn.fit(train_emb, train_labels)
    knn_scores = det_knn.score(test_emb)

    # --- 纯 heterophily 分数 (alpha=1) ---
    det_het = HeterophilyEnhancedFixed(input_dim=dim, k=k, alpha=1.0, verbose=False)
    det_het.fit(train_emb, train_labels)
    het_scores = det_het.score(test_emb)

    # --- 融合分数 (alpha=0.3, 默认) ---
    det_fused = HeterophilyEnhancedFixed(input_dim=dim, k=k, alpha=0.3, verbose=False)
    det_fused.fit(train_emb, train_labels)
    fused_scores = det_fused.score(test_emb)

    # --- 分组分析 ---
    ood_mask = test_labels == 1
    id_mask = test_labels == 0
    n_id = id_mask.sum()
    n_ood = ood_mask.sum()

    # OOD 样本的 KNN 分数
    ood_knn = knn_scores[ood_mask]

    # 按 KNN 距离分组: near(30%), middle(40%), far(30%)
    near_threshold = np.percentile(ood_knn, 30)
    far_threshold = np.percentile(ood_knn, 70)

    # 在整个测试集上找到 near/far OOD 的索引
    near_ood_idx = np.where(ood_mask & (knn_scores <= near_threshold))[0]
    mid_ood_idx = np.where(ood_mask & (knn_scores > near_threshold) & (knn_scores < far_threshold))[0]
    far_ood_idx = np.where(ood_mask & (knn_scores >= far_threshold))[0]
    id_idx = np.where(id_mask)[0]

    def compute_group_auroc(group_ood_idx, group_name):
        """对一组 OOD 样本 (vs 全部 ID) 计算各方法的 AUROC"""
        if len(group_ood_idx) == 0:
            return None

        labels = np.concatenate([np.zeros(len(id_idx)), np.ones(len(group_ood_idx))])
        combined_idx = np.concatenate([id_idx, group_ood_idx])

        knn_auc = roc_auc_score(labels, knn_scores[combined_idx])
        het_auc = roc_auc_score(labels, het_scores[combined_idx])
        fused_auc = roc_auc_score(labels, fused_scores[combined_idx])

        # heterophily 相对于随机(0.5) 的贡献 / KNN 相对于随机的贡献
        knn_above_chance = knn_auc - 0.5
        het_above_chance = het_auc - 0.5

        if abs(knn_above_chance) < 1e-6:
            het_relative_gain = float("nan")
        else:
            het_relative_gain = het_above_chance / knn_above_chance

        return {
            "n_ood": int(len(group_ood_idx)),
            "knn_auroc": float(round(knn_auc, 4)),
            "het_auroc": float(round(het_auc, 4)),
            "fused_auroc": float(round(fused_auc, 4)),
            "het_relative_gain": float(round(het_relative_gain, 4)),
            "knn_score_range": [
                float(round(knn_scores[group_ood_idx].min(), 4)),
                float(round(knn_scores[group_ood_idx].max(), 4)),
            ],
        }

    # 全局 AUROC
    overall_knn_auc = roc_auc_score(test_labels, knn_scores)
    overall_het_auc = roc_auc_score(test_labels, het_scores)
    overall_fused_auc = roc_auc_score(test_labels, fused_scores)

    results = {
        "dataset": dataset_name,
        "k": k,
        "n_id_test": int(n_id),
        "n_ood_test": int(n_ood),
        "embedding_dim": dim,
        "overall": {
            "knn_auroc": float(round(overall_knn_auc, 4)),
            "het_auroc": float(round(overall_het_auc, 4)),
            "fused_auroc": float(round(overall_fused_auc, 4)),
        },
        "near": compute_group_auroc(near_ood_idx, "near"),
        "middle": compute_group_auroc(mid_ood_idx, "middle"),
        "far": compute_group_auroc(far_ood_idx, "far"),
    }

    return results


# ---------------------------------------------------------------------------
# 打印 & 决策
# ---------------------------------------------------------------------------

def print_results(results: dict):
    ds = results["dataset"]
    print(f"\n{'='*60}")
    print(f"  {ds.upper()}  (k={results['k']}, dim={results['embedding_dim']})")
    print(f"  ID test: {results['n_id_test']},  OOD test: {results['n_ood_test']}")
    print(f"{'='*60}")

    # Overall
    o = results["overall"]
    print(f"\n  Overall  AUROC  — KNN: {o['knn_auroc']:.4f}  |  Het: {o['het_auroc']:.4f}  |  Fused(a=0.3): {o['fused_auroc']:.4f}")

    for group in ["near", "middle", "far"]:
        g = results[group]
        if g is None:
            continue
        tag = {"near": "Near-OOD (bottom 30%)", "middle": "Mid-OOD  (30-70%)", "far": "Far-OOD  (top 30%)"}[group]
        print(f"\n  {tag}  [n={g['n_ood']}, knn_range={g['knn_score_range']}]")
        print(f"    KNN AUROC:          {g['knn_auroc']:.4f}")
        print(f"    Het AUROC:          {g['het_auroc']:.4f}")
        print(f"    Fused AUROC:        {g['fused_auroc']:.4f}")
        print(f"    Het relative gain:  {g['het_relative_gain']:.2%}")

    print()


def make_decision(all_results: dict):
    """根据数据做出 go/no-go 决策"""
    print("\n" + "=" * 60)
    print("  DECISION")
    print("=" * 60)

    go = True
    reasons = []
    for ds, res in all_results.items():
        near_gain = res["near"]["het_relative_gain"]
        far_gain = res["far"]["het_relative_gain"]

        near_ok = near_gain > 0.50
        far_ok = far_gain < 0.30

        status_near = "PASS" if near_ok else "FAIL"
        status_far = "PASS" if far_ok else "FAIL"

        reasons.append(
            f"  {ds}: near_gain={near_gain:.2%} [{status_near}]  "
            f"far_gain={far_gain:.2%} [{status_far}]"
        )

        if not (near_ok and far_ok):
            go = False

    for r in reasons:
        print(r)

    if go:
        print("\n  >>> PASS: heterophily 在 near-OOD 有显著价值，继续实现门控融合 <<<")
    else:
        print("\n  >>> CONDITIONAL: 部分条件未满足，需进一步分析 <<<")
        print("    建议: 检查不同 k 值 / 不同 embedding 是否改变结论")

    return go


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    output_dir = PROJECT_ROOT / "results" / "phase0"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "near_far_verification.json"

    all_results = {}

    for dataset in ["clinc150", "banking77"]:
        print(f"\n{'#'*60}")
        print(f"  Processing: {dataset}")
        print(f"{'#'*60}")

        results = near_far_breakdown(dataset, k=10)
        print_results(results)
        all_results[dataset] = results

    decision = make_decision(all_results)

    # 保存 JSON
    output = {
        "description": "Phase 0: near-OOD vs far-OOD heterophily verification",
        "hypothesis": {
            "near_ood": "het_relative_gain > 50%",
            "far_ood": "het_relative_gain < 30%",
        },
        "decision": "GO - proceed with gated fusion" if decision else "CONDITIONAL - needs further analysis",
        "results": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {output_path}")
