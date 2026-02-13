#!/usr/bin/env python3
"""
RW3 Kill-Switch Determination Experiment - Part B: RAG Route
Embedding anisotropy analysis, spectral whitening, and retrieval graph features
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import defaultdict
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
SEEDS = [42, 123, 456]
np.random.seed(42)
torch.manual_seed(42)

# Output directory
RESULTS_DIR = Path("/home/user/OOD-project/rw3_pilot/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("RW3 RAG Route Kill-Switch Experiment")
print("=" * 60)

# ============================================================
# B0: Environment Setup
# ============================================================
print("\n[B0] Setting up environment...")

from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import faiss

# Load embedding model
print("  Loading all-MiniLM-L6-v2 model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
EMBEDDING_DIM = 384

# ============================================================
# B1: Embedding Extraction and Anisotropy Baseline
# ============================================================
print("\n[B1] Loading CLINC150 and extracting embeddings...")

def load_clinc150():
    """Load CLINC150 dataset with OOD split"""
    ds = load_dataset("clinc_oos", "plus")

    data = {
        'train': {'texts': [], 'labels': [], 'is_ood': []},
        'val': {'texts': [], 'labels': [], 'is_ood': []},
        'test': {'texts': [], 'labels': [], 'is_ood': []}
    }

    for split_name, split_data in [('train', ds['train']), ('val', ds['validation']), ('test', ds['test'])]:
        for item in split_data:
            data[split_name]['texts'].append(item['text'])
            data[split_name]['labels'].append(item['intent'])
            # intent 150 is the OOD class in CLINC150
            data[split_name]['is_ood'].append(item['intent'] == 150)

    return data

def extract_embeddings(texts, batch_size=64):
    """Extract embeddings using sentence-transformers"""
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    # L2 normalize
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings

# Load CLINC150
clinc_data = load_clinc150()
print(f"  CLINC150 loaded:")
print(f"    Train: {len(clinc_data['train']['texts'])} samples")
print(f"    Val: {len(clinc_data['val']['texts'])} samples")
print(f"    Test: {len(clinc_data['test']['texts'])} samples")

# Extract embeddings for all splits
embeddings = {}
for split in ['train', 'val', 'test']:
    embeddings[split] = extract_embeddings(clinc_data[split]['texts'])
    print(f"    {split} embeddings shape: {embeddings[split].shape}")

# Get ID and OOD splits
train_is_ood = np.array(clinc_data['train']['is_ood'])
test_is_ood = np.array(clinc_data['test']['is_ood'])
test_labels = np.array(clinc_data['test']['labels'])

train_id_emb = embeddings['train'][~train_is_ood]
train_id_labels = np.array(clinc_data['train']['labels'])[~train_is_ood]

test_id_emb = embeddings['test'][~test_is_ood]
test_ood_emb = embeddings['test'][test_is_ood]

print(f"\n  ID/OOD split:")
print(f"    Train ID: {len(train_id_emb)}")
print(f"    Test ID: {len(test_id_emb)}")
print(f"    Test OOD: {len(test_ood_emb)}")

# Define near/medium/far OOD based on similarity to nearest ID sample
def compute_ood_severity(ood_emb, id_emb):
    """Classify OOD samples by cosine similarity to nearest ID sample"""
    # Compute cosine similarity matrix
    sims = np.dot(ood_emb, id_emb.T)
    max_sims = sims.max(axis=1)

    # near-OOD: sim > 0.7 (or 0.6 if <100 samples)
    near_threshold = 0.7
    if (max_sims > near_threshold).sum() < 100:
        near_threshold = 0.6

    near_mask = max_sims > near_threshold
    far_mask = max_sims < 0.4
    medium_mask = ~near_mask & ~far_mask

    return {
        'near': near_mask,
        'medium': medium_mask,
        'far': far_mask,
        'max_sims': max_sims
    }

ood_severity = compute_ood_severity(test_ood_emb, train_id_emb)
print(f"\n  OOD severity distribution:")
print(f"    Near-OOD (sim > {0.6 if ood_severity['near'].sum() < 100 else 0.7}): {ood_severity['near'].sum()}")
print(f"    Medium-OOD (0.4-0.7): {ood_severity['medium'].sum()}")
print(f"    Far-OOD (sim < 0.4): {ood_severity['far'].sum()}")

# Measure anisotropy
def measure_anisotropy(emb):
    """Measure embedding space anisotropy"""
    # Covariance matrix eigenvalues
    cov = np.cov(emb.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    total_var = eigenvalues.sum()

    metrics = {
        'top1_var_ratio': eigenvalues[0] / total_var,
        'top5_var_ratio': eigenvalues[:5].sum() / total_var,
        'top10_var_ratio': eigenvalues[:10].sum() / total_var,
    }

    # Random pair cosine similarity (sample 10000 pairs)
    n = len(emb)
    n_pairs = min(10000, n * (n-1) // 2)
    idx1 = np.random.randint(0, n, n_pairs)
    idx2 = np.random.randint(0, n, n_pairs)
    # Avoid same index
    valid = idx1 != idx2
    idx1, idx2 = idx1[valid], idx2[valid]

    pair_sims = (emb[idx1] * emb[idx2]).sum(axis=1)
    metrics['mean_pair_sim'] = pair_sims.mean()
    metrics['std_pair_sim'] = pair_sims.std()

    return metrics

anisotropy_baseline = measure_anisotropy(train_id_emb)
print(f"\n  Anisotropy baseline:")
print(f"    Top-1 variance ratio: {anisotropy_baseline['top1_var_ratio']:.4f}")
print(f"    Top-5 variance ratio: {anisotropy_baseline['top5_var_ratio']:.4f}")
print(f"    Top-10 variance ratio: {anisotropy_baseline['top10_var_ratio']:.4f}")
print(f"    Mean pair cosine sim: {anisotropy_baseline['mean_pair_sim']:.4f}")

# kNN-based OOD detection
def compute_knn_scores(query_emb, ref_emb, k=20):
    """Compute kNN distance scores for OOD detection"""
    index = faiss.IndexFlatIP(ref_emb.shape[1])  # Inner product = cosine for normalized vectors
    index.add(ref_emb.astype(np.float32))

    sims, _ = index.search(query_emb.astype(np.float32), k)
    # Distance = 1 - similarity (higher = more OOD)
    distances = 1 - sims.mean(axis=1)
    return distances

# Compute baseline OOD scores
id_scores_baseline = compute_knn_scores(test_id_emb, train_id_emb)
ood_scores_baseline = compute_knn_scores(test_ood_emb, train_id_emb)

# Cohen's d for near-OOD
near_ood_scores = ood_scores_baseline[ood_severity['near']]
def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return abs(group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0

baseline_cohens_d_near = cohens_d(id_scores_baseline, near_ood_scores)
baseline_cohens_d_all = cohens_d(id_scores_baseline, ood_scores_baseline)

# AUROC
baseline_auroc_all = roc_auc_score(
    np.concatenate([np.zeros(len(test_id_emb)), np.ones(len(test_ood_emb))]),
    np.concatenate([id_scores_baseline, ood_scores_baseline])
)

# Near-OOD AUROC
near_ood_emb = test_ood_emb[ood_severity['near']]
baseline_auroc_near = roc_auc_score(
    np.concatenate([np.zeros(len(test_id_emb)), np.ones(len(near_ood_emb))]),
    np.concatenate([id_scores_baseline, near_ood_scores])
) if len(near_ood_emb) > 0 else 0

print(f"\n  Baseline OOD detection (kNN, k=20):")
print(f"    All-OOD AUROC: {baseline_auroc_all:.4f}")
print(f"    Near-OOD AUROC: {baseline_auroc_near:.4f}")
print(f"    Near-OOD Cohen's d: {baseline_cohens_d_near:.4f}")
print(f"    All-OOD Cohen's d: {baseline_cohens_d_all:.4f}")

# Save baseline results
with open(RESULTS_DIR / "baseline_anisotropy.md", "w") as f:
    f.write("# Baseline Embedding Anisotropy Analysis\n\n")
    f.write("## Dataset: CLINC150\n\n")
    f.write(f"- Train ID samples: {len(train_id_emb)}\n")
    f.write(f"- Test ID samples: {len(test_id_emb)}\n")
    f.write(f"- Test OOD samples: {len(test_ood_emb)}\n")
    f.write(f"  - Near-OOD: {ood_severity['near'].sum()}\n")
    f.write(f"  - Medium-OOD: {ood_severity['medium'].sum()}\n")
    f.write(f"  - Far-OOD: {ood_severity['far'].sum()}\n\n")

    f.write("## Anisotropy Metrics\n\n")
    f.write(f"| Metric | Value |\n")
    f.write(f"|--------|-------|\n")
    f.write(f"| Top-1 variance ratio | {anisotropy_baseline['top1_var_ratio']:.4f} |\n")
    f.write(f"| Top-5 variance ratio | {anisotropy_baseline['top5_var_ratio']:.4f} |\n")
    f.write(f"| Top-10 variance ratio | {anisotropy_baseline['top10_var_ratio']:.4f} |\n")
    f.write(f"| Mean pair cosine sim | {anisotropy_baseline['mean_pair_sim']:.4f} |\n\n")

    f.write("## OOD Detection Performance (kNN, k=20)\n\n")
    f.write(f"| Metric | Value |\n")
    f.write(f"|--------|-------|\n")
    f.write(f"| All-OOD AUROC | {baseline_auroc_all:.4f} |\n")
    f.write(f"| Near-OOD AUROC | {baseline_auroc_near:.4f} |\n")
    f.write(f"| Near-OOD Cohen's d | {baseline_cohens_d_near:.4f} |\n")
    f.write(f"| All-OOD Cohen's d | {baseline_cohens_d_all:.4f} |\n")

# ============================================================
# B2: Spectral Whitening Ablation
# ============================================================
print("\n[B2] Running spectral whitening ablation...")

def spectral_whitening(embeddings, k=1):
    """All-but-the-top: Remove top-k principal components (Mu & Viswanath 2018)"""
    mean = embeddings.mean(axis=0)
    centered = embeddings - mean

    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    # Remove top-k directions
    whitened = centered.copy()
    for i in range(k):
        component = Vt[i]
        whitened = whitened - np.outer(whitened @ component, component)

    # Re-normalize
    norms = np.linalg.norm(whitened, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1)
    whitened = whitened / norms

    return whitened, mean, Vt[:k]

# Test different k values
k_values = [0, 1, 2, 3, 5, 10]
whitening_results = []

for k in k_values:
    print(f"  Testing k={k}...")

    if k == 0:
        # No whitening
        train_w = train_id_emb
        test_id_w = test_id_emb
        test_ood_w = test_ood_emb
    else:
        # Apply whitening
        train_w, mean, components = spectral_whitening(train_id_emb, k)
        # Apply same transformation to test data
        test_id_centered = test_id_emb - mean
        test_ood_centered = test_ood_emb - mean
        for comp in components:
            test_id_centered = test_id_centered - np.outer(test_id_centered @ comp, comp)
            test_ood_centered = test_ood_centered - np.outer(test_ood_centered @ comp, comp)
        # Normalize
        test_id_w = test_id_centered / np.linalg.norm(test_id_centered, axis=1, keepdims=True)
        test_ood_w = test_ood_centered / np.linalg.norm(test_ood_centered, axis=1, keepdims=True)

    # Compute kNN scores
    id_scores = compute_knn_scores(test_id_w, train_w)
    ood_scores = compute_knn_scores(test_ood_w, train_w)

    # OOD severity for whitened embeddings
    near_scores = ood_scores[ood_severity['near']]
    medium_scores = ood_scores[ood_severity['medium']]
    far_scores = ood_scores[ood_severity['far']]

    # Compute metrics
    auroc_all = roc_auc_score(
        np.concatenate([np.zeros(len(test_id_w)), np.ones(len(test_ood_w))]),
        np.concatenate([id_scores, ood_scores])
    )

    auroc_near = roc_auc_score(
        np.concatenate([np.zeros(len(test_id_w)), np.ones(ood_severity['near'].sum())]),
        np.concatenate([id_scores, near_scores])
    ) if ood_severity['near'].sum() > 0 else 0

    auroc_medium = roc_auc_score(
        np.concatenate([np.zeros(len(test_id_w)), np.ones(ood_severity['medium'].sum())]),
        np.concatenate([id_scores, medium_scores])
    ) if ood_severity['medium'].sum() > 0 else 0

    auroc_far = roc_auc_score(
        np.concatenate([np.zeros(len(test_id_w)), np.ones(ood_severity['far'].sum())]),
        np.concatenate([id_scores, far_scores])
    ) if ood_severity['far'].sum() > 0 else 0

    # Cohen's d
    d_near = cohens_d(id_scores, near_scores) if len(near_scores) > 0 else 0
    d_all = cohens_d(id_scores, ood_scores)

    # Anisotropy after whitening
    aniso = measure_anisotropy(train_w)

    whitening_results.append({
        'k': k,
        'mean_pair_sim': aniso['mean_pair_sim'],
        'top1_var_ratio': aniso['top1_var_ratio'],
        'cohens_d_near': d_near,
        'cohens_d_all': d_all,
        'auroc_all': auroc_all,
        'auroc_near': auroc_near,
        'auroc_medium': auroc_medium,
        'auroc_far': auroc_far
    })

whitening_df = pd.DataFrame(whitening_results)

# Find optimal k
best_k_near = whitening_df.loc[whitening_df['cohens_d_near'].idxmax(), 'k']
best_cohens_d = whitening_df['cohens_d_near'].max()

print(f"\n  Whitening ablation results:")
print(whitening_df.to_string(index=False))
print(f"\n  Best k for near-OOD Cohen's d: k={int(best_k_near)} (d={best_cohens_d:.4f})")

# Save whitening results
with open(RESULTS_DIR / "whitening_ablation.md", "w") as f:
    f.write("# Spectral Whitening Ablation\n\n")
    f.write("## Results by k (number of removed principal components)\n\n")
    f.write("| k | Mean Pair Sim | Top-1 Var | Cohen's d (near) | Cohen's d (all) | AUROC (all) | AUROC (near) | AUROC (med) | AUROC (far) |\n")
    f.write("|---|---------------|-----------|------------------|-----------------|-------------|--------------|-------------|-------------|\n")
    for _, row in whitening_df.iterrows():
        f.write(f"| {int(row['k'])} | {row['mean_pair_sim']:.4f} | {row['top1_var_ratio']:.4f} | "
                f"{row['cohens_d_near']:.4f} | {row['cohens_d_all']:.4f} | "
                f"{row['auroc_all']:.4f} | {row['auroc_near']:.4f} | "
                f"{row['auroc_medium']:.4f} | {row['auroc_far']:.4f} |\n")

    f.write(f"\n## Key Finding\n\n")
    f.write(f"Best k = {int(best_k_near)} achieves Cohen's d = {best_cohens_d:.4f} on near-OOD\n")
    f.write(f"Baseline (k=0) Cohen's d = {whitening_df[whitening_df['k']==0]['cohens_d_near'].values[0]:.4f}\n")

# Plot Cohen's d vs k
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.plot(whitening_df['k'], whitening_df['cohens_d_near'], 'o-', label="Near-OOD", linewidth=2, markersize=8)
ax.plot(whitening_df['k'], whitening_df['cohens_d_all'], 's--', label="All-OOD", linewidth=2, markersize=8)
ax.axhline(y=0.8, color='red', linestyle=':', label="Target (d=0.8)")
ax.set_xlabel('k (removed components)')
ax.set_ylabel("Cohen's d")
ax.set_title("Effect of Spectral Whitening on OOD Separation")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(whitening_df['k'], whitening_df['auroc_near'], 'o-', label="Near-OOD", linewidth=2, markersize=8)
ax.plot(whitening_df['k'], whitening_df['auroc_medium'], 's-', label="Medium-OOD", linewidth=2, markersize=8)
ax.plot(whitening_df['k'], whitening_df['auroc_far'], '^-', label="Far-OOD", linewidth=2, markersize=8)
ax.plot(whitening_df['k'], whitening_df['auroc_all'], 'd--', label="All-OOD", linewidth=2, markersize=8)
ax.set_xlabel('k (removed components)')
ax.set_ylabel("AUROC")
ax.set_title("AUROC by OOD Severity")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "fig_cohens_d_vs_k.png", dpi=150, bbox_inches='tight')
plt.savefig(RESULTS_DIR / "fig_auroc_by_severity.png", dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# B3: Conformal Prediction Analysis
# ============================================================
print("\n[B3] Running conformal prediction analysis...")

def split_conformal_ood_detection(train_emb, cal_emb, test_id_emb, test_ood_emb, alpha=0.1, k=20):
    """
    Split conformal prediction for OOD detection.
    Calibrate threshold on cal_emb (ID samples only), then detect OOD in test.
    """
    # Use train as reference for kNN
    cal_scores = compute_knn_scores(cal_emb, train_emb, k=k)
    test_id_scores = compute_knn_scores(test_id_emb, train_emb, k=k)
    test_ood_scores = compute_knn_scores(test_ood_emb, train_emb, k=k)

    # Conformal threshold: (1-alpha) quantile of calibration scores
    q_hat = np.quantile(cal_scores, 1 - alpha)

    # Detect OOD: score > threshold
    id_detected_as_ood = (test_id_scores > q_hat).mean()  # False positive rate
    ood_detected = (test_ood_scores > q_hat).mean()  # True positive rate

    return {
        'threshold': q_hat,
        'id_fpr': id_detected_as_ood,
        'ood_tpr': ood_detected,
        'test_id_scores': test_id_scores,
        'test_ood_scores': test_ood_scores
    }

# Split ID training data for calibration
np.random.seed(42)
n_train = len(train_id_emb)
cal_size = int(0.3 * n_train)
indices = np.random.permutation(n_train)
cal_indices = indices[:cal_size]
train_indices = indices[cal_size:]

train_for_cp = train_id_emb[train_indices]
cal_for_cp = train_id_emb[cal_indices]

# Run CP on original embeddings
cp_baseline = split_conformal_ood_detection(train_for_cp, cal_for_cp, test_id_emb, test_ood_emb, alpha=0.1)

# Run CP on best whitened embeddings
best_k = int(best_k_near)
train_w, mean, components = spectral_whitening(train_id_emb, best_k)
train_for_cp_w = train_w[train_indices]
cal_for_cp_w = train_w[cal_indices]

# Apply whitening to test
test_id_centered = test_id_emb - mean
test_ood_centered = test_ood_emb - mean
for comp in components:
    test_id_centered = test_id_centered - np.outer(test_id_centered @ comp, comp)
    test_ood_centered = test_ood_centered - np.outer(test_ood_centered @ comp, comp)
test_id_w = test_id_centered / np.linalg.norm(test_id_centered, axis=1, keepdims=True)
test_ood_w = test_ood_centered / np.linalg.norm(test_ood_centered, axis=1, keepdims=True)

cp_whitened = split_conformal_ood_detection(train_for_cp_w, cal_for_cp_w, test_id_w, test_ood_w, alpha=0.1)

# Detection rates by OOD severity
near_det_baseline = (cp_baseline['test_ood_scores'][ood_severity['near']] > cp_baseline['threshold']).mean()
near_det_whitened = (cp_whitened['test_ood_scores'][ood_severity['near']] > cp_whitened['threshold']).mean()
medium_det_baseline = (cp_baseline['test_ood_scores'][ood_severity['medium']] > cp_baseline['threshold']).mean()
medium_det_whitened = (cp_whitened['test_ood_scores'][ood_severity['medium']] > cp_whitened['threshold']).mean()
far_det_baseline = (cp_baseline['test_ood_scores'][ood_severity['far']] > cp_baseline['threshold']).mean()
far_det_whitened = (cp_whitened['test_ood_scores'][ood_severity['far']] > cp_whitened['threshold']).mean()

print(f"  Conformal Prediction Results (alpha=0.1):")
print(f"    Baseline:")
print(f"      Threshold: {cp_baseline['threshold']:.4f}")
print(f"      ID FPR: {cp_baseline['id_fpr']:.4f}")
print(f"      OOD TPR (all): {cp_baseline['ood_tpr']:.4f}")
print(f"      Near-OOD TPR: {near_det_baseline:.4f}")
print(f"    Whitened (k={best_k}):")
print(f"      Threshold: {cp_whitened['threshold']:.4f}")
print(f"      ID FPR: {cp_whitened['id_fpr']:.4f}")
print(f"      OOD TPR (all): {cp_whitened['ood_tpr']:.4f}")
print(f"      Near-OOD TPR: {near_det_whitened:.4f}")

cp_improvement = (near_det_whitened - near_det_baseline) * 100
print(f"    Near-OOD detection improvement: {cp_improvement:.1f} percentage points")

# ============================================================
# B4: Retrieval Graph Features
# ============================================================
print("\n[B4] Validating retrieval graph features...")

def compute_retrieval_features(query_emb, corpus_emb, corpus_labels, k=10):
    """Compute retrieval graph features for OOD detection"""
    index = faiss.IndexFlatIP(corpus_emb.shape[1])
    index.add(corpus_emb.astype(np.float32))

    sims, indices = index.search(query_emb.astype(np.float32), k)

    features = {
        'mean_sim': sims.mean(axis=1),
        'std_sim': sims.std(axis=1),
        'retrieval_gap': sims[:, 0] - sims[:, -1],  # top-1 vs top-k
        'label_purity': []
    }

    # Label purity: fraction of most common label in retrieved docs
    for i in range(len(query_emb)):
        retrieved_labels = corpus_labels[indices[i]]
        unique, counts = np.unique(retrieved_labels, return_counts=True)
        purity = counts.max() / k
        features['label_purity'].append(purity)

    features['label_purity'] = np.array(features['label_purity'])

    return features

# Compute features for ID and OOD test samples
id_features = compute_retrieval_features(test_id_emb, train_id_emb, train_id_labels)
ood_features = compute_retrieval_features(test_ood_emb, train_id_emb, train_id_labels)

# Test each feature as OOD score
feature_names = ['mean_sim', 'std_sim', 'retrieval_gap', 'label_purity']
feature_results = []

for feat in feature_names:
    id_vals = id_features[feat]
    ood_vals = ood_features[feat]

    # For similarity-based features, lower = more OOD
    if feat in ['mean_sim', 'label_purity']:
        scores_id = -id_vals
        scores_ood = -ood_vals
    else:
        scores_id = id_vals
        scores_ood = ood_vals

    auroc = roc_auc_score(
        np.concatenate([np.zeros(len(id_vals)), np.ones(len(ood_vals))]),
        np.concatenate([scores_id, scores_ood])
    )

    # Near-OOD AUROC
    near_scores = scores_ood[ood_severity['near']]
    auroc_near = roc_auc_score(
        np.concatenate([np.zeros(len(id_vals)), np.ones(len(near_scores))]),
        np.concatenate([scores_id, near_scores])
    ) if len(near_scores) > 0 else 0

    feature_results.append({
        'feature': feat,
        'auroc_all': auroc,
        'auroc_near': auroc_near
    })

feature_df = pd.DataFrame(feature_results)
print(f"\n  Individual feature AUROC:")
print(feature_df.to_string(index=False))

# Combine features with kNN score using logistic regression
def combine_features_with_knn(id_features, ood_features, id_knn_scores, ood_knn_scores, ood_severity):
    """Combine retrieval features with kNN score"""
    # Build feature matrix
    id_X = np.column_stack([
        id_features['mean_sim'],
        id_features['std_sim'],
        id_features['retrieval_gap'],
        id_features['label_purity'],
        id_knn_scores
    ])

    ood_X = np.column_stack([
        ood_features['mean_sim'],
        ood_features['std_sim'],
        ood_features['retrieval_gap'],
        ood_features['label_purity'],
        ood_knn_scores
    ])

    y_id = np.zeros(len(id_X))
    y_ood = np.ones(len(ood_X))

    # Split for training/testing
    np.random.seed(42)
    id_train_idx = np.random.choice(len(id_X), len(id_X)//2, replace=False)
    id_test_idx = np.setdiff1d(np.arange(len(id_X)), id_train_idx)

    ood_train_idx = np.random.choice(len(ood_X), len(ood_X)//2, replace=False)
    ood_test_idx = np.setdiff1d(np.arange(len(ood_X)), ood_train_idx)

    X_train = np.vstack([id_X[id_train_idx], ood_X[ood_train_idx]])
    y_train = np.concatenate([y_id[id_train_idx], y_ood[ood_train_idx]])

    X_test = np.vstack([id_X[id_test_idx], ood_X[ood_test_idx]])
    y_test = np.concatenate([y_id[id_test_idx], y_ood[ood_test_idx]])

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train_scaled, y_train)

    # Get probabilities
    probs = clf.predict_proba(X_test_scaled)[:, 1]

    # Compute AUROC
    auroc_combined = roc_auc_score(y_test, probs)

    # Near-OOD subset
    n_id_test = len(id_test_idx)
    ood_test_severity_near = ood_severity['near'][ood_test_idx]
    near_mask_in_test = np.concatenate([np.zeros(n_id_test, dtype=bool), ood_test_severity_near])
    id_mask_in_test = np.concatenate([np.ones(n_id_test, dtype=bool), np.zeros(len(ood_test_idx), dtype=bool)])

    if near_mask_in_test.sum() > 0:
        combined_near_mask = id_mask_in_test | near_mask_in_test
        auroc_near = roc_auc_score(y_test[combined_near_mask], probs[combined_near_mask])
    else:
        auroc_near = 0

    return auroc_combined, auroc_near

# Pure kNN baseline (using whitened embeddings for fair comparison)
knn_auroc_all = whitening_df[whitening_df['k'] == best_k]['auroc_all'].values[0]
knn_auroc_near = whitening_df[whitening_df['k'] == best_k]['auroc_near'].values[0]

# Combined features AUROC
# Recompute features on whitened embeddings
id_features_w = compute_retrieval_features(test_id_w, train_w, train_id_labels)
ood_features_w = compute_retrieval_features(test_ood_w, train_w, train_id_labels)

id_knn_scores_w = compute_knn_scores(test_id_w, train_w)
ood_knn_scores_w = compute_knn_scores(test_ood_w, train_w)

combined_auroc_all, combined_auroc_near = combine_features_with_knn(
    id_features_w, ood_features_w, id_knn_scores_w, ood_knn_scores_w, ood_severity
)

print(f"\n  Feature combination results:")
print(f"    Pure kNN (whitened, k={best_k}):")
print(f"      All-OOD AUROC: {knn_auroc_all:.4f}")
print(f"      Near-OOD AUROC: {knn_auroc_near:.4f}")
print(f"    kNN + Graph Features (LR):")
print(f"      All-OOD AUROC: {combined_auroc_all:.4f}")
print(f"      Near-OOD AUROC: {combined_auroc_near:.4f}")
print(f"    Near-OOD improvement: {(combined_auroc_near - knn_auroc_near) * 100:.2f}%")

# Plot retrieval graph analysis
fig, ax = plt.subplots(figsize=(8, 6))
methods = ['kNN only', 'kNN + Graph\nFeatures']
aurocs_near = [knn_auroc_near, combined_auroc_near]
aurocs_all = [knn_auroc_all, combined_auroc_all]

x = np.arange(len(methods))
width = 0.35
ax.bar(x - width/2, aurocs_near, width, label='Near-OOD', color='coral')
ax.bar(x + width/2, aurocs_all, width, label='All-OOD', color='steelblue')
ax.set_ylabel('AUROC')
ax.set_title('Effect of Retrieval Graph Features')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()
ax.set_ylim(0.5, 1.0)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(RESULTS_DIR / "fig_retrieval_graph.png", dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# B5: Banking77 Cross-Dataset Validation
# ============================================================
print("\n[B5] Running Banking77 cross-dataset validation...")

def load_banking77():
    """Load Banking77 dataset with OOD split"""
    ds = load_dataset("banking77")

    # Use first 50 intents as ID, last 27 as OOD
    id_intents = set(range(50))
    ood_intents = set(range(50, 77))

    data = {
        'train': {'texts': [], 'labels': [], 'is_ood': []},
        'test': {'texts': [], 'labels': [], 'is_ood': []}
    }

    for item in ds['train']:
        data['train']['texts'].append(item['text'])
        data['train']['labels'].append(item['label'])
        data['train']['is_ood'].append(item['label'] in ood_intents)

    for item in ds['test']:
        data['test']['texts'].append(item['text'])
        data['test']['labels'].append(item['label'])
        data['test']['is_ood'].append(item['label'] in ood_intents)

    return data

banking_data = load_banking77()
print(f"  Banking77 loaded:")
print(f"    Train: {len(banking_data['train']['texts'])} samples")
print(f"    Test: {len(banking_data['test']['texts'])} samples")

# Extract embeddings
banking_emb = {}
for split in ['train', 'test']:
    banking_emb[split] = extract_embeddings(banking_data[split]['texts'])

# Split ID/OOD
banking_train_is_ood = np.array(banking_data['train']['is_ood'])
banking_test_is_ood = np.array(banking_data['test']['is_ood'])
banking_train_labels = np.array(banking_data['train']['labels'])

banking_train_id = banking_emb['train'][~banking_train_is_ood]
banking_train_id_labels = banking_train_labels[~banking_train_is_ood]
banking_test_id = banking_emb['test'][~banking_test_is_ood]
banking_test_ood = banking_emb['test'][banking_test_is_ood]

print(f"    Train ID: {len(banking_train_id)}")
print(f"    Test ID: {len(banking_test_id)}")
print(f"    Test OOD: {len(banking_test_ood)}")

# Compute OOD severity for Banking77
banking_severity = compute_ood_severity(banking_test_ood, banking_train_id)
print(f"    Near-OOD: {banking_severity['near'].sum()}")
print(f"    Medium-OOD: {banking_severity['medium'].sum()}")
print(f"    Far-OOD: {banking_severity['far'].sum()}")

# Baseline detection
banking_id_scores = compute_knn_scores(banking_test_id, banking_train_id)
banking_ood_scores = compute_knn_scores(banking_test_ood, banking_train_id)

banking_baseline_d = cohens_d(banking_id_scores, banking_ood_scores[banking_severity['near']])
banking_baseline_auroc = roc_auc_score(
    np.concatenate([np.zeros(len(banking_test_id)), np.ones(banking_severity['near'].sum())]),
    np.concatenate([banking_id_scores, banking_ood_scores[banking_severity['near']]])
) if banking_severity['near'].sum() > 0 else 0

print(f"\n  Banking77 baseline (near-OOD):")
print(f"    Cohen's d: {banking_baseline_d:.4f}")
print(f"    AUROC: {banking_baseline_auroc:.4f}")

# Test whitening on Banking77
banking_whitening_results = []
for k in [0, 1, 2, 3, 5, 10]:
    if k == 0:
        train_w = banking_train_id
        test_id_w = banking_test_id
        test_ood_w = banking_test_ood
    else:
        train_w, mean, components = spectral_whitening(banking_train_id, k)
        test_id_centered = banking_test_id - mean
        test_ood_centered = banking_test_ood - mean
        for comp in components:
            test_id_centered = test_id_centered - np.outer(test_id_centered @ comp, comp)
            test_ood_centered = test_ood_centered - np.outer(test_ood_centered @ comp, comp)
        test_id_w = test_id_centered / np.linalg.norm(test_id_centered, axis=1, keepdims=True)
        test_ood_w = test_ood_centered / np.linalg.norm(test_ood_centered, axis=1, keepdims=True)

    id_scores = compute_knn_scores(test_id_w, train_w)
    ood_scores = compute_knn_scores(test_ood_w, train_w)

    near_scores = ood_scores[banking_severity['near']]
    d_near = cohens_d(id_scores, near_scores) if len(near_scores) > 0 else 0

    auroc_near = roc_auc_score(
        np.concatenate([np.zeros(len(test_id_w)), np.ones(len(near_scores))]),
        np.concatenate([id_scores, near_scores])
    ) if len(near_scores) > 0 else 0

    banking_whitening_results.append({
        'k': k,
        'cohens_d_near': d_near,
        'auroc_near': auroc_near
    })

banking_whitening_df = pd.DataFrame(banking_whitening_results)
best_k_banking = banking_whitening_df.loc[banking_whitening_df['cohens_d_near'].idxmax(), 'k']
best_d_banking = banking_whitening_df['cohens_d_near'].max()

print(f"\n  Banking77 whitening results:")
print(banking_whitening_df.to_string(index=False))
print(f"  Best k={int(best_k_banking)} achieves Cohen's d={best_d_banking:.4f}")

# ============================================================
# B6: Kill-Switch Determination
# ============================================================
print("\n" + "=" * 60)
print("RAG ROUTE KILL-SWITCH DETERMINATION")
print("=" * 60)

# Collect all judgments
judgments = {}

# [1] Whitening significantly improves near-OOD Cohen's d?
baseline_d = whitening_df[whitening_df['k'] == 0]['cohens_d_near'].values[0]
best_d = whitening_df['cohens_d_near'].max()

print(f"\n[1] Spectral whitening improves near-OOD separation?")
print(f"    Original Cohen's d (near-OOD): {baseline_d:.4f}")
print(f"    Best whitened Cohen's d (k={int(best_k_near)}): {best_d:.4f}")
print(f"    Improvement: {best_d - baseline_d:.4f}")

judgments['whitening_improves_d'] = best_d > 0.8
print(f"    JUDGMENT (d > 0.8): {'YES' if judgments['whitening_improves_d'] else 'NO'}")

# [2] CP detection rate improves?
print(f"\n[2] Conformal prediction detection rate improves?")
print(f"    Original near-OOD detection rate: {near_det_baseline:.4f}")
print(f"    Whitened near-OOD detection rate: {near_det_whitened:.4f}")
print(f"    Improvement: {cp_improvement:.1f} percentage points")

judgments['cp_improves'] = cp_improvement >= 10.0
print(f"    JUDGMENT (≥10 pp improvement): {'YES' if judgments['cp_improves'] else 'NO'}")

# [3] Retrieval graph features add value?
graph_improvement = (combined_auroc_near - knn_auroc_near) * 100

print(f"\n[3] Retrieval graph features add incremental value?")
print(f"    Pure kNN near-OOD AUROC: {knn_auroc_near:.4f}")
print(f"    kNN + Graph features AUROC: {combined_auroc_near:.4f}")
print(f"    Improvement: {graph_improvement:.2f}%")

judgments['graph_adds_value'] = graph_improvement >= 2.0
print(f"    JUDGMENT (≥2% improvement): {'YES' if judgments['graph_adds_value'] else 'NO'}")

# [4] Cross-dataset generalization?
clinc_passes = best_d > 0.8
banking_passes = best_d_banking > 0.8

print(f"\n[4] Cross-dataset generalization?")
print(f"    CLINC150 best Cohen's d: {best_d:.4f} ({'PASS' if clinc_passes else 'FAIL'})")
print(f"    Banking77 best Cohen's d: {best_d_banking:.4f} ({'PASS' if banking_passes else 'FAIL'})")

judgments['generalizes'] = clinc_passes and banking_passes
print(f"    JUDGMENT (both datasets pass): {'YES' if judgments['generalizes'] else 'NO'}")

# Final S2 determination
criteria_passed = sum([
    judgments['whitening_improves_d'],
    judgments['cp_improves'],
    judgments['graph_adds_value'],
    judgments['generalizes']
])

s2_pass = criteria_passed >= 3

print("\n" + "=" * 60)
print(f"S2 FINAL DETERMINATION: {'YES' if s2_pass else 'NO'}")
print(f"Criteria passed: {criteria_passed}/4 (need ≥3)")
print("=" * 60)

if s2_pass:
    print("→ RAG route methodology is VIABLE")
    print("→ Proceed with RAG direction + TextOOD-Bench validation")
else:
    print("→ RAG route does NOT meet criteria")
    print("→ Recommend: Retreat to P5 (Geometric Adaptive CP + TextOOD-Bench baseline)")

    failures = []
    if not judgments['whitening_improves_d']:
        failures.append(f"Whitening Cohen's d = {best_d:.4f} < 0.8")
    if not judgments['cp_improves']:
        failures.append(f"CP improvement = {cp_improvement:.1f}pp < 10pp")
    if not judgments['graph_adds_value']:
        failures.append(f"Graph feature improvement = {graph_improvement:.2f}% < 2%")
    if not judgments['generalizes']:
        failures.append("Cross-dataset generalization failed")
    print(f"Failure reasons: {'; '.join(failures)}")

# Store results for report
S2_RESULT = s2_pass
S2_DETAILS = {
    'whitening_improves_d': judgments['whitening_improves_d'],
    'cp_improves': judgments['cp_improves'],
    'graph_adds_value': judgments['graph_adds_value'],
    'generalizes': judgments['generalizes'],
    'baseline_cohens_d': baseline_d,
    'best_cohens_d': best_d,
    'best_k': int(best_k_near),
    'cp_improvement_pp': cp_improvement,
    'graph_improvement_pct': graph_improvement,
    'near_det_baseline': near_det_baseline,
    'near_det_whitened': near_det_whitened,
    'knn_auroc_near': knn_auroc_near,
    'combined_auroc_near': combined_auroc_near,
    'banking_best_d': best_d_banking,
    'criteria_passed': criteria_passed
}

# Save S2 result
import pickle
with open(RESULTS_DIR / "s2_result.pkl", "wb") as f:
    pickle.dump({'S2_RESULT': S2_RESULT, 'S2_DETAILS': S2_DETAILS}, f)

print("\n[Part B Complete]")
print(f"Results saved to: {RESULTS_DIR}")
