#!/usr/bin/env python3
"""
Part B: RAG Route Real Data Validation
RW3 Kill-Switch Determination - v2 (Real Data Only)

Covers: B1-B7 (Data loading through final determination)
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
np.random.seed(42)

OUTPUT_DIR = '/home/user/OOD-project/rw3_pilot_v2/results'
EMBED_DIR = os.path.join(OUTPUT_DIR, 'embeddings')
os.makedirs(EMBED_DIR, exist_ok=True)

# ============================================
# B1: Data Loading
# ============================================

def load_clinc150():
    """Load CLINC150 dataset"""
    print("\n[B1] Loading CLINC150 dataset...")
    from datasets import load_dataset

    # Try different loading strategies
    try:
        ds = load_dataset("clinc_oos", "plus", trust_remote_code=True)
        print("  Loaded clinc_oos/plus configuration")
    except Exception as e1:
        try:
            ds = load_dataset("clinc_oos", "small", trust_remote_code=True)
            print("  Loaded clinc_oos/small configuration")
        except Exception as e2:
            try:
                ds = load_dataset("clinc_oos", trust_remote_code=True)
                print("  Loaded clinc_oos default configuration")
            except Exception as e3:
                print(f"  ERROR: Failed to load CLINC150: {e1}, {e2}, {e3}")
                return None

    # Process the dataset
    data = {
        'train': {'texts': [], 'labels': [], 'is_ood': []},
        'validation': {'texts': [], 'labels': [], 'is_ood': []},
        'test': {'texts': [], 'labels': [], 'is_ood': []}
    }

    for split in ['train', 'validation', 'test']:
        if split not in ds:
            continue
        for item in ds[split]:
            text = item['text']
            intent = item['intent']
            # OOD is labeled as intent 42 or 'oos' in CLINC150
            is_ood = (intent == 42) or (str(intent).lower() == 'oos')

            data[split]['texts'].append(text)
            data[split]['labels'].append(intent)
            data[split]['is_ood'].append(is_ood)

    # Convert to numpy
    for split in data:
        for key in data[split]:
            data[split][key] = np.array(data[split][key])

    # Statistics
    total_train = len(data['train']['texts'])
    total_test = len(data['test']['texts'])
    n_ood_train = data['train']['is_ood'].sum()
    n_ood_test = data['test']['is_ood'].sum()
    n_classes = len(set(data['train']['labels'][~data['train']['is_ood']]))

    print(f"  Train: {total_train} samples, {n_ood_train} OOD ({100*n_ood_train/total_train:.1f}%)")
    print(f"  Test: {total_test} samples, {n_ood_test} OOD ({100*n_ood_test/total_test:.1f}%)")
    print(f"  ID classes: {n_classes}")

    return data


def load_banking77():
    """Load Banking77 dataset with near-OOD split"""
    print("\n[B1] Loading Banking77 dataset...")
    from datasets import load_dataset

    try:
        ds = load_dataset("PolyAI/banking77", trust_remote_code=True)
        print("  Loaded PolyAI/banking77")
    except Exception as e1:
        try:
            ds = load_dataset("banking77", trust_remote_code=True)
            print("  Loaded banking77")
        except Exception as e2:
            print(f"  ERROR: Failed to load Banking77: {e1}, {e2}")
            return None

    # Banking77: 77 intent classes
    # Near-OOD setup: first 50 intents = ID, last 27 intents = OOD
    ID_CLASSES = set(range(50))
    OOD_CLASSES = set(range(50, 77))

    data = {
        'train': {'texts': [], 'labels': [], 'is_ood': []},
        'test': {'texts': [], 'labels': [], 'is_ood': []}
    }

    for split in ['train', 'test']:
        if split not in ds:
            continue
        for item in ds[split]:
            text = item['text']
            label = item['label']
            is_ood = label in OOD_CLASSES

            data[split]['texts'].append(text)
            data[split]['labels'].append(label)
            data[split]['is_ood'].append(is_ood)

    # Convert to numpy
    for split in data:
        for key in data[split]:
            data[split][key] = np.array(data[split][key])

    # Statistics
    total_train = len(data['train']['texts'])
    total_test = len(data['test']['texts'])
    n_ood_train = data['train']['is_ood'].sum()
    n_ood_test = data['test']['is_ood'].sum()

    print(f"  Train: {total_train} samples, {n_ood_train} OOD ({100*n_ood_train/total_train:.1f}%)")
    print(f"  Test: {total_test} samples, {n_ood_test} OOD ({100*n_ood_test/total_test:.1f}%)")
    print(f"  ID classes: 50, OOD classes: 27 (near-OOD setup)")

    return data


# ============================================
# B2: Embedding Extraction
# ============================================

def load_embedding_model(model_name):
    """Load embedding model"""
    from sentence_transformers import SentenceTransformer

    print(f"  Loading model: {model_name}")
    start = time.time()
    try:
        model = SentenceTransformer(model_name)
        print(f"    Loaded in {time.time()-start:.1f}s, dim={model.get_sentence_embedding_dimension()}")
        return model
    except Exception as e:
        print(f"    ERROR loading {model_name}: {e}")
        return None


def extract_embeddings(texts, model, batch_size=64):
    """Extract embeddings for a list of texts"""
    start = time.time()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    # L2 normalize
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    print(f"    Extracted {len(embeddings)} embeddings in {time.time()-start:.1f}s")
    return embeddings


def get_or_compute_embeddings(data, dataset_name, model, model_name):
    """Get embeddings from cache or compute"""
    model_short = model_name.split('/')[-1].replace('-', '_')
    cache_path = os.path.join(EMBED_DIR, f'{dataset_name}_{model_short}.npz')

    if os.path.exists(cache_path):
        print(f"  Loading cached embeddings from {cache_path}")
        cached = np.load(cache_path)
        return {split: cached[split] for split in cached.files}

    print(f"  Computing embeddings for {dataset_name} with {model_name}...")
    embeddings = {}
    for split in data:
        if len(data[split]['texts']) == 0:
            continue
        print(f"    {split}:")
        embeddings[split] = extract_embeddings(data[split]['texts'].tolist(), model)

    # Save cache
    np.savez(cache_path, **embeddings)
    print(f"  Saved embeddings to {cache_path}")

    return embeddings


# ============================================
# B3: Anisotropy Baseline Measurement
# ============================================

def measure_anisotropy(embeddings, n_pairs=10000):
    """Measure embedding space anisotropy"""
    n = len(embeddings)

    # Compute covariance eigenvalues
    centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)[::-1]  # descending

    total_var = eigenvalues.sum()
    top1_ratio = eigenvalues[0] / total_var
    top5_ratio = eigenvalues[:5].sum() / total_var
    top10_ratio = eigenvalues[:10].sum() / total_var

    # Random pair cosine similarity
    idx1 = np.random.choice(n, min(n_pairs, n), replace=True)
    idx2 = np.random.choice(n, min(n_pairs, n), replace=True)

    cosines = np.sum(embeddings[idx1] * embeddings[idx2], axis=1)
    mean_cosine = cosines.mean()

    is_anisotropic = mean_cosine > 0.3

    return {
        'top1_var_ratio': top1_ratio,
        'top5_var_ratio': top5_ratio,
        'top10_var_ratio': top10_ratio,
        'mean_pair_cosine': mean_cosine,
        'is_anisotropic': is_anisotropic,
        'eigenvalues': eigenvalues
    }


def compute_ood_severity_groups(id_embeddings, ood_embeddings, id_labels=None):
    """
    Compute OOD severity groups based on similarity to ID data.
    Uses percentiles for robustness across different models.
    """
    # Compute similarity of each OOD to nearest ID
    # Using cosine similarity (embeddings are L2-normalized)
    similarities = ood_embeddings @ id_embeddings.T
    max_similarities = similarities.max(axis=1)

    # Define groups by percentiles
    p25 = np.percentile(max_similarities, 25)
    p75 = np.percentile(max_similarities, 75)

    groups = {
        'near': max_similarities >= p75,  # Top 25% most similar to ID
        'medium': (max_similarities >= p25) & (max_similarities < p75),
        'far': max_similarities < p25  # Bottom 25% least similar to ID
    }

    stats = {
        'near_count': groups['near'].sum(),
        'medium_count': groups['medium'].sum(),
        'far_count': groups['far'].sum(),
        'near_sim_range': (max_similarities[groups['near']].min(), max_similarities[groups['near']].max()) if groups['near'].any() else (0, 0),
        'medium_sim_range': (max_similarities[groups['medium']].min(), max_similarities[groups['medium']].max()) if groups['medium'].any() else (0, 0),
        'far_sim_range': (max_similarities[groups['far']].min(), max_similarities[groups['far']].max()) if groups['far'].any() else (0, 0),
    }

    return groups, max_similarities, stats


def compute_baseline_ood_scores(id_train_emb, test_emb, k=20):
    """Compute baseline OOD scores"""
    # kNN distance
    similarities = test_emb @ id_train_emb.T
    topk_sims = np.sort(similarities, axis=1)[:, -k:]
    knn_score = -topk_sims.mean(axis=1)  # Negative similarity = distance

    # Centroid distance
    centroid = id_train_emb.mean(axis=0)
    centroid_score = 1 - test_emb @ centroid

    # Mahalanobis distance (simplified)
    try:
        centered = id_train_emb - id_train_emb.mean(axis=0)
        cov = np.cov(centered.T)
        cov_inv = np.linalg.pinv(cov + 1e-6 * np.eye(cov.shape[0]))
        test_centered = test_emb - id_train_emb.mean(axis=0)
        mahal_score = np.sqrt(np.sum(test_centered @ cov_inv * test_centered, axis=1))
    except:
        mahal_score = np.zeros(len(test_emb))

    return {
        'knn': knn_score,
        'centroid': centroid_score,
        'mahalanobis': mahal_score
    }


def compute_auroc_and_cohens_d(id_scores, ood_scores, ood_groups=None):
    """Compute AUROC and Cohen's d for OOD detection"""
    # Full AUROC
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])

    try:
        full_auroc = roc_auc_score(labels, scores)
    except:
        full_auroc = 0.5

    # Cohen's d
    mean_diff = ood_scores.mean() - id_scores.mean()
    pooled_std = np.sqrt((id_scores.std()**2 + ood_scores.std()**2) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

    result = {
        'full_auroc': full_auroc,
        'cohens_d': cohens_d,
    }

    # Group-specific metrics
    if ood_groups is not None:
        for group_name, mask in ood_groups.items():
            if mask.sum() > 0:
                group_ood_scores = ood_scores[mask]
                group_labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(group_ood_scores))])
                group_scores = np.concatenate([id_scores, group_ood_scores])
                try:
                    result[f'{group_name}_auroc'] = roc_auc_score(group_labels, group_scores)
                except:
                    result[f'{group_name}_auroc'] = 0.5

                # Cohen's d for this group
                mean_diff = group_ood_scores.mean() - id_scores.mean()
                pooled_std = np.sqrt((id_scores.std()**2 + group_ood_scores.std()**2) / 2)
                result[f'{group_name}_cohens_d'] = mean_diff / pooled_std if pooled_std > 0 else 0

    return result


# ============================================
# B4: Spectral Whitening
# ============================================

def spectral_whitening(embeddings, directions, k):
    """Apply all-but-the-top: remove top-k principal components"""
    if k == 0:
        return embeddings.copy()

    centered = embeddings - embeddings.mean(axis=0)

    for i in range(min(k, len(directions))):
        component = directions[i]
        centered = centered - np.outer(centered @ component, component)

    # Re-normalize
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms[norms == 0] = 1
    centered = centered / norms

    return centered


def compute_pca_directions(id_train_emb):
    """Compute PCA directions on ID training data only"""
    centered = id_train_emb - id_train_emb.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    return Vt  # Principal components


def run_whitening_ablation(id_train_emb, id_test_emb, ood_emb, ood_groups, k_values=[0, 1, 2, 3, 5, 10, 20]):
    """Run whitening ablation experiment"""
    # Compute PCA directions on ID training data
    pca_directions = compute_pca_directions(id_train_emb)
    id_train_mean = id_train_emb.mean(axis=0)

    results = []

    for k in k_values:
        # Apply whitening
        id_train_white = spectral_whitening(id_train_emb, pca_directions, k)
        id_test_white = spectral_whitening(id_test_emb, pca_directions, k)
        ood_white = spectral_whitening(ood_emb, pca_directions, k)

        # Measure anisotropy
        aniso = measure_anisotropy(id_train_white)

        # Compute OOD scores
        scores = compute_baseline_ood_scores(id_train_white,
                                             np.vstack([id_test_white, ood_white]), k=20)

        n_id_test = len(id_test_white)
        id_scores = scores['knn'][:n_id_test]
        ood_scores = scores['knn'][n_id_test:]

        # Compute metrics
        metrics = compute_auroc_and_cohens_d(id_scores, ood_scores, ood_groups)

        results.append({
            'k': k,
            'mean_pair_cosine': aniso['mean_pair_cosine'],
            **metrics
        })

    return pd.DataFrame(results)


# ============================================
# B5: Conformal Prediction Detection
# ============================================

def conformal_prediction_detection(id_train_emb, id_cal_emb, id_test_emb, ood_emb,
                                   ood_groups, alpha_values=[0.05, 0.10, 0.20], k=20):
    """Run conformal prediction OOD detection"""
    # Compute non-conformity scores (kNN distance)
    def compute_knn_scores(query_emb, ref_emb, k):
        sims = query_emb @ ref_emb.T
        topk_sims = np.sort(sims, axis=1)[:, -k:]
        return -topk_sims.mean(axis=1)

    # Calibration scores
    cal_scores = compute_knn_scores(id_cal_emb, id_train_emb, k)

    # Test scores
    id_test_scores = compute_knn_scores(id_test_emb, id_train_emb, k)
    ood_scores = compute_knn_scores(ood_emb, id_train_emb, k)

    results = []

    for alpha in alpha_values:
        # Compute threshold at (1-alpha) quantile
        q_hat = np.quantile(cal_scores, 1 - alpha)

        # Detection rates
        id_fpr = (id_test_scores > q_hat).mean()  # False positive rate

        row = {
            'alpha': alpha,
            'q_hat': q_hat,
            'id_fpr': id_fpr,
        }

        # Per-group detection rates
        for group_name, mask in ood_groups.items():
            if mask.sum() > 0:
                detection_rate = (ood_scores[mask] > q_hat).mean()
                row[f'{group_name}_detection'] = detection_rate

        # Full OOD detection rate
        row['ood_detection'] = (ood_scores > q_hat).mean()

        results.append(row)

    return pd.DataFrame(results)


# ============================================
# B6: Retrieval Graph Features
# ============================================

def compute_retrieval_graph_features(query_emb, corpus_emb, corpus_labels, k=10):
    """Compute retrieval graph features for each query"""
    import faiss

    # Build FAISS index
    dim = corpus_emb.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
    index.add(corpus_emb.astype(np.float32))

    # Search
    sims, indices = index.search(query_emb.astype(np.float32), k)

    features = {
        'mean_sim': sims.mean(axis=1),
        'std_sim': sims.std(axis=1),
        'retrieval_gap': sims[:, 0] - sims[:, -1],  # top-1 - top-k
        'sim_drop_rate': (sims[:, 0] - sims[:, 4]) / 4 if k >= 5 else np.zeros(len(query_emb)),
    }

    # Label purity
    retrieved_labels = corpus_labels[indices]
    purity = np.zeros(len(query_emb))
    for i in range(len(query_emb)):
        unique, counts = np.unique(retrieved_labels[i], return_counts=True)
        purity[i] = counts.max() / k
    features['label_purity'] = purity

    return features


def evaluate_graph_features(id_train_emb, id_train_labels, id_cal_emb, id_test_emb,
                            ood_emb, ood_groups, k=10):
    """Evaluate graph features for OOD detection"""
    # Compute features for calibration, test, and OOD
    cal_features = compute_retrieval_graph_features(id_cal_emb, id_train_emb, id_train_labels, k)
    test_features = compute_retrieval_graph_features(id_test_emb, id_train_emb, id_train_labels, k)
    ood_features = compute_retrieval_graph_features(ood_emb, id_train_emb, id_train_labels, k)

    results = {}

    # Individual feature AUROCs
    for feat_name in cal_features:
        id_scores = test_features[feat_name]
        ood_scores = ood_features[feat_name]

        # For mean_sim and label_purity, lower = more OOD, so negate
        if feat_name in ['mean_sim', 'label_purity']:
            id_scores = -id_scores
            ood_scores = -ood_scores

        metrics = compute_auroc_and_cohens_d(id_scores, ood_scores, ood_groups)
        results[feat_name] = metrics

    # kNN baseline (from B3)
    knn_scores = compute_baseline_ood_scores(id_train_emb,
                                             np.vstack([id_test_emb, ood_emb]), k=20)['knn']
    n_id_test = len(id_test_emb)
    knn_id = knn_scores[:n_id_test]
    knn_ood = knn_scores[n_id_test:]
    knn_metrics = compute_auroc_and_cohens_d(knn_id, knn_ood, ood_groups)
    results['knn_only'] = knn_metrics

    # Combined: kNN + all graph features (logistic regression)
    # Train on calibration set
    cal_knn = compute_baseline_ood_scores(id_train_emb, id_cal_emb, k=20)['knn']

    # Need OOD samples for training - use a small holdout
    n_ood_train = min(100, len(ood_emb) // 2)
    ood_train_idx = np.random.choice(len(ood_emb), n_ood_train, replace=False)
    ood_test_mask = np.ones(len(ood_emb), dtype=bool)
    ood_test_mask[ood_train_idx] = False

    ood_train_features = compute_retrieval_graph_features(ood_emb[ood_train_idx],
                                                          id_train_emb, id_train_labels, k)
    ood_train_knn = compute_baseline_ood_scores(id_train_emb, ood_emb[ood_train_idx], k=20)['knn']

    # Build training features
    X_id_train = np.column_stack([cal_knn] + [cal_features[f] for f in cal_features])
    X_ood_train = np.column_stack([ood_train_knn] + [ood_train_features[f] for f in ood_train_features])

    X_train = np.vstack([X_id_train, X_ood_train])
    y_train = np.concatenate([np.zeros(len(X_id_train)), np.ones(len(X_ood_train))])

    # Scale and train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)

    # Test features
    test_knn = compute_baseline_ood_scores(id_train_emb, id_test_emb, k=20)['knn']
    X_id_test = np.column_stack([test_knn] + [test_features[f] for f in test_features])

    ood_test_knn = compute_baseline_ood_scores(id_train_emb, ood_emb[ood_test_mask], k=20)['knn']
    ood_test_features_subset = {f: ood_features[f][ood_test_mask] for f in ood_features}
    X_ood_test = np.column_stack([ood_test_knn] + [ood_test_features_subset[f] for f in ood_test_features_subset])

    X_test = np.vstack([X_id_test, X_ood_test])
    X_test_scaled = scaler.transform(X_test)

    combined_scores = clf.predict_proba(X_test_scaled)[:, 1]

    n_id = len(X_id_test)
    combined_id = combined_scores[:n_id]
    combined_ood = combined_scores[n_id:]

    # Adjust groups for test subset
    ood_groups_test = {name: mask[ood_test_mask] for name, mask in ood_groups.items()}

    combined_metrics = compute_auroc_and_cohens_d(combined_id, combined_ood, ood_groups_test)
    results['knn_plus_graph'] = combined_metrics

    # kNN + label_purity only
    X_id_simple = np.column_stack([test_knn, test_features['label_purity']])
    X_ood_simple = np.column_stack([ood_test_knn, ood_test_features_subset['label_purity']])

    X_simple = np.vstack([X_id_simple, X_ood_simple])
    X_simple_scaled = StandardScaler().fit_transform(X_simple)

    clf_simple = LogisticRegression(max_iter=1000, random_state=42)
    X_train_simple = np.column_stack([cal_knn, cal_features['label_purity']])
    X_train_ood_simple = np.column_stack([ood_train_knn, ood_train_features['label_purity']])
    X_train_all_simple = np.vstack([X_train_simple, X_train_ood_simple])
    X_train_scaled_simple = StandardScaler().fit_transform(X_train_all_simple)
    clf_simple.fit(X_train_scaled_simple, y_train)

    simple_scores = clf_simple.predict_proba(X_simple_scaled)[:, 1]
    simple_id = simple_scores[:n_id]
    simple_ood = simple_scores[n_id:]

    simple_metrics = compute_auroc_and_cohens_d(simple_id, simple_ood, ood_groups_test)
    results['knn_plus_purity'] = simple_metrics

    return results


# ============================================
# B7: RAG Route Kill-Switch Determination
# ============================================

def run_full_experiment(data, dataset_name, model, model_name):
    """Run full experiment for one dataset + model combination"""
    print(f"\n{'='*60}")
    print(f"Experiment: {dataset_name} + {model_name}")
    print(f"{'='*60}")

    # Get embeddings
    embeddings = get_or_compute_embeddings(data, dataset_name, model, model_name)

    # Split ID data: 60% train, 20% cal, 20% test
    train_emb = embeddings['train']
    train_labels = data['train']['labels']
    train_is_ood = data['train']['is_ood']

    # Separate ID and OOD from train
    id_train_mask = ~train_is_ood
    id_train_emb = train_emb[id_train_mask]
    id_train_labels = train_labels[id_train_mask]

    # For test split
    test_emb = embeddings['test']
    test_is_ood = data['test']['is_ood']

    id_test_mask = ~test_is_ood
    ood_test_mask = test_is_ood

    id_test_emb = test_emb[id_test_mask]
    ood_test_emb = test_emb[ood_test_mask]

    # Split ID train into train (60%), cal (20%), test-id (from test set)
    n_id_train = len(id_train_emb)
    idx = np.random.permutation(n_id_train)

    n_train = int(0.75 * n_id_train)
    n_cal = n_id_train - n_train

    actual_train_idx = idx[:n_train]
    cal_idx = idx[n_train:]

    id_train_emb_final = id_train_emb[actual_train_idx]
    id_train_labels_final = id_train_labels[actual_train_idx]
    id_cal_emb = id_train_emb[cal_idx]

    print(f"\n  ID train: {len(id_train_emb_final)}, ID cal: {len(id_cal_emb)}, ID test: {len(id_test_emb)}")
    print(f"  OOD test: {len(ood_test_emb)}")

    # B3: Anisotropy
    print("\n[B3] Measuring anisotropy baseline...")
    aniso = measure_anisotropy(id_train_emb_final)
    print(f"  Top-1 variance ratio: {aniso['top1_var_ratio']:.3f}")
    print(f"  Mean pair cosine: {aniso['mean_pair_cosine']:.3f}")
    print(f"  Anisotropic: {aniso['is_anisotropic']}")

    # OOD severity groups
    print("\n[B3] Computing OOD severity groups...")
    ood_groups, ood_sims, group_stats = compute_ood_severity_groups(id_train_emb_final, ood_test_emb)
    print(f"  Near-OOD: {group_stats['near_count']} (sim range: {group_stats['near_sim_range']})")
    print(f"  Medium-OOD: {group_stats['medium_count']} (sim range: {group_stats['medium_sim_range']})")
    print(f"  Far-OOD: {group_stats['far_count']} (sim range: {group_stats['far_sim_range']})")

    # Baseline OOD detection
    print("\n[B3] Computing baseline OOD scores...")
    baseline_scores = compute_baseline_ood_scores(id_train_emb_final,
                                                  np.vstack([id_test_emb, ood_test_emb]), k=20)

    baseline_metrics = {}
    for score_name, scores in baseline_scores.items():
        n_id = len(id_test_emb)
        id_scores = scores[:n_id]
        ood_scores = scores[n_id:]
        metrics = compute_auroc_and_cohens_d(id_scores, ood_scores, ood_groups)
        baseline_metrics[score_name] = metrics
        print(f"  {score_name}: full AUROC={metrics['full_auroc']:.3f}, near AUROC={metrics.get('near_auroc', 'N/A'):.3f}, Cohen's d={metrics['cohens_d']:.2f}")

    # B4: Whitening ablation
    print("\n[B4] Running whitening ablation...")
    whitening_results = run_whitening_ablation(
        id_train_emb_final, id_test_emb, ood_test_emb, ood_groups,
        k_values=[0, 1, 2, 3, 5, 10, 20]
    )
    print(whitening_results[['k', 'mean_pair_cosine', 'full_auroc', 'near_auroc', 'cohens_d', 'near_cohens_d']])

    # Find best k
    best_idx = whitening_results['near_cohens_d'].idxmax()
    best_k = int(whitening_results.loc[best_idx, 'k'])
    best_near_cohens_d = whitening_results.loc[best_idx, 'near_cohens_d']
    original_near_cohens_d = whitening_results[whitening_results['k'] == 0]['near_cohens_d'].values[0]

    print(f"\n  Best k* = {best_k} (near Cohen's d: {original_near_cohens_d:.2f} -> {best_near_cohens_d:.2f})")

    # B5: Conformal prediction
    print("\n[B5] Running conformal prediction detection...")

    # Compute for original (k=0)
    cp_original = conformal_prediction_detection(
        id_train_emb_final, id_cal_emb, id_test_emb, ood_test_emb, ood_groups
    )
    print("  Original embeddings:")
    print(cp_original[['alpha', 'id_fpr', 'near_detection', 'medium_detection', 'far_detection']])

    # Compute for whitened (k=best_k)
    pca_dirs = compute_pca_directions(id_train_emb_final)
    id_train_white = spectral_whitening(id_train_emb_final, pca_dirs, best_k)
    id_cal_white = spectral_whitening(id_cal_emb, pca_dirs, best_k)
    id_test_white = spectral_whitening(id_test_emb, pca_dirs, best_k)
    ood_white = spectral_whitening(ood_test_emb, pca_dirs, best_k)

    cp_whitened = conformal_prediction_detection(
        id_train_white, id_cal_white, id_test_white, ood_white, ood_groups
    )
    print(f"\n  Whitened embeddings (k={best_k}):")
    print(cp_whitened[['alpha', 'id_fpr', 'near_detection', 'medium_detection', 'far_detection']])

    # Detection rate improvement at alpha=0.10
    orig_near_det = cp_original[cp_original['alpha'] == 0.10]['near_detection'].values[0]
    white_near_det = cp_whitened[cp_whitened['alpha'] == 0.10]['near_detection'].values[0]
    cp_improvement = (white_near_det - orig_near_det) * 100  # percentage points

    print(f"\n  Near-OOD detection improvement (α=0.10): {cp_improvement:+.1f}pp")

    # B6: Graph features
    print("\n[B6] Computing retrieval graph features...")
    graph_results = evaluate_graph_features(
        id_train_emb_final, id_train_labels_final, id_cal_emb, id_test_emb,
        ood_test_emb, ood_groups, k=10
    )

    print("  Individual feature AUROCs (near-OOD):")
    for feat, metrics in graph_results.items():
        print(f"    {feat}: near AUROC={metrics.get('near_auroc', 'N/A'):.3f}")

    knn_only_near = graph_results['knn_only'].get('near_auroc', 0)
    knn_graph_near = graph_results['knn_plus_graph'].get('near_auroc', 0)
    graph_increment = (knn_graph_near - knn_only_near) * 100

    print(f"\n  Graph feature increment (near AUROC): {graph_increment:+.1f}%")

    # Compile results
    return {
        'dataset': dataset_name,
        'model': model_name,
        'anisotropy': aniso,
        'baseline_metrics': baseline_metrics,
        'whitening_results': whitening_results,
        'best_k': best_k,
        'original_near_cohens_d': original_near_cohens_d,
        'best_near_cohens_d': best_near_cohens_d,
        'cp_original': cp_original,
        'cp_whitened': cp_whitened,
        'cp_improvement': cp_improvement,
        'graph_results': graph_results,
        'knn_only_near_auroc': knn_only_near,
        'knn_graph_near_auroc': knn_graph_near,
        'graph_increment': graph_increment,
    }


def create_visualizations(all_results, output_dir):
    """Create visualization figures"""
    # Figure 1: Cohen's d vs k for all experiments
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, (key, result) in enumerate(all_results.items()):
        ax = axes[idx // 2, idx % 2]
        df = result['whitening_results']

        ax.plot(df['k'], df['cohens_d'], 'b-o', label="Full OOD")
        ax.plot(df['k'], df['near_cohens_d'], 'r-s', label="Near-OOD")
        ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='d=0.8 threshold')
        ax.set_xlabel('k (components removed)')
        ax.set_ylabel("Cohen's d")
        ax.set_title(f"{result['dataset']} + {result['model'].split('/')[-1]}")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_cohens_d_vs_k.png'), dpi=150)
    plt.close()
    print(f"Saved fig_cohens_d_vs_k.png")

    # Figure 2: AUROC breakdown
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, (key, result) in enumerate(all_results.items()):
        ax = axes[idx // 2, idx % 2]
        df = result['whitening_results']

        ax.plot(df['k'], df['full_auroc'], 'b-o', label="Full")
        ax.plot(df['k'], df['near_auroc'], 'r-s', label="Near")
        ax.plot(df['k'], df['medium_auroc'], 'g-^', label="Medium")
        ax.plot(df['k'], df['far_auroc'], 'm-d', label="Far")
        ax.set_xlabel('k (components removed)')
        ax.set_ylabel('AUROC')
        ax.set_title(f"{result['dataset']} + {result['model'].split('/')[-1]}")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(0.5, 1.0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_auroc_breakdown.png'), dpi=150)
    plt.close()
    print(f"Saved fig_auroc_breakdown.png")

    # Figure 3: Graph features comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    x_labels = []
    knn_only = []
    knn_graph = []

    for key, result in all_results.items():
        x_labels.append(f"{result['dataset']}\n{result['model'].split('/')[-1]}")
        knn_only.append(result['knn_only_near_auroc'])
        knn_graph.append(result['knn_graph_near_auroc'])

    x = np.arange(len(x_labels))
    width = 0.35

    ax.bar(x - width/2, knn_only, width, label='kNN only', color='steelblue')
    ax.bar(x + width/2, knn_graph, width, label='kNN + Graph', color='coral')

    ax.set_ylabel('Near-OOD AUROC')
    ax.set_title('Retrieval Graph Feature Increment')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_retrieval_graph_features.png'), dpi=150)
    plt.close()
    print(f"Saved fig_retrieval_graph_features.png")


def make_final_determination(all_results, output_dir):
    """Make RAG route kill-switch determination"""
    print("\n" + "=" * 60)
    print("B7: RAG ROUTE KILL-SWITCH DETERMINATION")
    print("=" * 60)

    # Condition 1: Whitening improves near-OOD Cohen's d
    cond1_results = []
    for key, result in all_results.items():
        passes = result['best_near_cohens_d'] > 0.8
        cond1_results.append({
            'exp': key,
            'original_d': result['original_near_cohens_d'],
            'best_d': result['best_near_cohens_d'],
            'best_k': result['best_k'],
            'passes': passes
        })

    cond1_pass_count = sum(r['passes'] for r in cond1_results)
    cond1_total = len(cond1_results)
    cond1 = cond1_pass_count >= 2

    print(f"\n═══ 条件1：光谱白化改善near-OOD分离度 ═══")
    for r in cond1_results:
        print(f"  {r['exp']}: d={r['original_d']:.2f} -> {r['best_d']:.2f} (k*={r['best_k']}) {'✓' if r['passes'] else '✗'}")
    print(f"  通过: {cond1_pass_count}/{cond1_total} (需≥2)")
    print(f"  判定: {'YES' if cond1 else 'NO'}")

    # Condition 2: CP detection rate improvement >= 10pp
    cond2_results = []
    for key, result in all_results.items():
        improvement = result['cp_improvement']
        passes = improvement >= 10
        cond2_results.append({
            'exp': key,
            'improvement': improvement,
            'passes': passes
        })

    cond2_pass_count = sum(r['passes'] for r in cond2_results)
    cond2 = cond2_pass_count >= 2

    print(f"\n═══ 条件2：CP检测率显著提升 ═══")
    for r in cond2_results:
        print(f"  {r['exp']}: {r['improvement']:+.1f}pp {'✓' if r['passes'] else '✗'}")
    print(f"  通过: {cond2_pass_count}/{cond1_total} (需≥2)")
    print(f"  判定: {'YES' if cond2 else 'NO'}")

    # Condition 3: Graph feature increment >= 2%
    cond3_results = []
    for key, result in all_results.items():
        increment = result['graph_increment']
        passes = increment >= 2
        cond3_results.append({
            'exp': key,
            'increment': increment,
            'passes': passes
        })

    cond3_pass_count = sum(r['passes'] for r in cond3_results)
    cond3 = cond3_pass_count >= 2

    print(f"\n═══ 条件3：检索图特征增量 ═══")
    for r in cond3_results:
        print(f"  {r['exp']}: {r['increment']:+.1f}% {'✓' if r['passes'] else '✗'}")
    print(f"  通过: {cond3_pass_count}/{cond1_total} (需≥2)")
    print(f"  判定: {'YES' if cond3 else 'NO'}")

    # Condition 4: Cross-dataset + cross-model robustness
    datasets = set(r['dataset'] for r in all_results.values())
    models = set(r['model'] for r in all_results.values())

    # Check if condition 1 passes on both datasets
    cond1_by_dataset = {}
    cond1_by_model = {}
    for key, result in all_results.items():
        ds = result['dataset']
        model = result['model']
        passes = result['best_near_cohens_d'] > 0.8
        cond1_by_dataset[ds] = cond1_by_dataset.get(ds, False) or passes
        cond1_by_model[model] = cond1_by_model.get(model, False) or passes

    cond4_dataset = all(cond1_by_dataset.values())
    cond4_model = all(cond1_by_model.values())
    cond4 = cond4_dataset and cond4_model

    print(f"\n═══ 条件4：跨数据集+跨模型鲁棒性 ═══")
    print(f"  条件1在两个数据集上都YES? {cond4_dataset}")
    print(f"  条件1在两个模型上都YES? {cond4_model}")
    print(f"  判定: {'YES' if cond4 else 'NO'}")

    # Final determination
    n_yes = sum([cond1, cond2, cond3, cond4])
    s2_pass = n_yes >= 3

    print(f"\n{'='*60}")
    print(f"S2总判定 = {'YES' if s2_pass else 'NO'}")
    print(f"（4个条件中{n_yes}个YES，需≥3）")

    if s2_pass:
        verdict = "走RAG方向"
        reason = f"4个条件中{n_yes}个满足（≥3），RAG路线验证通过"
    else:
        verdict = "退守P5"
        reason = f"只有{n_yes}个条件满足（<3），RAG路线验证失败"

    # Find most promising signal
    best_signal = max(cond1_results, key=lambda x: x['best_d'])
    biggest_risk = "near-OOD分离度可能不够稳定" if not cond4 else "需要更多数据集验证"

    print(f"\n决策: {verdict}")
    print(f"原因: {reason}")
    print(f"最有前景的信号: {best_signal['exp']} (Cohen's d={best_signal['best_d']:.2f})")
    print(f"最大障碍/风险: {biggest_risk}")
    print("=" * 60)

    # Save determination report
    with open(os.path.join(output_dir, 'rag_route_determination.md'), 'w') as f:
        f.write("# RAG Route Kill-Switch Determination (v2, Real Data)\n\n")
        f.write("```\n")
        f.write("═══════════════════════════════════════\n")
        f.write("RAG路线 KILL-SWITCH 判定（v2，真实数据）\n")
        f.write("═══════════════════════════════════════\n\n")

        f.write("实验配置：\n")
        for key, result in all_results.items():
            f.write(f"  {key}: {result['dataset']} + {result['model']}\n")

        f.write("\n═══ 条件1：光谱白化改善near-OOD分离度 ═══\n")
        for r in cond1_results:
            f.write(f"{r['exp']}:\n")
            f.write(f"  原始 Cohen's d = {r['original_d']:.2f}，最优k*={r['best_k']}，白化后 Cohen's d = {r['best_d']:.2f}\n")
        f.write(f"\n{cond1_pass_count}组实验中，白化后Cohen's d > 0.8的组数 = {cond1_pass_count}/4\n")
        f.write(f"判定：{'YES' if cond1 else 'NO'}（≥2组达到0.8为YES）\n")

        f.write("\n═══ 条件2：CP检测率显著提升 ═══\n")
        f.write("near-OOD检测率提升（α=0.10）：\n")
        for r in cond2_results:
            f.write(f"  {r['exp']}: {r['improvement']:+.1f}pp\n")
        f.write(f"\n{cond2_pass_count}组中提升≥10pp的组数 = {cond2_pass_count}/4\n")
        f.write(f"判定：{'YES' if cond2 else 'NO'}（≥2组达到10pp为YES）\n")

        f.write("\n═══ 条件3：检索图特征增量 ═══\n")
        f.write("near-OOD AUROC增量（kNN+图特征 vs 纯kNN）：\n")
        for r in cond3_results:
            f.write(f"  {r['exp']}: {r['increment']:+.1f}%\n")
        f.write(f"\n{cond3_pass_count}组中增量≥2%的组数 = {cond3_pass_count}/4\n")
        f.write(f"判定：{'YES' if cond3 else 'NO'}（≥2组达到2%为YES）\n")

        f.write("\n═══ 条件4：跨数据集+跨模型鲁棒性 ═══\n")
        f.write(f"条件1在两个数据集上都YES？ {cond4_dataset}\n")
        f.write(f"条件1在两个模型上都YES？ {cond4_model}\n")
        f.write(f"判定：{'YES' if cond4 else 'NO'}\n")

        f.write("\n═══════════════════════════════════════\n")
        f.write(f"S2总判定 = {'YES' if s2_pass else 'NO'}\n")
        f.write(f"（4个条件中{n_yes}个YES则S2=YES）\n\n")
        f.write(f"决策：{verdict}\n")
        f.write(f"原因：{reason}\n")
        f.write(f"最有前景的信号：{best_signal['exp']} (d={best_signal['best_d']:.2f})\n")
        f.write(f"最大障碍/风险：{biggest_risk}\n")
        f.write("═══════════════════════════════════════\n")
        f.write("```\n")

    return {
        'cond1': cond1,
        'cond2': cond2,
        'cond3': cond3,
        'cond4': cond4,
        's2_pass': s2_pass,
        'verdict': verdict,
        'reason': reason,
    }


# ============================================
# Main Execution
# ============================================

def main():
    print("=" * 60)
    print("Part B: RAG Route Real Data Validation")
    print("RW3 Kill-Switch Determination v2")
    print("=" * 60)

    # B1: Load datasets
    clinc_data = load_clinc150()
    banking_data = load_banking77()

    if clinc_data is None and banking_data is None:
        print("\nERROR: Failed to load any dataset. Aborting.")
        return

    # B2: Load models
    print("\n[B2] Loading embedding models...")

    model_light = load_embedding_model('all-MiniLM-L6-v2')

    # Try BGE first, fall back to mpnet
    model_medium = load_embedding_model('BAAI/bge-base-en-v1.5')
    if model_medium is None:
        model_medium = load_embedding_model('all-mpnet-base-v2')

    if model_light is None or model_medium is None:
        print("\nERROR: Failed to load embedding models. Aborting.")
        return

    # Run experiments
    all_results = {}

    datasets = []
    if clinc_data is not None:
        datasets.append(('clinc150', clinc_data))
    if banking_data is not None:
        datasets.append(('banking77', banking_data))

    models = [
        ('all-MiniLM-L6-v2', model_light),
        ('BAAI/bge-base-en-v1.5', model_medium),
    ]

    for ds_name, ds_data in datasets:
        for model_name, model in models:
            key = f"{ds_name}_{model_name.split('/')[-1]}"
            try:
                result = run_full_experiment(ds_data, ds_name, model, model_name)
                all_results[key] = result
            except Exception as e:
                print(f"ERROR in {key}: {e}")
                import traceback
                traceback.print_exc()

    if len(all_results) < 2:
        print("\nERROR: Not enough successful experiments. Aborting.")
        return

    # Create visualizations
    print("\n[B6] Creating visualizations...")
    create_visualizations(all_results, OUTPUT_DIR)

    # Final determination
    determination = make_final_determination(all_results, OUTPUT_DIR)

    # Save detailed results
    save_detailed_results(all_results, OUTPUT_DIR)

    return all_results, determination


def save_detailed_results(all_results, output_dir):
    """Save detailed results to markdown files"""

    # Baseline metrics
    with open(os.path.join(output_dir, 'baseline_metrics.md'), 'w') as f:
        f.write("# Baseline OOD Detection Metrics\n\n")
        f.write("**Data Source**: Real CLINC150 and Banking77 datasets\n\n")

        for key, result in all_results.items():
            f.write(f"## {key}\n\n")
            f.write("### Anisotropy\n")
            f.write(f"- Top-1 variance ratio: {result['anisotropy']['top1_var_ratio']:.3f}\n")
            f.write(f"- Mean pair cosine: {result['anisotropy']['mean_pair_cosine']:.3f}\n")
            f.write(f"- Anisotropic: {result['anisotropy']['is_anisotropic']}\n\n")

            f.write("### Baseline OOD Scores (k=0)\n")
            f.write("| Score | Full AUROC | Near AUROC | Cohen's d |\n")
            f.write("|-------|------------|------------|----------|\n")
            for score_name, metrics in result['baseline_metrics'].items():
                f.write(f"| {score_name} | {metrics['full_auroc']:.3f} | {metrics.get('near_auroc', 'N/A'):.3f} | {metrics['cohens_d']:.2f} |\n")
            f.write("\n")

    # Whitening ablation
    with open(os.path.join(output_dir, 'whitening_ablation.md'), 'w') as f:
        f.write("# Spectral Whitening Ablation\n\n")

        for key, result in all_results.items():
            f.write(f"## {key}\n\n")
            df = result['whitening_results']
            f.write(df.to_markdown(index=False))
            f.write(f"\n\n**Best k* = {result['best_k']}** (Cohen's d: {result['original_near_cohens_d']:.2f} -> {result['best_near_cohens_d']:.2f})\n\n")

    # CP detection rates
    with open(os.path.join(output_dir, 'cp_detection_rates.md'), 'w') as f:
        f.write("# Conformal Prediction Detection Rates\n\n")

        for key, result in all_results.items():
            f.write(f"## {key}\n\n")
            f.write("### Original Embeddings\n")
            f.write(result['cp_original'].to_markdown(index=False))
            f.write(f"\n\n### Whitened Embeddings (k={result['best_k']})\n")
            f.write(result['cp_whitened'].to_markdown(index=False))
            f.write(f"\n\n**Improvement (α=0.10)**: {result['cp_improvement']:+.1f}pp\n\n")

    # Graph feature ablation
    with open(os.path.join(output_dir, 'graph_feature_ablation.md'), 'w') as f:
        f.write("# Retrieval Graph Feature Ablation\n\n")

        for key, result in all_results.items():
            f.write(f"## {key}\n\n")
            f.write("| Feature | Full AUROC | Near AUROC | Cohen's d |\n")
            f.write("|---------|------------|------------|----------|\n")
            for feat_name, metrics in result['graph_results'].items():
                f.write(f"| {feat_name} | {metrics['full_auroc']:.3f} | {metrics.get('near_auroc', 'N/A'):.3f} | {metrics['cohens_d']:.2f} |\n")
            f.write(f"\n**Graph increment (near AUROC)**: {result['graph_increment']:+.1f}%\n\n")

    print(f"Saved detailed results to {output_dir}")


if __name__ == '__main__':
    main()
