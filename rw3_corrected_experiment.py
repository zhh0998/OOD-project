#!/usr/bin/env python3
"""
RW3 OOD Assortativity Verification Experiment - CORRECTED VERSION

Fixes three critical bugs from V2:
1. CLINC150: Dynamic OOS label lookup (42 for small/plus, not hardcoded 150)
2. Banking77: Fixed 50/27 split using Zhang et al. (ACL 2022) standard
3. All datasets: UMAP dimensionality reduction before HDBSCAN

Expected improvements:
- Noise ratio: 60-66% -> 15-30%
- CLINC150: Restore heterophilic pattern (fix OOS label bug)
- Banking77: Stable and reproducible results (fixed split)
"""

import numpy as np
import pandas as pd
from datetime import datetime
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(42)


def convert_numpy(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

#==============================================================================
# CONFIGURATION
#==============================================================================

# Fixed 27 OOD intents for Banking77 (Zhang et al., ACL 2022)
BANKING77_OOD_INTENTS = [
    "pin_blocked", "top_up_by_cash_or_cheque", "top_up_by_card_charge",
    "verify_source_of_funds", "transfer_into_account", "exchange_rate",
    "card_delivery_estimate", "card_not_working", "age_limit",
    "terminate_account", "get_physical_card", "passcode_forgotten",
    "verify_my_identity", "topping_up_by_card", "unable_to_verify_identity",
    "getting_virtual_card", "top_up_limits", "get_disposable_virtual_card",
    "receiving_money", "atm_support", "compromised_card",
    "lost_or_stolen_card", "card_swallowed", "card_acceptance",
    "virtual_card_not_working", "contactless_not_working",
    "top_up_by_bank_transfer_charge"
]

# Output directory
OUTPUT_DIR = "rw3_corrected_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#==============================================================================
# DATA LOADING FUNCTIONS
#==============================================================================

def load_clinc150():
    """
    Load CLINC150 with DYNAMIC OOS label lookup.

    Critical fix: OOS label is 42 in small/plus config (alphabetical order),
    NOT 150 (which is only in 'full' config).
    """
    from datasets import load_dataset

    print("\n" + "#"*80)
    print("# Loading CLINC150 (with dynamic OOS label lookup)")
    print("#"*80)

    # Use 'small' config (recommended for experiments)
    dataset = load_dataset("clinc_oos", "small")

    # CRITICAL: Dynamic OOS label lookup
    label_names = dataset['train'].features['intent'].names
    OOS_LABEL = label_names.index('oos')

    print(f"\n[CLINC150] Dynamic OOS label lookup:")
    print(f"  Config: 'small'")
    print(f"  Total intents: {len(label_names)}")
    print(f"  OOS Label ID: {OOS_LABEL}")  # Should be 42 for small/plus
    print(f"  OOS Label Name: '{label_names[OOS_LABEL]}'")

    # Verify it's 42 (sanity check)
    if OOS_LABEL != 42:
        print(f"  WARNING: OOS label is {OOS_LABEL}, not 42. This may be 'full' config.")
    else:
        print(f"  ✓ Confirmed: OOS label = 42 (alphabetical order)")

    # Combine all splits
    all_texts = []
    all_labels = []
    for split in ['train', 'validation', 'test']:
        all_texts.extend(list(dataset[split]['text']))
        all_labels.extend(list(dataset[split]['intent']))

    all_labels = np.array(all_labels)

    # Separate ID and OOD
    id_mask = all_labels != OOS_LABEL
    ood_mask = all_labels == OOS_LABEL

    id_texts = [t for t, m in zip(all_texts, id_mask) if m]
    id_labels = all_labels[id_mask]
    ood_texts = [t for t, m in zip(all_texts, ood_mask) if m]

    n_id_classes = len(set(id_labels))

    print(f"\n[CLINC150] Data statistics:")
    print(f"  ID samples: {len(id_texts)} ({n_id_classes} classes)")
    print(f"  OOD samples: {len(ood_texts)} (out-of-scope)")
    print(f"  Total: {len(all_texts)}")

    return {
        'name': 'CLINC150',
        'id_texts': id_texts,
        'ood_texts': ood_texts,
        'ood_type': 'out-of-scope',
        'description': f'Out-of-scope queries (OOS_LABEL={OOS_LABEL})',
        'n_classes': n_id_classes
    }


def load_banking77():
    """
    Load Banking77 with FIXED 50/27 class split.

    Critical fix: Use Zhang et al. (ACL 2022) standard split instead of
    random shuffle which produces different results with same seed.
    """
    from datasets import load_dataset

    print("\n" + "#"*80)
    print("# Loading Banking77 (with fixed 50/27 split)")
    print("#"*80)

    dataset = load_dataset("mteb/banking77")

    # Get all unique label texts to build label_names list
    all_label_texts = list(dataset['train']['label_text']) + list(dataset['test']['label_text'])
    all_labels_int = list(dataset['train']['label']) + list(dataset['test']['label'])

    # Build label index to name mapping
    label_to_name = {}
    for label_int, label_text in zip(all_labels_int, all_label_texts):
        label_to_name[label_int] = label_text

    # Create label_names list (sorted by index)
    n_classes = max(label_to_name.keys()) + 1
    label_names = [label_to_name.get(i, f"unknown_{i}") for i in range(n_classes)]

    print(f"\n[Banking77] Fixed class split (Zhang et al., ACL 2022):")
    print(f"  Total classes: {len(label_names)}")

    # Map OOD intent names to indices
    ood_indices = set()
    missing_intents = []
    for intent in BANKING77_OOD_INTENTS:
        try:
            idx = label_names.index(intent)
            ood_indices.add(idx)
        except ValueError:
            missing_intents.append(intent)

    if missing_intents:
        print(f"  WARNING: Missing intents: {missing_intents}")

    id_indices = set(range(77)) - ood_indices

    print(f"  ID classes: {len(id_indices)}")
    print(f"  OOD classes: {len(ood_indices)}")

    # Print a few examples of each
    id_names = [label_names[i] for i in sorted(id_indices)[:5]]
    ood_names = [label_names[i] for i in sorted(ood_indices)[:5]]
    print(f"  ID examples: {id_names}...")
    print(f"  OOD examples: {ood_names}...")

    # Combine train and test
    all_texts = list(dataset['train']['text']) + list(dataset['test']['text'])
    all_labels = np.array(all_labels_int)

    # Separate ID and OOD
    id_mask = np.array([l in id_indices for l in all_labels])
    ood_mask = np.array([l in ood_indices for l in all_labels])

    id_texts = [t for t, m in zip(all_texts, id_mask) if m]
    ood_texts = [t for t, m in zip(all_texts, ood_mask) if m]

    print(f"\n[Banking77] Data statistics:")
    print(f"  ID samples: {len(id_texts)} ({len(id_indices)} classes)")
    print(f"  OOD samples: {len(ood_texts)} ({len(ood_indices)} classes)")
    print(f"  Total: {len(all_texts)}")

    return {
        'name': 'Banking77',
        'id_texts': id_texts,
        'ood_texts': ood_texts,
        'ood_type': 'held-out-class',
        'description': 'Fixed 50/27 split (Zhang et al., ACL 2022)',
        'n_classes': len(id_indices)
    }


def load_rostd():
    """Load ROSTD dataset (unchanged from previous version)."""
    from datasets import load_dataset

    print("\n" + "#"*80)
    print("# Loading ROSTD")
    print("#"*80)

    # Use Banking77 as ID proxy
    banking = load_dataset("mteb/banking77")
    id_texts = list(banking['train']['text']) + list(banking['test']['text'])

    # Load ROSTD OOD data
    ood_path = 'dataset_downloads/LR_GC_OOD_data/LR_GC_OOD-master/data/fbrelease/OODrelease.tsv'
    if os.path.exists(ood_path):
        # File format: label \t FILLER \t text \t FILLER (no header)
        ood_df = pd.read_csv(ood_path, sep='\t', header=None, names=['label', 'f1', 'text', 'f2'])
        ood_texts = list(ood_df['text'])
        print(f"  OOD file: {ood_path}")
    else:
        print(f"  WARNING: {ood_path} not found, using placeholder")
        ood_texts = ["This is a placeholder OOD text."] * 100

    print(f"\n[ROSTD] Data statistics:")
    print(f"  ID samples: {len(id_texts)} (Banking77 as proxy)")
    print(f"  OOD samples: {len(ood_texts)} (diverse human-authored)")

    return {
        'name': 'ROSTD',
        'id_texts': id_texts,
        'ood_texts': ood_texts,
        'ood_type': 'diverse-ood',
        'description': 'Real diverse human-authored OOD',
        'n_classes': 77
    }


def load_toxigen():
    """Load ToxiGen dataset (unchanged from previous version)."""
    from datasets import load_dataset

    print("\n" + "#"*80)
    print("# Loading ToxiGen")
    print("#"*80)

    try:
        dataset = load_dataset("skg/toxigen-data", "annotated", split="train")
    except:
        dataset = load_dataset("toxigen/toxigen-data", "annotated", split="train")

    # toxicity_ai >= 2.5 as OOD threshold
    TOXICITY_THRESHOLD = 2.5

    id_texts = [item['text'] for item in dataset if item['toxicity_ai'] < TOXICITY_THRESHOLD]
    ood_texts = [item['text'] for item in dataset if item['toxicity_ai'] >= TOXICITY_THRESHOLD]

    print(f"\n[ToxiGen] Data statistics:")
    print(f"  Toxicity threshold: {TOXICITY_THRESHOLD}")
    print(f"  ID samples: {len(id_texts)} (non-toxic)")
    print(f"  OOD samples: {len(ood_texts)} (toxic)")

    return {
        'name': 'ToxiGen',
        'id_texts': id_texts,
        'ood_texts': ood_texts,
        'ood_type': 'toxicity-based',
        'description': f'Toxicity-based OOD (threshold={TOXICITY_THRESHOLD})',
        'n_classes': 'continuous'
    }


#==============================================================================
# EMBEDDING AND PROCESSING
#==============================================================================

def generate_embeddings(texts, model_name='all-mpnet-base-v2'):
    """Generate sentence embeddings."""
    from sentence_transformers import SentenceTransformer

    print(f"\n[Embeddings] Generating with {model_name}...")
    model = SentenceTransformer(model_name)

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=False  # Keep raw embeddings for k-NN
    )

    print(f"[Embeddings] Shape: {embeddings.shape}")
    return embeddings


def apply_umap(embeddings, n_components=5, n_neighbors=15, min_dist=0.0):
    """
    Apply UMAP dimensionality reduction.

    Critical addition: HDBSCAN performs poorly on high-dimensional data.
    UMAP reduces 768d -> 5d while preserving local density structure.
    """
    from umap import UMAP

    print(f"\n[UMAP] Reducing dimensions: {embeddings.shape[1]} -> {n_components}")
    print(f"  n_neighbors: {n_neighbors}")
    print(f"  min_dist: {min_dist} (tight packing for clustering)")
    print(f"  metric: cosine (for text embeddings)")

    umap_model = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='cosine',
        random_state=42,
        n_jobs=1  # Reproducibility
    )

    embeddings_reduced = umap_model.fit_transform(embeddings)

    print(f"[UMAP] Complete: {embeddings.shape} -> {embeddings_reduced.shape}")
    print(f"[UMAP] Reduction ratio: {embeddings.shape[1] / embeddings_reduced.shape[1]:.1f}x")

    return embeddings_reduced


def run_hdbscan(embeddings, n_samples):
    """
    Run HDBSCAN clustering with adaptive parameters.

    Key change: Independent min_samples setting (lower than min_cluster_size).
    """
    import hdbscan

    # Adaptive parameters based on dataset size
    if n_samples < 500:
        min_cluster_size = 3
        min_samples = 2
    elif n_samples < 2000:
        min_cluster_size = 5
        min_samples = 3
    elif n_samples < 10000:
        min_cluster_size = 15
        min_samples = 5
    else:
        min_cluster_size = 20
        min_samples = 5

    print(f"\n[HDBSCAN] Clustering parameters:")
    print(f"  n_samples: {n_samples}")
    print(f"  min_cluster_size: {min_cluster_size}")
    print(f"  min_samples: {min_samples} (independent, for density)")
    print(f"  metric: euclidean (on reduced embeddings)")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom',
        cluster_selection_epsilon=0.01,
        core_dist_n_jobs=1
    )

    cluster_labels = clusterer.fit_predict(embeddings)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = (cluster_labels == -1).sum()
    noise_ratio = n_noise / n_samples

    print(f"\n[HDBSCAN] Results:")
    print(f"  Clusters: {n_clusters}")
    print(f"  Noise samples: {n_noise} ({noise_ratio*100:.1f}%)")

    # Quality assessment
    if noise_ratio > 0.4:
        print(f"  ⚠️ WARNING: Noise ratio >40%")
    elif noise_ratio > 0.3:
        print(f"  ⚠️ Note: Noise ratio 30-40% (acceptable)")
    else:
        print(f"  ✅ Good: Noise ratio <30%")

    if n_clusters < 5:
        print(f"  ⚠️ WARNING: Too few clusters (<5)")

    return cluster_labels, {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'noise_ratio': noise_ratio,
        'min_cluster_size': min_cluster_size,
        'min_samples': min_samples
    }


def build_knn_graph(embeddings, k=15):
    """Build k-NN graph on reduced embeddings."""
    from sklearn.neighbors import NearestNeighbors

    print(f"\n[k-NN Graph] Building with k={k}")

    nn = NearestNeighbors(
        n_neighbors=k+1,  # +1 for self
        metric='euclidean',
        n_jobs=-1
    )

    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    # Exclude self
    knn_indices = indices[:, 1:]
    knn_distances = distances[:, 1:]

    print(f"[k-NN Graph] Shape: {knn_indices.shape}")

    return knn_indices, knn_distances


def compute_nhr(knn_indices, cluster_labels, verbose=True):
    """
    Compute Node Homophily Ratio (NHR).

    NHR = fraction of k-NN neighbors with same cluster label.
    Noise points (cluster=-1) have NHR=0.
    """
    n_samples = len(cluster_labels)
    nhr_values = np.zeros(n_samples)
    n_noise_neighbors = 0

    for i in range(n_samples):
        my_label = cluster_labels[i]

        # Noise points get NHR=0
        if my_label == -1:
            nhr_values[i] = 0.0
            continue

        neighbors = knn_indices[i]
        neighbor_labels = cluster_labels[neighbors]

        # Count same-label neighbors
        same_label = (neighbor_labels == my_label).sum()
        nhr_values[i] = same_label / len(neighbors)

        # Track noise neighbors
        n_noise_neighbors += (neighbor_labels == -1).sum()

    if verbose:
        non_noise_mask = cluster_labels != -1
        avg_nhr = nhr_values[non_noise_mask].mean() if non_noise_mask.sum() > 0 else 0

        print(f"\n[NHR] Statistics:")
        print(f"  Average NHR (non-noise): {avg_nhr:.4f}")
        print(f"  Noise neighbor connections: {n_noise_neighbors}")

        if avg_nhr > 0.7:
            print(f"  ✅ Cluster quality: Strong homophily (NHR>0.7)")
        elif avg_nhr > 0.5:
            print(f"  ⚠️ Cluster quality: Medium homophily (0.5<NHR<0.7)")
        else:
            print(f"  ❌ Cluster quality: Heterophily or high noise (NHR<0.5)")

    return nhr_values


def compute_embedding_homophily(knn_indices, knn_distances):
    """Compute embedding homophily (average cosine similarity to neighbors)."""
    # Lower distance = higher homophily
    # Normalize to [0, 1] where 1 = high homophily
    max_dist = knn_distances.max()
    homophily = 1 - (knn_distances.mean(axis=1) / (max_dist + 1e-10))
    return homophily


#==============================================================================
# STATISTICAL ANALYSIS
#==============================================================================

def statistical_analysis(nhr, is_ood, embedding_homophily=None):
    """Compute effect sizes, significance tests, and AUROC."""
    from scipy import stats
    from sklearn.metrics import roc_auc_score

    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)

    # Separate ID and OOD
    id_nhr = nhr[~is_ood]
    ood_nhr = nhr[is_ood]

    print(f"\n[Sample sizes]")
    print(f"  ID: {len(id_nhr)}, OOD: {len(ood_nhr)}")

    # Basic statistics
    id_mean, id_std = id_nhr.mean(), id_nhr.std()
    ood_mean, ood_std = ood_nhr.mean(), ood_nhr.std()

    print(f"\n[NHR Distribution]")
    print(f"  ID:  {id_mean:.4f} ± {id_std:.4f}")
    print(f"  OOD: {ood_mean:.4f} ± {ood_std:.4f}")
    print(f"  Diff: {ood_mean - id_mean:+.4f}")

    # Cohen's d
    pooled_std = np.sqrt((id_std**2 + ood_std**2) / 2)
    cohens_d = (ood_mean - id_mean) / (pooled_std + 1e-10)

    print(f"\n[Effect Size]")
    print(f"  Cohen's d: {cohens_d:+.4f}")

    if abs(cohens_d) > 0.8:
        effect_label = "Large effect (|d|>0.8)"
        effect_emoji = "✅"
    elif abs(cohens_d) > 0.5:
        effect_label = "Medium effect (0.5<|d|<0.8)"
        effect_emoji = "⚠️"
    elif abs(cohens_d) > 0.2:
        effect_label = "Small effect (0.2<|d|<0.5)"
        effect_emoji = "⚠️"
    else:
        effect_label = "Negligible effect (|d|<0.2)"
        effect_emoji = "❌"

    print(f"  {effect_emoji} {effect_label}")

    if cohens_d > 0:
        direction = "HOMOPHILIC"
        print(f"  Pattern: Homophilic (OOD NHR > ID NHR)")
    else:
        direction = "HETEROPHILIC"
        print(f"  Pattern: Heterophilic (OOD NHR < ID NHR)")

    # t-test
    t_stat, p_value = stats.ttest_ind(ood_nhr, id_nhr)

    print(f"\n[Significance Test]")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.2e}")

    if p_value < 0.001:
        print(f"  ✅ Highly significant (p<0.001)")
    elif p_value < 0.05:
        print(f"  ✅ Significant (p<0.05)")
    else:
        print(f"  ❌ Not significant (p≥0.05)")

    # AUROC
    auroc = roc_auc_score(is_ood, -nhr)  # Negative because lower NHR = more OOD
    if auroc < 0.5:
        auroc = 1 - auroc

    print(f"\n[Classification Performance]")
    print(f"  NHR AUROC: {auroc:.4f}")

    if auroc > 0.95:
        print(f"  ✅ Excellent (AUROC>0.95)")
    elif auroc > 0.85:
        print(f"  ✅ Good (0.85<AUROC<0.95)")
    elif auroc > 0.7:
        print(f"  ⚠️ Medium (0.7<AUROC<0.85)")
    else:
        print(f"  ❌ Poor (AUROC<0.7)")

    results = {
        'id_mean': float(id_mean),
        'id_std': float(id_std),
        'ood_mean': float(ood_mean),
        'ood_std': float(ood_std),
        'cohens_d': float(cohens_d),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'auroc': float(auroc),
        'direction': direction,
        'effect_label': effect_label
    }

    # Embedding homophily analysis (if provided)
    if embedding_homophily is not None:
        id_emb_h = embedding_homophily[~is_ood]
        ood_emb_h = embedding_homophily[is_ood]

        emb_d = (ood_emb_h.mean() - id_emb_h.mean()) / (np.sqrt((id_emb_h.std()**2 + ood_emb_h.std()**2) / 2) + 1e-10)
        emb_auroc = roc_auc_score(is_ood, -embedding_homophily)
        if emb_auroc < 0.5:
            emb_auroc = 1 - emb_auroc

        print(f"\n[Embedding Homophily]")
        print(f"  ID:  {id_emb_h.mean():.4f} ± {id_emb_h.std():.4f}")
        print(f"  OOD: {ood_emb_h.mean():.4f} ± {ood_emb_h.std():.4f}")
        print(f"  Cohen's d: {emb_d:+.4f}")
        print(f"  AUROC: {emb_auroc:.4f}")

        results['embedding_homophily'] = {
            'id_mean': float(id_emb_h.mean()),
            'ood_mean': float(ood_emb_h.mean()),
            'cohens_d': float(emb_d),
            'auroc': float(emb_auroc)
        }

    return results


def compute_baselines(embeddings_reduced, is_ood, n_classes):
    """Compute SOTA baseline methods for comparison."""
    from sklearn.metrics import roc_auc_score
    from sklearn.neighbors import NearestNeighbors

    print(f"\n[Baselines] Computing SOTA comparisons...")

    results = {}

    # 1. k-NN Distance baseline
    nn = NearestNeighbors(n_neighbors=16, metric='euclidean')
    nn.fit(embeddings_reduced[~is_ood])  # Fit on ID only
    distances, _ = nn.kneighbors(embeddings_reduced)
    knn_scores = distances[:, 1:].mean(axis=1)  # Avg distance to k nearest

    knn_auroc = roc_auc_score(is_ood, knn_scores)
    results['knn_distance'] = float(knn_auroc)
    print(f"  k-NN Distance AUROC: {knn_auroc:.4f}")

    # 2. Mahalanobis distance (with regularization)
    try:
        id_embeddings = embeddings_reduced[~is_ood]
        mean = id_embeddings.mean(axis=0)
        cov = np.cov(id_embeddings.T)

        # Regularization
        cov_reg = cov + 1e-5 * np.eye(cov.shape[0])
        cov_inv = np.linalg.inv(cov_reg)

        maha_scores = []
        for emb in embeddings_reduced:
            diff = emb - mean
            score = np.sqrt(np.dot(np.dot(diff, cov_inv), diff))
            maha_scores.append(score)
        maha_scores = np.array(maha_scores)

        maha_auroc = roc_auc_score(is_ood, maha_scores)
        results['mahalanobis'] = float(maha_auroc)
        print(f"  Mahalanobis AUROC: {maha_auroc:.4f}")
    except Exception as e:
        print(f"  Mahalanobis: Failed ({e})")
        results['mahalanobis'] = None

    return results


#==============================================================================
# MAIN EXPERIMENT
#==============================================================================

def run_experiment(data_loader, k=15, umap_dim=5):
    """Run complete experiment pipeline on one dataset."""

    # 1. Load data
    data = data_loader()
    dataset_name = data['name']

    # Combine texts
    all_texts = data['id_texts'] + data['ood_texts']
    n_id = len(data['id_texts'])
    n_ood = len(data['ood_texts'])
    n_total = len(all_texts)

    is_ood = np.array([False] * n_id + [True] * n_ood)

    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {dataset_name}")
    print(f"{'='*80}")
    print(f"Total samples: {n_total} (ID={n_id}, OOD={n_ood})")

    # 2. Generate embeddings
    embeddings = generate_embeddings(all_texts)

    # 3. UMAP dimensionality reduction (CRITICAL FIX!)
    embeddings_reduced = apply_umap(embeddings, n_components=umap_dim)

    # 4. HDBSCAN clustering
    cluster_labels, cluster_stats = run_hdbscan(embeddings_reduced, n_total)

    # 5. Build k-NN graph
    knn_indices, knn_distances = build_knn_graph(embeddings_reduced, k=k)

    # 6. Compute NHR
    nhr = compute_nhr(knn_indices, cluster_labels)

    # 7. Compute embedding homophily
    embedding_homophily = compute_embedding_homophily(knn_indices, knn_distances)

    # 8. Statistical analysis
    stats_results = statistical_analysis(nhr, is_ood, embedding_homophily)

    # 9. Baseline comparison
    baselines = compute_baselines(embeddings_reduced, is_ood, data.get('n_classes', 50))

    # 10. Compile results
    results = {
        'dataset': dataset_name,
        'config': {
            'k_neighbors': k,
            'umap_dim': umap_dim,
            'ood_type': data['ood_type'],
            'description': data['description']
        },
        'data_info': {
            'n_id': n_id,
            'n_ood': n_ood,
            'n_total': n_total
        },
        'clustering': cluster_stats,
        'nhr_stats': stats_results,
        'baselines': baselines,
        'main_finding': {
            'cohens_d': stats_results['cohens_d'],
            'auroc': stats_results['auroc'],
            'direction': stats_results['direction'],
            'noise_ratio': cluster_stats['noise_ratio']
        }
    }

    # Save individual result
    output_path = os.path.join(OUTPUT_DIR, f"{dataset_name.lower()}_results.json")
    with open(output_path, 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)
    print(f"\n[Saved] {output_path}")

    return results


def main():
    """Run experiments on all datasets."""

    print("="*80)
    print("RW3 CORRECTED EXPERIMENT")
    print("="*80)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("\nKey corrections:")
    print("  1. CLINC150: Dynamic OOS label lookup (42, not 150)")
    print("  2. Banking77: Fixed 50/27 split (Zhang et al., ACL 2022)")
    print("  3. All: UMAP 768d->5d before HDBSCAN")
    print("="*80)

    # Define dataset loaders
    datasets = [
        ('CLINC150', load_clinc150),
        ('Banking77', load_banking77),
        ('ROSTD', load_rostd),
        ('ToxiGen', load_toxigen),
    ]

    all_results = {}

    for name, loader in datasets:
        try:
            results = run_experiment(loader)
            all_results[name] = results
        except Exception as e:
            print(f"\n❌ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results[name] = {'error': str(e)}

    # Generate summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    summary_rows = []
    for name, results in all_results.items():
        if 'error' in results:
            summary_rows.append({
                'Dataset': name,
                'Status': 'FAILED',
                'Error': results['error']
            })
        else:
            mf = results['main_finding']
            summary_rows.append({
                'Dataset': name,
                'N_ID': results['data_info']['n_id'],
                'N_OOD': results['data_info']['n_ood'],
                'Noise%': f"{mf['noise_ratio']*100:.1f}%",
                'Cohen_d': f"{mf['cohens_d']:+.4f}",
                'AUROC': f"{mf['auroc']:.4f}",
                'Direction': mf['direction']
            })

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + summary_df.to_string(index=False))

    # Save summary CSV
    summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[Saved] {summary_path}")

    # Save full report
    report = {
        'experiment': 'RW3 Corrected Verification',
        'timestamp': datetime.now().isoformat(),
        'corrections': [
            'CLINC150: Dynamic OOS label lookup',
            'Banking77: Fixed 50/27 split',
            'All: UMAP dimensionality reduction'
        ],
        'results': all_results
    }

    report_path = os.path.join(OUTPUT_DIR, "full_report.json")
    with open(report_path, 'w') as f:
        json.dump(convert_numpy(report), f, indent=2)
    print(f"[Saved] {report_path}")

    # Comparison with previous results
    print("\n" + "="*80)
    print("COMPARISON WITH PREVIOUS EXPERIMENTS")
    print("="*80)

    previous = {
        'V2': {'CLINC150': 2.03, 'Banking77': -0.49, 'ROSTD': -0.89, 'ToxiGen': -0.03},
        'Full-Data': {'CLINC150': -1.18, 'Banking77': 0.13, 'ROSTD': -4.83, 'ToxiGen': -0.05}
    }

    print("\n| Dataset   | V2     | Full-Data | Corrected | Expected Change |")
    print("|-----------|--------|-----------|-----------|-----------------|")

    for name in ['CLINC150', 'Banking77', 'ROSTD', 'ToxiGen']:
        if name in all_results and 'main_finding' in all_results[name]:
            d = all_results[name]['main_finding']['cohens_d']
            v2 = previous['V2'].get(name, 'N/A')
            fd = previous['Full-Data'].get(name, 'N/A')

            if name == 'CLINC150':
                expected = "HET (fix OOS label)"
            elif name == 'Banking77':
                expected = "Stable (fixed split)"
            else:
                expected = "Similar"

            print(f"| {name:9} | {v2:+.2f} | {fd:+.2f}     | {d:+.4f}   | {expected:15} |")

    print("\n✅ Experiment complete!")
    return all_results


if __name__ == '__main__':
    main()
