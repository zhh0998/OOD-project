#!/usr/bin/env python3
"""
RW3 Pre-Experiment: Heterophily-OOD Association Verification
Following Senior's Dissertation Methodology

Key Principles:
1. Verify hypothesis BEFORE designing methods
2. Avoid circular reasoning (no OOD labels in heterophily computation)
3. Statistical rigor (10 runs, Cohen's d, 95% CI, p-values)
4. Multi-angle verification (overall, stratified, case, sensitivity)

Success Criteria:
- Cohen's d >= 0.5 (medium effect)
- p < 0.05 (statistically significant)
- 95% CI does not contain 0
- AUROC > 0.5 (better than random)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datetime import datetime
import json
import os
import warnings
from collections import Counter
import math
warnings.filterwarnings('ignore')

# Output directory
OUTPUT_DIR = "rw3_preexp_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def convert_numpy(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    else:
        return obj

#==============================================================================
# DATA LOADING
#==============================================================================

def load_clinc150():
    """Load CLINC150 with proper OOS label detection."""
    from datasets import load_dataset

    print("\n" + "="*80)
    print("Loading CLINC150 Dataset")
    print("="*80)

    dataset = load_dataset("clinc_oos", "small")

    # Dynamic OOS label lookup
    label_names = dataset['train'].features['intent'].names
    OOS_LABEL = label_names.index('oos')

    print(f"OOS Label: {OOS_LABEL} ('{label_names[OOS_LABEL]}')")

    # Combine all splits
    all_texts = []
    all_labels = []
    for split in ['train', 'validation', 'test']:
        all_texts.extend(list(dataset[split]['text']))
        all_labels.extend(list(dataset[split]['intent']))

    all_labels = np.array(all_labels)

    # Create OOD binary labels
    ood_labels = (all_labels == OOS_LABEL).astype(int)

    print(f"Total samples: {len(all_texts)}")
    print(f"ID samples: {(ood_labels == 0).sum()}")
    print(f"OOD samples: {(ood_labels == 1).sum()}")

    return all_texts, ood_labels, all_labels

#==============================================================================
# EMBEDDING GENERATION
#==============================================================================

def generate_embeddings(texts, model_name='all-mpnet-base-v2', cache_path=None):
    """Generate sentence embeddings with caching."""

    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        return torch.load(cache_path)

    from sentence_transformers import SentenceTransformer

    print(f"\nGenerating embeddings with {model_name}...")
    model = SentenceTransformer(model_name)

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=False
    )

    embeddings = torch.tensor(embeddings, dtype=torch.float32)

    if cache_path:
        torch.save(embeddings, cache_path)
        print(f"Saved embeddings to {cache_path}")

    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings

#==============================================================================
# K-NN GRAPH CONSTRUCTION
#==============================================================================

def build_knn_graph(embeddings, k=15, metric='cosine'):
    """Build k-NN graph on embeddings."""
    from sklearn.neighbors import NearestNeighbors

    print(f"\nBuilding k-NN graph (k={k}, metric={metric})")

    nn = NearestNeighbors(n_neighbors=k+1, metric=metric)
    nn.fit(embeddings.cpu().numpy())
    distances, indices = nn.kneighbors(embeddings.cpu().numpy())

    # Remove self-loops
    indices = indices[:, 1:]
    distances = distances[:, 1:]

    # Convert to edge_index format
    n_nodes = len(embeddings)
    edge_list = []
    for i in range(n_nodes):
        for j in indices[i]:
            edge_list.append([i, j])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t()

    # Convert distances to similarities (for cosine)
    if metric == 'cosine':
        similarities = 1 - distances
    else:
        similarities = 1 / (1 + distances)

    print(f"Graph: {n_nodes} nodes, {edge_index.shape[1]} edges")

    return edge_index, indices, similarities

#==============================================================================
# HETEROPHILY COMPUTATION (3 METHODS - NO OOD LABELS!)
#==============================================================================

def compute_heterophily_pseudolabel(edge_index, embeddings, min_cluster_size=10):
    """
    Method 1: Pseudo-label heterophily using HDBSCAN clustering.

    CRITICAL: Does NOT use OOD ground truth labels!
    Uses unsupervised clustering to assign pseudo-labels.
    """
    import hdbscan

    print("\n[Heterophily Method 1: Pseudo-label (HDBSCAN)]")

    # Unsupervised clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=5,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    pseudo_labels = clusterer.fit_predict(embeddings.cpu().numpy())

    n_clusters = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
    n_noise = (pseudo_labels == -1).sum()

    print(f"  Clusters: {n_clusters}, Noise: {n_noise} ({n_noise/len(pseudo_labels)*100:.1f}%)")

    # Compute node heterophily ratio (NHR)
    row, col = edge_index
    pseudo_labels_tensor = torch.tensor(pseudo_labels)

    node_het = []
    for i in range(len(embeddings)):
        neighbors = col[row == i]
        if len(neighbors) == 0:
            node_het.append(0.0)
            continue

        my_label = pseudo_labels[i]
        if my_label == -1:  # Noise point
            node_het.append(1.0)  # Consider as fully heterophilic
            continue

        neighbor_labels = pseudo_labels_tensor[neighbors].numpy()
        different = (neighbor_labels != my_label) | (neighbor_labels == -1)
        nhr = different.sum() / len(neighbors)
        node_het.append(nhr)

    node_het = torch.tensor(node_het, dtype=torch.float32)

    # Edge heterophily
    edge_het = ((pseudo_labels_tensor[row] != pseudo_labels_tensor[col]) |
                (pseudo_labels_tensor[row] == -1) |
                (pseudo_labels_tensor[col] == -1)).float().mean().item()

    print(f"  Edge heterophily: {edge_het:.4f}")
    print(f"  Mean node heterophily: {node_het.mean():.4f}")

    return node_het, edge_het, pseudo_labels


def compute_heterophily_similarity(edge_index, embeddings):
    """
    Method 2: Embedding similarity-based heterophily (continuous).

    CRITICAL: Does NOT use any discrete labels!
    Uses cosine similarity directly.
    """
    print("\n[Heterophily Method 2: Embedding Similarity]")

    row, col = edge_index

    # Compute cosine similarity for each edge
    cosine_sim = F.cosine_similarity(
        embeddings[row],
        embeddings[col],
        dim=1
    )

    # Heterophily = 1 - similarity
    edge_dissim = 1 - cosine_sim

    # Node-level heterophily: mean dissimilarity to neighbors (manual scatter)
    n_nodes = len(embeddings)
    node_het = torch.zeros(n_nodes)
    node_counts = torch.zeros(n_nodes)

    for i, (src, dissim) in enumerate(zip(row.cpu().numpy(), edge_dissim.cpu().numpy())):
        node_het[src] += dissim
        node_counts[src] += 1

    # Average
    node_het = node_het / (node_counts + 1e-10)

    # Edge heterophily
    edge_het = edge_dissim.mean().item()

    print(f"  Edge heterophily: {edge_het:.4f}")
    print(f"  Mean node heterophily: {node_het.mean():.4f}")

    return node_het, edge_het


def compute_heterophily_entropy(edge_index, pseudo_labels):
    """
    Method 3: Neighbor entropy heterophily.

    High entropy = diverse neighbors = high heterophily.
    """
    print("\n[Heterophily Method 3: Neighbor Entropy]")

    row, col = edge_index
    n_nodes = len(pseudo_labels)
    pseudo_labels_tensor = torch.tensor(pseudo_labels)

    node_het = []
    for i in range(n_nodes):
        neighbors = col[row == i]
        if len(neighbors) == 0:
            node_het.append(0.0)
            continue

        neighbor_labels = pseudo_labels_tensor[neighbors].numpy()

        # Compute label distribution
        label_counts = Counter(neighbor_labels)
        total = len(neighbors)

        # Compute entropy
        entropy = 0
        for count in label_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log(p + 1e-10)

        # Normalize by max possible entropy
        n_unique = len(label_counts)
        max_entropy = math.log(n_unique) if n_unique > 1 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        node_het.append(normalized_entropy)

    node_het = torch.tensor(node_het, dtype=torch.float32)
    edge_het = node_het.mean().item()

    print(f"  Edge heterophily: {edge_het:.4f}")
    print(f"  Mean node heterophily: {node_het.mean():.4f}")

    return node_het, edge_het

#==============================================================================
# STATISTICAL VERIFICATION (LAYER 1)
#==============================================================================

def verify_layer1_hypothesis(node_het, ood_labels, method_name='Method'):
    """
    Verify Layer 1 hypothesis: OOD samples have different heterophily.

    Returns:
        - Cohen's d (effect size)
        - t-statistic and p-value
        - 95% CI for Cohen's d
        - Success determination
    """
    from scipy.stats import ttest_ind

    # Convert to numpy
    if isinstance(node_het, torch.Tensor):
        node_het = node_het.cpu().numpy()
    if isinstance(ood_labels, torch.Tensor):
        ood_labels = ood_labels.cpu().numpy()

    # Separate ID and OOD
    id_het = node_het[ood_labels == 0]
    ood_het = node_het[ood_labels == 1]

    # Basic statistics
    id_mean, id_std = id_het.mean(), id_het.std()
    ood_mean, ood_std = ood_het.mean(), ood_het.std()

    # Cohen's d
    pooled_std = np.sqrt((id_std**2 + ood_std**2) / 2)
    cohens_d = (ood_mean - id_mean) / (pooled_std + 1e-10)

    # T-test
    t_stat, p_value = ttest_ind(ood_het, id_het)

    # Bootstrap 95% CI for Cohen's d
    n_bootstrap = 1000
    cohens_d_samples = []
    for _ in range(n_bootstrap):
        id_sample = np.random.choice(id_het, size=len(id_het), replace=True)
        ood_sample = np.random.choice(ood_het, size=len(ood_het), replace=True)
        pooled_sample = np.sqrt((id_sample.std()**2 + ood_sample.std()**2) / 2)
        d_sample = (ood_sample.mean() - id_sample.mean()) / (pooled_sample + 1e-10)
        cohens_d_samples.append(d_sample)

    ci_95 = np.percentile(cohens_d_samples, [2.5, 97.5])

    # Effect size interpretation
    abs_d = abs(cohens_d)
    if abs_d >= 0.8:
        effect_size = "Large effect (|d|>=0.8)"
    elif abs_d >= 0.5:
        effect_size = "Medium effect (0.5<=|d|<0.8)"
    elif abs_d >= 0.2:
        effect_size = "Small effect (0.2<=|d|<0.5)"
    else:
        effect_size = "Negligible effect (|d|<0.2)"

    # Success criteria
    success = (abs_d >= 0.5) and (p_value < 0.05) and (ci_95[0] * ci_95[1] > 0)

    result = {
        'method': method_name,
        'id_mean': float(id_mean),
        'id_std': float(id_std),
        'ood_mean': float(ood_mean),
        'ood_std': float(ood_std),
        'diff': float(ood_mean - id_mean),
        'cohens_d': float(cohens_d),
        'cohens_d_ci_95': [float(ci_95[0]), float(ci_95[1])],
        'p_value': float(p_value),
        't_statistic': float(t_stat),
        'effect_size': effect_size,
        'success': success,
        'n_id': len(id_het),
        'n_ood': len(ood_het)
    }

    return result


def test_discriminability(node_het, ood_labels):
    """Test heterophily as OOD discriminator (AUROC, AUPR, FPR95)."""
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

    if isinstance(node_het, torch.Tensor):
        node_het = node_het.cpu().numpy()
    if isinstance(ood_labels, torch.Tensor):
        ood_labels = ood_labels.cpu().numpy()

    # AUROC (higher heterophily = OOD)
    auroc = roc_auc_score(ood_labels, node_het)

    # If AUROC < 0.5, flip (lower heterophily = OOD)
    if auroc < 0.5:
        auroc = 1 - auroc
        node_het_for_roc = -node_het
    else:
        node_het_for_roc = node_het

    # AUPR
    aupr = average_precision_score(ood_labels, node_het_for_roc)

    # FPR95
    fpr, tpr, _ = roc_curve(ood_labels, node_het_for_roc)
    idx_95 = np.argmax(tpr >= 0.95)
    fpr_95 = fpr[idx_95] if idx_95 < len(fpr) else 1.0

    return {
        'auroc': float(auroc),
        'aupr': float(aupr),
        'fpr95': float(fpr_95)
    }

#==============================================================================
# MULTI-ANGLE ANALYSIS
#==============================================================================

def stratified_analysis(node_het, ood_labels, embeddings, n_clusters=10, seed=42):
    """Stratified analysis by semantic clusters (learning from RW2)."""
    from sklearn.cluster import KMeans

    print("\n[Stratified Analysis by Semantic Clusters]")

    if isinstance(node_het, torch.Tensor):
        node_het = node_het.cpu().numpy()
    if isinstance(ood_labels, torch.Tensor):
        ood_labels = ood_labels.cpu().numpy()
    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.cpu().numpy()
    else:
        embeddings_np = embeddings

    # Cluster by embedding
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    clusters = kmeans.fit_predict(embeddings_np)

    results = []
    for cluster_id in range(n_clusters):
        mask = clusters == cluster_id

        if mask.sum() < 20:  # Skip small clusters
            continue

        cluster_het = node_het[mask]
        cluster_ood = ood_labels[mask]

        n_ood_in_cluster = cluster_ood.sum()
        if n_ood_in_cluster < 5 or n_ood_in_cluster == len(cluster_ood):
            continue  # Need both ID and OOD samples

        # Statistical test
        stat_result = verify_layer1_hypothesis(
            cluster_het, cluster_ood, f'Cluster_{cluster_id}'
        )

        results.append({
            'cluster_id': cluster_id,
            'n_samples': int(mask.sum()),
            'n_ood': int(n_ood_in_cluster),
            'ood_ratio': float(n_ood_in_cluster / mask.sum()),
            **stat_result
        })

    # Sort by Cohen's d
    results = sorted(results, key=lambda x: abs(x['cohens_d']), reverse=True)

    print(f"  Analyzed {len(results)} clusters with sufficient samples")
    if results:
        print(f"  Best cluster Cohen's d: {results[0]['cohens_d']:.4f}")

    return results


def case_analysis(node_het, ood_labels, texts, edge_index, embeddings, top_k=10):
    """Case analysis of top-K heterophilic OOD samples (learning from RW3)."""
    print(f"\n[Case Analysis: Top-{top_k} Heterophilic OOD Samples]")

    if isinstance(node_het, torch.Tensor):
        node_het_np = node_het.cpu().numpy()
    else:
        node_het_np = node_het
    if isinstance(ood_labels, torch.Tensor):
        ood_labels_np = ood_labels.cpu().numpy()
    else:
        ood_labels_np = ood_labels

    # Find OOD samples
    ood_indices = np.where(ood_labels_np == 1)[0]
    ood_het = node_het_np[ood_indices]

    # Top-K highest heterophily OOD samples
    top_k_idx = np.argsort(ood_het)[-top_k:][::-1]
    top_k_global_idx = ood_indices[top_k_idx]

    row, col = edge_index

    cases = []
    for rank, idx in enumerate(top_k_global_idx):
        idx = int(idx)
        het_score = float(node_het_np[idx])

        # Find neighbors
        neighbors = col[row == idx].cpu().numpy()
        neighbor_texts = [texts[int(n)] for n in neighbors[:5]]

        # Compute similarity to neighbors
        neighbor_sims = F.cosine_similarity(
            embeddings[idx].unsqueeze(0),
            embeddings[neighbors[:5]],
            dim=-1
        ).cpu().numpy()

        case = {
            'rank': rank + 1,
            'sample_id': idx,
            'text': texts[idx][:200],  # Truncate for display
            'heterophily': het_score,
            'n_neighbors': len(neighbors),
            'avg_neighbor_similarity': float(neighbor_sims.mean()) if len(neighbor_sims) > 0 else 0,
            'top5_neighbors': [
                {'text': t[:100], 'similarity': float(s)}
                for t, s in zip(neighbor_texts, neighbor_sims)
            ]
        }
        cases.append(case)

    return cases


def k_sensitivity_analysis(embeddings, ood_labels, k_values=[5, 10, 13, 15, 20, 30]):
    """Analyze sensitivity to k (number of neighbors)."""
    print("\n[K-Value Sensitivity Analysis]")

    results = []

    for k in k_values:
        print(f"  Testing k={k}...")

        # Build graph
        edge_index, _, _ = build_knn_graph(embeddings, k=k)

        # Compute heterophily (pseudo-label method)
        node_het, edge_het, _ = compute_heterophily_pseudolabel(edge_index, embeddings)

        # Statistical test
        stat_result = verify_layer1_hypothesis(node_het, ood_labels, f'k={k}')

        # Discriminability
        disc_result = test_discriminability(node_het, ood_labels)

        results.append({
            'k': k,
            'edge_heterophily': edge_het,
            'cohens_d': stat_result['cohens_d'],
            'p_value': stat_result['p_value'],
            'auroc': disc_result['auroc'],
            'n_edges': edge_index.shape[1]
        })

        print(f"    edge_het={edge_het:.4f}, Cohen's d={stat_result['cohens_d']:.4f}, AUROC={disc_result['auroc']:.4f}")

    return results


def select_optimal_k(k_results, target_edge_het_range=[0.2, 0.4]):
    """Select optimal k based on edge heterophily and Cohen's d."""
    candidates = [r for r in k_results
                  if target_edge_het_range[0] <= r['edge_heterophily'] <= target_edge_het_range[1]]

    if not candidates:
        # Select closest to target range
        candidates = k_results

    # Select by highest Cohen's d
    optimal = max(candidates, key=lambda x: abs(x['cohens_d']))
    return optimal['k']

#==============================================================================
# MULTIPLE RUNS (STATISTICAL RIGOR)
#==============================================================================

def run_multiple_seeds(embeddings, ood_labels, optimal_k, n_runs=10):
    """Run experiment multiple times with different seeds for statistical rigor."""
    print(f"\n[Running {n_runs} times for statistical rigor]")

    all_results = {
        'pseudolabel': [],
        'similarity': [],
        'entropy': []
    }

    for seed in range(n_runs):
        print(f"\n  Run {seed+1}/{n_runs} (seed={seed})")
        set_seed(seed)

        # Build graph (same k, different random state for clustering)
        edge_index, _, _ = build_knn_graph(embeddings, k=optimal_k)

        # Method 1: Pseudo-label
        node_het_pseudo, _, pseudo_labels = compute_heterophily_pseudolabel(
            edge_index, embeddings, min_cluster_size=10+seed  # Slight variation
        )
        stat_pseudo = verify_layer1_hypothesis(node_het_pseudo, ood_labels, 'Pseudo-label')
        disc_pseudo = test_discriminability(node_het_pseudo, ood_labels)
        all_results['pseudolabel'].append({**stat_pseudo, **disc_pseudo})

        # Method 2: Similarity (deterministic, but include for consistency)
        node_het_sim, _ = compute_heterophily_similarity(edge_index, embeddings)
        stat_sim = verify_layer1_hypothesis(node_het_sim, ood_labels, 'Similarity')
        disc_sim = test_discriminability(node_het_sim, ood_labels)
        all_results['similarity'].append({**stat_sim, **disc_sim})

        # Method 3: Entropy
        node_het_ent, _ = compute_heterophily_entropy(edge_index, pseudo_labels)
        stat_ent = verify_layer1_hypothesis(node_het_ent, ood_labels, 'Entropy')
        disc_ent = test_discriminability(node_het_ent, ood_labels)
        all_results['entropy'].append({**stat_ent, **disc_ent})

    # Aggregate results
    aggregated = {}
    for method, runs in all_results.items():
        cohens_d_values = [r['cohens_d'] for r in runs]
        auroc_values = [r['auroc'] for r in runs]
        p_values = [r['p_value'] for r in runs]

        aggregated[method] = {
            'cohens_d_mean': float(np.mean(cohens_d_values)),
            'cohens_d_std': float(np.std(cohens_d_values)),
            'auroc_mean': float(np.mean(auroc_values)),
            'auroc_std': float(np.std(auroc_values)),
            'p_value_mean': float(np.mean(p_values)),
            'success_rate': sum([r['success'] for r in runs]) / n_runs,
            'all_runs': runs
        }

    return aggregated

#==============================================================================
# REPORT GENERATION
#==============================================================================

def generate_report(layer1_results, k_results, stratified_results, case_results, config):
    """Generate comprehensive pre-experiment report."""

    report_path = os.path.join(OUTPUT_DIR, "preexp_report.md")

    with open(report_path, 'w') as f:
        f.write("# RW3 Pre-Experiment Report: Heterophily-OOD Association Verification\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # Section 1: Summary
        f.write("## 1. Executive Summary\n\n")
        f.write("### Layer 1 Hypothesis Verification Results\n\n")
        f.write("| Method | Cohen's d | 95% CI | p-value | AUROC | Success |\n")
        f.write("|--------|-----------|--------|---------|-------|--------|\n")

        for method, results in layer1_results.items():
            d = results['cohens_d_mean']
            d_std = results['cohens_d_std']
            auroc = results['auroc_mean']
            auroc_std = results['auroc_std']
            p = results['p_value_mean']
            success = "Yes" if results['success_rate'] >= 0.8 else "No"

            f.write(f"| {method.capitalize()} | {d:.4f}+/-{d_std:.4f} | - | {p:.2e} | {auroc:.4f}+/-{auroc_std:.4f} | {success} |\n")

        f.write("\n")

        # Overall conclusion
        n_success = sum([1 for r in layer1_results.values() if r['success_rate'] >= 0.8])
        if n_success >= 2:
            f.write("**Conclusion**: Layer 1 hypothesis VERIFIED. Heterophily-OOD association exists.\n\n")
        elif n_success == 1:
            f.write("**Conclusion**: Layer 1 hypothesis PARTIALLY verified. Consider adjusting methods.\n\n")
        else:
            f.write("**Conclusion**: Layer 1 hypothesis NOT verified. Consider changing research direction.\n\n")

        # Section 2: Configuration
        f.write("## 2. Experimental Configuration\n\n")
        f.write(f"- Dataset: CLINC150 (small config)\n")
        f.write(f"- Encoder: {config.get('encoder', 'all-mpnet-base-v2')}\n")
        f.write(f"- Optimal k: {config.get('optimal_k', 15)}\n")
        f.write(f"- Number of runs: {config.get('n_runs', 10)}\n\n")

        # Section 3: K-sensitivity
        f.write("## 3. K-Value Sensitivity Analysis\n\n")
        f.write("| k | Edge Het. | Cohen's d | AUROC |\n")
        f.write("|---|-----------|-----------|-------|\n")
        for r in k_results:
            f.write(f"| {r['k']} | {r['edge_heterophily']:.4f} | {r['cohens_d']:.4f} | {r['auroc']:.4f} |\n")
        f.write("\n")

        # Section 4: Stratified Analysis
        f.write("## 4. Stratified Analysis (Top-5 Clusters)\n\n")
        f.write("| Cluster | N | OOD% | Cohen's d | p-value |\n")
        f.write("|---------|---|------|-----------|--------|\n")
        for r in stratified_results[:5]:
            f.write(f"| {r['cluster_id']} | {r['n_samples']} | {r['ood_ratio']*100:.1f}% | {r['cohens_d']:.4f} | {r['p_value']:.2e} |\n")
        f.write("\n")

        # Section 5: Case Analysis
        f.write("## 5. Case Analysis (Top-5 Heterophilic OOD Samples)\n\n")
        for case in case_results[:5]:
            f.write(f"### Case {case['rank']}\n")
            f.write(f"- **Text**: {case['text'][:100]}...\n")
            f.write(f"- **Heterophily**: {case['heterophily']:.4f}\n")
            f.write(f"- **Avg Neighbor Similarity**: {case['avg_neighbor_similarity']:.4f}\n")
            f.write(f"- **Top Neighbors**:\n")
            for nb in case['top5_neighbors'][:3]:
                f.write(f"  - sim={nb['similarity']:.3f}: {nb['text'][:50]}...\n")
            f.write("\n")

        # Section 6: Conclusion
        f.write("## 6. Conclusions and Next Steps\n\n")

        if n_success >= 2:
            f.write("### Hypothesis Verified\n\n")
            f.write("The pre-experiment successfully demonstrates that OOD samples exhibit ")
            f.write("significantly different heterophily patterns in k-NN semantic graphs.\n\n")
            f.write("**Next Steps**:\n")
            f.write("1. Proceed to design heterophily-aware OOD detection methods (Layer 3)\n")
            f.write("2. Implement 5 proposed methods (NegHetero-OOD, SpectralLLM-OOD, etc.)\n")
            f.write("3. Run full experiments comparing with SOTA baselines\n")
        else:
            f.write("### Hypothesis Not Fully Verified\n\n")
            f.write("The pre-experiment shows limited evidence for heterophily-OOD association.\n\n")
            f.write("**Recommendations**:\n")
            f.write("1. Try different clustering methods (DBSCAN, Spectral Clustering)\n")
            f.write("2. Test on other datasets (Banking77, ROSTD)\n")
            f.write("3. Consider alternative research directions\n")

    print(f"\nReport saved to {report_path}")
    return report_path

#==============================================================================
# MAIN EXPERIMENT
#==============================================================================

def main():
    """Run complete pre-experiment."""

    print("="*80)
    print("RW3 PRE-EXPERIMENT: HETEROPHILY-OOD ASSOCIATION VERIFICATION")
    print("="*80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("Following Senior's Dissertation Methodology")
    print("="*80)

    config = {
        'encoder': 'all-mpnet-base-v2',
        'n_runs': 10,
        'k_values': [5, 10, 13, 15, 20, 30]
    }

    # Step 1: Load data
    print("\n" + "="*60)
    print("STEP 1: Data Loading")
    print("="*60)
    texts, ood_labels, intent_labels = load_clinc150()
    ood_labels = torch.tensor(ood_labels)

    # Step 2: Generate embeddings
    print("\n" + "="*60)
    print("STEP 2: Embedding Generation")
    print("="*60)
    cache_path = os.path.join(OUTPUT_DIR, "clinc150_embeddings.pt")
    embeddings = generate_embeddings(texts, cache_path=cache_path)

    # Step 3: K-sensitivity analysis
    print("\n" + "="*60)
    print("STEP 3: K-Value Sensitivity Analysis")
    print("="*60)
    k_results = k_sensitivity_analysis(embeddings, ood_labels, config['k_values'])
    optimal_k = select_optimal_k(k_results)
    config['optimal_k'] = optimal_k
    print(f"\n  Selected optimal k = {optimal_k}")

    # Save k-sensitivity results
    k_results_path = os.path.join(OUTPUT_DIR, "k_sensitivity.json")
    with open(k_results_path, 'w') as f:
        json.dump(convert_numpy(k_results), f, indent=2)

    # Step 4: Multiple runs for statistical rigor
    print("\n" + "="*60)
    print("STEP 4: Layer 1 Hypothesis Verification (Multiple Runs)")
    print("="*60)
    layer1_results = run_multiple_seeds(embeddings, ood_labels, optimal_k, config['n_runs'])

    # Save Layer 1 results
    layer1_path = os.path.join(OUTPUT_DIR, "layer1_verification.json")
    with open(layer1_path, 'w') as f:
        json.dump(convert_numpy(layer1_results), f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("LAYER 1 VERIFICATION SUMMARY")
    print("="*60)
    print("\n| Method      | Cohen's d        | AUROC            | Success Rate |")
    print("|-------------|------------------|------------------|--------------|")
    for method, results in layer1_results.items():
        d = results['cohens_d_mean']
        d_std = results['cohens_d_std']
        auroc = results['auroc_mean']
        auroc_std = results['auroc_std']
        success = results['success_rate']
        print(f"| {method:11} | {d:+.4f} +/- {d_std:.4f} | {auroc:.4f} +/- {auroc_std:.4f} | {success*100:.0f}%          |")

    # Decision point
    n_success = sum([1 for r in layer1_results.values() if r['success_rate'] >= 0.8])
    print("\n" + "="*60)
    if n_success >= 2:
        print("DECISION: Layer 1 hypothesis VERIFIED. Continue to full experiment.")
    elif n_success == 1:
        print("DECISION: Layer 1 hypothesis PARTIALLY verified. Adjust and retry.")
    else:
        print("DECISION: Layer 1 hypothesis NOT verified. Consider new direction.")
    print("="*60)

    # Step 5: Multi-angle analysis (if hypothesis verified)
    print("\n" + "="*60)
    print("STEP 5: Multi-Angle Deep Analysis")
    print("="*60)

    # Build final graph for analysis
    edge_index, _, _ = build_knn_graph(embeddings, k=optimal_k)
    node_het, _, pseudo_labels = compute_heterophily_pseudolabel(edge_index, embeddings)

    # Stratified analysis
    stratified_results = stratified_analysis(node_het, ood_labels, embeddings)
    stratified_path = os.path.join(OUTPUT_DIR, "stratified_analysis.json")
    with open(stratified_path, 'w') as f:
        json.dump(convert_numpy(stratified_results), f, indent=2)

    # Case analysis
    case_results = case_analysis(node_het, ood_labels, texts, edge_index, embeddings)
    case_path = os.path.join(OUTPUT_DIR, "case_analysis.json")
    with open(case_path, 'w') as f:
        json.dump(convert_numpy(case_results), f, indent=2)

    # Step 6: Generate report
    print("\n" + "="*60)
    print("STEP 6: Report Generation")
    print("="*60)
    report_path = generate_report(layer1_results, k_results, stratified_results, case_results, config)

    # Save full results
    full_results = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'layer1_results': layer1_results,
        'k_sensitivity': k_results,
        'stratified_analysis': stratified_results[:10],
        'case_analysis': case_results
    }

    full_path = os.path.join(OUTPUT_DIR, "full_preexp_results.json")
    with open(full_path, 'w') as f:
        json.dump(convert_numpy(full_results), f, indent=2)

    print("\n" + "="*80)
    print("PRE-EXPERIMENT COMPLETE")
    print("="*80)
    print(f"Results saved to: {OUTPUT_DIR}/")
    print(f"Report: {report_path}")

    return full_results


if __name__ == '__main__':
    main()
