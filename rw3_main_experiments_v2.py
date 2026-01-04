#!/usr/bin/env python3
"""
RW3 Main Experiments V2 - Corrected Version
============================================

Key Corrections:
1. Reproduce pre-experiment's 96.46% AUROC with EmbeddingSimilarity
2. Add 10+ baseline methods (including MSP, Energy, MaxLogit)
3. Use real OOD datasets (Banking77-OOS, ROSTD)
4. Staged NegHetero-OOD implementation
"""

import os
import json
import random
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.covariance import EmpiricalCovariance
from sentence_transformers import SentenceTransformer

# Set seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'encoder': 'sentence-transformers/all-mpnet-base-v2',
    'batch_size': 64,
    'output_dir': Path('rw3_experiments_v2'),

    # Model hyperparameters
    'hidden_dim': 256,
    'output_dim': 128,
    'k_neighbors': 10,
    'n_epochs': 10,
    'lr': 1e-3,

    # Success thresholds
    'stage1_min_auroc': 90.0,  # Must reproduce pre-experiment
    'stage2_target_auroc': 92.0,  # Heterophily enhancement target
}

# Create directories
CONFIG['output_dir'].mkdir(exist_ok=True)
(CONFIG['output_dir'] / 'data').mkdir(exist_ok=True)
(CONFIG['output_dir'] / 'results').mkdir(exist_ok=True)
(CONFIG['output_dir'] / 'models').mkdir(exist_ok=True)

# Initialize encoder
print(f"Loading encoder: {CONFIG['encoder']}")
ENCODER = SentenceTransformer(CONFIG['encoder'])

# ============================================================================
# Part 1: Data Preparation (Real Datasets Only)
# ============================================================================

def load_banking77_oos():
    """
    Load REAL Banking77-OOS dataset.
    This is the critical near-OOD benchmark.
    """
    print("\n[Data] Loading Banking77-OOS (Real Dataset)...")

    base_path = Path('rw3_main_experiments/data/banking77_oos')

    if not base_path.exists():
        raise FileNotFoundError(
            "Banking77-OOS not found. Run:\n"
            "git clone https://github.com/jianguoz/Few-Shot-Intent-Detection tmp_fewshot\n"
            "cp -r tmp_fewshot/Datasets/BANKING77-OOS rw3_main_experiments/data/banking77_oos"
        )

    def read_data(split_path):
        texts_file = split_path / 'seq.in'
        labels_file = split_path / 'label'

        with open(texts_file, 'r') as f:
            texts = [line.strip() for line in f]
        with open(labels_file, 'r') as f:
            labels = [line.strip() for line in f]

        return texts, labels

    # Load training data (ID only)
    train_texts, train_labels_str = read_data(base_path / 'train')

    # Create label mapping
    unique_labels = sorted(set(train_labels_str))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    train_labels = [label2id[l] for l in train_labels_str]

    # Load test ID data
    test_texts, test_labels_str = read_data(base_path / 'test')
    test_id_texts = [t for t, l in zip(test_texts, test_labels_str) if l != 'oos']
    test_id_labels = [label2id[l] for t, l in zip(test_texts, test_labels_str) if l != 'oos' and l in label2id]

    # Load ID-OOS (Near-OOD) - This is the key benchmark!
    id_oos_texts, id_oos_labels = read_data(base_path / 'id-oos' / 'test')
    test_ood_texts = [t for t, l in zip(id_oos_texts, id_oos_labels) if l == 'oos']

    # Also load OOD-OOS (Far-OOD) for comparison
    ood_oos_texts, _ = read_data(base_path / 'ood-oos' / 'test')

    print(f"  Train ID: {len(train_texts)} samples, {len(unique_labels)} classes")
    print(f"  Test ID: {len(test_id_texts)} samples")
    print(f"  Test OOD (Near-OOD): {len(test_ood_texts)} samples")
    print(f"  Test OOD (Far-OOD): {len(ood_oos_texts)} samples")

    return {
        'train_texts': train_texts,
        'train_labels': train_labels,
        'test_id_texts': test_id_texts,
        'test_ood_texts': test_ood_texts,  # Near-OOD (harder!)
        'test_ood_far_texts': ood_oos_texts,  # Far-OOD
        'n_classes': len(unique_labels),
        'label2id': label2id
    }


def load_clinc150():
    """Load CLINC150 dataset (already working)."""
    print("\n[Data] Loading CLINC150...")

    try:
        from datasets import load_dataset
        dataset = load_dataset("clinc_oos", "plus")

        # Process splits
        train_texts = [s['text'] for s in dataset['train'] if s['intent'] != 150]
        train_labels = [s['intent'] for s in dataset['train'] if s['intent'] != 150]

        test_all = dataset['test']
        test_id_texts = [s['text'] for s in test_all if s['intent'] != 150]
        test_id_labels = [s['intent'] for s in test_all if s['intent'] != 150]

        # Collect OOD from test and validation
        test_ood_texts = [s['text'] for s in test_all if s['intent'] == 150]
        val_ood_texts = [s['text'] for s in dataset['validation'] if s['intent'] == 150]
        test_ood_texts = test_ood_texts + val_ood_texts

        print(f"  Train ID: {len(train_texts)} samples")
        print(f"  Test ID: {len(test_id_texts)} samples")
        print(f"  Test OOD: {len(test_ood_texts)} samples")

        return {
            'train_texts': train_texts,
            'train_labels': train_labels,
            'test_id_texts': test_id_texts,
            'test_ood_texts': test_ood_texts,
            'n_classes': 150
        }

    except Exception as e:
        print(f"  Error loading CLINC150: {e}")
        raise


def generate_embeddings(texts: List[str], cache_path: Optional[Path] = None):
    """Generate embeddings with caching."""

    if cache_path and cache_path.exists():
        print(f"  Loading cached embeddings from {cache_path}")
        return torch.load(cache_path)

    print(f"  Generating embeddings for {len(texts)} texts...")
    embeddings = ENCODER.encode(texts, convert_to_tensor=True, show_progress_bar=True)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, cache_path)
        print(f"  Cached to {cache_path}")

    return embeddings


# ============================================================================
# Part 2: Evaluation Metrics
# ============================================================================

def compute_metrics(scores_id: np.ndarray, scores_ood: np.ndarray) -> Dict:
    """Compute OOD detection metrics."""
    scores = np.concatenate([scores_id, scores_ood])
    labels = np.concatenate([np.zeros(len(scores_id)), np.ones(len(scores_ood))])

    # AUROC
    try:
        auroc = roc_auc_score(labels, scores) * 100
    except:
        auroc = 50.0

    # FPR@95
    fpr95 = compute_fpr_at_tpr(labels, scores, 0.95) * 100

    # AUPR-Out
    try:
        precision, recall, _ = precision_recall_curve(labels, scores)
        aupr_out = auc(recall, precision) * 100
    except:
        aupr_out = 50.0

    return {
        'auroc': auroc,
        'fpr95': fpr95,
        'aupr_out': aupr_out
    }


def compute_fpr_at_tpr(labels, scores, tpr_target=0.95):
    """Compute FPR at given TPR."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.argmin(np.abs(tpr - tpr_target))
    return fpr[idx]


# ============================================================================
# Part 3: Baseline Methods (10+ methods)
# ============================================================================

class EmbeddingSimilarityOOD:
    """
    CRITICAL: Reproduce pre-experiment's 96.46% AUROC.

    This is the validated baseline that must be matched before
    any enhancement is attempted.

    Pre-experiment results on CLINC150:
    - Cohen's d: 2.78
    - AUROC: 96.46%
    """

    def __init__(self, k: int = 10):
        self.k = k
        self.train_embeddings = None

    def fit(self, train_embeddings: torch.Tensor):
        """Store normalized training embeddings."""
        if isinstance(train_embeddings, torch.Tensor):
            train_embeddings = train_embeddings.cpu()
        self.train_embeddings = F.normalize(train_embeddings, p=2, dim=1)

    def score(self, test_embeddings: torch.Tensor) -> np.ndarray:
        """
        Compute OOD scores based on average similarity to k-nearest neighbors.

        OOD samples have LOWER similarity to training data → HIGHER OOD score.
        """
        if isinstance(test_embeddings, torch.Tensor):
            test_embeddings = test_embeddings.cpu()

        test_emb_norm = F.normalize(test_embeddings, p=2, dim=1)

        # Compute cosine similarity to all training samples
        similarities = torch.mm(test_emb_norm, self.train_embeddings.t())  # (n_test, n_train)

        # Get top-k similarities (highest = most similar)
        top_k_sim, _ = torch.topk(similarities, k=self.k, dim=1)
        avg_similarity = top_k_sim.mean(dim=1)

        # OOD score: low similarity = high OOD
        ood_scores = 1 - avg_similarity

        return ood_scores.numpy()


class MahalanobisDetector:
    """Mahalanobis distance detector (Lee et al., NeurIPS 2018)."""

    def __init__(self):
        self.mean = None
        self.precision = None

    def fit(self, train_embeddings: torch.Tensor):
        if isinstance(train_embeddings, torch.Tensor):
            train_embeddings = train_embeddings.cpu().numpy()

        self.mean = np.mean(train_embeddings, axis=0)
        cov = EmpiricalCovariance().fit(train_embeddings)
        self.precision = cov.precision_

    def score(self, embeddings: torch.Tensor) -> np.ndarray:
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        diff = embeddings - self.mean
        mahal = np.sqrt(np.sum(np.dot(diff, self.precision) * diff, axis=1))
        return mahal


class KNNDetector:
    """KNN distance detector (Sun et al., ICML 2022)."""

    def __init__(self, k: int = 10):
        self.k = k
        self.knn = None

    def fit(self, train_embeddings: torch.Tensor):
        if isinstance(train_embeddings, torch.Tensor):
            train_embeddings = train_embeddings.cpu().numpy()

        # Normalize
        train_embeddings = train_embeddings / (np.linalg.norm(train_embeddings, axis=1, keepdims=True) + 1e-8)

        self.knn = NearestNeighbors(n_neighbors=self.k, metric='cosine')
        self.knn.fit(train_embeddings)

    def score(self, embeddings: torch.Tensor) -> np.ndarray:
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        distances, _ = self.knn.kneighbors(embeddings)
        return distances.mean(axis=1)


class LOFDetector:
    """Local Outlier Factor detector."""

    def __init__(self, n_neighbors: int = 20):
        self.n_neighbors = n_neighbors
        self.lof = None

    def fit(self, train_embeddings: torch.Tensor):
        if isinstance(train_embeddings, torch.Tensor):
            train_embeddings = train_embeddings.cpu().numpy()

        train_embeddings = train_embeddings / (np.linalg.norm(train_embeddings, axis=1, keepdims=True) + 1e-8)

        self.lof = LocalOutlierFactor(n_neighbors=self.n_neighbors, novelty=True, metric='cosine')
        self.lof.fit(train_embeddings)

    def score(self, embeddings: torch.Tensor) -> np.ndarray:
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        return -self.lof.score_samples(embeddings)


class CosineDistanceDetector:
    """Average cosine distance to training centroids."""

    def __init__(self):
        self.centroid = None

    def fit(self, train_embeddings: torch.Tensor):
        if isinstance(train_embeddings, torch.Tensor):
            train_embeddings = train_embeddings.cpu().numpy()

        train_embeddings = train_embeddings / (np.linalg.norm(train_embeddings, axis=1, keepdims=True) + 1e-8)
        self.centroid = np.mean(train_embeddings, axis=0)
        self.centroid = self.centroid / (np.linalg.norm(self.centroid) + 1e-8)

    def score(self, embeddings: torch.Tensor) -> np.ndarray:
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        similarities = np.dot(embeddings, self.centroid)
        return 1 - similarities


# Classifier-based methods (MSP, Energy, MaxLogit)

class SimpleClassifier(nn.Module):
    """Simple linear classifier for MSP/Energy/MaxLogit."""

    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        return self.fc(x)


def train_classifier(train_embeddings: torch.Tensor, train_labels: List[int],
                     n_classes: int, n_epochs: int = 5) -> SimpleClassifier:
    """Train a simple classifier for MSP/Energy/MaxLogit methods."""

    device = CONFIG['device']

    if isinstance(train_embeddings, torch.Tensor):
        train_embeddings = train_embeddings.clone().to(device)
    else:
        train_embeddings = torch.tensor(train_embeddings, dtype=torch.float32).to(device)

    train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)

    input_dim = train_embeddings.shape[1]
    classifier = SimpleClassifier(input_dim, n_classes).to(device)

    optimizer = Adam(classifier.parameters(), lr=2e-4)
    criterion = nn.CrossEntropyLoss()

    classifier.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        logits = classifier(train_embeddings)
        loss = criterion(logits, train_labels)
        loss.backward()
        optimizer.step()

    classifier.eval()
    return classifier


class MSPDetector:
    """Maximum Softmax Probability (Hendrycks & Gimpel, ICLR 2017)."""

    def __init__(self, classifier: SimpleClassifier):
        self.classifier = classifier

    def score(self, embeddings: torch.Tensor) -> np.ndarray:
        device = next(self.classifier.parameters()).device

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.clone().to(device)
        else:
            embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = self.classifier(embeddings)
            probs = F.softmax(logits, dim=1)
            max_probs = probs.max(dim=1)[0]

        # Low confidence = OOD
        return (1 - max_probs).cpu().numpy()


class EnergyDetector:
    """Energy-based OOD detection (Liu et al., NeurIPS 2020)."""

    def __init__(self, classifier: SimpleClassifier, T: float = 1.0):
        self.classifier = classifier
        self.T = T

    def score(self, embeddings: torch.Tensor) -> np.ndarray:
        device = next(self.classifier.parameters()).device

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.clone().to(device)
        else:
            embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = self.classifier(embeddings)
            energy = -self.T * torch.logsumexp(logits / self.T, dim=1)

        # Higher energy = OOD (negate for consistency)
        return (-energy).cpu().numpy()


class MaxLogitDetector:
    """MaxLogit OOD detection (Hendrycks et al., ICML 2022)."""

    def __init__(self, classifier: SimpleClassifier):
        self.classifier = classifier

    def score(self, embeddings: torch.Tensor) -> np.ndarray:
        device = next(self.classifier.parameters()).device

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.clone().to(device)
        else:
            embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = self.classifier(embeddings)
            max_logits = logits.max(dim=1)[0]

        # Low max logit = OOD
        return (-max_logits).cpu().numpy()


# ============================================================================
# Part 4: Staged NegHetero-OOD Implementation
# ============================================================================

class HeterophilyEnhancedOOD(EmbeddingSimilarityOOD):
    """
    Stage 2: Heterophily-enhanced OOD detection.

    Builds on EmbeddingSimilarity (Stage 1) with heterophily modulation.
    Must maintain Stage 1 performance as baseline.
    """

    def __init__(self, k: int = 10, alpha: float = 0.3, use_heterophily: bool = True):
        super().__init__(k=k)
        self.alpha = alpha  # Heterophily weight (conservative)
        self.use_heterophily = use_heterophily
        self.train_labels = None

    def fit(self, train_embeddings: torch.Tensor, train_labels: Optional[List[int]] = None):
        super().fit(train_embeddings)
        if train_labels is not None:
            self.train_labels = torch.tensor(train_labels)

    def score(self, test_embeddings: torch.Tensor) -> np.ndarray:
        # Stage 1: Base embedding similarity score
        base_scores = super().score(test_embeddings)

        if not self.use_heterophily or self.train_labels is None:
            return base_scores

        # Stage 2: Heterophily enhancement
        if isinstance(test_embeddings, torch.Tensor):
            test_embeddings = test_embeddings.cpu()

        test_emb_norm = F.normalize(test_embeddings, p=2, dim=1)

        # Find k-nearest neighbors
        similarities = torch.mm(test_emb_norm, self.train_embeddings.t())
        _, top_k_indices = torch.topk(similarities, k=self.k, dim=1)

        # Compute neighbor heterophily
        heterophily_scores = self._compute_heterophily(top_k_indices)

        # Combine: base_score + alpha * heterophily
        # Higher heterophily → higher OOD score
        enhanced_scores = base_scores + self.alpha * heterophily_scores

        return enhanced_scores

    def _compute_heterophily(self, neighbor_indices: torch.Tensor) -> np.ndarray:
        """
        Compute label heterophily of k-nearest neighbors.

        High heterophily = neighbors from diverse classes = potentially OOD
        """
        n_test = neighbor_indices.shape[0]
        heterophily = np.zeros(n_test)

        for i in range(n_test):
            neighbor_labels = self.train_labels[neighbor_indices[i]]
            unique_labels = torch.unique(neighbor_labels)
            # Normalize by k
            heterophily[i] = len(unique_labels) / self.k

        return heterophily


# ============================================================================
# Part 5: Main Experiment Runner
# ============================================================================

def run_all_baselines(dataset_name: str, train_emb: torch.Tensor, train_labels: List[int],
                      test_id_emb: torch.Tensor, test_ood_emb: torch.Tensor,
                      n_classes: int) -> Dict:
    """Run all baseline methods."""

    print(f"\n{'='*60}")
    print(f"Running Baselines on {dataset_name}")
    print(f"{'='*60}")

    results = {}

    # 1. EmbeddingSimilarity (CRITICAL - must reproduce pre-experiment)
    print("\n[1/10] EmbeddingSimilarity (Pre-experiment baseline)...")
    emb_sim = EmbeddingSimilarityOOD(k=10)
    emb_sim.fit(train_emb)
    scores_id = emb_sim.score(test_id_emb)
    scores_ood = emb_sim.score(test_ood_emb)
    results['EmbeddingSimilarity'] = compute_metrics(scores_id, scores_ood)
    print(f"  AUROC: {results['EmbeddingSimilarity']['auroc']:.2f}%")

    # 2. Mahalanobis
    print("\n[2/10] Mahalanobis...")
    mahal = MahalanobisDetector()
    mahal.fit(train_emb)
    scores_id = mahal.score(test_id_emb)
    scores_ood = mahal.score(test_ood_emb)
    results['Mahalanobis'] = compute_metrics(scores_id, scores_ood)
    print(f"  AUROC: {results['Mahalanobis']['auroc']:.2f}%")

    # 3. KNN-10
    print("\n[3/10] KNN (k=10)...")
    knn10 = KNNDetector(k=10)
    knn10.fit(train_emb)
    scores_id = knn10.score(test_id_emb)
    scores_ood = knn10.score(test_ood_emb)
    results['KNN-10'] = compute_metrics(scores_id, scores_ood)
    print(f"  AUROC: {results['KNN-10']['auroc']:.2f}%")

    # 4. KNN-200
    print("\n[4/10] KNN (k=200)...")
    knn200 = KNNDetector(k=200)
    knn200.fit(train_emb)
    scores_id = knn200.score(test_id_emb)
    scores_ood = knn200.score(test_ood_emb)
    results['KNN-200'] = compute_metrics(scores_id, scores_ood)
    print(f"  AUROC: {results['KNN-200']['auroc']:.2f}%")

    # 5. LOF
    print("\n[5/10] LOF...")
    lof = LOFDetector(n_neighbors=20)
    lof.fit(train_emb)
    scores_id = lof.score(test_id_emb)
    scores_ood = lof.score(test_ood_emb)
    results['LOF'] = compute_metrics(scores_id, scores_ood)
    print(f"  AUROC: {results['LOF']['auroc']:.2f}%")

    # 6. Cosine Distance
    print("\n[6/10] CosineDistance...")
    cosine = CosineDistanceDetector()
    cosine.fit(train_emb)
    scores_id = cosine.score(test_id_emb)
    scores_ood = cosine.score(test_ood_emb)
    results['CosineDistance'] = compute_metrics(scores_id, scores_ood)
    print(f"  AUROC: {results['CosineDistance']['auroc']:.2f}%")

    # Train classifier for MSP/Energy/MaxLogit
    print("\n  Training classifier for MSP/Energy/MaxLogit...")
    classifier = train_classifier(train_emb, train_labels, n_classes)

    # 7. MSP
    print("\n[7/10] MSP...")
    msp = MSPDetector(classifier)
    scores_id = msp.score(test_id_emb)
    scores_ood = msp.score(test_ood_emb)
    results['MSP'] = compute_metrics(scores_id, scores_ood)
    print(f"  AUROC: {results['MSP']['auroc']:.2f}%")

    # 8. Energy
    print("\n[8/10] Energy...")
    energy = EnergyDetector(classifier)
    scores_id = energy.score(test_id_emb)
    scores_ood = energy.score(test_ood_emb)
    results['Energy'] = compute_metrics(scores_id, scores_ood)
    print(f"  AUROC: {results['Energy']['auroc']:.2f}%")

    # 9. MaxLogit
    print("\n[9/10] MaxLogit...")
    maxlogit = MaxLogitDetector(classifier)
    scores_id = maxlogit.score(test_id_emb)
    scores_ood = maxlogit.score(test_ood_emb)
    results['MaxLogit'] = compute_metrics(scores_id, scores_ood)
    print(f"  AUROC: {results['MaxLogit']['auroc']:.2f}%")

    # 10. HeterophilyEnhanced (Stage 2)
    print("\n[10/10] HeterophilyEnhanced (Our Method)...")
    het_ood = HeterophilyEnhancedOOD(k=10, alpha=0.3, use_heterophily=True)
    het_ood.fit(train_emb, train_labels)
    scores_id = het_ood.score(test_id_emb)
    scores_ood = het_ood.score(test_ood_emb)
    results['HeterophilyEnhanced'] = compute_metrics(scores_id, scores_ood)
    print(f"  AUROC: {results['HeterophilyEnhanced']['auroc']:.2f}%")

    return results


def run_experiments():
    """Main experiment runner."""

    print("="*80)
    print("RW3 EXPERIMENTS V2: CORRECTED VERSION")
    print("="*80)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Device: {CONFIG['device']}")

    all_results = {}

    # =========================================================================
    # CLINC150 (Primary benchmark)
    # =========================================================================
    print("\n" + "#"*80)
    print("# DATASET: CLINC150 (Primary Benchmark)")
    print("#"*80)

    try:
        data = load_clinc150()

        # Generate embeddings
        cache_dir = CONFIG['output_dir'] / 'data' / 'clinc150'
        train_emb = generate_embeddings(data['train_texts'], cache_dir / 'train.pt')
        test_id_emb = generate_embeddings(data['test_id_texts'], cache_dir / 'test_id.pt')
        test_ood_emb = generate_embeddings(data['test_ood_texts'], cache_dir / 'test_ood.pt')

        # Run baselines
        results = run_all_baselines(
            'clinc150', train_emb, data['train_labels'],
            test_id_emb, test_ood_emb, data['n_classes']
        )

        all_results['clinc150'] = results

        # Validation checkpoint
        emb_sim_auroc = results['EmbeddingSimilarity']['auroc']
        if emb_sim_auroc < CONFIG['stage1_min_auroc']:
            warnings.warn(
                f"EmbeddingSimilarity AUROC ({emb_sim_auroc:.2f}%) < {CONFIG['stage1_min_auroc']}%\n"
                f"Pre-experiment achieved 96.46%. Implementation may have issues."
            )

        # Save results
        with open(CONFIG['output_dir'] / 'results' / 'clinc150.json', 'w') as f:
            json.dump(results, f, indent=2)

    except Exception as e:
        print(f"Error on CLINC150: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # Banking77-OOS (Near-OOD Benchmark - KEY!)
    # =========================================================================
    print("\n" + "#"*80)
    print("# DATASET: Banking77-OOS (Near-OOD Benchmark)")
    print("#"*80)

    try:
        data = load_banking77_oos()

        # Generate embeddings
        cache_dir = CONFIG['output_dir'] / 'data' / 'banking77_oos'
        train_emb = generate_embeddings(data['train_texts'], cache_dir / 'train.pt')
        test_id_emb = generate_embeddings(data['test_id_texts'], cache_dir / 'test_id.pt')
        test_ood_emb = generate_embeddings(data['test_ood_texts'], cache_dir / 'test_ood.pt')

        # Run baselines
        results = run_all_baselines(
            'banking77_oos', train_emb, data['train_labels'],
            test_id_emb, test_ood_emb, data['n_classes']
        )

        all_results['banking77_oos'] = results

        # Save results
        with open(CONFIG['output_dir'] / 'results' / 'banking77_oos.json', 'w') as f:
            json.dump(results, f, indent=2)

    except FileNotFoundError as e:
        print(f"Banking77-OOS not available: {e}")
    except Exception as e:
        print(f"Error on Banking77-OOS: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # Summary Report
    # =========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    # Generate summary table
    methods = ['EmbeddingSimilarity', 'HeterophilyEnhanced', 'Mahalanobis',
               'LOF', 'KNN-10', 'MSP', 'Energy', 'MaxLogit']

    print("\n### AUROC (%) Comparison\n")
    header = "| Method |"
    for ds in all_results.keys():
        header += f" {ds} |"
    header += " Avg |"
    print(header)
    print("|" + "--------|" * (len(all_results) + 2))

    for method in methods:
        row = f"| {method} |"
        aurocs = []
        for ds, res in all_results.items():
            if method in res:
                auroc = res[method]['auroc']
                aurocs.append(auroc)
                row += f" {auroc:.2f} |"
            else:
                row += " - |"
        if aurocs:
            row += f" **{np.mean(aurocs):.2f}** |"
        else:
            row += " - |"
        print(row)

    # Find best method
    print("\n### Key Findings\n")
    for ds, res in all_results.items():
        best_method = max(res.keys(), key=lambda m: res[m]['auroc'])
        best_auroc = res[best_method]['auroc']
        our_auroc = res.get('HeterophilyEnhanced', {}).get('auroc', 0)
        improvement = our_auroc - best_auroc if best_method != 'HeterophilyEnhanced' else 0

        print(f"**{ds}**:")
        print(f"  - Best baseline: {best_method} ({best_auroc:.2f}%)")
        print(f"  - Our method: HeterophilyEnhanced ({our_auroc:.2f}%)")
        if improvement > 0:
            print(f"  - Improvement: +{improvement:.2f}%")
        else:
            print(f"  - Gap: {improvement:.2f}%")

    # Save complete results
    with open(CONFIG['output_dir'] / 'results' / 'complete_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print("EXPERIMENTS COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {CONFIG['output_dir'] / 'results'}")

    return all_results


if __name__ == '__main__':
    run_experiments()
