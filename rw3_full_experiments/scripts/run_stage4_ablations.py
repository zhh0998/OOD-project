#!/usr/bin/env python3
"""
Stage 4: Ablation Experiments (A0-A13)
Comprehensive ablation studies for the paper.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt

from src.audit.leakage_audit import LeakageAuditor, DeliberateLeakageChecker
from src.methods.whitening import SpectralWhitening, compute_cohens_d
from src.methods.knn_scorer import KNNScorer
from src.methods.graph_features import GraphFeatureExtractor
from src.methods.fusion import UnsupervisedFusion, FusionAblation
from src.methods.conformal import ConformalPredictor, RiskCoverageAnalyzer
from src.evaluation.metrics import compute_ood_metrics, compute_all_metrics_with_ci
from src.evaluation.risk_coverage import RiskCoverageEvaluator
from src.embeddings.anisotropy_metrics import compute_anisotropy_metrics


class AblationRunner:
    """Runner for all ablation experiments."""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.ablation_dir = self.output_dir / "ablations"
        self.ablation_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.auditor = LeakageAuditor(output_dir=str(self.output_dir / "audit"))

    def load_embeddings(self, dataset: str, model: str) -> Dict[str, np.ndarray]:
        """Load cached embeddings."""
        cache_dir = Path("cache/embeddings") / dataset / model
        embeddings = {}
        for split in ['train_id', 'cal_id', 'test_id', 'test_ood']:
            path = cache_dir / f"{split}.npy"
            if path.exists():
                embeddings[split] = np.load(path)
        return embeddings

    def load_labels(self, dataset: str) -> Dict[str, np.ndarray]:
        """Load labels from data."""
        data_path = Path("data/processed") / dataset / "data.json"
        labels = {}
        with open(data_path) as f:
            data = json.load(f)
        for split in ['train_id', 'cal_id', 'test_id', 'test_ood']:
            if split in data:
                labels[split] = np.array(data[split]['labels'])
        return labels

    def run_a0_deliberate_leakage(self, dataset: str = 'clinc150',
                                   model: str = 'minilm') -> Dict:
        """
        A0: Deliberate Leakage Sanity Check
        Proves audit credibility by showing leakage inflates scores.
        """
        print("\n" + "=" * 60)
        print("A0: DELIBERATE LEAKAGE SANITY CHECK")
        print("=" * 60)

        embeddings = self.load_embeddings(dataset, model)
        labels = self.load_labels(dataset)

        # Test labels
        n_test_id = len(embeddings['test_id'])
        n_test_ood = len(embeddings['test_ood'])
        test_labels = np.concatenate([np.zeros(n_test_id), np.ones(n_test_ood)])

        def compute_metrics(whitened_dict, labels_dict):
            """Compute AUROC and Cohen's d after whitening."""
            test_emb = np.vstack([whitened_dict['test_id'], whitened_dict['test_ood']])

            # Simple kNN scoring
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=20, metric='cosine')
            nn.fit(whitened_dict['train_id'])
            dists, _ = nn.kneighbors(test_emb)
            scores = dists.mean(axis=1)

            from sklearn.metrics import roc_auc_score
            auroc = roc_auc_score(test_labels, scores)

            id_scores = scores[:n_test_id]
            ood_scores = scores[n_test_id:]
            cohens_d = compute_cohens_d(id_scores, ood_scores)

            return {'auroc': auroc, 'cohens_d': cohens_d}

        checker = DeliberateLeakageChecker(self.auditor)
        results = checker.compare_settings(embeddings, labels, compute_metrics)

        # Save results
        with open(self.ablation_dir / "a0_leakage_check.json", 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def run_a1_whitening_k_sweep(self, datasets: List[str],
                                  models: List[str]) -> Dict:
        """
        A1: Whitening k value sweep
        k ∈ {0, 1, 2, 3, 5, 10, 20, 50}
        """
        print("\n" + "=" * 60)
        print("A1: WHITENING K VALUE SWEEP")
        print("=" * 60)

        k_values = [0, 1, 2, 3, 5, 10, 20, 50]
        results = []

        for dataset in datasets:
            for model in models:
                embeddings = self.load_embeddings(dataset, model)
                labels = self.load_labels(dataset)

                if len(embeddings) < 4:
                    continue

                # Prepare test labels
                n_test_id = len(embeddings['test_id'])
                n_test_ood = len(embeddings['test_ood'])
                test_labels = np.concatenate([np.zeros(n_test_id), np.ones(n_test_ood)])
                test_emb = np.vstack([embeddings['test_id'], embeddings['test_ood']])

                for k in k_values:
                    print(f"  {dataset}/{model}/k={k}")

                    # Apply whitening
                    whitener = SpectralWhitening(k_remove=k)
                    whitener.fit(embeddings['train_id'])

                    whitened_train = whitener.transform(embeddings['train_id'])
                    whitened_test = whitener.transform(test_emb)

                    # kNN scoring
                    knn = KNNScorer(k=20)
                    knn.fit(whitened_train, labels['train_id'])
                    scores, _, _ = knn.score(whitened_test)

                    # Metrics
                    metrics = compute_ood_metrics(scores, test_labels)

                    results.append({
                        'dataset': dataset,
                        'model': model,
                        'k_whitening': k,
                        'auroc': metrics['auroc'],
                        'cohens_d': metrics['cohens_d'],
                        'fpr95': metrics['fpr95']
                    })

        df = pd.DataFrame(results)
        df.to_csv(self.ablation_dir / "a1_whitening_k_sweep.csv", index=False)

        # Plot
        self._plot_a1_k_sweep(df)

        return results

    def _plot_a1_k_sweep(self, df: pd.DataFrame):
        """Plot k-sweep results."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for metric, ax in zip(['auroc', 'cohens_d', 'fpr95'], axes):
            for (dataset, model), group in df.groupby(['dataset', 'model']):
                ax.plot(group['k_whitening'], group[metric],
                        marker='o', label=f"{dataset}/{model}")
            ax.set_xlabel('k (components removed)')
            ax.set_ylabel(metric.upper())
            ax.set_title(f'{metric.upper()} vs Whitening k')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.figures_dir / "fig4_whitening_k_sweep.png", dpi=150)
        plt.close()

    def run_a3_whitening_vs_graph(self, datasets: List[str],
                                   models: List[str]) -> Dict:
        """
        A3: Whitening vs Graph Features Independent Contribution
        Most critical ablation for understanding method components.
        """
        print("\n" + "=" * 60)
        print("A3: WHITENING vs GRAPH FEATURES")
        print("=" * 60)

        results = []

        for dataset in datasets:
            for model in models:
                embeddings = self.load_embeddings(dataset, model)
                labels = self.load_labels(dataset)

                if len(embeddings) < 4:
                    continue

                print(f"  {dataset}/{model}")

                # Prepare data
                n_test_id = len(embeddings['test_id'])
                n_test_ood = len(embeddings['test_ood'])
                test_labels = np.concatenate([np.zeros(n_test_id), np.ones(n_test_ood)])

                # (a) kNN only
                scores_a = self._run_knn_only(embeddings, labels)
                metrics_a = compute_ood_metrics(scores_a, test_labels)

                # (b) kNN + whitening
                scores_b = self._run_knn_whitening(embeddings, labels)
                metrics_b = compute_ood_metrics(scores_b, test_labels)

                # (c) kNN + graph features (no whitening)
                scores_c = self._run_knn_graph(embeddings, labels)
                metrics_c = compute_ood_metrics(scores_c, test_labels)

                # (d) kNN + whitening + graph features (Ours)
                scores_d = self._run_full(embeddings, labels)
                metrics_d = compute_ood_metrics(scores_d, test_labels)

                for setting, metrics in [('a_knn_only', metrics_a),
                                          ('b_knn_whitening', metrics_b),
                                          ('c_knn_graph', metrics_c),
                                          ('d_ours', metrics_d)]:
                    results.append({
                        'dataset': dataset,
                        'model': model,
                        'setting': setting,
                        'auroc': metrics['auroc'],
                        'cohens_d': metrics['cohens_d'],
                        'fpr95': metrics['fpr95']
                    })

        df = pd.DataFrame(results)
        df.to_csv(self.ablation_dir / "a3_whitening_vs_graph.csv", index=False)

        # Analysis
        pivot = df.pivot_table(index=['dataset', 'model'],
                               columns='setting', values='auroc')
        print("\nAUROC by Setting:")
        print(pivot.to_string())

        return results

    def _run_knn_only(self, embeddings: Dict, labels: Dict) -> np.ndarray:
        """(a) kNN only - no whitening, no graph features."""
        knn = KNNScorer(k=20)
        knn.fit(embeddings['train_id'], labels['train_id'])
        test_emb = np.vstack([embeddings['test_id'], embeddings['test_ood']])
        scores, _, _ = knn.score(test_emb)
        return scores

    def _run_knn_whitening(self, embeddings: Dict, labels: Dict) -> np.ndarray:
        """(b) kNN + whitening."""
        whitener = SpectralWhitening(k_remove=3)
        whitener.fit(embeddings['train_id'])

        whitened = {k: whitener.transform(v) for k, v in embeddings.items()}

        knn = KNNScorer(k=20)
        knn.fit(whitened['train_id'], labels['train_id'])
        test_emb = np.vstack([whitened['test_id'], whitened['test_ood']])
        scores, _, _ = knn.score(test_emb)
        return scores

    def _run_knn_graph(self, embeddings: Dict, labels: Dict) -> np.ndarray:
        """(c) kNN + graph features (no whitening)."""
        knn = KNNScorer(k=20)
        knn.fit(embeddings['train_id'], labels['train_id'])

        test_emb = np.vstack([embeddings['test_id'], embeddings['test_ood']])
        scores, distances, indices = knn.score(test_emb)

        # Graph features
        graph = GraphFeatureExtractor(topK=20)
        neighbor_labels = knn.get_neighbor_labels(indices)
        features = graph.extract_all(distances, indices, neighbor_labels, test_emb)

        # Fusion
        cal_scores, cal_dists, cal_idx = knn.score(embeddings['cal_id'])
        cal_neighbor_labels = knn.get_neighbor_labels(cal_idx)
        cal_features = graph.extract_all(cal_dists, cal_idx, cal_neighbor_labels,
                                         embeddings['cal_id'])

        fusion = UnsupervisedFusion(method="zscore_mean")
        fusion.fit(cal_scores, cal_features)
        return fusion.transform(scores, features)

    def _run_full(self, embeddings: Dict, labels: Dict) -> np.ndarray:
        """(d) Full method: kNN + whitening + graph features."""
        whitener = SpectralWhitening(k_remove=3)
        whitener.fit(embeddings['train_id'])
        whitened = {k: whitener.transform(v) for k, v in embeddings.items()}

        knn = KNNScorer(k=20)
        knn.fit(whitened['train_id'], labels['train_id'])

        test_emb = np.vstack([whitened['test_id'], whitened['test_ood']])
        scores, distances, indices = knn.score(test_emb)

        graph = GraphFeatureExtractor(topK=20)
        neighbor_labels = knn.get_neighbor_labels(indices)
        features = graph.extract_all(distances, indices, neighbor_labels, test_emb)

        cal_scores, cal_dists, cal_idx = knn.score(whitened['cal_id'])
        cal_neighbor_labels = knn.get_neighbor_labels(cal_idx)
        cal_features = graph.extract_all(cal_dists, cal_idx, cal_neighbor_labels,
                                         whitened['cal_id'])

        fusion = UnsupervisedFusion(method="zscore_mean")
        fusion.fit(cal_scores, cal_features)
        return fusion.transform(scores, features)

    def run_a9_banking77_robustness(self) -> Dict:
        """
        A9: Banking77 Multi-Random Partition Robustness
        5 random seeds + alphabetical, report mean ± 95% CI.
        """
        print("\n" + "=" * 60)
        print("A9: BANKING77 ROBUSTNESS")
        print("=" * 60)

        seeds = [42, 123, 456, 789, 2024]
        results = []

        # Alphabetical partition
        for partition in ['alpha'] + [f'seed{s}' for s in seeds]:
            dataset = f'banking77_{partition}'
            for model in ['minilm', 'bge', 'e5', 'mpnet']:
                embeddings = self.load_embeddings(dataset, model)
                labels = self.load_labels(dataset)

                if len(embeddings) < 4:
                    continue

                print(f"  {dataset}/{model}")

                n_test_id = len(embeddings['test_id'])
                n_test_ood = len(embeddings['test_ood'])
                test_labels = np.concatenate([np.zeros(n_test_id), np.ones(n_test_ood)])

                scores = self._run_full(embeddings, labels)
                metrics = compute_ood_metrics(scores, test_labels)

                results.append({
                    'partition': partition,
                    'model': model,
                    'auroc': metrics['auroc'],
                    'cohens_d': metrics['cohens_d'],
                    'fpr95': metrics['fpr95']
                })

        df = pd.DataFrame(results)
        df.to_csv(self.ablation_dir / "a9_banking77_robustness.csv", index=False)

        # Compute statistics
        random_results = df[df['partition'].str.startswith('seed')]
        alpha_results = df[df['partition'] == 'alpha']

        print("\nRandom Partitions (mean ± std):")
        print(random_results.groupby('model').agg({
            'auroc': ['mean', 'std'],
            'cohens_d': ['mean', 'std']
        }))

        print("\nAlphabetical Partition:")
        print(alpha_results[['model', 'auroc', 'cohens_d']])

        return results

    def run_a12_istar_response(self, datasets: List[str],
                                models: List[str]) -> Dict:
        """
        A12: I-STAR Response Experiment
        Show whitening may hurt classification but help OOD detection.
        """
        print("\n" + "=" * 60)
        print("A12: I-STAR RESPONSE")
        print("=" * 60)

        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score

        results = []

        for dataset in datasets:
            for model in models:
                embeddings = self.load_embeddings(dataset, model)
                labels = self.load_labels(dataset)

                if len(embeddings) < 4:
                    continue

                print(f"  {dataset}/{model}")

                # Anisotropy metrics
                aniso = compute_anisotropy_metrics(embeddings['train_id'])

                # (a) Original -> classification accuracy
                clf_orig = KNeighborsClassifier(n_neighbors=5)
                clf_orig.fit(embeddings['train_id'], labels['train_id'])
                acc_orig = accuracy_score(labels['test_id'],
                                          clf_orig.predict(embeddings['test_id']))

                # (b) Whitened -> classification accuracy
                whitener = SpectralWhitening(k_remove=3)
                whitener.fit(embeddings['train_id'])
                whitened_train = whitener.transform(embeddings['train_id'])
                whitened_test = whitener.transform(embeddings['test_id'])

                clf_white = KNeighborsClassifier(n_neighbors=5)
                clf_white.fit(whitened_train, labels['train_id'])
                acc_white = accuracy_score(labels['test_id'],
                                           clf_white.predict(whitened_test))

                # (c) Original -> OOD AUROC
                test_all = np.vstack([embeddings['test_id'], embeddings['test_ood']])
                test_labels = np.concatenate([
                    np.zeros(len(embeddings['test_id'])),
                    np.ones(len(embeddings['test_ood']))
                ])

                knn = KNNScorer(k=20)
                knn.fit(embeddings['train_id'], labels['train_id'])
                scores_orig, _, _ = knn.score(test_all)
                auroc_orig = compute_ood_metrics(scores_orig, test_labels)['auroc']

                # (d) Whitened -> OOD AUROC
                whitened_all = whitener.transform(test_all)
                knn_white = KNNScorer(k=20)
                knn_white.fit(whitened_train, labels['train_id'])
                scores_white, _, _ = knn_white.score(whitened_all)
                auroc_white = compute_ood_metrics(scores_white, test_labels)['auroc']

                results.append({
                    'dataset': dataset,
                    'model': model,
                    'effective_dim': aniso['effective_dim'],
                    'acc_original': acc_orig,
                    'acc_whitened': acc_white,
                    'acc_delta': acc_white - acc_orig,
                    'auroc_original': auroc_orig,
                    'auroc_whitened': auroc_white,
                    'auroc_delta': auroc_white - auroc_orig
                })

        df = pd.DataFrame(results)
        df.to_csv(self.ablation_dir / "a12_istar_response.csv", index=False)

        # Plot I-STAR scatter
        self._plot_istar_scatter(df)

        return results

    def _plot_istar_scatter(self, df: pd.DataFrame):
        """Plot I-STAR response scatter."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Effective dim vs Classification delta
        ax1 = axes[0]
        ax1.scatter(df['effective_dim'], df['acc_delta'] * 100, s=100)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Effective Dimension')
        ax1.set_ylabel('Classification Accuracy Δ (%)')
        ax1.set_title('Whitening Effect on Classification\n(I-STAR: may hurt)')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Effective dim vs OOD AUROC delta
        ax2 = axes[1]
        ax2.scatter(df['effective_dim'], df['auroc_delta'] * 100, s=100)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Effective Dimension')
        ax2.set_ylabel('OOD AUROC Δ (%)')
        ax2.set_title('Whitening Effect on OOD Detection\n(Our goal: should help)')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.figures_dir / "fig2_istar_response.png", dpi=150)
        plt.close()

    def run_all_ablations(self, datasets: List[str] = None,
                          models: List[str] = None):
        """Run all ablation experiments."""
        if datasets is None:
            datasets = ['clinc150', 'banking77_alpha']
        if models is None:
            models = ['minilm', 'bge']

        print("=" * 60)
        print("STAGE 4: ABLATION EXPERIMENTS")
        print("=" * 60)

        # A0: Deliberate leakage
        self.run_a0_deliberate_leakage()

        # A1: Whitening k sweep
        self.run_a1_whitening_k_sweep(datasets, models)

        # A3: Whitening vs Graph
        self.run_a3_whitening_vs_graph(datasets, models)

        # A9: Banking77 robustness
        self.run_a9_banking77_robustness()

        # A12: I-STAR response
        self.run_a12_istar_response(datasets, models)

        print("\n" + "=" * 60)
        print("STAGE 4 CHECKPOINT")
        print("=" * 60)
        print("Ablations completed: A0, A1, A3, A9, A12")
        print(f"Results saved to: {self.ablation_dir}")
        print(f"Figures saved to: {self.figures_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage 4: Ablation Experiments")
    parser.add_argument('--datasets', type=str, default='clinc150,banking77_alpha')
    parser.add_argument('--models', type=str, default='minilm,bge')

    args = parser.parse_args()

    datasets = args.datasets.split(',')
    models = args.models.split(',')

    runner = AblationRunner()
    runner.run_all_ablations(datasets, models)


if __name__ == "__main__":
    main()
