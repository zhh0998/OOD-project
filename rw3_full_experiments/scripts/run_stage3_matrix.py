#!/usr/bin/env python3
"""
Stage 3: Main Experiment Matrix
Runs D×M×A experiments with checkpoint/resume support.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.audit.leakage_audit import LeakageAuditor
from src.methods.whitening import SpectralWhitening, compute_cohens_d
from src.methods.knn_scorer import KNNScorer
from src.methods.graph_features import GraphFeatureExtractor
from src.methods.fusion import UnsupervisedFusion
from src.methods.conformal import ConformalPredictor, RiskCoverageAnalyzer
from src.baselines.embedding_based import (
    KNNDistanceBaseline, LOFBaseline, IsolationForestBaseline,
    MahalanobisBaseline, CentroidDistanceBaseline
)
from src.baselines.sota_methods import MahalanobisPlusPlusBaseline, FLaTSBaseline
from src.evaluation.metrics import compute_ood_metrics, compute_all_metrics_with_ci
from src.evaluation.risk_coverage import RiskCoverageEvaluator


class ExperimentRunner:
    """Main experiment runner with checkpoint support."""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "manifest.json"
        self.manifest = self._load_manifest()
        self.auditor = LeakageAuditor(output_dir=str(self.output_dir / "audit"))

    def _load_manifest(self) -> Dict:
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                return json.load(f)
        return {"version": "4.1", "completed": [], "failed": [], "remaining": []}

    def _save_manifest(self):
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)

    def _is_completed(self, dataset: str, model: str, method: str, seed: int) -> bool:
        key = f"{dataset}_{model}_{method}_{seed}"
        return any(c.get('key') == key for c in self.manifest['completed'])

    def _mark_completed(self, dataset: str, model: str, method: str, seed: int, results: Dict):
        key = f"{dataset}_{model}_{method}_{seed}"
        self.manifest['completed'].append({
            'key': key,
            'dataset': dataset,
            'model': model,
            'method': method,
            'seed': seed,
            'timestamp': datetime.now().isoformat(),
            'auroc': results.get('auroc', {}).get('value', 0)
        })
        self._save_manifest()

    def run_ours(self, embeddings: Dict[str, np.ndarray],
                 labels: Dict[str, np.ndarray],
                 k_whitening: int = 3,
                 k_knn: int = 20) -> Tuple[np.ndarray, Dict]:
        """Run our full pipeline."""
        self.auditor.reset()

        # 1. Whitening (fit on ID-train only)
        whitener = SpectralWhitening(k_remove=k_whitening)
        whitener.fit(embeddings['train_id'], audit_log=self.auditor)

        # Apply whitening to all splits
        whitened = {
            split: whitener.transform(emb)
            for split, emb in embeddings.items()
        }

        # 2. KNN scorer (fit on ID-train)
        knn = KNNScorer(k=k_knn)
        knn.fit(whitened['train_id'], labels['train_id'], audit_log=self.auditor)

        # Get scores and neighbors for all test data
        test_all = np.vstack([whitened['test_id'], whitened['test_ood']])
        scores_all, distances, indices = knn.score(test_all)

        # Scores for calibration
        cal_scores, cal_distances, cal_indices = knn.score(whitened['cal_id'])

        # 3. Graph features
        graph_extractor = GraphFeatureExtractor(topK=k_knn)

        # Get neighbor labels for cal and test
        cal_neighbor_labels = knn.get_neighbor_labels(cal_indices)
        test_neighbor_labels = knn.get_neighbor_labels(indices)

        # Extract features
        cal_features = graph_extractor.extract_all(
            cal_distances, cal_indices, cal_neighbor_labels,
            embeddings=whitened['cal_id']
        )
        test_features = graph_extractor.extract_all(
            distances, indices, test_neighbor_labels,
            embeddings=test_all
        )

        # 4. Fusion (fit on ID-cal)
        fusion = UnsupervisedFusion(method="zscore_mean")
        fusion.fit(cal_scores, cal_features, audit_log=self.auditor)

        # Final scores
        final_scores = fusion.transform(scores_all, test_features)

        # Also get knn-only scores for comparison
        knn_only_scores = scores_all

        return final_scores, {
            'knn_scores': knn_only_scores,
            'cal_scores': cal_scores,
            'whitener': whitener,
            'knn': knn,
            'fusion': fusion
        }

    def run_baseline(self, method_name: str,
                     embeddings: Dict[str, np.ndarray],
                     labels: Dict[str, np.ndarray]) -> np.ndarray:
        """Run a baseline method."""
        train_emb = embeddings['train_id']
        test_emb = np.vstack([embeddings['test_id'], embeddings['test_ood']])
        train_labels = labels['train_id']

        if method_name == 'B1_knn':
            model = KNNDistanceBaseline(k=20)
            model.fit(train_emb, self.auditor)
            return model.score(test_emb)

        elif method_name == 'B2_lof':
            model = LOFBaseline(k=20)
            model.fit(train_emb, self.auditor)
            return model.score(test_emb)

        elif method_name == 'B3_iforest':
            model = IsolationForestBaseline()
            model.fit(train_emb, self.auditor)
            return model.score(test_emb)

        elif method_name == 'B4_mahal':
            model = MahalanobisBaseline()
            model.fit(train_emb, self.auditor)
            return model.score(test_emb)

        elif method_name == 'B5_centroid':
            model = CentroidDistanceBaseline()
            model.fit(train_emb, train_labels, self.auditor)
            return model.score(test_emb)

        elif method_name == 'B8_mahal_pp':
            model = MahalanobisPlusPlusBaseline()
            model.fit(train_emb, self.auditor)
            return model.score(test_emb)

        elif method_name == 'B9_flats':
            model = FLaTSBaseline()
            model.fit(train_emb, self.auditor)
            return model.score(test_emb)

        else:
            raise ValueError(f"Unknown baseline: {method_name}")

    def run_single_experiment(self,
                              dataset: str,
                              model: str,
                              method: str,
                              seed: int,
                              embeddings: Dict[str, np.ndarray],
                              labels: Dict[str, np.ndarray],
                              ood_groups: Optional[np.ndarray] = None) -> Dict:
        """Run a single experiment configuration."""
        np.random.seed(seed)

        # Prepare test labels
        n_test_id = len(embeddings['test_id'])
        n_test_ood = len(embeddings['test_ood'])
        test_labels = np.concatenate([
            np.zeros(n_test_id),
            np.ones(n_test_ood)
        ])

        # Run method
        if method == 'ours':
            scores, extras = self.run_ours(embeddings, labels)
            cal_scores = extras['cal_scores']
        else:
            scores = self.run_baseline(method, embeddings, labels)
            # Get cal scores for CP
            if method == 'B1_knn':
                model = KNNDistanceBaseline(k=20)
                model.fit(embeddings['train_id'])
                cal_scores = model.score(embeddings['cal_id'])
            else:
                cal_scores = None

        # Compute metrics
        metrics = compute_all_metrics_with_ci(
            scores, test_labels, ood_groups,
            n_bootstrap=1000, confidence=0.95
        )

        # Risk-coverage analysis
        if cal_scores is not None:
            rc_eval = RiskCoverageEvaluator()
            rc_metrics = rc_eval.compute_all_metrics(
                scores, test_labels, cal_scores
            )
            metrics['risk_coverage'] = rc_metrics

        # Save audit log
        self.auditor.save_log(dataset, model, method)

        return metrics

    def run_matrix(self,
                   datasets: List[str],
                   models: List[str],
                   methods: List[str],
                   seeds: List[int],
                   resume: bool = True):
        """Run full experiment matrix."""
        results = []

        total = len(datasets) * len(models) * len(methods) * len(seeds)
        completed = 0
        skipped = 0

        print(f"\nRunning experiment matrix: {total} combinations")
        print(f"Datasets: {datasets}")
        print(f"Models: {models}")
        print(f"Methods: {methods}")
        print(f"Seeds: {seeds}")

        for dataset in datasets:
            # Load embeddings
            for model in models:
                # Load cached embeddings
                cache_dir = Path("cache/embeddings") / dataset / model

                embeddings = {}
                labels = {}
                ood_groups = None

                for split in ['train_id', 'cal_id', 'test_id', 'test_ood']:
                    cache_path = cache_dir / f"{split}.npy"
                    if cache_path.exists():
                        embeddings[split] = np.load(cache_path)

                # Load labels from data
                data_path = Path("data/processed") / dataset / "data.json"
                if data_path.exists():
                    with open(data_path) as f:
                        data = json.load(f)
                    for split in ['train_id', 'cal_id', 'test_id', 'test_ood']:
                        if split in data:
                            labels[split] = np.array(data[split]['labels'])
                    if 'ood_group' in data.get('test_ood', {}):
                        ood_groups = np.array(data['test_ood']['ood_group'])

                if len(embeddings) < 4:
                    print(f"Skipping {dataset}/{model}: embeddings not found")
                    continue

                for method in methods:
                    for seed in seeds:
                        completed += 1

                        # Check if already done
                        if resume and self._is_completed(dataset, model, method, seed):
                            skipped += 1
                            continue

                        print(f"\n[{completed}/{total}] {dataset}/{model}/{method}/seed{seed}")

                        try:
                            metrics = self.run_single_experiment(
                                dataset, model, method, seed,
                                embeddings, labels, ood_groups
                            )

                            result = {
                                'dataset': dataset,
                                'model': model,
                                'method': method,
                                'seed': seed,
                                **{k: v.get('value', v) if isinstance(v, dict) else v
                                   for k, v in metrics.items() if k != 'risk_coverage'}
                            }

                            if 'risk_coverage' in metrics:
                                result['auc_rc'] = metrics['risk_coverage'].get('auc_rc', 0)
                                result['best_utility'] = metrics['risk_coverage'].get('best_utility', 0)

                            results.append(result)
                            self._mark_completed(dataset, model, method, seed, metrics)

                            print(f"  AUROC: {metrics['auroc']['value']:.4f}")
                            if 'risk_coverage' in metrics:
                                print(f"  AUC-RC: {metrics['risk_coverage']['auc_rc']:.4f}")

                        except Exception as e:
                            print(f"  ERROR: {e}")
                            self.manifest['failed'].append({
                                'dataset': dataset,
                                'model': model,
                                'method': method,
                                'seed': seed,
                                'error': str(e)
                            })
                            self._save_manifest()

        # Save results
        if results:
            df = pd.DataFrame(results)
            df.to_csv(self.output_dir / "raw" / "stage3_results.csv", index=False)

            # Generate summary tables
            self._generate_summary_tables(df)

        print(f"\n{'=' * 60}")
        print("STAGE 3 CHECKPOINT")
        print("=" * 60)
        print(f"Completed: {completed - skipped}")
        print(f"Skipped (already done): {skipped}")
        print(f"Failed: {len(self.manifest['failed'])}")

    def _generate_summary_tables(self, df: pd.DataFrame):
        """Generate summary tables from results."""
        tables_dir = self.output_dir / "tables"
        tables_dir.mkdir(exist_ok=True)

        # Table 1: Overall AUROC
        if 'auroc' in df.columns:
            pivot = df.pivot_table(
                index='dataset',
                columns='method',
                values='auroc',
                aggfunc='mean'
            )
            pivot.to_csv(tables_dir / "table1_overall_auroc.csv")

        # Table 3: AUC-RC (Risk-Coverage)
        if 'auc_rc' in df.columns:
            pivot = df.pivot_table(
                index='dataset',
                columns='method',
                values='auc_rc',
                aggfunc='mean'
            )
            pivot.to_csv(tables_dir / "table3_auc_rc.csv")

        print(f"Saved summary tables to {tables_dir}")


def main():
    parser = argparse.ArgumentParser(description="Stage 3: Main Experiment Matrix")
    parser.add_argument('--datasets', type=str, default='clinc150,banking77_alpha',
                        help='Comma-separated dataset names')
    parser.add_argument('--models', type=str, default='minilm,bge',
                        help='Comma-separated model names')
    parser.add_argument('--methods', type=str, default='ours,B1_knn,B2_lof,B3_iforest,B4_mahal',
                        help='Comma-separated method names')
    parser.add_argument('--seeds', type=str, default='0,1,2',
                        help='Comma-separated seeds')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Resume from checkpoint')
    parser.add_argument('--no-resume', action='store_false', dest='resume',
                        help='Start fresh')

    args = parser.parse_args()

    datasets = args.datasets.split(',')
    models = args.models.split(',')
    methods = args.methods.split(',')
    seeds = [int(s) for s in args.seeds.split(',')]

    runner = ExperimentRunner()
    runner.run_matrix(datasets, models, methods, seeds, resume=args.resume)


if __name__ == "__main__":
    main()
