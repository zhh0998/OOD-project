"""
Leakage Audit Module
Ensures no data leakage in PCA, whitening, scaling, and fusion operations.
All fit operations are logged with source split, sample count, and timestamp.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np


class LeakageAuditor:
    """
    Auditor for tracking all fit operations and ensuring no data leakage.

    Iron Rules (铁律):
    1. PCA/whitening: fit only on ID-train (SVD)
    2. Centering mean: only use ID-train mean, apply to all splits
    3. StandardScaler: fit only on ID-train or ID-cal
    4. No test samples in any training set
    5. Graph features: no OOD labels for training (unsupervised fusion only)
    6. All fit operations logged with sample count, source split, seed
    """

    def __init__(self, output_dir: str = "results/audit"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_log: List[str] = []
        self.fit_records: List[Dict[str, Any]] = []

    def log_fit(self,
                operation: str,
                fit_split: str,
                n_samples: int,
                additional_info: Optional[Dict[str, Any]] = None,
                seed: Optional[int] = None):
        """Log a fit operation with all relevant details."""
        timestamp = datetime.now().isoformat()

        # Validate fit_split against allowed values
        allowed_splits = {
            "whitening": ["id_train"],
            "pca": ["id_train"],
            "centering": ["id_train"],
            "scaler": ["id_train", "id_cal"],
            "fusion": ["id_cal"],
            "knn_index": ["id_train"],
        }

        op_type = operation.lower().split("_")[0]
        if op_type in allowed_splits:
            if fit_split not in allowed_splits[op_type]:
                raise ValueError(
                    f"LEAKAGE VIOLATION: {operation} fitted on {fit_split}, "
                    f"allowed: {allowed_splits[op_type]}"
                )

        record = {
            "timestamp": timestamp,
            "operation": operation,
            "fit_split": fit_split,
            "n_samples": n_samples,
            "seed": seed,
            "additional_info": additional_info or {}
        }
        self.fit_records.append(record)

        # Format log line
        log_line = f"[AUDIT] {operation}: fit_split={fit_split}, N_fit={n_samples}"
        if seed is not None:
            log_line += f", seed={seed}"
        if additional_info:
            for k, v in additional_info.items():
                log_line += f", {k}={v}"
        log_line += f", timestamp={timestamp}"

        self.current_log.append(log_line)
        print(log_line)

    def log_bootstrap(self, n_iter: int, stratified: bool = True):
        """Log bootstrap configuration."""
        log_line = f"[AUDIT] bootstrap: n_iter={n_iter}, stratified={stratified}"
        self.current_log.append(log_line)
        print(log_line)

    def log_seed(self, seed: int):
        """Log random seed."""
        timestamp = datetime.now().isoformat()
        log_line = f"[AUDIT] seed={seed}, timestamp={timestamp}"
        self.current_log.append(log_line)
        print(log_line)

    def save_log(self, dataset: str, model: str, method: str):
        """Save audit log to file."""
        filename = f"{dataset}_{model}_{method}.log"
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            f.write(f"# Audit Log: {dataset} / {model} / {method}\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write("# " + "=" * 60 + "\n\n")
            for line in self.current_log:
                f.write(line + "\n")

        # Also save structured JSON
        json_filepath = filepath.with_suffix('.json')
        with open(json_filepath, 'w') as f:
            json.dump({
                "dataset": dataset,
                "model": model,
                "method": method,
                "fit_records": self.fit_records,
                "log_lines": self.current_log
            }, f, indent=2)

    def reset(self):
        """Reset log for new experiment."""
        self.current_log = []
        self.fit_records = []

    def validate_no_ood_labels_in_fusion(self, fusion_method: str):
        """Validate that fusion doesn't use OOD labels."""
        allowed_methods = ["zscore_mean", "rank_average", "fisher_pvalue", "weighted_sum"]
        if fusion_method not in allowed_methods:
            raise ValueError(
                f"LEAKAGE VIOLATION: fusion method '{fusion_method}' not in allowed list. "
                f"Allowed (unsupervised): {allowed_methods}"
            )
        log_line = f"[AUDIT] fusion: method={fusion_method}, ood_labels_used=NONE"
        self.current_log.append(log_line)
        print(log_line)


class DeliberateLeakageChecker:
    """
    A0: Deliberate Leakage Sanity Check
    Run experiments with intentional leakage to show how much scores inflate.
    This goes in the Appendix to prove audit credibility.
    """

    def __init__(self, auditor: LeakageAuditor):
        self.auditor = auditor
        self.results: Dict[str, Dict[str, float]] = {}

    def run_correct_setting(self, embeddings_dict: dict, labels_dict: dict):
        """
        (a) Correct: train-only mean + train-only PCA
        """
        from ..methods.whitening import SpectralWhitening

        # This is the correct way - fit only on ID-train
        whitener = SpectralWhitening(k_remove=3)
        whitener.fit(embeddings_dict['train_id'], audit_log=self.auditor)

        # Apply to all splits using the same transform
        whitened = {
            split: whitener.transform(emb)
            for split, emb in embeddings_dict.items()
        }
        return whitened

    def run_leaked_each_split_mean(self, embeddings_dict: dict, labels_dict: dict):
        """
        (b) Wrong 1: each-split mean (centering with per-split mean)
        """
        whitened = {}
        for split, emb in embeddings_dict.items():
            # LEAK: using split's own mean instead of train mean
            mean = emb.mean(axis=0)
            centered = emb - mean
            whitened[split] = centered
        return whitened

    def run_leaked_test_fit_scaler(self, embeddings_dict: dict, labels_dict: dict):
        """
        (c) Wrong 2: test-fit scaler (fit on test+OOD)
        """
        from sklearn.preprocessing import StandardScaler

        # Combine test_id and test_ood for fitting (LEAK!)
        test_all = np.vstack([
            embeddings_dict['test_id'],
            embeddings_dict['test_ood']
        ])

        scaler = StandardScaler()
        scaler.fit(test_all)  # LEAK: fitting on test data

        whitened = {
            split: scaler.transform(emb)
            for split, emb in embeddings_dict.items()
        }
        return whitened

    def compare_settings(self,
                         embeddings_dict: dict,
                         labels_dict: dict,
                         compute_metrics_fn) -> Dict[str, Dict[str, float]]:
        """
        Compare all settings and return metric differences.

        Args:
            embeddings_dict: dict with keys train_id, cal_id, test_id, test_ood
            labels_dict: corresponding labels
            compute_metrics_fn: function(whitened_dict, labels_dict) -> metrics_dict

        Returns:
            dict with keys 'correct', 'leaked_each_split', 'leaked_test_fit'
        """
        print("\n" + "="*60)
        print("A0: DELIBERATE LEAKAGE SANITY CHECK")
        print("="*60)

        # (a) Correct setting
        print("\n(a) Running CORRECT setting (train-only mean + train-only PCA)...")
        whitened_correct = self.run_correct_setting(embeddings_dict, labels_dict)
        metrics_correct = compute_metrics_fn(whitened_correct, labels_dict)

        # (b) Leaked: each-split mean
        print("\n(b) Running LEAKED setting (each-split mean)...")
        whitened_leaked1 = self.run_leaked_each_split_mean(embeddings_dict, labels_dict)
        metrics_leaked1 = compute_metrics_fn(whitened_leaked1, labels_dict)

        # (c) Leaked: test-fit scaler
        print("\n(c) Running LEAKED setting (test-fit scaler)...")
        whitened_leaked2 = self.run_leaked_test_fit_scaler(embeddings_dict, labels_dict)
        metrics_leaked2 = compute_metrics_fn(whitened_leaked2, labels_dict)

        self.results = {
            'correct': metrics_correct,
            'leaked_each_split_mean': metrics_leaked1,
            'leaked_test_fit_scaler': metrics_leaked2
        }

        # Print comparison
        print("\n" + "-"*60)
        print("SANITY CHECK RESULTS:")
        print("-"*60)
        for setting, metrics in self.results.items():
            print(f"\n{setting}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

        # Calculate inflation
        print("\n" + "-"*60)
        print("INFLATION FROM LEAKAGE:")
        print("-"*60)
        for metric in metrics_correct.keys():
            correct_val = metrics_correct[metric]
            for leaked_name in ['leaked_each_split_mean', 'leaked_test_fit_scaler']:
                leaked_val = self.results[leaked_name][metric]
                inflation = leaked_val - correct_val
                inflation_pct = (inflation / abs(correct_val)) * 100 if correct_val != 0 else float('inf')
                print(f"  {leaked_name} - {metric}: +{inflation:.4f} ({inflation_pct:+.1f}%)")

        return self.results
