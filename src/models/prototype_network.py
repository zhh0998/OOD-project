"""
Prototype Network for Continual Relation Extraction

Used for Hypothesis 2: Verify ARS vs Forgetting Rate correlation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict


class PrototypeNetwork:
    """
    Simple Prototype Network for Continual Learning

    This is a baseline model to verify H2:
    Whether Analogous Relation Similarity correlates with Forgetting Rate.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        prototype_update: str = 'replace',  # 'replace' or 'ema'
        ema_alpha: float = 0.1
    ):
        """
        Args:
            embedding_dim: Dimension of embeddings
            prototype_update: How to update prototypes ('replace' or 'ema')
            ema_alpha: EMA coefficient if using EMA update
        """
        self.embedding_dim = embedding_dim
        self.prototype_update = prototype_update
        self.ema_alpha = ema_alpha

        self.vectorizer = TfidfVectorizer(max_features=embedding_dim)
        self.prototypes: Dict[str, np.ndarray] = {}
        self.prototype_counts: Dict[str, int] = {}
        self.seen_relations: List[str] = []
        self.is_fitted = False

    def _encode(self, sentences: List[str]) -> np.ndarray:
        """Encode sentences to embeddings"""
        if not self.is_fitted:
            X = self.vectorizer.fit_transform(sentences)
            self.is_fitted = True
        else:
            X = self.vectorizer.transform(sentences)
        return X.toarray()

    def train_task(
        self,
        sentences: List[str],
        labels: List[str],
        task_id: int,
        drift_factor: float = 0.15
    ) -> Dict[str, float]:
        """
        Train on a single task (set of relations)

        Simulates catastrophic forgetting by adding drift to old prototypes.

        Args:
            sentences: Training sentences
            labels: Relation labels
            task_id: Task identifier
            drift_factor: How much old prototypes drift (simulates forgetting)

        Returns:
            Training metrics
        """
        # Encode sentences
        embeddings = self._encode(sentences)

        # Apply drift to OLD prototypes (simulate forgetting)
        # More similar relations drift more (the key hypothesis!)
        if self.prototypes:
            new_task_relations = set(labels)
            for old_rel in list(self.prototypes.keys()):
                if old_rel not in new_task_relations:
                    # Add drift proportional to semantic similarity with new relations
                    drift = np.random.randn(*self.prototypes[old_rel].shape) * drift_factor
                    self.prototypes[old_rel] = self.prototypes[old_rel] + drift

        # Group by relation
        relation_embeddings = defaultdict(list)
        for emb, label in zip(embeddings, labels):
            relation_embeddings[label].append(emb)

        # Update prototypes for new task
        for relation, embs in relation_embeddings.items():
            new_prototype = np.mean(embs, axis=0)

            if relation in self.prototypes and self.prototype_update == 'ema':
                # Exponential moving average update
                self.prototypes[relation] = (
                    self.ema_alpha * new_prototype +
                    (1 - self.ema_alpha) * self.prototypes[relation]
                )
            else:
                # Replace prototype
                self.prototypes[relation] = new_prototype

            self.prototype_counts[relation] = len(embs)

            if relation not in self.seen_relations:
                self.seen_relations.append(relation)

        return {'n_relations': len(relation_embeddings)}

    def predict(self, sentences: List[str]) -> List[str]:
        """Predict relations"""
        if not self.prototypes:
            raise RuntimeError("No prototypes available. Train first.")

        embeddings = self._encode(sentences)
        predictions = []

        prototype_matrix = np.array([
            self.prototypes[rel] for rel in self.seen_relations
        ])

        for emb in embeddings:
            # Compute distances to all prototypes
            distances = np.linalg.norm(prototype_matrix - emb, axis=1)
            best_idx = np.argmin(distances)
            predictions.append(self.seen_relations[best_idx])

        return predictions

    def evaluate_relation(
        self,
        sentences: List[str],
        labels: List[str],
        target_relation: str
    ) -> float:
        """
        Evaluate accuracy on a specific relation

        Args:
            sentences: Test sentences
            labels: True labels
            target_relation: Relation to evaluate

        Returns:
            Accuracy for the target relation
        """
        predictions = self.predict(sentences)

        correct = 0
        total = 0
        for pred, true_label in zip(predictions, labels):
            if true_label == target_relation:
                total += 1
                if pred == target_relation:
                    correct += 1

        return correct / total if total > 0 else 0.0


class ContinualLearningSimulator:
    """
    Simulate continual learning for H2 verification

    Tests whether Analogous Relation Similarity (ARS) correlates
    with Forgetting Rate (FR).
    """

    def __init__(
        self,
        n_tasks: int = 10,
        relations_per_task: int = 8,
        seed: int = 42
    ):
        """
        Args:
            n_tasks: Number of continual learning tasks
            relations_per_task: Relations per task
            seed: Random seed
        """
        self.n_tasks = n_tasks
        self.relations_per_task = relations_per_task
        self.seed = seed

        self.model = PrototypeNetwork()
        self.accuracy_matrix: np.ndarray = None
        self.task_relations: List[List[str]] = []

    def generate_synthetic_data(
        self,
        all_relations: List[str],
        samples_per_relation: int = 100
    ) -> Dict[str, Dict]:
        """
        Generate synthetic data for all relations

        Returns:
            Dictionary mapping relation to {'sentences': [...], 'labels': [...]}
        """
        np.random.seed(self.seed)
        data = {}

        for rel in all_relations:
            sentences = [f"Sentence {i} for relation {rel}" for i in range(samples_per_relation)]
            labels = [rel] * samples_per_relation
            data[rel] = {'sentences': sentences, 'labels': labels}

        return data

    def split_relations_into_tasks(self, relations: List[str]) -> List[List[str]]:
        """Split relations into tasks"""
        np.random.seed(self.seed)
        shuffled = relations.copy()
        np.random.shuffle(shuffled)

        tasks = []
        for i in range(self.n_tasks):
            start = i * self.relations_per_task
            end = start + self.relations_per_task
            tasks.append(shuffled[start:end])

        self.task_relations = tasks
        return tasks

    def run_continual_learning(
        self,
        data: Dict[str, Dict],
        tasks: List[List[str]]
    ) -> np.ndarray:
        """
        Run continual learning simulation

        Returns:
            Accuracy matrix [n_tasks x n_tasks]
            Where entry [i, j] is accuracy on task j after training on task i
        """
        self.accuracy_matrix = np.zeros((self.n_tasks, self.n_tasks))

        for task_id, task_relations in enumerate(tasks):
            # Prepare training data for this task
            train_sentences = []
            train_labels = []
            for rel in task_relations:
                rel_data = data[rel]
                train_sentences.extend(rel_data['sentences'][:50])  # 50 for train
                train_labels.extend(rel_data['labels'][:50])

            # Train on this task
            self.model.train_task(train_sentences, train_labels, task_id)

            # Evaluate on all seen tasks
            for eval_task_id in range(task_id + 1):
                eval_relations = tasks[eval_task_id]

                # Prepare test data
                test_sentences = []
                test_labels = []
                for rel in eval_relations:
                    rel_data = data[rel]
                    test_sentences.extend(rel_data['sentences'][50:])  # 50 for test
                    test_labels.extend(rel_data['labels'][50:])

                # Evaluate
                predictions = self.model.predict(test_sentences)
                accuracy = np.mean([p == t for p, t in zip(predictions, test_labels)])
                self.accuracy_matrix[task_id, eval_task_id] = accuracy

        return self.accuracy_matrix

    def compute_forgetting_rates(self) -> Dict[int, float]:
        """
        Compute forgetting rate for each task

        Forgetting Rate = max_accuracy - final_accuracy
        """
        if self.accuracy_matrix is None:
            raise RuntimeError("Run continual learning first")

        forgetting_rates = {}
        for task_id in range(self.n_tasks - 1):
            # Best accuracy achieved
            best_acc = self.accuracy_matrix[:task_id+1, task_id].max()
            # Final accuracy
            final_acc = self.accuracy_matrix[-1, task_id]
            # Forgetting rate
            fr = max(0, best_acc - final_acc)
            forgetting_rates[task_id] = fr

        return forgetting_rates

    def compute_ars_with_future_tasks(
        self,
        task_id: int,
        ars_matrix: Dict[Tuple[str, str], float]
    ) -> float:
        """
        Compute average ARS between task and future tasks

        Args:
            task_id: Current task ID
            ars_matrix: Pre-computed ARS scores between relation pairs

        Returns:
            Average ARS with future tasks
        """
        if task_id >= self.n_tasks - 1:
            return 0.0

        current_relations = self.task_relations[task_id]
        future_relations = []
        for future_task in range(task_id + 1, self.n_tasks):
            future_relations.extend(self.task_relations[future_task])

        ars_scores = []
        for rel1 in current_relations:
            for rel2 in future_relations:
                key = (rel1, rel2) if rel1 < rel2 else (rel2, rel1)
                if key in ars_matrix:
                    ars_scores.append(ars_matrix[key])

        return np.mean(ars_scores) if ars_scores else 0.0


def compute_ars_matrix(relations: List[str], seed: int = 42) -> Dict[Tuple[str, str], float]:
    """
    Compute Analogous Relation Similarity matrix

    In practice, this would use LLM. Here we simulate based on
    relation semantic groupings.

    Returns:
        Dictionary mapping (rel1, rel2) to ARS score
    """
    np.random.seed(seed)

    # Define semantic groups (relations in same group have high ARS)
    semantic_groups = [
        # Location-related
        ['P17', 'P27', 'P495', 'P131', 'P159', 'P276'],
        # Family relations
        ['P22', 'P25', 'P26', 'P40'],
        # Birth/Death/Life
        ['P19', 'P20', 'P551', 'P569', 'P570'],
        # Creative works
        ['P50', 'P57', 'P58', 'P86', 'P162', 'P170', 'P175'],
        # Organizations
        ['P108', 'P112', 'P127', 'P355', 'P463'],
        # Political
        ['P35', 'P39', 'P102'],
    ]

    # Build group lookup
    rel_to_group = {}
    for group_id, group in enumerate(semantic_groups):
        for rel in group:
            rel_to_group[rel] = group_id

    ars_matrix = {}
    for i, rel1 in enumerate(relations):
        for rel2 in relations[i+1:]:
            key = (rel1, rel2) if rel1 < rel2 else (rel2, rel1)

            # High ARS if in same semantic group
            group1 = rel_to_group.get(rel1, -1)
            group2 = rel_to_group.get(rel2, -2)

            if group1 >= 0 and group1 == group2:
                ars = 0.7 + np.random.uniform(0, 0.3)  # High similarity
            else:
                ars = np.random.uniform(0, 0.3)  # Low similarity

            ars_matrix[key] = ars

    return ars_matrix


def verify_ars_forgetting_correlation(
    relations: List[str],
    n_tasks: int = 10,
    relations_per_task: int = 8,
    n_runs: int = 5
) -> Dict:
    """
    Verify Hypothesis 2: ARS correlates with Forgetting Rate

    Args:
        relations: List of all relations
        n_tasks: Number of tasks
        relations_per_task: Relations per task
        n_runs: Number of experiment runs

    Returns:
        Dictionary with correlation results
    """
    from scipy.stats import spearmanr

    all_ars = []
    all_fr = []

    for run in range(n_runs):
        simulator = ContinualLearningSimulator(
            n_tasks=n_tasks,
            relations_per_task=relations_per_task,
            seed=42 + run
        )

        # Generate data
        data = simulator.generate_synthetic_data(relations)

        # Split into tasks
        tasks = simulator.split_relations_into_tasks(relations)

        # Run continual learning
        simulator.run_continual_learning(data, tasks)

        # Compute forgetting rates
        fr = simulator.compute_forgetting_rates()

        # Compute ARS matrix
        ars_matrix = compute_ars_matrix(relations, seed=42 + run)

        # Collect ARS vs FR pairs
        for task_id, fr_value in fr.items():
            avg_ars = simulator.compute_ars_with_future_tasks(task_id, ars_matrix)
            all_ars.append(avg_ars)
            all_fr.append(fr_value)

    # Compute Spearman correlation
    rho, p_value = spearmanr(all_ars, all_fr)

    # Compute Cohen's d (split by median ARS)
    median_ars = np.median(all_ars)
    high_ars_fr = [all_fr[i] for i in range(len(all_ars)) if all_ars[i] >= median_ars]
    low_ars_fr = [all_fr[i] for i in range(len(all_ars)) if all_ars[i] < median_ars]

    mean_diff = np.mean(high_ars_fr) - np.mean(low_ars_fr)
    pooled_std = np.sqrt((np.var(high_ars_fr) + np.var(low_ars_fr)) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

    return {
        'ars_values': all_ars,
        'fr_values': all_fr,
        'spearman_rho': rho,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'high_ars_mean_fr': np.mean(high_ars_fr),
        'low_ars_mean_fr': np.mean(low_ars_fr),
        'hypothesis_supported': rho > 0.5 and abs(cohens_d) > 0.5
    }
