"""
Data Loader Module
Handles loading and preprocessing of all 6 datasets with unified split format.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups


@dataclass
class DataCard:
    """Data card for dataset documentation."""
    name: str
    train_id_n: int
    train_id_classes: int
    cal_id_n: int
    test_id_n: int
    test_ood_n: int
    test_ood_near: int
    test_ood_medium: int
    test_ood_far: int
    near_ood_definition: str
    source: str
    license: str

    def to_markdown(self) -> str:
        return f"""[DATA CARD] {self.name}:
  train_id: N={self.train_id_n}, classes={self.train_id_classes}
  cal_id: N={self.cal_id_n}
  test_id: N={self.test_id_n}
  test_ood: N={self.test_ood_n} (near={self.test_ood_near}, medium={self.test_ood_medium}, far={self.test_ood_far})
  near_ood_definition: "{self.near_ood_definition}"
  source: {self.source}
  license: {self.license}
"""


class DatasetLoader:
    """
    Unified dataset loader for all 6 datasets.

    Output format for each dataset:
    {
        "train_id": {"texts": [...], "labels": [...]},
        "cal_id": {"texts": [...], "labels": [...]},
        "test_id": {"texts": [...], "labels": [...]},
        "test_ood": {"texts": [...], "labels": [...], "ood_group": [...]}
    }
    """

    def __init__(self, output_dir: str = "data/processed", cache_dir: str = "cache/datasets"):
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_clinc150(self, cal_ratio: float = 0.2, seed: int = 42) -> Tuple[Dict, DataCard]:
        """
        Load CLINC150 dataset.
        150 in-scope intents = ID, OOS (out-of-scope) = OOD.
        Near-OOD: OOS semantically similar to ID.
        """
        from datasets import load_dataset

        print("Loading CLINC150...")
        # Try different config names for compatibility
        try:
            ds = load_dataset("DeepPavlov/clinc150", "default")
        except Exception:
            try:
                ds = load_dataset("clinc_oos", "plus")
            except Exception:
                ds = load_dataset("clinc_oos")

        # Collect data by split
        train_texts, train_labels = [], []
        val_texts, val_labels = [], []
        test_texts, test_labels = [], []
        oos_texts = []

        # Get text field name (different dataset versions use different names)
        text_field = 'utterance' if 'utterance' in ds['train'].column_names else 'text'

        # OOS samples have None label in this version
        # Process training split
        for item in ds['train']:
            text = item[text_field]
            label = item['label']
            if label is None:  # OOS class has None label
                oos_texts.append(text)
            else:
                train_texts.append(text)
                train_labels.append(label)

        # Process validation split
        for item in ds['validation']:
            text = item[text_field]
            label = item['label']
            if label is None:
                oos_texts.append(text)
            else:
                val_texts.append(text)
                val_labels.append(label)

        # Process test split
        for item in ds['test']:
            text = item[text_field]
            label = item['label']
            if label is None:
                oos_texts.append(text)
            else:
                test_texts.append(text)
                test_labels.append(label)

        # Combine train + val for ID-train (common practice)
        all_train_texts = train_texts + val_texts
        all_train_labels = train_labels + val_labels

        # Split ID-train into train and calibration
        train_texts_final, cal_texts, train_labels_final, cal_labels = train_test_split(
            all_train_texts, all_train_labels,
            test_size=cal_ratio,
            random_state=seed,
            stratify=all_train_labels
        )

        # OOD groups: for CLINC150, all OOS are "near" semantically
        # (they're designed to be confusable with in-scope intents)
        ood_groups = ['near'] * len(oos_texts)

        data = {
            "train_id": {"texts": train_texts_final, "labels": train_labels_final},
            "cal_id": {"texts": cal_texts, "labels": cal_labels},
            "test_id": {"texts": test_texts, "labels": test_labels},
            "test_ood": {"texts": oos_texts, "labels": [-1] * len(oos_texts), "ood_group": ood_groups}
        }

        card = DataCard(
            name="CLINC150",
            train_id_n=len(train_texts_final),
            train_id_classes=150,
            cal_id_n=len(cal_texts),
            test_id_n=len(test_texts),
            test_ood_n=len(oos_texts),
            test_ood_near=len(oos_texts),
            test_ood_medium=0,
            test_ood_far=0,
            near_ood_definition="OOS samples designed to be confusable with in-scope",
            source="DeepPavlov/clinc150",
            license="CC-BY-3.0"
        )

        return data, card

    def load_banking77(self, n_id: int = 50, n_ood: int = 27,
                       partition: str = "random", seed: int = 42,
                       cal_ratio: float = 0.2) -> Tuple[Dict, DataCard]:
        """
        Load Banking77 dataset with random or alphabetical partition.

        Args:
            partition: "random" or "alphabetical"
            seed: random seed for partition (only used if partition="random")
        """
        from datasets import load_dataset

        print(f"Loading Banking77 (partition={partition}, seed={seed})...")
        ds = load_dataset("banking77", trust_remote_code=True)

        # Get all unique labels
        all_labels = sorted(set(ds['train']['label']))
        label_names = ds['train'].features['label'].names

        if partition == "alphabetical":
            # Sort by label name alphabetically
            sorted_labels = sorted(range(len(label_names)), key=lambda x: label_names[x])
            id_labels = set(sorted_labels[:n_id])
            ood_labels = set(sorted_labels[n_id:])
        else:
            # Random partition
            np.random.seed(seed)
            shuffled = np.random.permutation(all_labels)
            id_labels = set(shuffled[:n_id])
            ood_labels = set(shuffled[n_id:])

        # Separate data
        train_texts, train_labels = [], []
        test_id_texts, test_id_labels = [], []
        test_ood_texts, test_ood_labels = [], []

        for item in ds['train']:
            if item['label'] in id_labels:
                train_texts.append(item['text'])
                train_labels.append(item['label'])

        for item in ds['test']:
            if item['label'] in id_labels:
                test_id_texts.append(item['text'])
                test_id_labels.append(item['label'])
            else:
                test_ood_texts.append(item['text'])
                test_ood_labels.append(item['label'])

        # Split train into train and cal
        train_texts_final, cal_texts, train_labels_final, cal_labels = train_test_split(
            train_texts, train_labels,
            test_size=cal_ratio,
            random_state=seed,
            stratify=train_labels
        )

        # OOD groups: determine based on semantic similarity
        # For Banking77, all held-out intents are semantically similar (near-OOD)
        ood_groups = ['near'] * len(test_ood_texts)

        data = {
            "train_id": {"texts": train_texts_final, "labels": train_labels_final},
            "cal_id": {"texts": cal_texts, "labels": cal_labels},
            "test_id": {"texts": test_id_texts, "labels": test_id_labels},
            "test_ood": {"texts": test_ood_texts, "labels": test_ood_labels, "ood_group": ood_groups}
        }

        card = DataCard(
            name=f"Banking77_{partition}_seed{seed}",
            train_id_n=len(train_texts_final),
            train_id_classes=n_id,
            cal_id_n=len(cal_texts),
            test_id_n=len(test_id_texts),
            test_ood_n=len(test_ood_texts),
            test_ood_near=len(test_ood_texts),
            test_ood_medium=0,
            test_ood_far=0,
            near_ood_definition=f"Hold-out {n_ood} intents ({partition} partition)",
            source="banking77",
            license="CC-BY-4.0"
        )

        return data, card

    def load_hwu64(self, holdout_domains: Optional[List[str]] = None,
                   cal_ratio: float = 0.2, seed: int = 42) -> Tuple[Dict, DataCard]:
        """
        Load HWU64 dataset. Hold out domains for OOD.
        """
        from datasets import load_dataset

        print("Loading HWU64...")
        # Try multiple sources
        ds = None
        for source in ["nlu_evaluation_data", "silicone"]:
            try:
                if source == "silicone":
                    ds = load_dataset(source, "hwu")
                else:
                    ds = load_dataset(source)
                print(f"Loaded HWU64 from {source}")
                break
            except Exception as e:
                print(f"Could not load from {source}: {e}")
                continue

        if ds is None:
            # Create minimal synthetic data for structure validation
            print("Warning: Using synthetic HWU64 data for structure validation")
            return self._create_synthetic_intent_data("HWU64", 64, cal_ratio, seed)

        # Default holdout domains (related to similar topics = near-OOD)
        if holdout_domains is None:
            holdout_domains = ["alarm", "calendar", "datetime"]

        train_texts, train_labels = [], []
        test_id_texts, test_id_labels = [], []
        test_ood_texts, test_ood_labels = [], []
        ood_groups = []

        # Process splits
        for split in ['train', 'test']:
            for item in ds[split]:
                text = item.get('text', item.get('utterance', ''))
                label = item.get('label', item.get('intent', 0))
                domain = item.get('domain', 'unknown')

                if domain in holdout_domains:
                    if split == 'test':
                        test_ood_texts.append(text)
                        test_ood_labels.append(label)
                        ood_groups.append('near')  # Same domain type = near
                else:
                    if split == 'train':
                        train_texts.append(text)
                        train_labels.append(label)
                    else:
                        test_id_texts.append(text)
                        test_id_labels.append(label)

        # Split train into train and cal
        if len(train_texts) > 0:
            train_texts_final, cal_texts, train_labels_final, cal_labels = train_test_split(
                train_texts, train_labels,
                test_size=cal_ratio,
                random_state=seed,
                stratify=train_labels if len(set(train_labels)) > 1 else None
            )
        else:
            train_texts_final, cal_texts = [], []
            train_labels_final, cal_labels = [], []

        data = {
            "train_id": {"texts": train_texts_final, "labels": train_labels_final},
            "cal_id": {"texts": cal_texts, "labels": cal_labels},
            "test_id": {"texts": test_id_texts, "labels": test_id_labels},
            "test_ood": {"texts": test_ood_texts, "labels": test_ood_labels, "ood_group": ood_groups}
        }

        n_classes = len(set(train_labels_final)) if train_labels_final else 0

        card = DataCard(
            name="HWU64",
            train_id_n=len(train_texts_final),
            train_id_classes=n_classes,
            cal_id_n=len(cal_texts),
            test_id_n=len(test_id_texts),
            test_ood_n=len(test_ood_texts),
            test_ood_near=len(test_ood_texts),
            test_ood_medium=0,
            test_ood_far=0,
            near_ood_definition=f"Hold-out domains: {holdout_domains}",
            source="hwu64",
            license="CC-BY-SA-4.0"
        )

        return data, card

    def _create_synthetic_intent_data(self, name: str, n_intents: int,
                                        cal_ratio: float, seed: int) -> Tuple[Dict, DataCard]:
        """Create synthetic intent data for structure validation."""
        np.random.seed(seed)
        n_train = 100 * n_intents
        n_test = 20 * n_intents
        n_ood = 20 * (n_intents // 4)

        train_texts = [f"synthetic intent query {i} for class {i % n_intents}" for i in range(n_train)]
        train_labels = [i % n_intents for i in range(n_train)]

        test_id_texts = [f"synthetic test query {i} for class {i % n_intents}" for i in range(n_test)]
        test_id_labels = [i % n_intents for i in range(n_test)]

        test_ood_texts = [f"synthetic ood query {i}" for i in range(n_ood)]
        test_ood_labels = [-1] * n_ood
        ood_groups = ['near'] * n_ood

        train_texts_final, cal_texts, train_labels_final, cal_labels = train_test_split(
            train_texts, train_labels, test_size=cal_ratio, random_state=seed
        )

        data = {
            "train_id": {"texts": train_texts_final, "labels": train_labels_final},
            "cal_id": {"texts": cal_texts, "labels": cal_labels},
            "test_id": {"texts": test_id_texts, "labels": test_id_labels},
            "test_ood": {"texts": test_ood_texts, "labels": test_ood_labels, "ood_group": ood_groups}
        }

        card = DataCard(
            name=f"{name}_synthetic",
            train_id_n=len(train_texts_final),
            train_id_classes=n_intents,
            cal_id_n=len(cal_texts),
            test_id_n=len(test_id_texts),
            test_ood_n=len(test_ood_texts),
            test_ood_near=len(test_ood_texts),
            test_ood_medium=0,
            test_ood_far=0,
            near_ood_definition="synthetic",
            source="synthetic",
            license="N/A"
        )

        return data, card

    def load_massive(self, holdout_domains: Optional[List[str]] = None,
                     cal_ratio: float = 0.2, seed: int = 42) -> Tuple[Dict, DataCard]:
        """
        Load MASSIVE (en-US) dataset. Hold out domains for OOD.
        """
        from datasets import load_dataset

        print("Loading MASSIVE (en-US)...")
        try:
            ds = load_dataset("AmazonScience/massive", "en-US")
        except Exception as e:
            print(f"Could not load MASSIVE: {e}")
            print("Warning: Using synthetic MASSIVE data for structure validation")
            return self._create_synthetic_intent_data("MASSIVE", 60, cal_ratio, seed)

        # Get unique domains
        all_domains = list(set(item['scenario'] for item in ds['train']))

        if holdout_domains is None:
            # Select 3 related domains for near-OOD
            np.random.seed(seed)
            holdout_domains = np.random.choice(all_domains, size=min(3, len(all_domains)), replace=False).tolist()

        train_texts, train_labels = [], []
        test_id_texts, test_id_labels = [], []
        test_ood_texts, test_ood_labels = [], []
        ood_groups = []

        label_to_id = {}

        for split in ['train', 'validation', 'test']:
            for item in ds[split]:
                text = item['utt']
                intent = item['intent']
                domain = item['scenario']

                if intent not in label_to_id:
                    label_to_id[intent] = len(label_to_id)
                label = label_to_id[intent]

                if domain in holdout_domains:
                    if split == 'test':
                        test_ood_texts.append(text)
                        test_ood_labels.append(label)
                        ood_groups.append('near')
                else:
                    if split == 'train':
                        train_texts.append(text)
                        train_labels.append(label)
                    elif split == 'test':
                        test_id_texts.append(text)
                        test_id_labels.append(label)

        # Split train into train and cal
        train_texts_final, cal_texts, train_labels_final, cal_labels = train_test_split(
            train_texts, train_labels,
            test_size=cal_ratio,
            random_state=seed,
            stratify=train_labels
        )

        data = {
            "train_id": {"texts": train_texts_final, "labels": train_labels_final},
            "cal_id": {"texts": cal_texts, "labels": cal_labels},
            "test_id": {"texts": test_id_texts, "labels": test_id_labels},
            "test_ood": {"texts": test_ood_texts, "labels": test_ood_labels, "ood_group": ood_groups}
        }

        card = DataCard(
            name="MASSIVE_en-US",
            train_id_n=len(train_texts_final),
            train_id_classes=len(set(train_labels_final)),
            cal_id_n=len(cal_texts),
            test_id_n=len(test_id_texts),
            test_ood_n=len(test_ood_texts),
            test_ood_near=len(test_ood_texts),
            test_ood_medium=0,
            test_ood_far=0,
            near_ood_definition=f"Hold-out domains: {holdout_domains}",
            source="AmazonScience/massive",
            license="CC-BY-4.0"
        )

        return data, card

    def load_20newsgroups(self, id_categories: Optional[List[str]] = None,
                          cal_ratio: float = 0.2, seed: int = 42) -> Tuple[Dict, DataCard]:
        """
        Load 20Newsgroups dataset. Hold out 5 categories for OOD.
        Categories grouped by topic similarity for near/medium/far.
        """
        print("Loading 20Newsgroups...")

        # Define category groups by topic relatedness
        category_groups = {
            'comp': ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
                     'comp.sys.mac.hardware', 'comp.windows.x'],
            'rec': ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'],
            'sci': ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'],
            'misc': ['misc.forsale'],
            'talk': ['talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast', 'talk.religion.misc'],
            'alt': ['alt.atheism'],
            'soc': ['soc.religion.christian']
        }

        all_categories = [cat for group in category_groups.values() for cat in group]

        if id_categories is None:
            # Use 15 categories as ID, 5 as OOD
            np.random.seed(seed)
            shuffled = np.random.permutation(all_categories)
            id_categories = list(shuffled[:15])
            ood_categories = list(shuffled[15:])
        else:
            ood_categories = [c for c in all_categories if c not in id_categories]

        # Load data
        train_data = fetch_20newsgroups(subset='train', categories=None, remove=('headers', 'footers', 'quotes'))
        test_data = fetch_20newsgroups(subset='test', categories=None, remove=('headers', 'footers', 'quotes'))

        train_texts, train_labels = [], []
        test_id_texts, test_id_labels = [], []
        test_ood_texts, test_ood_labels = [], []
        ood_groups = []

        id_categories_set = set(id_categories)

        # Process train
        for text, label_idx in zip(train_data.data, train_data.target):
            cat = train_data.target_names[label_idx]
            if cat in id_categories_set:
                train_texts.append(text)
                train_labels.append(int(label_idx))  # Convert to Python int

        # Process test
        for text, label_idx in zip(test_data.data, test_data.target):
            cat = test_data.target_names[label_idx]
            if cat in id_categories_set:
                test_id_texts.append(text)
                test_id_labels.append(int(label_idx))  # Convert to Python int
            else:
                test_ood_texts.append(text)
                test_ood_labels.append(int(label_idx))  # Convert to Python int
                # Determine OOD group based on topic relatedness
                # Find which group this belongs to
                cat_group = None
                for gname, gcats in category_groups.items():
                    if cat in gcats:
                        cat_group = gname
                        break
                # Check if any ID category is in same group
                id_groups = set()
                for id_cat in id_categories:
                    for gname, gcats in category_groups.items():
                        if id_cat in gcats:
                            id_groups.add(gname)
                if cat_group in id_groups:
                    ood_groups.append('near')
                elif cat_group in ['comp', 'sci', 'rec']:  # Related technical
                    ood_groups.append('medium')
                else:
                    ood_groups.append('far')

        # Split train into train and cal
        train_texts_final, cal_texts, train_labels_final, cal_labels = train_test_split(
            train_texts, train_labels,
            test_size=cal_ratio,
            random_state=seed,
            stratify=train_labels
        )

        data = {
            "train_id": {"texts": train_texts_final, "labels": train_labels_final},
            "cal_id": {"texts": cal_texts, "labels": cal_labels},
            "test_id": {"texts": test_id_texts, "labels": test_id_labels},
            "test_ood": {"texts": test_ood_texts, "labels": test_ood_labels, "ood_group": ood_groups}
        }

        near_count = sum(1 for g in ood_groups if g == 'near')
        medium_count = sum(1 for g in ood_groups if g == 'medium')
        far_count = sum(1 for g in ood_groups if g == 'far')

        card = DataCard(
            name="20Newsgroups",
            train_id_n=len(train_texts_final),
            train_id_classes=len(set(train_labels_final)),
            cal_id_n=len(cal_texts),
            test_id_n=len(test_id_texts),
            test_ood_n=len(test_ood_texts),
            test_ood_near=near_count,
            test_ood_medium=medium_count,
            test_ood_far=far_count,
            near_ood_definition="Same topic group as ID categories",
            source="sklearn.datasets.fetch_20newsgroups",
            license="Public Domain"
        )

        return data, card

    def load_nq_open(self, n_samples: int = 3000, cal_ratio: float = 0.2,
                     seed: int = 42) -> Tuple[Dict, DataCard]:
        """
        Load NQ-Open subset for retrieval gating task.
        ID: queries that can retrieve gold passage
        OOD: queries that cannot retrieve gold passage (based on embedding similarity)
        """
        from datasets import load_dataset

        print("Loading NQ-Open...")
        ds = load_dataset("nq_open", trust_remote_code=True)

        # Sample subset
        np.random.seed(seed)
        train_indices = np.random.choice(len(ds['train']), min(n_samples, len(ds['train'])), replace=False)
        val_indices = np.random.choice(len(ds['validation']), min(n_samples // 3, len(ds['validation'])), replace=False)

        train_texts = [ds['train'][int(i)]['question'] for i in train_indices]
        train_answers = [ds['train'][int(i)]['answer'] for i in train_indices]

        val_texts = [ds['validation'][int(i)]['question'] for i in val_indices]
        val_answers = [ds['validation'][int(i)]['answer'] for i in val_indices]

        # For now, create synthetic ID/OOD split based on question characteristics
        # ID: factual questions with clear answers
        # OOD: ambiguous or unanswerable-looking questions
        # (In practice, this would be determined by retrieval success)

        all_texts = train_texts + val_texts
        all_answers = train_answers + val_answers

        # Use answer length/complexity as proxy for retrieval difficulty
        train_id_texts, train_id_labels = [], []
        test_texts, test_labels, test_ood = [], [], []

        for i, (text, answer) in enumerate(zip(all_texts, all_answers)):
            # Heuristic: longer answers = harder to retrieve = more likely OOD
            avg_answer_len = np.mean([len(a) for a in answer]) if answer else 0
            is_hard = avg_answer_len > 50 or len(answer) > 3

            if i < int(len(all_texts) * 0.7):  # 70% train
                train_id_texts.append(text)
                train_id_labels.append(0)  # All ID for training
            else:  # 30% test
                test_texts.append(text)
                if is_hard:
                    test_labels.append(-1)  # OOD
                    test_ood.append('near')  # All retrieval failures are "near"
                else:
                    test_labels.append(0)  # ID
                    test_ood.append(None)

        # Split test into ID and OOD
        test_id_texts = [t for t, l in zip(test_texts, test_labels) if l == 0]
        test_id_labels = [l for l in test_labels if l == 0]
        test_ood_texts = [t for t, l in zip(test_texts, test_labels) if l == -1]
        test_ood_labels = [l for l in test_labels if l == -1]
        ood_groups = [g for g in test_ood if g is not None]

        # Split train into train and cal
        train_texts_final, cal_texts, train_labels_final, cal_labels = train_test_split(
            train_id_texts, train_id_labels,
            test_size=cal_ratio,
            random_state=seed
        )

        data = {
            "train_id": {"texts": train_texts_final, "labels": train_labels_final},
            "cal_id": {"texts": cal_texts, "labels": cal_labels},
            "test_id": {"texts": test_id_texts, "labels": test_id_labels},
            "test_ood": {"texts": test_ood_texts, "labels": test_ood_labels, "ood_group": ood_groups}
        }

        card = DataCard(
            name="NQ-Open",
            train_id_n=len(train_texts_final),
            train_id_classes=1,  # Binary: retrievable or not
            cal_id_n=len(cal_texts),
            test_id_n=len(test_id_texts),
            test_ood_n=len(test_ood_texts),
            test_ood_near=len(test_ood_texts),
            test_ood_medium=0,
            test_ood_far=0,
            near_ood_definition="Queries with difficult/ambiguous retrieval targets",
            source="nq_open",
            license="Apache-2.0"
        )

        return data, card

    def save_dataset(self, data: Dict, card: DataCard, dataset_name: str):
        """Save processed dataset and data card."""
        output_path = self.output_dir / dataset_name
        output_path.mkdir(parents=True, exist_ok=True)

        # Save data
        with open(output_path / "data.json", 'w') as f:
            json.dump(data, f)

        # Save card
        with open(output_path / "card.json", 'w') as f:
            json.dump(asdict(card), f, indent=2)

        with open(output_path / "card.md", 'w') as f:
            f.write(card.to_markdown())

        print(f"Saved {dataset_name} to {output_path}")

    def load_all_datasets(self, banking77_seeds: List[int] = None) -> Dict[str, Tuple[Dict, DataCard]]:
        """Load all 6 datasets with multiple Banking77 partitions."""
        if banking77_seeds is None:
            banking77_seeds = [42, 123, 456, 789, 2024]

        datasets = {}

        # D1: CLINC150
        data, card = self.load_clinc150()
        datasets['clinc150'] = (data, card)
        self.save_dataset(data, card, 'clinc150')

        # D2: Banking77 (alphabetical + random seeds)
        data, card = self.load_banking77(partition='alphabetical')
        datasets['banking77_alpha'] = (data, card)
        self.save_dataset(data, card, 'banking77_alpha')

        for seed in banking77_seeds:
            data, card = self.load_banking77(partition='random', seed=seed)
            datasets[f'banking77_seed{seed}'] = (data, card)
            self.save_dataset(data, card, f'banking77_seed{seed}')

        # D3: HWU64
        data, card = self.load_hwu64()
        datasets['hwu64'] = (data, card)
        self.save_dataset(data, card, 'hwu64')

        # D4: MASSIVE
        data, card = self.load_massive()
        datasets['massive'] = (data, card)
        self.save_dataset(data, card, 'massive')

        # D5: 20Newsgroups
        data, card = self.load_20newsgroups()
        datasets['newsgroups'] = (data, card)
        self.save_dataset(data, card, 'newsgroups')

        # D6: NQ-Open
        data, card = self.load_nq_open()
        datasets['nq_open'] = (data, card)
        self.save_dataset(data, card, 'nq_open')

        return datasets
