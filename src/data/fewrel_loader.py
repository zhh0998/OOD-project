"""
FewRel Dataset Loader
For Hypothesis 2 (Analogous Relation Forgetting)
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
from tqdm import tqdm


class FewRelSample:
    """Single sample in FewRel dataset"""
    def __init__(self, tokens: List[str], head: dict, tail: dict, relation: str):
        self.tokens = tokens
        self.head = head  # {'word': str, 'pos': [start, end]}
        self.tail = tail
        self.relation = relation

    @property
    def sentence(self) -> str:
        return ' '.join(self.tokens)

    @property
    def head_entity(self) -> str:
        return self.head.get('word', '')

    @property
    def tail_entity(self) -> str:
        return self.tail.get('word', '')


class FewRelDataset:
    """
    FewRel Dataset for Continual Relation Extraction

    Used for H2: Testing Analogous Relation Similarity vs Forgetting Rate
    """

    # Standard 80 relations in FewRel
    RELATION_DESCRIPTIONS = {
        'P17': 'country',
        'P19': 'place of birth',
        'P20': 'place of death',
        'P22': 'father',
        'P25': 'mother',
        'P26': 'spouse',
        'P27': 'country of citizenship',
        'P30': 'continent',
        'P31': 'instance of',
        'P35': 'head of state',
        'P36': 'capital',
        'P37': 'official language',
        'P39': 'position held',
        'P40': 'child',
        'P50': 'author',
        'P57': 'director',
        'P58': 'screenwriter',
        'P69': 'educated at',
        'P86': 'composer',
        'P102': 'member of political party',
        'P108': 'employer',
        'P112': 'founded by',
        'P118': 'league',
        'P123': 'publisher',
        'P127': 'owned by',
        'P131': 'located in administrative territorial entity',
        'P136': 'genre',
        'P137': 'operator',
        'P140': 'religion',
        'P150': 'contains administrative territorial entity',
        'P155': 'follows',
        'P156': 'followed by',
        'P159': 'headquarters location',
        'P161': 'cast member',
        'P162': 'producer',
        'P166': 'award received',
        'P170': 'creator',
        'P171': 'parent taxon',
        'P175': 'performer',
        'P176': 'manufacturer',
        'P178': 'developer',
        'P179': 'series',
        'P190': 'sister city',
        'P194': 'legislative body',
        'P241': 'military branch',
        'P264': 'record label',
        'P272': 'production company',
        'P276': 'location',
        'P279': 'subclass of',
        'P355': 'subsidiary',
        'P361': 'part of',
        'P364': 'original language of work',
        'P400': 'platform',
        'P403': 'mouth of the watercourse',
        'P449': 'original network',
        'P463': 'member of',
        'P466': 'occupant',
        'P495': 'country of origin',
        'P527': 'has part',
        'P551': 'residence',
        'P674': 'characters',
        'P706': 'located on terrain feature',
        'P710': 'participant',
        'P737': 'influenced by',
        'P740': 'location of formation',
        'P750': 'distributor',
        'P800': 'notable work',
        'P807': 'separated from',
        'P840': 'narrative location',
        'P937': 'work location',
        'P991': 'successful candidate',
        'P1001': 'applies to jurisdiction',
        'P1056': 'product or material produced',
        'P1198': 'unemployment rate',
        'P1303': 'instrument',
        'P1344': 'participant of',
        'P1376': 'capital of',
        'P1408': 'licensed to broadcast to',
        'P1411': 'nominated for',
        'P1412': 'languages spoken or written'
    }

    # Define analogous relation groups (semantically similar relations)
    ANALOGOUS_GROUPS = [
        ['P17', 'P27', 'P495'],  # Country-related
        ['P19', 'P20', 'P551', 'P937'],  # Place-related (birth, death, residence, work)
        ['P22', 'P25', 'P40', 'P26'],  # Family relations
        ['P35', 'P39', 'P102'],  # Political positions
        ['P112', 'P170', 'P178'],  # Creator-related (founder, creator, developer)
        ['P50', 'P57', 'P58', 'P86', 'P162'],  # Creative works (author, director, etc.)
        ['P123', 'P264', 'P272', 'P750'],  # Company-related (publisher, label, etc.)
        ['P127', 'P355', 'P137'],  # Ownership relations
        ['P36', 'P1376'],  # Capital relations
        ['P155', 'P156', 'P807'],  # Sequence relations
    ]

    def __init__(self, data_dir: str = './fewrel'):
        self.data_dir = data_dir
        self.train_data: Dict[str, List[FewRelSample]] = {}
        self.val_data: Dict[str, List[FewRelSample]] = {}
        self.relations: List[str] = list(self.RELATION_DESCRIPTIONS.keys())

    def load_train(self, path: Optional[str] = None) -> Dict[str, List[FewRelSample]]:
        """Load training data"""
        if path is None:
            path = os.path.join(self.data_dir, 'train_wiki.json')
        self.train_data = self._load_data(path)
        return self.train_data

    def load_val(self, path: Optional[str] = None) -> Dict[str, List[FewRelSample]]:
        """Load validation data"""
        if path is None:
            path = os.path.join(self.data_dir, 'val_wiki.json')
        self.val_data = self._load_data(path)
        return self.val_data

    def _load_data(self, path: str) -> Dict[str, List[FewRelSample]]:
        """Load data from file"""
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Using synthetic data for demonstration.")
            return self._generate_synthetic_data()

        with open(path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        data = {}
        for relation, samples in raw_data.items():
            data[relation] = []
            for sample in samples:
                tokens = sample['tokens']
                head = {
                    'word': ' '.join(tokens[sample['h'][2][0][0]:sample['h'][2][0][1]]),
                    'pos': sample['h'][2][0]
                }
                tail = {
                    'word': ' '.join(tokens[sample['t'][2][0][0]:sample['t'][2][0][1]]),
                    'pos': sample['t'][2][0]
                }
                data[relation].append(FewRelSample(tokens, head, tail, relation))

        return data

    def _generate_synthetic_data(self, samples_per_relation: int = 700) -> Dict[str, List[FewRelSample]]:
        """Generate synthetic data for demonstration"""
        np.random.seed(42)
        data = {}

        for rel in self.relations:
            data[rel] = []
            for i in range(samples_per_relation):
                tokens = f"Entity_H_{i} is related to Entity_T_{i} through {rel}".split()
                head = {'word': f'Entity_H_{i}', 'pos': [0, 1]}
                tail = {'word': f'Entity_T_{i}', 'pos': [4, 5]}
                data[rel].append(FewRelSample(tokens, head, tail, rel))

        return data

    def get_continual_tasks(
        self,
        num_tasks: int = 10,
        relations_per_task: int = 8,
        seed: int = 42
    ) -> List[List[str]]:
        """
        Split relations into continual learning tasks

        Returns: List of tasks, each containing relation IDs
        """
        np.random.seed(seed)
        relations = self.relations.copy()
        np.random.shuffle(relations)

        tasks = []
        for i in range(num_tasks):
            start = i * relations_per_task
            end = start + relations_per_task
            tasks.append(relations[start:end])

        return tasks

    def compute_analogous_score(self, rel1: str, rel2: str) -> float:
        """
        Compute Analogous Relation Score (ARS) between two relations

        This is a rule-based approximation. In the actual experiment,
        we use LLM to compute semantic similarity.
        """
        # Check if in same analogous group
        for group in self.ANALOGOUS_GROUPS:
            if rel1 in group and rel2 in group:
                return 0.8 + np.random.uniform(0, 0.2)  # High similarity

        # Check description similarity (simple word overlap)
        desc1 = self.RELATION_DESCRIPTIONS.get(rel1, '')
        desc2 = self.RELATION_DESCRIPTIONS.get(rel2, '')

        words1 = set(desc1.lower().split())
        words2 = set(desc2.lower().split())

        if words1 and words2:
            overlap = len(words1 & words2) / min(len(words1), len(words2))
            return overlap * 0.5

        return np.random.uniform(0, 0.3)  # Low similarity

    def get_task_data(
        self,
        task_relations: List[str],
        split: str = 'train'
    ) -> Dict[str, List[FewRelSample]]:
        """Get data for specific task relations"""
        data = self.train_data if split == 'train' else self.val_data
        return {rel: data.get(rel, []) for rel in task_relations}


def load_fewrel(data_dir: str = './fewrel') -> FewRelDataset:
    """Convenience function to load FewRel dataset"""
    dataset = FewRelDataset(data_dir)
    dataset.load_train()
    dataset.load_val()
    return dataset
