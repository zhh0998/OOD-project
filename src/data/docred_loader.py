"""
DocRED / Re-DocRED Dataset Loader
For Hypothesis 4 (Path Length vs False Negative Rate)
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import networkx as nx


class DocREDMention:
    """Entity mention in a document"""
    def __init__(self, name: str, pos: List[int], sent_id: int, entity_id: int):
        self.name = name
        self.pos = pos  # [start, end] token positions
        self.sent_id = sent_id
        self.entity_id = entity_id


class DocREDEntity:
    """Entity with multiple mentions"""
    def __init__(self, entity_id: int, entity_type: str):
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.mentions: List[DocREDMention] = []

    def add_mention(self, mention: DocREDMention):
        self.mentions.append(mention)

    @property
    def name(self) -> str:
        """Return the most common mention name"""
        names = [m.name for m in self.mentions]
        return max(set(names), key=names.count) if names else ""

    @property
    def sentence_ids(self) -> Set[int]:
        """Return all sentence IDs where this entity appears"""
        return {m.sent_id for m in self.mentions}


class DocREDRelation:
    """Relation between entities"""
    def __init__(self, head_id: int, tail_id: int, relation: str, evidence: List[int]):
        self.head_id = head_id
        self.tail_id = tail_id
        self.relation = relation
        self.evidence = evidence  # Sentence IDs that support this relation


class DocREDDocument:
    """A document with entities and relations"""
    def __init__(self, doc_data: dict):
        self.title = doc_data.get('title', '')
        self.sentences: List[List[str]] = doc_data.get('sents', [])
        self.entities: Dict[int, DocREDEntity] = {}
        self.relations: List[DocREDRelation] = []
        self.graph: Optional[nx.Graph] = None

        self._parse_entities(doc_data.get('vertexSet', []))
        self._parse_relations(doc_data.get('labels', []))

    def _parse_entities(self, vertex_set: List[List[dict]]):
        """Parse entity vertex set"""
        for entity_id, mentions in enumerate(vertex_set):
            if not mentions:
                continue
            entity_type = mentions[0].get('type', 'UNK')
            entity = DocREDEntity(entity_id, entity_type)

            for m in mentions:
                mention = DocREDMention(
                    name=m.get('name', ''),
                    pos=m.get('pos', [0, 0]),
                    sent_id=m.get('sent_id', 0),
                    entity_id=entity_id
                )
                entity.add_mention(mention)

            self.entities[entity_id] = entity

    def _parse_relations(self, labels: List[dict]):
        """Parse relation labels"""
        for label in labels:
            relation = DocREDRelation(
                head_id=label.get('h', 0),
                tail_id=label.get('t', 0),
                relation=label.get('r', ''),
                evidence=label.get('evidence', [])
            )
            self.relations.append(relation)

    def build_graph(self) -> nx.Graph:
        """
        Build document heterogeneous graph for path analysis

        Nodes: Mentions (M), Entities (E), Sentences (S)
        Edges:
            - M-E: mention belongs to entity
            - M-S: mention appears in sentence
            - S-S: consecutive sentences
        """
        G = nx.Graph()

        # Add sentence nodes
        for sent_id in range(len(self.sentences)):
            G.add_node(f'S_{sent_id}', type='sentence')

        # Add consecutive sentence edges
        for sent_id in range(len(self.sentences) - 1):
            G.add_edge(f'S_{sent_id}', f'S_{sent_id + 1}', type='consecutive')

        # Add entity and mention nodes
        for entity_id, entity in self.entities.items():
            G.add_node(f'E_{entity_id}', type='entity')

            for i, mention in enumerate(entity.mentions):
                m_id = f'M_{entity_id}_{i}'
                G.add_node(m_id, type='mention')

                # Mention-Entity edge
                G.add_edge(m_id, f'E_{entity_id}', type='belong_to')

                # Mention-Sentence edge
                G.add_edge(m_id, f'S_{mention.sent_id}', type='in_sentence')

        self.graph = G
        return G

    def compute_path_length(self, head_id: int, tail_id: int) -> int:
        """
        Compute shortest path length between two entities

        Returns the minimum number of hops between entity nodes.
        """
        if self.graph is None:
            self.build_graph()

        head_node = f'E_{head_id}'
        tail_node = f'E_{tail_id}'

        try:
            path_length = nx.shortest_path_length(self.graph, head_node, tail_node)
            return path_length
        except nx.NetworkXNoPath:
            return -1  # No path exists

    def get_entity_distance(self, head_id: int, tail_id: int) -> int:
        """
        Compute sentence-level distance between entities

        Returns minimum sentence distance between any two mentions.
        """
        head_sents = self.entities[head_id].sentence_ids
        tail_sents = self.entities[tail_id].sentence_ids

        min_distance = float('inf')
        for h_sent in head_sents:
            for t_sent in tail_sents:
                dist = abs(h_sent - t_sent)
                min_distance = min(min_distance, dist)

        return int(min_distance) if min_distance != float('inf') else -1

    @property
    def num_sentences(self) -> int:
        return len(self.sentences)

    @property
    def num_entities(self) -> int:
        return len(self.entities)

    @property
    def full_text(self) -> str:
        return ' '.join([' '.join(sent) for sent in self.sentences])


class DocREDDataset:
    """
    DocRED / Re-DocRED Dataset for Document-level Relation Extraction

    Used for H4: Testing Path Length vs False Negative Rate
    """

    # DocRED relation types (96 relations)
    RELATION_TYPES = [
        'P6', 'P17', 'P19', 'P20', 'P22', 'P25', 'P26', 'P27', 'P30', 'P31',
        'P35', 'P36', 'P37', 'P39', 'P40', 'P50', 'P54', 'P57', 'P58', 'P69',
        'P86', 'P102', 'P108', 'P112', 'P118', 'P123', 'P127', 'P131', 'P136',
        'P137', 'P140', 'P150', 'P155', 'P156', 'P159', 'P161', 'P162', 'P166',
        'P170', 'P171', 'P172', 'P175', 'P176', 'P178', 'P179', 'P190', 'P194',
        'P205', 'P206', 'P241', 'P264', 'P272', 'P276', 'P279', 'P355', 'P361',
        'P364', 'P400', 'P403', 'P449', 'P463', 'P466', 'P495', 'P527', 'P551',
        'P569', 'P570', 'P571', 'P576', 'P577', 'P580', 'P582', 'P585', 'P607',
        'P674', 'P676', 'P706', 'P710', 'P737', 'P740', 'P749', 'P750', 'P800',
        'P807', 'P840', 'P937', 'P1001', 'P1056', 'P1198', 'P1336', 'P1344',
        'P1365', 'P1366', 'P1376', 'P1412', 'P1441'
    ]

    def __init__(self, data_dir: str = './docred'):
        self.data_dir = data_dir
        self.train_docs: List[DocREDDocument] = []
        self.dev_docs: List[DocREDDocument] = []
        self.test_docs: List[DocREDDocument] = []

    def load_train(self, path: Optional[str] = None) -> List[DocREDDocument]:
        """Load training documents"""
        if path is None:
            path = os.path.join(self.data_dir, 'train_annotated.json')
        self.train_docs = self._load_data(path)
        return self.train_docs

    def load_dev(self, path: Optional[str] = None) -> List[DocREDDocument]:
        """Load development documents"""
        if path is None:
            # Try revised version first, then original
            path = os.path.join(self.data_dir, 'dev_revised.json')
            if not os.path.exists(path):
                path = os.path.join(self.data_dir, 'dev.json')
        self.dev_docs = self._load_data(path)
        return self.dev_docs

    def load_test(self, path: Optional[str] = None) -> List[DocREDDocument]:
        """Load test documents"""
        if path is None:
            path = os.path.join(self.data_dir, 'test.json')
        self.test_docs = self._load_data(path)
        return self.test_docs

    def _load_data(self, path: str) -> List[DocREDDocument]:
        """Load data from JSON file"""
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Using synthetic data for demonstration.")
            return self._generate_synthetic_docs()

        with open(path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        docs = []
        for doc_data in tqdm(raw_data, desc=f"Loading {os.path.basename(path)}"):
            docs.append(DocREDDocument(doc_data))

        return docs

    def _generate_synthetic_docs(self, n_docs: int = 500) -> List[DocREDDocument]:
        """Generate synthetic documents for demonstration"""
        np.random.seed(42)
        docs = []

        for doc_id in range(n_docs):
            # Generate 5-10 sentences
            n_sents = np.random.randint(5, 11)
            sents = [[f"Word_{i}_{j}" for j in range(10)] for i in range(n_sents)]

            # Generate 3-8 entities
            n_entities = np.random.randint(3, 9)
            vertex_set = []
            for ent_id in range(n_entities):
                n_mentions = np.random.randint(1, 4)
                mentions = []
                for m in range(n_mentions):
                    sent_id = np.random.randint(0, n_sents)
                    mentions.append({
                        'name': f'Entity_{ent_id}',
                        'pos': [0, 2],
                        'sent_id': sent_id,
                        'type': 'PER' if ent_id % 2 == 0 else 'LOC'
                    })
                vertex_set.append(mentions)

            # Generate relations (with varying path lengths)
            labels = []
            for _ in range(np.random.randint(2, 6)):
                h = np.random.randint(0, n_entities)
                t = np.random.randint(0, n_entities)
                if h != t:
                    labels.append({
                        'h': h,
                        't': t,
                        'r': np.random.choice(self.RELATION_TYPES),
                        'evidence': [np.random.randint(0, n_sents)]
                    })

            doc_data = {
                'title': f'Doc_{doc_id}',
                'sents': sents,
                'vertexSet': vertex_set,
                'labels': labels
            }
            docs.append(DocREDDocument(doc_data))

        return docs

    def analyze_path_lengths(self, docs: Optional[List[DocREDDocument]] = None) -> Dict:
        """
        Analyze path length distribution across documents

        Returns statistics about path lengths for all relations.
        """
        if docs is None:
            docs = self.dev_docs if self.dev_docs else self.train_docs

        path_lengths = []
        sent_distances = []

        for doc in tqdm(docs, desc="Analyzing path lengths"):
            doc.build_graph()

            for rel in doc.relations:
                path_len = doc.compute_path_length(rel.head_id, rel.tail_id)
                sent_dist = doc.get_entity_distance(rel.head_id, rel.tail_id)

                if path_len > 0:
                    path_lengths.append(path_len)
                if sent_dist >= 0:
                    sent_distances.append(sent_dist)

        return {
            'path_lengths': path_lengths,
            'sent_distances': sent_distances,
            'mean_path_length': np.mean(path_lengths) if path_lengths else 0,
            'mean_sent_distance': np.mean(sent_distances) if sent_distances else 0,
            'path_length_std': np.std(path_lengths) if path_lengths else 0,
            'sent_distance_std': np.std(sent_distances) if sent_distances else 0
        }


def load_docred(data_dir: str = './docred') -> DocREDDataset:
    """Convenience function to load DocRED dataset"""
    dataset = DocREDDataset(data_dir)
    dataset.load_train()
    dataset.load_dev()
    return dataset
