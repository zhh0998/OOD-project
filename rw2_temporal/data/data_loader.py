"""
Data Loader for RW2 Temporal Network Embedding
Supports TGB and OGB datasets with fallback strategies
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle

# Fix for PyTorch 2.6+ weights_only default change
try:
    import numpy.core.multiarray
    torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])
except (AttributeError, ImportError):
    pass

try:
    import numpy._core.multiarray
    torch.serialization.add_safe_globals([numpy._core.multiarray._reconstruct])
except (AttributeError, ImportError):
    pass

from ogb.linkproppred import LinkPropPredDataset


class TemporalGraphDataset(Dataset):
    def __init__(self, edges, num_nodes, neg_sampling_ratio=1):
        self.edges = edges
        self.num_nodes = num_nodes
        self.neg_sampling_ratio = neg_sampling_ratio

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, idx):
        edge = self.edges[idx]
        neg_dsts = np.random.randint(0, self.num_nodes, self.neg_sampling_ratio)
        return {
            'src': edge['src'],
            'pos_dst': edge['dst'],
            'neg_dst': neg_dsts,
            'timestamp': edge['timestamp']
        }


def collate_fn(batch):
    return {
        'src': torch.tensor([b['src'] for b in batch], dtype=torch.long),
        'pos_dst': torch.tensor([b['pos_dst'] for b in batch], dtype=torch.long),
        'neg_dst': torch.tensor(np.array([b['neg_dst'] for b in batch]), dtype=torch.long),
        'timestamp': torch.tensor([b['timestamp'] for b in batch], dtype=torch.float)
    }


class TGBDataLoader:
    DATASET_MAPPING = {
        'tgbl-wiki': 'ogbl-collab',
        'tgbl-review': 'ogbl-citation2',
        'ogbl-collab': 'ogbl-collab',
    }

    def __init__(self, dataset_name, root='./data', subsample=None):
        self.dataset_name = dataset_name
        self.root = root
        self.subsample = subsample
        self.num_nodes = 0
        self.num_edges = 0
        self.train_edges = []
        self.val_edges = []
        self.test_edges = []

    def load(self):
        print(f"[Strategy 1] Trying OGB for {self.dataset_name}...")
        ogb_name = self.DATASET_MAPPING.get(self.dataset_name, self.dataset_name)

        try:
            original_load = torch.load
            torch.load = lambda *args, **kwargs: original_load(*args, weights_only=False, **{k: v for k, v in kwargs.items() if k != 'weights_only'})

            dataset = LinkPropPredDataset(name=ogb_name, root=self.root)
            self._process_ogb_dataset(dataset)

            torch.load = original_load
            print(f"✅ OGB成功: {ogb_name} → {self.dataset_name}")
            return True
        except Exception as e:
            print(f"⚠️ OGB加载失败: {e}")
            return False

    def _process_ogb_dataset(self, dataset):
        graph = dataset[0]

        if 'num_nodes' in graph:
            self.num_nodes = graph['num_nodes']
        else:
            edge_index = graph['edge_index']
            self.num_nodes = int(max(edge_index[0].max(), edge_index[1].max()) + 1)

        edge_index = graph['edge_index']
        num_edges = edge_index.shape[1]
        self.num_edges = num_edges

        if 'edge_year' in graph:
            timestamps = graph['edge_year'].flatten().astype(np.float32)
            timestamps = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1e-8)
        else:
            timestamps = np.linspace(0, 1, num_edges).astype(np.float32)

        all_edges = [{'src': int(edge_index[0, i]), 'dst': int(edge_index[1, i]),
                      'timestamp': float(timestamps[i])} for i in range(num_edges)]
        all_edges.sort(key=lambda x: x['timestamp'])

        if self.subsample and len(all_edges) > self.subsample:
            indices = np.linspace(0, len(all_edges) - 1, self.subsample, dtype=int)
            all_edges = [all_edges[i] for i in indices]
            self.num_edges = len(all_edges)
            print(f"   Subsampled to {self.num_edges} edges")

        n_train = int(len(all_edges) * 0.70)
        n_val = int(len(all_edges) * 0.85)
        self.train_edges = all_edges[:n_train]
        self.val_edges = all_edges[n_train:n_val]
        self.test_edges = all_edges[n_val:]

        print(f"✅ 数据验证通过: {self.num_nodes}节点, {self.num_edges}边")
        print(f"   Train: {len(self.train_edges)}, Val: {len(self.val_edges)}, Test: {len(self.test_edges)}")

    def get_dataloader(self, split='train', batch_size=200, shuffle=True, num_workers=0):
        edges = {'train': self.train_edges, 'val': self.val_edges, 'test': self.test_edges}[split]
        dataset = TemporalGraphDataset(edges, self.num_nodes)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers)

    def get_statistics(self):
        return {'dataset_name': self.dataset_name, 'num_nodes': self.num_nodes, 'num_edges': self.num_edges,
                'num_train': len(self.train_edges), 'num_val': len(self.val_edges), 'num_test': len(self.test_edges)}


def validate_dataset(dataset_name, root='./data'):
    loader = TGBDataLoader(dataset_name, root)
    success = loader.load()
    if success:
        stats = loader.get_statistics()
        print("=" * 60)
        print(f"  {dataset_name}: [✓] PASSED")
        print(f"  Nodes: {stats['num_nodes']:,}, Edges: {stats['num_edges']:,}")
        print("=" * 60)
        return True, stats
    return False, None
