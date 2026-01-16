"""
Negative sampling strategies for temporal link prediction.

Author: RW2 Temporal Network Embedding Project
"""

import numpy as np
import torch
from typing import Optional, Set, Tuple, List
from collections import defaultdict


class NegativeSampler:
    """
    Basic negative sampler for link prediction.
    """

    def __init__(
        self,
        num_nodes: int,
        num_negatives: int = 100,
        seed: int = 42
    ):
        self.num_nodes = num_nodes
        self.num_negatives = num_negatives
        self.rng = np.random.RandomState(seed)

    def sample(
        self,
        src: int,
        dst: int,
        exclude: Optional[Set[int]] = None
    ) -> np.ndarray:
        """
        Sample negative destinations for a given source.

        Args:
            src: Source node
            dst: True destination (will be excluded)
            exclude: Additional nodes to exclude

        Returns:
            Array of negative destination node IDs
        """
        exclude_set = {dst}
        if exclude is not None:
            exclude_set = exclude_set | exclude

        # Sample with rejection
        negatives = []
        while len(negatives) < self.num_negatives:
            candidates = self.rng.randint(0, self.num_nodes, size=self.num_negatives * 2)
            for c in candidates:
                if c not in exclude_set:
                    negatives.append(c)
                    if len(negatives) >= self.num_negatives:
                        break

        return np.array(negatives[:self.num_negatives])


class TemporalNegativeSampler:
    """
    Temporal-aware negative sampler.
    Samples negatives that don't have edges before the given timestamp.
    """

    def __init__(
        self,
        src_nodes: np.ndarray,
        dst_nodes: np.ndarray,
        timestamps: np.ndarray,
        num_nodes: int,
        num_negatives: int = 100,
        strategy: str = 'historical',
        seed: int = 42
    ):
        """
        Args:
            src_nodes: Source nodes of all edges
            dst_nodes: Destination nodes of all edges
            timestamps: Timestamps of all edges
            num_nodes: Total number of nodes
            num_negatives: Number of negative samples per edge
            strategy: 'historical' (exclude all past edges) or 'random'
            seed: Random seed
        """
        self.num_nodes = num_nodes
        self.num_negatives = num_negatives
        self.strategy = strategy
        self.rng = np.random.RandomState(seed)

        # Build temporal edge index
        self.temporal_edges = defaultdict(set)
        for s, d, t in zip(src_nodes, dst_nodes, timestamps):
            self.temporal_edges[(int(s), float(t))].add(int(d))

        # Precompute cumulative neighbors for efficient lookup
        self._build_cumulative_index(src_nodes, dst_nodes, timestamps)

    def _build_cumulative_index(
        self,
        src_nodes: np.ndarray,
        dst_nodes: np.ndarray,
        timestamps: np.ndarray
    ):
        """Build index for efficient historical neighbor lookup."""
        # Sort by timestamp
        sort_idx = np.argsort(timestamps)
        sorted_src = src_nodes[sort_idx]
        sorted_dst = dst_nodes[sort_idx]
        sorted_ts = timestamps[sort_idx]

        self.node_neighbors_by_time = defaultdict(list)
        self.unique_timestamps = np.unique(sorted_ts)

        # Track cumulative neighbors for each node
        cumulative_neighbors = defaultdict(set)

        current_time_idx = 0
        for i, (s, d, t) in enumerate(zip(sorted_src, sorted_dst, sorted_ts)):
            # Update when we move to a new timestamp
            while (current_time_idx < len(self.unique_timestamps) and
                   self.unique_timestamps[current_time_idx] < t):
                for node in cumulative_neighbors:
                    self.node_neighbors_by_time[node].append(
                        (self.unique_timestamps[current_time_idx],
                         cumulative_neighbors[node].copy())
                    )
                current_time_idx += 1

            cumulative_neighbors[int(s)].add(int(d))

    def get_historical_neighbors(self, node: int, timestamp: float) -> Set[int]:
        """Get all neighbors of a node before the given timestamp."""
        neighbors = set()

        if node not in self.node_neighbors_by_time:
            return neighbors

        for t, neighbor_set in self.node_neighbors_by_time[node]:
            if t < timestamp:
                neighbors = neighbor_set.copy()
            else:
                break

        return neighbors

    def sample(
        self,
        src: int,
        dst: int,
        timestamp: float
    ) -> np.ndarray:
        """
        Sample negative destinations considering temporal constraints.

        Args:
            src: Source node
            dst: True destination
            timestamp: Current timestamp

        Returns:
            Array of negative destination node IDs
        """
        if self.strategy == 'random':
            # Simple random sampling (exclude only the true edge)
            exclude_set = {dst}
        else:
            # Historical: exclude all nodes that src has connected to before
            exclude_set = self.get_historical_neighbors(src, timestamp)
            exclude_set.add(dst)

        # Sample with rejection
        negatives = []
        attempts = 0
        max_attempts = self.num_negatives * 10

        while len(negatives) < self.num_negatives and attempts < max_attempts:
            candidates = self.rng.randint(0, self.num_nodes, size=self.num_negatives)
            for c in candidates:
                if c not in exclude_set:
                    negatives.append(c)
                    if len(negatives) >= self.num_negatives:
                        break
            attempts += 1

        # If we can't find enough negatives, fall back to random
        if len(negatives) < self.num_negatives:
            while len(negatives) < self.num_negatives:
                c = self.rng.randint(0, self.num_nodes)
                if c != dst:
                    negatives.append(c)

        return np.array(negatives[:self.num_negatives])

    def sample_batch(
        self,
        src_batch: np.ndarray,
        dst_batch: np.ndarray,
        timestamp_batch: np.ndarray
    ) -> np.ndarray:
        """
        Sample negatives for a batch of edges.

        Args:
            src_batch: Source nodes [batch_size]
            dst_batch: Destination nodes [batch_size]
            timestamp_batch: Timestamps [batch_size]

        Returns:
            Negative samples [batch_size, num_negatives]
        """
        batch_size = len(src_batch)
        negatives = np.zeros((batch_size, self.num_negatives), dtype=np.int64)

        for i in range(batch_size):
            negatives[i] = self.sample(
                int(src_batch[i]),
                int(dst_batch[i]),
                float(timestamp_batch[i])
            )

        return negatives


class EdgeBankNegativeSampler:
    """
    EdgeBank-based negative sampler (from TGB benchmark).
    Uses historically observed edges as harder negatives.
    """

    def __init__(
        self,
        src_nodes: np.ndarray,
        dst_nodes: np.ndarray,
        timestamps: np.ndarray,
        num_nodes: int,
        num_negatives: int = 100,
        memory_mode: str = 'unlimited',
        time_window: Optional[float] = None,
        seed: int = 42
    ):
        """
        Args:
            memory_mode: 'unlimited' (all edges) or 'time_window'
            time_window: Window size for time_window mode
        """
        self.num_nodes = num_nodes
        self.num_negatives = num_negatives
        self.memory_mode = memory_mode
        self.time_window = time_window
        self.rng = np.random.RandomState(seed)

        # Build edge bank
        self.edge_bank = set()
        self.time_indexed_edges = []

        for s, d, t in zip(src_nodes, dst_nodes, timestamps):
            edge = (int(s), int(d))
            self.edge_bank.add(edge)
            self.time_indexed_edges.append((t, edge))

        # Sort by time
        self.time_indexed_edges.sort(key=lambda x: x[0])
        self.all_edges = list(self.edge_bank)

    def sample(
        self,
        src: int,
        dst: int,
        timestamp: float
    ) -> np.ndarray:
        """Sample negatives using EdgeBank strategy."""
        # Get candidate edges from bank
        if self.memory_mode == 'time_window' and self.time_window is not None:
            # Only use edges within time window
            min_time = timestamp - self.time_window
            candidates = [
                e for t, e in self.time_indexed_edges
                if min_time <= t < timestamp
            ]
        else:
            candidates = self.all_edges

        # Sample hard negatives (destinations of existing edges from src)
        hard_negatives = []
        for s, d in candidates:
            if s == src and d != dst:
                hard_negatives.append(d)

        # Fill with random negatives if needed
        negatives = list(set(hard_negatives))[:self.num_negatives]

        while len(negatives) < self.num_negatives:
            c = self.rng.randint(0, self.num_nodes)
            if c != dst and c not in negatives:
                negatives.append(c)

        return np.array(negatives[:self.num_negatives])


def create_negative_sampler(
    src_nodes: np.ndarray,
    dst_nodes: np.ndarray,
    timestamps: np.ndarray,
    num_nodes: int,
    strategy: str = 'temporal',
    num_negatives: int = 100,
    seed: int = 42,
    **kwargs
):
    """
    Factory function to create appropriate negative sampler.

    Args:
        strategy: 'random', 'temporal', or 'edgebank'
    """
    if strategy == 'random':
        return NegativeSampler(
            num_nodes=num_nodes,
            num_negatives=num_negatives,
            seed=seed
        )
    elif strategy == 'temporal':
        return TemporalNegativeSampler(
            src_nodes=src_nodes,
            dst_nodes=dst_nodes,
            timestamps=timestamps,
            num_nodes=num_nodes,
            num_negatives=num_negatives,
            strategy='historical',
            seed=seed
        )
    elif strategy == 'edgebank':
        return EdgeBankNegativeSampler(
            src_nodes=src_nodes,
            dst_nodes=dst_nodes,
            timestamps=timestamps,
            num_nodes=num_nodes,
            num_negatives=num_negatives,
            seed=seed,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown negative sampling strategy: {strategy}")


if __name__ == '__main__':
    print("Testing negative samplers...")

    # Create test data
    num_nodes = 1000
    num_edges = 5000

    rng = np.random.RandomState(42)
    src = rng.randint(0, num_nodes, num_edges)
    dst = rng.randint(0, num_nodes, num_edges)
    timestamps = np.sort(rng.rand(num_edges) * 1000)

    # Test basic sampler
    sampler = NegativeSampler(num_nodes, num_negatives=10)
    negs = sampler.sample(0, 1)
    print(f"Basic sampler: {negs.shape}")

    # Test temporal sampler
    temp_sampler = TemporalNegativeSampler(
        src, dst, timestamps, num_nodes, num_negatives=10
    )
    negs = temp_sampler.sample(0, 1, 500.0)
    print(f"Temporal sampler: {negs.shape}")

    # Test batch sampling
    batch_negs = temp_sampler.sample_batch(
        src[:32], dst[:32], timestamps[:32]
    )
    print(f"Batch sampling: {batch_negs.shape}")

    print("All tests passed!")
