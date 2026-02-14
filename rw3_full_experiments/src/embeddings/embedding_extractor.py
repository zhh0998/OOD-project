"""
Embedding Extractor Module
Handles embedding extraction and caching for all models.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
from tqdm import tqdm
import torch


class EmbeddingExtractor:
    """
    Unified embedding extractor for sentence transformers.

    Supported models:
    - all-MiniLM-L6-v2 (384 dim, high anisotropy)
    - bge-base-en-v1.5 (768 dim, low anisotropy)
    - e5-large-v2 (1024 dim, low anisotropy)
    - all-mpnet-base-v2 (768 dim, medium anisotropy)
    """

    MODEL_CONFIGS = {
        'minilm': {
            'name': 'all-MiniLM-L6-v2',
            'full_name': 'sentence-transformers/all-MiniLM-L6-v2',
            'dim': 384,
            'prefix': None
        },
        'bge': {
            'name': 'bge-base-en-v1.5',
            'full_name': 'BAAI/bge-base-en-v1.5',
            'dim': 768,
            'prefix': None  # BGE doesn't require prefix for short texts
        },
        'e5': {
            'name': 'e5-large-v2',
            'full_name': 'intfloat/e5-large-v2',
            'dim': 1024,
            'prefix': 'query: '  # E5 requires prefix
        },
        'mpnet': {
            'name': 'all-mpnet-base-v2',
            'full_name': 'sentence-transformers/all-mpnet-base-v2',
            'dim': 768,
            'prefix': None
        }
    }

    def __init__(self, cache_dir: str = "cache/embeddings", device: str = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.models = {}

    def _load_model(self, model_key: str):
        """Load a sentence transformer model."""
        if model_key not in self.models:
            from sentence_transformers import SentenceTransformer

            config = self.MODEL_CONFIGS[model_key]
            print(f"Loading model: {config['full_name']}...")
            self.models[model_key] = SentenceTransformer(config['full_name'], device=self.device)

        return self.models[model_key]

    def _get_cache_path(self, dataset: str, model: str, split: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / dataset / model / f"{split}.npy"

    def extract(self, texts: List[str], model_key: str, batch_size: int = 32,
                show_progress: bool = True) -> np.ndarray:
        """Extract embeddings for a list of texts."""
        config = self.MODEL_CONFIGS[model_key]
        model = self._load_model(model_key)

        # Add prefix if required
        if config['prefix']:
            texts = [config['prefix'] + t for t in texts]

        # Extract embeddings
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalize
        )

        return embeddings

    def extract_dataset(self, data: Dict, model_key: str, dataset_name: str,
                        use_cache: bool = True, batch_size: int = 32) -> Dict[str, np.ndarray]:
        """
        Extract embeddings for all splits of a dataset.

        Args:
            data: dict with keys train_id, cal_id, test_id, test_ood
            model_key: model identifier (minilm, bge, e5, mpnet)
            dataset_name: name for caching
            use_cache: whether to use/save cache

        Returns:
            dict with same keys, values are np.ndarray embeddings
        """
        embeddings = {}

        for split in ['train_id', 'cal_id', 'test_id', 'test_ood']:
            if split not in data:
                continue

            cache_path = self._get_cache_path(dataset_name, model_key, split)

            if use_cache and cache_path.exists():
                print(f"Loading cached {split} embeddings from {cache_path}")
                embeddings[split] = np.load(cache_path)
            else:
                texts = data[split]['texts']
                print(f"Extracting {split} embeddings ({len(texts)} samples)...")
                emb = self.extract(texts, model_key, batch_size)
                embeddings[split] = emb

                if use_cache:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(cache_path, emb)
                    print(f"Cached to {cache_path}")

        return embeddings

    def extract_all_models(self, data: Dict, dataset_name: str,
                           model_keys: List[str] = None,
                           use_cache: bool = True) -> Dict[str, Dict[str, np.ndarray]]:
        """Extract embeddings for all models."""
        if model_keys is None:
            model_keys = list(self.MODEL_CONFIGS.keys())

        all_embeddings = {}
        for model_key in model_keys:
            print(f"\n{'='*50}")
            print(f"Processing model: {model_key}")
            print(f"{'='*50}")
            all_embeddings[model_key] = self.extract_dataset(
                data, model_key, dataset_name, use_cache
            )

        return all_embeddings

    def get_model_dim(self, model_key: str) -> int:
        """Get embedding dimension for a model."""
        return self.MODEL_CONFIGS[model_key]['dim']
