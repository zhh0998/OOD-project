#!/usr/bin/env python3
"""
统一数据加载接口

支持两种实验协议:
1. Native OOS: 使用数据集自带的OOS样本 (CLINC150的oos_test)
2. Class-holdout: 随机选择部分类别作为OOD (TEXTOIR标准)

使用示例:
    # CLINC150 Native OOS
    data = load_ood_dataset("clinc150", "native_oos")

    # CLINC150 Class-holdout with 25% KIR
    data = load_ood_dataset("clinc150", "class_holdout", known_intent_ratio=0.25)

    # Banking77 with 50/27 split (FLatS标准)
    data = load_ood_dataset("banking77", "class_holdout", n_id_classes=50, n_ood_classes=27)
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any

from .ood_datasets import load_clinc150, load_banking77_oos, load_rostd
from .clinc_class_holdout import load_clinc150_class_holdout


def load_ood_dataset(
    dataset: str,
    protocol: str,
    known_intent_ratio: Optional[float] = None,
    n_id_classes: Optional[int] = None,
    n_ood_classes: Optional[int] = None,
    seed: int = 42,
    data_root: str = "data",
    **kwargs
) -> Dict[str, Any]:
    """
    统一数据加载接口

    Args:
        dataset: "clinc150" | "banking77" | "rostd"
        protocol: "native_oos" | "class_holdout"
        known_intent_ratio: KIR (仅class_holdout需要，用于CLINC150)
        n_id_classes: ID类别数 (用于Banking77，默认50)
        n_ood_classes: OOD类别数 (用于Banking77，默认27)
        seed: 随机种子
        data_root: 数据根目录
        **kwargs: 其他参数

    Returns:
        {
            'train_texts': List[str],
            'train_labels': List[int],  # intent IDs
            'test_texts': List[str],
            'test_labels': List[int],  # binary: 0=ID, 1=OOD
            'protocol': str,
            'dataset': str,
            'metadata': Dict
        }
    """
    data_root = Path(data_root)
    dataset = dataset.lower()

    if dataset == "clinc150":
        if protocol == "native_oos":
            train_texts, test_texts, test_labels, test_intents, train_labels = \
                load_clinc150(data_root / "clinc150")

            metadata = {
                'protocol': 'native_oos',
                'dataset': 'clinc150',
                'n_intents': 150,
                'train_size': len(train_texts),
                'test_id_size': test_labels.count(0),
                'test_ood_size': test_labels.count(1)
            }

            return {
                'train_texts': train_texts,
                'train_labels': train_labels,
                'test_texts': test_texts,
                'test_labels': test_labels,
                'protocol': protocol,
                'dataset': dataset,
                'metadata': metadata
            }

        elif protocol == "class_holdout":
            if known_intent_ratio is None:
                raise ValueError("class_holdout protocol requires known_intent_ratio parameter")

            train_texts, train_labels, test_texts, test_labels, metadata = \
                load_clinc150_class_holdout(
                    data_root / "clinc150",
                    known_intent_ratio=known_intent_ratio,
                    seed=seed,
                    **kwargs
                )

            return {
                'train_texts': train_texts,
                'train_labels': train_labels,
                'test_texts': test_texts,
                'test_labels': test_labels,
                'protocol': protocol,
                'dataset': dataset,
                'metadata': metadata
            }

        else:
            raise ValueError(f"Unknown protocol for clinc150: {protocol}")

    elif dataset == "banking77":
        # Banking77 使用 class_holdout 协议
        # 默认使用 FLatS 标准: 50 ID / 27 OOD

        if n_id_classes is None:
            n_id_classes = 50
        if n_ood_classes is None:
            n_ood_classes = 27

        train_texts, test_texts, test_labels, test_intents, train_labels = \
            load_banking77_oos(
                data_root / "banking77_oos",
                n_id_classes=n_id_classes,
                n_ood_classes=n_ood_classes,
                seed=seed
            )

        metadata = {
            'protocol': 'class_holdout',
            'dataset': 'banking77',
            'n_id_classes': n_id_classes,
            'n_ood_classes': n_ood_classes,
            'seed': seed,
            'train_size': len(train_texts),
            'test_id_size': test_labels.count(0),
            'test_ood_size': test_labels.count(1)
        }

        return {
            'train_texts': train_texts,
            'train_labels': train_labels,
            'test_texts': test_texts,
            'test_labels': test_labels,
            'protocol': protocol if protocol else 'class_holdout',
            'dataset': dataset,
            'metadata': metadata
        }

    elif dataset == "rostd":
        if protocol != "native_oos":
            print(f"[Warning] ROSTD only supports native_oos protocol, ignoring {protocol}")

        train_texts, test_texts, test_labels, test_intents, train_labels = \
            load_rostd(data_root / "rostd")

        metadata = {
            'protocol': 'native_oos',
            'dataset': 'rostd',
            'train_size': len(train_texts),
            'test_id_size': test_labels.count(0),
            'test_ood_size': test_labels.count(1)
        }

        return {
            'train_texts': train_texts,
            'train_labels': train_labels,
            'test_texts': test_texts,
            'test_labels': test_labels,
            'protocol': 'native_oos',
            'dataset': dataset,
            'metadata': metadata
        }

    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_experiment_configs() -> Dict:
    """
    获取所有实验配置

    Returns:
        字典，包含所有实验配置
    """
    configs = {
        # CLINC150 Native OOS (Far-OOD场景)
        'clinc150_native_oos': {
            'dataset': 'clinc150',
            'protocol': 'native_oos',
        },

        # CLINC150 Class-holdout (Near-OOD场景)
        'clinc150_kir25': {
            'dataset': 'clinc150',
            'protocol': 'class_holdout',
            'known_intent_ratio': 0.25,
        },
        'clinc150_kir50': {
            'dataset': 'clinc150',
            'protocol': 'class_holdout',
            'known_intent_ratio': 0.50,
        },
        'clinc150_kir75': {
            'dataset': 'clinc150',
            'protocol': 'class_holdout',
            'known_intent_ratio': 0.75,
        },

        # Banking77 (Near-OOD场景) - FLatS标准划分
        'banking77_50_27': {
            'dataset': 'banking77',
            'protocol': 'class_holdout',
            'n_id_classes': 50,
            'n_ood_classes': 27,
        },

        # ROSTD (Far-OOD场景)
        'rostd_native_oos': {
            'dataset': 'rostd',
            'protocol': 'native_oos',
        },
    }

    return configs


if __name__ == "__main__":
    print("="*80)
    print("测试统一数据加载接口")
    print("="*80)

    # 测试所有配置
    configs = get_experiment_configs()

    for name, config in configs.items():
        print(f"\n--- {name} ---")
        try:
            data = load_ood_dataset(**config, data_root="data")
            print(f"  Train: {len(data['train_texts'])} samples")
            print(f"  Test: {len(data['test_texts'])} samples "
                  f"(ID: {data['metadata']['test_id_size']}, "
                  f"OOD: {data['metadata']['test_ood_size']})")
            print(f"  ✅ 加载成功")
        except Exception as e:
            print(f"  ❌ 错误: {e}")
