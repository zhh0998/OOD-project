"""
数据加载工具 - 兼容旧接口的薄封装

统一调用 src/datasets/ood_datasets.py 作为唯一真源，
修复 train_labels 全 0 的 Bug（现返回真实类别 ID）。

Author: RW3 OOD Detection Project
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List

# 仓库根目录 / data
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# 统一使用 src/datasets/ood_datasets.py 作为"唯一真源"
from src.datasets.ood_datasets import (
    download_clinc150 as _download_clinc150,
    download_banking77_oos as _download_banking77_oos,
    download_rostd as _download_rostd,
    load_clinc150 as _load_clinc150,
    load_banking77_oos as _load_banking77_oos,
    load_rostd as _load_rostd,
)


def download_clinc150(data_dir: Optional[Path] = None) -> Path:
    if data_dir is None:
        data_dir = DATA_DIR / "clinc150"
    return _download_clinc150(data_dir)


def download_banking77_oos(data_dir: Optional[Path] = None) -> Path:
    if data_dir is None:
        data_dir = DATA_DIR / "banking77_oos"
    return _download_banking77_oos(data_dir)


def download_rostd(data_dir: Optional[Path] = None) -> Path:
    if data_dir is None:
        data_dir = DATA_DIR / "rostd"
    return _download_rostd(data_dir)


def load_clinc150(data_dir: Optional[Path] = None):
    """
    加载CLINC150数据集

    Returns:
        train_texts, test_texts, test_labels, test_intents, train_labels (真实类别id)
    """
    if data_dir is None:
        data_dir = DATA_DIR / "clinc150"
    return _load_clinc150(data_dir)


def load_banking77_oos(data_dir: Optional[Path] = None, oos_ratio: float = 0.25):
    """
    兼容旧接口：按比例切分 OOS intents。

    Returns:
        train_texts, test_texts, test_labels, test_intents, train_labels (真实类别id)
    """
    if data_dir is None:
        data_dir = DATA_DIR / "banking77_oos"

    total_classes = 77
    n_ood = int(total_classes * oos_ratio)
    n_id = total_classes - n_ood
    # 复现旧逻辑：固定 seed=42
    return _load_banking77_oos(
        data_dir=data_dir,
        n_id_classes=n_id,
        n_ood_classes=n_ood,
        seed=42,
    )


def load_rostd(data_dir: Optional[Path] = None):
    """
    加载ROSTD数据集

    Returns:
        train_texts, test_texts, test_labels, test_intents, train_labels (真实类别id)
    """
    if data_dir is None:
        data_dir = DATA_DIR / "rostd"
    return _load_rostd(data_dir)


def get_dataset(name: str):
    """
    获取指定数据集的加载函数

    Args:
        name: 数据集名称 ('clinc150', 'banking77', 'rostd')

    Returns:
        对应的 load_* 函数
    """
    name = name.lower()
    if name in ["clinc150", "clinc"]:
        return load_clinc150
    if name in ["banking77", "banking77_oos", "banking"]:
        return load_banking77_oos
    if name in ["rostd"]:
        return load_rostd
    raise ValueError(f"Unknown dataset: {name}")
