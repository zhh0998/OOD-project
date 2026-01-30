#!/usr/bin/env python3
"""
CLINC150 Class-holdout协议实现
符合TEXTOIR标准（25%/50%/75% KIR）

这是一种不同于Native OOS的实验协议：
- Native OOS: 使用数据集自带的OOS样本（1000条"oos"标签的测试样本）
- Class-holdout: 随机选择部分intent类别作为OOD，其余作为ID

Class-holdout更适合评估Near-OOD检测，因为OOD样本来自同一领域（对话意图）
"""

import json
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict

from .ood_datasets import download_clinc150


def load_clinc150_class_holdout(
    data_dir: Optional[Path] = None,
    known_intent_ratio: float = 0.25,
    seed: int = 42,
    include_native_oos: bool = False
) -> Tuple[List[str], List[int], List[str], List[int], Dict]:
    """
    CLINC150 Class-holdout协议

    Args:
        data_dir: 数据目录
        known_intent_ratio: KIR - Known Intent Ratio (0.25, 0.50, 0.75)
        seed: 随机种子
        include_native_oos: 是否将原生OOS样本加入测试集的OOD部分

    Returns:
        train_texts: 训练文本列表
        train_labels: 训练标签（known intent ID: 0 到 n_known-1）
        test_texts: 测试文本列表
        test_labels: 测试标签（0=ID, 1=OOD）
        metadata: 元数据字典
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "data" / "clinc150"

    data_file = data_dir / "data_full.json"

    if not data_file.exists():
        download_clinc150(data_dir)

    with open(data_file, 'r') as f:
        data = json.load(f)

    # 1. 获取所有ID意图（排除原生OOS）
    all_intents = sorted(set(
        intent for _, intent in data['train'] + data['val']
        if intent != 'oos'
    ))

    assert len(all_intents) == 150, f"CLINC150应有150个意图类别，实际{len(all_intents)}"

    # 2. 计算known和unknown意图数量
    n_known = int(len(all_intents) * known_intent_ratio)
    n_unknown = len(all_intents) - n_known

    # 3. 随机选择known intents
    np.random.seed(seed)
    known_intents = set(np.random.choice(all_intents, n_known, replace=False))
    unknown_intents = set(all_intents) - known_intents

    print(f"\n[CLINC150 Class-holdout] KIR={known_intent_ratio*100:.0f}%")
    print(f"  Known intents: {len(known_intents)}")
    print(f"  Unknown intents: {len(unknown_intents)}")
    print(f"  Random seed: {seed}")

    # 4. 建立intent到ID的映射（仅known intents）
    intent2id = {intent: idx for idx, intent in enumerate(sorted(known_intents))}

    # 5. 构建训练集（仅known intents的train+val）
    train_texts = []
    train_labels = []
    for text, intent in data['train'] + data['val']:
        if intent in known_intents:
            train_texts.append(text)
            train_labels.append(intent2id[intent])

    # 6. 构建测试集
    test_texts = []
    test_labels = []  # Binary: 0=ID, 1=OOD

    # ID测试样本（known intents的test部分）
    n_test_id = 0
    for text, intent in data['test']:
        if intent in known_intents:
            test_texts.append(text)
            test_labels.append(0)
            n_test_id += 1

    # OOD测试样本（unknown intents的test部分）
    n_test_ood_unknown = 0
    for text, intent in data['test']:
        if intent in unknown_intents:
            test_texts.append(text)
            test_labels.append(1)
            n_test_ood_unknown += 1

    # 可选：加入原生OOS样本
    n_test_ood_native = 0
    if include_native_oos:
        for text, intent in data['oos_test']:
            test_texts.append(text)
            test_labels.append(1)
            n_test_ood_native += 1

    # 元数据
    metadata = {
        'protocol': 'class_holdout',
        'dataset': 'clinc150',
        'kir': known_intent_ratio,
        'n_known_intents': len(known_intents),
        'n_unknown_intents': len(unknown_intents),
        'seed': seed,
        'include_native_oos': include_native_oos,
        'train_size': len(train_texts),
        'test_id_size': n_test_id,
        'test_ood_size': n_test_ood_unknown + n_test_ood_native,
        'test_ood_from_unknown': n_test_ood_unknown,
        'test_ood_from_native': n_test_ood_native,
        'known_intents': sorted(known_intents),
        'unknown_intents': sorted(unknown_intents)
    }

    print(f"  Training samples: {metadata['train_size']}")
    print(f"  Test ID samples: {metadata['test_id_size']}")
    print(f"  Test OOD samples: {metadata['test_ood_size']}")
    if include_native_oos:
        print(f"    - From unknown intents: {n_test_ood_unknown}")
        print(f"    - From native OOS: {n_test_ood_native}")

    return train_texts, train_labels, test_texts, test_labels, metadata


def load_clinc150_for_all_kirs(
    data_dir: Optional[Path] = None,
    kirs: List[float] = [0.25, 0.50, 0.75],
    seed: int = 42,
    include_native_oos: bool = False
) -> Dict:
    """
    加载所有KIR设置的CLINC150数据

    Returns:
        字典，key为KIR值，value为对应的数据
    """
    all_data = {}
    for kir in kirs:
        train_texts, train_labels, test_texts, test_labels, metadata = \
            load_clinc150_class_holdout(
                data_dir=data_dir,
                known_intent_ratio=kir,
                seed=seed,
                include_native_oos=include_native_oos
            )
        all_data[kir] = {
            'train_texts': train_texts,
            'train_labels': train_labels,
            'test_texts': test_texts,
            'test_labels': test_labels,
            'metadata': metadata
        }
    return all_data


if __name__ == "__main__":
    # 测试代码
    print("="*80)
    print("测试CLINC150 Class-holdout协议")
    print("="*80)

    for kir in [0.25, 0.50, 0.75]:
        train_texts, train_labels, test_texts, test_labels, metadata = \
            load_clinc150_class_holdout(known_intent_ratio=kir, seed=42)

        print(f"\nKIR={kir*100:.0f}% 验证:")
        print(f"  训练集唯一标签数: {len(set(train_labels))}")
        print(f"  应该等于 known intents: {metadata['n_known_intents']}")
        assert len(set(train_labels)) == metadata['n_known_intents'], "标签数不匹配！"
        print(f"  ✅ 验证通过")
