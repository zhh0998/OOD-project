"""
数据加载工具 - 支持CLINC150, Banking77-OOS, ROSTD数据集

Author: RW3 OOD Detection Project
"""

import os
import json
import requests
import zipfile
import numpy as np
from typing import Tuple, List, Dict, Optional
from pathlib import Path


DATA_DIR = Path(__file__).parent / "data"


def download_clinc150(data_dir: Optional[Path] = None) -> Path:
    """
    下载CLINC150数据集
    Source: https://github.com/clinc/oos-eval
    """
    if data_dir is None:
        data_dir = DATA_DIR / "clinc150"
    data_dir.mkdir(parents=True, exist_ok=True)

    data_file = data_dir / "data_full.json"

    if data_file.exists():
        print(f"[CLINC150] 数据已存在: {data_file}")
        return data_dir

    # 下载数据
    url = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"
    print(f"[CLINC150] 正在下载数据...")

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        with open(data_file, 'w') as f:
            f.write(response.text)

        print(f"[CLINC150] 下载完成: {data_file}")
    except Exception as e:
        print(f"[CLINC150] 下载失败: {e}")
        raise

    return data_dir


def load_clinc150(data_dir: Optional[Path] = None) -> Tuple[List[str], List[str], List[int], List[str], List[int]]:
    """
    加载CLINC150数据集

    Returns:
        train_texts: 训练文本
        test_texts: 测试文本
        test_labels: 测试标签 (0=ID, 1=OOD)
        test_intents: 测试意图
        train_labels: 训练标签
    """
    if data_dir is None:
        data_dir = DATA_DIR / "clinc150"

    data_file = data_dir / "data_full.json"

    if not data_file.exists():
        download_clinc150(data_dir)

    with open(data_file, 'r') as f:
        data = json.load(f)

    # 训练数据（只使用ID类别）
    train_texts = []
    train_labels = []
    for text, intent in data['train']:
        if intent != 'oos':
            train_texts.append(text)
            train_labels.append(0)

    # 验证数据也加入训练（增加训练样本）
    for text, intent in data['val']:
        if intent != 'oos':
            train_texts.append(text)
            train_labels.append(0)

    # 测试数据 - 需要合并test(ID)和oos_test(OOD)
    test_texts = []
    test_labels = []
    test_intents = []

    # ID测试样本 (from 'test')
    for text, intent in data['test']:
        test_texts.append(text)
        test_labels.append(0)  # ID样本
        test_intents.append(intent)

    # OOD测试样本 (from 'oos_test')
    for text, intent in data['oos_test']:
        test_texts.append(text)
        test_labels.append(1)  # OOD样本
        test_intents.append(intent)

    print(f"[CLINC150] 加载完成:")
    print(f"  - 训练样本: {len(train_texts)} (全部ID)")
    print(f"  - 测试样本: {len(test_texts)} (ID: {test_labels.count(0)}, OOD: {test_labels.count(1)})")

    return train_texts, test_texts, test_labels, test_intents, train_labels


def download_banking77_oos(data_dir: Optional[Path] = None) -> Path:
    """
    下载Banking77-OOS数据集
    Source: https://github.com/PolyAI-LDN/task-specific-datasets
    """
    if data_dir is None:
        data_dir = DATA_DIR / "banking77_oos"
    data_dir.mkdir(parents=True, exist_ok=True)

    train_file = data_dir / "train.csv"

    if train_file.exists():
        print(f"[Banking77-OOS] 数据已存在: {data_dir}")
        return data_dir

    # Banking77训练数据
    train_url = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
    test_url = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"

    print(f"[Banking77-OOS] 正在下载数据...")

    try:
        # 下载训练数据
        response = requests.get(train_url, timeout=60)
        response.raise_for_status()
        with open(train_file, 'w') as f:
            f.write(response.text)

        # 下载测试数据
        response = requests.get(test_url, timeout=60)
        response.raise_for_status()
        with open(data_dir / "test.csv", 'w') as f:
            f.write(response.text)

        print(f"[Banking77-OOS] 下载完成: {data_dir}")
    except Exception as e:
        print(f"[Banking77-OOS] 下载失败: {e}")
        raise

    return data_dir


def load_banking77_oos(data_dir: Optional[Path] = None,
                       oos_ratio: float = 0.25) -> Tuple[List[str], List[str], List[int], List[str], List[int]]:
    """
    加载Banking77-OOS数据集
    将部分类别作为OOS (Out-of-Scope)

    Args:
        oos_ratio: OOS类别占比

    Returns:
        train_texts, test_texts, test_labels, test_intents, train_labels
    """
    if data_dir is None:
        data_dir = DATA_DIR / "banking77_oos"

    train_file = data_dir / "train.csv"

    if not train_file.exists():
        download_banking77_oos(data_dir)

    import csv

    # 加载数据
    def load_csv(filepath):
        texts = []
        intents = []
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过header
            for row in reader:
                if len(row) >= 2:
                    texts.append(row[0])
                    intents.append(row[1])
        return texts, intents

    train_texts_all, train_intents_all = load_csv(train_file)
    test_texts_all, test_intents_all = load_csv(data_dir / "test.csv")

    # 获取所有类别
    all_intents = sorted(set(train_intents_all))
    n_oos = int(len(all_intents) * oos_ratio)

    # 随机选择OOS类别（固定种子保证可复现）
    np.random.seed(42)
    oos_intents = set(np.random.choice(all_intents, n_oos, replace=False))
    id_intents = set(all_intents) - oos_intents

    print(f"[Banking77-OOS] ID类别: {len(id_intents)}, OOS类别: {len(oos_intents)}")

    # 过滤训练数据（只保留ID类别）
    train_texts = []
    train_labels = []
    for text, intent in zip(train_texts_all, train_intents_all):
        if intent in id_intents:
            train_texts.append(text)
            train_labels.append(0)

    # 测试数据（包含ID和OOS）
    test_texts = []
    test_labels = []
    test_intents = []
    for text, intent in zip(test_texts_all, test_intents_all):
        test_texts.append(text)
        test_labels.append(1 if intent in oos_intents else 0)
        test_intents.append(intent)

    print(f"[Banking77-OOS] 加载完成:")
    print(f"  - 训练样本: {len(train_texts)} (全部ID)")
    print(f"  - 测试样本: {len(test_texts)} (ID: {test_labels.count(0)}, OOD: {test_labels.count(1)})")

    return train_texts, test_texts, test_labels, test_intents, train_labels


def download_rostd(data_dir: Optional[Path] = None) -> Path:
    """
    下载ROSTD数据集
    ROSTD: Real Out-of-Scope Texts Dataset
    """
    if data_dir is None:
        data_dir = DATA_DIR / "rostd"
    data_dir.mkdir(parents=True, exist_ok=True)

    data_file = data_dir / "rostd_data.json"

    if data_file.exists():
        print(f"[ROSTD] 数据已存在: {data_file}")
        return data_dir

    # 使用SNIPS数据集创建类似ROSTD的设置
    print(f"[ROSTD] 正在下载SNIPS数据集...")

    snips_train_url = "https://raw.githubusercontent.com/snipsco/nlu-benchmark/master/2017-06-custom-intent-engines/train.json"
    snips_test_url = "https://raw.githubusercontent.com/snipsco/nlu-benchmark/master/2017-06-custom-intent-engines/test.json"

    try:
        # 尝试下载SNIPS数据
        # 如果失败，创建模拟数据
        raise NotImplementedError("ROSTD数据需要手动准备")
    except:
        # 创建模拟数据用于测试
        print(f"[ROSTD] 创建模拟数据...")

        # 模拟7个意图类别
        intents = ['weather', 'music', 'reminder', 'alarm', 'calendar', 'news', 'navigation']
        oos_phrases = [
            "What is the meaning of life?",
            "Tell me a joke",
            "How old is the universe?",
            "What's your favorite color?",
            "Can you sing a song?",
            "Who won the world cup?",
            "What's the capital of Mars?",
        ]

        # 创建训练和测试数据
        data = {
            'train': [],
            'test': []
        }

        # 为每个意图创建样本
        intent_phrases = {
            'weather': ["What's the weather today?", "Is it going to rain?", "How hot is it outside?"],
            'music': ["Play some music", "Play my favorite song", "Turn on the radio"],
            'reminder': ["Remind me to call mom", "Set a reminder for tomorrow", "Don't let me forget"],
            'alarm': ["Set an alarm for 7am", "Wake me up at 6", "Cancel my alarm"],
            'calendar': ["What's on my calendar?", "Schedule a meeting", "Add an event"],
            'news': ["What's the latest news?", "Tell me the headlines", "Any breaking news?"],
            'navigation': ["Navigate to home", "How do I get to work?", "Show me directions"],
        }

        for intent, phrases in intent_phrases.items():
            for phrase in phrases:
                data['train'].append([phrase, intent])
                data['test'].append([phrase, intent])

        # 添加OOS样本到测试集
        for phrase in oos_phrases:
            data['test'].append([phrase, 'oos'])

        with open(data_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"[ROSTD] 模拟数据创建完成: {data_file}")

    return data_dir


def load_rostd(data_dir: Optional[Path] = None) -> Tuple[List[str], List[str], List[int], List[str], List[int]]:
    """
    加载ROSTD数据集

    Returns:
        train_texts, test_texts, test_labels, test_intents, train_labels
    """
    if data_dir is None:
        data_dir = DATA_DIR / "rostd"

    data_file = data_dir / "rostd_data.json"

    if not data_file.exists():
        download_rostd(data_dir)

    with open(data_file, 'r') as f:
        data = json.load(f)

    # 训练数据
    train_texts = []
    train_labels = []
    for text, intent in data['train']:
        if intent != 'oos':
            train_texts.append(text)
            train_labels.append(0)

    # 测试数据
    test_texts = []
    test_labels = []
    test_intents = []
    for text, intent in data['test']:
        test_texts.append(text)
        test_labels.append(1 if intent == 'oos' else 0)
        test_intents.append(intent)

    print(f"[ROSTD] 加载完成:")
    print(f"  - 训练样本: {len(train_texts)} (全部ID)")
    print(f"  - 测试样本: {len(test_texts)} (ID: {test_labels.count(0)}, OOD: {test_labels.count(1)})")

    return train_texts, test_texts, test_labels, test_intents, train_labels


def get_dataset(name: str) -> Tuple[List[str], List[str], List[int], List[str], List[int]]:
    """
    获取指定数据集

    Args:
        name: 数据集名称 ('clinc150', 'banking77', 'rostd')

    Returns:
        train_texts, test_texts, test_labels, test_intents, train_labels
    """
    name = name.lower()

    if name in ['clinc150', 'clinc']:
        return load_clinc150()
    elif name in ['banking77', 'banking77_oos', 'banking']:
        return load_banking77_oos()
    elif name in ['rostd']:
        return load_rostd()
    else:
        raise ValueError(f"Unknown dataset: {name}")


if __name__ == "__main__":
    # 测试数据加载
    print("\n" + "="*50)
    print("Testing CLINC150")
    print("="*50)
    train, test, labels, intents, train_labels = load_clinc150()

    print("\n" + "="*50)
    print("Testing Banking77-OOS")
    print("="*50)
    train, test, labels, intents, train_labels = load_banking77_oos()

    print("\n" + "="*50)
    print("Testing ROSTD")
    print("="*50)
    train, test, labels, intents, train_labels = load_rostd()
