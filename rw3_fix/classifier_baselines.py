"""
分类器基线OOD检测方法

实现以下经典方法：
1. MSP (Maximum Softmax Probability) - ICLR 2017
2. Energy-based OOD Detection - NeurIPS 2020
3. MaxLogit - arXiv 2022

Author: RW3 OOD Detection Project
"""

import numpy as np
from typing import Tuple, Optional
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available")


class SimpleClassifier(nn.Module):
    """
    简单的句子分类器

    用于基于分类器的OOD检测方法（MSP, Energy, MaxLogit）
    """

    def __init__(self, input_dim: int = 384, hidden_dim: int = 256,
                 num_classes: int = 150, dropout: float = 0.1):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

    def get_features(self, x):
        """获取倒数第二层特征"""
        for i, layer in enumerate(self.classifier):
            x = layer(x)
            if i == len(self.classifier) - 2:  # 倒数第二层
                return x
        return x


def train_classifier(train_embeddings: np.ndarray, train_labels: np.ndarray,
                     num_classes: int, epochs: int = 10,
                     batch_size: int = 64, lr: float = 1e-3,
                     verbose: bool = True) -> nn.Module:
    """
    训练分类器

    Args:
        train_embeddings: 训练集embeddings
        train_labels: 训练集标签（整数）
        num_classes: 类别数
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        verbose: 是否打印训练信息

    Returns:
        训练好的分类器
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for classifier training")

    # 处理标签
    if isinstance(train_labels[0], str):
        unique_labels = sorted(set(train_labels))
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        train_labels = np.array([label_to_idx[l] for l in train_labels])
    else:
        train_labels = np.array(train_labels)

    # 创建模型
    input_dim = train_embeddings.shape[1]
    model = SimpleClassifier(input_dim=input_dim, num_classes=num_classes)

    # 创建DataLoader
    dataset = TensorDataset(
        torch.FloatTensor(train_embeddings),
        torch.LongTensor(train_labels)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 训练
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for x, y in loader:
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)

        acc = correct / total
        avg_loss = total_loss / len(loader)

        if verbose and (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={acc:.4f}")

    model.eval()
    return model


class MSPDetector:
    """
    Maximum Softmax Probability (MSP)

    参考: A Baseline for Detecting Misclassified and Out-of-Distribution
          Examples in Neural Networks (ICLR 2017)

    原理: ID样本的最大softmax概率通常较高，OOD样本较低
    """

    def __init__(self, classifier: nn.Module, verbose: bool = True):
        self.classifier = classifier
        self.verbose = verbose

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """
        计算OOD分数

        Args:
            embeddings: 测试集embeddings

        Returns:
            OOD分数（1 - max_prob，越高越可能是OOD）
        """
        self.classifier.eval()
        with torch.no_grad():
            x = torch.FloatTensor(embeddings)
            logits = self.classifier(x)
            probs = F.softmax(logits, dim=1)
            max_probs = probs.max(dim=1)[0]

        # 1 - max_prob: 低置信度 = 高OOD分数
        ood_scores = (1 - max_probs).numpy()

        if self.verbose:
            print(f"[MSP] Scores: mean={ood_scores.mean():.4f}, std={ood_scores.std():.4f}")

        return ood_scores

    def score_with_fix(self, embeddings: np.ndarray,
                       labels: np.ndarray) -> Tuple[np.ndarray, float]:
        """计算分数并自动修复方向"""
        from sklearn.metrics import roc_auc_score

        scores = self.score(embeddings)

        auroc_orig = roc_auc_score(labels, scores)
        auroc_inv = roc_auc_score(labels, -scores)

        if auroc_inv > auroc_orig + 0.05:
            if self.verbose:
                print(f"[MSP] 修复分数反转: {auroc_orig:.4f} -> {auroc_inv:.4f}")
            return -scores, auroc_inv
        return scores, auroc_orig


class EnergyDetector:
    """
    Energy-based OOD Detection

    参考: Energy-based Out-of-distribution Detection (NeurIPS 2020)

    原理: E(x) = -T * log(sum(exp(f_i(x)/T)))
          ID样本能量低，OOD样本能量高
    """

    def __init__(self, classifier: nn.Module, temperature: float = 1.0,
                 verbose: bool = True):
        self.classifier = classifier
        self.temperature = temperature
        self.verbose = verbose

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """
        计算OOD分数（能量）

        Args:
            embeddings: 测试集embeddings

        Returns:
            能量分数（越高越可能是OOD）
        """
        self.classifier.eval()
        with torch.no_grad():
            x = torch.FloatTensor(embeddings)
            logits = self.classifier(x)

            # Energy = -T * logsumexp(logits/T)
            energy = -self.temperature * torch.logsumexp(
                logits / self.temperature, dim=1
            )

        # 高能量 = 高OOD分数（取负使其变为高=OOD）
        ood_scores = (-energy).numpy()

        if self.verbose:
            print(f"[Energy] Scores: mean={ood_scores.mean():.4f}, std={ood_scores.std():.4f}")

        return ood_scores

    def score_with_fix(self, embeddings: np.ndarray,
                       labels: np.ndarray) -> Tuple[np.ndarray, float]:
        """计算分数并自动修复方向"""
        from sklearn.metrics import roc_auc_score

        scores = self.score(embeddings)

        auroc_orig = roc_auc_score(labels, scores)
        auroc_inv = roc_auc_score(labels, -scores)

        if auroc_inv > auroc_orig + 0.05:
            if self.verbose:
                print(f"[Energy] 修复分数反转: {auroc_orig:.4f} -> {auroc_inv:.4f}")
            return -scores, auroc_inv
        return scores, auroc_orig


class MaxLogitDetector:
    """
    MaxLogit OOD Detection

    参考: Scaling Out-of-Distribution Detection for Real-World Settings (arXiv 2022)

    原理: 使用最大logit值而非softmax概率
          对于大规模分类器，MaxLogit比MSP更有效
    """

    def __init__(self, classifier: nn.Module, verbose: bool = True):
        self.classifier = classifier
        self.verbose = verbose

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """
        计算OOD分数

        Args:
            embeddings: 测试集embeddings

        Returns:
            OOD分数（-max_logit，越高越可能是OOD）
        """
        self.classifier.eval()
        with torch.no_grad():
            x = torch.FloatTensor(embeddings)
            logits = self.classifier(x)
            max_logits = logits.max(dim=1)[0]

        # -max_logit: 低logit = 高OOD分数
        ood_scores = (-max_logits).numpy()

        if self.verbose:
            print(f"[MaxLogit] Scores: mean={ood_scores.mean():.4f}, std={ood_scores.std():.4f}")

        return ood_scores

    def score_with_fix(self, embeddings: np.ndarray,
                       labels: np.ndarray) -> Tuple[np.ndarray, float]:
        """计算分数并自动修复方向"""
        from sklearn.metrics import roc_auc_score

        scores = self.score(embeddings)

        auroc_orig = roc_auc_score(labels, scores)
        auroc_inv = roc_auc_score(labels, -scores)

        if auroc_inv > auroc_orig + 0.05:
            if self.verbose:
                print(f"[MaxLogit] 修复分数反转: {auroc_orig:.4f} -> {auroc_inv:.4f}")
            return -scores, auroc_inv
        return scores, auroc_orig


class OdinDetector:
    """
    ODIN (Out-of-DIstribution detector for Neural networks)

    参考: Enhancing The Reliability of Out-of-distribution Image Detection
          in Neural Networks (ICLR 2018)

    原理: 使用温度缩放 + 输入扰动来增强ID/OOD区分
    """

    def __init__(self, classifier: nn.Module, temperature: float = 1000.0,
                 epsilon: float = 0.0014, verbose: bool = True):
        self.classifier = classifier
        self.temperature = temperature
        self.epsilon = epsilon
        self.verbose = verbose

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """
        计算ODIN分数

        Args:
            embeddings: 测试集embeddings

        Returns:
            OOD分数
        """
        self.classifier.eval()

        x = torch.FloatTensor(embeddings)
        x.requires_grad = True

        # 前向传播
        logits = self.classifier(x)
        scaled_logits = logits / self.temperature

        # 计算梯度
        probs = F.softmax(scaled_logits, dim=1)
        max_probs, _ = probs.max(dim=1)

        # 对输入求梯度
        loss = max_probs.sum()
        loss.backward()

        # 输入扰动
        gradient = x.grad.data
        gradient_sign = gradient.sign()

        # 扰动后的输入
        x_perturbed = x.data - self.epsilon * gradient_sign

        # 在扰动输入上计算分数
        with torch.no_grad():
            logits_perturbed = self.classifier(x_perturbed)
            scaled_logits_perturbed = logits_perturbed / self.temperature
            probs_perturbed = F.softmax(scaled_logits_perturbed, dim=1)
            max_probs_perturbed = probs_perturbed.max(dim=1)[0]

        ood_scores = (1 - max_probs_perturbed).numpy()

        if self.verbose:
            print(f"[ODIN] Scores: mean={ood_scores.mean():.4f}, std={ood_scores.std():.4f}")

        return ood_scores

    def score_with_fix(self, embeddings: np.ndarray,
                       labels: np.ndarray) -> Tuple[np.ndarray, float]:
        """计算分数并自动修复方向"""
        from sklearn.metrics import roc_auc_score

        scores = self.score(embeddings)

        auroc_orig = roc_auc_score(labels, scores)
        auroc_inv = roc_auc_score(labels, -scores)

        if auroc_inv > auroc_orig + 0.05:
            if self.verbose:
                print(f"[ODIN] 修复分数反转: {auroc_orig:.4f} -> {auroc_inv:.4f}")
            return -scores, auroc_inv
        return scores, auroc_orig


def test_classifier_baselines():
    """测试分类器基线方法"""
    if not TORCH_AVAILABLE:
        print("PyTorch not available, skipping test")
        return

    np.random.seed(42)
    torch.manual_seed(42)

    # 模拟数据
    n_train = 500
    n_test = 200
    dim = 384
    n_classes = 10

    # 训练数据
    train_emb = np.random.randn(n_train, dim).astype('float32')
    train_labels = np.random.randint(0, n_classes, n_train)

    # 测试数据
    test_id = np.random.randn(n_test // 2, dim).astype('float32')
    test_ood = np.random.randn(n_test // 2, dim).astype('float32') + 3
    test_emb = np.vstack([test_id, test_ood])
    test_labels = np.array([0] * (n_test // 2) + [1] * (n_test // 2))

    # 训练分类器
    print("Training classifier...")
    classifier = train_classifier(train_emb, train_labels, n_classes, epochs=5)

    # 测试各方法
    print("\nTesting MSP...")
    msp = MSPDetector(classifier)
    _, auroc_msp = msp.score_with_fix(test_emb, test_labels)
    print(f"MSP AUROC: {auroc_msp:.4f}")

    print("\nTesting Energy...")
    energy = EnergyDetector(classifier)
    _, auroc_energy = energy.score_with_fix(test_emb, test_labels)
    print(f"Energy AUROC: {auroc_energy:.4f}")

    print("\nTesting MaxLogit...")
    maxlogit = MaxLogitDetector(classifier)
    _, auroc_maxlogit = maxlogit.score_with_fix(test_emb, test_labels)
    print(f"MaxLogit AUROC: {auroc_maxlogit:.4f}")

    return {
        'MSP': auroc_msp,
        'Energy': auroc_energy,
        'MaxLogit': auroc_maxlogit
    }


if __name__ == "__main__":
    results = test_classifier_baselines()
    print(f"\n{'='*50}")
    print("Classifier Baselines Test Results:")
    for method, auroc in results.items():
        print(f"  {method}: AUROC = {auroc:.4f}")
    print(f"{'='*50}")
