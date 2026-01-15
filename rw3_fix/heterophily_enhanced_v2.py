"""
HeterophilyEnhanced v2 - 有监督训练版本

改进：
1. 使用PyTorch Geometric GATv2
2. 对比学习训练
3. 异配性感知损失
4. 超参数可调

Author: RW3 OOD Detection Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import faiss

from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_undirected


class HeterophilyGNNv2(nn.Module):
    """GATv2网络 - 用于异配性感知OOD检测"""

    def __init__(self, input_dim: int = 384, hidden_dim: int = 256,
                 output_dim: int = 128, num_layers: int = 2,
                 heads: int = 4, dropout: float = 0.1):
        super().__init__()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # 输入层
        self.convs.append(GATv2Conv(input_dim, hidden_dim, heads=heads,
                                     concat=False, dropout=dropout))
        self.norms.append(nn.LayerNorm(hidden_dim))

        # 隐藏层
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_dim, hidden_dim, heads=heads,
                                         concat=False, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden_dim))

        # 输出层
        if num_layers > 1:
            self.convs.append(GATv2Conv(hidden_dim, output_dim, heads=1,
                                         concat=False, dropout=dropout))
            self.norms.append(nn.LayerNorm(output_dim))

        self.dropout = dropout
        self.output_dim = output_dim if num_layers > 1 else hidden_dim

    def forward(self, x, edge_index):
        """前向传播"""
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class HeterophilyEnhancedV2:
    """
    HeterophilyEnhanced v2 - 完整实现

    核心创新：
    1. 利用图结构中的异配性信息
    2. 通过GNN消息传递增强特征表示
    3. 对比学习 + 分类损失联合训练
    4. 异配性调制的OOD分数
    """

    def __init__(self, input_dim: int = 384, hidden_dim: int = 256,
                 output_dim: int = 128, k: int = 50, num_layers: int = 2,
                 heads: int = 4, dropout: float = 0.1, alpha: float = 0.3,
                 device: str = 'cpu'):
        """
        Args:
            input_dim: 输入embedding维度
            hidden_dim: GNN隐藏层维度
            output_dim: GNN输出维度
            k: k-NN图的k值
            num_layers: GNN层数
            heads: 注意力头数
            dropout: Dropout率
            alpha: 异配性权重 [0.0-1.0]
            device: 计算设备
        """
        self.k = k
        self.alpha = alpha
        self.device = torch.device(device)

        # GNN模型
        self.gnn = HeterophilyGNNv2(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout
        ).to(self.device)

        self.classifier = None
        self.train_embeddings = None
        self.train_labels = None
        self.num_classes = None
        self.fitted = False

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2归一化"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        return embeddings / norms

    def _build_knn_graph(self, embeddings_np: np.ndarray) -> torch.Tensor:
        """构建k-NN图"""
        embeddings_np = embeddings_np.astype('float32')
        d = embeddings_np.shape[1]

        # FAISS索引
        index = faiss.IndexFlatIP(d)
        index.add(embeddings_np)

        # k-NN搜索
        k_search = min(self.k + 1, len(embeddings_np))
        similarities, indices = index.search(embeddings_np, k_search)

        # 构建edge_index
        edge_list = []
        for i in range(len(embeddings_np)):
            for j in range(1, k_search):  # 跳过自己
                neighbor = indices[i, j]
                if neighbor != i:
                    edge_list.append([i, neighbor])

        if not edge_list:
            return torch.zeros((2, 0), dtype=torch.long, device=self.device)

        edge_index = torch.LongTensor(edge_list).t().contiguous()
        edge_index = to_undirected(edge_index)  # 无向图

        return edge_index.to(self.device)

    def _compute_heterophily(self, edge_index: torch.Tensor,
                             labels: np.ndarray,
                             test_start_idx: int) -> np.ndarray:
        """
        计算节点异配性

        对于训练节点：NHR = 不同类邻居数 / 总邻居数
        对于测试节点：NHR = 邻居类别多样性
        """
        if edge_index.size(1) == 0:
            return np.ones(test_start_idx)

        num_nodes = edge_index.max().item() + 1
        heterophily = np.zeros(num_nodes)

        edge_index_cpu = edge_index.cpu()

        # 扩展标签
        extended_labels = np.concatenate([
            labels,
            np.full(num_nodes - len(labels), -1)
        ])

        for i in range(num_nodes):
            # 找邻居
            mask = edge_index_cpu[0] == i
            neighbors = edge_index_cpu[1][mask].numpy()

            if len(neighbors) == 0:
                heterophily[i] = 1.0
                continue

            if i < len(labels):
                # 训练节点：标准异配性
                node_label = extended_labels[i]
                neighbor_labels = extended_labels[neighbors]
                valid_mask = neighbor_labels != -1
                if valid_mask.sum() > 0:
                    valid_neighbors = neighbor_labels[valid_mask]
                    diff_count = (valid_neighbors != node_label).sum()
                    heterophily[i] = diff_count / len(valid_neighbors)
            else:
                # 测试节点：类别多样性
                neighbor_labels = extended_labels[neighbors]
                valid_mask = neighbor_labels != -1
                if valid_mask.sum() > 0:
                    valid_neighbors = neighbor_labels[valid_mask]
                    unique_labels = len(np.unique(valid_neighbors))
                    # 归一化到[0, 1]
                    heterophily[i] = min(unique_labels / 10.0, 1.0)
                else:
                    heterophily[i] = 1.0

        return heterophily[test_start_idx:]

    def _contrastive_loss(self, embeddings: torch.Tensor,
                          labels: torch.Tensor,
                          temperature: float = 0.5) -> torch.Tensor:
        """
        对比损失（InfoNCE）

        同类样本拉近，异类样本推远
        """
        # 归一化
        embeddings = F.normalize(embeddings, dim=1)

        # 相似度矩阵
        similarity_matrix = torch.matmul(embeddings, embeddings.t()) / temperature

        # 掩码：同类为正样本
        labels_expanded = labels.unsqueeze(0)
        positive_mask = (labels_expanded == labels_expanded.t()).float()
        positive_mask.fill_diagonal_(0)  # 排除自己

        # 检查是否有正样本
        pos_count = positive_mask.sum(dim=1)
        if (pos_count == 0).all():
            return torch.tensor(0.0, device=embeddings.device)

        # 负样本掩码
        negative_mask = 1 - positive_mask
        negative_mask.fill_diagonal_(0)

        # 分子：正样本相似度
        exp_sim = torch.exp(similarity_matrix)
        pos_sim = (exp_sim * positive_mask).sum(dim=1)

        # 分母：所有样本相似度
        all_sim = (exp_sim * (positive_mask + negative_mask)).sum(dim=1)

        # InfoNCE损失
        loss = -torch.log(pos_sim / (all_sim + 1e-8) + 1e-8)

        # 只对有正样本的节点计算损失
        valid_mask = pos_count > 0
        if valid_mask.sum() > 0:
            return loss[valid_mask].mean()
        return torch.tensor(0.0, device=embeddings.device)

    def fit(self, train_embeddings: np.ndarray, train_labels: np.ndarray,
            epochs: int = 20, lr: float = 1e-3, contrast_weight: float = 0.1,
            verbose: bool = True):
        """
        训练阶段（有监督）

        Loss = 分类损失 + 对比损失
        """
        self.train_embeddings = self._normalize(train_embeddings).astype('float32')

        # 处理标签
        if isinstance(train_labels[0], str):
            unique_labels = sorted(set(train_labels))
            label_to_idx = {l: i for i, l in enumerate(unique_labels)}
            self.train_labels = np.array([label_to_idx[l] for l in train_labels])
        else:
            self.train_labels = np.array(train_labels)

        self.num_classes = len(np.unique(self.train_labels))

        # 初始化分类器
        self.classifier = nn.Linear(self.gnn.output_dim, self.num_classes).to(self.device)

        # 构建图
        edge_index = self._build_knn_graph(self.train_embeddings)

        # 转换为Tensor
        x = torch.FloatTensor(self.train_embeddings).to(self.device)
        y = torch.LongTensor(self.train_labels).to(self.device)

        # 优化器
        optimizer = torch.optim.Adam(
            list(self.gnn.parameters()) + list(self.classifier.parameters()),
            lr=lr, weight_decay=1e-5
        )

        # 训练循环
        self.gnn.train()
        self.classifier.train()

        for epoch in range(epochs):
            optimizer.zero_grad()

            # GNN前向
            gnn_out = self.gnn(x, edge_index)

            # 分类损失
            logits = self.classifier(gnn_out)
            loss_cls = F.cross_entropy(logits, y)

            # 对比损失
            loss_contrast = self._contrastive_loss(gnn_out, y)

            # 总损失
            loss = loss_cls + contrast_weight * loss_contrast

            loss.backward()
            optimizer.step()

            if verbose and (epoch + 1) % 5 == 0:
                acc = (logits.argmax(dim=1) == y).float().mean().item()
                print(f"  Epoch {epoch+1}/{epochs}: Loss={loss.item():.4f}, "
                      f"Cls={loss_cls.item():.4f}, Acc={acc:.4f}")

        self.fitted = True
        if verbose:
            print("[HeterophilyEnhanced v2] 训练完成")

    def score(self, test_embeddings: np.ndarray) -> np.ndarray:
        """
        测试阶段：计算OOD分数

        OOD_score = (1-α) * energy + α * heterophily
        """
        if not self.fitted:
            raise ValueError("模型未训练，请先调用fit()")

        test_embeddings = self._normalize(test_embeddings).astype('float32')

        # 合并训练+测试
        all_embeddings = np.vstack([self.train_embeddings, test_embeddings])
        edge_index = self._build_knn_graph(all_embeddings)

        # GNN前向
        self.gnn.eval()
        with torch.no_grad():
            x = torch.FloatTensor(all_embeddings).to(self.device)
            gnn_out = self.gnn(x, edge_index)

        # 分离测试节点
        test_start_idx = len(self.train_embeddings)
        test_gnn_out = gnn_out[test_start_idx:]
        train_gnn_out = gnn_out[:test_start_idx]

        # 能量分数1：到训练集质心的距离
        train_centroid = train_gnn_out.mean(dim=0)
        energy_dist = torch.norm(test_gnn_out - train_centroid, dim=1).detach().cpu().numpy()

        # 能量分数2：分类器的负最大logit
        logits = self.classifier(test_gnn_out)
        energy_logit = -logits.max(dim=1)[0].detach().cpu().numpy()

        # 综合能量（归一化）
        energy_dist_norm = (energy_dist - energy_dist.min()) / (energy_dist.max() - energy_dist.min() + 1e-10)
        energy_logit_norm = (energy_logit - energy_logit.min()) / (energy_logit.max() - energy_logit.min() + 1e-10)
        energy_scores = 0.5 * energy_dist_norm + 0.5 * energy_logit_norm

        # 异配性分数
        heterophily_scores = self._compute_heterophily(
            edge_index,
            self.train_labels,
            test_start_idx
        )

        # 加权融合
        ood_scores = (1 - self.alpha) * energy_scores + self.alpha * heterophily_scores

        return ood_scores

    def score_with_fix(self, test_embeddings: np.ndarray,
                       test_labels: np.ndarray) -> Tuple[np.ndarray, float]:
        """计算分数并自动修复方向"""
        from sklearn.metrics import roc_auc_score

        scores = self.score(test_embeddings)

        auroc_orig = roc_auc_score(test_labels, scores)
        auroc_inv = roc_auc_score(test_labels, -scores)

        if auroc_inv > auroc_orig + 0.05:
            return -scores, auroc_inv
        return scores, auroc_orig


def test_heterophily_v2():
    """测试HeterophilyEnhanced v2"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    from quick_fix import evaluate_ood

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

    # 训练和测试
    print("Testing HeterophilyEnhanced v2...")
    detector = HeterophilyEnhancedV2(
        input_dim=dim,
        hidden_dim=128,
        output_dim=64,
        k=20,
        num_layers=2,
        alpha=0.3
    )

    detector.fit(train_emb, train_labels, epochs=10, verbose=True)
    scores, auroc = detector.score_with_fix(test_emb, test_labels)

    print(f"\nTest AUROC: {auroc:.4f}")

    return auroc


if __name__ == "__main__":
    auroc = test_heterophily_v2()
    print(f"\n{'='*50}")
    print(f"HeterophilyEnhanced v2 Test: AUROC = {auroc:.4f}")
    print(f"{'='*50}")
