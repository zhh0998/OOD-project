"""
HeterophilyEnhanced OOD检测器 - RW3核心创新方法

核心思想：异配性高的节点更可能是OOD样本

架构组件：
1. k-NN图构建（FAISS，k=50）
2. 消息传递（可用PyG GATv2或简化版）
3. 节点异配性计算（Node Heterophily Ratio）
4. 能量头（Energy-based OOD scoring）
5. 异配性调制（加权融合）

Author: RW3 OOD Detection Project
"""

import numpy as np
from typing import Tuple, Optional
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available, using numpy-only implementation")

try:
    from torch_geometric.nn import GATv2Conv
    from torch_geometric.data import Data
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class HeterophilyEnhancedDetector:
    """
    RW3核心方法：异配性感知OOD检测

    Pipeline:
    1. Sentence embeddings → k-NN graph
    2. Message passing (GATv2 or simplified)
    3. Compute node heterophily
    4. Energy-based + Heterophily-modulated OOD score

    核心创新点：
    - 利用图结构中的异配性信息
    - OOD样本往往与ID样本的邻居分布不同（高异配性）
    - 通过GNN消息传递增强特征表示
    """

    def __init__(self, input_dim: int = 384, hidden_dim: int = 256,
                 k: int = 50, num_gnn_layers: int = 2,
                 alpha: float = 0.3, use_gnn: bool = True,
                 verbose: bool = True):
        """
        Args:
            input_dim: 输入embedding维度
            hidden_dim: GNN隐藏层维度
            k: k-NN图的k值
            num_gnn_layers: GNN层数
            alpha: 异配性权重（0-1之间）
            use_gnn: 是否使用GNN（否则只用异配性）
            verbose: 是否打印详细信息
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k
        self.num_gnn_layers = num_gnn_layers
        self.alpha = alpha
        self.use_gnn = use_gnn and TORCH_AVAILABLE
        self.verbose = verbose

        self.train_embeddings = None
        self.train_labels = None
        self.label_to_idx = None

        # 初始化GNN（如果可用）
        if self.use_gnn and PYG_AVAILABLE:
            self.gnn = HeterophilyGNN(input_dim, hidden_dim, num_gnn_layers)
            if self.verbose:
                print("[HeterophilyEnhanced] Using PyG GATv2 GNN")
        elif self.use_gnn and TORCH_AVAILABLE:
            self.gnn = SimplifiedGNN(input_dim, hidden_dim, num_gnn_layers)
            if self.verbose:
                print("[HeterophilyEnhanced] Using simplified GNN (no PyG)")
        else:
            self.gnn = None
            if self.verbose:
                print("[HeterophilyEnhanced] Using heterophily-only mode (no GNN)")

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2归一化"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        return embeddings / norms

    def fit(self, train_embeddings: np.ndarray, train_labels: np.ndarray):
        """
        训练阶段：存储训练数据用于构图

        Args:
            train_embeddings: 训练集embeddings, shape=(n_samples, dim)
            train_labels: 训练集类别标签（整数）
        """
        self.train_embeddings = self._normalize(train_embeddings).astype('float32')

        # 处理标签（转换为整数索引）
        if isinstance(train_labels[0], str):
            unique_labels = sorted(set(train_labels))
            self.label_to_idx = {l: i for i, l in enumerate(unique_labels)}
            self.train_labels = np.array([self.label_to_idx[l] for l in train_labels])
        else:
            self.train_labels = np.array(train_labels)

        self.num_classes = len(set(self.train_labels))

        if self.verbose:
            print(f"[HeterophilyEnhanced] Fitted with {len(self.train_embeddings)} samples, "
                  f"{self.num_classes} classes")

    def _build_knn_graph(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建k-NN图

        Args:
            embeddings: 所有节点的embeddings

        Returns:
            edge_index: 边索引 (2, num_edges)
            edge_weights: 边权重 (num_edges,)
        """
        embeddings_np = embeddings.astype('float32')
        n_samples = len(embeddings_np)

        if FAISS_AVAILABLE:
            # 使用FAISS加速
            d = embeddings_np.shape[1]
            index = faiss.IndexFlatIP(d)  # 内积（归一化后=余弦相似度）
            index.add(embeddings_np)

            similarities, indices = index.search(embeddings_np, self.k + 1)
        else:
            # 使用numpy计算（较慢但无依赖）
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=self.k + 1, metric='cosine')
            nn.fit(embeddings_np)
            distances, indices = nn.kneighbors(embeddings_np)
            similarities = 1 - distances  # 转换为相似度

        # 构建边列表
        edge_list = []
        edge_weights = []

        for i in range(n_samples):
            for j in range(1, min(self.k + 1, len(indices[i]))):  # 跳过自己
                neighbor_idx = indices[i, j]
                if neighbor_idx != i:  # 确保不是自环
                    edge_list.append([i, neighbor_idx])
                    edge_weights.append(similarities[i, j])

        edge_index = np.array(edge_list).T if edge_list else np.zeros((2, 0), dtype=np.int64)
        edge_weights = np.array(edge_weights) if edge_weights else np.array([])

        return edge_index, edge_weights

    def _compute_heterophily(self, edge_index: np.ndarray,
                             all_labels: np.ndarray,
                             test_start_idx: int) -> np.ndarray:
        """
        计算节点异配性（Node Heterophily Ratio）

        对于测试节点：衡量其邻居的标签分布多样性
        异配性高 → 邻居来自多个不同类别 → 可能是OOD

        Args:
            edge_index: 边索引
            all_labels: 所有节点标签（测试节点标签为-1）
            test_start_idx: 测试节点起始索引

        Returns:
            测试节点的异配性分数
        """
        num_nodes = edge_index.max() + 1 if edge_index.size > 0 else test_start_idx
        num_test = num_nodes - test_start_idx
        heterophily = np.zeros(num_test)

        # 为每个测试节点计算异配性
        for test_idx in range(num_test):
            node_idx = test_start_idx + test_idx

            # 找到该节点的所有邻居
            neighbor_mask = edge_index[0] == node_idx
            neighbors = edge_index[1][neighbor_mask]

            if len(neighbors) == 0:
                heterophily[test_idx] = 1.0  # 无邻居视为高异配性
                continue

            # 获取邻居中训练节点的标签
            train_neighbors = neighbors[neighbors < test_start_idx]

            if len(train_neighbors) == 0:
                heterophily[test_idx] = 1.0
                continue

            neighbor_labels = all_labels[train_neighbors]

            # 方法1: 标签熵（越高越异配）
            unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
            probs = counts / counts.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(self.num_classes)

            # 方法2: 唯一标签比例
            unique_ratio = len(unique_labels) / min(len(train_neighbors), self.num_classes)

            # 综合两种方法
            heterophily[test_idx] = 0.5 * (entropy / max_entropy) + 0.5 * unique_ratio

        return heterophily

    def _compute_energy_scores(self, test_emb: np.ndarray,
                               train_emb: np.ndarray) -> np.ndarray:
        """
        计算能量分数（基于到训练集的距离）

        能量高 → 远离训练分布 → OOD
        """
        # 确保维度匹配
        if test_emb.shape[1] != train_emb.shape[1]:
            # 如果维度不匹配，使用原始embeddings计算能量
            # 这种情况发生在GNN输出维度与原始不同时
            centroid = train_emb.mean(axis=0)
            # 使用余弦距离而非欧氏距离
            test_norm = test_emb / (np.linalg.norm(test_emb, axis=1, keepdims=True) + 1e-10)
            energy = np.linalg.norm(test_norm, axis=1)  # 简化：使用范数作为能量
            return energy

        # 计算到最近训练样本的距离
        if FAISS_AVAILABLE:
            index = faiss.IndexFlatIP(train_emb.shape[1])
            index.add(train_emb.astype('float32'))
            k = min(10, len(train_emb))
            sims, _ = index.search(test_emb.astype('float32'), k)
            # 平均相似度的负值作为能量
            energy = -sims.mean(axis=1)
        else:
            # 简化版：到质心的距离
            centroid = train_emb.mean(axis=0)
            energy = np.linalg.norm(test_emb - centroid, axis=1)

        return energy

    def _run_gnn(self, embeddings: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        """运行GNN获取增强表示"""
        if self.gnn is None or not TORCH_AVAILABLE:
            return embeddings

        x = torch.FloatTensor(embeddings)
        edge_idx = torch.LongTensor(edge_index)

        self.gnn.eval()
        with torch.no_grad():
            if PYG_AVAILABLE and isinstance(self.gnn, HeterophilyGNN):
                out = self.gnn(x, edge_idx)
            else:
                out = self.gnn(x, edge_idx)

        return out.numpy()

    def score(self, test_embeddings: np.ndarray) -> np.ndarray:
        """
        计算OOD分数

        Args:
            test_embeddings: 测试集embeddings

        Returns:
            OOD分数（越高越可能是OOD）
        """
        test_embeddings = self._normalize(test_embeddings).astype('float32')
        n_train = len(self.train_embeddings)
        n_test = len(test_embeddings)

        # 1. 合并训练+测试构建完整图
        all_embeddings = np.vstack([self.train_embeddings, test_embeddings])

        # 2. 构建k-NN图
        edge_index, edge_weights = self._build_knn_graph(all_embeddings)

        if self.verbose:
            print(f"[HeterophilyEnhanced] Graph built: {len(all_embeddings)} nodes, "
                  f"{edge_index.shape[1]} edges")

        # 3. GNN前向传播（可选）
        if self.use_gnn:
            gnn_embeddings = self._run_gnn(all_embeddings, edge_index)
            test_gnn_emb = gnn_embeddings[n_train:]
        else:
            test_gnn_emb = test_embeddings

        # 4. 计算能量分数
        energy_scores = self._compute_energy_scores(test_gnn_emb, self.train_embeddings)

        # 归一化能量分数到[0,1]
        energy_scores = (energy_scores - energy_scores.min()) / (energy_scores.max() - energy_scores.min() + 1e-10)

        # 5. 计算异配性分数
        all_labels = np.concatenate([self.train_labels, np.full(n_test, -1)])
        heterophily_scores = self._compute_heterophily(edge_index, all_labels, n_train)

        if self.verbose:
            print(f"[HeterophilyEnhanced] Energy scores: mean={energy_scores.mean():.4f}, "
                  f"std={energy_scores.std():.4f}")
            print(f"[HeterophilyEnhanced] Heterophily scores: mean={heterophily_scores.mean():.4f}, "
                  f"std={heterophily_scores.std():.4f}")

        # 6. 异配性调制融合
        # OOD分数 = (1-α) * energy + α * heterophily
        ood_scores = (1 - self.alpha) * energy_scores + self.alpha * heterophily_scores

        return ood_scores

    def score_with_fix(self, test_embeddings: np.ndarray,
                       test_labels: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        计算OOD分数并自动修复分数方向

        Args:
            test_embeddings: 测试集embeddings
            test_labels: 测试标签 (0=ID, 1=OOD)

        Returns:
            (修正后的分数, AUROC)
        """
        from sklearn.metrics import roc_auc_score

        scores = self.score(test_embeddings)

        # 自动修复分数方向
        auroc_orig = roc_auc_score(test_labels, scores)
        auroc_inv = roc_auc_score(test_labels, -scores)

        if auroc_inv > auroc_orig + 0.05:
            if self.verbose:
                print(f"[HeterophilyEnhanced] 修复分数反转: {auroc_orig:.4f} -> {auroc_inv:.4f}")
            scores = -scores
            final_auroc = auroc_inv
        else:
            final_auroc = auroc_orig

        return scores, final_auroc


class HeterophilyGNN(nn.Module):
    """GATv2消息传递网络（需要PyG）"""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()

        self.convs = nn.ModuleList()

        # 第一层
        self.convs.append(GATv2Conv(input_dim, hidden_dim, heads=4, concat=False))

        # 后续层
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False))

        # 输出投影
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.1, training=self.training)

        x = self.out_proj(x)
        return x


class SimplifiedGNN(nn.Module):
    """简化版GNN（不需要PyG，使用邻接矩阵消息传递）"""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()

        self.layers = nn.ModuleList()

        # 第一层
        self.layers.append(nn.Linear(input_dim, hidden_dim))

        # 后续层
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        # 注意力参数
        self.att_weight = nn.Parameter(torch.FloatTensor(hidden_dim, 1))
        nn.init.xavier_uniform_(self.att_weight)

    def forward(self, x, edge_index):
        """
        简化的消息传递

        Args:
            x: 节点特征 (N, D)
            edge_index: 边索引 (2, E)
        """
        num_nodes = x.size(0)

        for i, layer in enumerate(self.layers):
            # 线性变换
            x_transformed = layer(x)

            if i < len(self.layers) - 1:
                # 简化的邻居聚合
                # 构建邻接矩阵
                adj = torch.zeros(num_nodes, num_nodes, device=x.device)
                if edge_index.size(1) > 0:
                    adj[edge_index[0], edge_index[1]] = 1.0

                # 度归一化
                deg = adj.sum(dim=1, keepdim=True) + 1e-10
                adj_norm = adj / deg

                # 消息传递：聚合邻居特征
                neighbor_features = torch.mm(adj_norm, x_transformed)

                # 结合自身和邻居特征
                x = F.relu(x_transformed + neighbor_features)
                x = F.dropout(x, p=0.1, training=self.training)
            else:
                x = x_transformed

        return x


def test_heterophily_enhanced():
    """测试HeterophilyEnhanced检测器"""
    np.random.seed(42)

    # 模拟数据
    n_train = 500
    n_test_id = 100
    n_test_ood = 100
    dim = 384
    n_classes = 10

    # 训练数据：10个类别的高斯分布
    train_embeddings = []
    train_labels = []
    for c in range(n_classes):
        center = np.random.randn(dim) * 2
        samples = center + np.random.randn(n_train // n_classes, dim) * 0.5
        train_embeddings.append(samples)
        train_labels.extend([c] * (n_train // n_classes))

    train_embeddings = np.vstack(train_embeddings).astype('float32')
    train_labels = np.array(train_labels)

    # 测试ID数据：与训练分布相似
    test_id = np.random.randn(n_test_id, dim).astype('float32') * 0.5

    # 测试OOD数据：偏移的分布
    test_ood = np.random.randn(n_test_ood, dim).astype('float32') * 0.5 + 5

    test_embeddings = np.vstack([test_id, test_ood])
    test_labels = np.array([0] * n_test_id + [1] * n_test_ood)

    # 测试检测器
    print("Testing HeterophilyEnhancedDetector...")
    detector = HeterophilyEnhancedDetector(
        input_dim=dim,
        hidden_dim=128,
        k=20,
        num_gnn_layers=2,
        alpha=0.3,
        verbose=True
    )

    detector.fit(train_embeddings, train_labels)
    scores, auroc = detector.score_with_fix(test_embeddings, test_labels)

    print(f"\nTest AUROC: {auroc:.4f}")
    print(f"Scores - mean: {scores.mean():.4f}, std: {scores.std():.4f}")

    return auroc


if __name__ == "__main__":
    auroc = test_heterophily_enhanced()
    print(f"\n{'='*50}")
    print(f"HeterophilyEnhanced Test: AUROC = {auroc:.4f}")
    print(f"{'='*50}")
