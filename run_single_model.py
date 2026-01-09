"""
RW2预实验 - 单模型快速测试版
仅测试NPPCTNE baseline, 1 epoch, 验证可行性
预计运行时间: 5-10分钟
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 修复PyTorch 2.6+ weights_only问题
# ============================================================
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# 设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

print(f"Device: {device}")
print("="*60)
print("单模型快速测试 - NPPCTNE (1 epoch)")
print("="*60)

# ============================================================
# 数据加载器
# ============================================================

class RealDataLoader:
    """真实数据加载器"""
    def __init__(self):
        self.data = None
        self.data_loaded = False
        self.data_source = None

        # 方案1: TGB
        try:
            from tgb.linkproppred.dataset import LinkPropPredDataset
            print("尝试加载TGB数据集 tgbl-wiki...")
            self.dataset = LinkPropPredDataset(name='tgbl-wiki', root='datasets')
            self.data = self.dataset.full_data
            self.data_loaded = True
            self.data_source = "TGB tgbl-wiki"
            num_nodes = len(torch.unique(torch.cat([
                torch.tensor(self.data['sources']),
                torch.tensor(self.data['destinations'])
            ])))
            print(f"TGB加载成功: {len(self.data['sources']):,} edges, {num_nodes:,} nodes")
        except Exception as e:
            print(f"TGB失败: {e}")

        # 方案2: OGB
        if not self.data_loaded:
            try:
                from ogb.linkproppred import LinkPropPredDataset
                print("尝试加载OGB数据集 ogbl-collab...")
                dataset = LinkPropPredDataset(name='ogbl-collab', root='datasets')
                edge_split = dataset.get_edge_split()
                self._convert_ogb_to_tgb_format(edge_split)
                self.data_loaded = True
                self.data_source = "OGB ogbl-collab"
                print(f"OGB加载成功: {len(self.data['sources']):,} edges")
            except Exception as e:
                print(f"OGB失败: {e}")

        if not self.data_loaded:
            raise RuntimeError("无法加载真实数据集！请安装: pip install py-tgb 或 pip install ogb")

    def _convert_ogb_to_tgb_format(self, edge_split):
        """将OGB格式转换为TGB格式"""
        train_edge = edge_split['train']['edge']
        train_year = edge_split['train']['year'].flatten()
        valid_edge = edge_split['valid']['edge']
        valid_year = edge_split['valid']['year'].flatten()
        test_edge = edge_split['test']['edge']
        test_year = edge_split['test']['year'].flatten()

        all_edges = np.vstack([train_edge, valid_edge, test_edge])
        all_years = np.concatenate([train_year, valid_year, test_year])

        n_train = len(train_edge)
        n_valid = len(valid_edge)
        n_test = len(test_edge)
        n_total = n_train + n_valid + n_test

        self.data = {
            'sources': torch.from_numpy(all_edges[:, 0].astype(np.int64)),
            'destinations': torch.from_numpy(all_edges[:, 1].astype(np.int64)),
            'timestamps': torch.from_numpy(all_years.astype(np.float32)),
            'train_mask': torch.zeros(n_total, dtype=torch.bool),
            'val_mask': torch.zeros(n_total, dtype=torch.bool),
            'test_mask': torch.zeros(n_total, dtype=torch.bool)
        }
        self.data['train_mask'][:n_train] = True
        self.data['val_mask'][n_train:n_train+n_valid] = True
        self.data['test_mask'][n_train+n_valid:] = True

        print(f"  Nodes: {int(all_edges.max()) + 1:,}, Train: {n_train:,}, Test: {n_test:,}")

    def get_train_data(self):
        mask = self.data['train_mask']
        return {
            'sources': self.data['sources'][mask],
            'destinations': self.data['destinations'][mask],
            'timestamps': self.data['timestamps'][mask]
        }

    def get_test_data(self):
        mask = self.data['test_mask']
        return {
            'sources': self.data['sources'][mask],
            'destinations': self.data['destinations'][mask],
            'timestamps': self.data['timestamps'][mask]
        }


# ============================================================
# 模型定义
# ============================================================

class TimeEncoder(nn.Module):
    """时间编码器"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timestamps):
        device = timestamps.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timestamps.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class NPPCTNE(nn.Module):
    """Baseline: 双响应机制"""
    def __init__(self, num_nodes, dim=128):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, dim)
        self.time_enc = TimeEncoder(dim)
        self.gru1 = nn.GRU(dim, 64, batch_first=True)
        self.gru2 = nn.GRU(dim, 64, batch_first=True)
        self.fc = nn.Linear(128, dim)

    def forward(self, src, dst, time):
        src_emb = self.node_emb(src)
        dst_emb = self.node_emb(dst)
        time_emb = self.time_enc(time)[:, :self.node_emb.embedding_dim]

        x = (src_emb + dst_emb + time_emb).unsqueeze(1)
        h1, _ = self.gru1(x)
        h2, _ = self.gru2(x)

        out = self.fc(torch.cat([h1.squeeze(1), h2.squeeze(1)], dim=-1))
        return out


# ============================================================
# 训练与评估
# ============================================================

def train_model(model, train_data, epochs=1, lr=0.001, batch_size=2048):
    """训练模型 - 优化版：增大batch_size提高速度"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    sources = train_data['sources']
    destinations = train_data['destinations']
    timestamps = train_data['timestamps']

    n_samples = len(sources)
    n_batches = (n_samples + batch_size - 1) // batch_size

    print(f"\n训练配置:")
    print(f"  - 样本数: {n_samples:,}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Batches/epoch: {n_batches}")
    print(f"  - Epochs: {epochs}")

    start_time = time.time()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        epoch_start = time.time()

        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{epochs}")

        for i in pbar:
            start_idx = i * batch_size
            end_idx = min((i+1) * batch_size, n_samples)

            batch_src = sources[start_idx:end_idx].to(device)
            batch_dst = destinations[start_idx:end_idx].to(device)
            batch_time = timestamps[start_idx:end_idx].to(device)

            # 前向传播
            emb = model(batch_src, batch_dst, batch_time)

            # 获取源节点和目标节点嵌入
            src_emb = model.node_emb(batch_src)
            dst_emb = model.node_emb(batch_dst)

            # 正样本分数
            pos_score = F.cosine_similarity(emb, dst_emb, dim=-1)

            # 负采样
            neg_dst = torch.randint(0, model.node_emb.num_embeddings, (len(batch_src),), device=device)
            neg_emb = model.node_emb(neg_dst)
            neg_score = F.cosine_similarity(emb, neg_emb, dim=-1)

            # BPR损失
            loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 更频繁的进度显示
            if (i + 1) % 100 == 0:
                elapsed = time.time() - epoch_start
                eta = elapsed / (i + 1) * (n_batches - i - 1)
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'ETA': f'{eta/60:.1f}min'
                })
            else:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / n_batches
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Time = {epoch_time:.1f}s")

    total_time = time.time() - start_time
    print(f"\n训练完成! 总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")

    return model, total_time


def evaluate_model(model, test_data, batch_size=200, n_eval_samples=1000):
    """评估模型（快速MRR）"""
    model.eval()

    sources = test_data['sources']
    destinations = test_data['destinations']
    timestamps = test_data['timestamps']

    n_samples = min(n_eval_samples, len(sources))
    print(f"\n评估配置:")
    print(f"  - 评估样本数: {n_samples}")

    mrrs = []
    start_time = time.time()

    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc="评估中"):
            end_idx = min(i + batch_size, n_samples)

            batch_src = sources[i:end_idx].to(device)
            batch_dst = destinations[i:end_idx].to(device)
            batch_time = timestamps[i:end_idx].to(device)

            # 获取嵌入
            emb = model(batch_src, batch_dst, batch_time)

            # 计算与所有节点的相似度
            all_node_emb = model.node_emb.weight
            scores = torch.matmul(emb, all_node_emb.t())

            # 计算MRR
            ranks = torch.argsort(torch.argsort(scores, dim=1, descending=True), dim=1) + 1
            target_ranks = ranks[torch.arange(len(batch_dst)), batch_dst]
            mrr = (1.0 / target_ranks.float()).mean().item()

            mrrs.append(mrr)

    eval_time = time.time() - start_time
    final_mrr = np.mean(mrrs)

    print(f"评估完成! 耗时: {eval_time:.1f}s")
    print(f"MRR: {final_mrr:.4f}")

    return final_mrr, eval_time


# ============================================================
# 主流程
# ============================================================

def main():
    print("\n" + "="*60)
    print("RW2预实验 - 单模型快速测试")
    print("模型: NPPCTNE (baseline)")
    print("Epochs: 1")
    print("="*60 + "\n")

    total_start = time.time()

    # 1. 加载数据
    print("[1/4] 加载真实数据...")
    loader = RealDataLoader()

    train_data = loader.get_train_data()
    test_data = loader.get_test_data()

    num_nodes = int(max(
        train_data['sources'].max(),
        train_data['destinations'].max(),
        test_data['sources'].max(),
        test_data['destinations'].max()
    )) + 1

    print(f"\n数据统计:")
    print(f"  - 数据源: {loader.data_source}")
    print(f"  - 节点数: {num_nodes:,}")
    print(f"  - 训练边数: {len(train_data['sources']):,}")
    print(f"  - 测试边数: {len(test_data['sources']):,}")

    # 2. 创建模型
    print("\n[2/4] 创建NPPCTNE模型...")
    model = NPPCTNE(num_nodes, dim=128)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  - 参数量: {n_params:,}")

    # 3. 训练
    print("\n[3/4] 训练模型 (1 epoch)...")
    model, train_time = train_model(
        model, train_data,
        epochs=1,
        lr=0.001,
        batch_size=2048  # 增大batch size加速
    )

    # 4. 评估
    print("\n[4/4] 评估模型...")
    mrr, eval_time = evaluate_model(model, test_data, n_eval_samples=1000)

    # 保存结果
    os.makedirs('results', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    # 保存模型
    torch.save(model.state_dict(), 'checkpoints/NPPCTNE_single_test.pth')

    # 保存结果JSON
    total_time = time.time() - total_start
    results = {
        'model': 'NPPCTNE',
        'epochs': 1,
        'mrr': mrr,
        'train_time_seconds': train_time,
        'eval_time_seconds': eval_time,
        'total_time_seconds': total_time,
        'data_source': loader.data_source,
        'num_nodes': num_nodes,
        'train_edges': len(train_data['sources']),
        'test_edges': len(test_data['sources']),
        'device': str(device)
    }

    with open('results/single_model_test.json', 'w') as f:
        json.dump(results, f, indent=2)

    # 打印总结
    print("\n" + "="*60)
    print("单模型测试完成!")
    print("="*60)
    print(f"\n结果:")
    print(f"  - 模型: NPPCTNE (baseline)")
    print(f"  - MRR: {mrr:.4f}")
    print(f"  - 训练时间: {train_time:.1f}s ({train_time/60:.1f}min)")
    print(f"  - 评估时间: {eval_time:.1f}s")
    print(f"  - 总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")

    print(f"\n生成文件:")
    print(f"  - checkpoints/NPPCTNE_single_test.pth")
    print(f"  - results/single_model_test.json")

    # 策略建议
    print("\n" + "="*60)
    print("后续策略建议:")
    print("="*60)

    if total_time < 600:  # < 10分钟
        print("  单模型测试 < 10分钟完成")
        print("  建议: 可以继续运行完整5模型实验（分批执行）")
        print("  预计每个模型: ~5-10分钟")
        print("  预计总时间: ~30-50分钟")
    elif total_time < 900:  # < 15分钟
        print("  单模型测试 < 15分钟完成")
        print("  建议: 继续分批运行5模型实验")
        print("  建议降低epochs到1或2")
    else:
        print("  单模型测试 > 15分钟")
        print("  建议: 考虑使用更小数据集(tgbl-wiki)或10%采样")

    return results


if __name__ == '__main__':
    results = main()
