"""
RW2预实验 - 真实数据版
严格要求：必须使用真实TGB/OGB数据集，禁止模拟数据
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
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

# ============================================================
# 真实数据加载器 - 禁止模拟数据
# ============================================================

class RealDataLoader:
    """真实数据加载器 - 禁止模拟数据"""
    def __init__(self):
        self.data = None
        self.data_loaded = False
        self.data_source = None

        # 方案1: TGB
        try:
            from tgb.linkproppred.dataset import LinkPropPredDataset
            print("✓ 尝试加载TGB数据集 tgbl-wiki...")
            self.dataset = LinkPropPredDataset(name='tgbl-wiki', root='datasets')
            self.data = self.dataset.full_data
            self.data_loaded = True
            self.data_source = "TGB tgbl-wiki (Wikipedia编辑网络)"
            num_nodes = len(torch.unique(torch.cat([
                torch.tensor(self.data['sources']),
                torch.tensor(self.data['destinations'])
            ])))
            print(f"✓ TGB加载成功: {len(self.data['sources']):,} edges, {num_nodes:,} nodes")
        except Exception as e:
            print(f"✗ TGB失败: {e}")

        # 方案2: OGB (ogbl-collab - 论文合作网络)
        if not self.data_loaded:
            try:
                from ogb.linkproppred import LinkPropPredDataset
                print("✓ 尝试加载OGB数据集 ogbl-collab...")
                dataset = LinkPropPredDataset(name='ogbl-collab', root='datasets')
                edge_split = dataset.get_edge_split()
                self._convert_ogb_to_tgb_format(edge_split)
                self.data_loaded = True
                self.data_source = "OGB ogbl-collab (论文合作网络)"
                print(f"✓ OGB加载成功: {len(self.data['sources']):,} edges")
            except Exception as e:
                print(f"✗ OGB失败: {e}")

        # 失败处理：报错而不是降级
        if not self.data_loaded:
            raise RuntimeError(
                "\n" + "="*60 + "\n"
                "❌ 无法加载真实数据集！\n"
                "请手动安装: pip install py-tgb 或 pip install ogb\n"
                "不允许使用模拟数据运行实验\n"
                "="*60
            )

    def _convert_ogb_to_tgb_format(self, edge_split):
        """将OGB格式转换为TGB格式"""
        train_edge = edge_split['train']['edge']
        train_year = edge_split['train']['year'].flatten()
        valid_edge = edge_split['valid']['edge']
        valid_year = edge_split['valid']['year'].flatten()
        test_edge = edge_split['test']['edge']
        test_year = edge_split['test']['year'].flatten()

        # 合并数据
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

        num_nodes = int(all_edges.max()) + 1
        print(f"  Nodes: {num_nodes:,}, Edges: {n_total:,}")
        print(f"  Train: {n_train:,}, Valid: {n_valid:,}, Test: {n_test:,}")

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


def validate_real_data(loader):
    """验证使用的是真实数据"""
    print("\n验证数据真实性...")

    # 检查1: 数据规模合理
    n_edges = len(loader.data['sources'])
    if n_edges < 50000:
        raise RuntimeError(f"❌ 数据量太小 ({n_edges:,})，可能是模拟数据")
    print(f"  ✓ 边数: {n_edges:,} (>50,000)")

    # 检查2: 节点ID范围合理
    max_node = max(loader.data['sources'].max().item(),
                   loader.data['destinations'].max().item())
    if max_node < 1000:
        raise RuntimeError(f"❌ 节点数太少 ({max_node:,})，可能是模拟数据")
    print(f"  ✓ 最大节点ID: {max_node:,} (>1,000)")

    # 检查3: 时间戳分布
    timestamps = loader.data['timestamps'].float()
    time_std = timestamps.std().item()
    time_range = timestamps.max().item() - timestamps.min().item()
    print(f"  ✓ 时间戳范围: {time_range:.2f}, 标准差: {time_std:.2f}")

    print("✅ 数据验证通过：确认使用真实数据集")
    print(f"   数据源: {loader.data_source}")

# ============================================================
# 模型定义（5个方案 - 简化实现）
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

class MoMentPP(nn.Module):
    """多模态融合"""
    def __init__(self, num_nodes, dim=172):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, dim)
        self.time_enc = TimeEncoder(dim)
        self.temporal = nn.GRU(dim, dim, batch_first=True)
        self.text_proj = nn.Linear(dim, dim)
        self.struct_gnn = nn.Linear(dim, dim)
        self.fusion = nn.Linear(dim*3, dim)

    def forward(self, src, dst, time):
        src_emb = self.node_emb(src)
        dst_emb = self.node_emb(dst)
        time_emb = self.time_enc(time)[:, :self.node_emb.embedding_dim]

        # 三模态
        temp, _ = self.temporal((src_emb + dst_emb + time_emb).unsqueeze(1))
        text = self.text_proj((src_emb + dst_emb) / 2)
        struct = F.relu(self.struct_gnn(src_emb + dst_emb))

        # 融合
        fused = torch.cat([temp.squeeze(1), text, struct], dim=-1)
        return self.fusion(fused)

class THGMamba(nn.Module):
    """异构图Mamba"""
    def __init__(self, num_nodes, dim=256):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, dim)
        self.time_enc = TimeEncoder(dim)
        self.enc1 = nn.GRU(dim, dim, batch_first=True)
        self.enc2 = nn.GRU(dim, dim, batch_first=True)
        self.enc3 = nn.GRU(dim, dim, batch_first=True)
        self.fc = nn.Linear(dim, dim)

    def forward(self, src, dst, time):
        src_emb = self.node_emb(src)
        dst_emb = self.node_emb(dst)
        time_emb = self.time_enc(time)[:, :self.node_emb.embedding_dim]

        x = (src_emb + dst_emb + time_emb).unsqueeze(1)
        h1, _ = self.enc1(x)
        h2, _ = self.enc2(h1)
        h3, _ = self.enc3(h2)

        return self.fc(h3.squeeze(1))

class TempMemLLM(nn.Module):
    """LLM增强记忆"""
    def __init__(self, num_nodes, dim=172):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, dim)
        self.time_enc = TimeEncoder(dim)
        self.memory = nn.GRU(dim, dim, batch_first=True)
        self.llm_proj = nn.Linear(dim, dim)
        self.fusion = nn.Linear(dim*2, dim)

    def forward(self, src, dst, time):
        src_emb = self.node_emb(src)
        dst_emb = self.node_emb(dst)
        time_emb = self.time_enc(time)[:, :self.node_emb.embedding_dim]

        x = (src_emb + dst_emb + time_emb).unsqueeze(1)
        mem, _ = self.memory(x)
        llm = self.llm_proj((src_emb + dst_emb) / 2)

        fused = torch.cat([mem.squeeze(1), llm], dim=-1)
        return self.fusion(fused)

class FreqTemporal(nn.Module):
    """频域增强"""
    def __init__(self, num_nodes, dim=172):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, dim)
        self.time_enc = TimeEncoder(dim)
        self.time_gnn = nn.Linear(dim, dim)
        self.freq_proj = nn.Linear(dim, dim)
        self.fusion = nn.Linear(dim*2, dim)

    def forward(self, src, dst, time):
        src_emb = self.node_emb(src)
        dst_emb = self.node_emb(dst)
        time_emb = self.time_enc(time)[:, :self.node_emb.embedding_dim]

        # 时域
        time_feat = F.relu(self.time_gnn(src_emb + dst_emb + time_emb))

        # 频域（简化：FFT）
        freq_feat = torch.fft.rfft(src_emb + dst_emb, dim=-1)
        freq_feat = torch.cat([freq_feat.real, freq_feat.imag], dim=-1)
        freq_feat = self.freq_proj(freq_feat[:, :self.node_emb.embedding_dim])

        # 融合
        fused = torch.cat([time_feat, freq_feat], dim=-1)
        return self.fusion(fused)

# ============================================================
# 训练与评估
# ============================================================

def train_model(model, train_data, epochs=5, lr=0.001, batch_size=1024):
    """训练模型"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    sources = train_data['sources']
    destinations = train_data['destinations']
    timestamps = train_data['timestamps']

    n_samples = len(sources)
    n_batches = (n_samples + batch_size - 1) // batch_size

    model.train()
    for epoch in range(epochs):
        total_loss = 0
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

            # 正样本分数：源-目标对
            pos_score = F.cosine_similarity(emb, dst_emb, dim=-1)

            # 负采样：随机目标节点
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
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    return model

def evaluate_model(model, test_data, batch_size=200):
    """评估模型（简化MRR）"""
    model.eval()

    sources = test_data['sources']
    destinations = test_data['destinations']
    timestamps = test_data['timestamps']

    n_samples = min(1000, len(sources))  # 只评估前1000个样本（快速）

    mrrs = []

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
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

    return np.mean(mrrs)

# ============================================================
# 主实验流程
# ============================================================

def main():
    print("\n" + "="*60)
    print("RW2预实验 - 真实数据版")
    print("严格要求：必须使用TGB/OGB真实数据集")
    print("="*60 + "\n")

    # 1. 加载真实数据
    print("[1/5] 加载真实数据...")
    loader = RealDataLoader()

    # 2. 验证数据真实性
    print("\n[2/5] 验证数据真实性...")
    validate_real_data(loader)

    train_data = loader.get_train_data()
    test_data = loader.get_test_data()

    num_nodes = int(max(
        train_data['sources'].max(),
        train_data['destinations'].max(),
        test_data['sources'].max(),
        test_data['destinations'].max()
    )) + 1

    print(f"\n节点数: {num_nodes:,}")

    # 3. 定义所有模型
    models = {
        'NPPCTNE': NPPCTNE(num_nodes, dim=128),
        'MoMent++': MoMentPP(num_nodes, dim=172),
        'THG-Mamba': THGMamba(num_nodes, dim=256),
        'TempMem-LLM': TempMemLLM(num_nodes, dim=172),
        'FreqTemporal': FreqTemporal(num_nodes, dim=172)
    }

    # 4. 训练和评估所有模型
    print("\n[3/5] 训练和评估模型...")
    results = {}

    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"训练 {name}")
        print('='*60)

        # 训练（优化：5个epoch，更大batch）
        model = train_model(model, train_data, epochs=5, lr=0.001, batch_size=1024)

        # 评估
        print(f"\n评估 {name}...")
        mrr = evaluate_model(model, test_data)

        results[name] = {
            'mrr': mrr,
            'model': model
        }

        print(f"{name} - MRR: {mrr:.4f}")

        # 保存模型
        torch.save(model.state_dict(), f'checkpoints/{name.replace("+", "P")}.pth')

    # 5. 生成报告
    print("\n[4/5] 生成性能对比报告...")

    baseline_mrr = results['NPPCTNE']['mrr']

    # 创建DataFrame
    df_data = []
    for name, result in results.items():
        improvement = (result['mrr'] - baseline_mrr) / baseline_mrr * 100
        df_data.append({
            '方法': name,
            'MRR': f"{result['mrr']:.4f}",
            '相对baseline': f"{improvement:+.2f}%",
            '通过': '✅' if improvement > 3 else '❌'
        })

    df = pd.DataFrame(df_data)

    # 保存CSV
    df.to_csv('results/performance_comparison.csv', index=False)

    # 打印表格
    print("\n" + "="*60)
    print("性能对比表")
    print("="*60)
    print(df.to_string(index=False))

    # 6. 可视化
    print("\n[5/5] 生成可视化...")

    plt.figure(figsize=(10, 6))
    names = [r['方法'] for r in df_data]
    mrrs = [results[name]['mrr'] for name in names]
    colors = ['red' if name == 'NPPCTNE' else 'green' if results[name]['mrr'] > baseline_mrr * 1.03 else 'orange'
              for name in names]

    bars = plt.bar(names, mrrs, color=colors, alpha=0.7)
    plt.axhline(y=baseline_mrr, color='r', linestyle='--', label='Baseline')
    plt.xlabel('方法')
    plt.ylabel('MRR')
    plt.title('RW2预实验性能对比')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/performance_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n✅ 实验完成！")
    print("\n生成的文件:")
    print("  - results/performance_comparison.csv")
    print("  - figures/performance_comparison.pdf")
    print("  - figures/performance_comparison.png")
    print("  - checkpoints/*.pth (5个模型)")

    # 7. 总结
    print("\n" + "="*60)
    print("实验总结")
    print("="*60)

    # 数据集信息
    print("\n## 数据集信息")
    print(f"  - 数据源: {loader.data_source}")
    n_nodes = num_nodes
    n_edges = len(loader.data['sources'])
    print(f"  - 节点数: {n_nodes:,}")
    print(f"  - 边数: {n_edges:,}")
    print(f"  - 数据类型: ✅ 真实网络数据")

    passed = sum(1 for r in df_data if r['通过'] == '✅') - 1  # 减去baseline
    print(f"\n通过性能测试的方法: {passed}/4")
    print(f"Baseline MRR: {baseline_mrr:.4f}")

    best_method = max(results.items(), key=lambda x: x[1]['mrr'])
    print(f"最佳方法: {best_method[0]} (MRR={best_method[1]['mrr']:.4f})")

    print("\n下一步:")
    if passed >= 2:
        print("  ✅ 至少2个方法通过，可以进入主实验阶段")
    else:
        print("  ⚠️ 通过的方法<2个，建议调整技术方案")

if __name__ == '__main__':
    main()
