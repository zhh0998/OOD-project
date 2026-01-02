"""
RW2预实验 - 精简执行版
只保留核心功能：5个模型训练 + 性能对比
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

# 设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

print(f"Device: {device}")
print("="*60)

# ============================================================
# 数据加载（简化版 - 使用TGB）
# ============================================================

class SimpleDataLoader:
    """简化的数据加载器"""
    def __init__(self):
        try:
            from tgb.linkproppred.dataset import LinkPropPredDataset
            print("Loading TGB dataset: tgbl-wiki-v2")
            self.dataset = LinkPropPredDataset(name='tgbl-wiki-v2', root='datasets')
            self.data = self.dataset.full_data

            # 统计
            num_nodes = len(torch.unique(torch.cat([
                self.data['sources'], self.data['destinations']
            ])))
            print(f"Nodes: {num_nodes:,}, Edges: {len(self.data['sources']):,}")

        except Exception as e:
            print(f"TGB加载失败: {e}")
            print("使用模拟数据...")
            self._create_dummy_data()

    def _create_dummy_data(self):
        """创建模拟数据用于快速测试"""
        n_samples = 10000
        n_nodes = 1000

        self.data = {
            'sources': torch.randint(0, n_nodes, (n_samples,)),
            'destinations': torch.randint(0, n_nodes, (n_samples,)),
            'timestamps': torch.arange(n_samples).float(),
            'train_mask': torch.zeros(n_samples, dtype=torch.bool),
            'val_mask': torch.zeros(n_samples, dtype=torch.bool),
            'test_mask': torch.zeros(n_samples, dtype=torch.bool)
        }

        # 70% train, 15% val, 15% test
        self.data['train_mask'][:7000] = True
        self.data['val_mask'][7000:8500] = True
        self.data['test_mask'][8500:] = True

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

def train_model(model, train_data, epochs=20, lr=0.001, batch_size=200):
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

            # 简单的链接预测损失（余弦相似度）
            pos_score = F.cosine_similarity(emb[:len(emb)//2], emb[len(emb)//2:], dim=-1)

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
    print("RW2预实验开始")
    print("="*60 + "\n")

    # 1. 加载数据
    print("[1/4] 加载数据...")
    loader = SimpleDataLoader()
    train_data = loader.get_train_data()
    test_data = loader.get_test_data()

    num_nodes = int(max(
        train_data['sources'].max(),
        train_data['destinations'].max(),
        test_data['sources'].max(),
        test_data['destinations'].max()
    )) + 1

    print(f"节点数: {num_nodes}")

    # 2. 定义所有模型
    models = {
        'NPPCTNE': NPPCTNE(num_nodes, dim=128),
        'MoMent++': MoMentPP(num_nodes, dim=172),
        'THG-Mamba': THGMamba(num_nodes, dim=256),
        'TempMem-LLM': TempMemLLM(num_nodes, dim=172),
        'FreqTemporal': FreqTemporal(num_nodes, dim=172)
    }

    # 3. 训练和评估所有模型
    print("\n[2/4] 训练和评估模型...")
    results = {}

    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"训练 {name}")
        print('='*60)

        # 训练（简化：只训练20个epoch）
        model = train_model(model, train_data, epochs=20, lr=0.001)

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

    # 4. 生成报告
    print("\n[3/4] 生成性能对比报告...")

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

    # 5. 可视化
    print("\n[4/4] 生成可视化...")

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

    # 6. 总结
    print("\n" + "="*60)
    print("实验总结")
    print("="*60)

    passed = sum(1 for r in df_data if r['通过'] == '✅') - 1  # 减去baseline
    print(f"通过性能测试的方法: {passed}/4")
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
