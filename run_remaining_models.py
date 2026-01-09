"""
RW2预实验 - 运行剩余4个模型
跳过已完成的NPPCTNE baseline，顺序执行：
1. MoMent++
2. THG-Mamba
3. TempMem-LLM
4. FreqTemporal

每个模型单独保存结果，最后生成综合对比报告
预计运行时间: ~12-15分钟
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
import os
import gc
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
print("运行剩余4个模型实验")
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
            raise RuntimeError("无法加载真实数据集！")

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


class MoMentPP(nn.Module):
    """MoMent++ - 多模态融合 (NeurIPS 2024 DTGB-inspired)"""
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
    """THG-Mamba - 时态异构图 + Mamba架构"""
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
    """TempMem-LLM - LLM增强时态记忆"""
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
    """FreqTemporal - 频域增强时态建模"""
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

def train_model(model, train_data, epochs=1, lr=0.001, batch_size=2048):
    """训练模型 - 优化版"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    sources = train_data['sources']
    destinations = train_data['destinations']
    timestamps = train_data['timestamps']

    n_samples = len(sources)
    n_batches = (n_samples + batch_size - 1) // batch_size

    start_time = time.time()

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

            # 获取目标节点嵌入
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
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    train_time = time.time() - start_time
    return model, train_time


def evaluate_model(model, test_data, batch_size=200, n_eval_samples=1000):
    """评估模型（快速MRR）"""
    model.eval()

    sources = test_data['sources']
    destinations = test_data['destinations']
    timestamps = test_data['timestamps']

    n_samples = min(n_eval_samples, len(sources))

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
    return np.mean(mrrs), eval_time


def cleanup_memory():
    """清理内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 主流程
# ============================================================

def main():
    print("\n" + "="*60)
    print("RW2预实验 - 运行剩余4个模型")
    print("="*60 + "\n")

    total_start = time.time()

    # 1. 加载数据
    print("[1/3] 加载真实数据...")
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

    # 2. 定义待运行的模型（跳过NPPCTNE）
    models_to_run = [
        ('MoMent++', lambda: MoMentPP(num_nodes, dim=172)),
        ('THG-Mamba', lambda: THGMamba(num_nodes, dim=256)),
        ('TempMem-LLM', lambda: TempMemLLM(num_nodes, dim=172)),
        ('FreqTemporal', lambda: FreqTemporal(num_nodes, dim=172)),
    ]

    # 加载baseline结果
    baseline_result = None
    if os.path.exists('results/single_model_test.json'):
        with open('results/single_model_test.json', 'r') as f:
            baseline_result = json.load(f)
        print(f"\n已加载Baseline结果: NPPCTNE MRR = {baseline_result['mrr']:.4f}")

    # 3. 顺序运行每个模型
    print("\n[2/3] 顺序运行4个模型...")
    all_results = {}

    # 先添加baseline结果
    if baseline_result:
        all_results['NPPCTNE'] = baseline_result

    for idx, (model_name, model_fn) in enumerate(models_to_run):
        print(f"\n{'='*60}")
        print(f"模型 {idx+1}/4: {model_name}")
        print('='*60)

        try:
            # 创建模型
            model = model_fn()
            n_params = sum(p.numel() for p in model.parameters())
            print(f"参数量: {n_params:,}")

            # 训练
            print(f"\n训练 {model_name} (1 epoch)...")
            model, train_time = train_model(model, train_data, epochs=1, lr=0.001, batch_size=2048)

            # 评估
            print(f"\n评估 {model_name}...")
            mrr, eval_time = evaluate_model(model, test_data, n_eval_samples=1000)

            total_time = train_time + eval_time

            # 保存模型
            model_filename = model_name.replace('+', 'P').replace('-', '_')
            torch.save(model.state_dict(), f'checkpoints/{model_filename}_test.pth')

            # 保存单模型结果
            result = {
                'model': model_name,
                'epochs': 1,
                'mrr': mrr,
                'train_time_seconds': train_time,
                'eval_time_seconds': eval_time,
                'total_time_seconds': total_time,
                'data_source': loader.data_source,
                'num_nodes': num_nodes,
                'device': str(device)
            }

            with open(f'results/model_{model_filename}_test.json', 'w') as f:
                json.dump(result, f, indent=2)

            all_results[model_name] = result

            print(f"\n{model_name} 完成!")
            print(f"  - MRR: {mrr:.4f}")
            print(f"  - 训练时间: {train_time:.1f}s")
            print(f"  - 总耗时: {total_time:.1f}s")

            # 与baseline对比
            if baseline_result:
                improvement = (mrr - baseline_result['mrr']) / baseline_result['mrr'] * 100
                passed = improvement >= 3.0
                print(f"  - vs Baseline: {improvement:+.2f}% {'PASS' if passed else 'FAIL'}")

            # 清理内存
            del model
            cleanup_memory()

        except Exception as e:
            print(f"\n{model_name} 运行失败: {e}")
            all_results[model_name] = {'model': model_name, 'error': str(e)}
            cleanup_memory()
            continue

    # 4. 生成综合对比报告
    print("\n[3/3] 生成综合对比报告...")

    baseline_mrr = baseline_result['mrr'] if baseline_result else 0

    comparison = {
        'experiment': 'RW2 Temporal Network Embedding Pre-experiment',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'data_source': loader.data_source,
        'num_nodes': num_nodes,
        'baseline_mrr': baseline_mrr,
        'models': [],
        'summary': {}
    }

    passed_count = 0
    for name, result in all_results.items():
        if 'error' in result:
            model_info = {
                'name': name,
                'status': 'FAILED',
                'error': result['error']
            }
        else:
            mrr = result['mrr']
            improvement = (mrr - baseline_mrr) / baseline_mrr * 100 if baseline_mrr > 0 else 0
            passed = improvement >= 3.0 or name == 'NPPCTNE'

            if passed and name != 'NPPCTNE':
                passed_count += 1

            model_info = {
                'name': name,
                'status': 'SUCCESS',
                'mrr': mrr,
                'improvement_pct': improvement,
                'passed_3pct_threshold': passed,
                'train_time_seconds': result.get('train_time_seconds', 0),
                'total_time_seconds': result.get('total_time_seconds', 0)
            }

        comparison['models'].append(model_info)

    comparison['summary'] = {
        'total_models': len(all_results),
        'successful_models': sum(1 for r in all_results.values() if 'error' not in r),
        'passed_3pct_threshold': passed_count,
        'best_model': max(
            [(k, v['mrr']) for k, v in all_results.items() if 'mrr' in v],
            key=lambda x: x[1],
            default=('N/A', 0)
        )[0],
        'total_experiment_time_seconds': time.time() - total_start
    }

    with open('results/all_models_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)

    # 打印总结
    total_time = time.time() - total_start

    print("\n" + "="*60)
    print("实验完成!")
    print("="*60)

    print("\n模型性能对比:")
    print(f"{'模型':<15} {'MRR':<10} {'vs Baseline':<15} {'通过3%?':<10}")
    print("-" * 50)

    for model_info in comparison['models']:
        name = model_info['name']
        if model_info['status'] == 'SUCCESS':
            mrr = model_info['mrr']
            imp = model_info['improvement_pct']
            passed = 'PASS' if model_info['passed_3pct_threshold'] else 'FAIL'
            if name == 'NPPCTNE':
                print(f"{name:<15} {mrr:.4f}     {'baseline':<15} {'---':<10}")
            else:
                print(f"{name:<15} {mrr:.4f}     {imp:+.2f}%          {passed:<10}")
        else:
            print(f"{name:<15} {'ERROR':<10} {'N/A':<15} {'N/A':<10}")

    print(f"\n总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"通过3%阈值的模型: {passed_count}/4")

    print("\n生成的文件:")
    print("  - results/all_models_comparison.json")
    for name in all_results:
        if 'error' not in all_results[name]:
            model_filename = name.replace('+', 'P').replace('-', '_')
            print(f"  - results/model_{model_filename}_test.json")
            print(f"  - checkpoints/{model_filename}_test.pth")

    return comparison


if __name__ == '__main__':
    results = main()
