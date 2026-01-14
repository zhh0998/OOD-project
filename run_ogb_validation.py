"""
RW2 主线A - OGB多数据集验证 (Multi-Dataset Validation)
在ogbl-citation2和ogbl-ddi上验证TempMem-LLM（各3次种子）

目标：验证TempMem-LLM在不同数据集上的泛化能力
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
# PyTorch兼容性修复 (for PyTorch 2.6+)
# ============================================================
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*60)
print("RW2 主线A - OGB Multi-Dataset Validation")
print("="*60)
print(f"Device: {device}")

# ============================================================
# 数据加载器
# ============================================================

class OGBDataLoader:
    """OGB数据集加载器 - 支持多种数据集"""

    def __init__(self, dataset_name):
        from ogb.linkproppred import LinkPropPredDataset

        self.dataset_name = dataset_name
        print(f"\nLoading {dataset_name}...")

        dataset = LinkPropPredDataset(name=dataset_name, root='dataset/')
        self.graph = dataset[0]
        edge_split = dataset.get_edge_split()

        self._convert_format(edge_split)
        self.data_source = dataset_name
        print(f"  Nodes: {self.num_nodes:,}")
        print(f"  Train edges: {len(self.train_sources):,}")
        print(f"  Test edges: {len(self.test_sources):,}")

    def _convert_format(self, edge_split):
        """转换OGB格式为统一格式"""
        # 处理不同数据集的格式差异
        if 'edge' in edge_split['train']:
            # ogbl-ddi, ogbl-collab 格式
            train_edge = edge_split['train']['edge']
            test_edge = edge_split['test']['edge']
            train_src = train_edge[:, 0]
            train_dst = train_edge[:, 1]
            test_src = test_edge[:, 0]
            test_dst = test_edge[:, 1]
        elif 'source_node' in edge_split['train']:
            # ogbl-citation2 格式
            train_src = edge_split['train']['source_node']
            train_dst = edge_split['train']['target_node']
            test_src = edge_split['test']['source_node']
            test_dst = edge_split['test']['target_node']
        else:
            raise ValueError(f"Unknown edge split format: {edge_split['train'].keys()}")

        # 提取边
        self.train_sources = torch.from_numpy(train_src.astype(np.int64))
        self.train_destinations = torch.from_numpy(train_dst.astype(np.int64))

        self.test_sources = torch.from_numpy(test_src.astype(np.int64))
        self.test_destinations = torch.from_numpy(test_dst.astype(np.int64))

        # 时间戳：如果有年份信息使用它，否则使用序号
        if 'year' in edge_split['train']:
            train_year = edge_split['train']['year'].flatten()
            test_year = edge_split['test']['year'].flatten()
            self.train_timestamps = torch.from_numpy(train_year.astype(np.float32))
            self.test_timestamps = torch.from_numpy(test_year.astype(np.float32))
        else:
            # 对于没有时间戳的数据集，使用边的序号作为时间
            self.train_timestamps = torch.arange(len(self.train_sources), dtype=torch.float32)
            self.test_timestamps = torch.arange(len(self.test_sources), dtype=torch.float32) + len(self.train_sources)

        # 计算节点数
        all_nodes = torch.cat([self.train_sources, self.train_destinations,
                               self.test_sources, self.test_destinations])
        self.num_nodes = int(all_nodes.max()) + 1

    def get_train_data(self):
        return {
            'sources': self.train_sources,
            'destinations': self.train_destinations,
            'timestamps': self.train_timestamps
        }

    def get_test_data(self):
        return {
            'sources': self.test_sources,
            'destinations': self.test_destinations,
            'timestamps': self.test_timestamps
        }


# ============================================================
# 模型定义
# ============================================================

class TimeEncoder(nn.Module):
    """时间编码器 - 使用正弦位置编码"""
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
    """NPPCTNE Baseline - Neural Point Process for Continuous-Time Network Embedding"""
    def __init__(self, num_nodes, dim=128):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, dim)
        self.time_enc = TimeEncoder(dim)
        self.temporal = nn.GRU(dim, dim, batch_first=True)
        self.fc = nn.Linear(dim, dim)

    def forward(self, src, dst, time):
        src_emb = self.node_emb(src)
        dst_emb = self.node_emb(dst)
        time_emb = self.time_enc(time)[:, :self.node_emb.embedding_dim]

        x = (src_emb + dst_emb + time_emb).unsqueeze(1)
        h, _ = self.temporal(x)
        return self.fc(h.squeeze(1))


class TempMemLLM(nn.Module):
    """TempMem-LLM - LLM增强时态记忆网络 (RW2最优模型)"""
    def __init__(self, num_nodes, dim=128):
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


# ============================================================
# 训练与评估
# ============================================================

def train_model(model, train_data, epochs=1, lr=0.001, batch_size=2048):
    """训练模型"""
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
        pbar = tqdm(range(n_batches), desc=f"Training", leave=False)

        for i in pbar:
            start_idx = i * batch_size
            end_idx = min((i+1) * batch_size, n_samples)

            batch_src = sources[start_idx:end_idx].to(device)
            batch_dst = destinations[start_idx:end_idx].to(device)
            batch_time = timestamps[start_idx:end_idx].to(device)

            emb = model(batch_src, batch_dst, batch_time)
            dst_emb = model.node_emb(batch_dst)

            pos_score = F.cosine_similarity(emb, dst_emb, dim=-1)

            neg_dst = torch.randint(0, model.node_emb.num_embeddings, (len(batch_src),), device=device)
            neg_emb = model.node_emb(neg_dst)
            neg_score = F.cosine_similarity(emb, neg_emb, dim=-1)

            loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    train_time = time.time() - start_time
    return model, train_time


def evaluate_model(model, test_data, batch_size=200, n_eval_samples=1000):
    """评估模型 - 计算MRR"""
    model.eval()

    sources = test_data['sources']
    destinations = test_data['destinations']
    timestamps = test_data['timestamps']

    n_samples = min(n_eval_samples, len(sources))

    # 为大规模数据集限制候选节点数
    num_nodes = model.node_emb.num_embeddings
    if num_nodes > 50000:
        # 对于大图，只对部分节点计算排名
        sample_nodes = min(10000, num_nodes)
    else:
        sample_nodes = num_nodes

    mrrs = []

    with torch.no_grad():
        # 预先计算所有节点嵌入（采样）
        if sample_nodes < num_nodes:
            candidate_nodes = torch.randperm(num_nodes)[:sample_nodes]
            all_node_emb = model.node_emb(candidate_nodes.to(device))
            node_mapping = {int(n): i for i, n in enumerate(candidate_nodes)}
        else:
            all_node_emb = model.node_emb.weight
            node_mapping = None

        for i in tqdm(range(0, n_samples, batch_size), desc="Evaluating", leave=False):
            end_idx = min(i + batch_size, n_samples)

            batch_src = sources[i:end_idx].to(device)
            batch_dst = destinations[i:end_idx].to(device)
            batch_time = timestamps[i:end_idx].to(device)

            emb = model(batch_src, batch_dst, batch_time)
            scores = torch.matmul(emb, all_node_emb.t())

            for j, dst in enumerate(batch_dst):
                dst_idx = dst.item()
                if node_mapping is not None:
                    if dst_idx not in node_mapping:
                        # 目标节点不在采样中，跳过
                        continue
                    dst_idx = node_mapping[dst_idx]

                # 计算排名
                score = scores[j, dst_idx].item()
                rank = (scores[j] > score).sum().item() + 1
                mrrs.append(1.0 / rank)

    return np.mean(mrrs) if mrrs else 0.0


def run_experiment_with_seed(model_class, num_nodes, train_data, test_data, seed,
                            dim=128, batch_size=2048, n_eval=1000):
    """使用指定种子运行单次实验"""
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # 创建模型
    model = model_class(num_nodes, dim=dim)

    # 训练
    model, train_time = train_model(model, train_data, epochs=1, batch_size=batch_size)

    # 评估
    mrr = evaluate_model(model, test_data, n_eval_samples=n_eval)

    # 清理显存
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return mrr, train_time


# ============================================================
# 统计分析
# ============================================================

def compute_cohens_d(group1, group2):
    """计算Cohen's d效应量"""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0

    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def welch_t_test(group1, group2):
    """Welch's t-test"""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0, 1.0

    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    se = np.sqrt(var1/n1 + var2/n2)
    if se == 0:
        return 0.0, 1.0

    t_stat = (mean1 - mean2) / se

    df_num = (var1/n1 + var2/n2)**2
    df_den = (var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1)
    df = df_num / df_den if df_den > 0 else 1

    try:
        from scipy import stats
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    except ImportError:
        p_value = 2 * np.exp(-0.5 * t_stat**2) if abs(t_stat) < 5 else 0.0001

    return t_stat, p_value


# ============================================================
# 主程序
# ============================================================

def main():
    # 配置
    DATASETS = ['ogbl-ddi', 'ogbl-citation2']
    SEEDS = [42, 123, 456]  # 3个随机种子

    all_results = {
        'experiment': 'RW2 主线A - OGB Multi-Dataset Validation',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'seeds': SEEDS,
        'datasets': {}
    }

    for dataset_name in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print("="*60)

        try:
            # 加载数据
            data_loader = OGBDataLoader(dataset_name)
            train_data = data_loader.get_train_data()
            test_data = data_loader.get_test_data()
            num_nodes = data_loader.num_nodes

            # 根据数据集大小调整维度和参数
            if num_nodes > 1000000:
                dim = 32  # 超大数据集用更小维度
                batch_size = 4096
                n_eval = 500
            elif num_nodes > 100000:
                dim = 64  # 大数据集用较小维度
                batch_size = 2048
                n_eval = 1000
            else:
                dim = 128
                batch_size = 2048
                n_eval = 1000

            print(f"\nUsing embedding dim: {dim}, batch_size: {batch_size}, n_eval: {n_eval}")

            results = {
                'NPPCTNE': {'mrrs': [], 'times': []},
                'TempMem-LLM': {'mrrs': [], 'times': []}
            }

            for seed in SEEDS:
                print(f"\n--- Seed {seed} ---")

                # NPPCTNE
                print(f"  Running NPPCTNE...")
                mrr, train_time = run_experiment_with_seed(
                    NPPCTNE, num_nodes, train_data, test_data, seed,
                    dim=dim, batch_size=batch_size, n_eval=n_eval
                )
                results['NPPCTNE']['mrrs'].append(float(mrr))
                results['NPPCTNE']['times'].append(float(train_time))
                print(f"    MRR: {mrr:.4f}, Time: {train_time:.1f}s")

                # TempMem-LLM
                print(f"  Running TempMem-LLM...")
                mrr, train_time = run_experiment_with_seed(
                    TempMemLLM, num_nodes, train_data, test_data, seed,
                    dim=dim, batch_size=batch_size, n_eval=n_eval
                )
                results['TempMem-LLM']['mrrs'].append(float(mrr))
                results['TempMem-LLM']['times'].append(float(train_time))
                print(f"    MRR: {mrr:.4f}, Time: {train_time:.1f}s")

            # 统计分析
            baseline_mrrs = results['NPPCTNE']['mrrs']
            tempmem_mrrs = results['TempMem-LLM']['mrrs']

            baseline_mean = np.mean(baseline_mrrs)
            baseline_std = np.std(baseline_mrrs, ddof=1) if len(baseline_mrrs) > 1 else 0
            tempmem_mean = np.mean(tempmem_mrrs)
            tempmem_std = np.std(tempmem_mrrs, ddof=1) if len(tempmem_mrrs) > 1 else 0

            improvement_pct = ((tempmem_mean - baseline_mean) / baseline_mean) * 100 if baseline_mean > 0 else 0
            cohens_d = compute_cohens_d(tempmem_mrrs, baseline_mrrs)
            t_stat, p_value = welch_t_test(tempmem_mrrs, baseline_mrrs)

            # 保存结果
            all_results['datasets'][dataset_name] = {
                'num_nodes': num_nodes,
                'num_train_edges': len(train_data['sources']),
                'num_test_edges': len(test_data['sources']),
                'embedding_dim': dim,
                'results': results,
                'statistics': {
                    'baseline_mean': float(baseline_mean),
                    'baseline_std': float(baseline_std),
                    'tempmem_mean': float(tempmem_mean),
                    'tempmem_std': float(tempmem_std),
                    'improvement_pct': float(improvement_pct),
                    'cohens_d': float(cohens_d),
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'passed_3pct': bool(improvement_pct >= 3.0),
                    'passed_significance': bool(p_value < 0.05)
                }
            }

            # 打印结果
            print(f"\n--- {dataset_name} Results ---")
            print(f"NPPCTNE:     {baseline_mean:.4f} ± {baseline_std:.4f}")
            print(f"TempMem-LLM: {tempmem_mean:.4f} ± {tempmem_std:.4f}")
            print(f"Improvement: {improvement_pct:+.2f}%")
            print(f"Cohen's d:   {cohens_d:.3f}")
            print(f"p-value:     {p_value:.4f}")

        except Exception as e:
            print(f"ERROR on {dataset_name}: {e}")
            all_results['datasets'][dataset_name] = {'error': str(e)}

    # ============================================================
    # 汇总报告
    # ============================================================
    print("\n" + "="*60)
    print("MULTI-DATASET VALIDATION SUMMARY")
    print("="*60)

    for ds_name, ds_results in all_results['datasets'].items():
        if 'error' in ds_results:
            print(f"\n{ds_name}: ERROR - {ds_results['error']}")
            continue

        stats = ds_results['statistics']
        status = "✅ PASS" if stats['passed_3pct'] and (stats['passed_significance'] or stats['cohens_d'] >= 0.5) else "❌ FAIL"
        print(f"\n{ds_name}:")
        print(f"  NPPCTNE:     {stats['baseline_mean']:.4f}")
        print(f"  TempMem-LLM: {stats['tempmem_mean']:.4f}")
        print(f"  Improvement: {stats['improvement_pct']:+.2f}% {status}")

    # 保存结果
    os.makedirs('results', exist_ok=True)
    output_file = 'results/ogb_multi_dataset_validation.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")
    print("="*60)

    return all_results


if __name__ == '__main__':
    main()
