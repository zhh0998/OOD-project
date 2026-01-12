"""
RW2预实验 - 多种子验证 (Multi-Seed Validation)
在OGB ogbl-collab数据集上运行TempMem-LLM和NPPCTNE baseline各5次
计算统计显著性和Cohen's d效应量

目标：验证TempMem-LLM相对于baseline的提升是否统计显著
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
# PyTorch兼容性修复
# ============================================================
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*60)
print("RW2 Multi-Seed Validation Experiment")
print("="*60)
print(f"Device: {device}")

# ============================================================
# 数据加载
# ============================================================

class RealDataLoader:
    """真实数据加载器"""
    def __init__(self):
        from ogb.linkproppred import LinkPropPredDataset
        print("Loading OGB ogbl-collab dataset...")
        dataset = LinkPropPredDataset(name='ogbl-collab', root='datasets')
        edge_split = dataset.get_edge_split()
        self._convert_ogb_format(edge_split)
        self.data_source = "OGB ogbl-collab"
        print(f"Loaded: {len(self.train_sources):,} train edges, {len(self.test_sources):,} test edges")

    def _convert_ogb_format(self, edge_split):
        train_edge = edge_split['train']['edge']
        train_year = edge_split['train']['year'].flatten()
        test_edge = edge_split['test']['edge']
        test_year = edge_split['test']['year'].flatten()

        self.train_sources = torch.from_numpy(train_edge[:, 0].astype(np.int64))
        self.train_destinations = torch.from_numpy(train_edge[:, 1].astype(np.int64))
        self.train_timestamps = torch.from_numpy(train_year.astype(np.float32))

        self.test_sources = torch.from_numpy(test_edge[:, 0].astype(np.int64))
        self.test_destinations = torch.from_numpy(test_edge[:, 1].astype(np.int64))
        self.test_timestamps = torch.from_numpy(test_year.astype(np.float32))

        all_nodes = torch.cat([self.train_sources, self.train_destinations,
                               self.test_sources, self.test_destinations])
        self.num_nodes = int(all_nodes.max()) + 1
        print(f"  Num nodes: {self.num_nodes:,}")

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
    def __init__(self, num_nodes, dim=172):
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
    """评估模型"""
    model.eval()

    sources = test_data['sources']
    destinations = test_data['destinations']
    timestamps = test_data['timestamps']

    n_samples = min(n_eval_samples, len(sources))

    mrrs = []

    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc="Evaluating", leave=False):
            end_idx = min(i + batch_size, n_samples)

            batch_src = sources[i:end_idx].to(device)
            batch_dst = destinations[i:end_idx].to(device)
            batch_time = timestamps[i:end_idx].to(device)

            emb = model(batch_src, batch_dst, batch_time)
            all_node_emb = model.node_emb.weight
            scores = torch.matmul(emb, all_node_emb.t())

            ranks = torch.argsort(torch.argsort(scores, dim=1, descending=True), dim=1) + 1
            target_ranks = ranks[torch.arange(len(batch_dst)), batch_dst]
            batch_mrr = (1.0 / target_ranks.float()).cpu().numpy()
            mrrs.extend(batch_mrr)

    return np.mean(mrrs)


def run_experiment_with_seed(model_class, num_nodes, train_data, test_data, seed):
    """使用指定种子运行单次实验"""
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # 创建模型
    model = model_class(num_nodes)

    # 训练
    model, train_time = train_model(model, train_data, epochs=1, batch_size=2048)

    # 评估
    mrr = evaluate_model(model, test_data)

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
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # 合并标准差
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))

    if pooled_std == 0:
        return 0.0

    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return d


def welch_t_test(group1, group2):
    """Welch's t-test (不假设方差相等)"""
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # t统计量
    se = np.sqrt(var1/n1 + var2/n2)
    if se == 0:
        return 0.0, 1.0

    t_stat = (mean1 - mean2) / se

    # 自由度 (Welch-Satterthwaite)
    df_num = (var1/n1 + var2/n2)**2
    df_den = (var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1)
    df = df_num / df_den if df_den > 0 else 1

    # p-value (two-tailed) 使用近似
    # 对于小样本，使用scipy更准确
    try:
        from scipy import stats
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    except ImportError:
        # 简化近似
        p_value = 2 * np.exp(-0.5 * t_stat**2) if abs(t_stat) < 5 else 0.0001

    return t_stat, p_value


# ============================================================
# 主程序
# ============================================================

def main():
    # 参数
    SEEDS = [42, 123, 456, 789, 2024]  # 5个随机种子

    print("\n[1/3] Loading data...")
    data_loader = RealDataLoader()
    train_data = data_loader.get_train_data()
    test_data = data_loader.get_test_data()
    num_nodes = data_loader.num_nodes

    results = {
        'NPPCTNE': {'mrrs': [], 'times': []},
        'TempMem-LLM': {'mrrs': [], 'times': []}
    }

    print(f"\n[2/3] Running experiments with {len(SEEDS)} seeds...")

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")

        # NPPCTNE
        print(f"  Running NPPCTNE...")
        mrr, train_time = run_experiment_with_seed(NPPCTNE, num_nodes, train_data, test_data, seed)
        results['NPPCTNE']['mrrs'].append(mrr)
        results['NPPCTNE']['times'].append(train_time)
        print(f"    MRR: {mrr:.4f}, Time: {train_time:.1f}s")

        # TempMem-LLM
        print(f"  Running TempMem-LLM...")
        mrr, train_time = run_experiment_with_seed(TempMemLLM, num_nodes, train_data, test_data, seed)
        results['TempMem-LLM']['mrrs'].append(mrr)
        results['TempMem-LLM']['times'].append(train_time)
        print(f"    MRR: {mrr:.4f}, Time: {train_time:.1f}s")

    # ============================================================
    # 统计分析
    # ============================================================
    print("\n[3/3] Statistical Analysis...")

    baseline_mrrs = results['NPPCTNE']['mrrs']
    tempmem_mrrs = results['TempMem-LLM']['mrrs']

    baseline_mean = np.mean(baseline_mrrs)
    baseline_std = np.std(baseline_mrrs, ddof=1)
    tempmem_mean = np.mean(tempmem_mrrs)
    tempmem_std = np.std(tempmem_mrrs, ddof=1)

    improvement_pct = ((tempmem_mean - baseline_mean) / baseline_mean) * 100

    # Cohen's d
    cohens_d = compute_cohens_d(tempmem_mrrs, baseline_mrrs)

    # Welch's t-test
    t_stat, p_value = welch_t_test(tempmem_mrrs, baseline_mrrs)

    # 判断是否通过
    passed_improvement = improvement_pct >= 3.0
    passed_significance = p_value < 0.05
    passed_effect_size = abs(cohens_d) >= 0.5
    overall_passed = passed_improvement and (passed_significance or passed_effect_size)

    # ============================================================
    # 输出结果
    # ============================================================
    print("\n" + "="*60)
    print("MULTI-SEED VALIDATION RESULTS")
    print("="*60)

    print(f"\nNPPCTNE (Baseline):")
    print(f"  MRRs: {[f'{m:.4f}' for m in baseline_mrrs]}")
    print(f"  Mean: {baseline_mean:.4f} ± {baseline_std:.4f}")

    print(f"\nTempMem-LLM:")
    print(f"  MRRs: {[f'{m:.4f}' for m in tempmem_mrrs]}")
    print(f"  Mean: {tempmem_mean:.4f} ± {tempmem_std:.4f}")

    print(f"\nStatistical Analysis:")
    print(f"  Improvement: {improvement_pct:+.2f}%")
    print(f"  Cohen's d: {cohens_d:.3f} ({'large' if abs(cohens_d)>=0.8 else 'medium' if abs(cohens_d)>=0.5 else 'small'})")
    print(f"  Welch's t-test: t={t_stat:.3f}, p={p_value:.4f}")

    print(f"\nSuccess Criteria:")
    print(f"  [{'✓' if passed_improvement else '✗'}] Improvement ≥ 3%: {improvement_pct:.2f}%")
    print(f"  [{'✓' if passed_significance else '✗'}] p-value < 0.05: {p_value:.4f}")
    print(f"  [{'✓' if passed_effect_size else '✗'}] Cohen's d ≥ 0.5: {cohens_d:.3f}")
    print(f"\n  Overall: {'✅ PASSED' if overall_passed else '❌ NOT PASSED'}")

    # ============================================================
    # 保存结果
    # ============================================================
    os.makedirs('results', exist_ok=True)

    validation_results = {
        'experiment': 'RW2 Multi-Seed Validation',
        'dataset': data_loader.data_source,
        'num_nodes': num_nodes,
        'seeds': SEEDS,
        'results': {
            'NPPCTNE': {
                'mrrs': baseline_mrrs,
                'mean_mrr': float(baseline_mean),
                'std_mrr': float(baseline_std),
                'times': results['NPPCTNE']['times']
            },
            'TempMem-LLM': {
                'mrrs': tempmem_mrrs,
                'mean_mrr': float(tempmem_mean),
                'std_mrr': float(tempmem_std),
                'times': results['TempMem-LLM']['times']
            }
        },
        'statistics': {
            'improvement_pct': float(improvement_pct),
            'cohens_d': float(cohens_d),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'passed_improvement_threshold': bool(passed_improvement),
            'passed_statistical_significance': bool(passed_significance),
            'passed_effect_size': bool(passed_effect_size),
            'overall_passed': bool(overall_passed)
        }
    }

    with open('results/multi_seed_validation.json', 'w') as f:
        json.dump(validation_results, f, indent=2)

    print(f"\nResults saved to: results/multi_seed_validation.json")
    print("="*60)

    return validation_results


if __name__ == '__main__':
    main()
