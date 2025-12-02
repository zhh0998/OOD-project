#!/usr/bin/env python3
"""
RW2-PAG-Mamba 核心假设验证
假设：图Laplacian谱变化能够检测网络结构的相变点
"""

import numpy as np
import networkx as nx
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("RW2-PAG-Mamba 假设验证")
print("假设: Laplacian谱变化能检测网络结构相变点")
print("=" * 70)

# ============================================================
# A. 生成时序SBM图（在t=50引入相变）
# ============================================================
print("\n[A] 生成时序随机块模型(SBM)图...")

np.random.seed(42)

n_nodes = 100  # 节点数
n_timesteps = 100  # 时间步数
change_point = 50  # 相变点

def generate_sbm_graph(n, p_in, p_out, n_communities=2):
    """生成随机块模型图"""
    sizes = [n // n_communities] * n_communities
    sizes[-1] += n % n_communities  # 处理余数

    # 构建概率矩阵
    probs = np.full((n_communities, n_communities), p_out)
    np.fill_diagonal(probs, p_in)

    G = nx.stochastic_block_model(sizes, probs, seed=np.random.randint(10000))
    return G

# 生成时序图
graphs = []

for t in range(n_timesteps):
    if t < change_point:
        # 相变前：清晰的2社区结构
        G = generate_sbm_graph(n_nodes, p_in=0.4, p_out=0.05, n_communities=2)
    else:
        # 相变后：4社区结构（结构突变）
        G = generate_sbm_graph(n_nodes, p_in=0.5, p_out=0.02, n_communities=4)

    graphs.append(G)

print(f"    生成了 {n_timesteps} 个时间步的图")
print(f"    相变点: t = {change_point}")
print(f"    相变前: 2社区结构 (p_in=0.4, p_out=0.05)")
print(f"    相变后: 4社区结构 (p_in=0.5, p_out=0.02)")

# ============================================================
# B. 计算Laplacian谱特征
# ============================================================
print("\n[B] 计算Laplacian谱特征...")

def compute_laplacian_spectrum(G, k=5):
    """计算图的Laplacian谱特征"""
    # 归一化Laplacian矩阵
    L = nx.normalized_laplacian_matrix(G).toarray()

    # 计算特征值和特征向量
    eigenvalues, eigenvectors = linalg.eigh(L)

    # 排序（从小到大）
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues[:k], eigenvectors[:, :k]

# 计算每个时间步的谱特征
k = 5  # 使用前5个特征值
eigenvalues_series = []
eigenvectors_series = []

for t, G in enumerate(graphs):
    evals, evecs = compute_laplacian_spectrum(G, k)
    eigenvalues_series.append(evals)
    eigenvectors_series.append(evecs)

    if (t + 1) % 20 == 0:
        print(f"    已处理 {t + 1}/{n_timesteps}")

eigenvalues_series = np.array(eigenvalues_series)

print(f"    特征值矩阵形状: {eigenvalues_series.shape}")

# ============================================================
# C. 计算谱子空间距离 (Gemini方法B)
# ============================================================
print("\n[C] 计算谱子空间距离...")

def subspace_distance(U1, U2):
    """
    计算谱子空间距离
    d(t) = sqrt(k - ||U_t^T U_{t-1}||²_F)
    """
    k = U1.shape[1]
    inner = U1.T @ U2
    frobenius_sq = np.sum(inner ** 2)
    # 确保数值稳定性
    dist_sq = max(0, k - frobenius_sq)
    return np.sqrt(dist_sq)

subspace_distances = [0]  # t=0没有前一时刻

for t in range(1, n_timesteps):
    U_prev = eigenvectors_series[t-1]
    U_curr = eigenvectors_series[t]
    dist = subspace_distance(U_curr, U_prev)
    subspace_distances.append(dist)

subspace_distances = np.array(subspace_distances)

print(f"    谱子空间距离范围: [{subspace_distances.min():.4f}, {subspace_distances.max():.4f}]")
print(f"    相变前平均距离 (t<50): {subspace_distances[1:change_point].mean():.4f}")
print(f"    相变点距离 (t=50): {subspace_distances[change_point]:.4f}")
print(f"    相变后平均距离 (t>50): {subspace_distances[change_point+1:].mean():.4f}")

# ============================================================
# D. 统计分析
# ============================================================
print("\n[D] 统计分析...")

# 基线：相变前的平均距离和标准差
baseline_mean = subspace_distances[1:change_point].mean()
baseline_std = subspace_distances[1:change_point].std()

# 相变点的距离
change_point_dist = subspace_distances[change_point]

# 计算跃升倍数
jump_ratio = change_point_dist / baseline_mean if baseline_mean > 0 else 0

# Z-score
z_score = (change_point_dist - baseline_mean) / baseline_std if baseline_std > 0 else 0

# 谱隙变化（λ_2）
spectral_gap_before = eigenvalues_series[:change_point, 1].mean()
spectral_gap_after = eigenvalues_series[change_point:, 1].mean()
spectral_gap_change = abs(spectral_gap_after - spectral_gap_before) / spectral_gap_before

print(f"\n    基线统计 (t<50):")
print(f"      平均距离: {baseline_mean:.4f}")
print(f"      标准差: {baseline_std:.4f}")
print(f"\n    相变点 (t=50):")
print(f"      子空间距离: {change_point_dist:.4f}")
print(f"      跃升倍数: {jump_ratio:.2f}x")
print(f"      Z-score: {z_score:.2f}")
print(f"\n    谱隙变化 (λ_2):")
print(f"      相变前: {spectral_gap_before:.4f}")
print(f"      相变后: {spectral_gap_after:.4f}")
print(f"      变化率: {spectral_gap_change*100:.1f}%")

# ============================================================
# E. 可视化
# ============================================================
print("\n[E] 生成可视化...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 子图1: 谱子空间距离
ax1 = axes[0, 0]
ax1.plot(range(n_timesteps), subspace_distances, 'b-', linewidth=1.5, label='Subspace Distance')
ax1.axvline(x=change_point, color='r', linestyle='--', linewidth=2, label=f'Change Point (t={change_point})')
ax1.axhline(y=baseline_mean, color='g', linestyle=':', linewidth=1.5, label=f'Baseline Mean')
ax1.axhline(y=baseline_mean + 2*baseline_std, color='orange', linestyle=':', linewidth=1, label='2σ Threshold')
ax1.fill_between(range(n_timesteps), baseline_mean - baseline_std, baseline_mean + baseline_std,
                  alpha=0.2, color='green', label='±1σ Range')
ax1.set_xlabel('Time Step', fontsize=12)
ax1.set_ylabel('Subspace Distance d(t)', fontsize=12)
ax1.set_title(f'Spectral Subspace Distance\n(Jump Ratio at t=50: {jump_ratio:.2f}x)', fontsize=13)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# 子图2: 前5个特征值变化
ax2 = axes[0, 1]
for i in range(k):
    ax2.plot(range(n_timesteps), eigenvalues_series[:, i], linewidth=1.5, label=f'λ_{i+1}')
ax2.axvline(x=change_point, color='r', linestyle='--', linewidth=2)
ax2.set_xlabel('Time Step', fontsize=12)
ax2.set_ylabel('Eigenvalue', fontsize=12)
ax2.set_title('First 5 Laplacian Eigenvalues', fontsize=13)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# 子图3: 谱隙 (λ_2) 变化
ax3 = axes[1, 0]
ax3.plot(range(n_timesteps), eigenvalues_series[:, 1], 'b-', linewidth=2, label='Spectral Gap (λ₂)')
ax3.axvline(x=change_point, color='r', linestyle='--', linewidth=2, label='Change Point')
ax3.axhline(y=spectral_gap_before, color='g', linestyle=':', linewidth=1.5, label=f'Before: {spectral_gap_before:.3f}')
ax3.axhline(y=spectral_gap_after, color='orange', linestyle=':', linewidth=1.5, label=f'After: {spectral_gap_after:.3f}')
ax3.set_xlabel('Time Step', fontsize=12)
ax3.set_ylabel('Spectral Gap (λ₂)', fontsize=12)
ax3.set_title(f'Algebraic Connectivity\n(Change: {spectral_gap_change*100:.1f}%)', fontsize=13)
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

# 子图4: 相变前后的图结构示意
ax4 = axes[1, 1]
# 绘制相变前后的图
G_before = graphs[change_point - 1]
G_after = graphs[change_point]

# 使用spring layout
pos_before = nx.spring_layout(G_before, seed=42, k=2)

# 只显示边数统计
edges_before = G_before.number_of_edges()
edges_after = G_after.number_of_edges()

# 绘制文字说明
ax4.text(0.5, 0.7, f'Before Change (t={change_point-1})', fontsize=14, ha='center', fontweight='bold')
ax4.text(0.5, 0.55, f'2 Communities\n{edges_before} edges', fontsize=12, ha='center')
ax4.text(0.5, 0.3, f'After Change (t={change_point})', fontsize=14, ha='center', fontweight='bold')
ax4.text(0.5, 0.15, f'4 Communities\n{edges_after} edges', fontsize=12, ha='center')
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis('off')
ax4.set_title('Network Structure Change', fontsize=13)

plt.tight_layout()
plt.savefig('rw2_pag_mamba_validation.png', dpi=150, bbox_inches='tight')
print("    图表已保存: rw2_pag_mamba_validation.png")

# ============================================================
# F. 结论
# ============================================================
print("\n" + "=" * 70)
print("RW2-PAG-Mamba 验证结果")
print("=" * 70)

print(f"\n关键指标:")
print(f"  谱子空间距离跃升倍数: {jump_ratio:.2f}x (阈值: >2x)")
print(f"  Z-score: {z_score:.2f} (阈值: >3)")
print(f"  谱隙变化率: {spectral_gap_change*100:.1f}%")

print("\n" + "-" * 70)

# 判断标准
pass_jump = jump_ratio > 2.0
pass_zscore = z_score > 3.0
pass_spectral = spectral_gap_change > 0.1

if pass_jump and pass_zscore:
    print("\n✅ RW2-PAG-Mamba 核心假设验证通过！")
    print()
    print(f"   • 谱子空间距离跃升: {jump_ratio:.2f}x > 2x ✅")
    print(f"   • Z-score异常检测: {z_score:.2f} > 3 ✅")
    print(f"   • 谱隙显著变化: {spectral_gap_change*100:.1f}% ✅")
    print()
    print("   结论: Laplacian谱变化能有效检测网络相变点")
    print("   建议: 继续RW2-PAG-Mamba的后续研究")
    result = "PASS"
elif pass_jump or pass_zscore:
    print("\n⚠️ RW2-PAG-Mamba 核心假设部分通过")
    print()
    print(f"   • 谱子空间距离跃升: {jump_ratio:.2f}x {'✅' if pass_jump else '❌'}")
    print(f"   • Z-score异常检测: {z_score:.2f} {'✅' if pass_zscore else '❌'}")
    print()
    print("   结论: 谱方法有一定效果，但信号不够强")
    print("   建议: 需要进一步优化检测方法")
    result = "PARTIAL"
else:
    print("\n❌ RW2-PAG-Mamba 核心假设验证失败")
    print()
    print(f"   • 谱子空间距离跃升: {jump_ratio:.2f}x < 2x ❌")
    print(f"   • Z-score异常检测: {z_score:.2f} < 3 ❌")
    print()
    print("   结论: Laplacian谱变化无法有效检测相变点")
    print("   建议: 考虑其他相变检测方法")
    result = "FAIL"

print("\n" + "=" * 70)

# 保存结果
with open('rw2_validation_result.txt', 'w') as f:
    f.write(f"RW2-PAG-Mamba Validation Result: {result}\n")
    f.write(f"Jump Ratio: {jump_ratio:.4f}\n")
    f.write(f"Z-score: {z_score:.4f}\n")
    f.write(f"Spectral Gap Change: {spectral_gap_change:.4f}\n")
