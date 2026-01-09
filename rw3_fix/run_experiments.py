"""
RW3 OOD检测实验 - 修复版完整实验脚本

修复3个关键Bug后运行完整实验:
1. OOD分数方向反转
2. L2归一化缺失
3. k-NN距离计算错误

Author: RW3 OOD Detection Project
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from quick_fix import FixedKNNDetector, MahalanobisDetector, LOFDetector, evaluate_ood
from data_loader import get_dataset


class EmbeddingExtractor:
    """Embedding提取器 - 使用RoBERTa-base"""

    def __init__(self, model_name: str = "roberta-base", device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[Embedding] 加载模型: {model_name}")
        print(f"[Embedding] 设备: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def extract(self, texts: List[str], batch_size: int = 32,
                max_length: int = 128, show_progress: bool = True) -> np.ndarray:
        """
        提取文本embeddings

        Args:
            texts: 文本列表
            batch_size: 批次大小
            max_length: 最大序列长度
            show_progress: 是否显示进度

        Returns:
            embeddings: shape=(n_samples, hidden_dim)
        """
        embeddings = []
        n_batches = (len(texts) + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)

                # 使用[CLS] token的embedding
                batch_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_emb)

                if show_progress and (i // batch_size + 1) % 10 == 0:
                    print(f"  进度: {i // batch_size + 1}/{n_batches} batches")

        return np.vstack(embeddings)


def run_single_experiment(dataset_name: str, extractor: EmbeddingExtractor,
                          k: int = 50, verbose: bool = True) -> Dict:
    """
    运行单个数据集的实验

    Args:
        dataset_name: 数据集名称
        extractor: Embedding提取器
        k: k近邻数量
        verbose: 是否打印详细信息

    Returns:
        实验结果字典
    """
    print(f"\n{'='*60}")
    print(f"数据集: {dataset_name.upper()}")
    print(f"{'='*60}")

    # 1. 加载数据
    print("\n[Step 1] 加载数据...")
    train_texts, test_texts, test_labels, test_intents, train_labels = get_dataset(dataset_name)
    test_labels = np.array(test_labels)

    # 2. 提取embeddings
    print("\n[Step 2] 提取embeddings...")
    print(f"  训练集: {len(train_texts)} 样本")
    train_emb = extractor.extract(train_texts, show_progress=verbose)

    print(f"  测试集: {len(test_texts)} 样本")
    test_emb = extractor.extract(test_texts, show_progress=verbose)

    print(f"  Embedding维度: {train_emb.shape[1]}")

    # 3. 运行各种检测器
    print("\n[Step 3] 运行OOD检测器...")
    results = {}

    # 3.1 Fixed k-NN (主方法)
    print("\n--- Fixed k-NN ---")
    knn = FixedKNNDetector(k=k, verbose=verbose)
    knn.fit(train_emb)
    knn_scores, knn_auroc = knn.score_with_fix(test_emb, test_labels)
    results['KNN'] = evaluate_ood(test_labels, knn_scores, auto_fix=False, verbose=False)
    print(f"  AUROC: {results['KNN']['auroc']:.4f} ({results['KNN']['auroc']*100:.2f}%)")
    print(f"  AUPR:  {results['KNN']['aupr']:.4f}")
    print(f"  FPR95: {results['KNN']['fpr95']:.4f}")

    # 3.2 Mahalanobis
    print("\n--- Mahalanobis ---")
    try:
        maha = MahalanobisDetector(verbose=verbose)
        maha.fit(train_emb)
        maha_scores, maha_auroc = maha.score_with_fix(test_emb, test_labels)
        results['Mahalanobis'] = evaluate_ood(test_labels, maha_scores, auto_fix=False, verbose=False)
        print(f"  AUROC: {results['Mahalanobis']['auroc']:.4f}")
    except Exception as e:
        print(f"  失败: {e}")
        results['Mahalanobis'] = {'auroc': 0, 'aupr': 0, 'fpr95': 1}

    # 3.3 LOF
    print("\n--- LOF ---")
    lof = LOFDetector(k=min(k, 20), verbose=verbose)
    lof.fit(train_emb)
    lof_scores, lof_auroc = lof.score_with_fix(test_emb, test_labels)
    results['LOF'] = evaluate_ood(test_labels, lof_scores, auto_fix=False, verbose=False)
    print(f"  AUROC: {results['LOF']['auroc']:.4f}")

    # 3.4 不同k值的k-NN
    print("\n--- k-NN (不同k值) ---")
    for k_val in [10, 30, 50, 100]:
        knn_k = FixedKNNDetector(k=k_val, verbose=False)
        knn_k.fit(train_emb)
        scores_k, auroc_k = knn_k.score_with_fix(test_emb, test_labels)
        results[f'KNN_k{k_val}'] = evaluate_ood(test_labels, scores_k, auto_fix=False, verbose=False)
        print(f"  k={k_val}: AUROC={results[f'KNN_k{k_val}']['auroc']:.4f}")

    # 汇总
    print(f"\n{'='*60}")
    print(f"{dataset_name.upper()} 结果汇总")
    print(f"{'='*60}")
    print(f"{'方法':<15} {'AUROC':<10} {'AUPR':<10} {'FPR@95':<10}")
    print(f"{'-'*45}")
    for method in ['KNN', 'Mahalanobis', 'LOF']:
        r = results[method]
        print(f"{method:<15} {r['auroc']:.4f}     {r['aupr']:.4f}     {r['fpr95']:.4f}")

    return {
        'dataset': dataset_name,
        'n_train': len(train_texts),
        'n_test': len(test_texts),
        'n_ood': int(test_labels.sum()),
        'n_id': int(len(test_labels) - test_labels.sum()),
        'results': results
    }


def run_all_experiments(datasets: List[str] = None, k: int = 50,
                        output_dir: Path = None) -> Dict:
    """
    运行所有数据集的实验

    Args:
        datasets: 数据集列表
        k: k近邻数量
        output_dir: 输出目录

    Returns:
        所有实验结果
    """
    if datasets is None:
        datasets = ['clinc150', 'banking77']

    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化Embedding提取器
    extractor = EmbeddingExtractor()

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'k': k,
            'model': extractor.model_name,
            'datasets': datasets
        },
        'experiments': {}
    }

    # 运行各数据集实验
    for dataset in datasets:
        try:
            result = run_single_experiment(dataset, extractor, k=k)
            all_results['experiments'][dataset] = result
        except Exception as e:
            print(f"\n[错误] {dataset} 实验失败: {e}")
            import traceback
            traceback.print_exc()
            all_results['experiments'][dataset] = {'error': str(e)}

    # 保存结果
    output_file = output_dir / "fixed_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n结果已保存: {output_file}")

    # 生成论文级表格
    generate_paper_table(all_results, output_dir)

    return all_results


def generate_paper_table(results: Dict, output_dir: Path):
    """生成论文级结果表格"""

    print("\n" + "="*70)
    print("论文级结果表格")
    print("="*70)

    # 表1: 主要结果对比
    print("\n表1: 修复版OOD检测结果 (AUROC %)")
    print("-"*70)
    print(f"{'Dataset':<15} {'KNN':<12} {'Mahalanobis':<12} {'LOF':<12} {'Best':<12}")
    print("-"*70)

    for dataset, exp in results.get('experiments', {}).items():
        if 'error' in exp:
            continue

        r = exp['results']
        knn = r.get('KNN', {}).get('auroc', 0) * 100
        maha = r.get('Mahalanobis', {}).get('auroc', 0) * 100
        lof = r.get('LOF', {}).get('auroc', 0) * 100
        best = max(knn, maha, lof)

        print(f"{dataset:<15} {knn:>10.2f}% {maha:>10.2f}% {lof:>10.2f}% {best:>10.2f}%")

    print("-"*70)

    # 表2: v2 vs v3对比
    print("\n表2: Bug修复前后对比")
    print("-"*70)
    print(f"{'Dataset':<15} {'v2 (有Bug)':<15} {'v3 (修复后)':<15} {'改善':<12}")
    print("-"*70)

    # v2结果（从问题描述中）
    v2_results = {
        'clinc150': 72.86,
        'banking77': 86.95
    }

    for dataset, exp in results.get('experiments', {}).items():
        if 'error' in exp:
            continue

        v2 = v2_results.get(dataset, 'N/A')
        v3 = exp['results'].get('KNN', {}).get('auroc', 0) * 100
        if isinstance(v2, float):
            improve = v3 - v2
            print(f"{dataset:<15} {v2:>12.2f}% {v3:>12.2f}% {improve:>+10.2f}%")
        else:
            print(f"{dataset:<15} {'N/A':>12} {v3:>12.2f}% {'N/A':>10}")

    print("-"*70)

    # 表3: k值敏感性分析
    print("\n表3: k值敏感性分析 (AUROC %)")
    print("-"*70)
    print(f"{'Dataset':<15} {'k=10':<12} {'k=30':<12} {'k=50':<12} {'k=100':<12}")
    print("-"*70)

    for dataset, exp in results.get('experiments', {}).items():
        if 'error' in exp:
            continue

        r = exp['results']
        k10 = r.get('KNN_k10', {}).get('auroc', 0) * 100
        k30 = r.get('KNN_k30', {}).get('auroc', 0) * 100
        k50 = r.get('KNN_k50', {}).get('auroc', 0) * 100
        k100 = r.get('KNN_k100', {}).get('auroc', 0) * 100

        print(f"{dataset:<15} {k10:>10.2f}% {k30:>10.2f}% {k50:>10.2f}% {k100:>10.2f}%")

    print("-"*70)

    # 保存LaTeX表格
    latex_file = output_dir / "paper_table.tex"
    with open(latex_file, 'w') as f:
        f.write("% 论文级结果表格\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{OOD Detection Results (AUROC \\%)}\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("Dataset & KNN & Mahalanobis & LOF \\\\\n")
        f.write("\\midrule\n")

        for dataset, exp in results.get('experiments', {}).items():
            if 'error' in exp:
                continue
            r = exp['results']
            knn = r.get('KNN', {}).get('auroc', 0) * 100
            maha = r.get('Mahalanobis', {}).get('auroc', 0) * 100
            lof = r.get('LOF', {}).get('auroc', 0) * 100
            f.write(f"{dataset.upper()} & {knn:.2f} & {maha:.2f} & {lof:.2f} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"\nLaTeX表格已保存: {latex_file}")


def main():
    parser = argparse.ArgumentParser(description="RW3 OOD Detection Experiments (Fixed Version)")
    parser.add_argument("--datasets", nargs="+", default=["clinc150", "banking77"],
                        help="Datasets to run experiments on")
    parser.add_argument("--k", type=int, default=50, help="k for k-NN")
    parser.add_argument("--output", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else None

    print("="*70)
    print("RW3 OOD Detection Experiments - Fixed Version")
    print("="*70)
    print(f"数据集: {args.datasets}")
    print(f"k值: {args.k}")
    print("="*70)

    results = run_all_experiments(
        datasets=args.datasets,
        k=args.k,
        output_dir=output_dir
    )

    print("\n" + "="*70)
    print("实验完成!")
    print("="*70)


if __name__ == "__main__":
    main()
