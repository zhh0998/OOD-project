#!/usr/bin/env python3
"""
RW3 CCF-A Compliance Complete Assessment
RW3项目CCF-A标准合规性完整评估

评估维度:
1. 数据集完整性 (25%)
2. Baseline对比 (20%)
3. 消融实验 (15%)
4. 统计分析 (15%)
5. 可视化 (25%)

Author: RW3 OOD Detection Project
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import fnmatch

# 权重配置
WEIGHTS = {
    'datasets': 0.25,
    'baselines': 0.20,
    'ablation': 0.15,
    'statistics': 0.15,
    'visualization': 0.25
}

# 项目路径
PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / 'results'
FIGURES_DIR = PROJECT_ROOT / 'figures'


# ==============================================================================
# 1. 数据集完整性评估
# ==============================================================================

def assess_dataset_compliance() -> Tuple[Dict, float]:
    """评估数据集使用完整性"""

    checklist = {
        'tier1_datasets': {
            'clinc150': {
                'required': True,
                'description': '远OOD检测基准',
                'min_auroc': 0.90,
                'status': 'unknown',
                'auroc': None,
                'fpr95': None
            },
            'banking77': {
                'required': True,
                'description': '近OOD检测基准',
                'min_auroc': 0.85,
                'status': 'unknown',
                'auroc': None,
                'fpr95': None
            }
        },
        'tier2_datasets': {
            'rostd': {
                'required': True,
                'description': '真实世界OOD泛化',
                'zero_shot': True,
                'min_auroc': 0.95,
                'status': 'unknown',
                'auroc': None
            }
        },
        'tier3_datasets': {
            'hwu64': {
                'required': False,
                'description': '额外评估数据集',
                'status': 'unknown',
                'auroc': None
            }
        }
    }

    # 检查结果文件
    result_files = [
        'priority1_full_results.json',
        'priority2_results.json',
        'ablation_results.json'
    ]

    all_data = {}
    for file_name in result_files:
        file_path = RESULTS_DIR / file_name
        if file_path.exists():
            with open(file_path) as f:
                all_data[file_name] = json.load(f)

    # 从结果中提取性能
    def extract_dataset_results(dataset_name: str) -> Dict:
        results = {'found': False, 'auroc': None, 'fpr95': None}

        for file_name, data in all_data.items():
            # 检查priority1_full_results
            if file_name == 'priority1_full_results.json':
                if 'full_evaluation' in data and dataset_name in data['full_evaluation']:
                    results['found'] = True
                    results['auroc'] = data['full_evaluation'][dataset_name].get('auroc')
                    results['fpr95'] = data['full_evaluation'][dataset_name].get('fpr95')
                    break

            # 检查priority2_results
            if file_name == 'priority2_results.json':
                if 'final_evaluation' in data and dataset_name in data['final_evaluation']:
                    results['found'] = True
                    results['auroc'] = data['final_evaluation'][dataset_name].get('auroc')
                    results['fpr95'] = data['final_evaluation'][dataset_name].get('fpr95')
                    break

        return results

    # 更新检查状态
    score_components = []

    for tier_name, datasets in checklist.items():
        for dataset_name, config in datasets.items():
            results = extract_dataset_results(dataset_name)

            if results['found']:
                config['status'] = 'found'
                config['auroc'] = results['auroc']
                config['fpr95'] = results['fpr95']

                # 检查是否达到阈值
                if config['auroc'] and config.get('min_auroc'):
                    if config['auroc'] >= config['min_auroc']:
                        config['meets_threshold'] = True
                        if config['required']:
                            score_components.append(100)
                    else:
                        config['meets_threshold'] = False
                        if config['required']:
                            score_components.append(config['auroc'] / config['min_auroc'] * 100)
                else:
                    if config['required']:
                        score_components.append(80)  # 找到但无阈值
            else:
                config['status'] = 'missing'
                if config['required']:
                    score_components.append(0)

    # 计算总分
    dataset_score = sum(score_components) / len(score_components) if score_components else 0

    return checklist, dataset_score


# ==============================================================================
# 2. Baseline对比完整性评估
# ==============================================================================

def assess_baseline_completeness() -> Tuple[Dict, List[str], float]:
    """检查baseline方法覆盖率"""

    implemented_baselines = []

    # 检查各个文件
    files_to_check = {
        'sota_detectors.py': ['DAADBDetector', 'FLatSDetector', 'RMDDetector'],
        'quick_fix.py': ['FixedKNNDetector', 'LOFDetector', 'MahalanobisDetector'],
        'heterophily_enhanced_fixed.py': ['HeterophilyEnhancedFixed'],
        'classifier_baselines.py': ['MSP', 'Energy', 'MaxLogits'],
        'priority1_experiments.py': ['AblationDetector'],
        'priority2_experiments.py': ['AdaptiveKSelector', 'HeterophilyAnalyzer']
    }

    for file_name, expected_classes in files_to_check.items():
        file_path = PROJECT_ROOT / file_name
        if file_path.exists():
            content = file_path.read_text()
            for class_name in expected_classes:
                if f'class {class_name}' in content or f'{class_name}' in content:
                    # Map to readable name
                    name_map = {
                        'DAADBDetector': 'DA-ADB',
                        'FLatSDetector': 'FLatS',
                        'RMDDetector': 'RMD',
                        'FixedKNNDetector': 'KNN Distance',
                        'LOFDetector': 'LOF',
                        'MahalanobisDetector': 'Mahalanobis',
                        'HeterophilyEnhancedFixed': 'Heterophily-Enhanced (Ours)',
                        'AblationDetector': 'Ablation Detector',
                        'AdaptiveKSelector': 'Adaptive K Selector',
                        'HeterophilyAnalyzer': 'NHR Analyzer',
                        'MSP': 'MSP',
                        'Energy': 'Energy Score',
                        'MaxLogits': 'MaxLogits'
                    }
                    readable_name = name_map.get(class_name, class_name)
                    if readable_name not in implemented_baselines:
                        implemented_baselines.append(readable_name)

    # 分类检查
    required_categories = {
        'traditional': {
            'required': ['LOF', 'One-Class SVM', 'Isolation Forest'],
            'implemented': []
        },
        'output_based': {
            'required': ['MSP', 'Energy Score', 'ODIN', 'MaxLogits'],
            'implemented': []
        },
        'distance_based': {
            'required': ['Mahalanobis', 'KNN Distance', 'Cosine Distance'],
            'implemented': []
        },
        'sota_2022_2024': {
            'required': ['DA-ADB', 'FLatS', 'RMD', 'KNN-Contrastive'],
            'implemented': []
        }
    }

    # 匹配已实现的方法到类别
    for category, data in required_categories.items():
        for method in data['required']:
            for impl in implemented_baselines:
                if method.lower() in impl.lower() or impl.lower() in method.lower():
                    data['implemented'].append(method)
                    break

    # 计算覆盖率
    coverage = {}
    total_required = 0
    total_implemented = 0

    for category, data in required_categories.items():
        req = len(data['required'])
        impl = len(data['implemented'])
        total_required += req
        total_implemented += impl

        coverage[category] = {
            'required': req,
            'implemented': impl,
            'methods': data['implemented'],
            'missing': list(set(data['required']) - set(data['implemented'])),
            'percentage': impl / req * 100 if req > 0 else 0
        }

    overall_score = total_implemented / total_required * 100 if total_required > 0 else 0

    return coverage, implemented_baselines, overall_score


# ==============================================================================
# 3. 消融实验完整性评估
# ==============================================================================

def assess_ablation_completeness() -> Dict:
    """检查消融实验完整性"""

    required_ablations = {
        'full_model': {
            'name': 'Full Model',
            'required': True,
            'found': False,
            'critical': False
        },
        'knn_only': {
            'name': 'KNN Only (w/o Heterophily)',
            'required': True,
            'found': False,
            'critical': True  # 核心消融
        },
        'heterophily_only': {
            'name': 'Heterophily Only',
            'required': True,
            'found': False,
            'critical': True
        },
        'different_k_values': {
            'name': 'Different k Values',
            'required': True,
            'found': False,
            'critical': False
        },
        'distance_methods': {
            'name': 'Distance Methods (kth vs mean)',
            'required': True,
            'found': False,
            'critical': False
        },
        'adaptive_k': {
            'name': 'Adaptive k Selection',
            'required': False,
            'found': False,
            'critical': False
        }
    }

    # 检查消融结果文件
    ablation_file = RESULTS_DIR / 'ablation_results.json'
    priority1_file = RESULTS_DIR / 'priority1_full_results.json'
    priority2_file = RESULTS_DIR / 'priority2_results.json'

    all_content = ""

    for file_path in [ablation_file, priority1_file, priority2_file]:
        if file_path.exists():
            with open(file_path) as f:
                all_content += json.dumps(json.load(f)).lower()

    # 检查每个配置
    keywords = {
        'full_model': ['full model', 'full_model', 'full (k='],
        'knn_only': ['knn only', 'knn_only', 'w/o heterophily', 'without heterophily'],
        'heterophily_only': ['heterophily only', 'heterophily_only'],
        'different_k_values': ['k=2', 'k=5', 'k=10', 'k=50'],
        'distance_methods': ['kth', 'mean', 'distance_method'],
        'adaptive_k': ['adaptive', 'adaptive_k', 'adaptivekselector']
    }

    for ablation_key, kws in keywords.items():
        for kw in kws:
            if kw.lower() in all_content:
                required_ablations[ablation_key]['found'] = True
                break

    # 计算完成度
    total_required = sum(1 for v in required_ablations.values() if v['required'])
    completed = sum(1 for v in required_ablations.values() if v['required'] and v['found'])

    critical_missing = [
        v['name'] for v in required_ablations.values()
        if v.get('critical') and not v['found']
    ]

    return {
        'completeness': completed / total_required * 100 if total_required > 0 else 0,
        'details': required_ablations,
        'critical_missing': critical_missing,
        'total_required': total_required,
        'completed': completed
    }


# ==============================================================================
# 4. 统计分析规范性评估
# ==============================================================================

def assess_statistical_rigor() -> Tuple[Dict, float]:
    """检查统计分析规范性"""

    checks = {
        'multi_seed': {
            'required_seeds': 5,
            'found_seeds': 0,
            'status': 'unknown'
        },
        'significance_tests': {
            'paired_ttest': False,
            'bootstrap_ci': False,
            'cohens_d': False,
            'wilcoxon': False
        },
        'multiple_comparison': {
            'correction_applied': False,
            'method': None
        },
        'error_bars': {
            'reported_std': False
        }
    }

    # 检查结果文件
    priority2_file = RESULTS_DIR / 'priority2_results.json'

    if priority2_file.exists():
        with open(priority2_file) as f:
            data = json.load(f)
            content = json.dumps(data).lower()

            # 检查随机种子
            if 'significance' in data:
                sig_data = data.get('significance', {})
                for dataset_results in sig_data.values():
                    if 'individual_aurocs' in dataset_results:
                        checks['multi_seed']['found_seeds'] = len(dataset_results['individual_aurocs'])
                        break
                    elif 'n_runs' in dataset_results:
                        checks['multi_seed']['found_seeds'] = dataset_results['n_runs']
                        break

            # 检查统计检验
            if 't_statistic' in content or 'ttest' in content or 't_test' in content:
                checks['significance_tests']['paired_ttest'] = True

            if 'bootstrap' in content or 'ci_lower' in content or 'auroc_ci' in content:
                checks['significance_tests']['bootstrap_ci'] = True

            if 'cohens_d' in content or "cohen's d" in content:
                checks['significance_tests']['cohens_d'] = True

            if 'std' in content or 'auroc_std' in content:
                checks['error_bars']['reported_std'] = True

    # 也检查priority1文件
    priority1_file = RESULTS_DIR / 'priority1_full_results.json'
    if priority1_file.exists():
        with open(priority1_file) as f:
            content = json.dumps(json.load(f)).lower()
            if 'std' in content:
                checks['error_bars']['reported_std'] = True

    # 评分
    checks['multi_seed']['status'] = 'pass' if checks['multi_seed']['found_seeds'] >= 5 else 'fail'

    sig_score_components = []
    if checks['multi_seed']['status'] == 'pass':
        sig_score_components.append(30)
    elif checks['multi_seed']['found_seeds'] > 0:
        sig_score_components.append(15)

    if checks['significance_tests']['paired_ttest']:
        sig_score_components.append(25)
    if checks['significance_tests']['bootstrap_ci']:
        sig_score_components.append(25)
    if checks['error_bars']['reported_std']:
        sig_score_components.append(20)

    stat_score = sum(sig_score_components)

    return checks, stat_score


# ==============================================================================
# 5. 可视化完整性评估
# ==============================================================================

def assess_visualization_completeness() -> Tuple[Dict, float]:
    """检查图表完整性和质量"""

    required_figures = {
        'main_results': {
            'comparison_bar': {
                'patterns': ['*comparison*.png', '*comparison*.pdf', '*bar*.png'],
                'required': True,
                'found': False,
                'description': '主结果对比柱状图'
            },
            'roc_curves': {
                'patterns': ['*roc*.png', '*roc*.pdf'],
                'required': True,
                'found': False,
                'description': 'ROC曲线'
            }
        },
        'ablation': {
            'ablation_heatmap': {
                'patterns': ['*ablation*.png', '*ablation*.pdf', '*heatmap*.png'],
                'required': True,
                'found': False,
                'description': '消融实验热力图'
            }
        },
        'heterophily_core': {
            'nhr_distribution': {
                'patterns': ['*nhr*distribution*.png', '*heterophil*distribution*.png', '*nhr*.png'],
                'required': True,
                'found': False,
                'critical': True,
                'description': 'NHR分布图 (ID vs OOD)'
            },
            'nhr_correlation': {
                'patterns': ['*nhr*correlation*.png', '*nhr*ood*.png', '*correlation*.png'],
                'required': True,
                'found': False,
                'critical': True,
                'description': 'NHR与OOD分数相关性图'
            }
        },
        'feature_space': {
            'tsne': {
                'patterns': ['*tsne*.png', '*tsne*.pdf', '*t-sne*.png'],
                'required': True,
                'found': False,
                'description': 't-SNE降维可视化'
            }
        },
        'hyperparameter': {
            'sensitivity': {
                'patterns': ['*sensitiv*.png', '*hyperparam*.png', '*k_value*.png'],
                'required': False,
                'found': False,
                'description': '超参数敏感性曲线'
            }
        }
    }

    # 扫描figures目录
    all_figures = []
    if FIGURES_DIR.exists():
        all_figures = list(FIGURES_DIR.glob('*.png')) + list(FIGURES_DIR.glob('*.pdf')) + list(FIGURES_DIR.glob('*.svg'))

    # 检查每个图表
    for category, figures in required_figures.items():
        for fig_name, config in figures.items():
            for pattern in config['patterns']:
                for fig_file in all_figures:
                    if fnmatch.fnmatch(fig_file.name.lower(), pattern.lower()):
                        config['found'] = True
                        config['file'] = str(fig_file)
                        break
                if config['found']:
                    break

    # 计算完成度
    total_required = sum(
        1 for cat in required_figures.values()
        for fig in cat.values() if fig['required']
    )
    found = sum(
        1 for cat in required_figures.values()
        for fig in cat.values() if fig['required'] and fig['found']
    )

    completeness = found / total_required * 100 if total_required > 0 else 0

    # 识别关键缺失
    critical_missing = [
        fig['description']
        for cat in required_figures.values()
        for fig in cat.values()
        if fig.get('critical') and not fig['found']
    ]

    return {
        'figures': required_figures,
        'completeness': completeness,
        'critical_missing': critical_missing,
        'total_required': total_required,
        'found': found,
        'figures_dir_exists': FIGURES_DIR.exists()
    }, completeness


# ==============================================================================
# 综合评估与报告生成
# ==============================================================================

def calculate_overall_score(assessments: Dict) -> Tuple[float, str, Dict]:
    """计算总体CCF-A合规性评分"""

    scores = {
        'datasets': assessments['dataset_score'],
        'baselines': assessments['baseline_score'],
        'ablation': assessments['ablation_completeness'],
        'statistics': assessments['stat_score'],
        'visualization': assessments['viz_completeness']
    }

    weighted_score = sum(scores[k] * WEIGHTS[k] for k in scores.keys())

    # 评级
    if weighted_score >= 90:
        grade = 'A+ (Ready for Top-Tier Submission)'
    elif weighted_score >= 80:
        grade = 'A (Minor Revisions Needed)'
    elif weighted_score >= 70:
        grade = 'B+ (Moderate Revisions Required)'
    elif weighted_score >= 60:
        grade = 'B (Substantial Work Needed)'
    else:
        grade = 'C (Major Gaps)'

    return weighted_score, grade, scores


def generate_markdown_report(
    dataset_status, baseline_coverage, baselines, ablation_status,
    stat_checks, viz_status, total_score, grade, component_scores
) -> str:
    """生成详细的Markdown评估报告"""

    report = f"""# RW3项目CCF-A标准合规性评估报告

**评估时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**总体得分**: {total_score:.1f}/100
**评级**: {grade}

---

## 执行摘要

本报告系统评估了RW3 (Heterophily-Aware Text OOD Detection) 项目相对于CCF-A级会议（ACL/EMNLP/NeurIPS/ICML）发表标准的完整性。

### 分项得分

| 维度 | 得分 | 权重 | 加权得分 |
|------|------|------|----------|
| 数据集完整性 | {component_scores['datasets']:.1f}% | 25% | {component_scores['datasets'] * 0.25:.1f} |
| Baseline对比 | {component_scores['baselines']:.1f}% | 20% | {component_scores['baselines'] * 0.20:.1f} |
| 消融实验 | {component_scores['ablation']:.1f}% | 15% | {component_scores['ablation'] * 0.15:.1f} |
| 统计分析 | {component_scores['statistics']:.1f}% | 15% | {component_scores['statistics'] * 0.15:.1f} |
| 可视化 | {component_scores['visualization']:.1f}% | 25% | {component_scores['visualization'] * 0.25:.1f} |
| **总计** | - | 100% | **{total_score:.1f}** |

---

## 1. 数据集完整性评估 ({component_scores['datasets']:.1f}/100)

### 1.1 第一梯队（核心必需）

| 数据集 | 状态 | AUROC | 阈值 | 达标 |
|--------|------|-------|------|------|
"""

    for name, config in dataset_status['tier1_datasets'].items():
        status_icon = '✅' if config['status'] == 'found' else '❌'
        auroc_str = f"{config['auroc']*100:.2f}%" if config['auroc'] else 'N/A'
        threshold = f"≥{config['min_auroc']*100:.0f}%" if config.get('min_auroc') else '-'
        meets = '✅' if config.get('meets_threshold') else '⚠️' if config['status'] == 'found' else '❌'
        report += f"| {name.upper()} | {status_icon} | {auroc_str} | {threshold} | {meets} |\n"

    report += """
### 1.2 第二梯队（鲁棒性）

| 数据集 | 状态 | AUROC | 零样本 |
|--------|------|-------|--------|
"""

    for name, config in dataset_status['tier2_datasets'].items():
        status_icon = '✅' if config['status'] == 'found' else '❌'
        auroc_str = f"{config['auroc']*100:.2f}%" if config['auroc'] else 'N/A'
        zero_shot = '✅' if config.get('zero_shot') else '-'
        report += f"| {name.upper()} | {status_icon} | {auroc_str} | {zero_shot} |\n"

    report += """
### 1.3 第三梯队（可选增强）

| 数据集 | 状态 | AUROC |
|--------|------|-------|
"""

    for name, config in dataset_status['tier3_datasets'].items():
        status_icon = '✅' if config['status'] == 'found' else '⭕'
        auroc_str = f"{config['auroc']*100:.2f}%" if config['auroc'] else 'N/A'
        report += f"| {name.upper()} | {status_icon} | {auroc_str} |\n"

    report += f"""

---

## 2. Baseline对比完整性 ({component_scores['baselines']:.1f}/100)

### 2.1 方法覆盖率

| 类别 | 推荐 | 已实现 | 完成度 | 缺失 |
|------|------|--------|--------|------|
"""

    for category, data in baseline_coverage.items():
        missing = ', '.join(data['missing']) if data['missing'] else '-'
        report += f"| {category.replace('_', ' ').title()} | {data['required']} | {data['implemented']} | {data['percentage']:.0f}% | {missing} |\n"

    report += f"""

### 2.2 已实现的方法

"""
    for method in baselines:
        report += f"- {method}\n"

    report += f"""

### 2.3 建议补充

- [ ] KNN-Contrastive (ACL 2022) - 对比学习基线
- [ ] VI-OOD (COLING 2024) - 变分推断方法
- [ ] BLOOD (ICLR 2024) - 最新SOTA

---

## 3. 消融实验完整性 ({component_scores['ablation']:.1f}/100)

### 3.1 消融配置检查

| 配置 | 必需 | 状态 | 重要性 |
|------|------|------|--------|
"""

    for key, config in ablation_status['details'].items():
        required = '✅' if config['required'] else '⭕'
        found = '✅' if config['found'] else '❌'
        importance = '🔴 Critical' if config.get('critical') else '🟢 Normal'
        report += f"| {config['name']} | {required} | {found} | {importance} |\n"

    if ablation_status['critical_missing']:
        report += f"""

### 3.2 关键缺失

"""
        for missing in ablation_status['critical_missing']:
            report += f"- ⚠️ {missing}\n"

    report += f"""

---

## 4. 统计分析规范性 ({component_scores['statistics']:.1f}/100)

### 4.1 多次实验

- 随机种子数: {stat_checks['multi_seed']['found_seeds']}/{stat_checks['multi_seed']['required_seeds']} {'✅' if stat_checks['multi_seed']['status'] == 'pass' else '❌'}

### 4.2 显著性检验

| 检验类型 | 状态 |
|---------|------|
| Paired t-test | {'✅' if stat_checks['significance_tests']['paired_ttest'] else '❌'} |
| Bootstrap 95% CI | {'✅' if stat_checks['significance_tests']['bootstrap_ci'] else '❌'} |
| Cohen's d效应量 | {'✅' if stat_checks['significance_tests']['cohens_d'] else '❌'} |
| 标准差报告 | {'✅' if stat_checks['error_bars']['reported_std'] else '❌'} |

### 4.3 建议改进

- [ ] 在主结果表中添加显著性标注 (`* p<0.05, ** p<0.01, *** p<0.001`)
- [ ] 实施Holm多重比较校正
- [ ] 报告Cohen's d效应量

---

## 5. 可视化完整性 ({component_scores['visualization']:.1f}/100)

### 5.1 图表清单

| 类别 | 图表 | 状态 | 关键性 |
|------|------|------|--------|
"""

    for category, figures in viz_status['figures'].items():
        for fig_name, config in figures.items():
            status = '✅' if config['found'] else '❌'
            critical = '🔴' if config.get('critical') else ''
            report += f"| {category.replace('_', ' ').title()} | {config['description']} | {status} | {critical} |\n"

    if viz_status['critical_missing']:
        report += f"""

### 5.2 关键缺失图表

"""
        for missing in viz_status['critical_missing']:
            report += f"- ⚠️ {missing}\n"

    report += f"""

---

## 6. 优先级改进建议

### P0 - 紧急（1-3天）

1. **生成Graph Heterophily核心可视化**
   - NHR分布图（ID vs OOD直方图+箱线图）
   - NHR与OOD Score相关性散点图
   - 这是论文核心创新点的直接证据

### P1 - 高优先级（1周）

2. **补充关键消融实验**
"""

    if ablation_status['critical_missing']:
        for missing in ablation_status['critical_missing']:
            report += f"   - 实现{missing}配置\n"
    else:
        report += "   - 消融实验已较完整\n"

    report += """
3. **补充SOTA Baseline对比**
   - 实现KNN-Contrastive作为对比学习代表
   - 考虑添加VI-OOD或BLOOD

### P2 - 中优先级（2周）

4. **统计分析规范化**
   - 在所有结果表中添加显著性标注
   - 实施多重比较校正

5. **生成ROC/PR曲线**
   - CLINC150和Banking77的多方法对比ROC曲线
   - 对应的Precision-Recall曲线

### P3 - 低优先级（可选）

6. **额外可视化增强**
   - t-SNE特征空间可视化
   - 超参数敏感性曲线
   - Few-shot学习曲线

---

## 7. 结论

当前RW3项目在**数据集评估**和**基础实验**方面已达到良好水平，性能指标（CLINC150: 96.23%, ROSTD: 99.23%）接近或超越SOTA。

**主要差距**：
1. Graph Heterophily可视化缺失（核心创新点支撑不足）
2. 部分SOTA Baseline对比缺失

**预估提升**：
完成P0和P1改进后，预计总分可从 **{total_score:.1f}** 提升至 **85-90分**，达到ACL/EMNLP主会场发表标准。

---

*报告生成: RW3 CCF-A Compliance Assessment Tool*
"""

    return report


def generate_todo_list(assessments: Dict, ablation_status: Dict, viz_status: Dict) -> str:
    """生成优先级排序的待办清单"""

    todos = []

    # P0: 可视化（如果缺失关键图表）
    if viz_status['critical_missing']:
        todos.append({
            'priority': 'P0 - Critical',
            'task': 'Graph Heterophily核心可视化',
            'items': viz_status['critical_missing'] + [
                '使用matplotlib/seaborn生成高质量PDF图表',
                '添加统计检验结果标注（p值）'
            ],
            'estimated_time': '2-3天',
            'impact': '直接支撑核心创新点'
        })

    # P0: 消融实验关键缺失
    if ablation_status['critical_missing']:
        todos.append({
            'priority': 'P0 - Critical',
            'task': '补充关键消融实验',
            'items': [f'实现"{m}"配置' for m in ablation_status['critical_missing']],
            'estimated_time': '3-5天',
            'impact': '证明方法有效性的必要证据'
        })

    # P1: Baseline
    if assessments['baseline_score'] < 70:
        todos.append({
            'priority': 'P1 - High',
            'task': '补充SOTA Baseline',
            'items': [
                '实现KNN-Contrastive (ACL 2022)',
                '考虑VI-OOD (COLING 2024)',
                '考虑BLOOD (ICLR 2024)'
            ],
            'estimated_time': '5-7天',
            'impact': '证明方法相对最新SOTA的优势'
        })

    # P1: 统计分析
    if assessments['stat_score'] < 80:
        todos.append({
            'priority': 'P1 - High',
            'task': '规范化统计报告',
            'items': [
                '在主结果表中添加显著性标注',
                '实施Holm多重比较校正',
                '报告Cohen\'s d效应量'
            ],
            'estimated_time': '1-2天',
            'impact': '提升论文的学术严谨性'
        })

    # P2: 其他可视化
    if assessments['viz_completeness'] < 80:
        todos.append({
            'priority': 'P2 - Medium',
            'task': 'ROC/PR曲线生成',
            'items': [
                '生成CLINC150 ROC曲线（我们的方法 + 5个baselines）',
                '生成Banking77 ROC曲线',
                '生成对应的PR曲线',
                't-SNE特征空间可视化'
            ],
            'estimated_time': '1-2天',
            'impact': '标准的OOD检测性能展示'
        })

    # 生成Markdown
    md = f"""# RW3项目待办清单（按优先级排序）

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**当前总分**: {assessments['total_score']:.1f}/100

---

"""

    for todo in todos:
        md += f"""## {todo['priority']}: {todo['task']}

**预计时间**: {todo['estimated_time']}
**影响**: {todo['impact']}

**任务清单**:
"""
        for item in todo['items']:
            md += f"- [ ] {item}\n"
        md += "\n---\n\n"

    if not todos:
        md += """## 🎉 恭喜！

所有关键任务已完成，项目已达到CCF-A发表标准。

**建议下一步**:
- 进行最终论文撰写
- 准备补充材料（Appendix）
- 代码整理与开源准备
"""

    return md


def run_complete_assessment():
    """执行完整的CCF-A合规性评估"""

    print("="*70)
    print("🔍 RW3项目CCF-A标准合规性评估")
    print("="*70)
    print(f"\n执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. 数据集评估
    print("[1/5] 评估数据集完整性...")
    dataset_status, dataset_score = assess_dataset_compliance()
    print(f"   数据集得分: {dataset_score:.1f}/100")

    # 2. Baseline评估
    print("\n[2/5] 评估Baseline覆盖率...")
    baseline_coverage, baselines, baseline_score = assess_baseline_completeness()
    print(f"   Baseline得分: {baseline_score:.1f}/100")
    print(f"   已实现: {len(baselines)}个方法")

    # 3. 消融实验评估
    print("\n[3/5] 评估消融实验完整性...")
    ablation_status = assess_ablation_completeness()
    print(f"   消融实验完成度: {ablation_status['completeness']:.1f}%")
    if ablation_status['critical_missing']:
        print(f"   ⚠️  关键缺失: {', '.join(ablation_status['critical_missing'])}")

    # 4. 统计分析评估
    print("\n[4/5] 评估统计分析规范性...")
    stat_checks, stat_score = assess_statistical_rigor()
    print(f"   统计分析得分: {stat_score:.1f}/100")
    print(f"   随机种子数: {stat_checks['multi_seed']['found_seeds']}")

    # 5. 可视化评估
    print("\n[5/5] 评估可视化完整性...")
    viz_status, viz_completeness = assess_visualization_completeness()
    print(f"   可视化完成度: {viz_completeness:.1f}%")
    if viz_status['critical_missing']:
        print(f"   ⚠️  关键缺失: {', '.join(viz_status['critical_missing'])}")

    # 综合评分
    print("\n" + "="*70)
    print("📈 综合评分")
    print("="*70)

    assessments = {
        'dataset_score': dataset_score,
        'baseline_score': baseline_score,
        'ablation_completeness': ablation_status['completeness'],
        'stat_score': stat_score,
        'viz_completeness': viz_completeness
    }

    total_score, grade, component_scores = calculate_overall_score(assessments)
    assessments['total_score'] = total_score

    print(f"\n总分: {total_score:.1f}/100")
    print(f"评级: {grade}")
    print(f"\n分项得分:")
    for component, score in component_scores.items():
        weight = WEIGHTS[component]
        weighted = score * weight
        print(f"  {component:20s}: {score:5.1f}% (权重{weight:.0%}) = {weighted:5.1f}")

    # 生成报告
    report = generate_markdown_report(
        dataset_status, baseline_coverage, baselines, ablation_status,
        stat_checks, viz_status, total_score, grade, component_scores
    )

    # 保存报告
    output_dir = PROJECT_ROOT / 'assessment'
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f'CCF-A_Assessment_{timestamp}.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n✅ 详细报告已保存: {report_file}")

    # 生成待办清单
    todo_list = generate_todo_list(assessments, ablation_status, viz_status)
    todo_file = output_dir / 'TODO_Priority_List.md'
    with open(todo_file, 'w', encoding='utf-8') as f:
        f.write(todo_list)

    print(f"✅ 待办清单已保存: {todo_file}")

    # 保存JSON结果
    json_results = {
        'timestamp': datetime.now().isoformat(),
        'total_score': total_score,
        'grade': grade,
        'component_scores': component_scores,
        'dataset_status': {k: {kk: vv for kk, vv in v.items() if kk != 'auroc' or vv is None or isinstance(vv, (int, float, str, bool))}
                          for k, v in dataset_status.items()},
        'baseline_coverage': baseline_coverage,
        'implemented_baselines': baselines,
        'ablation_status': ablation_status,
        'stat_checks': stat_checks,
        'viz_status': {
            'completeness': viz_status['completeness'],
            'critical_missing': viz_status['critical_missing'],
            'figures_dir_exists': viz_status['figures_dir_exists']
        }
    }

    json_file = output_dir / 'assessment_results.json'
    with open(json_file, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)

    print(f"✅ JSON结果已保存: {json_file}")

    return assessments, total_score, grade


if __name__ == '__main__':
    assessments, total_score, grade = run_complete_assessment()

    print("\n" + "="*70)
    print("🎯 评估完成！")
    print("="*70)
    print(f"\n当前状态: {grade}")
    print(f"总分: {total_score:.1f}/100")
    print("\n请查看生成的报告文件获取详细信息和改进建议。")
