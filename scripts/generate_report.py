#!/usr/bin/env python3
"""
Generate Preliminary Experiment Report

Aggregates results from all hypothesis verification experiments
and generates a comprehensive markdown report.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_results(results_dir: str) -> Dict[str, Dict]:
    """Load all hypothesis test results"""
    results = {}

    for h_id in ['h1', 'h2', 'h3', 'h4', 'h5']:
        result_file = os.path.join(results_dir, h_id, f'{h_id}_results.json')
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                results[h_id] = json.load(f)
        else:
            results[h_id] = None

    return results


def generate_summary_table(results: Dict[str, Dict]) -> str:
    """Generate markdown summary table"""
    table = """
| Hypothesis | Description | Primary Metric | Threshold | Value | p-value | Result |
|------------|-------------|----------------|-----------|-------|---------|--------|
"""

    hypothesis_info = {
        'h1': {
            'name': 'H1 (LDA-LLM)',
            'desc': 'Distribution Shift vs F1',
            'metric': 'Pearson r',
            'threshold': '> 0.8'
        },
        'h2': {
            'name': 'H2 (LLM-RFCRE)',
            'desc': 'ARS vs Forgetting Rate',
            'metric': 'Spearman ρ',
            'threshold': '> 0.5'
        },
        'h3': {
            'name': 'H3 (LLM-PUQ)',
            'desc': 'PDI vs Noise Rate',
            'metric': 'Pearson r',
            'threshold': '> 0.5'
        },
        'h4': {
            'name': 'H4 (HGT-LC)',
            'desc': 'Path Length vs FN Rate',
            'metric': "Cohen's d",
            'threshold': '> 0.5'
        },
        'h5': {
            'name': 'H5 (PGCDN)',
            'desc': 'Bag Size vs Reliability',
            'metric': "Cohen's d",
            'threshold': '0.5-0.8'
        }
    }

    for h_id, info in hypothesis_info.items():
        result = results.get(h_id)

        if result is None:
            table += f"| {info['name']} | {info['desc']} | {info['metric']} | {info['threshold']} | N/A | N/A | ⏳ Pending |\n"
            continue

        passed = result.get('passed', False)
        status = "✅ Passed" if passed else "❌ Failed"

        # Extract metrics
        if h_id == 'h1':
            value = result.get('correlation', {}).get('pearson_r', 'N/A')
            p_val = result.get('correlation', {}).get('p_value', 'N/A')
        elif h_id == 'h2':
            value = result.get('correlation', {}).get('spearman_rho', 'N/A')
            p_val = result.get('correlation', {}).get('p_value', 'N/A')
        elif h_id == 'h3':
            value = result.get('correlation', {}).get('pearson_r', 'N/A')
            p_val = result.get('correlation', {}).get('p_value', 'N/A')
        elif h_id == 'h4':
            value = result.get('statistics', {}).get('cohens_d', 'N/A')
            p_val = result.get('statistics', {}).get('p_value', 'N/A')
        elif h_id == 'h5':
            value = result.get('statistics', {}).get('cohens_d', 'N/A')
            p_val = result.get('statistics', {}).get('p_value', 'N/A')

        if isinstance(value, float):
            value = f"{value:.4f}"
        if isinstance(p_val, float):
            p_val = f"{p_val:.4e}"

        table += f"| {info['name']} | {info['desc']} | {info['metric']} | {info['threshold']} | {value} | {p_val} | {status} |\n"

    return table


def generate_detailed_section(h_id: str, result: Dict) -> str:
    """Generate detailed section for a hypothesis"""
    if result is None:
        return f"\n### {h_id.upper()}: Pending\n\nExperiment not yet run.\n"

    sections = []
    sections.append(f"\n### {result.get('hypothesis', h_id.upper())}: {result.get('description', '')}")
    sections.append("")

    # Status
    passed = result.get('passed', False)
    status = "**PASSED** ✅" if passed else "**FAILED** ❌"
    sections.append(f"**Status**: {status}")
    sections.append("")

    # Statistics
    sections.append("#### Statistics")
    sections.append("")

    if 'correlation' in result:
        corr = result['correlation']
        for key, value in corr.items():
            if isinstance(value, float):
                sections.append(f"- **{key}**: {value:.4f}")
            else:
                sections.append(f"- **{key}**: {value}")

    if 'statistics' in result:
        stats = result['statistics']
        for key, value in stats.items():
            if isinstance(value, float):
                sections.append(f"- **{key}**: {value:.4f}")
            elif isinstance(value, (int, str)):
                sections.append(f"- **{key}**: {value}")

    sections.append("")

    # Group analysis if available
    if 'group_analysis' in result:
        sections.append("#### Group Analysis")
        sections.append("")
        for group, data in result['group_analysis'].items():
            if isinstance(data, dict):
                sections.append(f"- **{group}**:")
                for k, v in data.items():
                    if isinstance(v, float):
                        sections.append(f"  - {k}: {v:.4f}")
                    else:
                        sections.append(f"  - {k}: {v}")
            else:
                if isinstance(data, float):
                    sections.append(f"- **{group}**: {data:.4f}")
                else:
                    sections.append(f"- **{group}**: {data}")
        sections.append("")

    # Threshold
    if 'threshold' in result:
        sections.append("#### Threshold Criteria")
        sections.append("")
        for key, value in result['threshold'].items():
            sections.append(f"- {key}: {value}")
        sections.append("")

    return "\n".join(sections)


def generate_recommendations(results: Dict[str, Dict]) -> str:
    """Generate recommendations based on results"""
    passed_count = sum(1 for r in results.values() if r and r.get('passed', False))
    total = len(results)

    sections = []
    sections.append("\n## Recommendations")
    sections.append("")

    if passed_count >= 4:
        sections.append("### ✅ Proceed to Full Experiments")
        sections.append("")
        sections.append(f"**{passed_count}/{total}** hypotheses passed verification.")
        sections.append("")
        sections.append("The preliminary evidence strongly supports the research directions.")
        sections.append("Recommended next steps:")
        sections.append("")
        sections.append("1. Proceed with full method implementation for passed hypotheses")
        sections.append("2. Run comprehensive experiments on standard benchmarks")
        sections.append("3. Prepare for publication submission")

    elif passed_count >= 2:
        sections.append("### ⚠️ Partial Success - Review and Adjust")
        sections.append("")
        sections.append(f"**{passed_count}/{total}** hypotheses passed verification.")
        sections.append("")
        sections.append("Recommendations:")
        sections.append("")
        sections.append("1. Focus on methods for passed hypotheses")
        sections.append("2. Re-evaluate failed hypotheses:")

        for h_id, result in results.items():
            if result and not result.get('passed', True):
                sections.append(f"   - {h_id.upper()}: Consider alternative verification methods")

        sections.append("3. Consider adjusting research scope")

    else:
        sections.append("### ❌ Re-evaluate Research Direction")
        sections.append("")
        sections.append(f"**{passed_count}/{total}** hypotheses passed verification.")
        sections.append("")
        sections.append("The preliminary evidence does not strongly support the current research directions.")
        sections.append("Recommendations:")
        sections.append("")
        sections.append("1. Review hypothesis formulations")
        sections.append("2. Consider alternative research questions")
        sections.append("3. Conduct additional literature review")
        sections.append("4. Consult with advisor before proceeding")

    sections.append("")
    return "\n".join(sections)


def generate_report(results_dir: str, output_path: str) -> str:
    """Generate the full preliminary experiment report"""
    results = load_results(results_dir)

    # Count results
    passed_count = sum(1 for r in results.values() if r and r.get('passed', False))
    total = len([r for r in results.values() if r is not None])

    report = []

    # Header
    report.append("# RW1 Preliminary Experiment Report")
    report.append("")
    report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("---")
    report.append("")

    # Executive Summary
    report.append("## Executive Summary")
    report.append("")
    report.append(f"This report presents the results of preliminary experiments to verify")
    report.append(f"5 research hypotheses for the RW1 research work on Remote Supervision")
    report.append(f"Relation Extraction.")
    report.append("")
    report.append(f"**Overall Result**: {passed_count}/{total} hypotheses passed verification")
    report.append("")

    # Summary Table
    report.append("## Summary Table")
    report.append("")
    report.append(generate_summary_table(results))
    report.append("")

    # Detailed Results
    report.append("---")
    report.append("")
    report.append("## Detailed Results")
    report.append("")

    for h_id in ['h1', 'h2', 'h3', 'h4', 'h5']:
        report.append(generate_detailed_section(h_id, results.get(h_id)))

    # Recommendations
    report.append("---")
    report.append(generate_recommendations(results))

    # Methodology Notes
    report.append("---")
    report.append("")
    report.append("## Methodology Notes")
    report.append("")
    report.append("### Statistical Standards")
    report.append("")
    report.append("- **Significance level**: p < 0.05")
    report.append("- **Effect size interpretation**: Cohen's d (0.2=small, 0.5=medium, 0.8=large)")
    report.append("- **Correlation interpretation**: |r| (0.3=weak, 0.5=moderate, 0.7=strong)")
    report.append("")
    report.append("### Avoiding Circular Reasoning")
    report.append("")
    report.append("All experiments were designed to avoid circular reasoning:")
    report.append("- No target labels used in feature computation")
    report.append("- Independent validation methods (e.g., human annotations for H5)")
    report.append("- Synthetic noise injection for controlled experiments (H3)")
    report.append("")

    # Appendix
    report.append("---")
    report.append("")
    report.append("## Appendix: File Locations")
    report.append("")
    report.append("### Results")
    report.append("")
    for h_id in ['h1', 'h2', 'h3', 'h4', 'h5']:
        report.append(f"- `results/{h_id}/{h_id}_results.json`")
    report.append("")
    report.append("### Figures")
    report.append("")
    for h_id in ['h1', 'h2', 'h3', 'h4', 'h5']:
        report.append(f"- `results/{h_id}/figures/`")
    report.append("")

    # Join and save
    report_text = "\n".join(report)

    with open(output_path, 'w') as f:
        f.write(report_text)

    print(f"Report generated: {output_path}")
    return report_text


def main():
    parser = argparse.ArgumentParser(
        description="Generate Preliminary Experiment Report"
    )
    parser.add_argument(
        '--results_dir', type=str, default='./results',
        help='Directory containing experiment results'
    )
    parser.add_argument(
        '--output', type=str, default='./results/preliminary_experiment_report.md',
        help='Output path for the report'
    )

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    generate_report(args.results_dir, args.output)


if __name__ == '__main__':
    main()
