#!/usr/bin/env python3
"""
RW3 Kill-Switch Determination Experiment - Part A: Agent Route
Based on AgentDojo real structure analysis + simulated traces
"""

import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import networkx as nx
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEEDS = [42, 123, 456]
np.random.seed(42)

# Output directories
RESULTS_DIR = Path("/home/user/OOD-project/rw3_pilot/results")
AGENTDOJO_DIR = Path("/home/user/OOD-project/rw3_pilot/agentdojo")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("RW3 Agent Route Kill-Switch Experiment")
print("=" * 60)

# ============================================================
# A1: AgentDojo Data Structure Exploration
# ============================================================
print("\n[A1] Analyzing AgentDojo Data Structure...")

def analyze_agentdojo_structure():
    """Extract real structure from AgentDojo repository"""

    results = {
        "suites": {},
        "attacks": [],
        "tools": [],
        "total_user_tasks": 0,
        "total_injection_tasks": 0,
        "models_tested": []
    }

    # Define suites and their tools
    suites = ["banking", "workspace", "travel", "slack"]

    # Count tasks from v1 suite (main version)
    v1_path = AGENTDOJO_DIR / "src/agentdojo/default_suites/v1"

    for suite in suites:
        suite_path = v1_path / suite
        user_tasks_file = suite_path / "user_tasks.py"
        injection_tasks_file = suite_path / "injection_tasks.py"

        user_count = 0
        injection_count = 0

        if user_tasks_file.exists():
            content = user_tasks_file.read_text()
            user_count = content.count("@task_suite.register_user_task")

        if injection_tasks_file.exists():
            content = injection_tasks_file.read_text()
            injection_count = content.count("@task_suite.register_injection_task")

        results["suites"][suite] = {
            "user_tasks": user_count,
            "injection_tasks": injection_count,
            "security_tests": user_count * injection_count  # Each user_task x injection_task combo
        }
        results["total_user_tasks"] += user_count
        results["total_injection_tasks"] += injection_count

    # Extract attack types from attacks directory
    attacks_path = AGENTDOJO_DIR / "src/agentdojo/attacks"
    attack_types = [
        "direct", "ignore_previous", "system_message", "injecagent",
        "important_instructions", "important_instructions_no_user_name",
        "important_instructions_no_model_name", "important_instructions_no_names",
        "important_instructions_wrong_model_name", "important_instructions_wrong_user_name",
        "tool_knowledge", "dos", "swearwords_dos", "captcha_dos",
        "offensive_email_dos", "felony_dos"
    ]
    results["attacks"] = attack_types

    # Extract tool types from tools directory
    tools_path = v1_path / "tools"
    if tools_path.exists():
        tool_files = [f.stem for f in tools_path.glob("*.py") if f.stem != "__init__"]
        results["tools"] = tool_files

    # Get models tested from runs directory
    runs_path = AGENTDOJO_DIR / "runs"
    if runs_path.exists():
        results["models_tested"] = [d.name for d in runs_path.iterdir() if d.is_dir()]

    # Calculate total security tests (matches paper's ~629)
    total_security_tests = sum(s["security_tests"] for s in results["suites"].values())
    results["total_security_tests"] = total_security_tests

    return results

def parse_benchmark_results():
    """Parse actual benchmark results from runs directory"""

    runs_path = AGENTDOJO_DIR / "runs"
    if not runs_path.exists():
        return None

    all_results = []

    # Sample models to analyze
    models_to_analyze = ["gpt-4o-2024-05-13", "claude-3-5-sonnet-20240620", "claude-3-opus-20240229"]

    for model in models_to_analyze:
        model_path = runs_path / model
        if not model_path.exists():
            continue

        for suite in ["banking", "workspace", "travel", "slack"]:
            suite_path = model_path / suite
            if not suite_path.exists():
                continue

            for task_dir in suite_path.iterdir():
                if not task_dir.is_dir():
                    continue

                for attack_dir in task_dir.iterdir():
                    if not attack_dir.is_dir():
                        continue

                    attack_type = attack_dir.name

                    for result_file in attack_dir.glob("*.json"):
                        try:
                            with open(result_file) as f:
                                data = json.load(f)

                            # Extract tool calls from messages
                            tool_calls = []
                            for msg in data.get("messages", []):
                                if msg.get("tool_calls"):
                                    for tc in msg["tool_calls"]:
                                        tool_calls.append(tc.get("function", "unknown"))

                            all_results.append({
                                "model": model,
                                "suite": suite,
                                "user_task": task_dir.name,
                                "injection_task": result_file.stem,
                                "attack_type": attack_type,
                                "utility": data.get("utility", False),
                                "security": data.get("security", False),
                                "duration": data.get("duration", 0),
                                "num_tool_calls": len(tool_calls),
                                "tools_used": tool_calls,
                                "is_attacked": attack_type != "none"
                            })
                        except Exception as e:
                            continue

    if all_results:
        return pd.DataFrame(all_results)
    return None

# Run A1 analysis
structure = analyze_agentdojo_structure()
benchmark_df = parse_benchmark_results()

# Write A1 analysis report
with open(RESULTS_DIR / "agentdojo_analysis.md", "w") as f:
    f.write("# AgentDojo Data Structure Analysis\n\n")
    f.write("## Overview\n\n")
    f.write(f"- Total User Tasks: {structure['total_user_tasks']} (paper reports ~97)\n")
    f.write(f"- Total Injection Tasks: {structure['total_injection_tasks']}\n")
    f.write(f"- Total Security Test Combinations: {structure['total_security_tests']} (paper reports ~629)\n")
    f.write(f"- Attack Types: {len(structure['attacks'])}\n")
    f.write(f"- Tool Categories: {len(structure['tools'])}\n")
    f.write(f"- Models Benchmarked: {len(structure['models_tested'])}\n\n")

    f.write("## Suite Breakdown\n\n")
    f.write("| Suite | User Tasks | Injection Tasks | Security Tests |\n")
    f.write("|-------|------------|-----------------|----------------|\n")
    for suite, data in structure['suites'].items():
        f.write(f"| {suite} | {data['user_tasks']} | {data['injection_tasks']} | {data['security_tests']} |\n")

    f.write("\n## Attack Types\n\n")
    for i, attack in enumerate(structure['attacks'], 1):
        f.write(f"{i}. {attack}\n")

    f.write("\n## Tool Categories\n\n")
    for tool in structure['tools']:
        f.write(f"- {tool}\n")

    if benchmark_df is not None:
        f.write("\n## Benchmark Results Summary\n\n")
        f.write(f"Total traces analyzed: {len(benchmark_df)}\n\n")

        # Safety rates by model
        f.write("### Safety Rate by Model\n\n")
        attacked = benchmark_df[benchmark_df['is_attacked']]
        safety_by_model = attacked.groupby('model')['security'].mean()
        f.write("| Model | Safety Rate (attacked traces) |\n")
        f.write("|-------|-------------------------------|\n")
        for model, rate in safety_by_model.items():
            f.write(f"| {model} | {rate:.2%} |\n")

        # Safety rates by attack type
        f.write("\n### Safety Rate by Attack Type\n\n")
        safety_by_attack = attacked.groupby('attack_type')['security'].mean().sort_values(ascending=False)
        f.write("| Attack Type | Safety Rate |\n")
        f.write("|-------------|-------------|\n")
        for attack, rate in safety_by_attack.items():
            f.write(f"| {attack} | {rate:.2%} |\n")

        # Safety rates by suite
        f.write("\n### Safety Rate by Suite (Tool Domain)\n\n")
        safety_by_suite = attacked.groupby('suite')['security'].mean()
        f.write("| Suite | Safety Rate |\n")
        f.write("|-------|-------------|\n")
        for suite, rate in safety_by_suite.items():
            f.write(f"| {suite} | {rate:.2%} |\n")

print(f"  - Total user tasks: {structure['total_user_tasks']}")
print(f"  - Total injection tasks: {structure['total_injection_tasks']}")
print(f"  - Total security test combinations: {structure['total_security_tests']}")
print(f"  - Attack types: {len(structure['attacks'])}")
print(f"  - Tool categories: {len(structure['tools'])}")
print(f"  - Benchmark traces parsed: {len(benchmark_df) if benchmark_df is not None else 0}")

# ============================================================
# A2: Simulate Agent Traces Based on Real Structure
# ============================================================
print("\n[A2] Generating Simulated Agent Traces...")

@dataclass
class ToolCall:
    step: int
    tool_type: int
    tool_name: str
    is_attacked: bool
    base_risk: float
    actual_risk: float
    dag_depth: int

@dataclass
class AgentTrace:
    trace_id: int
    tools: List[ToolCall]
    dag: nx.DiGraph
    attack_type: str
    task_success: bool
    safety_violation: bool
    suite: str

# Tool types based on AgentDojo real structure
TOOL_TYPES = {
    0: {"name": "banking_client", "base_risk": 0.35, "weight": 0.15, "category": "high_risk"},
    1: {"name": "email_client", "base_risk": 0.15, "weight": 0.20, "category": "medium_risk"},
    2: {"name": "calendar_client", "base_risk": 0.08, "weight": 0.15, "category": "low_risk"},
    3: {"name": "cloud_drive", "base_risk": 0.25, "weight": 0.15, "category": "medium_risk"},
    4: {"name": "web_browse", "base_risk": 0.45, "weight": 0.10, "category": "high_risk"},
    5: {"name": "slack", "base_risk": 0.20, "weight": 0.15, "category": "medium_risk"},
    6: {"name": "travel_booking", "base_risk": 0.30, "weight": 0.10, "category": "high_risk"},
}

# Attack types based on AgentDojo
ATTACK_TYPES = ["none", "injection", "prompt_manipulation", "tool_knowledge", "dos"]
ATTACK_RISK_MULTIPLIER = {
    "none": 1.0,
    "injection": 3.5,  # important_instructions style
    "prompt_manipulation": 2.5,  # ignore_previous style
    "tool_knowledge": 4.0,  # Provides explicit tool sequences
    "dos": 1.5  # Denial of service
}

# Suite mapping to understand tool domain groupings
SUITE_TOOL_MAPPING = {
    "banking": [0, 1],  # banking_client, email
    "workspace": [1, 2, 3],  # email, calendar, cloud_drive
    "travel": [6, 1, 2],  # travel_booking, email, calendar
    "slack": [5, 1, 3]  # slack, email, cloud_drive
}

# Simulation parameters (based on AgentDojo observations)
N_TRACES = 2000
TRACE_LENGTH = (2, 7)
ATTACK_RATE = 0.30  # 30% traces attacked
PROPAGATION_FACTOR = 1.5

def simulate_traces(n=N_TRACES, seed=42):
    """Generate simulated agent traces based on real AgentDojo structure"""
    np.random.seed(seed)
    traces = []

    suites = list(SUITE_TOOL_MAPPING.keys())
    tool_weights = np.array([TOOL_TYPES[i]["weight"] for i in range(len(TOOL_TYPES))])
    tool_weights = tool_weights / tool_weights.sum()

    for trace_id in range(n):
        # Select suite
        suite = np.random.choice(suites)
        suite_tools = SUITE_TOOL_MAPPING[suite]

        # Determine trace length
        trace_len = np.random.randint(TRACE_LENGTH[0], TRACE_LENGTH[1] + 1)

        # Decide if attacked
        is_trace_attacked = np.random.random() < ATTACK_RATE
        attack_type = np.random.choice(ATTACK_TYPES[1:]) if is_trace_attacked else "none"

        # Build tool call sequence
        tools = []
        dag = nx.DiGraph()
        cumulative_risk = 0.0

        # 70% of tools from suite-specific set, 30% from global
        for step in range(trace_len):
            if np.random.random() < 0.7 and suite_tools:
                tool_type = np.random.choice(suite_tools)
            else:
                tool_type = np.random.choice(len(TOOL_TYPES), p=tool_weights)

            tool_info = TOOL_TYPES[tool_type]
            base_risk = tool_info["base_risk"]

            # Risk propagation from previous high-risk tools
            propagation_bonus = min(0.3, cumulative_risk * 0.2)

            # Attack multiplier
            attack_mult = ATTACK_RISK_MULTIPLIER[attack_type] if is_trace_attacked else 1.0

            # Calculate actual risk
            actual_risk = min(0.95, (base_risk + propagation_bonus) * attack_mult)

            # Determine if this specific tool call is the injection point
            is_this_attacked = is_trace_attacked and step == np.random.randint(0, trace_len)

            # DAG depth (linear chain with possible branch at step 3+)
            dag_depth = step + 1
            dag.add_node(step)
            if step > 0:
                dag.add_edge(step - 1, step)
                # 30% chance of additional dependency for longer traces
                if step >= 3 and np.random.random() < 0.3:
                    dag.add_edge(max(0, step - 2), step)

            tools.append(ToolCall(
                step=step,
                tool_type=tool_type,
                tool_name=tool_info["name"],
                is_attacked=is_this_attacked,
                base_risk=base_risk,
                actual_risk=actual_risk,
                dag_depth=dag_depth
            ))

            # Update cumulative risk
            if actual_risk > 0.3:
                cumulative_risk += actual_risk * PROPAGATION_FACTOR

        # Determine outcomes
        max_risk = max(t.actual_risk for t in tools)
        avg_risk = np.mean([t.actual_risk for t in tools])

        # Safety violation probability based on max risk
        safety_violation = np.random.random() < max_risk if is_trace_attacked else np.random.random() < 0.05

        # Task success (inversely related to risk, but generally high)
        task_success = np.random.random() > (avg_risk * 0.3)

        traces.append(AgentTrace(
            trace_id=trace_id,
            tools=tools,
            dag=dag,
            attack_type=attack_type,
            task_success=task_success,
            safety_violation=safety_violation,
            suite=suite
        ))

    return traces

# Generate traces for multiple seeds
all_traces_by_seed = {}
for seed in SEEDS:
    all_traces_by_seed[seed] = simulate_traces(N_TRACES, seed)

# Use primary seed for main analysis
traces = all_traces_by_seed[42]

# Convert to DataFrame for analysis
def traces_to_df(traces):
    rows = []
    for trace in traces:
        for tool in trace.tools:
            rows.append({
                "trace_id": trace.trace_id,
                "step": tool.step,
                "tool_type": tool.tool_type,
                "tool_name": tool.tool_name,
                "attack_type": trace.attack_type,
                "is_attacked": tool.is_attacked,
                "base_risk": tool.base_risk,
                "actual_risk": tool.actual_risk,
                "dag_depth": tool.dag_depth,
                "task_success": trace.task_success,
                "safety_violation": trace.safety_violation,
                "suite": trace.suite,
                "trace_length": len(trace.tools),
                "max_trace_risk": max(t.actual_risk for t in trace.tools),
                "tool_category": TOOL_TYPES[tool.tool_type]["category"]
            })
    return pd.DataFrame(rows)

df = traces_to_df(traces)

# Save simulated traces
df.to_csv(RESULTS_DIR / "simulated_traces.csv", index=False)
print(f"  - Generated {len(traces)} traces with {len(df)} total tool calls")
print(f"  - Attack rate: {df.groupby('trace_id')['attack_type'].first().apply(lambda x: x != 'none').mean():.1%}")

# ============================================================
# A3: Worst-Group Analysis (Core Judgment)
# ============================================================
print("\n[A3] Performing Worst-Group Analysis...")

def compute_worst_group_stats(df, group_col):
    """Compute risk statistics by group"""
    trace_level = df.groupby('trace_id').agg({
        group_col: 'first',
        'safety_violation': 'first',
        'task_success': 'first',
        'attack_type': 'first',
        'max_trace_risk': 'first'
    }).reset_index()

    stats_by_group = trace_level.groupby(group_col).agg({
        'safety_violation': ['mean', 'std', 'count'],
        'task_success': 'mean',
        'max_trace_risk': 'mean'
    }).round(4)

    stats_by_group.columns = ['risk_rate', 'risk_std', 'n_traces', 'success_rate', 'avg_max_risk']
    stats_by_group = stats_by_group.reset_index()

    return stats_by_group

def statistical_tests(df, group_col):
    """Run statistical tests for group differences"""
    trace_level = df.groupby('trace_id').agg({
        group_col: 'first',
        'safety_violation': 'first',
    }).reset_index()

    groups = trace_level.groupby(group_col)['safety_violation'].apply(list)

    # Kruskal-Wallis test (non-parametric ANOVA)
    if len(groups) >= 2:
        h_stat, p_value = stats.kruskal(*groups.values)
    else:
        h_stat, p_value = 0, 1.0

    # Cohen's d between worst and best groups
    group_means = trace_level.groupby(group_col)['safety_violation'].mean()
    worst_group = group_means.idxmax()
    best_group = group_means.idxmin()

    worst_data = trace_level[trace_level[group_col] == worst_group]['safety_violation']
    best_data = trace_level[trace_level[group_col] == best_group]['safety_violation']

    pooled_std = np.sqrt((worst_data.std()**2 + best_data.std()**2) / 2)
    cohens_d = (worst_data.mean() - best_data.mean()) / pooled_std if pooled_std > 0 else 0

    return {
        'h_stat': h_stat,
        'p_value': p_value,
        'cohens_d': abs(cohens_d),
        'worst_group': worst_group,
        'best_group': best_group,
        'worst_risk': group_means.max(),
        'best_risk': group_means.min(),
        'avg_risk': group_means.mean(),
        'ratio': group_means.max() / group_means.mean() if group_means.mean() > 0 else 0
    }

# Analyze by different grouping dimensions
dimensions = {
    'tool_name': 'Tool Type',
    'attack_type': 'Attack Type',
    'tool_category': 'Tool Risk Category',
    'suite': 'Agent Suite'
}

# Also create depth bins
df['depth_bin'] = pd.cut(df['dag_depth'], bins=[0, 2, 4, 10], labels=['shallow', 'medium', 'deep'])
dimensions['depth_bin'] = 'DAG Depth'

worst_group_results = {}
for col, name in dimensions.items():
    group_stats = compute_worst_group_stats(df, col)
    test_results = statistical_tests(df, col)
    worst_group_results[col] = {
        'name': name,
        'stats': group_stats,
        'tests': test_results
    }

# Create worst group table
with open(RESULTS_DIR / "worst_group_table.md", "w") as f:
    f.write("# Worst-Group Risk Analysis\n\n")

    for col, result in worst_group_results.items():
        f.write(f"## By {result['name']}\n\n")

        stats_df = result['stats']
        f.write("| Group | Risk Rate | Std | N Traces | Success Rate | Avg Max Risk |\n")
        f.write("|-------|-----------|-----|----------|--------------|---------------|\n")
        for _, row in stats_df.iterrows():
            f.write(f"| {row[col]} | {row['risk_rate']:.3f} | {row['risk_std']:.3f} | {int(row['n_traces'])} | {row['success_rate']:.3f} | {row['avg_max_risk']:.3f} |\n")

        tests = result['tests']
        f.write(f"\n**Statistical Tests:**\n")
        f.write(f"- Kruskal-Wallis H: {tests['h_stat']:.2f}, p = {tests['p_value']:.2e}\n")
        f.write(f"- Worst group: {tests['worst_group']} (risk = {tests['worst_risk']:.3f})\n")
        f.write(f"- Best group: {tests['best_group']} (risk = {tests['best_risk']:.3f})\n")
        f.write(f"- **Worst/Average ratio: {tests['ratio']:.2f}**\n")
        f.write(f"- Cohen's d (worst vs best): {tests['cohens_d']:.2f}\n\n")

# Generate visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Risk by tool type
ax = axes[0, 0]
tool_stats = worst_group_results['tool_name']['stats']
bars = ax.bar(tool_stats['tool_name'], tool_stats['risk_rate'],
              yerr=tool_stats['risk_std'], capsize=3, color='steelblue', alpha=0.8)
ax.axhline(y=tool_stats['risk_rate'].mean(), color='red', linestyle='--', label=f"Mean: {tool_stats['risk_rate'].mean():.3f}")
ax.set_xlabel('Tool Type')
ax.set_ylabel('Safety Violation Rate')
ax.set_title('Risk by Tool Type')
ax.tick_params(axis='x', rotation=45)
ax.legend()

# Plot 2: Risk by attack type
ax = axes[0, 1]
attack_stats = worst_group_results['attack_type']['stats']
colors = ['green' if x == 'none' else 'red' for x in attack_stats['attack_type']]
bars = ax.bar(attack_stats['attack_type'], attack_stats['risk_rate'],
              yerr=attack_stats['risk_std'], capsize=3, color=colors, alpha=0.8)
ax.axhline(y=attack_stats['risk_rate'].mean(), color='blue', linestyle='--', label=f"Mean: {attack_stats['risk_rate'].mean():.3f}")
ax.set_xlabel('Attack Type')
ax.set_ylabel('Safety Violation Rate')
ax.set_title('Risk by Attack Type')
ax.tick_params(axis='x', rotation=45)
ax.legend()

# Plot 3: Risk by DAG depth
ax = axes[1, 0]
depth_stats = worst_group_results['depth_bin']['stats']
ax.bar(depth_stats['depth_bin'].astype(str), depth_stats['risk_rate'],
       yerr=depth_stats['risk_std'], capsize=3, color='darkorange', alpha=0.8)
ax.axhline(y=depth_stats['risk_rate'].mean(), color='red', linestyle='--', label=f"Mean: {depth_stats['risk_rate'].mean():.3f}")
ax.set_xlabel('DAG Depth')
ax.set_ylabel('Safety Violation Rate')
ax.set_title('Risk by DAG Depth')
ax.legend()

# Plot 4: Risk by suite
ax = axes[1, 1]
suite_stats = worst_group_results['suite']['stats']
ax.bar(suite_stats['suite'], suite_stats['risk_rate'],
       yerr=suite_stats['risk_std'], capsize=3, color='purple', alpha=0.8)
ax.axhline(y=suite_stats['risk_rate'].mean(), color='red', linestyle='--', label=f"Mean: {suite_stats['risk_rate'].mean():.3f}")
ax.set_xlabel('Agent Suite')
ax.set_ylabel('Safety Violation Rate')
ax.set_title('Risk by Suite (Tool Domain)')
ax.legend()

plt.tight_layout()
plt.savefig(RESULTS_DIR / "fig_risk_by_tool_type.png", dpi=150, bbox_inches='tight')
plt.savefig(RESULTS_DIR / "fig_risk_by_attack_type.png", dpi=150, bbox_inches='tight')
plt.close()

# Separate detailed plots
for col, result in worst_group_results.items():
    fig, ax = plt.subplots(figsize=(10, 6))
    stats_df = result['stats']
    ax.bar(range(len(stats_df)), stats_df['risk_rate'],
           yerr=stats_df['risk_std'], capsize=3, alpha=0.8)
    ax.set_xticks(range(len(stats_df)))
    ax.set_xticklabels(stats_df[col], rotation=45, ha='right')
    ax.axhline(y=stats_df['risk_rate'].mean(), color='red', linestyle='--',
               label=f"Mean: {stats_df['risk_rate'].mean():.3f}")
    ax.set_ylabel('Safety Violation Rate')
    ax.set_title(f'Risk by {result["name"]}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"fig_risk_by_{col}.png", dpi=150, bbox_inches='tight')
    plt.close()

# Print key findings
print("  Worst-Group Analysis Results:")
for col, result in worst_group_results.items():
    tests = result['tests']
    print(f"    - {result['name']}: worst/avg ratio = {tests['ratio']:.2f}, p = {tests['p_value']:.2e}")

# ============================================================
# A4: Graph-Conditional CP Routing vs Baselines
# ============================================================
print("\n[A4] Implementing and Comparing Routing Strategies...")

def split_data(traces, train_ratio=0.6, cal_ratio=0.2):
    """Split traces into train/calibration/test sets"""
    n = len(traces)
    indices = np.random.permutation(n)

    train_end = int(n * train_ratio)
    cal_end = train_end + int(n * cal_ratio)

    train_traces = [traces[i] for i in indices[:train_end]]
    cal_traces = [traces[i] for i in indices[train_end:cal_end]]
    test_traces = [traces[i] for i in indices[cal_end:]]

    return train_traces, cal_traces, test_traces

# Routing strategies
def no_guard(trace):
    """Strategy 1: Accept all"""
    return "accept"

def always_abstain(trace):
    """Strategy 2: Reject all"""
    return "abstain"

def heuristic_guard(trace, risky_tools={"web_browse", "banking_client", "travel_booking"}):
    """Strategy 3: Rule-based - reject if high-risk tools present"""
    if any(t.tool_name in risky_tools for t in trace.tools):
        return "abstain"
    return "accept"

def compute_global_threshold(cal_traces, alpha=0.1):
    """Compute global threshold for unconditional CP"""
    # Use max risk in each trace as the score
    safe_traces = [t for t in cal_traces if not t.safety_violation]
    scores = [max(tc.actual_risk for tc in t.tools) for t in safe_traces]
    if not scores:
        return 0.5
    threshold = np.quantile(scores, 1 - alpha)
    return threshold

def compute_group_thresholds(cal_traces, alpha=0.1):
    """Compute per-group thresholds for graph-conditional CP"""
    thresholds = {}

    # Group by (tool_category, depth_bin)
    groups = defaultdict(list)
    for trace in cal_traces:
        if not trace.safety_violation:  # Only use safe traces for calibration
            for tool in trace.tools:
                depth_bin = "shallow" if tool.dag_depth <= 2 else ("medium" if tool.dag_depth <= 4 else "deep")
                category = TOOL_TYPES[tool.tool_type]["category"]
                groups[(category, depth_bin)].append(tool.actual_risk)

    for group_key, scores in groups.items():
        if scores:
            thresholds[group_key] = np.quantile(scores, 1 - alpha)
        else:
            thresholds[group_key] = 0.5  # Default

    return thresholds

def unconditional_cp_router(trace, global_threshold):
    """Strategy 4: Unconditional CP - single global threshold"""
    max_risk = max(t.actual_risk for t in trace.tools)
    if max_risk > global_threshold:
        return "abstain"
    return "accept"

def graph_conditional_cp_router(trace, thresholds):
    """Strategy 5: Graph-conditional CP - per-group thresholds"""
    for tool in trace.tools:
        depth_bin = "shallow" if tool.dag_depth <= 2 else ("medium" if tool.dag_depth <= 4 else "deep")
        category = TOOL_TYPES[tool.tool_type]["category"]
        group_key = (category, depth_bin)

        threshold = thresholds.get(group_key, 0.5)
        if tool.actual_risk > threshold:
            return "abstain"
    return "accept"

def evaluate_strategy(test_traces, strategy_fn, *args):
    """Evaluate a routing strategy on test traces"""
    results = {
        'total': len(test_traces),
        'benign_traces': 0,
        'attacked_traces': 0,
        'benign_accepted': 0,
        'attacked_abstained': 0,
        'violations_accepted': 0,  # Safety violations that were accepted (bad)
        'by_tool_type': defaultdict(lambda: {'total': 0, 'violations_accepted': 0}),
        'by_attack_type': defaultdict(lambda: {'total': 0, 'violations_accepted': 0})
    }

    for trace in test_traces:
        decision = strategy_fn(trace, *args)
        is_attacked = trace.attack_type != "none"

        if is_attacked:
            results['attacked_traces'] += 1
            if decision == "abstain":
                results['attacked_abstained'] += 1
        else:
            results['benign_traces'] += 1
            if decision == "accept":
                results['benign_accepted'] += 1

        # Track violations that slip through
        if trace.safety_violation and decision == "accept":
            results['violations_accepted'] += 1

            # Track by tool type (use first tool as representative)
            first_tool = trace.tools[0].tool_name
            results['by_tool_type'][first_tool]['violations_accepted'] += 1
            results['by_attack_type'][trace.attack_type]['violations_accepted'] += 1

        # Track totals by group
        first_tool = trace.tools[0].tool_name
        results['by_tool_type'][first_tool]['total'] += 1
        results['by_attack_type'][trace.attack_type]['total'] += 1

    # Compute metrics
    results['benign_utility'] = results['benign_accepted'] / max(1, results['benign_traces'])
    results['safety_rate'] = results['attacked_abstained'] / max(1, results['attacked_traces'])
    results['overall_risk'] = results['violations_accepted'] / results['total']

    # Worst-group risk
    tool_risks = {}
    for tool, data in results['by_tool_type'].items():
        if data['total'] > 0:
            tool_risks[tool] = data['violations_accepted'] / data['total']
    results['worst_group_risk'] = max(tool_risks.values()) if tool_risks else 0
    results['tool_risks'] = tool_risks

    return results

# Run evaluation across multiple seeds
def run_full_evaluation(seed):
    """Run complete evaluation for one seed"""
    np.random.seed(seed)
    traces = all_traces_by_seed[seed]
    train, cal, test = split_data(traces)

    # Compute thresholds
    global_thresh = compute_global_threshold(cal)
    group_thresholds = compute_group_thresholds(cal)

    # Evaluate all strategies
    results = {}
    results['no_guard'] = evaluate_strategy(test, no_guard)
    results['always_abstain'] = evaluate_strategy(test, always_abstain)
    results['heuristic'] = evaluate_strategy(test, heuristic_guard)
    results['unconditional_cp'] = evaluate_strategy(test, unconditional_cp_router, global_thresh)
    results['graph_conditional_cp'] = evaluate_strategy(test, graph_conditional_cp_router, group_thresholds)

    return results

# Run for all seeds and aggregate
all_results = {seed: run_full_evaluation(seed) for seed in SEEDS}

# Aggregate results
def aggregate_results(all_results):
    """Compute mean and std across seeds"""
    strategies = list(all_results[SEEDS[0]].keys())
    metrics = ['benign_utility', 'safety_rate', 'overall_risk', 'worst_group_risk']

    aggregated = {}
    for strategy in strategies:
        aggregated[strategy] = {}
        for metric in metrics:
            values = [all_results[seed][strategy][metric] for seed in SEEDS]
            aggregated[strategy][metric] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
    return aggregated

agg_results = aggregate_results(all_results)

# Write routing comparison table
with open(RESULTS_DIR / "agent_routing_comparison.md", "w") as f:
    f.write("# Agent Routing Strategy Comparison\n\n")
    f.write(f"Results averaged over {len(SEEDS)} random seeds: {SEEDS}\n\n")

    f.write("## Performance Metrics\n\n")
    f.write("| Strategy | Benign Utility | Safety Rate | Overall Risk | Worst-Group Risk |\n")
    f.write("|----------|----------------|-------------|--------------|------------------|\n")

    for strategy, metrics in agg_results.items():
        f.write(f"| {strategy} | "
                f"{metrics['benign_utility']['mean']:.3f}±{metrics['benign_utility']['std']:.3f} | "
                f"{metrics['safety_rate']['mean']:.3f}±{metrics['safety_rate']['std']:.3f} | "
                f"{metrics['overall_risk']['mean']:.3f}±{metrics['overall_risk']['std']:.3f} | "
                f"{metrics['worst_group_risk']['mean']:.3f}±{metrics['worst_group_risk']['std']:.3f} |\n")

    f.write("\n## Key Comparisons\n\n")

    no_guard_utility = agg_results['no_guard']['benign_utility']['mean']
    gc_cp_utility = agg_results['graph_conditional_cp']['benign_utility']['mean']
    utility_drop = (no_guard_utility - gc_cp_utility) / no_guard_utility * 100

    uc_cp_wg = agg_results['unconditional_cp']['worst_group_risk']['mean']
    gc_cp_wg = agg_results['graph_conditional_cp']['worst_group_risk']['mean']
    wg_improvement = (uc_cp_wg - gc_cp_wg) / uc_cp_wg * 100 if uc_cp_wg > 0 else 0

    f.write(f"- **Utility preservation**: Graph-CP utility drop = {utility_drop:.1f}% (vs No-Guard)\n")
    f.write(f"- **Worst-group improvement**: Graph-CP improves worst-group risk by {wg_improvement:.1f}% (vs Unconditional CP)\n")

# Generate comparison plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

strategies = list(agg_results.keys())
x = np.arange(len(strategies))

# Plot 1: Risk-Utility Tradeoff
ax = axes[0]
utilities = [agg_results[s]['benign_utility']['mean'] for s in strategies]
risks = [agg_results[s]['overall_risk']['mean'] for s in strategies]
colors = ['red', 'gray', 'orange', 'blue', 'green']
for i, s in enumerate(strategies):
    ax.scatter(utilities[i], 1-risks[i], s=150, c=colors[i], label=s, alpha=0.8)
ax.set_xlabel('Benign Utility')
ax.set_ylabel('Safety (1 - Risk)')
ax.set_title('Risk-Utility Tradeoff')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 2: Worst-Group Risk Comparison
ax = axes[1]
wg_risks = [agg_results[s]['worst_group_risk']['mean'] for s in strategies]
wg_stds = [agg_results[s]['worst_group_risk']['std'] for s in strategies]
bars = ax.bar(x, wg_risks, yerr=wg_stds, capsize=3, color=colors, alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([s.replace('_', '\n') for s in strategies], fontsize=8)
ax.set_ylabel('Worst-Group Risk')
ax.set_title('Worst-Group Risk by Strategy')

# Plot 3: Safety vs Utility bars
ax = axes[2]
width = 0.35
ax.bar(x - width/2, utilities, width, label='Benign Utility', color='steelblue', alpha=0.8)
ax.bar(x + width/2, [agg_results[s]['safety_rate']['mean'] for s in strategies],
       width, label='Safety Rate', color='forestgreen', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([s.replace('_', '\n') for s in strategies], fontsize=8)
ax.set_ylabel('Rate')
ax.set_title('Utility vs Safety by Strategy')
ax.legend()

plt.tight_layout()
plt.savefig(RESULTS_DIR / "fig_risk_utility_tradeoff.png", dpi=150, bbox_inches='tight')
plt.savefig(RESULTS_DIR / "fig_worst_group_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

# Per-tool risk comparison plot
fig, ax = plt.subplots(figsize=(12, 6))
results_seed42 = all_results[42]
tool_names = list(TOOL_TYPES.values())
tool_names = [t['name'] for t in tool_names]

uc_risks = [results_seed42['unconditional_cp']['tool_risks'].get(t, 0) for t in tool_names]
gc_risks = [results_seed42['graph_conditional_cp']['tool_risks'].get(t, 0) for t in tool_names]

x = np.arange(len(tool_names))
width = 0.35
ax.bar(x - width/2, uc_risks, width, label='Unconditional CP', color='coral', alpha=0.8)
ax.bar(x + width/2, gc_risks, width, label='Graph-Conditional CP', color='seagreen', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(tool_names, rotation=45, ha='right')
ax.set_ylabel('Risk Rate')
ax.set_title('Per-Tool Risk: Unconditional vs Graph-Conditional CP')
ax.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "fig_conditional_vs_unconditional.png", dpi=150, bbox_inches='tight')
plt.close()

print("  Strategy Comparison (mean ± std):")
for strategy, metrics in agg_results.items():
    print(f"    {strategy}:")
    print(f"      Benign Utility: {metrics['benign_utility']['mean']:.3f}±{metrics['benign_utility']['std']:.3f}")
    print(f"      Worst-Group Risk: {metrics['worst_group_risk']['mean']:.3f}±{metrics['worst_group_risk']['std']:.3f}")

# ============================================================
# A5: Kill-Switch Determination
# ============================================================
print("\n" + "=" * 60)
print("AGENT ROUTE KILL-SWITCH DETERMINATION")
print("=" * 60)

# Collect all judgments
judgments = {}

# [1] Worst-group phenomenon exists?
# Check across all dimensions
wg_ratios = {col: result['tests']['ratio'] for col, result in worst_group_results.items()}
wg_pvals = {col: result['tests']['p_value'] for col, result in worst_group_results.items()}

max_ratio = max(wg_ratios.values())
dims_with_ratio_ge3 = sum(1 for r in wg_ratios.values() if r >= 3.0)
dims_with_sig = sum(1 for p in wg_pvals.values() if p < 0.01)

print(f"\n[1] Worst-group phenomenon exists?")
print(f"    Worst/average risk ratios by dimension:")
for col, ratio in wg_ratios.items():
    p = wg_pvals[col]
    status = "✓" if ratio >= 3.0 and p < 0.01 else "✗"
    print(f"      {col}: {ratio:.2f} (p={p:.2e}) {status}")
print(f"    Max ratio = {max_ratio:.2f}")
print(f"    Dimensions with ratio ≥ 3: {dims_with_ratio_ge3}")
print(f"    Dimensions with p < 0.01: {dims_with_sig}")

judgments['wg_exists'] = max_ratio >= 3.0 and dims_with_sig >= 1
print(f"    JUDGMENT: {'YES' if judgments['wg_exists'] else 'NO'}")

# [2] Graph-conditional CP fixes worst-group?
uc_wg = agg_results['unconditional_cp']['worst_group_risk']['mean']
gc_wg = agg_results['graph_conditional_cp']['worst_group_risk']['mean']
wg_improvement = (uc_wg - gc_wg) / uc_wg * 100 if uc_wg > 0 else 0

print(f"\n[2] Graph-conditional CP fixes worst-group?")
print(f"    Unconditional CP worst-group risk = {uc_wg:.3f}")
print(f"    Graph-Conditional CP worst-group risk = {gc_wg:.3f}")
print(f"    Improvement = {wg_improvement:.1f}%")

judgments['gc_fixes_wg'] = wg_improvement >= 30.0
print(f"    JUDGMENT (≥30% improvement): {'YES' if judgments['gc_fixes_wg'] else 'NO'}")

# [3] Benign utility preserved?
no_guard_util = agg_results['no_guard']['benign_utility']['mean']
gc_util = agg_results['graph_conditional_cp']['benign_utility']['mean']
util_drop = (no_guard_util - gc_util) / no_guard_util * 100

print(f"\n[3] Benign utility preserved?")
print(f"    No-Guard benign utility = {no_guard_util:.3f}")
print(f"    Graph-Conditional CP utility = {gc_util:.3f}")
print(f"    Utility drop = {util_drop:.1f}%")

judgments['utility_preserved'] = util_drop < 5.0
print(f"    JUDGMENT (<5% drop): {'YES' if judgments['utility_preserved'] else 'NO'}")

# [4] Conclusions hold across ≥2 dimensions?
dims_passing = []
for col, result in worst_group_results.items():
    if result['tests']['ratio'] >= 3.0 and result['tests']['p_value'] < 0.01:
        dims_passing.append(col)

print(f"\n[4] Conclusions hold across ≥2 dimensions?")
print(f"    Dimensions passing (ratio≥3, p<0.01): {dims_passing}")
print(f"    Count: {len(dims_passing)}")

judgments['multi_dim'] = len(dims_passing) >= 2
print(f"    JUDGMENT: {'YES' if judgments['multi_dim'] else 'NO'}")

# Final S1 determination
s1_pass = all([
    judgments['wg_exists'],
    judgments['gc_fixes_wg'],
    judgments['utility_preserved'],
    judgments['multi_dim']
])

print("\n" + "=" * 60)
print(f"S1 FINAL DETERMINATION: {'YES' if s1_pass else 'NO'}")
print("=" * 60)

if s1_pass:
    print("→ Agent route methodology is VIABLE on simulated data")
    print("→ Proceed with Agent direction + real AgentDojo validation")
else:
    print("→ Agent route does NOT meet all criteria")
    print("→ Execute Part B (RAG route verification)")

    # Identify failure reasons
    failures = []
    if not judgments['wg_exists']:
        failures.append("Worst-group phenomenon not significant (ratio<3 or p≥0.01)")
    if not judgments['gc_fixes_wg']:
        failures.append(f"Graph-CP improvement insufficient ({wg_improvement:.1f}% < 30%)")
    if not judgments['utility_preserved']:
        failures.append(f"Utility drop too large ({util_drop:.1f}% ≥ 5%)")
    if not judgments['multi_dim']:
        failures.append(f"Only {len(dims_passing)} dimension(s) pass, need ≥2")

    print(f"Failure reasons: {'; '.join(failures)}")

# Store S1 result for report
S1_RESULT = s1_pass
S1_DETAILS = {
    'wg_exists': judgments['wg_exists'],
    'gc_fixes_wg': judgments['gc_fixes_wg'],
    'utility_preserved': judgments['utility_preserved'],
    'multi_dim': judgments['multi_dim'],
    'max_wg_ratio': max_ratio,
    'wg_improvement_pct': wg_improvement,
    'utility_drop_pct': util_drop,
    'dims_passing': dims_passing,
    'uc_wg_risk': uc_wg,
    'gc_wg_risk': gc_wg,
    'no_guard_utility': no_guard_util,
    'gc_utility': gc_util
}

# Save S1 result to file for Part B to check
import pickle
with open(RESULTS_DIR / "s1_result.pkl", "wb") as f:
    pickle.dump({'S1_RESULT': S1_RESULT, 'S1_DETAILS': S1_DETAILS}, f)

print("\n[Part A Complete]")
print(f"Results saved to: {RESULTS_DIR}")
