"""
Visualization Utilities for Hypothesis Verification

Publication-quality plots following academic standards.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List, Optional, Tuple, Union
import os

# Use non-interactive backend for server environments
matplotlib.use('Agg')


def set_publication_style():
    """Set matplotlib style for publication-quality figures"""
    plt.style.use('seaborn-v0_8-whitegrid')

    plt.rcParams.update({
        # Font settings
        'font.family': 'serif',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,

        # Figure settings
        'figure.figsize': (8, 6),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',

        # Line settings
        'lines.linewidth': 2,
        'lines.markersize': 8,

        # Grid settings
        'grid.alpha': 0.3,

        # Legend settings
        'legend.frameon': True,
        'legend.framealpha': 0.9,
    })


def plot_correlation_scatter(
    x: Union[List[float], np.ndarray],
    y: Union[List[float], np.ndarray],
    xlabel: str = "X",
    ylabel: str = "Y",
    title: str = "Correlation Analysis",
    correlation_type: str = "pearson",
    show_regression: bool = True,
    show_ci: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Create scatter plot with correlation analysis

    Args:
        x, y: Data points
        xlabel, ylabel: Axis labels
        title: Plot title
        correlation_type: 'pearson' or 'spearman'
        show_regression: Show regression line
        show_ci: Show 95% confidence interval
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    set_publication_style()
    x = np.array(x)
    y = np.array(y)

    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    ax.scatter(x, y, alpha=0.6, edgecolors='white', linewidth=0.5, s=80, c='steelblue')

    # Compute correlation
    from scipy import stats
    if correlation_type == 'pearson':
        r, p = stats.pearsonr(x, y)
        corr_label = f'Pearson r = {r:.3f} (p = {p:.3e})'
    else:
        r, p = stats.spearmanr(x, y)
        corr_label = f'Spearman ρ = {r:.3f} (p = {p:.3e})'

    # Regression line
    if show_regression:
        z = np.polyfit(x, y, 1)
        p_line = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p_line(x_line), 'r-', linewidth=2, label=corr_label)

        # Confidence interval
        if show_ci:
            # Simple bootstrap CI for the regression line
            n_boot = 1000
            y_boot = np.zeros((n_boot, len(x_line)))
            for i in range(n_boot):
                idx = np.random.choice(len(x), size=len(x), replace=True)
                z_boot = np.polyfit(x[idx], y[idx], 1)
                y_boot[i] = np.poly1d(z_boot)(x_line)

            ci_lower = np.percentile(y_boot, 2.5, axis=0)
            ci_upper = np.percentile(y_boot, 97.5, axis=0)
            ax.fill_between(x_line, ci_lower, ci_upper, alpha=0.2, color='red')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best')

    # Add significance marker
    significance = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    ax.text(0.05, 0.95, significance, transform=ax.transAxes,
            fontsize=14, fontweight='bold', verticalalignment='top')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path)
        print(f"Figure saved to: {save_path}")

    return fig


def plot_grouped_boxplot(
    groups: List[Union[List[float], np.ndarray]],
    group_labels: List[str],
    ylabel: str = "Value",
    title: str = "Group Comparison",
    show_points: bool = True,
    show_significance: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    colors: Optional[List[str]] = None
) -> plt.Figure:
    """
    Create grouped boxplot for comparing distributions

    Args:
        groups: List of data arrays for each group
        group_labels: Labels for each group
        ylabel: Y-axis label
        title: Plot title
        show_points: Show individual data points
        show_significance: Show significance bars
        save_path: Path to save figure
        figsize: Figure size
        colors: Custom colors for groups

    Returns:
        matplotlib Figure object
    """
    set_publication_style()

    fig, ax = plt.subplots(figsize=figsize)

    if colors is None:
        colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))

    # Create boxplot
    bp = ax.boxplot(groups, labels=group_labels, patch_artist=True)

    # Style boxplot
    for i, (patch, color) in enumerate(zip(bp['boxes'], colors)):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add individual points
    if show_points:
        for i, group in enumerate(groups):
            x = np.random.normal(i + 1, 0.04, size=len(group))
            ax.scatter(x, group, alpha=0.4, s=20, c='black', zorder=3)

    # Add significance bars between first and last group
    if show_significance and len(groups) >= 2:
        from scipy import stats

        # Compare first and last groups
        t_stat, p_value = stats.ttest_ind(groups[0], groups[-1])

        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."

        # Draw significance bar
        y_max = max([max(g) for g in groups])
        y_bar = y_max * 1.1
        ax.plot([1, len(groups)], [y_bar, y_bar], 'k-', linewidth=1)
        ax.text((1 + len(groups)) / 2, y_bar * 1.02, significance,
                ha='center', fontsize=12, fontweight='bold')

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path)
        print(f"Figure saved to: {save_path}")

    return fig


def plot_fitted_line(
    x: Union[List[float], np.ndarray],
    y: Union[List[float], np.ndarray],
    xlabel: str = "X",
    ylabel: str = "Y",
    title: str = "Fitted Relationship",
    degree: int = 1,
    equation_position: str = 'upper left',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Create scatter plot with polynomial fitted line

    Args:
        x, y: Data points
        xlabel, ylabel: Axis labels
        title: Plot title
        degree: Polynomial degree for fitting
        equation_position: Position for equation text
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    set_publication_style()
    x = np.array(x)
    y = np.array(y)

    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    ax.scatter(x, y, alpha=0.7, s=100, c='steelblue', edgecolors='white', linewidth=0.5)

    # Fit polynomial
    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = p(x_fit)

    ax.plot(x_fit, y_fit, 'r-', linewidth=2.5, label='Fitted curve')

    # Create equation string
    if degree == 1:
        equation = f'y = {coeffs[0]:.3f}x + {coeffs[1]:.3f}'
    else:
        terms = []
        for i, c in enumerate(coeffs):
            power = degree - i
            if power == 0:
                terms.append(f'{c:.3f}')
            elif power == 1:
                terms.append(f'{c:.3f}x')
            else:
                terms.append(f'{c:.3f}x^{power}')
        equation = 'y = ' + ' + '.join(terms)

    # Compute R-squared
    y_pred = p(x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Add equation and R-squared
    position_map = {
        'upper left': (0.05, 0.95),
        'upper right': (0.95, 0.95),
        'lower left': (0.05, 0.05),
        'lower right': (0.95, 0.05)
    }
    pos = position_map.get(equation_position, (0.05, 0.95))
    ha = 'left' if 'left' in equation_position else 'right'
    va = 'top' if 'upper' in equation_position else 'bottom'

    ax.text(pos[0], pos[1], f'{equation}\nR² = {r_squared:.4f}',
            transform=ax.transAxes, fontsize=11,
            verticalalignment=va, horizontalalignment=ha,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path)
        print(f"Figure saved to: {save_path}")

    return fig


def plot_distribution_comparison(
    dist1: Dict[str, float],
    dist2: Dict[str, float],
    label1: str = "Distribution 1",
    label2: str = "Distribution 2",
    xlabel: str = "Category",
    ylabel: str = "Probability",
    title: str = "Distribution Comparison",
    show_divergence: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    top_k: int = 15
) -> plt.Figure:
    """
    Compare two distributions as bar charts

    Args:
        dist1, dist2: Dictionaries mapping categories to probabilities
        label1, label2: Labels for distributions
        xlabel, ylabel: Axis labels
        title: Plot title
        show_divergence: Show JS divergence value
        save_path: Path to save figure
        figsize: Figure size
        top_k: Show only top K categories

    Returns:
        matplotlib Figure object
    """
    set_publication_style()

    # Get all keys and sort by dist1 values
    all_keys = sorted(set(dist1.keys()) | set(dist2.keys()),
                      key=lambda k: dist1.get(k, 0), reverse=True)[:top_k]

    x = np.arange(len(all_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)

    values1 = [dist1.get(k, 0) for k in all_keys]
    values2 = [dist2.get(k, 0) for k in all_keys]

    bars1 = ax.bar(x - width/2, values1, width, label=label1, color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, values2, width, label=label2, color='coral', alpha=0.8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([k.split('/')[-1][:15] for k in all_keys], rotation=45, ha='right')
    ax.legend()

    # Show JS divergence
    if show_divergence:
        from .statistics import compute_js_divergence
        js = compute_js_divergence(dist1, dist2)
        ax.text(0.95, 0.95, f'JS Divergence = {js:.4f}',
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path)
        print(f"Figure saved to: {save_path}")

    return fig


def plot_heatmap(
    matrix: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str = "Heatmap",
    xlabel: str = "",
    ylabel: str = "",
    cmap: str = "RdYlBu_r",
    annotate: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Create heatmap visualization

    Args:
        matrix: 2D numpy array
        row_labels, col_labels: Labels for rows and columns
        title: Plot title
        xlabel, ylabel: Axis labels
        cmap: Colormap name
        annotate: Show values in cells
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    set_publication_style()

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(matrix, cmap=cmap, aspect='auto')

    # Set ticks
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right')
    ax.set_yticklabels(row_labels)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    # Annotate cells
    if annotate:
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                              ha='center', va='center', color='black', fontsize=8)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path)
        print(f"Figure saved to: {save_path}")

    return fig


def plot_multi_line(
    data: Dict[str, Tuple[List[float], List[float]]],
    xlabel: str = "X",
    ylabel: str = "Y",
    title: str = "Multi-line Plot",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Create multi-line plot

    Args:
        data: Dictionary mapping label to (x_values, y_values) tuples
        xlabel, ylabel: Axis labels
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    set_publication_style()

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.Set1(np.linspace(0, 1, len(data)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h']

    for i, (label, (x, y)) in enumerate(data.items()):
        ax.plot(x, y, '-', color=colors[i], marker=markers[i % len(markers)],
                label=label, markersize=8, linewidth=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path)
        print(f"Figure saved to: {save_path}")

    return fig
