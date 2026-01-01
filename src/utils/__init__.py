# Utility modules
from .statistics import (
    compute_pearson_correlation,
    compute_spearman_correlation,
    compute_cohens_d,
    compute_js_divergence,
    compute_kl_divergence,
    bootstrap_confidence_interval,
    paired_t_test,
    independent_t_test
)

from .visualization import (
    plot_correlation_scatter,
    plot_grouped_boxplot,
    plot_fitted_line,
    plot_distribution_comparison,
    plot_heatmap,
    set_publication_style
)

__all__ = [
    # Statistics
    'compute_pearson_correlation',
    'compute_spearman_correlation',
    'compute_cohens_d',
    'compute_js_divergence',
    'compute_kl_divergence',
    'bootstrap_confidence_interval',
    'paired_t_test',
    'independent_t_test',
    # Visualization
    'plot_correlation_scatter',
    'plot_grouped_boxplot',
    'plot_fitted_line',
    'plot_distribution_comparison',
    'plot_heatmap',
    'set_publication_style'
]
