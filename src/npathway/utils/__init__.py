"""
Utility modules for nPathway.

This subpackage contains I/O helpers for gene set file formats (GMT, GMX)
and visualization routines for gene program analysis and benchmarking.

Modules
-------
gmt_io
    Read/write GMT and GMX gene set files, convert between representations.
visualization
    Plotting functions for embeddings, heatmaps, benchmarks, and comparisons.
"""

from __future__ import annotations

from npathway.utils.gmt_io import (
    programs_to_df,
    read_gmt,
    read_gmx,
    weighted_programs_to_gmt,
    write_gmt,
    write_gmx,
)
from npathway.utils.visualization import (
    plot_benchmark_summary,
    plot_context_specificity,
    plot_cross_model_consistency,
    plot_embedding_umap,
    plot_enrichment_comparison,
    plot_program_overlap_heatmap,
    plot_program_sizes,
)

__all__ = [
    "read_gmt",
    "read_gmx",
    "write_gmt",
    "write_gmx",
    "programs_to_df",
    "weighted_programs_to_gmt",
    "plot_embedding_umap",
    "plot_program_overlap_heatmap",
    "plot_enrichment_comparison",
    "plot_program_sizes",
    "plot_benchmark_summary",
    "plot_cross_model_consistency",
    "plot_context_specificity",
]
