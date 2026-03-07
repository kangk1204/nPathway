"""
nPathway: Foundation Model-Derived Gene Programs for Context-Aware Gene Set Enrichment Analysis.

This package provides tools for extracting gene embeddings from pre-trained
single-cell RNA-seq foundation models, discovering data-driven gene programs,
and performing enrichment analysis that replaces or augments traditional
curated pathway databases (KEGG, GO, Reactome, MSigDB).

Subpackages
-----------
embedding
    Gene embedding extraction from foundation models (scGPT, Geneformer, etc.).
discovery
    Gene program discovery via clustering, topic models, and attention networks.
evaluation
    Benchmarking and evaluation of discovered gene programs.
utils
    I/O utilities, visualization, and helper functions.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "nPathway Contributors"
__license__ = "MIT"

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
from npathway.pipeline import (
    BulkDynamicConfig,
    BulkDynamicResult,
    ValidationReport,
    run_bulk_dynamic_pipeline,
    validate_bulk_input_files,
    validate_scrna_pseudobulk_input,
)
from npathway.reporting import (
    DashboardArtifacts,
    DashboardConfig,
    build_dynamic_dashboard_package,
)
from npathway.validation import (
    SensitivitySummaryTables,
    analyze_perturbation_robustness,
    compare_external_reproducibility,
    summarize_bootstrap_stability,
    summarize_external_reproducibility,
    summarize_hyperparameter_sensitivity,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    # I/O functions
    "read_gmt",
    "read_gmx",
    "write_gmt",
    "write_gmx",
    "programs_to_df",
    "weighted_programs_to_gmt",
    # Visualization functions
    "plot_embedding_umap",
    "plot_program_overlap_heatmap",
    "plot_enrichment_comparison",
    "plot_program_sizes",
    "plot_benchmark_summary",
    "plot_cross_model_consistency",
    "plot_context_specificity",
    # Bulk dynamic pathway pipeline
    "BulkDynamicConfig",
    "BulkDynamicResult",
    "ValidationReport",
    "run_bulk_dynamic_pipeline",
    "validate_bulk_input_files",
    "validate_scrna_pseudobulk_input",
    "DashboardConfig",
    "DashboardArtifacts",
    "build_dynamic_dashboard_package",
    # Validation suite
    "SensitivitySummaryTables",
    "summarize_bootstrap_stability",
    "summarize_hyperparameter_sensitivity",
    "analyze_perturbation_robustness",
    "compare_external_reproducibility",
    "summarize_external_reproducibility",
]
