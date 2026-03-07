"""Pipeline entrypoints for end-to-end nPathway workflows."""

from __future__ import annotations

from npathway.pipeline.bulk_dynamic import (
    BulkDynamicConfig,
    BulkDynamicResult,
    run_bulk_dynamic_pipeline,
)
from npathway.pipeline.gsea_comparison import (
    GSEAComparisonResult,
    compare_curated_vs_dynamic_gsea,
    load_ranked_gene_table,
)
from npathway.pipeline.input_validation import (
    ValidationReport,
    validate_bulk_input_files,
    validate_scrna_pseudobulk_input,
)

__all__ = [
    "BulkDynamicConfig",
    "BulkDynamicResult",
    "run_bulk_dynamic_pipeline",
    "GSEAComparisonResult",
    "compare_curated_vs_dynamic_gsea",
    "load_ranked_gene_table",
    "ValidationReport",
    "validate_bulk_input_files",
    "validate_scrna_pseudobulk_input",
]
