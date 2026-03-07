"""Evaluation and benchmarking subpackage.

This subpackage provides tools for evaluating discovered gene programs
against curated pathway databases, perturbation datasets, and other
baselines. It implements the benchmarking framework described in the
nPathway methodology for assessing biological coherence, context-specificity,
statistical power, and cross-model robustness.

Modules
-------
base
    Abstract base class for all benchmarks.
enrichment
    Core enrichment analysis engine (Fisher, GSEA, ssGSEA, weighted GSEA).
metrics
    Shared metric functions (Jaccard, coherence, redundancy, coverage, novelty).
benchmark_perturbation
    Recovery of known biology from perturbation screens (Perturb-seq, CRISPR).
benchmark_context
    Context-specificity evaluation across cell types and tissues.
benchmark_power
    Statistical power comparison in enrichment analysis tasks.
benchmark_robustness
    Cross-model robustness and consensus program identification.
benchmark_real
    Real-data benchmarks using public scRNA-seq datasets and MSigDB.
"""

from __future__ import annotations

from npathway.evaluation.base import BaseBenchmark
from npathway.evaluation.benchmark_context import ContextSpecificityBenchmark
from npathway.evaluation.benchmark_perturbation import PerturbationBenchmark
from npathway.evaluation.benchmark_power import PowerBenchmark
from npathway.evaluation.benchmark_real import (
    CellTypeMarkerBenchmark,
    RealDataBenchmark,
)
from npathway.evaluation.benchmark_robustness import CrossModelBenchmark
from npathway.evaluation.enrichment import (
    aucell_score,
    effect_size_metrics,
    leading_edge_analysis,
    leading_edge_overlap,
    preranked_gsea,
    run_enrichment,
    ssgsea_score,
    weighted_enrichment,
)
from npathway.evaluation.manuscript_consistency import (
    ConsistencyIssue,
    validate_manuscript_consistency,
)
from npathway.evaluation.metrics import (
    adjusted_rand_index,
    benjamini_hochberg_fdr,
    benjamini_hochberg_fdr_grouped,
    biological_coherence,
    bootstrap_metric,
    compare_methods,
    comprehensive_evaluation,
    compute_overlap_matrix,
    coverage,
    effect_size_cohens_d,
    fold_enrichment_with_ci,
    functional_coherence_go,
    inter_program_distance,
    jaccard_similarity,
    normalized_mutual_info,
    novelty_score,
    permutation_test_enrichment,
    paired_rank_biserial_correlation,
    program_redundancy,
    program_specificity,
    program_stability,
    programs_to_labels,
    regulatory_coherence,
)
from npathway.evaluation.submission_stats import (
    GateFailure,
    SubmissionStatsConfig,
    SubmissionStatsResult,
    build_submission_stats_package,
)

__all__ = [
    # Base
    "BaseBenchmark",
    # Benchmarks
    "PerturbationBenchmark",
    "ContextSpecificityBenchmark",
    "PowerBenchmark",
    "CrossModelBenchmark",
    "RealDataBenchmark",
    "CellTypeMarkerBenchmark",
    # Enrichment
    "run_enrichment",
    "preranked_gsea",
    "ssgsea_score",
    "weighted_enrichment",
    "aucell_score",
    "effect_size_metrics",
    "leading_edge_analysis",
    "leading_edge_overlap",
    # Metrics - Basic
    "jaccard_similarity",
    "compute_overlap_matrix",
    "biological_coherence",
    "program_redundancy",
    "coverage",
    "novelty_score",
    "benjamini_hochberg_fdr",
    "benjamini_hochberg_fdr_grouped",
    "adjusted_rand_index",
    "normalized_mutual_info",
    "programs_to_labels",
    # Metrics - Advanced
    "program_stability",
    "bootstrap_metric",
    "permutation_test_enrichment",
    "functional_coherence_go",
    "regulatory_coherence",
    "program_specificity",
    "inter_program_distance",
    "comprehensive_evaluation",
    "compare_methods",
    "paired_rank_biserial_correlation",
    "effect_size_cohens_d",
    "fold_enrichment_with_ci",
    # Manuscript consistency
    "ConsistencyIssue",
    "validate_manuscript_consistency",
    # Submission statistics package
    "GateFailure",
    "SubmissionStatsConfig",
    "SubmissionStatsResult",
    "build_submission_stats_package",
]
