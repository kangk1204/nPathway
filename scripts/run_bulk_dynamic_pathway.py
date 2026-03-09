#!/usr/bin/env python3
"""Run dynamic pathway discovery on bulk RNA-seq for a two-group contrast.

Input contract:
- matrix CSV/TSV: first column is gene ID, remaining columns are samples by default
- metadata CSV/TSV: one row per sample with sample and group columns
- hard minimum: 2 samples in each requested group
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from npathway.pipeline import (
    BulkDynamicConfig,
    run_bulk_dynamic_pipeline,
    validate_bulk_input_files,
)
from npathway.reporting import DashboardConfig, build_dynamic_dashboard_package


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        required=True,
        help="Path to gene/sample matrix (CSV/TSV). First column must be gene ID unless using samples_by_genes orientation.",
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="Path to metadata table (CSV/TSV) with one row per sample.",
    )
    parser.add_argument(
        "--group-col",
        required=True,
        help="Metadata column defining the two contrast groups.",
    )
    parser.add_argument("--group-a", required=True, help="First group label.")
    parser.add_argument("--group-b", required=True, help="Second group label.")
    parser.add_argument("--sample-col", default="sample", help="Metadata sample ID column.")
    parser.add_argument(
        "--matrix-orientation",
        default="genes_by_samples",
        choices=["genes_by_samples", "samples_by_genes"],
        help="Input matrix orientation. genes_by_samples means rows=genes and columns=samples after the first ID column.",
    )
    parser.add_argument(
        "--sep",
        default=None,
        help="Delimiter for matrix/metadata (default: auto infer).",
    )
    parser.add_argument(
        "--raw-counts",
        action="store_true",
        help="Treat matrix values as raw counts and apply CPM+log1p normalization.",
    )
    parser.add_argument(
        "--ranked-genes",
        default=None,
        help="Optional external ranked gene table. When provided, nPathway GSEA uses this ranking instead of building one from the internal DE table.",
    )
    parser.add_argument(
        "--ranked-gene-col",
        default="gene",
        help="Gene column name in --ranked-genes.",
    )
    parser.add_argument(
        "--ranked-score-col",
        default="score",
        help="Score column name in --ranked-genes.",
    )
    parser.add_argument(
        "--ranked-sep",
        default=None,
        help="Delimiter override for --ranked-genes (default: auto infer).",
    )
    parser.add_argument(
        "--discovery-method",
        default="ensemble",
        choices=["ensemble", "kmeans", "leiden", "spectral", "hdbscan"],
        help="Program discovery method (default: ensemble = consensus of kmeans+leiden).",
    )
    parser.add_argument(
        "--n-programs",
        type=int,
        default=20,
        help="Target number of programs for kmeans/spectral.",
    )
    parser.add_argument("--k-neighbors", type=int, default=15, help="kNN neighbors.")
    parser.add_argument("--resolution", type=float, default=1.0, help="Leiden resolution.")
    parser.add_argument("--n-components", type=int, default=30, help="Embedding dimension.")
    parser.add_argument("--n-diffusion-steps", type=int, default=3, help="Diffusion iterations.")
    parser.add_argument("--diffusion-alpha", type=float, default=0.5, help="Diffusion self-weight.")
    parser.add_argument(
        "--de-test",
        default="welch",
        choices=["welch", "mwu"],
        help="DE test for two-group comparison.",
    )
    parser.add_argument("--de-alpha", type=float, default=0.05, help="DE FDR threshold.")
    parser.add_argument(
        "--min-abs-logfc-for-claim",
        type=float,
        default=0.2,
        help="Minimum mean abs(logFC) gate for claim support.",
    )
    parser.add_argument("--gsea-n-perm", type=int, default=1000, help="GSEA permutations.")
    parser.add_argument(
        "--min-genes-per-program-claim",
        type=int,
        default=10,
        help="Minimum program size gate for claim support.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=0,
        help="Bootstrap runs for stability estimation (0 disables).",
    )
    parser.add_argument(
        "--min-stability-for-claim",
        type=float,
        default=0.25,
        help="Minimum stability gate when bootstrap is enabled.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--annotate-programs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Annotate programs with closest reference sets (MSigDB/custom GMT). Default: true.",
    )
    parser.add_argument(
        "--annotation-collections",
        default="hallmark,go_bp,kegg",
        help=(
            "Comma-separated annotation collections. Supports MSigDB collections "
            "(for example hallmark, go_bp, go_cc, go_mf, kegg, c2_cp, c7, "
            "msigdb_reactome) and public collections "
            "(reactome, wikipathways, pathwaycommons)."
        ),
    )
    parser.add_argument(
        "--annotation-species",
        default="human",
        choices=["human", "mouse"],
        help="Species for annotation collections and public reference downloads.",
    )
    parser.add_argument(
        "--annotation-gmt",
        default=None,
        help="Optional custom GMT file for program naming annotation.",
    )
    parser.add_argument(
        "--annotation-topk-per-program",
        type=int,
        default=15,
        help="Top reference matches saved per program for heatmap/reporting.",
    )
    parser.add_argument(
        "--annotation-min-jaccard-for-label",
        type=float,
        default=0.03,
        help="Minimum Jaccard to use reference-derived label (else Unmatched).",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory.")
    parser.add_argument(
        "--with-dashboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build interactive dashboard package after the run. Default: true.",
    )
    parser.add_argument(
        "--dashboard-output-dir",
        default=None,
        help="Dashboard output directory (default: <output-dir>/dashboard).",
    )
    parser.add_argument(
        "--dashboard-top-k",
        type=int,
        default=20,
        help="Top-K rows used in dashboard focused plots.",
    )
    parser.add_argument(
        "--dashboard-no-pdf",
        action="store_true",
        help="Dashboard figures PNG only (skip PDF).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable INFO logs.")
    return parser.parse_args()


def _require_int_at_least(name: str, value: int, minimum: int) -> None:
    """Validate an integer CLI argument lower bound."""
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")


def _require_float_range(
    name: str,
    value: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    min_inclusive: bool = True,
    max_inclusive: bool = True,
) -> None:
    """Validate a float CLI argument range."""
    if minimum is not None:
        if min_inclusive and value < minimum:
            raise ValueError(f"{name} must be >= {minimum}.")
        if not min_inclusive and value <= minimum:
            raise ValueError(f"{name} must be > {minimum}.")
    if maximum is not None:
        if max_inclusive and value > maximum:
            raise ValueError(f"{name} must be <= {maximum}.")
        if not max_inclusive and value >= maximum:
            raise ValueError(f"{name} must be < {maximum}.")


def _validate_cli_args(args: argparse.Namespace) -> None:
    """Fail fast on malformed CLI parameters."""
    if args.group_a == args.group_b:
        raise ValueError("--group-a and --group-b must be different labels.")
    _require_int_at_least("--n-programs", args.n_programs, 1)
    _require_int_at_least("--k-neighbors", args.k_neighbors, 1)
    _require_float_range("--resolution", args.resolution, minimum=0.0, min_inclusive=False)
    _require_int_at_least("--n-components", args.n_components, 1)
    _require_int_at_least("--n-diffusion-steps", args.n_diffusion_steps, 0)
    _require_float_range("--diffusion-alpha", args.diffusion_alpha, minimum=0.0, maximum=1.0)
    _require_float_range("--de-alpha", args.de_alpha, minimum=0.0, maximum=1.0, min_inclusive=False)
    _require_float_range(
        "--min-abs-logfc-for-claim",
        args.min_abs_logfc_for_claim,
        minimum=0.0,
    )
    _require_int_at_least("--gsea-n-perm", args.gsea_n_perm, 1)
    _require_int_at_least("--min-genes-per-program-claim", args.min_genes_per_program_claim, 1)
    _require_int_at_least("--n-bootstrap", args.n_bootstrap, 0)
    _require_float_range("--min-stability-for-claim", args.min_stability_for_claim, minimum=0.0, maximum=1.0)
    _require_int_at_least("--annotation-topk-per-program", args.annotation_topk_per_program, 1)
    _require_float_range(
        "--annotation-min-jaccard-for-label",
        args.annotation_min_jaccard_for_label,
        minimum=0.0,
        maximum=1.0,
    )
    _require_int_at_least("--dashboard-top-k", args.dashboard_top_k, 1)


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    _validate_cli_args(args)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    report = validate_bulk_input_files(
        matrix_path=args.matrix,
        metadata_path=args.metadata,
        sample_col=args.sample_col,
        group_col=args.group_col,
        group_a=args.group_a,
        group_b=args.group_b,
        matrix_orientation=args.matrix_orientation,
        sep=args.sep,
        raw_counts=bool(args.raw_counts),
    )
    for warning in report.warnings:
        logging.warning("Input validation: %s", warning)

    outdir = Path(args.output_dir) if args.output_dir else Path(
        "results"
    ) / f"bulk_dynamic_pathway_{date.today().strftime('%Y%m%d')}"

    config = BulkDynamicConfig(
        matrix_path=args.matrix,
        metadata_path=args.metadata,
        group_col=args.group_col,
        group_a=args.group_a,
        group_b=args.group_b,
        sample_col=args.sample_col,
        matrix_orientation=args.matrix_orientation,
        sep=args.sep,
        raw_counts=bool(args.raw_counts),
        discovery_method=args.discovery_method,
        n_programs=args.n_programs,
        k_neighbors=args.k_neighbors,
        resolution=args.resolution,
        n_components=args.n_components,
        n_diffusion_steps=args.n_diffusion_steps,
        diffusion_alpha=args.diffusion_alpha,
        de_test=args.de_test,
        de_alpha=args.de_alpha,
        min_abs_logfc_for_claim=args.min_abs_logfc_for_claim,
        gsea_n_perm=args.gsea_n_perm,
        min_genes_per_program_claim=args.min_genes_per_program_claim,
        n_bootstrap=args.n_bootstrap,
        min_stability_for_claim=args.min_stability_for_claim,
        random_seed=args.seed,
        annotate_programs=bool(args.annotate_programs),
        annotation_collections=tuple(
            x.strip() for x in str(args.annotation_collections).split(",") if x.strip()
        ),
        annotation_species=args.annotation_species,
        annotation_gmt_path=args.annotation_gmt,
        annotation_topk_per_program=args.annotation_topk_per_program,
        annotation_min_jaccard_for_label=args.annotation_min_jaccard_for_label,
        ranked_genes_path=args.ranked_genes,
        ranked_genes_sep=args.ranked_sep,
        ranked_gene_col=args.ranked_gene_col,
        ranked_score_col=args.ranked_score_col,
    )

    result = run_bulk_dynamic_pipeline(config=config, output_dir=outdir)
    print("Bulk dynamic pathway run completed.")
    print(f"- output_dir: {result.output_dir}")
    print(f"- n_samples: {result.n_samples}")
    print(f"- n_genes: {result.n_genes}")
    print(f"- n_programs: {result.n_programs}")
    print(f"- n_sig_de_genes: {result.n_sig_de_genes}")
    if result.stability_mean_best_match_jaccard is not None:
        print(
            "- stability_mean_best_match_jaccard: "
            f"{result.stability_mean_best_match_jaccard:.3f}"
        )
    if result.n_annotated_programs is not None:
        print(f"- n_annotated_programs: {result.n_annotated_programs}")

    if args.with_dashboard:
        dashboard_dir = (
            Path(args.dashboard_output_dir)
            if args.dashboard_output_dir
            else Path(result.output_dir)
        )
        dashboard_cfg = DashboardConfig(
            results_dir=result.output_dir,
            output_dir=str(dashboard_dir),
            title=f"nPathway Dashboard: {args.group_a} vs {args.group_b}",
            top_k=args.dashboard_top_k,
            include_pdf=not args.dashboard_no_pdf,
        )
        artifacts = build_dynamic_dashboard_package(dashboard_cfg)
        print("- dashboard_html:", artifacts.html_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "Hint: validate the input first with "
            "`python scripts/validate_npathway_inputs.py bulk ...`",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
