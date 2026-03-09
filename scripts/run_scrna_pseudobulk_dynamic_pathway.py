#!/usr/bin/env python3
"""Run dynamic pathway discovery on scRNA-seq case/control data via pseudobulk.

Input contract:
- input is a .h5ad AnnData object with cells x genes in adata.X
- adata.obs must contain a sample/donor column and a group/condition column
- each biological sample must map to exactly one group label
- hard minimum: 2 pseudobulk samples in each requested group after filtering
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import anndata as ad
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from npathway.data import compute_pseudobulk
from npathway.pipeline import (
    BulkDynamicConfig,
    run_bulk_dynamic_pipeline,
    validate_scrna_pseudobulk_input,
)
from npathway.reporting import DashboardConfig, build_dynamic_dashboard_package

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adata", required=True, help="Path to input AnnData (.h5ad).")
    parser.add_argument(
        "--sample-col",
        required=True,
        help="adata.obs column defining biological replicate / donor / sample ID for pseudobulk aggregation.",
    )
    parser.add_argument(
        "--group-col",
        required=True,
        help="adata.obs column defining the case/control contrast at sample level.",
    )
    parser.add_argument("--group-a", required=True, help="First group label.")
    parser.add_argument("--group-b", required=True, help="Second group label.")
    parser.add_argument(
        "--subset-col",
        default=None,
        help="Optional adata.obs column used to subset cells before pseudobulk (for example cell_type).",
    )
    parser.add_argument(
        "--subset-value",
        default=None,
        help="Optional value in --subset-col to retain before pseudobulk.",
    )
    parser.add_argument(
        "--layer",
        default=None,
        help="Optional AnnData layer to aggregate instead of adata.X / adata.raw.X.",
    )
    parser.add_argument(
        "--use-raw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Aggregate from adata.raw when available (default: true).",
    )
    parser.add_argument(
        "--min-cells-per-sample",
        type=int,
        default=10,
        help="Minimum cells required in each pseudobulk sample after optional subsetting.",
    )
    parser.add_argument(
        "--bulk-raw-counts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat the generated pseudobulk matrix as raw counts for CPM+log1p normalization in the bulk pipeline (default: true).",
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
    _require_int_at_least("--min-cells-per-sample", args.min_cells_per_sample, 1)
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


def _validate_obs_column(adata: ad.AnnData, column: str, role: str) -> None:
    """Validate that an AnnData obs column exists."""
    if column not in adata.obs.columns:
        raise KeyError(
            f"adata.obs is missing {role} column '{column}'. "
            f"Available columns: {list(adata.obs.columns)}"
        )


def _subset_cells(
    adata: ad.AnnData,
    *,
    subset_col: str | None,
    subset_value: str | None,
) -> ad.AnnData:
    """Optionally subset cells before pseudobulk aggregation."""
    if subset_col is None and subset_value is None:
        return adata
    if not subset_col or subset_value is None:
        raise ValueError("Both --subset-col and --subset-value must be provided together.")
    _validate_obs_column(adata, subset_col, "subset")
    mask = adata.obs[subset_col].astype(str) == str(subset_value)
    if int(mask.sum()) == 0:
        raise ValueError(
            f"No cells matched {subset_col}={subset_value!r}. "
            "Check the subset label before running pseudobulk."
        )
    logger.info("Subsetting to %d cells where %s=%s", int(mask.sum()), subset_col, subset_value)
    return adata[mask].copy()


def _build_sample_metadata(
    adata: ad.AnnData,
    *,
    sample_col: str,
    group_col: str,
) -> pd.DataFrame:
    """Extract one row per pseudobulk sample with a stable group label."""
    _validate_obs_column(adata, sample_col, "sample")
    _validate_obs_column(adata, group_col, "group")

    meta = adata.obs[[sample_col, group_col]].copy()
    if meta[sample_col].isna().any():
        raise ValueError(f"adata.obs sample column '{sample_col}' contains missing values.")
    if meta[group_col].isna().any():
        raise ValueError(f"adata.obs group column '{group_col}' contains missing values.")
    meta[sample_col] = meta[sample_col].astype(str)
    meta[group_col] = meta[group_col].astype(str)

    n_groups = meta.groupby(sample_col, observed=False)[group_col].nunique(dropna=False)
    bad = n_groups[n_groups > 1]
    if not bad.empty:
        bad_samples = ", ".join(bad.index.astype(str).tolist())
        raise ValueError(
            "Each pseudobulk sample must map to exactly one group label. "
            f"Conflicting samples: {bad_samples}"
        )

    return meta.drop_duplicates(subset=[sample_col]).reset_index(drop=True)


def _prepare_pseudobulk_inputs(
    *,
    adata: ad.AnnData,
    sample_col: str,
    group_col: str,
    group_a: str,
    group_b: str,
    min_cells_per_sample: int,
    layer: str | None,
    use_raw: bool,
    output_dir: Path,
) -> tuple[Path, Path, Path, pd.DataFrame]:
    """Aggregate pseudobulk and write reproducible matrix/metadata inputs."""
    pb_adata = compute_pseudobulk(
        adata,
        groupby=sample_col,
        layer=layer,
        use_raw=use_raw,
    )
    sample_meta = _build_sample_metadata(adata, sample_col=sample_col, group_col=group_col)

    pb_obs = pb_adata.obs.copy()
    pb_obs[sample_col] = pb_obs[sample_col].astype(str)
    merged_meta = pb_obs[[sample_col, "n_cells"]].merge(
        sample_meta,
        on=sample_col,
        how="left",
        validate="1:1",
    )
    if merged_meta[group_col].isna().any():
        raise ValueError("Failed to align pseudobulk samples with sample-level metadata.")

    if min_cells_per_sample < 1:
        raise ValueError("--min-cells-per-sample must be >= 1.")
    keep = merged_meta["n_cells"].astype(int) >= int(min_cells_per_sample)
    if not bool(keep.any()):
        raise ValueError(
            "No pseudobulk samples remained after applying --min-cells-per-sample."
        )
    pb_adata = pb_adata[keep.to_numpy()].copy()
    merged_meta = merged_meta.loc[keep].reset_index(drop=True)

    labels = merged_meta[group_col].astype(str)
    n_a = int(labels.eq(group_a).sum())
    n_b = int(labels.eq(group_b).sum())
    if n_a < 2 or n_b < 2:
        raise ValueError(
            "Pseudobulk analysis requires at least 2 samples in each requested group "
            f"after filtering. Observed: {group_a}={n_a}, {group_b}={n_b}."
        )

    inputs_dir = output_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    pb_h5ad_path = inputs_dir / "pseudobulk.h5ad"
    pb_adata.write_h5ad(pb_h5ad_path)

    sample_ids = merged_meta[sample_col].astype(str).tolist()
    matrix = pd.DataFrame(
        pb_adata.X.T,
        index=pb_adata.var_names.astype(str),
        columns=sample_ids,
    ).reset_index()
    matrix.columns = ["gene"] + sample_ids
    matrix_path = inputs_dir / "pseudobulk_matrix.csv"
    matrix.to_csv(matrix_path, index=False)

    metadata_path = inputs_dir / "pseudobulk_metadata.csv"
    merged_meta.to_csv(metadata_path, index=False)
    return matrix_path, metadata_path, pb_h5ad_path, merged_meta


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    _validate_cli_args(args)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    report = validate_scrna_pseudobulk_input(
        adata_path=args.adata,
        sample_col=args.sample_col,
        group_col=args.group_col,
        group_a=args.group_a,
        group_b=args.group_b,
        subset_col=args.subset_col,
        subset_value=args.subset_value,
        layer=args.layer,
        use_raw=bool(args.use_raw),
        min_cells_per_sample=args.min_cells_per_sample,
    )
    for warning in report.warnings:
        logging.warning("Input validation: %s", warning)

    outdir = Path(args.output_dir) if args.output_dir else Path(
        "results"
    ) / f"scrna_pseudobulk_dynamic_pathway_{date.today().strftime('%Y%m%d')}"
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading AnnData from %s", args.adata)
    adata = ad.read_h5ad(args.adata)
    adata = _subset_cells(
        adata,
        subset_col=args.subset_col,
        subset_value=args.subset_value,
    )
    matrix_path, metadata_path, pb_h5ad_path, pb_meta = _prepare_pseudobulk_inputs(
        adata=adata,
        sample_col=args.sample_col,
        group_col=args.group_col,
        group_a=args.group_a,
        group_b=args.group_b,
        min_cells_per_sample=args.min_cells_per_sample,
        layer=args.layer,
        use_raw=bool(args.use_raw),
        output_dir=outdir,
    )

    config = BulkDynamicConfig(
        matrix_path=str(matrix_path),
        metadata_path=str(metadata_path),
        group_col=args.group_col,
        group_a=args.group_a,
        group_b=args.group_b,
        sample_col=args.sample_col,
        matrix_orientation="genes_by_samples",
        raw_counts=bool(args.bulk_raw_counts),
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
    )

    result = run_bulk_dynamic_pipeline(config=config, output_dir=outdir)
    print("scRNA pseudobulk dynamic pathway run completed.")
    print(f"- output_dir: {result.output_dir}")
    print(f"- pseudobulk_h5ad: {pb_h5ad_path}")
    print(f"- pseudobulk_matrix: {matrix_path}")
    print(f"- pseudobulk_metadata: {metadata_path}")
    print(f"- n_pseudobulk_samples: {len(pb_meta)}")
    print(f"- n_group_a_samples: {int(pb_meta[args.group_col].astype(str).eq(args.group_a).sum())}")
    print(f"- n_group_b_samples: {int(pb_meta[args.group_col].astype(str).eq(args.group_b).sum())}")
    print(f"- n_programs: {result.n_programs}")
    print(f"- n_sig_de_genes: {result.n_sig_de_genes}")

    if args.with_dashboard:
        dashboard_dir = (
            Path(args.dashboard_output_dir)
            if args.dashboard_output_dir
            else Path(result.output_dir)
        )
        subset_suffix = (
            f" | {args.subset_col}={args.subset_value}"
            if args.subset_col and args.subset_value is not None
            else ""
        )
        dashboard_cfg = DashboardConfig(
            results_dir=result.output_dir,
            output_dir=str(dashboard_dir),
            title=f"nPathway Dashboard: {args.group_a} vs {args.group_b}{subset_suffix}",
            top_k=args.dashboard_top_k,
            include_pdf=not args.dashboard_no_pdf,
        )
        artifacts = build_dynamic_dashboard_package(dashboard_cfg)
        print(f"- dashboard_html: {artifacts.html_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "Hint: validate the input first with "
            "`python scripts/validate_npathway_inputs.py scrna ...`",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
