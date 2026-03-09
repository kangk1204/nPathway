"""Installed demo runner for nPathway."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from npathway.data import compute_pseudobulk
from npathway.pipeline import (
    BulkDynamicConfig,
    run_bulk_dynamic_pipeline,
    validate_bulk_input_files,
    validate_scrna_pseudobulk_input,
)
from npathway.reporting import DashboardConfig, build_dynamic_dashboard_package

logger = logging.getLogger(__name__)

_BULK_DEMO_GENES: tuple[str, ...] = (
    "HEG1", "IRAK1", "NES", "FBXO2", "FOXO4", "ITIH5", "NWD1", "TXNRD1", "NPIPA8", "RHOQ",
    "PPP2R1B", "ADNP2", "SLC7A1", "NQO1", "SAP30L", "MAFK", "CLIP2", "MAPK4", "LINC00472",
    "UNC5B", "RGS5", "CHI3L1", "CAVIN1", "SYNM", "KIF1C", "PFKP", "NPAS2", "JMJD6", "ABL2",
    "COL27A1", "DYNC1LI2", "INSIG1", "SAMD4A", "CYP2U1-AS1", "PLOD2", "SMTN", "FRAS1",
    "SLC6A12", "HECA", "RBMS2", "ITPRID2", "WNT5A", "MTCO2P12", "FN1", "USPL1", "DOCK1",
    "NHERF2", "FLT1", "SLC5A3", "EPAS1", "LINC02449", "MT1M", "TUBA1C", "CDKL5", "SYNE2",
    "CCPG1", "PHF21A", "PLCB3", "COG1", "ITGA1", "NOX4", "HAP1", "FAM66D", "KLHL2",
    "ADAM9", "ITGB1", "SCARNA2", "SFT2D2", "CALU", "CD34", "PI4KAP1", "HSPE1", "NPTX2",
    "ENAH", "STK24", "TMTC1", "WFS1", "CMTM4", "PTMA", "HSPA6", "CSNK1A1", "APLN", "WNT2B",
    "RCOR3", "GTF2IP4", "BAIAP3", "ELOVL5", "COL11A1", "MTR", "TMCC2", "RNU4-1", "GDF11",
    "MPHOSPH9", "BACE2", "AIF1L", "AKAP12", "PDE10A", "COL5A3", "PIK3R3", "CD59", "LDLR",
    "TUBAL3", "QSER1", "ANKRD9", "PIP5K1C", "MAP4K4", "TARBP1", "CHD3", "EMP1", "PITRM1",
    "HIPK2", "DDR2", "HMBOX1", "LINC02822", "IFRD1", "FZD4", "QDPR", "TRIM41", "ELL2",
    "MT1X", "MT1F", "ZNF710-AS1", "TXNIP", "TUBB6", "ICA1", "AGO4", "SLCO4A1", "PTPRB",
)

_SCRNA_DEMO_GENES: tuple[str, ...] = (
    "IL7R", "LTB", "MALAT1", "IL32", "TRBC1", "TRBC2", "CD3D", "CD3E", "CD2", "PTPRC",
    "CCR7", "SELL", "LTBP1", "MAL", "SAT1", "TIGIT", "CXCR4", "GZMK", "CCL5", "NKG7",
    "CTSW", "IFITM1", "IFITM3", "HLA-DRA", "HLA-DRB1", "TYMP", "CXCL13", "LST1", "FCER1G",
    "AIF1", "TYROBP", "TREM2", "APOE", "SPP1", "FTH1", "FTL", "B2M", "HLA-A", "HLA-B",
    "STAT1", "IRF7", "ISG15", "IFI6", "MX1", "OAS1", "IFI44L", "SAMHD1", "PSMB8",
)


def _default_output_dir(mode: str) -> Path:
    stamp = date.today().strftime("%Y%m%d")
    return Path("results") / f"npathway_demo_{mode}_{stamp}"


def _write_gmt(path: Path, gene_sets: dict[str, list[str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for name, genes in gene_sets.items():
            handle.write(name + "\tNA\t" + "\t".join(genes) + "\n")


def _generate_bulk_demo_inputs(output_dir: Path) -> tuple[Path, Path, Path]:
    inputs_dir = output_dir / "demo_inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    genes = list(_BULK_DEMO_GENES)
    samples = [f"S{i}" for i in range(1, 13)]

    matrix = rng.normal(loc=6.0, scale=0.45, size=(len(genes), len(samples)))
    matrix[:24, :6] += 1.8
    matrix[24:48, 6:] += 1.6
    matrix = np.clip(matrix, 0.0, None)

    matrix_df = pd.DataFrame(matrix, index=genes, columns=samples).reset_index()
    matrix_df.columns = ["gene"] + samples
    matrix_path = inputs_dir / "bulk_matrix_case_ctrl_demo.csv"
    matrix_df.to_csv(matrix_path, index=False)

    metadata = pd.DataFrame(
        {
            "sample": samples,
            "condition": ["case"] * 6 + ["control"] * 6,
        }
    )
    metadata_path = inputs_dir / "bulk_metadata_case_ctrl_demo.csv"
    metadata.to_csv(metadata_path, index=False)

    gmt_path = inputs_dir / "bulk_reference_demo.gmt"
    _write_gmt(
        gmt_path,
        {
            "Case_Inflammatory_Module": genes[:24],
            "Control_Metabolic_Module": genes[24:48],
            "Shared_Background_Module": genes[60:95],
        },
    )
    return matrix_path, metadata_path, gmt_path


def _build_sample_metadata(adata: ad.AnnData, sample_col: str, group_col: str) -> pd.DataFrame:
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
        raise ValueError(
            "Each pseudobulk sample must map to exactly one group label. "
            f"Conflicting samples: {', '.join(bad.index.astype(str).tolist())}"
        )
    return meta.drop_duplicates(subset=[sample_col]).reset_index(drop=True)


def _generate_scrna_demo_inputs(output_dir: Path) -> tuple[Path, Path]:
    inputs_dir = output_dir / "demo_inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    genes = list(_SCRNA_DEMO_GENES[:40])
    donors = ["case_1", "case_2", "ctrl_1", "ctrl_2"]
    rows: list[np.ndarray] = []
    obs_rows: list[dict[str, str]] = []

    for donor in donors:
        condition = "case" if donor.startswith("case") else "control"
        for _ in range(12):
            counts = rng.poisson(lam=4.0, size=len(genes)).astype(np.float32)
            if condition == "case":
                counts[:8] += 6.0
            else:
                counts[8:16] += 5.0
            rows.append(counts)
            obs_rows.append(
                {
                    "donor_id": donor,
                    "condition": condition,
                    "cell_type": "CD4_T",
                }
            )

    adata = ad.AnnData(
        X=np.vstack(rows),
        obs=pd.DataFrame(obs_rows, index=[f"cell_{i}" for i in range(len(rows))]),
        var=pd.DataFrame(index=genes),
    )
    adata.raw = adata.copy()
    adata_path = inputs_dir / "demo_scrna_case_ctrl.h5ad"
    adata.write_h5ad(adata_path)

    gmt_path = inputs_dir / "scrna_reference_demo.gmt"
    _write_gmt(
        gmt_path,
        {
            "Case_Tcell_Activation": genes[:16],
            "Control_Homeostatic_Module": genes[8:24],
            "Shared_Tcell_Background": genes[24:36],
        },
    )
    return adata_path, gmt_path


def _run_bulk_demo(output_dir: Path, *, with_dashboard: bool) -> None:
    matrix_path, metadata_path, gmt_path = _generate_bulk_demo_inputs(output_dir)
    report = validate_bulk_input_files(
        matrix_path=matrix_path,
        metadata_path=metadata_path,
        sample_col="sample",
        group_col="condition",
        group_a="case",
        group_b="control",
        raw_counts=False,
    )
    for warning in report.warnings:
        logger.warning("Input validation: %s", warning)

    result = run_bulk_dynamic_pipeline(
        config=BulkDynamicConfig(
            matrix_path=str(matrix_path),
            metadata_path=str(metadata_path),
            group_col="condition",
            group_a="case",
            group_b="control",
            sample_col="sample",
            matrix_orientation="genes_by_samples",
            raw_counts=False,
            discovery_method="kmeans",
            n_programs=8,
            n_components=8,
            gsea_n_perm=50,
            annotate_programs=True,
            annotation_collections=tuple(),
            annotation_gmt_path=str(gmt_path),
            annotation_topk_per_program=8,
        ),
        output_dir=output_dir,
    )
    print("nPathway bulk demo completed.")
    print(f"- output_dir: {result.output_dir}")
    print(f"- demo_matrix: {matrix_path}")
    print(f"- demo_metadata: {metadata_path}")
    print(f"- demo_annotation_gmt: {gmt_path}")
    print(f"- n_programs: {result.n_programs}")

    if with_dashboard:
        dashboard_dir = Path(result.output_dir)
        artifacts = build_dynamic_dashboard_package(
            DashboardConfig(
                results_dir=result.output_dir,
                output_dir=str(dashboard_dir),
                title="nPathway Demo Dashboard: case vs control",
                top_k=12,
                include_pdf=True,
            )
        )
        print(f"- dashboard_html: {artifacts.html_path}")


def _run_scrna_demo(output_dir: Path, *, with_dashboard: bool) -> None:
    adata_path, gmt_path = _generate_scrna_demo_inputs(output_dir)
    report = validate_scrna_pseudobulk_input(
        adata_path=adata_path,
        sample_col="donor_id",
        group_col="condition",
        group_a="case",
        group_b="control",
        subset_col="cell_type",
        subset_value="CD4_T",
        use_raw=True,
        min_cells_per_sample=10,
    )
    for warning in report.warnings:
        logger.warning("Input validation: %s", warning)

    adata = ad.read_h5ad(adata_path)
    pb_adata = compute_pseudobulk(adata, groupby="donor_id", use_raw=True)
    sample_meta = _build_sample_metadata(adata, sample_col="donor_id", group_col="condition")
    pb_obs = pb_adata.obs.copy()
    pb_obs["donor_id"] = pb_obs["donor_id"].astype(str)
    merged_meta = pb_obs[["donor_id", "n_cells"]].merge(
        sample_meta,
        on="donor_id",
        how="left",
        validate="1:1",
    )

    keep = merged_meta["n_cells"].astype(int) >= 10
    pb_adata = pb_adata[keep.to_numpy()].copy()
    merged_meta = merged_meta.loc[keep].reset_index(drop=True)

    inputs_dir = output_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    pb_h5ad_path = inputs_dir / "pseudobulk.h5ad"
    pb_adata.write_h5ad(pb_h5ad_path)

    sample_ids = merged_meta["donor_id"].astype(str).tolist()
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

    result = run_bulk_dynamic_pipeline(
        config=BulkDynamicConfig(
            matrix_path=str(matrix_path),
            metadata_path=str(metadata_path),
            group_col="condition",
            group_a="case",
            group_b="control",
            sample_col="donor_id",
            matrix_orientation="genes_by_samples",
            raw_counts=True,
            discovery_method="kmeans",
            n_programs=4,
            n_components=3,
            gsea_n_perm=50,
            annotate_programs=True,
            annotation_collections=tuple(),
            annotation_gmt_path=str(gmt_path),
            annotation_topk_per_program=6,
        ),
        output_dir=output_dir,
    )
    print("nPathway scRNA demo completed.")
    print(f"- output_dir: {result.output_dir}")
    print(f"- demo_adata: {adata_path}")
    print(f"- pseudobulk_h5ad: {pb_h5ad_path}")
    print(f"- pseudobulk_matrix: {matrix_path}")
    print(f"- pseudobulk_metadata: {metadata_path}")
    print(f"- demo_annotation_gmt: {gmt_path}")
    print(f"- n_programs: {result.n_programs}")

    if with_dashboard:
        dashboard_dir = Path(result.output_dir)
        artifacts = build_dynamic_dashboard_package(
            DashboardConfig(
                results_dir=result.output_dir,
                output_dir=str(dashboard_dir),
                title="nPathway Demo Dashboard: scRNA case vs control",
                top_k=12,
                include_pdf=True,
            )
        )
        print(f"- dashboard_html: {artifacts.html_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse demo runner CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="Enable INFO logging.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    bulk = subparsers.add_parser("bulk", help="Run the installed bulk demo.")
    bulk.add_argument("--output-dir", default=None, help="Output directory for bulk demo results.")
    bulk.add_argument(
        "--with-dashboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build the interactive HTML dashboard (default: true).",
    )

    scrna = subparsers.add_parser("scrna", help="Run the installed scRNA pseudobulk demo.")
    scrna.add_argument("--output-dir", default=None, help="Output directory for scRNA demo results.")
    scrna.add_argument(
        "--with-dashboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build the interactive HTML dashboard (default: true).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    try:
        args = parse_args(argv)
        logging.basicConfig(
            level=logging.INFO if args.verbose else logging.WARNING,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
        output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(args.mode)
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.mode == "bulk":
            _run_bulk_demo(output_dir, with_dashboard=bool(args.with_dashboard))
        else:
            _run_scrna_demo(output_dir, with_dashboard=bool(args.with_dashboard))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
