"""Beginner-friendly scRNA pseudobulk workflow with auto-detection and cell-type batch execution."""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd

from npathway.cli.bulk_workflow import _detect_r_dependencies, run_batch_aware_bulk_workflow
from npathway.data import compute_pseudobulk
from npathway.pipeline import BulkDynamicConfig, run_bulk_dynamic_pipeline
from npathway.pipeline.gsea_comparison import compare_curated_vs_dynamic_gsea
from npathway.reporting import DashboardConfig, build_dynamic_dashboard_package

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Auto-detected scRNA column mapping and associated warnings."""
    sample_col: str | None
    group_col: str | None
    cell_type_col: str | None
    batch_col: str | None
    covariate_cols: list[str]
    warnings: list[str]


@dataclass
class CellTypePlan:
    """Execution plan and eligibility summary for one cell-type subset."""
    label: str
    total_cells: int
    retained_samples: int
    n_group_a: int
    n_group_b: int
    eligible: bool
    reason: str


@dataclass
class CellTypeRunResult:
    """Serialized outcome for one executed cell-type analysis run."""
    label: str
    output_dir: str
    dashboard_html: str | None
    batch_qc_dir: str | None
    status: str
    total_cells: int
    retained_samples: int
    n_group_a: int
    n_group_b: int
    ranking_source: str
    n_programs: int
    n_sig_de_genes: int
    comparison_dir: str | None
    anchor_program: str | None
    anchor_reference: str | None
    anchor_jaccard: float | None


_SAMPLE_CANDIDATES = (
    "donor_id",
    "donor",
    "sample_id",
    "sample",
    "patient_id",
    "patient",
    "subject_id",
    "subject",
    "individual",
    "orig_ident",
    "orig.ident",
)
_GROUP_CANDIDATES = (
    "condition",
    "group",
    "diagnosis",
    "disease",
    "status",
    "case_control",
    "casecontrol",
    "phenotype",
)
_CELLTYPE_CANDIDATES = (
    "cell_type",
    "celltype",
    "broad_cell_type",
    "major_cell_type",
    "annotation",
    "cell_label",
    "celllabel",
    "cell_ontology_class",
    "cluster_annotation",
    "cell_class",
)
_BATCH_CANDIDATES = (
    "batch",
    "batch_id",
    "library",
    "library_id",
    "dataset",
    "run",
    "plate",
    "pool",
    "chemistry",
    "lane",
)
_COVARIATE_CANDIDATES = ("age", "sex", "gender", "pmi", "rin")


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    return slug or "all_cells"


def _csv_to_tuple(value: str | None) -> tuple[str, ...]:
    if value is None:
        return tuple()
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def _pick_best_column(
    obs: pd.DataFrame,
    candidates: tuple[str, ...],
    *,
    require_repeated: bool = False,
    min_unique: int = 2,
    max_unique: int | None = None,
    exclude: set[str] | None = None,
) -> str | None:
    exclude = exclude or set()
    obs_columns = list(obs.columns)
    scored: list[tuple[int, str]] = []
    for column in obs_columns:
        if column in exclude:
            continue
        values = obs[column].dropna().astype(str)
        if values.empty:
            continue
        nunique = int(values.nunique())
        if nunique < min_unique:
            continue
        if require_repeated and nunique >= len(values):
            continue
        if max_unique is not None and nunique > max_unique:
            continue
        norm_col = _normalize(column)
        score = -999
        for rank, candidate in enumerate(candidates):
            norm_candidate = _normalize(candidate)
            if norm_col == norm_candidate:
                score = max(score, 200 - rank)
            elif norm_candidate in norm_col:
                score = max(score, 120 - rank)
        if score <= -999:
            continue
        if require_repeated:
            score += 10
        score -= nunique // 100
        scored.append((score, column))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], obs_columns.index(item[1])))
    return scored[0][1]


def detect_obs_columns(
    adata: ad.AnnData,
    *,
    group_col: str | None,
    sample_col: str | None,
    cell_type_col: str | None,
    batch_col: str | None,
    covariate_cols: tuple[str, ...],
) -> DetectionResult:
    obs = adata.obs.copy()
    warnings: list[str] = []

    if group_col is None:
        group_col = _pick_best_column(obs, _GROUP_CANDIDATES, min_unique=2, max_unique=12)
        if group_col is None:
            warnings.append(
                "Could not auto-detect a condition/group column from adata.obs. Pass --condition-col explicitly."
            )
    elif group_col not in obs.columns:
        raise KeyError(f"adata.obs is missing condition/group column '{group_col}'.")

    used = {c for c in [group_col] if c}

    if sample_col is None:
        sample_col = _pick_best_column(
            obs,
            _SAMPLE_CANDIDATES,
            require_repeated=True,
            min_unique=2,
            max_unique=max(2, min(len(obs) - 1, 100000)),
            exclude=used,
        )
        if sample_col is None:
            warnings.append(
                "Could not auto-detect a donor/sample column from adata.obs. Pass --sample-col explicitly."
            )
    elif sample_col not in obs.columns:
        raise KeyError(f"adata.obs is missing sample/donor column '{sample_col}'.")
    if sample_col:
        used.add(sample_col)

    if cell_type_col is None:
        cell_type_col = _pick_best_column(
            obs,
            _CELLTYPE_CANDIDATES,
            min_unique=2,
            max_unique=max(2, min(200, len(obs) - 1)),
            exclude=used,
        )
        if cell_type_col is None:
            warnings.append(
                "Could not auto-detect a cell-type column. The workflow will run on all cells together unless --cell-type-col is provided."
            )
    elif cell_type_col not in obs.columns:
        raise KeyError(f"adata.obs is missing cell-type column '{cell_type_col}'.")
    if cell_type_col:
        used.add(cell_type_col)

    if batch_col is None:
        batch_col = _pick_best_column(
            obs,
            _BATCH_CANDIDATES,
            min_unique=2,
            max_unique=max(2, min(96, len(obs) - 1)),
            exclude=used,
        )
    elif batch_col not in obs.columns:
        raise KeyError(f"adata.obs is missing batch column '{batch_col}'.")
    if batch_col:
        used.add(batch_col)

    if covariate_cols:
        resolved = []
        for column in covariate_cols:
            if column not in obs.columns:
                raise KeyError(f"adata.obs is missing covariate column '{column}'.")
            resolved.append(column)
        covariates = resolved
    else:
        covariates = []
        for preferred in _COVARIATE_CANDIDATES:
            for column in obs.columns:
                if column in used or column in covariates:
                    continue
                if _normalize(column) != _normalize(preferred):
                    continue
                values = obs[column].dropna().astype(str)
                if values.empty or int(values.nunique()) < 2:
                    continue
                covariates.append(column)
                break

    return DetectionResult(
        sample_col=sample_col,
        group_col=group_col,
        cell_type_col=cell_type_col,
        batch_col=batch_col,
        covariate_cols=covariates,
        warnings=warnings,
    )


def _build_sample_level_metadata(
    adata: ad.AnnData,
    *,
    sample_col: str,
    stable_cols: tuple[str, ...],
) -> pd.DataFrame:
    required = [sample_col, *stable_cols]
    missing = [col for col in required if col not in adata.obs.columns]
    if missing:
        raise KeyError(f"adata.obs is missing required columns: {missing}")

    meta = adata.obs[required].copy()
    for column in required:
        if meta[column].isna().any():
            raise ValueError(f"adata.obs column '{column}' contains missing values.")
        meta[column] = meta[column].astype(str)

    bad_columns: list[str] = []
    for column in stable_cols:
        nunique = meta.groupby(sample_col, observed=False)[column].nunique(dropna=False)
        if bool((nunique > 1).any()):
            bad_columns.append(column)
    if bad_columns:
        raise ValueError(
            "Each pseudobulk sample must map to exactly one value for the following columns: "
            + ", ".join(sorted(bad_columns))
        )
    return meta.drop_duplicates(subset=[sample_col]).reset_index(drop=True)


def _prepare_pseudobulk_inputs(
    *,
    adata: ad.AnnData,
    sample_col: str,
    group_col: str,
    group_a: str,
    group_b: str,
    batch_col: str | None,
    covariate_cols: tuple[str, ...],
    min_cells_per_sample: int,
    layer: str | None,
    use_raw: bool,
    output_dir: Path,
) -> tuple[Path, Path, Path, pd.DataFrame]:
    pb_adata = compute_pseudobulk(adata, groupby=sample_col, layer=layer, use_raw=use_raw)
    stable_cols = tuple(col for col in [group_col, batch_col, *covariate_cols] if col)
    sample_meta = _build_sample_level_metadata(adata, sample_col=sample_col, stable_cols=stable_cols)

    pb_obs = pb_adata.obs.copy()
    pb_obs[sample_col] = pb_obs[sample_col].astype(str)
    merged_meta = pb_obs[[sample_col, "n_cells"]].merge(sample_meta, on=sample_col, how="left", validate="1:1")
    if merged_meta[group_col].isna().any():
        raise ValueError("Failed to align pseudobulk samples with sample-level metadata.")

    keep = merged_meta["n_cells"].astype(int) >= int(min_cells_per_sample)
    if not bool(keep.any()):
        raise ValueError("No pseudobulk samples remained after applying --min-cells-per-sample.")
    pb_adata = pb_adata[keep.to_numpy()].copy()
    merged_meta = merged_meta.loc[keep].reset_index(drop=True)

    labels = merged_meta[group_col].astype(str)
    n_a = int(labels.eq(group_a).sum())
    n_b = int(labels.eq(group_b).sum())
    if n_a < 2 or n_b < 2:
        raise ValueError(
            "Pseudobulk analysis requires at least 2 samples in each requested group after filtering. "
            f"Observed: {group_a}={n_a}, {group_b}={n_b}."
        )

    inputs_dir = output_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    pb_h5ad_path = inputs_dir / "pseudobulk.h5ad"
    pb_adata.write_h5ad(pb_h5ad_path)

    sample_ids = merged_meta[sample_col].astype(str).tolist()
    matrix = pd.DataFrame(pb_adata.X.T, index=pb_adata.var_names.astype(str), columns=sample_ids).reset_index()
    matrix.columns = ["gene"] + sample_ids
    matrix_path = inputs_dir / "pseudobulk_matrix.csv"
    matrix.to_csv(matrix_path, index=False)

    metadata_path = inputs_dir / "pseudobulk_metadata.csv"
    merged_meta.to_csv(metadata_path, index=False)
    return matrix_path, metadata_path, pb_h5ad_path, merged_meta


def _subset_adata(adata: ad.AnnData, *, cell_type_col: str | None, cell_type_value: str | None) -> ad.AnnData:
    if cell_type_col is None or cell_type_value is None:
        return adata
    mask = adata.obs[cell_type_col].astype(str) == str(cell_type_value)
    if int(mask.sum()) == 0:
        raise ValueError(f"No cells matched {cell_type_col}={cell_type_value!r}.")
    return adata[mask].copy()


def _plan_cell_types(
    adata: ad.AnnData,
    *,
    sample_col: str,
    group_col: str,
    group_a: str,
    group_b: str,
    cell_type_col: str | None,
    min_cells_per_sample: int,
) -> list[CellTypePlan]:
    obs = adata.obs.copy()
    values = [None]
    if cell_type_col is not None:
        values = obs[cell_type_col].dropna().astype(str).value_counts().index.tolist()

    plans: list[CellTypePlan] = []
    for value in values:
        label = "all_cells" if value is None else str(value)
        subset_obs = obs if value is None else obs.loc[obs[cell_type_col].astype(str) == str(value)].copy()
        total_cells = int(len(subset_obs))
        if total_cells == 0:
            plans.append(CellTypePlan(label=label, total_cells=0, retained_samples=0, n_group_a=0, n_group_b=0, eligible=False, reason="no cells after subsetting"))
            continue
        meta = subset_obs[[sample_col, group_col]].copy()
        if meta[sample_col].isna().any() or meta[group_col].isna().any():
            plans.append(CellTypePlan(label=label, total_cells=total_cells, retained_samples=0, n_group_a=0, n_group_b=0, eligible=False, reason="missing sample or condition values"))
            continue
        meta[sample_col] = meta[sample_col].astype(str)
        meta[group_col] = meta[group_col].astype(str)
        nunique = meta.groupby(sample_col, observed=False)[group_col].nunique(dropna=False)
        if bool((nunique > 1).any()):
            plans.append(CellTypePlan(label=label, total_cells=total_cells, retained_samples=0, n_group_a=0, n_group_b=0, eligible=False, reason="some donors map to multiple condition labels"))
            continue
        cells_per_sample = subset_obs.groupby(sample_col, observed=False).size().rename("n_cells")
        retained = cells_per_sample.loc[cells_per_sample >= int(min_cells_per_sample)]
        if retained.empty:
            plans.append(CellTypePlan(label=label, total_cells=total_cells, retained_samples=0, n_group_a=0, n_group_b=0, eligible=False, reason="no donors remain after min-cells filtering"))
            continue
        dedup = meta.drop_duplicates(subset=[sample_col]).set_index(sample_col)
        dedup = dedup.loc[dedup.index.intersection(retained.index)].copy()
        n_a = int(dedup[group_col].eq(group_a).sum())
        n_b = int(dedup[group_col].eq(group_b).sum())
        eligible = n_a >= 2 and n_b >= 2
        reason = "eligible" if eligible else f"requires >=2 donors per group after filtering ({group_a}={n_a}, {group_b}={n_b})"
        plans.append(CellTypePlan(label=label, total_cells=total_cells, retained_samples=int(len(dedup)), n_group_a=n_a, n_group_b=n_b, eligible=eligible, reason=reason))
    plans.sort(key=lambda item: (-item.total_cells, item.label))
    return plans


def _select_cell_types(
    plans: list[CellTypePlan],
    *,
    requested: tuple[str, ...],
    max_cell_types: int,
    all_cell_types: bool,
) -> list[CellTypePlan]:
    eligible = [plan for plan in plans if plan.eligible]
    if requested:
        requested_set = {str(v) for v in requested}
        selected = [plan for plan in eligible if plan.label in requested_set]
        if not selected:
            raise ValueError("None of the requested cell types are eligible for analysis.")
        return selected
    if all_cell_types or len(eligible) <= max_cell_types:
        return eligible
    return eligible[:max_cell_types]


def _render_preflight_html(
    *,
    out_path: Path,
    adata_path: Path,
    adata: ad.AnnData,
    detection: DetectionResult,
    plans: list[CellTypePlan],
    selected: list[CellTypePlan],
    group_a: str,
    group_b: str,
    backend_message: str,
    wizard_only: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selected_labels = {plan.label for plan in selected}
    rows = "\n".join(
        f"<tr><td>{html.escape(plan.label)}</td><td>{plan.total_cells}</td><td>{plan.retained_samples}</td><td>{plan.n_group_a}</td><td>{plan.n_group_b}</td><td>{'yes' if plan.eligible else 'no'}</td><td>{html.escape(plan.reason)}</td><td>{'selected' if plan.label in selected_labels else ''}</td></tr>"
        for plan in plans
    )
    warnings = list(detection.warnings)
    warning_items = "".join(f"<li>{html.escape(item)}</li>" for item in warnings) or "<li>No preflight warnings.</li>"
    html_text = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>nPathway scRNA Easy Preflight</title>
  <style>
    body {{ margin: 0; padding: 28px; font: 15px/1.5 'IBM Plex Sans','Segoe UI',sans-serif; background: linear-gradient(180deg,#f7f2e8,#ece4d7); color: #1f2527; }}
    main {{ max-width: 1180px; margin: 0 auto; background: #fffdf8; border: 1px solid #d8d1c5; border-radius: 18px; padding: 28px; box-shadow: 0 24px 60px rgba(31,37,39,0.08); }}
    .hero {{ display: grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap: 12px; margin-top: 18px; }}
    .tile {{ border: 1px solid #d8d1c5; border-radius: 14px; padding: 14px; background: #fff; }}
    .eyebrow {{ color: #9d6b2f; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; font-weight: 700; }}
    .big {{ font-size: 28px; font-weight: 800; margin-top: 4px; }}
    .card {{ margin-top: 20px; border: 1px solid #d8d1c5; border-radius: 14px; padding: 18px; background: #fff; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid #e5ddd1; vertical-align: top; }}
    th {{ color: #0d5c63; background: #f7f4ee; }}
    code, pre {{ font-family: 'IBM Plex Mono','SFMono-Regular',monospace; }}
    pre {{ background: #f8f3ea; border: 1px solid #d8d1c5; border-radius: 12px; padding: 14px; white-space: pre-wrap; }}
  </style>
</head>
<body>
  <main>
    <h1>nPathway scRNA Easy Preflight</h1>
    <p>Input file: <code>{html.escape(str(adata_path))}</code></p>
    <div class=\"hero\">
      <div class=\"tile\"><div class=\"eyebrow\">Cells</div><div class=\"big\">{adata.n_obs}</div></div>
      <div class=\"tile\"><div class=\"eyebrow\">Genes</div><div class=\"big\">{adata.n_vars}</div></div>
      <div class=\"tile\"><div class=\"eyebrow\">Eligible cell types</div><div class=\"big\">{sum(plan.eligible for plan in plans)}</div></div>
      <div class=\"tile\"><div class=\"eyebrow\">Requested contrast</div><div class=\"big\">{html.escape(group_a)} vs {html.escape(group_b)}</div></div>
    </div>
    <section class=\"card\">
      <h2>Detected Columns</h2>
      <table>
        <tr><th>Role</th><th>Column</th></tr>
        <tr><td>sample / donor</td><td>{html.escape(str(detection.sample_col or 'NOT FOUND'))}</td></tr>
        <tr><td>condition / group</td><td>{html.escape(str(detection.group_col or 'NOT FOUND'))}</td></tr>
        <tr><td>cell type</td><td>{html.escape(str(detection.cell_type_col or 'all cells'))}</td></tr>
        <tr><td>batch</td><td>{html.escape(str(detection.batch_col or 'none'))}</td></tr>
        <tr><td>covariates</td><td>{html.escape(', '.join(detection.covariate_cols) if detection.covariate_cols else 'none')}</td></tr>
      </table>
    </section>
    <section class=\"card\">
      <h2>Backend</h2>
      <p>{html.escape(backend_message)}</p>
      <p>nPathway keeps scRNA ranking on donor-level pseudobulk counts. Cell-level integration methods such as Harmony are useful upstream for exploratory embeddings, but they are intentionally not used for differential ranking here to avoid over-correcting the pathway signal.</p>
      <p>{'Wizard-only mode was requested; no analyses were launched.' if wizard_only else 'This report was written before execution so the selected plan is inspectable.'}</p>
    </section>
    <section class=\"card\">
      <h2>Cell-Type Plan</h2>
      <table>
        <tr><th>Cell type</th><th>Total cells</th><th>Retained donors</th><th>{html.escape(group_a)}</th><th>{html.escape(group_b)}</th><th>Eligible</th><th>Reason</th><th>Selected</th></tr>
        {rows}
      </table>
    </section>
    <section class=\"card\">
      <h2>Warnings</h2>
      <ul>{warning_items}</ul>
    </section>
  </main>
</body>
</html>
"""
    out_path.write_text(html_text, encoding="utf-8")


def _render_run_index_html(
    *,
    out_path: Path,
    adata_path: Path,
    detection: DetectionResult,
    runs: list[CellTypeRunResult],
    preflight_path: Path,
    figure_ready_path: Path | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _safe_read_csv(path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None
        try:
            return pd.read_csv(path)
        except Exception:
            return None

    def _artifact_summary(run: CellTypeRunResult) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "dynamic_sig_hits": 0,
            "curated_sig_hits": 0,
            "lead_dynamic": "",
            "lead_curated": "",
            "dynamic_only_count": 0,
            "dynamic_only_genes": "",
        }
        run_base = Path(run.output_dir)
        dynamic_df = _safe_read_csv(run_base / "comparison" / "dynamic_gsea.csv")
        curated_df = _safe_read_csv(run_base / "comparison" / "curated_gsea.csv")
        focus_df = _safe_read_csv(run_base / "comparison" / "focus_gene_membership.csv")

        if dynamic_df is not None and not dynamic_df.empty and "fdr" in dynamic_df.columns:
            dynamic_df = dynamic_df.sort_values(["fdr", "p_value"], ascending=[True, True]).reset_index(drop=True)
            summary["dynamic_sig_hits"] = int((dynamic_df["fdr"].astype(float) <= 0.05).sum())
            lead = dynamic_df.iloc[0]
            summary["lead_dynamic"] = f"{lead['program']} (FDR={float(lead['fdr']):.3g})"

        if curated_df is not None and not curated_df.empty and "fdr" in curated_df.columns:
            curated_df = curated_df.sort_values(["fdr", "p_value"], ascending=[True, True]).reset_index(drop=True)
            summary["curated_sig_hits"] = int((curated_df["fdr"].astype(float) <= 0.05).sum())
            lead = curated_df.iloc[0]
            summary["lead_curated"] = f"{lead['program']} (FDR={float(lead['fdr']):.3g})"

        if focus_df is not None and not focus_df.empty:
            dynamic_mask = focus_df["in_any_dynamic_program"].astype(str).str.lower().eq("true")
            curated_mask = focus_df["in_any_curated_set"].astype(str).str.lower().eq("true")
            genes = focus_df.loc[dynamic_mask & ~curated_mask, "gene"].astype(str).tolist()
            summary["dynamic_only_count"] = len(genes)
            summary["dynamic_only_genes"] = ", ".join(genes[:6])
        return summary

    artifact_summaries = {run.label: _artifact_summary(run) for run in runs}
    comparison_runs = [run for run in runs if run.comparison_dir is not None]
    best_anchor_run = max(
        (run for run in comparison_runs if run.anchor_jaccard is not None),
        key=lambda item: float(item.anchor_jaccard),
        default=None,
    )
    best_dynamic_only_run = max(
        runs,
        key=lambda item: int(artifact_summaries[item.label]["dynamic_only_count"]),
        default=None,
    )
    total_dynamic_only = sum(int(artifact_summaries[run.label]["dynamic_only_count"]) for run in runs)
    total_dynamic_sig = sum(int(artifact_summaries[run.label]["dynamic_sig_hits"]) for run in runs)
    total_curated_sig = sum(int(artifact_summaries[run.label]["curated_sig_hits"]) for run in runs)

    rows = []
    for run in runs:
        artifact = artifact_summaries[run.label]
        run_base = Path(run.output_dir)
        summary_rel = (run_base / "summary.md").relative_to(out_path.parent).as_posix()
        comparison_rel = ""
        if run.comparison_dir is not None:
            comparison_rel = (Path(run.comparison_dir) / "summary.md").relative_to(out_path.parent).as_posix()
        dashboard_rel = ""
        if run.dashboard_html is not None:
            dashboard_rel = Path(run.dashboard_html).relative_to(out_path.parent).as_posix()
        batch_qc_rel = ""
        if run.batch_qc_dir is not None:
            candidate = Path(run.batch_qc_dir) / "pca_summary.json"
            if candidate.exists():
                batch_qc_rel = candidate.relative_to(out_path.parent).as_posix()
        comparison_link = ""
        if comparison_rel:
            comparison_link = f'<a href="{html.escape(comparison_rel)}">comparison</a>'
        dashboard_link = ""
        if dashboard_rel:
            dashboard_link = f'<a href="{html.escape(dashboard_rel)}">dashboard</a>'
        batch_qc_link = ""
        if batch_qc_rel:
            batch_qc_link = f'<a href="{html.escape(batch_qc_rel)}">batch QC</a>'
        rows.append(
            f"<tr><td>{html.escape(run.label)}</td><td>{run.total_cells}</td><td>{run.retained_samples}</td><td>{run.n_group_a}</td><td>{run.n_group_b}</td><td>{html.escape(run.ranking_source)}</td><td>{run.n_programs}</td><td>{run.n_sig_de_genes}</td><td>{artifact['dynamic_sig_hits']}</td><td>{artifact['curated_sig_hits']}</td><td>{html.escape(run.anchor_reference or '')}</td><td>{html.escape(run.anchor_program or '')}</td><td>{'' if run.anchor_jaccard is None else f'{run.anchor_jaccard:.3f}'}</td><td>{html.escape(str(artifact['dynamic_only_count']))}</td><td>{html.escape(str(artifact['dynamic_only_genes']))}</td><td><a href=\"{html.escape(summary_rel)}\">summary</a></td><td>{comparison_link}</td><td>{dashboard_link}</td><td>{batch_qc_link}</td></tr>"
        )
    strongest_anchor_copy = "No curated comparison was generated."
    if best_anchor_run is not None:
        strongest_anchor_copy = (
            f"{best_anchor_run.label}: {best_anchor_run.anchor_reference} <-> "
            f"{best_anchor_run.anchor_program} (Jaccard {float(best_anchor_run.anchor_jaccard):.3f})"
        )
    frontier_copy = "No dynamic-only focus genes were captured."
    if best_dynamic_only_run is not None:
        artifact = artifact_summaries[best_dynamic_only_run.label]
        frontier_copy = (
            f"{best_dynamic_only_run.label}: {artifact['dynamic_only_count']} focus genes were in dynamic programs "
            f"but not in curated references. {artifact['dynamic_only_genes']}"
        )
    thumbnail_cards: list[str] = []
    for run in runs:
        if run.dashboard_html is None:
            continue
        dashboard_dir = Path(run.dashboard_html).parent
        fig1 = dashboard_dir / "figures" / "figure_1_volcano.png"
        fig2 = dashboard_dir / "figures" / "figure_2_program_sizes.png"
        if not fig1.exists() or not fig2.exists():
            continue
        fig1_rel = fig1.relative_to(out_path.parent).as_posix()
        fig2_rel = fig2.relative_to(out_path.parent).as_posix()
        batch_qc_html = ""
        if run.batch_qc_dir is not None:
            qc_dir = Path(run.batch_qc_dir)
            pca_before = qc_dir / "pca_before.png"
            pca_after = qc_dir / "pca_after.png"
            corr_before = qc_dir / "correlation_before.png"
            corr_after = qc_dir / "correlation_after.png"
            if pca_before.exists() and pca_after.exists():
                pca_before_rel = pca_before.relative_to(out_path.parent).as_posix()
                pca_after_rel = pca_after.relative_to(out_path.parent).as_posix()
                corr_links: list[str] = []
                if corr_before.exists():
                    corr_links.append(
                        f'<a href="{html.escape(corr_before.relative_to(out_path.parent).as_posix())}">correlation before</a>'
                    )
                if corr_after.exists():
                    corr_links.append(
                        f'<a href="{html.escape(corr_after.relative_to(out_path.parent).as_posix())}">correlation after</a>'
                    )
                corr_links_html = " | ".join(corr_links)
                batch_qc_html = f"""
                <div class="qc-block">
                  <div class="eyebrow">Batch QC</div>
                  <p>Before/after PCA on donor-level pseudobulk samples.</p>
                  <div class="thumb-grid">
                    <a href="{html.escape(pca_before_rel)}"><img alt="{html.escape(run.label)} PCA before correction" src="{html.escape(pca_before_rel)}" /></a>
                    <a href="{html.escape(pca_after_rel)}"><img alt="{html.escape(run.label)} PCA after correction" src="{html.escape(pca_after_rel)}" /></a>
                  </div>
                  <p class="qc-links">{corr_links_html}</p>
                </div>
                """
        thumbnail_cards.append(
            f"""
            <article class="thumb-card">
              <div class="eyebrow">Cell Type Snapshot</div>
              <h3>{html.escape(run.label)}</h3>
              <p>{html.escape(run.anchor_reference or 'No curated anchor')} | ranking: {html.escape(run.ranking_source)}</p>
              <div class="thumb-grid">
                <a href="{html.escape(fig1_rel)}"><img alt="{html.escape(run.label)} volcano" src="{html.escape(fig1_rel)}" /></a>
                <a href="{html.escape(fig2_rel)}"><img alt="{html.escape(run.label)} program sizes" src="{html.escape(fig2_rel)}" /></a>
              </div>
              {batch_qc_html}
            </article>
            """
        )
    figure_ready_link = ""
    if figure_ready_path is not None:
        figure_ready_link = (
            f'<p><a href="{html.escape(figure_ready_path.relative_to(out_path.parent).as_posix())}">'
            "Open figure-ready export package</a></p>"
        )

    html_text = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>nPathway scRNA Easy Run Index</title>
  <style>
    body {{ margin: 0; padding: 28px; font: 15px/1.5 'IBM Plex Sans','Segoe UI',sans-serif; background: radial-gradient(circle at top left,#fff8ea,#ece2d5 60%,#e4d8ca); color: #1f2527; }}
    main {{ max-width: 1280px; margin: 0 auto; background: #fffdf8; border: 1px solid #d8d1c5; border-radius: 22px; padding: 30px; box-shadow: 0 28px 70px rgba(31,37,39,0.10); }}
    .intro {{ display: grid; grid-template-columns: 1.5fr 1fr; gap: 18px; margin-top: 18px; }}
    .headline {{ border: 1px solid #d8d1c5; border-radius: 18px; padding: 20px; background: linear-gradient(135deg,#fffdf7,#f7efe1); }}
    .headline p {{ color: #5f676b; }}
    .hero {{ display: grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap: 12px; margin-top: 18px; }}
    .tile {{ border: 1px solid #d8d1c5; border-radius: 14px; padding: 14px; background: #fff; }}
    .eyebrow {{ color: #9d6b2f; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; font-weight: 700; }}
    .big {{ font-size: 28px; font-weight: 800; margin-top: 4px; }}
    .card {{ margin-top: 20px; border: 1px solid #d8d1c5; border-radius: 14px; padding: 18px; background: #fff; }}
    .spotlights {{ display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 12px; margin-top: 20px; }}
    .spotlight {{ border: 1px solid #d8d1c5; border-radius: 16px; padding: 18px; background: linear-gradient(180deg,#fff,#faf3e8); }}
    .spotlight h3 {{ margin: 0 0 8px; }}
    .spotlight p {{ margin: 0; color: #5f676b; }}
    .steps {{ display: grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap: 12px; margin-top: 20px; }}
    .step {{ border: 1px solid #d8d1c5; border-radius: 16px; padding: 16px; background: linear-gradient(180deg,#fff,#f6efe5); }}
    .step strong {{ display: block; margin-bottom: 6px; color: #174c62; }}
    .thumb-section {{ margin-top: 20px; }}
    .thumb-gallery {{ display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 14px; }}
    .thumb-card {{ border: 1px solid #d8d1c5; border-radius: 16px; padding: 16px; background: #fff; }}
    .thumb-card h3 {{ margin: 6px 0 8px; }}
    .thumb-card p {{ color: #5f676b; margin: 0 0 12px; }}
    .thumb-grid {{ display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 10px; }}
    .thumb-grid img {{ width: 100%; display: block; border-radius: 12px; border: 1px solid #ddd3c5; }}
    .qc-block {{ margin-top: 14px; padding-top: 12px; border-top: 1px solid #e5ddd1; }}
    .qc-block p {{ color: #5f676b; margin: 0 0 10px; }}
    .qc-links a {{ font-size: 13px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid #e5ddd1; vertical-align: top; }}
    th {{ color: #174c62; background: #f7f4ee; }}
    a {{ color: #174c62; text-decoration: none; font-weight: 700; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <main>
    <h1>nPathway scRNA Easy Run Index</h1>
    <p>Input file: <code>{html.escape(str(adata_path))}</code></p>
    <section class=\"intro\">
      <div class=\"headline\">
        <div class=\"eyebrow\">Manuscript View</div>
        <h2>Cell-type-level pathway discovery, grounded against curated biology</h2>
        <p>This landing page is the decision layer for the beginner-friendly scRNA workflow. It keeps the preflight contract, per-cell-type outputs, and curated-vs-dynamic comparison in one place so the reviewer-facing story stays inspectable.</p>
      </div>
      <div class=\"headline\">
        <div class=\"eyebrow\">Navigation</div>
        <p><a href=\"{html.escape(preflight_path.relative_to(out_path.parent).as_posix())}\">Open preflight report</a></p>
        {figure_ready_link}
        <p>Detected sample column: <strong>{html.escape(str(detection.sample_col or 'NA'))}</strong></p>
        <p>Detected condition column: <strong>{html.escape(str(detection.group_col or 'NA'))}</strong></p>
        <p>Detected cell-type column: <strong>{html.escape(str(detection.cell_type_col or 'all cells'))}</strong></p>
      </div>
    </section>
    <div class=\"hero\">
      <div class=\"tile\"><div class=\"eyebrow\">Completed runs</div><div class=\"big\">{len(runs)}</div></div>
      <div class=\"tile\"><div class=\"eyebrow\">Dynamic sig hits</div><div class=\"big\">{total_dynamic_sig}</div></div>
      <div class=\"tile\"><div class=\"eyebrow\">Curated sig hits</div><div class=\"big\">{total_curated_sig}</div></div>
      <div class=\"tile\"><div class=\"eyebrow\">Dynamic-only focus genes</div><div class=\"big\">{total_dynamic_only}</div></div>
    </div>
    <section class=\"spotlights\">
      <article class=\"spotlight\">
        <div class=\"eyebrow\">Best Anchor</div>
        <h3>Strongest curated alignment</h3>
        <p>{html.escape(strongest_anchor_copy)}</p>
      </article>
      <article class=\"spotlight\">
        <div class=\"eyebrow\">Frontier Signal</div>
        <h3>Largest dynamic-only recovery</h3>
        <p>{html.escape(frontier_copy)}</p>
      </article>
    </section>
    <section class=\"steps\">
      <article class=\"step\">
        <div class=\"eyebrow\">Step 1</div>
        <strong>Check the contract</strong>
        <p>Open the preflight report first. It tells you which columns were detected, which cell types were eligible, and whether the run used the batch-aware backend.</p>
      </article>
      <article class=\"step\">
        <div class=\"eyebrow\">Step 2</div>
        <strong>Inspect the best anchor</strong>
        <p>Use the strongest curated alignment card to find the cell type where dynamic programs most clearly connect back to known biology.</p>
      </article>
      <article class=\"step\">
        <div class=\"eyebrow\">Step 3</div>
        <strong>Look for frontier signal</strong>
        <p>Then inspect the dynamic-only focus genes and thumbnails to see what the curated references missed in each cell type.</p>
      </article>
    </section>
    <section class=\"card thumb-section\">
      <h2>Representative Thumbnails</h2>
      <div class=\"thumb-gallery\">
        {''.join(thumbnail_cards) or '<p>No dashboard thumbnails were available for these runs.</p>'}
      </div>
    </section>
    <section class=\"card\">
      <h2>Cell-Type Runs</h2>
      <table>
        <tr><th>Cell type</th><th>Total cells</th><th>Retained donors</th><th>Case</th><th>Control</th><th>Ranking</th><th>Programs</th><th>Sig DE genes</th><th>Dynamic sig</th><th>Curated sig</th><th>Anchor ref</th><th>Anchor program</th><th>Jaccard</th><th>Dynamic-only focus</th><th>Example genes</th><th>Results</th><th>Comparison</th><th>Dashboard</th><th>Batch QC</th></tr>
        {''.join(rows)}
      </table>
    </section>
  </main>
</body>
</html>
"""
    out_path.write_text(html_text, encoding="utf-8")


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _write_figure_ready_export(
    *,
    outdir: Path,
    preflight_path: Path,
    summary_csv: Path,
    index_html: Path,
    runs: list[CellTypeRunResult],
) -> Path:
    export_dir = outdir / "figure_ready"
    export_dir.mkdir(parents=True, exist_ok=True)
    _copy_if_exists(preflight_path, export_dir / "preflight_report.html")
    _copy_if_exists(summary_csv, export_dir / "cell_type_run_summary.csv")
    _copy_if_exists(index_html, export_dir / "analysis_index.html")

    inventory_rows: list[dict[str, str]] = []
    strongest_anchor = max(
        (run for run in runs if run.anchor_jaccard is not None),
        key=lambda item: float(item.anchor_jaccard),
        default=None,
    )

    for run in runs:
        run_dir = export_dir / _slugify(run.label)
        run_dir.mkdir(parents=True, exist_ok=True)
        source_dir = Path(run.output_dir)
        inventory_base = {
            "cell_type": run.label,
        }

        for src_name, dst_name, category, description in [
            ("summary.md", "summary.md", "summary", "Top-level run summary"),
            ("workflow_manifest.json", "workflow_manifest.json", "manifest", "Run manifest"),
            ("comparison/summary.md", "comparison_summary.md", "comparison", "Curated-vs-dynamic summary"),
            ("comparison/dynamic_gsea.csv", "dynamic_gsea.csv", "comparison_table", "Dynamic-program GSEA table"),
            ("comparison/curated_gsea.csv", "curated_gsea.csv", "comparison_table", "Curated reference GSEA table"),
            ("comparison/focus_gene_membership.csv", "focus_gene_membership.csv", "comparison_table", "Focus-gene membership table"),
            ("dashboard/figures/figure_1_volcano.png", "figure_1_volcano.png", "figure", "Volcano figure"),
            ("dashboard/figures/figure_2_program_sizes.png", "figure_2_program_sizes.png", "figure", "Program-size figure"),
            ("prepared_inputs/qc/pca_before.png", "batch_pca_before.png", "batch_qc", "Pseudobulk PCA before correction"),
            ("prepared_inputs/qc/pca_after.png", "batch_pca_after.png", "batch_qc", "Pseudobulk PCA after correction"),
            ("prepared_inputs/qc/correlation_before.png", "batch_corr_before.png", "batch_qc", "Sample correlation heatmap before correction"),
            ("prepared_inputs/qc/correlation_after.png", "batch_corr_after.png", "batch_qc", "Sample correlation heatmap after correction"),
            ("prepared_inputs/qc/pca_summary.json", "batch_qc_summary.json", "batch_qc", "Batch-QC summary metadata"),
        ]:
            src = source_dir / src_name
            dst = run_dir / dst_name
            if _copy_if_exists(src, dst):
                inventory_rows.append(
                    {
                        **inventory_base,
                        "category": category,
                        "description": description,
                        "source_path": str(src),
                        "exported_path": str(dst),
                    }
                )

    pd.DataFrame(
        inventory_rows,
        columns=["cell_type", "category", "description", "source_path", "exported_path"],
    ).to_csv(export_dir / "figure_inventory.csv", index=False)

    caption_lines = [
        "# nPathway Figure-Ready Notes",
        "",
        "## Recommended Primary Panels",
        "",
        "1. Use `figure_1_volcano.png` plus `figure_2_program_sizes.png` from the strongest anchor cell type for the main pathway-discovery figure.",
        "2. Use `batch_pca_before.png` and `batch_pca_after.png` to document the effect of known batch correction and guarded SVA on donor-level pseudobulk samples.",
        "3. Use `focus_gene_membership.csv` to highlight genes recovered by dynamic programs but missing from curated references.",
        "",
        "## Strongest Anchor",
        "",
    ]
    if strongest_anchor is None:
        caption_lines.append("- No curated comparison anchor was available.")
    else:
        caption_lines.append(
            f"- {strongest_anchor.label}: {strongest_anchor.anchor_reference} vs {strongest_anchor.anchor_program} "
            f"(Jaccard {float(strongest_anchor.anchor_jaccard):.3f})"
        )
    (export_dir / "caption_starter.md").write_text("\n".join(caption_lines) + "\n", encoding="utf-8")

    manifest = {
        "preflight_report": str(export_dir / "preflight_report.html"),
        "analysis_index_html": str(export_dir / "analysis_index.html"),
        "cell_type_run_summary_csv": str(export_dir / "cell_type_run_summary.csv"),
        "figure_inventory_csv": str(export_dir / "figure_inventory.csv"),
        "caption_starter_md": str(export_dir / "caption_starter.md"),
        "cell_types": [run.label for run in runs],
    }
    (export_dir / "figure_ready_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return export_dir


def _batch_aware_backend_available(force_simple: bool, surrogate_variable_mode: str) -> tuple[bool, str]:
    if force_simple:
        return False, "Batch-aware bulk ranking was disabled by --force-simple-backend; using the simpler pseudobulk fallback."
    try:
        availability = _detect_r_dependencies(raw_counts=True, surrogate_variable_mode=surrogate_variable_mode)
    except RuntimeError as exc:
        if surrogate_variable_mode == "on":
            raise
        return False, f"Batch-aware bulk ranking is unavailable: {exc}"
    if surrogate_variable_mode == "auto":
        if availability.get("sva", False):
            return (
                True,
                "Batch-aware bulk ranking is available. edgeR/limma will be used on donor-level pseudobulk counts, "
                "and guarded SVA will run automatically when residual degrees of freedom are sufficient.",
            )
        return (
            True,
            "Batch-aware bulk ranking is available. edgeR/limma will be used on donor-level pseudobulk counts; "
            "guarded SVA auto mode was requested but the R package 'sva' is not installed, so known batch/covariate adjustment only will be used.",
        )
    if surrogate_variable_mode == "on":
        return (
            True,
            "Batch-aware bulk ranking is available with required SVA support. edgeR/limma and guarded surrogate-variable adjustment will be used when feasible.",
        )
    return True, "Batch-aware bulk ranking is available. edgeR/limma will be used on donor-level pseudobulk counts when possible."


def _run_simple_subset(
    *,
    matrix_path: Path,
    metadata_path: Path,
    sample_col: str,
    group_col: str,
    group_a: str,
    group_b: str,
    outdir: Path,
    discovery_method: str,
    n_programs: int,
    k_neighbors: int,
    resolution: float,
    n_components: int,
    n_diffusion_steps: int,
    diffusion_alpha: float,
    gsea_n_perm: int,
    seed: int,
    annotate_programs: bool,
    annotation_collections: tuple[str, ...],
    annotation_species: str,
    annotation_gmt: str | None,
    annotation_topk_per_program: int,
    annotation_min_jaccard_for_label: float,
    curated_gmt: str | None,
    focus_genes: tuple[str, ...],
    with_dashboard: bool,
    dashboard_top_k: int,
    include_pdf: bool,
) -> dict[str, object]:
    result = run_bulk_dynamic_pipeline(
        config=BulkDynamicConfig(
            matrix_path=str(matrix_path),
            metadata_path=str(metadata_path),
            group_col=group_col,
            group_a=group_a,
            group_b=group_b,
            sample_col=sample_col,
            matrix_orientation="genes_by_samples",
            raw_counts=True,
            discovery_method=discovery_method,
            n_programs=n_programs,
            k_neighbors=k_neighbors,
            resolution=resolution,
            n_components=n_components,
            n_diffusion_steps=n_diffusion_steps,
            diffusion_alpha=diffusion_alpha,
            gsea_n_perm=gsea_n_perm,
            random_seed=seed,
            annotate_programs=annotate_programs,
            annotation_collections=annotation_collections,
            annotation_species=annotation_species,
            annotation_gmt_path=annotation_gmt,
            annotation_topk_per_program=annotation_topk_per_program,
            annotation_min_jaccard_for_label=annotation_min_jaccard_for_label,
        ),
        output_dir=outdir,
    )
    comparison_result = None
    if curated_gmt:
        comparison_result = compare_curated_vs_dynamic_gsea(
            ranked_genes_path=Path(result.output_dir) / "differential" / "ranked_genes_for_gsea.csv",
            dynamic_gmt_path=Path(result.output_dir) / "discovery" / "dynamic_programs.gmt",
            curated_gmt_path=curated_gmt,
            output_dir=Path(result.output_dir) / "comparison",
            focus_genes=list(focus_genes),
            n_perm=gsea_n_perm,
            seed=seed,
        )
    dashboard_html = None
    if with_dashboard:
        artifacts = build_dynamic_dashboard_package(
            DashboardConfig(
                results_dir=result.output_dir,
                output_dir=str(Path(result.output_dir)),
                title=f"nPathway Dashboard: {group_a} vs {group_b}",
                top_k=dashboard_top_k,
                include_pdf=include_pdf,
            )
        )
        dashboard_html = artifacts.html_path
    (outdir / "workflow_manifest.json").write_text(
        json.dumps(
            {
                "ranking_source": "simple_pseudobulk_internal_de",
                "curated_gmt": curated_gmt,
                "focus_genes": list(focus_genes),
                "result": asdict(result),
                "comparison": asdict(comparison_result) if comparison_result is not None else None,
                "dashboard_html": dashboard_html,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "result": result,
        "comparison_result": comparison_result,
        "ranking_source": "simple_pseudobulk_internal_de",
        "dashboard_html_path": dashboard_html,
    }


def _run_single_cell_type(
    *,
    adata: ad.AnnData,
    plan: CellTypePlan,
    cell_type_col: str | None,
    sample_col: str,
    group_col: str,
    group_a: str,
    group_b: str,
    batch_col: str | None,
    covariate_cols: tuple[str, ...],
    min_cells_per_sample: int,
    layer: str | None,
    use_raw: bool,
    outdir: Path,
    use_batch_aware_backend: bool,
    discovery_method: str,
    n_programs: int,
    k_neighbors: int,
    resolution: float,
    n_components: int,
    n_diffusion_steps: int,
    diffusion_alpha: float,
    gsea_n_perm: int,
    seed: int,
    annotate_programs: bool,
    annotation_collections: tuple[str, ...],
    annotation_species: str,
    annotation_gmt: str | None,
    annotation_topk_per_program: int,
    annotation_min_jaccard_for_label: float,
    curated_gmt: str | None,
    focus_genes: tuple[str, ...],
    with_dashboard: bool,
    dashboard_top_k: int,
    include_pdf: bool,
    surrogate_variable_mode: str,
    sva_max_n_sv: int,
    sva_min_residual_df: int,
    verbose: bool,
) -> CellTypeRunResult:
    cell_type_value = None if plan.label == "all_cells" else plan.label
    subset = _subset_adata(adata, cell_type_col=cell_type_col, cell_type_value=cell_type_value)
    matrix_path, metadata_path, _pb_h5ad_path, pb_meta = _prepare_pseudobulk_inputs(
        adata=subset,
        sample_col=sample_col,
        group_col=group_col,
        group_a=group_a,
        group_b=group_b,
        batch_col=batch_col,
        covariate_cols=covariate_cols,
        min_cells_per_sample=min_cells_per_sample,
        layer=layer,
        use_raw=use_raw,
        output_dir=outdir,
    )

    if use_batch_aware_backend:
        namespace = argparse.Namespace(
            matrix=str(matrix_path),
            metadata=str(metadata_path),
            sample_col=sample_col,
            group_col=group_col,
            group_a=group_a,
            group_b=group_b,
            matrix_orientation="genes_by_samples",
            sep=",",
            raw_counts=True,
            batch_col=batch_col,
            covariate_cols=",".join(covariate_cols),
            surrogate_variable_mode=surrogate_variable_mode,
            sva_max_n_sv=sva_max_n_sv,
            sva_min_residual_df=sva_min_residual_df,
            ranked_genes=None,
            ranked_gene_col="gene",
            ranked_score_col="score",
            ranked_sep=None,
            curated_gmt=curated_gmt,
            focus_genes=",".join(focus_genes),
            discovery_method=discovery_method,
            n_programs=n_programs,
            k_neighbors=k_neighbors,
            resolution=resolution,
            n_components=n_components,
            n_diffusion_steps=n_diffusion_steps,
            diffusion_alpha=diffusion_alpha,
            de_test="welch",
            de_alpha=0.05,
            min_abs_logfc_for_claim=0.2,
            gsea_n_perm=gsea_n_perm,
            min_genes_per_program_claim=10,
            n_bootstrap=0,
            min_stability_for_claim=0.25,
            seed=seed,
            annotate_programs=annotate_programs,
            annotation_collections=",".join(annotation_collections),
            annotation_species=annotation_species,
            annotation_gmt=annotation_gmt,
            annotation_topk_per_program=annotation_topk_per_program,
            annotation_min_jaccard_for_label=annotation_min_jaccard_for_label,
            output_dir=str(outdir),
            with_dashboard=with_dashboard,
            dashboard_output_dir=None,
            dashboard_top_k=dashboard_top_k,
            dashboard_no_pdf=not include_pdf,
            verbose=verbose,
        )
        run = run_batch_aware_bulk_workflow(namespace)
    else:
        run = _run_simple_subset(
            matrix_path=matrix_path,
            metadata_path=metadata_path,
            sample_col=sample_col,
            group_col=group_col,
            group_a=group_a,
            group_b=group_b,
            outdir=outdir,
            discovery_method=discovery_method,
            n_programs=n_programs,
            k_neighbors=k_neighbors,
            resolution=resolution,
            n_components=n_components,
            n_diffusion_steps=n_diffusion_steps,
            diffusion_alpha=diffusion_alpha,
            gsea_n_perm=gsea_n_perm,
            seed=seed,
            annotate_programs=annotate_programs,
            annotation_collections=annotation_collections,
            annotation_species=annotation_species,
            annotation_gmt=annotation_gmt,
            annotation_topk_per_program=annotation_topk_per_program,
            annotation_min_jaccard_for_label=annotation_min_jaccard_for_label,
            curated_gmt=curated_gmt,
            focus_genes=focus_genes,
            with_dashboard=with_dashboard,
            dashboard_top_k=dashboard_top_k,
            include_pdf=include_pdf,
        )

    result = run["result"]
    comparison_result = run["comparison_result"]
    return CellTypeRunResult(
        label=plan.label,
        output_dir=result.output_dir,
        dashboard_html=run.get("dashboard_html_path"),
        batch_qc_dir=str(Path(run["prepared"].batch_qc_dir)) if "prepared" in run else None,
        status="completed",
        total_cells=int(subset.n_obs),
        retained_samples=int(len(pb_meta)),
        n_group_a=int(pb_meta[group_col].astype(str).eq(group_a).sum()),
        n_group_b=int(pb_meta[group_col].astype(str).eq(group_b).sum()),
        ranking_source=str(run["ranking_source"]),
        n_programs=int(result.n_programs),
        n_sig_de_genes=int(result.n_sig_de_genes),
        comparison_dir=None if comparison_result is None else str(comparison_result.output_dir),
        anchor_program=None if comparison_result is None else comparison_result.anchor_program,
        anchor_reference=None if comparison_result is None else comparison_result.anchor_reference,
        anchor_jaccard=None if comparison_result is None else float(comparison_result.anchor_jaccard),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adata", required=True, help="Path to annotated AnnData (.h5ad).")
    parser.add_argument("--condition-col", "--group-col", dest="group_col", default=None, help="Condition/group column in adata.obs. If omitted, nPathway will try to detect it.")
    parser.add_argument("--sample-col", default=None, help="Donor/sample column in adata.obs. If omitted, nPathway will try to detect it.")
    parser.add_argument("--cell-type-col", default=None, help="Cell-type column in adata.obs. If omitted, nPathway will try to detect it.")
    parser.add_argument("--batch-col", default=None, help="Optional batch column in adata.obs. If omitted, nPathway will try to detect it.")
    parser.add_argument("--covariate-cols", default="", help="Optional comma-separated nuisance covariates. Leave empty for auto-detect of common columns like age/sex/PMI/RIN.")
    parser.add_argument("--case", "--group-a", dest="group_a", required=True, help="Case / target group label.")
    parser.add_argument("--control", "--group-b", dest="group_b", required=True, help="Control / reference group label.")
    parser.add_argument("--cell-types", default="", help="Optional comma-separated cell types to run explicitly.")
    parser.add_argument("--all-cell-types", action="store_true", help="Run every eligible cell type instead of only the top major cell types.")
    parser.add_argument("--max-cell-types", type=int, default=6, help="Maximum number of cell types to run when --all-cell-types is not used.")
    parser.add_argument("--min-cells-per-sample", type=int, default=10, help="Minimum cells per donor/sample required after cell-type subsetting.")
    parser.add_argument("--layer", default=None, help="Optional AnnData layer to aggregate instead of adata.X / adata.raw.X.")
    parser.add_argument("--use-raw", action=argparse.BooleanOptionalAction, default=True, help="Aggregate from adata.raw when available (default: true).")
    parser.add_argument("--force-simple-backend", action="store_true", help="Skip the batch-aware edgeR/limma backend and use the simpler pseudobulk fallback.")
    parser.add_argument(
        "--surrogate-variable-mode",
        default="auto",
        choices=["off", "auto", "on"],
        help="Guarded surrogate-variable mode for the batch-aware backend. auto uses SVA when available and safe.",
    )
    parser.add_argument("--sva-max-n-sv", type=int, default=3, help="Maximum number of surrogate variables in guarded SVA mode.")
    parser.add_argument(
        "--sva-min-residual-df",
        type=int,
        default=3,
        help="Minimum residual degrees of freedom required before guarded SVA will run.",
    )
    parser.add_argument("--curated-gmt", default=None, help="Optional curated GMT for same-ranked-list comparison after each run.")
    parser.add_argument("--annotation-gmt", default=None, help="Optional GMT used to annotate program labels.")
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
    parser.add_argument("--annotate-programs", action=argparse.BooleanOptionalAction, default=True, help="Annotate discovered programs (default: true).")
    parser.add_argument("--annotation-topk-per-program", type=int, default=15, help="Top reference matches saved per program.")
    parser.add_argument("--annotation-min-jaccard-for-label", type=float, default=0.03, help="Minimum Jaccard required to adopt a reference-derived label.")
    parser.add_argument("--focus-genes", default="", help="Optional comma-separated genes to track in comparison outputs.")
    parser.add_argument("--discovery-method", default="ensemble", choices=["ensemble", "kmeans", "leiden", "spectral", "hdbscan"], help="Program discovery method (default: ensemble = consensus of kmeans+leiden).")
    parser.add_argument("--n-programs", type=int, default=20, help="Target number of programs for kmeans/spectral.")
    parser.add_argument("--k-neighbors", type=int, default=15, help="kNN neighbors for discovery.")
    parser.add_argument("--resolution", type=float, default=1.0, help="Leiden resolution.")
    parser.add_argument("--n-components", type=int, default=30, help="Embedding dimension.")
    parser.add_argument("--n-diffusion-steps", type=int, default=3, help="Diffusion iterations.")
    parser.add_argument("--diffusion-alpha", type=float, default=0.5, help="Diffusion self-weight.")
    parser.add_argument("--gsea-n-perm", type=int, default=1000, help="GSEA permutations.")
    parser.add_argument("--with-dashboard", action=argparse.BooleanOptionalAction, default=True, help="Build a dashboard for each run (default: true).")
    parser.add_argument("--with-pdf", action="store_true", help="Also export dashboard figures as PDF.")
    parser.add_argument("--dashboard-top-k", type=int, default=20, help="Top-K rows used in dashboard plots.")
    parser.add_argument(
        "--figure-ready-export",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy the key figures, batch-QC panels, and comparison tables into a figure-ready package (default: true).",
    )
    parser.add_argument("--wizard-only", action="store_true", help="Only write the preflight report and stop before running analyses.")
    parser.add_argument("--output-dir", default=None, help="Output directory.")
    parser.add_argument("--verbose", action="store_true", help="Enable INFO logging.")
    return parser.parse_args(argv)


def _validate_cli_args(args: argparse.Namespace) -> None:
    if args.group_a == args.group_b:
        raise ValueError("--case and --control must be different labels.")
    if args.force_simple_backend and args.surrogate_variable_mode == "on":
        raise ValueError("--surrogate-variable-mode on is incompatible with --force-simple-backend.")
    if args.min_cells_per_sample < 1:
        raise ValueError("--min-cells-per-sample must be >= 1.")
    if args.max_cell_types < 1:
        raise ValueError("--max-cell-types must be >= 1.")
    if args.n_programs < 1:
        raise ValueError("--n-programs must be >= 1.")
    if args.n_components < 1:
        raise ValueError("--n-components must be >= 1.")
    if args.gsea_n_perm < 1:
        raise ValueError("--gsea-n-perm must be >= 1.")
    if args.sva_max_n_sv < 1:
        raise ValueError("--sva-max-n-sv must be >= 1.")
    if args.sva_min_residual_df < 1:
        raise ValueError("--sva-min-residual-df must be >= 1.")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_cli_args(args)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

    outdir = Path(args.output_dir) if args.output_dir else Path("results") / f"scrna_{args.case}_vs_{args.control}_{args.discovery_method}_{date.today().strftime('%Y%m%d')}"
    outdir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.adata)
    detection = detect_obs_columns(
        adata,
        group_col=args.group_col,
        sample_col=args.sample_col,
        cell_type_col=args.cell_type_col,
        batch_col=args.batch_col,
        covariate_cols=_csv_to_tuple(args.covariate_cols),
    )
    if detection.sample_col is None or detection.group_col is None:
        raise ValueError(
            "Could not determine required sample/group columns automatically. "
            f"Available obs columns: {list(adata.obs.columns)}"
        )

    backend_available, backend_message = _batch_aware_backend_available(
        bool(args.force_simple_backend),
        str(args.surrogate_variable_mode),
    )
    plans = _plan_cell_types(
        adata,
        sample_col=detection.sample_col,
        group_col=detection.group_col,
        group_a=args.group_a,
        group_b=args.group_b,
        cell_type_col=detection.cell_type_col,
        min_cells_per_sample=args.min_cells_per_sample,
    )
    selected = _select_cell_types(
        plans,
        requested=_csv_to_tuple(args.cell_types),
        max_cell_types=args.max_cell_types,
        all_cell_types=bool(args.all_cell_types),
    )

    preflight_path = outdir / "preflight_report.html"
    _render_preflight_html(
        out_path=preflight_path,
        adata_path=Path(args.adata),
        adata=adata,
        detection=detection,
        plans=plans,
        selected=selected,
        group_a=args.group_a,
        group_b=args.group_b,
        backend_message=backend_message,
        wizard_only=bool(args.wizard_only),
    )
    (outdir / "preflight_summary.json").write_text(
        json.dumps(
            {
                "adata_path": str(args.adata),
                "n_cells": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
                "detected_columns": asdict(detection),
                "plans": [asdict(plan) for plan in plans],
                "selected": [asdict(plan) for plan in selected],
                "backend_message": backend_message,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"- preflight_html: {preflight_path}")

    if args.wizard_only:
        print("nPathway scRNA easy preflight completed.")
        print(f"- detected_sample_col: {detection.sample_col}")
        print(f"- detected_group_col: {detection.group_col}")
        print(f"- detected_cell_type_col: {detection.cell_type_col}")
        print(f"- selected_runs: {len(selected)}")
        return
    if not selected:
        raise ValueError(
            "No eligible cell types were found for the requested contrast. "
            f"Open the preflight report for details: {preflight_path}"
        )

    annotation_gmt = args.annotation_gmt
    annotate_programs = bool(args.annotate_programs)
    if args.curated_gmt and annotation_gmt is None:
        annotation_gmt = args.curated_gmt
        annotate_programs = True

    run_results: list[CellTypeRunResult] = []
    analyses_dir = outdir / "analyses"
    analyses_dir.mkdir(parents=True, exist_ok=True)

    for plan in selected:
        run_dir = analyses_dir / _slugify(plan.label)
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Running scRNA easy workflow for cell type '%s'", plan.label)
        run = _run_single_cell_type(
            adata=adata,
            plan=plan,
            cell_type_col=detection.cell_type_col,
            sample_col=detection.sample_col,
            group_col=detection.group_col,
            group_a=args.group_a,
            group_b=args.group_b,
            batch_col=detection.batch_col,
            covariate_cols=tuple(detection.covariate_cols),
            min_cells_per_sample=args.min_cells_per_sample,
            layer=args.layer,
            use_raw=bool(args.use_raw),
            outdir=run_dir,
            use_batch_aware_backend=backend_available,
            discovery_method=args.discovery_method,
            n_programs=args.n_programs,
            k_neighbors=args.k_neighbors,
            resolution=args.resolution,
            n_components=args.n_components,
            n_diffusion_steps=args.n_diffusion_steps,
            diffusion_alpha=args.diffusion_alpha,
            gsea_n_perm=args.gsea_n_perm,
            seed=42,
            annotate_programs=annotate_programs,
            annotation_collections=_csv_to_tuple(args.annotation_collections),
            annotation_species=args.annotation_species,
            annotation_gmt=annotation_gmt,
            annotation_topk_per_program=args.annotation_topk_per_program,
            annotation_min_jaccard_for_label=args.annotation_min_jaccard_for_label,
            curated_gmt=args.curated_gmt,
            focus_genes=_csv_to_tuple(args.focus_genes),
            with_dashboard=bool(args.with_dashboard),
            dashboard_top_k=args.dashboard_top_k,
            include_pdf=bool(args.with_pdf),
            surrogate_variable_mode=str(args.surrogate_variable_mode),
            sva_max_n_sv=int(args.sva_max_n_sv),
            sva_min_residual_df=int(args.sva_min_residual_df),
            verbose=bool(args.verbose),
        )
        run_results.append(run)
        print(f"- completed_cell_type: {run.label}")
        print(f"  output_dir: {run.output_dir}")
        print(f"  ranking_source: {run.ranking_source}")
        print(f"  n_programs: {run.n_programs}")
        if run.comparison_dir is not None:
            print(f"  comparison_dir: {run.comparison_dir}")
            print(f"  anchor_reference: {run.anchor_reference}")
            print(f"  anchor_program: {run.anchor_program}")

    summary_df = pd.DataFrame([asdict(run) for run in run_results])
    summary_csv = outdir / "cell_type_run_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    index_html = outdir / "analysis_index.html"
    figure_ready_dir = outdir / "figure_ready" if args.figure_ready_export else None
    _render_run_index_html(
        out_path=index_html,
        adata_path=Path(args.adata),
        detection=detection,
        runs=run_results,
        preflight_path=preflight_path,
        figure_ready_path=None if figure_ready_dir is None else figure_ready_dir / "figure_ready_manifest.json",
    )
    if args.figure_ready_export:
        figure_ready_dir = _write_figure_ready_export(
            outdir=outdir,
            preflight_path=preflight_path,
            summary_csv=summary_csv,
            index_html=index_html,
            runs=run_results,
        )
    (outdir / "scrna_easy_manifest.json").write_text(
        json.dumps(
            {
                "adata_path": str(args.adata),
                "detected_columns": asdict(detection),
                "backend_message": backend_message,
                "selected_runs": [asdict(run) for run in run_results],
                "summary_csv": str(summary_csv),
                "analysis_index_html": str(index_html),
                "figure_ready_dir": None if figure_ready_dir is None else str(figure_ready_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("nPathway scRNA easy workflow completed.")
    print(f"- output_dir: {outdir}")
    print(f"- selected_runs: {len(run_results)}")
    print(f"- summary_csv: {summary_csv}")
    print(f"- analysis_index_html: {index_html}")
    if figure_ready_dir is not None:
        print(f"- figure_ready_dir: {figure_ready_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "Hint: start with `npathway run scrna --wizard-only ...` to inspect auto-detected columns and eligible cell types before the full run.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
