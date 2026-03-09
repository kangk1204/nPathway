"""Three-agent dashboard builder for dynamic pathway analysis outputs."""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.errors import EmptyDataError

from npathway.evaluation.enrichment import _compute_enrichment_score
from npathway.evaluation.pathway_annotation import (
    family_interpretation_score,
    reference_relevance_band,
    reference_relevance_score,
    source_interpretation_score,
)
from npathway.utils import read_gmt, write_gmt

logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Configuration for dashboard package generation."""

    results_dir: str
    output_dir: str
    title: str = "nPathway Dynamic Pathway Dashboard"
    top_k: int = 20
    include_pdf: bool = True


@dataclass
class DashboardArtifacts:
    """Output paths from dashboard package generation."""

    html_path: str
    figure_dir: str
    table_dir: str
    summary_table_path: str


@dataclass
class _ResultBundle:
    """Normalized bundle of analysis data tables."""

    de_df: pd.DataFrame
    gsea_df: pd.DataFrame
    program_sizes_df: pd.DataFrame
    context_df: pd.DataFrame
    fisher_df: pd.DataFrame | None
    program_long_df: pd.DataFrame | None
    annotation_df: pd.DataFrame | None
    overlap_df: pd.DataFrame | None
    ranked_genes_df: pd.DataFrame | None
    reference_gene_sets: dict[str, list[str]] | None
    curated_panel_gsea_df: pd.DataFrame | None
    curated_panel_gene_sets: dict[str, list[str]] | None


class ResultIngestionAgent:
    """Agent 1: Validate and ingest analysis outputs.

    Supports both the new organized subdirectory layout and the legacy flat layout.

    New layout::

        results_dir/
        ├── differential/de_results.csv, ranked_genes_for_gsea.csv
        ├── enrichment/enrichment_gsea_with_claim_gates.csv, enrichment_fisher.csv, contextual_membership_scores.csv
        ├── discovery/dynamic_program_sizes.csv, dynamic_programs_long.csv, dynamic_programs.gmt
        ├── membership/program_gene_membership_ranked.csv, program_gene_membership_top20.csv
        └── annotation/program_annotation_matches.csv, program_reference_overlap_long.csv

    Legacy layout: all files directly in results_dir/.
    """

    # (logical_name, new_subpath, legacy_flat_name)
    _FILE_MAP: tuple[tuple[str, str, str], ...] = (
        ("de_results", "differential/de_results.csv", "de_results.csv"),
        ("gsea", "enrichment/enrichment_gsea_with_claim_gates.csv", "enrichment_gsea_with_claim_gates.csv"),
        ("program_sizes", "discovery/dynamic_program_sizes.csv", "dynamic_program_sizes.csv"),
        ("context", "enrichment/contextual_membership_scores.csv", "contextual_membership_scores.csv"),
        ("fisher", "enrichment/enrichment_fisher.csv", "enrichment_fisher.csv"),
        ("program_long", "discovery/dynamic_programs_long.csv", "dynamic_programs_long.csv"),
        ("annotation", "annotation/program_annotation_matches.csv", "program_annotation_matches.csv"),
        ("overlap", "annotation/program_reference_overlap_long.csv", "program_reference_overlap_long.csv"),
        ("ranked_genes", "differential/ranked_genes_for_gsea.csv", "ranked_genes_for_gsea.csv"),
    )

    REQUIRED_KEYS: tuple[str, ...] = ("de_results", "gsea", "program_sizes", "context")

    def _resolve_path(self, results_dir: Path, new_sub: str, legacy: str) -> Path | None:
        """Return the first existing path (new layout preferred, legacy fallback)."""
        new_path = results_dir / new_sub
        if new_path.exists():
            return new_path
        legacy_path = results_dir / legacy
        if legacy_path.exists():
            return legacy_path
        return None

    def load(self, results_dir: Path) -> _ResultBundle:
        """Load required and optional result files."""
        resolved: dict[str, Path | None] = {}
        for key, new_sub, legacy in self._FILE_MAP:
            resolved[key] = self._resolve_path(results_dir, new_sub, legacy)

        missing = [k for k in self.REQUIRED_KEYS if resolved[k] is None]
        if missing:
            raise FileNotFoundError(
                "Missing required files in results_dir: " + ", ".join(missing)
            )

        de_df = pd.read_csv(resolved["de_results"])  # type: ignore[arg-type]
        gsea_df = pd.read_csv(resolved["gsea"])  # type: ignore[arg-type]
        program_sizes_df = pd.read_csv(resolved["program_sizes"])  # type: ignore[arg-type]
        context_df = pd.read_csv(resolved["context"])  # type: ignore[arg-type]

        fisher_df = self._read_optional_csv(resolved["fisher"])
        program_long_df = self._read_optional_csv(resolved["program_long"])
        annotation_df = self._read_optional_csv(resolved["annotation"])
        overlap_df = self._read_optional_csv(resolved["overlap"])
        ranked_genes_df = self._load_ranked_genes(results_dir, resolved["ranked_genes"])
        reference_gene_sets = self._load_reference_gene_sets(results_dir)
        curated_panel_gsea_df = self._load_curated_panel_gsea(results_dir)
        curated_panel_gene_sets = self._load_curated_panel_gene_sets(results_dir)

        self._validate_columns(de_df, gsea_df, program_sizes_df, context_df)
        return _ResultBundle(
            de_df=de_df,
            gsea_df=gsea_df,
            program_sizes_df=program_sizes_df,
            context_df=context_df,
            fisher_df=fisher_df,
            program_long_df=program_long_df,
            annotation_df=annotation_df,
            overlap_df=overlap_df,
            ranked_genes_df=ranked_genes_df,
            reference_gene_sets=reference_gene_sets,
            curated_panel_gsea_df=curated_panel_gsea_df,
            curated_panel_gene_sets=curated_panel_gene_sets,
        )

    @staticmethod
    def _read_optional_csv(path: Path | None) -> pd.DataFrame | None:
        """Read optional CSV and return None for missing/empty files."""
        if path is None or not path.exists():
            return None
        try:
            return pd.read_csv(path)
        except EmptyDataError:
            return None

    def _load_ranked_genes(
        self,
        results_dir: Path,
        direct_path: Path | None,
    ) -> pd.DataFrame | None:
        """Load ranked genes from canonical output or manifest pointers."""
        if direct_path is not None and direct_path.exists():
            return pd.read_csv(direct_path)

        manifest_candidates = [results_dir / "run_manifest.json", results_dir / "workflow_manifest.json"]
        for manifest_path in manifest_candidates:
            if not manifest_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            candidate = (
                manifest.get("config", {}).get("ranked_genes_path")
                or manifest.get("prepared", {}).get("ranked_genes_path")
            )
            if candidate:
                path = self._resolve_manifest_path(results_dir, str(candidate))
                if path.exists():
                    return pd.read_csv(path)
        return None

    def _load_reference_gene_sets(self, results_dir: Path) -> dict[str, list[str]] | None:
        """Load curated/reference GMT if the analysis manifest records it."""
        manifest_candidates = [results_dir / "run_manifest.json", results_dir / "workflow_manifest.json"]
        for manifest_path in manifest_candidates:
            if not manifest_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            candidate = (
                manifest.get("config", {}).get("annotation_gmt_path")
                or manifest.get("curated_gmt")
            )
            if candidate:
                path = self._resolve_manifest_path(results_dir, str(candidate))
                if path.exists():
                    try:
                        return read_gmt(str(path))
                    except Exception:
                        logger.warning("Failed to read reference GMT for dashboard: %s", path)
        return None

    def _load_curated_panel_gsea(self, results_dir: Path) -> pd.DataFrame | None:
        """Load canonical curated panel GSEA output when present."""
        candidates = [
            results_dir / "comparison" / "curated_panel_gsea.csv",
            results_dir / "curated_panel_gsea.csv",
        ]
        for path in candidates:
            if path.exists():
                return pd.read_csv(path)
        return None

    def _load_curated_panel_gene_sets(self, results_dir: Path) -> dict[str, list[str]] | None:
        """Load canonical curated panel GMT when present."""
        candidates = [
            results_dir / "comparison" / "curated_panel_gene_sets.gmt",
            results_dir / "curated_panel_gene_sets.gmt",
        ]
        for path in candidates:
            if path.exists():
                try:
                    return read_gmt(str(path))
                except Exception:
                    logger.warning("Failed to read curated panel GMT for dashboard: %s", path)
                    return None
        return None

    @staticmethod
    def _resolve_manifest_path(results_dir: Path, raw_path: str) -> Path:
        """Resolve manifest paths recorded as absolute, cwd-relative, or results-relative."""
        candidate = Path(str(raw_path))
        if candidate.is_absolute():
            return candidate

        cwd_relative = (Path.cwd() / candidate).resolve()
        if cwd_relative.exists():
            return cwd_relative

        results_relative = (results_dir / candidate).resolve()
        if results_relative.exists():
            return results_relative

        parent_relative = (results_dir.parent / candidate).resolve()
        if parent_relative.exists():
            return parent_relative

        return cwd_relative

    @staticmethod
    def _validate_columns(
        de_df: pd.DataFrame,
        gsea_df: pd.DataFrame,
        program_sizes_df: pd.DataFrame,
        context_df: pd.DataFrame,
    ) -> None:
        """Validate key columns needed by downstream agents."""
        required_de = {"gene", "logfc_a_minus_b", "p_value", "fdr"}
        required_gsea = {"program", "nes", "p_value", "fdr"}
        required_size = {"program", "n_genes"}
        required_context = {"program", "gene", "base_membership"}

        missing_de = required_de - set(de_df.columns)
        missing_gsea = required_gsea - set(gsea_df.columns)
        missing_size = required_size - set(program_sizes_df.columns)
        missing_context = required_context - set(context_df.columns)
        context_prob_cols = [c for c in context_df.columns if c.startswith("prob_")]
        context_score_cols = [c for c in context_df.columns if c.startswith("contextual_score_")]
        has_or_can_derive_context_shift = (
            "context_shift" in context_df.columns
            or len(context_prob_cols) >= 2
            or len(context_score_cols) >= 2
            or context_df.empty
        )

        if missing_de or missing_gsea or missing_size or missing_context or not has_or_can_derive_context_shift:
            parts: list[str] = []
            if missing_de:
                parts.append(f"de_results: {sorted(missing_de)}")
            if missing_gsea:
                parts.append(f"gsea_results: {sorted(missing_gsea)}")
            if missing_size:
                parts.append(f"program_sizes: {sorted(missing_size)}")
            if missing_context:
                parts.append(f"context_scores: {sorted(missing_context)}")
            if not has_or_can_derive_context_shift:
                parts.append(
                    "context_scores: missing 'context_shift' and no compatible fallback "
                    "columns (need >=2 'prob_*' or >=2 'contextual_score_*')."
                )
            raise ValueError("Input schema mismatch: " + "; ".join(parts))


class VisualDesignAgent:
    """Agent 2: Build figure-like publication style plots."""

    def create_static_figures(
        self,
        bundle: _ResultBundle,
        figure_dir: Path,
        top_k: int,
        include_pdf: bool,
    ) -> None:
        """Create static figure assets."""
        figure_dir.mkdir(parents=True, exist_ok=True)
        self._set_style()

        self._plot_volcano(bundle.de_df, figure_dir, include_pdf)
        self._plot_program_sizes(bundle.program_sizes_df, figure_dir, top_k, include_pdf)
        self._plot_claim_gates(bundle.gsea_df, figure_dir, include_pdf)
        self._plot_context_shift(bundle.context_df, figure_dir, top_k, include_pdf)
        self._plot_multi_pathway_reference_view(bundle, figure_dir, include_pdf)
        self._plot_multi_pathway_enrichment_curves(bundle, figure_dir, include_pdf)
        self._plot_reference_ranking_calibration(bundle, figure_dir, include_pdf)

    @staticmethod
    def _set_style() -> None:
        sns.set_theme(style="whitegrid", context="talk")
        plt.rcParams.update(
            {
                "figure.dpi": 150,
                "savefig.dpi": 300,
                "savefig.bbox": "tight",
                "axes.spines.top": False,
                "axes.spines.right": False,
                "font.family": "sans-serif",
            }
        )

    @staticmethod
    def _save(fig: plt.Figure, stem: Path, include_pdf: bool) -> None:
        fig.savefig(stem.with_suffix(".png"))
        if include_pdf:
            fig.savefig(stem.with_suffix(".pdf"))
        plt.close(fig)

    def _plot_volcano(self, de_df: pd.DataFrame, figure_dir: Path, include_pdf: bool) -> None:
        df = de_df.copy()
        df["neglog10p"] = -np.log10(np.clip(df["p_value"].astype(float), 1e-300, 1.0))
        df["sig"] = df["fdr"].astype(float) <= 0.05

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(
            df.loc[~df["sig"], "logfc_a_minus_b"],
            df.loc[~df["sig"], "neglog10p"],
            s=14,
            alpha=0.55,
            c="#4A5C6A",
            label="Not significant",
        )
        ax.scatter(
            df.loc[df["sig"], "logfc_a_minus_b"],
            df.loc[df["sig"], "neglog10p"],
            s=20,
            alpha=0.85,
            c="#E4572E",
            label="FDR <= 0.05",
        )
        ax.axhline(-np.log10(0.05), linestyle="--", linewidth=1.2, color="#999999")
        ax.axvline(0.0, linestyle="-", linewidth=1.0, color="#777777")
        ax.set_xlabel("logFC (Group A - Group B)")
        ax.set_ylabel("-log10(p-value)")
        ax.set_title("Figure 1. Differential Signal Volcano")
        ax.legend(frameon=False)
        self._save(fig, figure_dir / "figure_1_volcano", include_pdf)

    def _plot_program_sizes(
        self,
        program_sizes_df: pd.DataFrame,
        figure_dir: Path,
        top_k: int,
        include_pdf: bool,
    ) -> None:
        top = (
            program_sizes_df.sort_values("n_genes", ascending=False)
            .head(min(top_k, 14))
            .copy()
        )
        top["program_display_axis"] = top["program"].astype(str).map(
            lambda p: _display_program_name_axis(p, max_words=2, max_chars=26)
        )
        fig_height = min(15.5, max(7.2, 0.56 * len(top) + 2.0))
        fig, ax = plt.subplots(figsize=(11.4, fig_height))
        sns.barplot(
            data=top,
            x="n_genes",
            y="program_display_axis",
            hue="program_display_axis",
            dodge=False,
            legend=False,
            orient="h",
            palette="crest",
            ax=ax,
        )
        ax.set_xlabel("Gene Count")
        ax.set_ylabel("Program (ID + pathway)")
        ax.tick_params(axis="y", labelsize=9.5)
        ax.set_title("Figure 2. Program Size Landscape")
        self._save(fig, figure_dir / "figure_2_program_sizes", include_pdf)

    def _plot_claim_gates(self, gsea_df: pd.DataFrame, figure_dir: Path, include_pdf: bool) -> None:
        gate_cols = [c for c in gsea_df.columns if c.startswith("gate_")]
        if not gate_cols:
            gate_counts = pd.DataFrame({"gate": ["No gate columns"], "pass_rate": [0.0]})
        else:
            rates = []
            for c in gate_cols:
                s = gsea_df[c]
                valid = s.dropna()
                rate = float(valid.eq(True).mean()) if len(valid) > 0 else np.nan
                rates.append({"gate": c, "pass_rate": rate})
            gate_counts = pd.DataFrame(rates)

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(
            data=gate_counts,
            x="pass_rate",
            y="gate",
            hue="gate",
            dodge=False,
            legend=False,
            orient="h",
            palette="mako",
            ax=ax,
        )
        ax.set_xlim(0, 1)
        ax.set_xlabel("Pass Fraction")
        ax.set_ylabel("Claim Gate")
        ax.set_title("Figure 3. Claim-Gate Pass Rates")
        self._save(fig, figure_dir / "figure_3_claim_gates", include_pdf)

    def _plot_context_shift(
        self,
        context_df: pd.DataFrame,
        figure_dir: Path,
        top_k: int,
        include_pdf: bool,
    ) -> None:
        df = _ensure_context_metrics(context_df)
        top = df.sort_values("abs_context_evidence", ascending=False).head(top_k)
        if top.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No context rows available", ha="center", va="center", fontsize=12)
            ax.set_axis_off()
            ax.set_title("Figure 4. Top Context-Evidence Genes")
            self._save(fig, figure_dir / "figure_4_context_shift", include_pdf)
            return
        top["program_display_short"] = top["program"].astype(str).map(_display_program_name_short)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=top,
            x="context_evidence",
            y="gene",
            hue="program_display_short",
            dodge=False,
            palette="viridis",
            ax=ax,
        )
        ax.axvline(0.0, color="#333333", linewidth=1.0)
        ax.set_xlabel("Context evidence = context_shift x -log10(p)")
        ax.set_ylabel("Gene")
        ax.set_title("Figure 4. Top Context-Evidence Genes")
        ax.legend(title="Program", loc="best", fontsize=7, title_fontsize=8, frameon=False)
        self._save(fig, figure_dir / "figure_4_context_shift", include_pdf)

    def _plot_multi_pathway_reference_view(
        self,
        bundle: _ResultBundle,
        figure_dir: Path,
        include_pdf: bool,
    ) -> None:
        rows = _prepare_multi_pathway_rows(
            overlap_df=bundle.overlap_df,
            gsea_df=bundle.gsea_df,
            top_k_programs=12,
            top_n_refs=6,
        )
        if rows.empty:
            fig, ax = plt.subplots(figsize=(10, 5.4))
            ax.text(
                0.5,
                0.5,
                "No program-reference overlap data found.\nRun with --annotate-programs to populate this figure.",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.set_axis_off()
            ax.set_title("Figure 5. Multi-Pathway Reference View")
            self._save(fig, figure_dir / "figure_5_multi_pathway", include_pdf)
            return

        lead_program = rows["program"].iloc[0]
        if "program" in bundle.gsea_df.columns and "fdr" in bundle.gsea_df.columns:
            gsea_rank = bundle.gsea_df.copy()
            gsea_rank["fdr"] = _coerce_numeric_series(gsea_rank["fdr"], fill_value=1.0)
            lead_program = gsea_rank.sort_values("fdr", ascending=True)["program"].astype(str).iloc[0]
        subset = rows[rows["program"].astype(str) == str(lead_program)].copy()
        subset = subset.sort_values(["jaccard", "overlap_n"], ascending=[True, True])
        subset["reference_display_axis"] = subset["reference_display"].astype(str).map(
            lambda value: _chunk_text(value, width=26).replace("<br>", "\n")
        )

        fig, ax = plt.subplots(figsize=(10, min(8.2, max(4.8, 0.74 * len(subset) + 1.7))))
        ax.hlines(
            y=subset["reference_display_axis"],
            xmin=0,
            xmax=subset["jaccard"],
            color="#9bb7d4",
            linewidth=4,
            alpha=0.85,
        )
        scatter = ax.scatter(
            subset["jaccard"],
            subset["reference_display_axis"],
            s=np.clip(subset["overlap_n"].to_numpy(dtype=float) * 7.0, 60.0, 420.0),
            c=subset["overlap_n"],
            cmap="viridis",
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )
        for _, row in subset.iterrows():
            ax.text(
                float(row["jaccard"]) + 0.012,
                row["reference_display_axis"],
                f"{int(float(row['overlap_n']))} genes",
                va="center",
                fontsize=8,
                color="#34495e",
            )
        ax.set_xlabel("Program-reference Jaccard overlap")
        ax.set_ylabel("Curated pathway")
        ax.set_title(
            f"Figure 5. Multi-Pathway Reference View for {_display_program_name_short(str(lead_program), max_words=4, max_chars=38)}"
        )
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.84, pad=0.02)
        cbar.set_label("Overlap genes")
        self._save(fig, figure_dir / "figure_5_multi_pathway", include_pdf)

    def _plot_multi_pathway_enrichment_curves(
        self,
        bundle: _ResultBundle,
        figure_dir: Path,
        include_pdf: bool,
    ) -> None:
        """Render a classic GSEA-like multi-pathway running-sum figure."""
        curve_records = _prepare_multi_pathway_curve_records(
            ranked_genes_df=bundle.ranked_genes_df,
            reference_gene_sets=bundle.reference_gene_sets,
            overlap_df=bundle.overlap_df,
            gsea_df=bundle.gsea_df,
            top_k_programs=12,
            top_n_refs=5,
            max_points=800,
            max_hits=180,
        )
        if not curve_records:
            curve_records = _prepare_global_gsea_curve_records(
                ranked_genes_df=bundle.ranked_genes_df,
                reference_gene_sets=(
                    bundle.curated_panel_gene_sets
                    if bundle.curated_panel_gene_sets
                    else bundle.reference_gene_sets
                ),
                reference_score_df=bundle.curated_panel_gsea_df,
                top_n_refs=6,
                max_points=800,
                max_hits=180,
            )
        if not curve_records:
            fig, ax = plt.subplots(figsize=(10.2, 5.6))
            ax.text(
                0.5,
                0.5,
                "No ranked-gene or reference-GMT inputs were found.\nRun with annotation and ranked GSEA outputs to populate enrichment curves.",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.set_axis_off()
            ax.set_title("Figure 6. Multi-Pathway Enrichment Curves")
            self._save(fig, figure_dir / "figure_6_multi_pathway_enrichment_curves", include_pdf)
            return

        lead_program = str(curve_records[0]["program"])
        if lead_program != "__global__" and {"program", "fdr"}.issubset(bundle.gsea_df.columns):
            ranked = bundle.gsea_df.copy()
            ranked["fdr"] = _coerce_numeric_series(ranked["fdr"], fill_value=1.0)
            if not ranked.empty:
                lead_program = str(
                    ranked.sort_values(["fdr", "nes"], ascending=[True, False]).iloc[0]["program"]
                )
        subset = [row for row in curve_records if str(row["program"]) == lead_program][:5]
        if not subset:
            subset = curve_records[:5]
            lead_program = str(subset[0]["program"])

        palette = sns.color_palette("viridis", n_colors=max(len(subset), 3))
        fig, (ax_curve, ax_hits) = plt.subplots(
            2,
            1,
            figsize=(11.2, 7.4),
            sharex=True,
            gridspec_kw={"height_ratios": [4.6, 1.2], "hspace": 0.06},
        )

        for idx, row in enumerate(subset):
            color = palette[idx]
            x_vals = np.asarray(row["x_points"], dtype=float)
            y_vals = np.asarray(row["y_points"], dtype=float)
            label = f"{row['reference_display']} | ES {float(row['es']):.2f}"
            ax_curve.plot(x_vals, y_vals, color=color, linewidth=2.2, label=label)
            hit_positions = np.asarray(row["hit_positions"], dtype=float)
            if hit_positions.size > 0:
                ax_hits.vlines(
                    hit_positions,
                    idx + 0.15,
                    idx + 0.85,
                    color=color,
                    linewidth=0.9,
                    alpha=0.95,
                )

        ax_curve.axhline(0.0, color="#5c6770", linewidth=1.0, linestyle="--", alpha=0.9)
        ax_curve.set_ylabel("Running enrichment score")
        title_suffix = (
            "Global ranked-list view"
            if lead_program == "__global__"
            else _display_program_name_short(lead_program, max_words=4, max_chars=42)
        )
        ax_curve.set_title(f"Figure 6. Multi-Pathway Enrichment Curves for {title_suffix}")
        ax_curve.legend(
            loc="upper right",
            frameon=False,
            fontsize=8,
            title="Curated pathways",
            title_fontsize=9,
        )

        ax_hits.set_ylabel("Gene hits")
        ax_hits.set_xlabel("Rank in preranked gene list")
        ax_hits.set_yticks(np.arange(len(subset)) + 0.5)
        ax_hits.set_yticklabels(
            [
                _chunk_text(str(row["reference_display"]), width=22).replace("<br>", "\n")
                for row in subset
            ],
            fontsize=8,
        )
        ax_hits.set_ylim(0, len(subset))
        ax_hits.invert_yaxis()
        ax_hits.grid(False)
        for spine in ("top", "right"):
            ax_hits.spines[spine].set_visible(False)

        self._save(fig, figure_dir / "figure_6_multi_pathway_enrichment_curves", include_pdf)

    def _plot_reference_ranking_calibration(
        self,
        bundle: _ResultBundle,
        figure_dir: Path,
        include_pdf: bool,
    ) -> None:
        """Show what changed when family ranking moved from raw overlap to interpretation score."""
        calib = _prepare_reference_ranking_calibration_rows(
            overlap_df=bundle.overlap_df,
            gsea_df=bundle.gsea_df,
            top_k_programs=30,
            top_n_display=8,
        )
        if calib.empty:
            fig, ax = plt.subplots(figsize=(10.2, 5.4))
            ax.text(
                0.5,
                0.5,
                "No reference-family overlap data were available.\nRun with pathway annotation enabled to calibrate family ranking.",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.set_axis_off()
            ax.set_title("Figure 7. Reference Ranking Calibration")
            self._save(fig, figure_dir / "figure_7_reference_ranking_calibration", include_pdf)
            return

        raw_view = calib.sort_values(["raw_rank", "best_jaccard"], ascending=[True, False]).head(8).copy()
        prioritized_view = calib.sort_values(
            ["prioritized_rank", "interpretation_score"],
            ascending=[True, False],
        ).head(8).copy()
        raw_view["family_axis"] = raw_view["family_display"].astype(str).map(
            lambda value: _chunk_text(value, width=28).replace("<br>", "\n")
        )
        prioritized_view["family_axis"] = prioritized_view["family_display"].astype(str).map(
            lambda value: _chunk_text(value, width=28).replace("<br>", "\n")
        )

        fig, (ax_raw, ax_prior) = plt.subplots(
            1,
            2,
            figsize=(13.0, 5.6),
            gridspec_kw={"width_ratios": [1.0, 1.1], "wspace": 0.30},
        )

        raw_palette = ["#7b8794" if not bool(v) else "#4c78a8" for v in raw_view["prioritized_top"]]
        ax_raw.barh(
            raw_view["family_axis"].iloc[::-1],
            raw_view["best_jaccard"].iloc[::-1],
            color=raw_palette[::-1],
            edgecolor="white",
            linewidth=1.0,
        )
        ax_raw.set_xlabel("Best Jaccard")
        ax_raw.set_title("Raw overlap ranking", loc="left", fontsize=12, fontweight="bold")
        ax_raw.grid(axis="x", alpha=0.18)
        for _, row in raw_view.iterrows():
            ax_raw.text(
                float(row["best_jaccard"]) + 0.008,
                row["family_axis"],
                f"raw #{int(row['raw_rank'])}",
                va="center",
                fontsize=8,
                color="#34495e",
            )

        prior_colors = [
            {"Disease-prioritized": "#c0392b", "Supportive": "#d68910", "Background": "#95a5a6"}.get(
                str(band),
                "#95a5a6",
            )
            for band in prioritized_view["priority_band"]
        ]
        ax_prior.barh(
            prioritized_view["family_axis"].iloc[::-1],
            prioritized_view["interpretation_score"].iloc[::-1],
            color=prior_colors[::-1],
            edgecolor="white",
            linewidth=1.0,
        )
        ax_prior.set_xlabel("Interpretation score")
        ax_prior.set_title("Interpretation-prioritized ranking", loc="left", fontsize=12, fontweight="bold")
        ax_prior.grid(axis="x", alpha=0.18)
        for _, row in prioritized_view.iterrows():
            ax_prior.text(
                float(row["interpretation_score"]) + 0.15,
                row["family_axis"],
                f"{row['priority_band']} | raw #{int(row['raw_rank']) if pd.notna(row['raw_rank']) else 'NA'}",
                va="center",
                fontsize=8,
                color="#34495e",
            )

        fig.suptitle(
            "Figure 7. Reference Ranking Calibration",
            x=0.06,
            y=0.99,
            ha="left",
            fontsize=14,
            fontweight="bold",
        )
        fig.text(
            0.06,
            0.01,
            "The right panel is an interpretation-layer heuristic that reorders family labels after discovery. "
            "It does not change enrichment statistics or discovery assignments.",
            ha="left",
            va="bottom",
            fontsize=9,
            color="#34495e",
        )
        self._save(fig, figure_dir / "figure_7_reference_ranking_calibration", include_pdf)


class DashboardPublishingAgent:
    """Agent 3: Publish interactive dashboard + publication tables."""

    def publish(
        self,
        bundle: _ResultBundle,
        output_dir: Path,
        figure_dir: Path,
        table_dir: Path,
        title: str,
        top_k: int,
        include_pdf: bool,
    ) -> DashboardArtifacts:
        """Build interactive HTML and summary tables."""
        output_dir.mkdir(parents=True, exist_ok=True)
        table_dir.mkdir(parents=True, exist_ok=True)
        bundle.context_df = _ensure_context_metrics(bundle.context_df)
        gsea_rank = bundle.gsea_df.copy()
        if "fdr" in gsea_rank.columns:
            gsea_rank["fdr"] = _coerce_numeric_series(gsea_rank["fdr"], fill_value=1.0)

        top_enriched = gsea_rank.sort_values("fdr", ascending=True).head(top_k).copy()
        top_context_evidence = (
            bundle.context_df.sort_values("abs_context_evidence", ascending=False)
            .head(top_k)
            .copy()
        )
        top_context_shift = (
            bundle.context_df.sort_values("abs_shift", ascending=False)
            .head(top_k)
            .copy()
        )
        top_context = top_context_evidence
        gene_membership_top = (
            bundle.context_df.sort_values(
                ["program", "base_membership"],
                ascending=[True, False],
            )
            .groupby("program", as_index=False, sort=False)
            .head(15)
            .reset_index(drop=True)
        )
        gate_summary = self._claim_gate_summary(bundle.gsea_df)
        headline_summary = self._headline_summary(bundle)
        reference_source_hits = _prepare_reference_source_rows(
            overlap_df=bundle.overlap_df,
            gsea_df=bundle.gsea_df,
            top_k_programs=30,
            top_n_refs_per_source=12,
        )
        reference_family_hits = _prepare_reference_family_rows(
            overlap_df=bundle.overlap_df,
            gsea_df=bundle.gsea_df,
            top_k_programs=30,
            top_n_families=24,
        )
        ranking_calibration = _prepare_reference_ranking_calibration_rows(
            overlap_df=bundle.overlap_df,
            gsea_df=bundle.gsea_df,
            top_k_programs=30,
            top_n_display=10,
        )

        top_enriched.to_csv(table_dir / "top_enriched_programs.csv", index=False)
        top_context_shift.to_csv(table_dir / "top_context_shift_genes.csv", index=False)
        top_context_evidence.to_csv(table_dir / "top_context_evidence_genes.csv", index=False)
        gene_membership_top.to_csv(table_dir / "top_genes_per_program.csv", index=False)
        gate_summary.to_csv(table_dir / "claim_gate_summary.csv", index=False)
        headline_summary.to_csv(table_dir / "headline_summary.csv", index=False)
        _prepare_multi_pathway_rows(bundle.overlap_df, bundle.gsea_df, top_k_programs=30, top_n_refs=8).to_csv(
            table_dir / "top_multi_pathway_hits.csv",
            index=False,
        )
        reference_source_hits.to_csv(table_dir / "reference_source_hits.csv", index=False)
        reference_family_hits.to_csv(table_dir / "reference_family_hits.csv", index=False)
        ranking_calibration.to_csv(table_dir / "reference_ranking_calibration.csv", index=False)
        if bundle.curated_panel_gsea_df is not None:
            bundle.curated_panel_gsea_df.to_csv(table_dir / "curated_panel_gsea.csv", index=False)
        if bundle.curated_panel_gene_sets is not None:
            write_gmt(bundle.curated_panel_gene_sets, table_dir / "curated_panel_gene_sets.gmt")
        if bundle.annotation_df is not None:
            bundle.annotation_df.to_csv(table_dir / "program_annotation_matches.csv", index=False)
        if bundle.overlap_df is not None:
            bundle.overlap_df.to_csv(table_dir / "program_reference_overlap_long.csv", index=False)
        if bundle.program_long_df is not None:
            bundle.program_long_df.to_csv(table_dir / "program_gene_membership_long.csv", index=False)

        summary_table_path = table_dir / "headline_summary.csv"
        html_path = output_dir / "index.html"
        html = self._build_html(
            title=title,
            bundle=bundle,
            top_enriched=top_enriched,
            top_context=top_context,
            gene_membership_top=gene_membership_top,
            gate_summary=gate_summary,
            headline_summary=headline_summary,
            reference_source_hits=reference_source_hits,
            reference_family_hits=reference_family_hits,
            ranking_calibration=ranking_calibration,
            figure_dir=figure_dir,
            table_dir=table_dir,
            include_pdf=include_pdf,
            overlap_df=bundle.overlap_df,
        )
        html_path.write_text(html, encoding="utf-8")
        summary_md_path = output_dir / "summary.md"
        summary_md_path.write_text(
            self._build_summary_markdown(
                title=title,
                headline_summary=headline_summary,
                gate_summary=gate_summary,
                narrative_summary=self._narrative_summary(bundle, self._program_summary(bundle), gate_summary),
                reference_family_hits=reference_family_hits,
                reference_source_hits=reference_source_hits,
                ranking_calibration=ranking_calibration,
                include_pdf=include_pdf,
            ),
            encoding="utf-8",
        )

        return DashboardArtifacts(
            html_path=str(html_path),
            figure_dir=str(figure_dir),
            table_dir=str(table_dir),
            summary_table_path=str(summary_table_path),
        )

    @staticmethod
    def _claim_gate_summary(gsea_df: pd.DataFrame) -> pd.DataFrame:
        gate_cols = [c for c in gsea_df.columns if c.startswith("gate_")]
        rows: list[dict[str, Any]] = []
        for c in gate_cols:
            valid = gsea_df[c].dropna()
            pass_count = int(valid.eq(True).sum())
            n = int(len(valid))
            rows.append(
                {
                    "gate": c,
                    "gate_display": _humanize_gate_name(c),
                    "n_evaluable": n,
                    "n_pass": pass_count,
                    "pass_rate": float(pass_count / n) if n > 0 else np.nan,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _headline_summary(bundle: _ResultBundle) -> pd.DataFrame:
        de_sig = int((bundle.de_df["fdr"] <= 0.05).sum())
        n_programs = int(bundle.program_sizes_df["program"].nunique())
        n_claim_supported = (
            int(bundle.gsea_df["claim_supported"].sum())
            if "claim_supported" in bundle.gsea_df.columns
            else 0
        )
        return pd.DataFrame(
            [
                {
                    "metric": "n_genes_tested",
                    "metric_display": "Genes tested",
                    "value": int(len(bundle.de_df)),
                },
                {
                    "metric": "n_significant_de_genes",
                    "metric_display": "Significant DE genes",
                    "value": de_sig,
                },
                {
                    "metric": "n_programs_discovered",
                    "metric_display": "Programs discovered",
                    "value": n_programs,
                },
                {
                    "metric": "n_claim_supported_programs",
                    "metric_display": "Claim-supported programs",
                    "value": n_claim_supported,
                },
            ]
        )

    @staticmethod
    def _signal_interpretation_alert(n_de_sig: int, n_claim_supported: int) -> dict[str, str] | None:
        """Describe cohorts where program-level signal remains despite sparse DEG signal."""
        if n_claim_supported <= 0:
            return None
        if n_de_sig == 0:
            return {
                "title": "Sparse single-gene signal",
                "text": (
                    "No genes pass single-gene FDR <= 0.05 in this run, but claim-supported programs remain. "
                    "Read supported programs as coordinated low-amplitude shifts across many genes rather than as isolated DEG hits."
                ),
            }
        if n_de_sig <= 5:
            return {
                "title": "Limited single-gene signal",
                "text": (
                    f"Only {n_de_sig} genes pass single-gene FDR <= 0.05. Treat supported programs as weak-but-coordinated transcriptome structure and inspect ranked-gene and leading-edge views before making strong biological claims."
                ),
            }
        return None

    @staticmethod
    def _build_summary_markdown(
        title: str,
        headline_summary: pd.DataFrame,
        gate_summary: pd.DataFrame,
        narrative_summary: list[dict[str, str | None]],
        reference_family_hits: pd.DataFrame,
        reference_source_hits: pd.DataFrame,
        ranking_calibration: pd.DataFrame,
        include_pdf: bool,
    ) -> str:
        """Build a reviewer-first markdown summary alongside the interactive HTML."""
        metric_map = {
            str(row["metric_display"]): row["value"]
            for _, row in headline_summary.iterrows()
        }
        def _as_int(metric_name: str) -> int:
            try:
                return int(float(metric_map.get(metric_name, 0)))
            except (TypeError, ValueError):
                return 0

        signal_alert = DashboardPublishingAgent._signal_interpretation_alert(
            _as_int("Significant DE genes"),
            _as_int("Claim-supported programs"),
        )
        lead_lines = [
            f"- {str(item['label'])}: {str(item['text'])}"
            for item in narrative_summary[:3]
        ]
        top_family = reference_family_hits.iloc[0] if not reference_family_hits.empty else None
        top_source = reference_source_hits.iloc[0] if not reference_source_hits.empty else None
        calibration_shift = ranking_calibration.loc[
            ranking_calibration["raw_top"] != ranking_calibration["prioritized_top"]
        ].copy()
        calibration_note = (
            f"{len(calibration_shift)} family labels change top-10 status after interpretation prioritization."
            if not ranking_calibration.empty
            else "No reference-family calibration table was available."
        )
        gate_lines = []
        for _, row in gate_summary.iterrows():
            rate = float(row["pass_rate"]) if pd.notna(row["pass_rate"]) else float("nan")
            gate_lines.append(
                f"- {row['gate_display']}: {int(row['n_pass'])}/{int(row['n_evaluable'])} pass "
                f"({rate:.1%} when evaluable)"
            )
        figure_6_asset = "figures/figure_6_multi_pathway_enrichment_curves.pdf" if include_pdf else "figures/figure_6_multi_pathway_enrichment_curves.png"
        figure_7_asset = "figures/figure_7_reference_ranking_calibration.pdf" if include_pdf else "figures/figure_7_reference_ranking_calibration.png"
        return "\n".join(
            [
                f"# {title}",
                "",
                "## How to read this package",
                "- Start with `index.html` and select the lead enriched program in the Program Explorer.",
                "- Use the multi-pathway enrichment curves to compare how multiple curated pathways trace the same ranked gene list.",
                "- Treat disease-prioritized family ranking as an interpretation heuristic layered after discovery, not as a new significance test.",
                "",
                "## Headline counts",
                f"- Genes tested: {metric_map.get('Genes tested', 'NA')}",
                f"- Significant DE genes: {metric_map.get('Significant DE genes', 'NA')}",
                f"- Programs discovered: {metric_map.get('Programs discovered', 'NA')}",
                f"- Claim-supported programs: {metric_map.get('Claim-supported programs', 'NA')}",
                "",
                "## Lead findings",
                *(lead_lines or ["- No lead narrative summary was available."]),
                (
                    f"- {signal_alert['title']}: {signal_alert['text']}"
                    if signal_alert is not None
                    else "- Single-gene and program-level summaries are both available for cross-checking."
                ),
                (
                    f"- Top disease-prioritized family: {top_family['family_display']} "
                    f"(interpretation score {float(top_family['interpretation_score']):.2f}, "
                    f"best Jaccard {float(top_family['best_jaccard']):.3f})"
                    if top_family is not None
                    else "- Top disease-prioritized family: not available"
                ),
                (
                    f"- Strongest source layer: {top_source['source_display']} "
                    f"(interpretation score {float(top_source['interpretation_score']):.2f}, "
                    f"Jaccard {float(top_source['jaccard']):.3f})"
                    if top_source is not None
                    else "- Strongest source layer: not available"
                ),
                "",
                "## Claim-gate audit",
                *(gate_lines or ["- No claim-gate table was available."]),
                "",
                "## Ranking calibration",
                f"- {calibration_note}",
                "- `reference_ranking_calibration.csv` compares raw Jaccard ranking with interpretation-prioritized ranking.",
                "",
                "## Files to open first",
                "- `index.html`",
                "- `summary.md`",
                "- `tables/reference_family_hits.csv`",
                "- `tables/reference_ranking_calibration.csv`",
                f"- `{figure_6_asset}`",
                f"- `{figure_7_asset}`",
                "",
            ]
        )

    @staticmethod
    def _narrative_summary(
        bundle: _ResultBundle,
        program_summary: pd.DataFrame,
        gate_summary: pd.DataFrame,
    ) -> list[dict[str, str | None]]:
        """Create a concise narrative-first summary for the landing section."""
        bullets: list[dict[str, str | None]] = []

        lead_candidates = program_summary.dropna(subset=["fdr"]).copy()
        if not lead_candidates.empty:
            lead = lead_candidates.sort_values(["fdr", "nes"], ascending=[True, False]).iloc[0]
            bullets.append(
                {
                    "label": "Lead signal",
                    "text": (
                        f"{str(lead['program_display_short'])} is the top enrichment call "
                        f"(NES {float(lead['nes']):.2f}, FDR {float(lead['fdr']):.2e})."
                    ),
                    "program": str(lead["program"]),
                }
            )

        anchor_candidates = program_summary.dropna(subset=["top_reference_jaccard"]).copy()
        if not anchor_candidates.empty:
            anchor = anchor_candidates.sort_values("top_reference_jaccard", ascending=False).iloc[0]
            bullets.append(
                {
                    "label": "Best curated anchor",
                    "text": (
                        f"{str(anchor['program_display_short'])} aligns most strongly to "
                        f"{str(anchor['top_reference_name'])} "
                        f"(Jaccard {float(anchor['top_reference_jaccard']):.2f})."
                    ),
                    "program": str(anchor["program"]),
                }
            )

        context_candidates = program_summary.dropna(subset=["top_context_evidence_abs"]).copy()
        if not context_candidates.empty:
            context = context_candidates.sort_values("top_context_evidence_abs", ascending=False).iloc[0]
            bullets.append(
                {
                    "label": "Strongest context driver",
                    "text": (
                        f"{str(context['top_context_gene'])} is the clearest context-sensitive gene within "
                        f"{str(context['program_display_short'])} "
                        f"(evidence {float(context['top_context_evidence_abs']):.2f})."
                    ),
                    "program": str(context["program"]),
                }
            )

        n_enriched = int(len(bundle.gsea_df))
        n_claim_supported = int(program_summary["claim_supported"].fillna(False).astype(bool).sum())
        mean_gate_pass = (
            float(_coerce_numeric_series(gate_summary["pass_rate"], fill_value=0.0).mean())
            if not gate_summary.empty and "pass_rate" in gate_summary.columns
            else 0.0
        )
        bullets.append(
            {
                "label": "Decision frame",
                "text": (
                    f"{n_claim_supported} of {n_enriched} enriched programs currently clear the full "
                    f"claim gate stack; mean gate pass rate is {mean_gate_pass:.0%}."
                ),
                "program": None,
            }
        )

        de_sig = int((bundle.de_df["fdr"] <= 0.05).sum()) if "fdr" in bundle.de_df.columns else 0
        signal_alert = DashboardPublishingAgent._signal_interpretation_alert(de_sig, n_claim_supported)
        if signal_alert is not None:
            bullets.append(
                {
                    "label": signal_alert["title"],
                    "text": signal_alert["text"],
                    "program": None,
                }
            )

        return bullets[:5]

    @staticmethod
    def _hero_metrics(
        bundle: _ResultBundle,
        gate_summary: pd.DataFrame,
    ) -> list[dict[str, str | None]]:
        """Build compact KPI tiles for the dashboard hero."""
        context_df = _ensure_context_metrics(bundle.context_df)
        gsea_df = bundle.gsea_df.copy()
        if "fdr" in gsea_df.columns:
            gsea_df["fdr"] = _coerce_numeric_series(gsea_df["fdr"], fill_value=1.0)
        if "nes" in gsea_df.columns:
            gsea_df["nes"] = _coerce_numeric_series(gsea_df["nes"], fill_value=0.0)

        n_programs = int(bundle.program_sizes_df["program"].nunique())
        de_sig = int((bundle.de_df["fdr"] <= 0.05).sum())
        n_enriched = int(len(gsea_df))
        n_claim_supported = (
            int(_coerce_numeric_series(gsea_df["claim_supported"], fill_value=0.0).sum())
            if "claim_supported" in gsea_df.columns
            else 0
        )
        claim_rate = float(n_claim_supported / n_enriched) if n_enriched > 0 else 0.0
        median_program_size = (
            float(
                _coerce_numeric_series(bundle.program_sizes_df["n_genes"], fill_value=0.0).median()
            )
            if "n_genes" in bundle.program_sizes_df.columns and not bundle.program_sizes_df.empty
            else 0.0
        )
        mean_gate_pass = (
            float(_coerce_numeric_series(gate_summary["pass_rate"], fill_value=0.0).mean())
            if not gate_summary.empty and "pass_rate" in gate_summary.columns
            else 0.0
        )

        metrics: list[dict[str, str | None]] = [
            {
                "label": "Programs",
                "value": f"{n_programs}",
                "detail": "discovered",
                "program": None,
            },
            {
                "label": "Claim-supported",
                "value": f"{n_claim_supported}",
                "detail": f"{claim_rate:.0%} of enriched programs",
                "program": None,
            },
            {
                "label": "DE genes",
                "value": f"{de_sig}",
                "detail": (
                    "single-gene FDR sparse"
                    if de_sig == 0
                    else ("few genes at FDR <= 0.05" if de_sig <= 5 else "FDR <= 0.05")
                ),
                "program": None,
            },
            {
                "label": "Median size",
                "value": f"{median_program_size:.0f}",
                "detail": "genes per program",
                "program": None,
            },
            {
                "label": "Gate pass rate",
                "value": f"{mean_gate_pass:.0%}",
                "detail": "mean across claim gates",
                "program": None,
            },
        ]

        if not gsea_df.empty and {"program", "fdr", "nes"}.issubset(gsea_df.columns):
            lead = gsea_df.sort_values(["fdr", "nes"], ascending=[True, False]).iloc[0]
            metrics.append(
                {
                    "label": "Lead program",
                    "value": _display_program_name_short(str(lead["program"]), max_words=4, max_chars=42),
                    "detail": f"FDR {float(lead['fdr']):.2e} | NES {float(lead['nes']):.2f}",
                    "program": str(lead["program"]),
                }
            )

        if not context_df.empty and {"program", "gene", "abs_context_evidence"}.issubset(context_df.columns):
            strong = context_df.sort_values("abs_context_evidence", ascending=False).iloc[0]
            metrics.append(
                {
                    "label": "Strongest context gene",
                    "value": str(strong["gene"]),
                    "detail": (
                        f"{_display_program_name_short(str(strong['program']), max_words=4, max_chars=34)}"
                        f" | evidence {float(strong['abs_context_evidence']):.2f}"
                    ),
                    "program": str(strong["program"]),
                }
            )

        return metrics

    @staticmethod
    def _program_summary(bundle: _ResultBundle) -> pd.DataFrame:
        """Assemble a compact per-program summary used by the spotlight UI."""
        summary = bundle.program_sizes_df.copy()
        if summary.empty:
            summary = pd.DataFrame(columns=["program", "n_genes"])
        if "program" not in summary.columns:
            summary["program"] = pd.Series(dtype=str)
        if "n_genes" not in summary.columns:
            summary["n_genes"] = pd.Series(dtype=np.float64)
        summary["program"] = summary["program"].astype(str)
        summary = summary.loc[:, ["program", "n_genes"]].drop_duplicates("program").copy()
        summary["n_genes"] = _coerce_numeric_series(summary["n_genes"], fill_value=0.0)

        gsea_cols = [c for c in ["program", "nes", "fdr", "p_value", "claim_supported"] if c in bundle.gsea_df.columns]
        if gsea_cols:
            gsea = bundle.gsea_df.loc[:, gsea_cols].copy()
            gsea["program"] = gsea["program"].astype(str)
            if "fdr" in gsea.columns:
                gsea["fdr"] = _coerce_numeric_series(gsea["fdr"], fill_value=1.0)
            if "nes" in gsea.columns:
                gsea["nes"] = _coerce_numeric_series(gsea["nes"], fill_value=0.0)
            if "p_value" in gsea.columns:
                gsea["p_value"] = _coerce_numeric_series(gsea["p_value"], fill_value=1.0)
            if "claim_supported" in gsea.columns:
                gsea["claim_supported"] = gsea["claim_supported"].fillna(False).astype(bool)
            else:
                gsea["claim_supported"] = False
            order_cols = ["fdr", "nes"] if {"fdr", "nes"}.issubset(gsea.columns) else ["program"]
            ascending = [True, False] if order_cols == ["fdr", "nes"] else [True]
            gsea = gsea.sort_values(order_cols, ascending=ascending).drop_duplicates("program")
            summary = summary.merge(gsea, on="program", how="outer")

        context = _ensure_context_metrics(bundle.context_df)
        if not context.empty:
            context["program"] = context["program"].astype(str)
            context["base_membership"] = _coerce_numeric_series(context["base_membership"], fill_value=0.0)
            top_context = (
                context.sort_values(["program", "abs_context_evidence"], ascending=[True, False])
                .groupby("program", as_index=False, sort=False)
                .head(1)
                .loc[
                    :,
                    ["program", "gene", "context_evidence", "abs_context_evidence", "context_shift"],
                ]
                .rename(
                    columns={
                        "gene": "top_context_gene",
                        "context_evidence": "top_context_evidence",
                        "abs_context_evidence": "top_context_evidence_abs",
                        "context_shift": "top_context_shift",
                    }
                )
            )
            top_membership = (
                context.sort_values(["program", "base_membership"], ascending=[True, False])
                .groupby("program", as_index=False, sort=False)
                .head(1)
                .loc[:, ["program", "gene", "base_membership"]]
                .rename(
                    columns={
                        "gene": "top_membership_gene",
                        "base_membership": "top_membership",
                    }
                )
            )
            summary = summary.merge(top_context, on="program", how="left")
            summary = summary.merge(top_membership, on="program", how="left")

        overlap = _prepare_overlap_heatmap_long(
            overlap_df=bundle.overlap_df,
            gsea_df=bundle.gsea_df,
            top_k=max(50, int(max(len(summary), 1))),
        )
        if not overlap.empty:
            best_overlap = (
                overlap.sort_values(["program", "jaccard"], ascending=[True, False])
                .groupby("program", as_index=False, sort=False)
                .head(1)
                .rename(
                    columns={
                        "reference_name": "top_reference_name",
                        "jaccard": "top_reference_jaccard",
                    }
                )
            )
            summary = summary.merge(best_overlap, on="program", how="left")

        summary["program_display"] = summary["program"].map(_display_program_name)
        summary["program_display_short"] = summary["program"].map(_display_program_name_short)
        if "fdr" in summary.columns:
            summary["fdr"] = _coerce_numeric_series(summary["fdr"], fill_value=np.inf)
            summary = summary.sort_values(["fdr", "n_genes"], ascending=[True, False]).copy()
            summary["fdr"] = summary["fdr"].replace(np.inf, np.nan)
        else:
            summary = summary.sort_values("n_genes", ascending=False).copy()

        columns = [
            "program",
            "program_display",
            "program_display_short",
            "n_genes",
            "nes",
            "fdr",
            "p_value",
            "claim_supported",
            "top_context_gene",
            "top_context_evidence",
            "top_context_evidence_abs",
            "top_context_shift",
            "top_membership_gene",
            "top_membership",
            "top_reference_name",
            "top_reference_jaccard",
        ]
        for column in columns:
            if column not in summary.columns:
                summary[column] = np.nan
        return summary.loc[:, columns].reset_index(drop=True)

    @staticmethod
    def _story_cards(program_summary: pd.DataFrame) -> list[dict[str, str | None]]:
        """Create narrative focus cards that jump into key programs."""
        cards: list[dict[str, str | None]] = []

        lead_candidates = program_summary.dropna(subset=["fdr"]).copy()
        if not lead_candidates.empty:
            lead = lead_candidates.sort_values(["fdr", "nes"], ascending=[True, False]).iloc[0]
            cards.append(
                {
                    "eyebrow": "Lead enrichment",
                    "title": str(lead["program_display_short"]),
                    "detail": f"Lowest FDR program with NES {float(lead['nes']):.2f}.",
                    "program": str(lead["program"]),
                }
            )

        size_candidates = program_summary.dropna(subset=["n_genes"]).copy()
        if not size_candidates.empty:
            large = size_candidates.sort_values("n_genes", ascending=False).iloc[0]
            cards.append(
                {
                    "eyebrow": "Largest footprint",
                    "title": str(large["program_display_short"]),
                    "detail": f"{int(float(large['n_genes']))} genes contribute to this program.",
                    "program": str(large["program"]),
                }
            )

        context_candidates = program_summary.dropna(subset=["top_context_evidence_abs"]).copy()
        if not context_candidates.empty:
            shock = context_candidates.sort_values("top_context_evidence_abs", ascending=False).iloc[0]
            cards.append(
                {
                    "eyebrow": "Strongest context shift",
                    "title": str(shock["top_context_gene"]),
                    "detail": (
                        f"{str(shock['program_display_short'])} shows the sharpest evidence "
                        f"({float(shock['top_context_evidence_abs']):.2f})."
                    ),
                    "program": str(shock["program"]),
                }
            )

        overlap_candidates = program_summary.dropna(subset=["top_reference_jaccard"]).copy()
        if not overlap_candidates.empty:
            anchor = overlap_candidates.sort_values("top_reference_jaccard", ascending=False).iloc[0]
            cards.append(
                {
                    "eyebrow": "Reference anchor",
                    "title": str(anchor["top_reference_name"]),
                    "detail": (
                        f"{str(anchor['program_display_short'])} aligns best with a curated reference "
                        f"(Jaccard {float(anchor['top_reference_jaccard']):.2f})."
                    ),
                    "program": str(anchor["program"]),
                }
            )

        if not cards:
            cards.append(
                {
                    "eyebrow": "No program summary",
                    "title": "Interactive views remain available",
                    "detail": "Program metadata will appear here once enrichment and context tables are populated.",
                    "program": None,
                }
            )
        return cards[:4]

    def _build_html(
        self,
        title: str,
        bundle: _ResultBundle,
        top_enriched: pd.DataFrame,
        top_context: pd.DataFrame,
        gene_membership_top: pd.DataFrame,
        gate_summary: pd.DataFrame,
        headline_summary: pd.DataFrame,
        reference_source_hits: pd.DataFrame,
        reference_family_hits: pd.DataFrame,
        ranking_calibration: pd.DataFrame,
        figure_dir: Path,
        table_dir: Path,
        include_pdf: bool,
        overlap_df: pd.DataFrame | None,
    ) -> str:
        de_for_plot = bundle.de_df.copy()
        de_for_plot["neglog10p"] = -np.log10(np.clip(de_for_plot["p_value"], 1e-300, 1.0))
        de_for_plot["sig"] = de_for_plot["fdr"] <= 0.05

        sizes_for_plot = bundle.program_sizes_df.sort_values("n_genes", ascending=False).head(14).copy()
        gsea_for_plot = bundle.gsea_df.copy()
        if "fdr" in gsea_for_plot.columns:
            gsea_for_plot["fdr"] = _coerce_numeric_series(gsea_for_plot["fdr"], fill_value=1.0)
        gsea_for_plot = gsea_for_plot.sort_values("fdr", ascending=True).head(14).copy()
        context_for_plot = top_context.copy()
        context_program_df, membership_program_df, explorer_programs = _prepare_program_explorer_frames(
            context_df=bundle.context_df,
            gsea_df=bundle.gsea_df,
            top_k_programs=30,
            top_n_per_program=40,
        )
        if not explorer_programs:
            explorer_programs = sizes_for_plot["program"].astype(str).tolist()
        label_programs = list(
            dict.fromkeys(
                explorer_programs
                + sizes_for_plot["program"].astype(str).tolist()
                + gsea_for_plot["program"].astype(str).tolist()
            )
        )
        display_records = [
            {
                "program": p,
                "program_display": _display_program_name(p),
                "program_display_short": _display_program_name_short(p),
                "program_display_axis": _display_program_name_axis(p),
            }
            for p in label_programs
        ]

        top_enriched_view = top_enriched.copy()
        if "program" in top_enriched_view.columns:
            top_enriched_view.insert(
                1,
                "program_display",
                top_enriched_view["program"].astype(str).map(_display_program_name),
            )
        top_context_view = top_context.copy()
        if "program" in top_context_view.columns:
            top_context_view.insert(
                1,
                "program_display",
                top_context_view["program"].astype(str).map(_display_program_name),
            )
        gene_membership_view = gene_membership_top.copy()
        if "program" in gene_membership_view.columns:
            gene_membership_view.insert(
                1,
                "program_display",
                gene_membership_view["program"].astype(str).map(_display_program_name),
            )
        program_summary = self._program_summary(bundle)
        summary_table_view = (
            program_summary.loc[
                :,
                [
                    "program",
                    "program_display",
                    "n_genes",
                    "nes",
                    "fdr",
                    "top_reference_name",
                    "claim_supported",
                ],
            ]
            .rename(
                columns={
                    "program_display": "Program",
                    "n_genes": "Genes",
                    "nes": "NES",
                    "fdr": "FDR",
                    "top_reference_name": "Best curated match",
                    "claim_supported": "Claim status",
                }
            )
            .copy()
        )
        summary_table_view["Claim status"] = np.where(
            summary_table_view["Claim status"].fillna(False).astype(bool),
            "Supported",
            "Needs review",
        )
        summary_table_view["Best curated match"] = (
            summary_table_view["Best curated match"].fillna("No curated match").astype(str)
        )
        for numeric_col in ["Genes", "NES", "FDR"]:
            if numeric_col in summary_table_view.columns:
                summary_table_view[numeric_col] = _coerce_numeric_series(
                    summary_table_view[numeric_col],
                    fill_value=np.nan,
                ).round(3)
        context_driver_view = (
            program_summary.loc[
                :,
                [
                    "program",
                    "program_display",
                    "top_context_gene",
                    "top_context_evidence",
                    "top_context_shift",
                    "top_reference_name",
                ],
            ]
            .rename(
                columns={
                    "program_display": "Program",
                    "top_context_gene": "Context driver gene",
                    "top_context_evidence": "Context evidence",
                    "top_context_shift": "Context shift",
                    "top_reference_name": "Best curated match",
                }
            )
            .copy()
        )
        context_driver_view["Context driver gene"] = (
            context_driver_view["Context driver gene"].fillna("Not available").astype(str)
        )
        context_driver_view["Best curated match"] = (
            context_driver_view["Best curated match"].fillna("No curated match").astype(str)
        )
        for numeric_col in ["Context evidence", "Context shift"]:
            if numeric_col in context_driver_view.columns:
                context_driver_view[numeric_col] = _coerce_numeric_series(
                    context_driver_view[numeric_col],
                    fill_value=np.nan,
                ).round(3)
        core_gene_view = (
            program_summary.loc[
                :,
                [
                    "program",
                    "program_display",
                    "top_membership_gene",
                    "top_membership",
                    "n_genes",
                    "top_reference_name",
                ],
            ]
            .rename(
                columns={
                    "program_display": "Program",
                    "top_membership_gene": "Core gene",
                    "top_membership": "Core-weight score",
                    "n_genes": "Genes",
                    "top_reference_name": "Best curated match",
                }
            )
            .copy()
        )
        core_gene_view["Core gene"] = core_gene_view["Core gene"].fillna("Not available").astype(str)
        core_gene_view["Best curated match"] = (
            core_gene_view["Best curated match"].fillna("No curated match").astype(str)
        )
        for numeric_col in ["Core-weight score", "Genes"]:
            if numeric_col in core_gene_view.columns:
                core_gene_view[numeric_col] = _coerce_numeric_series(
                    core_gene_view[numeric_col],
                    fill_value=np.nan,
                ).round(3)
        multi_pathway_view = _prepare_multi_pathway_rows(
            overlap_df=overlap_df,
            gsea_df=bundle.gsea_df,
            top_k_programs=30,
            top_n_refs=8,
        ).copy()
        multi_pathway_curves = _prepare_multi_pathway_curve_records(
            ranked_genes_df=bundle.ranked_genes_df,
            reference_gene_sets=bundle.reference_gene_sets,
            overlap_df=overlap_df,
            gsea_df=bundle.gsea_df,
            top_k_programs=20,
            top_n_refs=4,
        )
        global_multi_pathway_curves = _prepare_global_gsea_curve_records(
            ranked_genes_df=bundle.ranked_genes_df,
            reference_gene_sets=(
                bundle.curated_panel_gene_sets
                if bundle.curated_panel_gene_sets
                else bundle.reference_gene_sets
            ),
            reference_score_df=bundle.curated_panel_gsea_df,
            top_n_refs=6,
        )
        if not multi_pathway_view.empty:
            multi_pathway_view["reference_display"] = (
                multi_pathway_view["reference_display"].fillna(multi_pathway_view["reference_name"]).astype(str)
            )
            for numeric_col in ["jaccard", "overlap_n", "program_n", "reference_n", "novel_gene_estimate"]:
                multi_pathway_view[numeric_col] = _coerce_numeric_series(
                    multi_pathway_view[numeric_col],
                    fill_value=np.nan,
                ).round(3)
        calibration_view = ranking_calibration.copy()
        if not calibration_view.empty:
            for numeric_col in [
                "disease_priority_score",
                "interpretation_score",
                "best_jaccard",
                "raw_rank",
                "prioritized_rank",
            ]:
                if numeric_col in calibration_view.columns:
                    calibration_view[numeric_col] = _coerce_numeric_series(
                        calibration_view[numeric_col],
                        fill_value=np.nan,
                    ).round(3)
        hero_metrics = self._hero_metrics(bundle, gate_summary)
        narrative_summary = self._narrative_summary(bundle, program_summary, gate_summary)
        story_cards = self._story_cards(program_summary)
        de_sig = int((bundle.de_df["fdr"] <= 0.05).sum()) if "fdr" in bundle.de_df.columns else 0
        n_claim_supported = int(program_summary["claim_supported"].fillna(False).astype(bool).sum())
        signal_alert = self._signal_interpretation_alert(de_sig, n_claim_supported)

        payload = {
            "de": _records(de_for_plot, ["gene", "logfc_a_minus_b", "neglog10p", "fdr", "sig"]),
            "sizes": _records(sizes_for_plot, ["program", "n_genes"]),
            "gsea": _records(gsea_for_plot, ["program", "nes", "fdr", "p_value"]),
            "context": _records(
                context_for_plot,
                [
                    "program",
                    "gene",
                    "base_membership",
                    "context_shift",
                    "signed_significance",
                    "context_evidence",
                ],
            ),
            "context_program": _records(
                context_program_df,
                [
                    "program",
                    "gene",
                    "base_membership",
                    "context_shift",
                    "signed_significance",
                    "context_evidence",
                    "abs_shift",
                    "abs_signed_significance",
                    "abs_context_evidence",
                ],
            ),
            "membership_program": _records(
                membership_program_df,
                [
                    "program",
                    "gene",
                    "base_membership",
                    "context_shift",
                    "signed_significance",
                    "context_evidence",
                    "abs_shift",
                    "abs_signed_significance",
                    "abs_context_evidence",
                ],
            ),
            "program_display": display_records,
            "program_summary": _records(
                program_summary,
                [
                    "program",
                    "program_display",
                    "program_display_short",
                    "n_genes",
                    "nes",
                    "fdr",
                    "p_value",
                    "claim_supported",
                    "top_context_gene",
                    "top_context_evidence",
                    "top_context_evidence_abs",
                    "top_context_shift",
                    "top_membership_gene",
                    "top_membership",
                    "top_reference_name",
                    "top_reference_jaccard",
                ],
            ),
            "hero_metrics": hero_metrics,
            "narrative_summary": narrative_summary,
            "story_cards": story_cards,
            "headline_summary": _records(headline_summary, ["metric", "metric_display", "value"]),
            "gate_summary": _records(
                gate_summary,
                ["gate", "gate_display", "n_evaluable", "n_pass", "pass_rate"],
            ),
            "top_enriched_table": _records(summary_table_view, list(summary_table_view.columns)),
            "top_context_table": _records(context_driver_view, list(context_driver_view.columns)),
            "gene_membership_table": _records(core_gene_view, list(core_gene_view.columns)),
            "heatmap": _records(
                _prepare_overlap_heatmap_long(
                    overlap_df=overlap_df,
                    gsea_df=bundle.gsea_df,
                    top_k=20,
                ),
                ["program", "reference_name", "reference_display", "jaccard"],
            ),
            "multi_pathway": _records(
                multi_pathway_view,
                [
                    "program",
                    "reference_name",
                    "reference_display",
                    "jaccard",
                    "overlap_n",
                    "program_n",
                    "reference_n",
                    "novel_gene_estimate",
                ],
            ),
            "multi_pathway_curves": multi_pathway_curves,
            "global_multi_pathway_curves": global_multi_pathway_curves,
            "reference_source_hits": _records(
                reference_source_hits,
                [
                    "source",
                    "source_display",
                    "program",
                    "reference_name",
                    "reference_display",
                    "priority_band",
                    "disease_priority_score",
                    "interpretation_score",
                    "jaccard",
                    "overlap_n",
                    "program_n",
                    "reference_n",
                    "novel_gene_estimate",
                ],
            ),
            "reference_family_hits": _records(
                reference_family_hits,
                [
                    "family_key",
                    "family_display",
                    "top_program",
                    "top_reference_name",
                    "top_reference_display",
                    "priority_band",
                    "disease_priority_score",
                    "interpretation_score",
                    "best_jaccard",
                    "mean_jaccard",
                    "programs_covered",
                    "references_merged",
                    "source_count",
                    "sources_display",
                ],
            ),
            "reference_ranking_calibration": _records(
                calibration_view,
                [
                    "family_key",
                    "family_display",
                    "priority_band",
                    "disease_priority_score",
                    "interpretation_score",
                    "best_jaccard",
                    "raw_rank",
                    "prioritized_rank",
                    "raw_top",
                    "prioritized_top",
                ],
            ),
        }
        payload_json = json.dumps(payload, ensure_ascii=True)

        hero_metrics_html = "".join(
            (
                f"<button type=\"button\" class=\"metric-card{' metric-card-action' if m.get('program') else ''}\""
                f" data-program=\"{_escape(str(m.get('program') or ''))}\">"
                f"<span class=\"metric-label\">{_escape(str(m['label']))}</span>"
                f"<strong>{_escape(str(m['value']))}</strong>"
                f"<small>{_escape(str(m['detail']))}</small>"
                "</button>"
            )
            for m in hero_metrics
        )
        narrative_html = "".join(
            (
                f"<button type=\"button\" class=\"summary-item{' summary-item-action' if bullet.get('program') else ''}\""
                f" data-program=\"{_escape(str(bullet.get('program') or ''))}\">"
                f"<span class=\"summary-label\">{_escape(str(bullet['label']))}</span>"
                f"<span class=\"summary-text\">{_escape(str(bullet['text']))}</span>"
                f"<span class=\"summary-cta\">{'Focus program' if bullet.get('program') else 'Study-level summary'}</span>"
                "</button>"
            )
            for bullet in narrative_summary
        )
        story_cards_html = "".join(
            (
                f"<button type=\"button\" class=\"story-card{' story-card-action' if c.get('program') else ''}\""
                f" data-program=\"{_escape(str(c.get('program') or ''))}\">"
                f"<span class=\"story-kicker\">{_escape(str(c['eyebrow']))}</span>"
                f"<h3>{_escape(str(c['title']))}</h3>"
                f"<p>{_escape(str(c['detail']))}</p>"
                f"<span class=\"story-cta\">{'Focus program' if c.get('program') else 'Summary'}</span>"
                "</button>"
            )
            for c in story_cards
        )
        guide_html = "".join(
            [
                (
                    "<a class=\"guide-step\" href=\"#program-interpretation\">"
                    "<strong>1. Pick a program</strong>"
                    "<span>Start in Program Explorer and lock onto the lead signal or search by pathway phrase.</span>"
                    "</a>"
                ),
                (
                    "<a class=\"guide-step\" href=\"#reference-evidence\">"
                    "<strong>2. Check the evidence chain</strong>"
                    "<span>Confirm enrichment, curated anchor overlap, and the top context-sensitive genes in the same place.</span>"
                    "</a>"
                ),
                (
                    "<a class=\"guide-step\" href=\"#downloads\">"
                    "<strong>3. Export the table you need</strong>"
                    "<span>Every major table is searchable, sortable, and downloadable from the dashboard itself.</span>"
                    "</a>"
                ),
            ]
        )
        hero_actions_html = "".join(
            [
                "<a class=\"hero-button\" href=\"#program-interpretation\">Open program interpretation</a>",
                "<a class=\"hero-button hero-button-ghost\" href=\"#reference-evidence\">Open reference evidence</a>",
                "<a class=\"hero-button hero-button-ghost\" href=\"#downloads\">Open downloads</a>",
                "<a class=\"hero-button hero-button-ghost\" href=\"#assets\">Open figure vault</a>",
            ]
        )
        study_alert_html = (
            f"<div class=\"alert-note\"><strong>{_escape(signal_alert['title'])}</strong>"
            f"<span>{_escape(signal_alert['text'])}</span></div>"
            if signal_alert is not None
            else ""
        )

        fig_rel = figure_dir.name
        table_rel = table_dir.name
        download_links: list[tuple[str, str]] = [
            ("Headline summary CSV", f"{table_rel}/headline_summary.csv"),
            ("Claim gate summary CSV", f"{table_rel}/claim_gate_summary.csv"),
            ("Top enriched programs CSV", f"{table_rel}/top_enriched_programs.csv"),
            ("Top context evidence CSV", f"{table_rel}/top_context_evidence_genes.csv"),
            ("Top genes per program CSV", f"{table_rel}/top_genes_per_program.csv"),
            ("Top multi-pathway hits CSV", f"{table_rel}/top_multi_pathway_hits.csv"),
            ("Reference source hits CSV", f"{table_rel}/reference_source_hits.csv"),
            ("Reference family hits CSV", f"{table_rel}/reference_family_hits.csv"),
            ("Reference ranking calibration CSV", f"{table_rel}/reference_ranking_calibration.csv"),
            ("Reviewer summary Markdown", "summary.md"),
            ("Figure 1 PNG", f"{fig_rel}/figure_1_volcano.png"),
            ("Figure 2 PNG", f"{fig_rel}/figure_2_program_sizes.png"),
            ("Figure 3 PNG", f"{fig_rel}/figure_3_claim_gates.png"),
            ("Figure 4 PNG", f"{fig_rel}/figure_4_context_shift.png"),
            ("Figure 5 PNG", f"{fig_rel}/figure_5_multi_pathway.png"),
            ("Figure 6 PNG", f"{fig_rel}/figure_6_multi_pathway_enrichment_curves.png"),
            ("Figure 7 PNG", f"{fig_rel}/figure_7_reference_ranking_calibration.png"),
        ]
        if bundle.curated_panel_gsea_df is not None:
            download_links.append(("Canonical curated panel CSV", f"{table_rel}/curated_panel_gsea.csv"))
        if bundle.curated_panel_gene_sets is not None:
            download_links.append(("Canonical curated panel GMT", f"{table_rel}/curated_panel_gene_sets.gmt"))
        if include_pdf:
            download_links.extend(
                [
                    ("Figure 1 PDF", f"{fig_rel}/figure_1_volcano.pdf"),
                    ("Figure 2 PDF", f"{fig_rel}/figure_2_program_sizes.pdf"),
                    ("Figure 3 PDF", f"{fig_rel}/figure_3_claim_gates.pdf"),
                    ("Figure 4 PDF", f"{fig_rel}/figure_4_context_shift.pdf"),
                    ("Figure 5 PDF", f"{fig_rel}/figure_5_multi_pathway.pdf"),
                    ("Figure 6 PDF", f"{fig_rel}/figure_6_multi_pathway_enrichment_curves.pdf"),
                    ("Figure 7 PDF", f"{fig_rel}/figure_7_reference_ranking_calibration.pdf"),
                ]
            )
        download_html = "".join(
            (
                f"<a class=\"download-link\" href=\"{_escape(path)}\" target=\"_blank\" rel=\"noreferrer\">"
                f"<span>{_escape(label)}</span><strong>Open</strong></a>"
            )
            for label, path in download_links
        )
        html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{_escape(title)}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg: #f5efe4;
      --bg-deep: #0d1b2a;
      --card: rgba(255, 251, 245, 0.92);
      --card-strong: #fffdf9;
      --text: #14213d;
      --muted: #5f6c7a;
      --accent: #d1495b;
      --accent-2: #276fbf;
      --accent-3: #2a9d8f;
      --gold: #c08a2d;
      --line: rgba(20, 33, 61, 0.10);
      --shadow: 0 18px 48px rgba(20, 33, 61, 0.10);
      --shadow-soft: 0 10px 22px rgba(20, 33, 61, 0.07);
      --radius: 22px;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      color: var(--text);
      font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at 8% 12%, rgba(39, 111, 191, 0.20) 0%, transparent 34%),
        radial-gradient(circle at 92% 10%, rgba(209, 73, 91, 0.18) 0%, transparent 28%),
        radial-gradient(circle at 60% 100%, rgba(42, 157, 143, 0.13) 0%, transparent 30%),
        linear-gradient(180deg, #fbf5ea 0%, #f5efe4 44%, #f8f2e8 100%);
    }}
    a {{ color: inherit; }}
    .wrap {{
      max-width: 1460px;
      margin: 0 auto;
      padding: 24px 24px 40px;
    }}
    .hero {{
      position: relative;
      overflow: hidden;
      background:
        linear-gradient(145deg, rgba(13, 27, 42, 0.98) 0%, rgba(24, 46, 78, 0.94) 45%, rgba(39, 111, 191, 0.90) 100%);
      color: #f6f7fb;
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 30px;
      padding: 28px 30px 30px;
      box-shadow: 0 24px 60px rgba(8, 18, 29, 0.26);
      margin-bottom: 18px;
      animation: fadeUp 480ms ease-out;
    }}
    .hero::before {{
      content: "";
      position: absolute;
      inset: auto -18% -32% auto;
      width: 420px;
      height: 420px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(255, 194, 77, 0.26) 0%, transparent 62%);
      pointer-events: none;
    }}
    .eyebrow {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      margin: 0 0 12px 0;
      padding: 7px 12px;
      border-radius: 999px;
      border: 1px solid rgba(255, 255, 255, 0.14);
      background: rgba(255, 255, 255, 0.08);
      color: #d9e5fb;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .hero-head {{
      display: grid;
      grid-template-columns: minmax(0, 1.7fr) minmax(260px, 1fr);
      gap: 18px;
      align-items: end;
      margin-bottom: 20px;
      position: relative;
      z-index: 1;
    }}
    .hero h1 {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
      font-size: clamp(2.1rem, 4vw, 3.35rem);
      line-height: 1.02;
      letter-spacing: -0.03em;
    }}
    .hero p {{
      margin: 10px 0 0 0;
      max-width: 780px;
      color: rgba(230, 236, 247, 0.88);
      font-size: 15px;
      line-height: 1.6;
    }}
    .hero-meta {{
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 10px;
      align-self: start;
    }}
    .hero-badge {{
      display: inline-flex;
      align-items: center;
      padding: 10px 12px;
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.10);
      border: 1px solid rgba(255, 255, 255, 0.12);
      color: #eef4ff;
      font-size: 12px;
      font-weight: 600;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
    }}
    .hero-stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      position: relative;
      z-index: 1;
    }}
    .hero-actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 16px;
      position: relative;
      z-index: 1;
    }}
    .hero-button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 42px;
      padding: 10px 14px;
      border-radius: 999px;
      border: 1px solid rgba(255, 255, 255, 0.14);
      background: rgba(255, 255, 255, 0.12);
      color: #f7fbff;
      font-size: 13px;
      font-weight: 700;
      text-decoration: none;
      cursor: pointer;
      transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
    }}
    .hero-button:hover {{
      transform: translateY(-1px);
      border-color: rgba(255, 255, 255, 0.26);
      background: rgba(255, 255, 255, 0.18);
    }}
    .hero-button-ghost {{
      background: rgba(255, 255, 255, 0.05);
    }}
    .hero-note {{
      margin-top: 14px;
      color: rgba(225, 233, 247, 0.82);
      font-size: 12px;
      line-height: 1.6;
      position: relative;
      z-index: 1;
    }}
    .metric-card {{
      padding: 14px 14px 15px;
      border-radius: 18px;
      border: 1px solid rgba(255, 255, 255, 0.10);
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.10), rgba(255, 255, 255, 0.05));
      color: #f9fbff;
      text-align: left;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.07);
    }}
    .metric-card-action {{
      cursor: pointer;
      transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
    }}
    .metric-card-action:hover {{
      transform: translateY(-2px);
      border-color: rgba(255, 255, 255, 0.28);
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.16), rgba(255, 255, 255, 0.08));
    }}
    .metric-label {{
      display: block;
      color: rgba(220, 231, 248, 0.72);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 7px;
    }}
    .metric-card strong {{
      display: block;
      font-size: 1.2rem;
      line-height: 1.1;
      margin-bottom: 4px;
    }}
    .metric-card small {{
      display: block;
      color: rgba(228, 235, 245, 0.78);
      font-size: 12px;
      line-height: 1.45;
    }}
    .section-nav {{
      position: sticky;
      top: 10px;
      z-index: 20;
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      padding: 12px;
      margin-bottom: 18px;
      border-radius: 18px;
      background: rgba(255, 251, 245, 0.74);
      backdrop-filter: blur(16px);
      border: 1px solid rgba(20, 33, 61, 0.08);
      box-shadow: var(--shadow-soft);
    }}
    .nav-chip {{
      padding: 10px 14px;
      border-radius: 999px;
      background: #fffdfa;
      border: 1px solid rgba(20, 33, 61, 0.08);
      color: #20324d;
      text-decoration: none;
      font-size: 13px;
      font-weight: 700;
      transition: transform 160ms ease, background 160ms ease, border-color 160ms ease;
    }}
    .nav-chip:hover {{
      transform: translateY(-1px);
      background: #f2f7ff;
      border-color: rgba(39, 111, 191, 0.24);
    }}
    .nav-actions {{
      margin-left: auto;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .nav-chip-button {{
      cursor: pointer;
      font-weight: 800;
    }}
    .layout-status {{
      align-self: center;
      color: #5f6c7a;
      font-size: 12px;
      font-weight: 700;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(12, minmax(0, 1fr));
      gap: 14px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow-soft);
      padding: 18px;
      overflow: hidden;
      animation: fadeUp 520ms ease-out;
    }}
    .card-strong {{
      background: linear-gradient(180deg, rgba(255, 253, 249, 0.98), rgba(255, 249, 241, 0.95));
      box-shadow: var(--shadow);
    }}
    .card[data-height="s"] .plot,
    .card[data-height="s"] .plot-sm {{
      height: 300px;
    }}
    .card[data-height="l"] .plot,
    .card[data-height="l"] .plot-sm {{
      height: 520px;
    }}
    .card[data-height="xl"] .plot,
    .card[data-height="xl"] .plot-sm {{
      height: 660px;
    }}
    .card[data-height="s"] .table-wrap {{
      max-height: 220px;
    }}
    .card[data-height="l"] .table-wrap {{
      max-height: 420px;
    }}
    .card[data-height="xl"] .table-wrap {{
      max-height: 560px;
    }}
    .span-12 {{ grid-column: span 12; }}
    .span-8 {{ grid-column: span 8; }}
    .span-7 {{ grid-column: span 7; }}
    .span-6 {{ grid-column: span 6; }}
    .span-5 {{ grid-column: span 5; }}
    .span-4 {{ grid-column: span 4; }}
    .span-3 {{ grid-column: span 3; }}
    h2 {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
      font-size: 1.3rem;
      line-height: 1.1;
      color: #13253f;
    }}
    .section-copy {{
      margin: 8px 0 0 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
    }}
    .section-head {{
      display: flex;
      justify-content: space-between;
      gap: 14px;
      margin-bottom: 14px;
      align-items: end;
    }}
    .section-kicker {{
      display: inline-flex;
      margin-bottom: 6px;
      color: var(--accent);
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .table-note {{
      margin: 0 0 10px 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.55;
    }}
    .alert-note {{
      margin: 12px 0 0 0;
      padding: 12px 14px;
      display: flex;
      flex-direction: column;
      gap: 4px;
      border-radius: 16px;
      border: 1px solid rgba(192, 138, 45, 0.24);
      background: linear-gradient(180deg, rgba(255, 248, 233, 0.95), rgba(255, 252, 244, 0.98));
      color: #6f4e12;
      font-size: 12.5px;
      line-height: 1.55;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.65);
    }}
    .alert-note strong {{
      color: #7a3f1d;
    }}
    .panel-tools {{
      display: none;
      align-items: center;
      gap: 8px;
      margin-left: auto;
    }}
    .layout-editing .panel-tools {{
      display: flex;
    }}
    .card-grip {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 36px;
      min-height: 36px;
      border-radius: 12px;
      border: 1px dashed rgba(20, 33, 61, 0.18);
      background: rgba(255, 255, 255, 0.85);
      color: #41566f;
      cursor: grab;
      font-size: 14px;
      font-weight: 800;
    }}
    .card-grip:active {{
      cursor: grabbing;
    }}
    .panel-select {{
      min-height: 36px;
      padding: 7px 10px;
      border-radius: 12px;
      border: 1px solid rgba(20, 33, 61, 0.10);
      background: rgba(255, 255, 255, 0.92);
      color: #13253f;
      font-size: 12px;
      font-weight: 700;
    }}
    .layout-editing .card {{
      outline: 2px dashed rgba(39, 111, 191, 0.18);
      outline-offset: 3px;
    }}
    .card-drop-target {{
      outline: 2px solid rgba(209, 73, 91, 0.35) !important;
      outline-offset: 3px;
    }}
    .plot {{ height: 420px; }}
    .plot-sm {{ height: 360px; }}
    .np-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12.5px;
      background: rgba(255, 255, 255, 0.88);
    }}
    .np-table th, .np-table td {{
      border-bottom: 1px solid rgba(20, 33, 61, 0.07);
      text-align: left;
      padding: 9px 10px;
      vertical-align: top;
      word-break: break-word;
    }}
    .np-table th {{
      background: rgba(242, 247, 255, 0.95);
      color: #1d3658;
      font-weight: 700;
      position: sticky;
      top: 0;
      backdrop-filter: blur(10px);
    }}
    .np-table tbody tr:hover {{
      background: rgba(39, 111, 191, 0.05);
    }}
    .table-wrap {{
      max-height: 320px;
      overflow: auto;
      border: 1px solid rgba(20, 33, 61, 0.08);
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.72);
    }}
    .toolbar {{
      display: grid;
      grid-template-columns: minmax(0, 1.4fr) minmax(260px, 1fr);
      gap: 16px;
      align-items: end;
    }}
    .toolbar-main {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }}
    .field-group {{
      display: flex;
      flex-direction: column;
      gap: 7px;
    }}
    .field-group label {{
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.02em;
      color: #223754;
    }}
    .field-group input,
    .field-group select,
    .toolbar-button {{
      width: 100%;
      min-height: 42px;
      padding: 10px 12px;
      border-radius: 14px;
      border: 1px solid rgba(20, 33, 61, 0.10);
      background: rgba(255, 255, 255, 0.88);
      color: #13253f;
      font-size: 13px;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.65);
    }}
    .toolbar-actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      justify-content: flex-end;
    }}
    .toolbar-button {{
      width: auto;
      cursor: pointer;
      font-weight: 700;
      transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
    }}
    .toolbar-button:hover {{
      transform: translateY(-1px);
      border-color: rgba(39, 111, 191, 0.28);
      background: rgba(242, 247, 255, 0.96);
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 9px 12px;
      border-radius: 999px;
      background: rgba(39, 111, 191, 0.09);
      border: 1px solid rgba(39, 111, 191, 0.16);
      color: #1f4c83;
      font-size: 12px;
      font-weight: 700;
    }}
    .hint {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.6;
      margin: 12px 0 0 0;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-top: 14px;
    }}
    .summary-item {{
      display: flex;
      flex-direction: column;
      gap: 8px;
      min-height: 148px;
      padding: 15px 16px;
      border-radius: 18px;
      border: 1px solid rgba(20, 33, 61, 0.08);
      background: rgba(255, 255, 255, 0.80);
      text-align: left;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.50);
    }}
    .summary-item-action {{
      cursor: pointer;
      transition: transform 160ms ease, border-color 160ms ease, box-shadow 160ms ease;
    }}
    .summary-item-action:hover {{
      transform: translateY(-2px);
      border-color: rgba(39, 111, 191, 0.24);
      box-shadow: 0 14px 28px rgba(20, 33, 61, 0.08);
    }}
    .summary-label {{
      color: var(--accent);
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .summary-text {{
      color: #1c3050;
      font-size: 13px;
      line-height: 1.6;
    }}
    .summary-cta {{
      margin-top: auto;
      color: #1f4c83;
      font-size: 12px;
      font-weight: 700;
    }}
    .guide-list {{
      display: grid;
      gap: 10px;
      margin-bottom: 16px;
    }}
    .guide-step {{
      display: block;
      padding: 13px 14px;
      border-radius: 16px;
      border: 1px solid rgba(20, 33, 61, 0.08);
      background: rgba(255, 255, 255, 0.82);
      text-decoration: none;
      transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
    }}
    .guide-step:hover {{
      transform: translateY(-1px);
      border-color: rgba(39, 111, 191, 0.20);
      background: rgba(245, 249, 255, 0.98);
    }}
    .guide-step strong {{
      display: block;
      margin-bottom: 5px;
      color: #13253f;
      font-size: 13px;
      font-weight: 800;
    }}
    .guide-step span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.55;
    }}
    .story-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 12px;
    }}
    .story-card {{
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      gap: 10px;
      min-height: 170px;
      padding: 16px;
      border-radius: 18px;
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.94), rgba(248, 251, 255, 0.88));
      border: 1px solid rgba(20, 33, 61, 0.08);
      text-align: left;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.55);
    }}
    .story-card-action {{
      cursor: pointer;
      transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
    }}
    .story-card-action:hover {{
      transform: translateY(-3px);
      border-color: rgba(209, 73, 91, 0.22);
      box-shadow: 0 16px 32px rgba(20, 33, 61, 0.10);
    }}
    .story-kicker {{
      color: var(--accent);
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .story-card h3 {{
      margin: 0;
      font-size: 1rem;
      line-height: 1.25;
      color: #12243d;
    }}
    .story-card p {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
    }}
    .story-cta {{
      color: #1f4c83;
      font-size: 12px;
      font-weight: 700;
    }}
    .download-list {{
      display: grid;
      gap: 10px;
    }}
    .download-link {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 12px 14px;
      border-radius: 16px;
      border: 1px solid rgba(20, 33, 61, 0.08);
      background: rgba(255, 255, 255, 0.82);
      text-decoration: none;
      transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
    }}
    .download-link:hover {{
      transform: translateY(-1px);
      background: rgba(245, 249, 255, 0.98);
      border-color: rgba(39, 111, 191, 0.18);
    }}
    .download-link span {{
      font-size: 13px;
      font-weight: 600;
      color: #223754;
    }}
    .download-link strong {{
      color: #1f4c83;
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .spotlight {{
      display: flex;
      flex-direction: column;
      gap: 12px;
      min-height: 240px;
    }}
    .spotlight-head {{
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .spotlight-title {{
      font-size: 1.05rem;
      font-weight: 800;
      line-height: 1.35;
      color: #13253f;
    }}
    .spotlight-badges {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .mini-pill {{
      display: inline-flex;
      align-items: center;
      padding: 7px 10px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 700;
      border: 1px solid rgba(20, 33, 61, 0.08);
      background: rgba(255, 255, 255, 0.84);
      color: #243a5b;
    }}
    .mini-pill.pass {{
      background: rgba(42, 157, 143, 0.10);
      border-color: rgba(42, 157, 143, 0.22);
      color: #1a675f;
    }}
    .mini-pill.warn {{
      background: rgba(209, 73, 91, 0.10);
      border-color: rgba(209, 73, 91, 0.20);
      color: #9a3745;
    }}
    .spotlight-metrics {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .spot-stat {{
      padding: 12px;
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.78);
      border: 1px solid rgba(20, 33, 61, 0.07);
    }}
    .spot-stat span {{
      display: block;
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      margin-bottom: 5px;
    }}
    .spot-stat strong {{
      display: block;
      color: #12243d;
      font-size: 1rem;
      line-height: 1.2;
    }}
    .spotlight-list {{
      display: grid;
      gap: 8px;
    }}
    .spotlight-row {{
      padding: 12px 13px;
      border-radius: 14px;
      background: rgba(248, 251, 255, 0.74);
      border: 1px solid rgba(20, 33, 61, 0.07);
    }}
    .spotlight-row strong {{
      display: block;
      font-size: 13px;
      color: #13253f;
      margin-bottom: 4px;
    }}
    .spotlight-row span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }}
    .fig-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .split-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.3fr) minmax(320px, 0.9fr);
      gap: 14px;
      align-items: start;
    }}
    .fig-grid img {{
      width: 100%;
      border: 1px solid rgba(20, 33, 61, 0.08);
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.88);
      box-shadow: var(--shadow-soft);
    }}
    .table-shell {{
      display: grid;
      gap: 10px;
    }}
    .table-toolbar {{
      display: grid;
      gap: 10px;
    }}
    .table-toolbar-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.4fr) repeat(3, minmax(120px, 0.42fr));
      gap: 10px;
    }}
    .table-status {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }}
    .table-button {{
      width: auto;
      min-height: 40px;
      padding: 10px 12px;
      border-radius: 14px;
      border: 1px solid rgba(20, 33, 61, 0.10);
      background: rgba(255, 255, 255, 0.88);
      color: #13253f;
      font-size: 12px;
      font-weight: 800;
      cursor: pointer;
      transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
    }}
    .table-button:hover {{
      transform: translateY(-1px);
      border-color: rgba(39, 111, 191, 0.24);
      background: rgba(242, 247, 255, 0.96);
    }}
    .table-row-action {{
      cursor: pointer;
    }}
    .table-row-action td:first-child {{
      font-weight: 700;
      color: #1d3658;
    }}
    .table-empty {{
      padding: 14px;
      color: #5a6b7f;
      font-size: 13px;
      line-height: 1.55;
    }}
    .tab-nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 14px;
    }}
    .tab-button {{
      appearance: none;
      border: 1px solid rgba(20, 33, 61, 0.10);
      background: rgba(255, 255, 255, 0.80);
      color: #1d3658;
      border-radius: 14px;
      min-height: 40px;
      padding: 10px 14px;
      font-size: 12px;
      font-weight: 800;
      cursor: pointer;
      transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
    }}
    .tab-button:hover {{
      transform: translateY(-1px);
      border-color: rgba(39, 111, 191, 0.24);
    }}
    .tab-button.active {{
      background: rgba(39, 111, 191, 0.12);
      border-color: rgba(39, 111, 191, 0.30);
      color: #123a66;
    }}
    .tab-panel {{
      display: none;
    }}
    .tab-panel.active {{
      display: block;
    }}
    @keyframes fadeUp {{
      from {{
        opacity: 0;
        transform: translateY(12px);
      }}
      to {{
        opacity: 1;
        transform: translateY(0);
      }}
    }}
    @media (max-width: 1180px) {{
      .hero-head,
      .toolbar,
      .toolbar-main {{
        grid-template-columns: 1fr;
      }}
      .hero-meta,
      .toolbar-actions,
      .nav-actions {{
        justify-content: flex-start;
      }}
      .span-7, .span-6, .span-5, .span-4, .span-3 {{ grid-column: span 12; }}
      .plot, .plot-sm {{ height: 360px; }}
      .split-grid {{
        grid-template-columns: 1fr;
      }}
    }}
    @media (max-width: 760px) {{
      .wrap {{ padding: 14px 14px 28px; }}
      .hero {{ padding: 20px; border-radius: 24px; }}
      .section-nav {{ top: 0; }}
      .hero-stats,
      .story-grid,
      .summary-grid,
      .spotlight-metrics,
      .fig-grid {{
        grid-template-columns: 1fr;
      }}
      .table-toolbar-grid {{
        grid-template-columns: 1fr;
      }}
      .plot, .plot-sm {{ height: 320px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero" id="top">
      <div class="hero-head">
        <div>
          <div class="eyebrow">Evidence-Linked Program Review</div>
          <h1>{_escape(title)}</h1>
          <p>Static pathway libraries are the reference layer, not the endpoint. This dashboard is built as a decision cockpit for dynamic gene programs: discover the signal, ground it against curated biology, inspect the evidence package, and decide which programs are ready to defend.</p>
        </div>
        <div class="hero-meta">
          <span class="hero-badge">Decision cockpit</span>
          <span class="hero-badge">Claim-gated outputs</span>
          <span class="hero-badge">Biological grounding</span>
          <span class="hero-badge">Program-to-pathway matching</span>
          <span class="hero-badge">Publication dossier</span>
        </div>
      </div>
      <div class="hero-stats">{hero_metrics_html}</div>
      <div class="hero-actions">{hero_actions_html}</div>
      <p class="hero-note">Open this file directly in your browser. No server is required, and every major table on the page can be searched, sorted, and exported.</p>
    </section>

    <nav class="section-nav">
      <a class="nav-chip" href="#study-summary">Study Summary</a>
      <a class="nav-chip" href="#program-interpretation">Program Interpretation</a>
      <a class="nav-chip" href="#reference-evidence">Reference Evidence</a>
      <a class="nav-chip" href="#downloads">Downloads / Exports</a>
      <div class="nav-actions">
        <button id="layout-toggle" type="button" class="nav-chip nav-chip-button">Customize layout</button>
        <button id="layout-reset" type="button" class="nav-chip nav-chip-button">Reset layout</button>
        <span id="layout-status" class="layout-status">Default layout</span>
      </div>
    </nav>

    <section class="grid dashboard-grid" id="dashboard-grid">
      <article class="card card-strong span-12" id="study-summary" data-card-id="summary-hub" data-height="l" data-tab-card="summary-hub">
        <div class="section-head">
          <div>
            <span class="section-kicker">Study Summary</span>
            <h2>Study Summary Hub</h2>
            <p class="section-copy">Open the reviewer summary first, then move into program-level interpretation and reference evidence. The goal is to make the biological claim and its supporting files visible in one pass.</p>
          </div>
        </div>
        <div class="tab-nav">
          <button type="button" class="tab-button active" data-tab-target="tab-how-to-read">How to Read This Page</button>
          <button type="button" class="tab-button" data-tab-target="tab-reviewer-checklist">Reviewer Checklist</button>
          <button type="button" class="tab-button" data-tab-target="tab-start-downloads">Start Here + Downloads</button>
          <button type="button" class="tab-button" data-tab-target="tab-headline-metrics">Headline Metrics</button>
          <button type="button" class="tab-button" data-tab-target="tab-claim-gates">Claim Gates</button>
        </div>
        <div class="tab-panel active" data-tab-panel="tab-how-to-read">
          <div class="section-head">
            <div>
              <h2>How to Read This Page</h2>
              <p class="section-copy">Use this package as an interpretation layer on top of dynamic program discovery rather than as a replacement for discovery itself.</p>
            </div>
          </div>
          {study_alert_html}
          <div class="guide-list">
            <div class="guide-step">
              <strong>Discovered program</strong>
              <span>A program is a data-derived gene module recovered from the expression matrix, not a predefined pathway taken from MSigDB or another database.</span>
            </div>
            <div class="guide-step">
              <strong>Curated pathway grounding</strong>
              <span>Curated references are used as the grounding layer. They tell you which known biology is most compatible with the discovered program.</span>
            </div>
            <div class="guide-step">
              <strong>Leading-edge genes</strong>
              <span>Leading-edge genes are the subset of hits driving a pathway curve near its enrichment peak. They help explain why one pathway peaks earlier or more sharply than another.</span>
            </div>
            <div class="guide-step">
              <strong>Interpretation ranking</strong>
              <span>Disease-prioritized family ranking is a downstream heuristic for readability. It does not change enrichment statistics, discovery assignments, or differential expression estimates.</span>
            </div>
          </div>
        </div>
        <div class="tab-panel" data-tab-panel="tab-reviewer-checklist">
          <div class="section-head">
            <div>
              <h2>Reviewer Checklist</h2>
              <p class="section-copy">These cards surface the first programs a reviewer will inspect: the strongest enrichment signal, the clearest curated anchor, the most defensible frontier program, and the sharpest context shift.</p>
            </div>
          </div>
          <div class="story-grid">{story_cards_html}</div>
          <div class="summary-grid">{narrative_html}</div>
        </div>
        <div class="tab-panel" data-tab-panel="tab-start-downloads">
          <div class="section-head">
            <div>
              <h2>Start Here</h2>
              <p class="section-copy">For a first pass, move in this order: lock onto a program, inspect its evidence chain, then export the exact table or PDF you need.</p>
            </div>
          </div>
          <div class="guide-list">{guide_html}</div>
          <div class="section-head">
            <div>
              <h2>Download Center</h2>
              <p class="section-copy">Direct access to the summary tables, calibration exports, and figure files used for manuscript assembly.</p>
            </div>
          </div>
          <div class="download-list">{download_html}</div>
        </div>
        <div class="tab-panel" data-tab-panel="tab-headline-metrics">
          <div class="section-head">
            <div>
              <h2>Headline Metrics</h2>
              <p class="section-copy">Study-level counts exported as the canonical headline summary. Search or sort directly in the browser.</p>
            </div>
          </div>
          <div id="headline-summary-table"></div>
        </div>
        <div class="tab-panel" data-tab-panel="tab-claim-gates">
          <div class="section-head">
            <div>
              <h2>Claim Gate Summary</h2>
              <p class="section-copy">Gate-level pass rates for manuscript-safe interpretation. Sort by pass fraction to spot weak gates immediately.</p>
            </div>
          </div>
          <div id="gate-summary-table"></div>
        </div>
      </article>

      <article class="card card-strong span-12" id="program-interpretation" data-card-id="explorer" data-height="m">
        <div class="section-head">
          <div>
            <span class="section-kicker">Program-Level Interpretation</span>
            <h2>Program Explorer</h2>
            <p class="section-copy">Search and lock onto a program, then let every plot and table follow it. Use this view to decide whether a program is anchored, frontier, or still too weak to support a claim.</p>
          </div>
        </div>
        <div class="toolbar">
          <div class="toolbar-main">
            <div class="field-group">
              <label for="program-search">Program search</label>
              <input id="program-search" type="search" placeholder="Search by program ID or pathway phrase" />
            </div>
            <div class="field-group">
              <label for="program-selector">Program</label>
              <select id="program-selector"></select>
            </div>
            <div class="field-group">
              <label for="context-metric-selector">Context metric</label>
              <select id="context-metric-selector">
                <option value="context_evidence">Context evidence (Recommended)</option>
                <option value="signed_significance">Signed significance</option>
                <option value="context_shift">Context shift (bounded)</option>
              </select>
            </div>
          </div>
          <div class="toolbar-actions">
            <button id="jump-best-program" type="button" class="toolbar-button">Lead program</button>
            <button id="jump-largest-program" type="button" class="toolbar-button">Largest program</button>
            <button id="jump-context-program" type="button" class="toolbar-button">Strongest context</button>
            <span id="program-pill" class="pill">Select a program</span>
          </div>
        </div>
        <p class="hint">Click a bar, point, or heatmap tile to sync all program views. Search narrows the selector live. The spotlight card below exposes the selected program's evidence chain. Use Customize layout if you want to rearrange or resize panels.</p>
      </article>

      <article class="card span-4" data-card-id="spotlight" data-height="m">
        <div class="section-head">
          <div>
            <h2>Program Spotlight</h2>
            <p class="section-copy">Compact per-program evidence summary with claim support, context driver, and best curated anchor.</p>
          </div>
        </div>
        <div id="program-spotlight" class="spotlight"></div>
      </article>
      <article class="card span-6" id="atlas" data-card-id="volcano" data-height="m">
        <div class="section-head">
          <div>
            <h2>Differential Signal Volcano</h2>
            <p class="section-copy">Global gene-level effect and significance landscape.</p>
          </div>
        </div>
        <div id="plot-volcano" class="plot-sm"></div>
      </article>
      <article class="card span-6" data-card-id="program-sizes" data-height="m">
        <div class="section-head">
          <div>
            <h2>Program Size Landscape</h2>
            <p class="section-copy">Top programs by size. Longer bars mean more genes were assigned to that program; use the Interactive Tables for the full list.</p>
          </div>
        </div>
        <div id="plot-sizes" class="plot-sm"></div>
      </article>

      <article class="card span-7" data-card-id="gsea" data-height="m">
        <div class="section-head">
          <div>
            <h2>Enriched Programs</h2>
            <p class="section-copy">Top enriched programs by significance. Use the Interactive Tables for the full dossier and claim status across all programs.</p>
          </div>
        </div>
        <div id="plot-gsea" class="plot"></div>
      </article>
      <article class="card span-5" id="evidence" data-card-id="overlap-heatmap" data-height="m">
        <div class="section-head">
          <div>
            <h2>Program vs Reference Overlap</h2>
            <p class="section-copy">Curated anchor map for the top dynamic programs.</p>
          </div>
        </div>
        <div id="plot-heatmap" class="plot"></div>
      </article>

      <article class="card span-12" data-card-id="multi-pathway" data-height="m">
        <div class="section-head">
          <div>
            <h2 id="multi-pathway-title">Selected Program: Multi-Pathway Enrichment Curves</h2>
            <p id="multi-pathway-copy" class="section-copy">Classic GSEA-style running enrichment curves for multiple curated pathways mapped onto the selected dynamic program. Use this to compare which references peak earliest, strongest, and most consistently along the ranked gene list.</p>
          </div>
        </div>
        <div class="split-grid">
          <div id="plot-multi-pathway" class="plot-sm"></div>
          <div id="multi-pathway-table"></div>
        </div>
      </article>

      <article class="card span-12" id="tables" data-card-id="table-hub" data-height="l" data-tab-card="table-hub">
        <div class="section-head">
          <div>
            <h2>Interactive Tables</h2>
            <p class="section-copy">Dense table views are grouped here to keep the dashboard readable. Switch tabs instead of scanning three side-by-side tables with long labels.</p>
          </div>
        </div>
        <div class="tab-nav">
          <button type="button" class="tab-button active" data-tab-target="tab-program-summary">Program Summary</button>
          <button type="button" class="tab-button" data-tab-target="tab-context-driver">Context Drivers</button>
          <button type="button" class="tab-button" data-tab-target="tab-core-gene">Core Genes</button>
        </div>
        <div class="tab-panel active" data-tab-panel="tab-program-summary">
          <p class="table-note">One row equals one discovered program. Use this as the compact claim dossier before drilling into gene-level detail.</p>
          <div id="top-enriched-table"></div>
        </div>
        <div class="tab-panel" data-tab-panel="tab-context-driver">
          <p class="table-note">This table highlights the single strongest context-sensitive gene per program. It is for interpretation, not a replacement differential-expression test.</p>
          <div id="top-context-table"></div>
        </div>
        <div class="tab-panel" data-tab-panel="tab-core-gene">
          <p class="table-note">These genes are most central to each program by program membership. High core-weight does not automatically mean high effect size.</p>
          <div id="gene-membership-table"></div>
        </div>
      </article>

      <article class="card span-12" id="reference-evidence" data-card-id="reference-layers" data-height="m">
        <div class="section-head">
          <div>
            <span class="section-kicker">Reference Evidence</span>
            <h2>Reference Layers</h2>
            <p class="section-copy">Keep knowledge layers separate first, then collapse near-duplicate labels across sources. Reactome, WikiPathways, Pathway Commons, GO, Hallmark, and KEGG can all support the same program, but they should not be interpreted as the same kind of evidence.</p>
          </div>
        </div>
        <div id="reference-source-tabs" class="tab-nav"></div>
        <div id="reference-source-summary" class="summary-grid"></div>
        <div id="reference-source-table"></div>
        <div class="section-head" style="margin-top:16px;">
          <div>
            <h2>Collapsed Reference Families</h2>
            <p class="section-copy">This table merges near-duplicate labels across sources with a conservative name-based collapse. Families are ranked by interpretation score first and overlap strength second.</p>
          </div>
        </div>
        <div id="reference-family-table"></div>
        <div class="section-head" style="margin-top:16px;">
          <div>
            <h2>Reference Ranking Calibration</h2>
            <p class="section-copy">This calibration view compares raw Jaccard ranking with interpretation-prioritized ranking so reviewers can see exactly which family labels moved and why.</p>
          </div>
        </div>
        <div id="reference-calibration-table"></div>
      </article>

      <article class="card span-7" data-card-id="context-plot" data-height="m">
        <div class="section-head">
          <div>
            <h2 id="context-title">Context evidence by gene</h2>
            <p id="context-copy" class="section-copy">Top 10 genes ranked by how strongly they help explain the case-control difference inside the selected program.</p>
          </div>
        </div>
        <div id="plot-context" class="plot"></div>
      </article>
      <article class="card span-5" data-card-id="context-table" data-height="m">
        <div class="section-head">
          <div>
            <h2 id="context-table-title">Top Context-Evidence Genes</h2>
            <p id="context-table-copy" class="section-copy">Use this to see the top 10 genes driving the selected program's biological interpretation.</p>
          </div>
        </div>
        <div id="context-program-table" class="table-wrap"></div>
      </article>

      <article class="card span-7" data-card-id="membership-plot" data-height="m">
        <div class="section-head">
          <div>
            <h2 id="membership-title">Core genes in selected program</h2>
            <p id="membership-copy" class="section-copy">Top 10 genes ranked by how strongly they belong to the selected program. This is a within-program centrality score, not an effect-size estimate.</p>
          </div>
        </div>
        <div id="plot-membership" class="plot"></div>
      </article>
      <article class="card span-5" data-card-id="membership-table" data-height="m">
        <div class="section-head">
          <div>
            <h2 id="membership-table-title">Core genes for selected program</h2>
            <p id="membership-table-copy" class="section-copy">These are the top 10 genes most central to the selected program, with context columns shown alongside for quick interpretation.</p>
          </div>
        </div>
        <div id="membership-program-table" class="table-wrap"></div>
      </article>

      <article class="card span-12" id="downloads" data-card-id="assets" data-height="m">
        <div class="section-head">
          <div>
            <span class="section-kicker">Downloads / Exports</span>
            <h2>Figure Vault</h2>
            <p class="section-copy">Static figure exports kept in the same package as the interactive dashboard.</p>
          </div>
        </div>
        <div class="fig-grid">
          <img alt="Figure 1 Volcano" src="{fig_rel}/figure_1_volcano.png" />
          <img alt="Figure 2 Program Sizes" src="{fig_rel}/figure_2_program_sizes.png" />
          <img alt="Figure 3 Claim Gates" src="{fig_rel}/figure_3_claim_gates.png" />
          <img alt="Figure 4 Context Evidence" src="{fig_rel}/figure_4_context_shift.png" />
          <img alt="Figure 5 Multi-Pathway Overlap" src="{fig_rel}/figure_5_multi_pathway.png" />
          <img alt="Figure 6 Multi-Pathway Enrichment Curves" src="{fig_rel}/figure_6_multi_pathway_enrichment_curves.png" />
          <img alt="Figure 7 Reference Ranking Calibration" src="{fig_rel}/figure_7_reference_ranking_calibration.png" />
        </div>
      </article>
    </section>
  </div>

  <script>
    const payload = {payload_json};

    const programDisplayRows = payload.program_display || [];
    const programSummary = payload.program_summary || [];
    const programSummaryMap = new Map(programSummary.map(d => [d.program, d]));
    const displayMapShort = new Map(
      programDisplayRows.map(d => [d.program, d.program_display_short || d.program_display || d.program])
    );
    const displayMapFull = new Map(
      programDisplayRows.map(d => [d.program, d.program_display || d.program_display_short || d.program])
    );
    const displayMapAxis = new Map(
      programDisplayRows.map(d => [d.program, d.program_display_axis || d.program_display_short || d.program_display || d.program])
    );
    const labelFor = (p) => displayMapShort.get(p) || p;
    const fullLabelFor = (p) => displayMapFull.get(p) || labelFor(p);
    const axisLabelFor = (p) => displayMapAxis.get(p) || labelFor(p);
    const wrapLabel = (s, width = 32) => {{
      const txt = String(s || "");
      if (txt.length <= width) return txt;
      if (txt.includes(" ")) {{
        const words = txt.split(/\\s+/);
        const lines = [];
        let cur = "";
        for (const w of words) {{
          const nxt = (cur + " " + w).trim();
          if (nxt.length <= width) cur = nxt;
          else {{ if (cur) lines.push(cur); cur = w; }}
        }}
        if (cur) lines.push(cur);
        return lines.join("<br>");
      }}
      const out = [];
      for (let i = 0; i < txt.length; i += width) out.push(txt.slice(i, i + width));
      return out.join("<br>");
    }};
    const rowPlotHeight = (nRows, minPx = 360, stepPx = 28, padPx = 120, maxPx = 920) => {{
      const n = Math.max(0, Number(nRows) || 0);
      return Math.min(maxPx, Math.max(minPx, padPx + n * stepPx));
    }};
    const panelHeightScale = {{s: 0.82, m: 1.0, l: 1.32, xl: 1.62}};
    const panelSpanClasses = ["span-3", "span-4", "span-5", "span-6", "span-7", "span-8", "span-12"];
    const esc = (x) => String(x ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    const plotConfig = {{
      displaylogo: false,
      responsive: true,
      modeBarButtonsToRemove: ["lasso2d", "select2d"],
    }};

    const programUniverse = [
      ...new Set([
        ...programDisplayRows.map(d => d.program),
        ...programSummary.map(d => d.program),
        ...(payload.gsea || []).map(d => d.program),
        ...(payload.sizes || []).map(d => d.program),
        ...(payload.heatmap || []).map(d => d.program),
      ]),
    ];
    let selectedProgram = programUniverse.length > 0 ? programUniverse[0] : null;
    let selectedContextMetric = "context_evidence";
    let selectedReferenceSource = null;
    let layoutEditing = false;
    let layoutCustomized = false;
    let layoutRerenderTimer = null;
    const layoutStorageKey = "npathway_dashboard_layout_v8";
    const defaultDashboardLayout = [
      {{id: "summary-hub", width: "12", height: "l"}},
      {{id: "explorer", width: "12", height: "m"}},
      {{id: "gsea", width: "8", height: "l"}},
      {{id: "spotlight", width: "4", height: "m"}},
      {{id: "multi-pathway", width: "12", height: "l"}},
      {{id: "reference-layers", width: "12", height: "m"}},
      {{id: "table-hub", width: "12", height: "l"}},
      {{id: "context-plot", width: "7", height: "l"}},
      {{id: "context-table", width: "5", height: "l"}},
      {{id: "membership-plot", width: "7", height: "l"}},
      {{id: "membership-table", width: "5", height: "l"}},
      {{id: "overlap-heatmap", width: "6", height: "l"}},
      {{id: "volcano", width: "6", height: "m"}},
      {{id: "program-sizes", width: "6", height: "l"}},
      {{id: "assets", width: "12", height: "m"}},
    ];
    const contextMetricLabels = {{
      context_evidence: "Context evidence",
      signed_significance: "Signed significance",
      context_shift: "Context shift (bounded)",
    }};
    const contextMetricAxis = {{
      context_evidence: "Context evidence = context_shift x -log10(p)",
      signed_significance: "Signed significance = sign(logFC) x -log10(p)",
      context_shift: "Context shift (bounded)",
    }};
    const contextMetricAbsCol = {{
      context_evidence: "abs_context_evidence",
      signed_significance: "abs_signed_significance",
      context_shift: "abs_shift",
    }};

    function numOrZero(x) {{
      const n = Number(x);
      return Number.isFinite(n) ? n : 0;
    }}

    function finiteOrNull(x) {{
      const n = Number(x);
      return Number.isFinite(n) ? n : null;
    }}

    function fmtFixed(x, digits = 2) {{
      const n = finiteOrNull(x);
      return n === null ? "NA" : n.toFixed(digits);
    }}

    function fmtExp(x) {{
      const n = finiteOrNull(x);
      return n === null ? "NA" : n.toExponential(2);
    }}

    function boolLike(x) {{
      return x === true || String(x).toLowerCase() === "true";
    }}

    function metricValue(row, metric) {{
      if (metric === "context_evidence") return numOrZero(row.context_evidence ?? row.context_shift);
      if (metric === "signed_significance") return numOrZero(row.signed_significance ?? row.context_shift);
      return numOrZero(row.context_shift);
    }}

    function metricAbsValue(row, metric) {{
      const absCol = contextMetricAbsCol[metric];
      if (absCol && row[absCol] !== undefined && row[absCol] !== null) return numOrZero(row[absCol]);
      return Math.abs(metricValue(row, metric));
    }}

    function panelHeightPreset(targetId) {{
      const el = document.getElementById(targetId);
      const card = el ? el.closest(".card") : null;
      return card?.dataset?.height || "m";
    }}

    function scaledPanelHeight(targetId, basePx) {{
      const scale = panelHeightScale[panelHeightPreset(targetId)] || 1.0;
      return Math.round(basePx * scale);
    }}

    function responsiveRowHeight(targetId, nRows, minPx = 360, stepPx = 28, padPx = 120, maxPx = 920) {{
      const scale = panelHeightScale[panelHeightPreset(targetId)] || 1.0;
      return rowPlotHeight(
        nRows,
        Math.round(minPx * scale),
        Math.round(stepPx * scale),
        Math.round(padPx * scale),
        Math.round(maxPx * scale),
      );
    }}

    function csvEscape(value) {{
      const text = String(value ?? "");
      if (/[",\\n]/.test(text)) return `"${{text.replace(/"/g, '""')}}"`;
      return text;
    }}

    function downloadRowsCsv(filename, rows, cols) {{
      const safeRows = Array.isArray(rows) ? rows : [];
      const safeCols = Array.isArray(cols) ? cols : [];
      const lines = [
        safeCols.map(csvEscape).join(","),
        ...safeRows.map((row) => safeCols.map((col) => csvEscape(row[col])).join(",")),
      ];
      const blob = new Blob([lines.join("\\n")], {{ type: "text/csv;charset=utf-8;" }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename || "npathway_table.csv";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    }}

    function normalizeSortValue(value) {{
      const numeric = Number(value);
      if (Number.isFinite(numeric) && String(value ?? "").trim() !== "") return numeric;
      return String(value ?? "").toLowerCase();
    }}

    function renderInteractiveTable(targetId, rows, cols, options = {{}}) {{
      const target = document.getElementById(targetId);
      if (!target) return;
      const safeRows = Array.isArray(rows) ? rows.slice() : [];
      const visibleCols = Array.isArray(cols) && cols.length > 0
        ? cols.slice()
        : (safeRows[0] ? Object.keys(safeRows[0]) : []);
      const defaultLimit = Number(options.limit);
      const limitChoices = Array.isArray(options.limitChoices) && options.limitChoices.length > 0
        ? options.limitChoices.map((v) => Number(v)).filter((v) => Number.isFinite(v) && v > 0)
        : [10, 20, 50, 100];
      const state = {{
        query: String(options.query || ""),
        sortBy: visibleCols.includes(options.sortBy) ? options.sortBy : (visibleCols[0] || ""),
        direction: options.direction === "asc" ? "asc" : "desc",
        limit: Number.isFinite(defaultLimit) && defaultLimit > 0
          ? defaultLimit
          : (limitChoices[0] || Math.max(safeRows.length, 1)),
      }};

      function filteredRows() {{
        const q = state.query.trim().toLowerCase();
        let working = safeRows.slice();
        if (q) {{
          working = working.filter((row) =>
            visibleCols.some((col) => String(row[col] ?? "").toLowerCase().includes(q))
          );
        }}
        if (state.sortBy) {{
          working.sort((a, b) => {{
            const av = normalizeSortValue(a[state.sortBy]);
            const bv = normalizeSortValue(b[state.sortBy]);
            if (av < bv) return state.direction === "asc" ? -1 : 1;
            if (av > bv) return state.direction === "asc" ? 1 : -1;
            return 0;
          }});
        }}
        return working;
      }}

      function render() {{
        const filtered = filteredRows();
        const shown = filtered.slice(0, state.limit);
        const toolbar = `
          <div class="table-toolbar">
            <div class="table-toolbar-grid">
              <div class="field-group">
                <label>Search rows</label>
                <input type="search" data-role="search" value="${{esc(state.query)}}" placeholder="Filter this table" />
              </div>
              <div class="field-group">
                <label>Sort by</label>
                <select data-role="sort-by">
                  ${{visibleCols.map((col) => `<option value="${{esc(col)}}" ${{col === state.sortBy ? "selected" : ""}}>${{esc(col)}}</option>`).join("")}}
                </select>
              </div>
              <div class="field-group">
                <label>Direction</label>
                <select data-role="direction">
                  <option value="desc" ${{state.direction === "desc" ? "selected" : ""}}>Descending</option>
                  <option value="asc" ${{state.direction === "asc" ? "selected" : ""}}>Ascending</option>
                </select>
              </div>
              <div class="field-group">
                <label>Rows shown</label>
                <select data-role="limit">
                  ${{limitChoices.map((value) => `<option value="${{value}}" ${{Number(value) === Number(state.limit) ? "selected" : ""}}>${{value}}</option>`).join("")}}
                </select>
              </div>
            </div>
            <div class="toolbar-actions">
              <span class="table-status">Showing ${{shown.length}} of ${{filtered.length}} rows</span>
              <button type="button" class="table-button" data-role="download">Download CSV</button>
            </div>
          </div>
        `;
        if (shown.length === 0) {{
          target.innerHTML = `<div class="table-shell">${{toolbar}}<div class="table-empty">No rows match the current filter.</div></div>`;
        }} else {{
          const head = "<thead><tr>" + visibleCols.map((col) => `<th>${{esc(col)}}</th>`).join("") + "</tr></thead>";
          const body = "<tbody>" + shown.map((row) => {{
            const clickableProgram = options.programKey ? String(row[options.programKey] ?? "") : "";
            const rowClass = clickableProgram ? "table-row-action" : "";
            const rowAttr = clickableProgram ? ` data-program="${{esc(clickableProgram)}}"` : "";
            return `<tr class="${{rowClass}}"${{rowAttr}}>` + visibleCols.map((col) => `<td>${{esc(row[col])}}</td>`).join("") + `</tr>`;
          }}).join("") + "</tbody>";
          target.innerHTML = `<div class="table-shell">${{toolbar}}<div class="table-wrap"><table class="np-table">${{head}}${{body}}</table></div></div>`;
        }}

        const searchEl = target.querySelector('[data-role="search"]');
        if (searchEl) {{
          searchEl.addEventListener("input", (ev) => {{
            state.query = ev.target.value || "";
            render();
          }});
        }}
        const sortEl = target.querySelector('[data-role="sort-by"]');
        if (sortEl) {{
          sortEl.addEventListener("change", (ev) => {{
            state.sortBy = ev.target.value || visibleCols[0] || "";
            render();
          }});
        }}
        const directionEl = target.querySelector('[data-role="direction"]');
        if (directionEl) {{
          directionEl.addEventListener("change", (ev) => {{
            state.direction = ev.target.value === "asc" ? "asc" : "desc";
            render();
          }});
        }}
        const limitEl = target.querySelector('[data-role="limit"]');
        if (limitEl) {{
          limitEl.addEventListener("change", (ev) => {{
            state.limit = Number(ev.target.value) || (limitChoices[0] || 10);
            render();
          }});
        }}
        const downloadEl = target.querySelector('[data-role="download"]');
        if (downloadEl) {{
          downloadEl.addEventListener("click", () => {{
            downloadRowsCsv(options.csvName || `${{targetId}}.csv`, filtered, visibleCols);
          }});
        }}
        if (options.programKey) {{
          target.querySelectorAll("tr[data-program]").forEach((rowEl) => {{
            rowEl.addEventListener("click", () => {{
              const program = rowEl.getAttribute("data-program");
              if (program) setProgram(program);
            }});
          }});
        }}
      }}

      render();
    }}

    function currentCardWidth(card) {{
      for (const cls of panelSpanClasses) {{
        if (card.classList.contains(cls)) return cls.replace("span-", "");
      }}
      return card.dataset.width || "6";
    }}

    function applyCardWidth(card, width) {{
      panelSpanClasses.forEach((cls) => card.classList.remove(cls));
      const widthText = String(width || "6");
      card.classList.add(`span-${{widthText}}`);
      card.dataset.width = widthText;
    }}

    function applyCardHeight(card, height) {{
      const value = ["s", "m", "l", "xl"].includes(String(height)) ? String(height) : "m";
      card.dataset.height = value;
    }}

    function resizeAllPlots() {{
      document.querySelectorAll(".plotly-graph-div").forEach((el) => {{
        try {{
          Plotly.Plots.resize(el);
        }} catch (_err) {{
          /* ignore */
        }}
      }});
    }}

    function scheduleResponsiveRerender() {{
      if (layoutRerenderTimer) {{
        window.clearTimeout(layoutRerenderTimer);
      }}
      layoutRerenderTimer = window.setTimeout(() => {{
        renderVolcanoPlot();
        renderProgramSizePlot();
        renderGseaPlot();
        renderHeatmapPlot();
        if (selectedProgram) {{
          renderProgramSpotlight();
          renderMultiPathwayForSelected();
          renderContextForSelected();
          renderMembershipForSelected();
        }}
        resizeAllPlots();
      }}, 90);
    }}

    function saveDashboardLayout() {{
      const grid = document.getElementById("dashboard-grid");
      if (!grid) return;
      const layout = Array.from(grid.querySelectorAll(":scope > .card")).map((card) => ({{
        id: card.dataset.cardId || "",
        width: currentCardWidth(card),
        height: card.dataset.height || "m",
      }}));
      localStorage.setItem(layoutStorageKey, JSON.stringify(layout));
      layoutCustomized = true;
      updateLayoutStatus();
      scheduleResponsiveRerender();
    }}

    function applyDashboardLayout(layout) {{
      const grid = document.getElementById("dashboard-grid");
      if (!grid) return;
      const cards = new Map(
        Array.from(grid.querySelectorAll(":scope > .card")).map((card) => [card.dataset.cardId || "", card])
      );
      layout.forEach((item) => {{
        const card = cards.get(String(item?.id || ""));
        if (!card) return;
        grid.appendChild(card);
        applyCardWidth(card, item?.width || currentCardWidth(card));
        applyCardHeight(card, item?.height || "m");
      }});
      scheduleResponsiveRerender();
    }}

    function loadDashboardLayout() {{
      const raw = localStorage.getItem(layoutStorageKey);
      if (!raw) return false;
      let layout = null;
      try {{
        layout = JSON.parse(raw);
      }} catch (_err) {{
        return false;
      }}
      if (!Array.isArray(layout)) return false;
      layoutCustomized = true;
      applyDashboardLayout(layout);
      return true;
    }}

    function resetDashboardLayout() {{
      localStorage.removeItem(layoutStorageKey);
      layoutCustomized = false;
      window.location.reload();
    }}

    function updateLayoutStatus() {{
      const el = document.getElementById("layout-status");
      if (!el) return;
      el.textContent = layoutEditing
        ? "Layout edit on: drag panels and change width/height"
        : (layoutCustomized ? "Custom layout saved" : "Default layout");
    }}

    function ensureCardTools(card) {{
      if (!card || card.querySelector(".panel-tools")) return;
      const sectionHead = card.querySelector(".section-head");
      if (!sectionHead) return;
      const tools = document.createElement("div");
      tools.className = "panel-tools";
      tools.innerHTML = `
        <button type="button" class="card-grip" title="Drag to move this panel">::</button>
        <select class="panel-select" data-role="width" title="Panel width">
          <option value="3">Width 3</option>
          <option value="4">Width 4</option>
          <option value="5">Width 5</option>
          <option value="6">Width 6</option>
          <option value="7">Width 7</option>
          <option value="8">Width 8</option>
          <option value="12">Width 12</option>
        </select>
        <select class="panel-select" data-role="height" title="Panel height">
          <option value="s">Height S</option>
          <option value="m">Height M</option>
          <option value="l">Height L</option>
          <option value="xl">Height XL</option>
        </select>
      `;
      sectionHead.appendChild(tools);

      const widthSelect = tools.querySelector('[data-role="width"]');
      const heightSelect = tools.querySelector('[data-role="height"]');
      if (widthSelect) {{
        widthSelect.value = currentCardWidth(card);
        widthSelect.addEventListener("change", (ev) => {{
          applyCardWidth(card, ev.target.value || "6");
          saveDashboardLayout();
        }});
      }}
      if (heightSelect) {{
        heightSelect.value = card.dataset.height || "m";
        heightSelect.addEventListener("change", (ev) => {{
          applyCardHeight(card, ev.target.value || "m");
          saveDashboardLayout();
        }});
      }}

      const grip = tools.querySelector(".card-grip");
      if (grip) {{
        grip.setAttribute("draggable", "true");
        grip.addEventListener("dragstart", (ev) => {{
          if (!layoutEditing) {{
            ev.preventDefault();
            return;
          }}
          card.classList.add("card-dragging");
          ev.dataTransfer.effectAllowed = "move";
          ev.dataTransfer.setData("text/plain", card.dataset.cardId || "");
        }});
        grip.addEventListener("dragend", () => {{
          card.classList.remove("card-dragging");
          document.querySelectorAll(".card-drop-target").forEach((el) => el.classList.remove("card-drop-target"));
        }});
      }}

      card.addEventListener("dragover", (ev) => {{
        if (!layoutEditing) return;
        const draggedId = ev.dataTransfer?.getData("text/plain");
        if (!draggedId || draggedId === card.dataset.cardId) return;
        ev.preventDefault();
        card.classList.add("card-drop-target");
      }});
      card.addEventListener("dragleave", () => {{
        card.classList.remove("card-drop-target");
      }});
      card.addEventListener("drop", (ev) => {{
        if (!layoutEditing) return;
        const draggedId = ev.dataTransfer?.getData("text/plain");
        if (!draggedId || draggedId === card.dataset.cardId) return;
        ev.preventDefault();
        card.classList.remove("card-drop-target");
        const grid = document.getElementById("dashboard-grid");
        const draggedCard = grid?.querySelector(`.card[data-card-id="${{draggedId}}"]`);
        if (!grid || !draggedCard) return;
        const rect = card.getBoundingClientRect();
        const after = ev.clientY > rect.top + rect.height / 2 || ev.clientX > rect.left + rect.width / 2;
        if (after) {{
          grid.insertBefore(draggedCard, card.nextSibling);
        }} else {{
          grid.insertBefore(draggedCard, card);
        }}
        saveDashboardLayout();
      }});
    }}

    function initLayoutEditor() {{
      const cards = document.querySelectorAll("#dashboard-grid > .card");
      cards.forEach((card) => ensureCardTools(card));
      const toggle = document.getElementById("layout-toggle");
      const reset = document.getElementById("layout-reset");
      if (toggle) {{
        toggle.addEventListener("click", () => {{
          layoutEditing = !layoutEditing;
          document.body.classList.toggle("layout-editing", layoutEditing);
          updateLayoutStatus();
        }});
      }}
      if (reset) {{
        reset.addEventListener("click", resetDashboardLayout);
      }}
      if (!loadDashboardLayout()) {{
        applyDashboardLayout(defaultDashboardLayout);
      }}
      updateLayoutStatus();
    }}

    function initTabCards() {{
      document.querySelectorAll("[data-tab-card]").forEach((card) => {{
        const buttons = Array.from(card.querySelectorAll("[data-tab-target]"));
        const panels = Array.from(card.querySelectorAll("[data-tab-panel]"));
        if (buttons.length === 0 || panels.length === 0) return;
        const activate = (targetId) => {{
          buttons.forEach((button) => {{
            button.classList.toggle("active", button.dataset.tabTarget === targetId);
          }});
          panels.forEach((panel) => {{
            const active = panel.dataset.tabPanel === targetId;
            panel.classList.toggle("active", active);
          }});
          resizeAllPlots();
        }};
        buttons.forEach((button) => {{
          button.addEventListener("click", () => activate(button.dataset.tabTarget || ""));
        }});
        activate(buttons[0].dataset.tabTarget || "");
      }});
    }}

    function renderVolcanoPlot() {{
      const volcano = payload.de || [];
      Plotly.react(
        "plot-volcano",
        [{{
          x: volcano.map(d => d.logfc_a_minus_b),
          y: volcano.map(d => d.neglog10p),
          text: volcano.map(d => `${{d.gene}}<br>FDR=${{Number(d.fdr).toExponential(2)}}`),
          mode: "markers",
          type: "scattergl",
          marker: {{
            size: 7,
            color: volcano.map(d => d.sig ? "#e4572e" : "#4a5c6a"),
            opacity: 0.75,
          }},
          hovertemplate: "%{{text}}<extra></extra>",
        }}],
        {{
          margin: {{l: 60, r: 24, t: 16, b: 60}},
          xaxis: {{title: "logFC (A - B)", automargin: true}},
          yaxis: {{title: "-log10(p-value)", automargin: true}},
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
        }},
        plotConfig
      );
    }}

    function renderProgramSizePlot() {{
      const sizes = payload.sizes || [];
      const sizesHeight = responsiveRowHeight("plot-sizes", sizes.length, 400, 34, 148, 1040);
      const sizesEl = document.getElementById("plot-sizes");
      if (sizesEl) sizesEl.style.height = `${{sizesHeight}}px`;
      Plotly.react(
        "plot-sizes",
        [{{
          x: sizes.map(d => d.n_genes),
          y: sizes.map(d => d.program),
          customdata: sizes.map(d => d.program),
          text: sizes.map(d => fullLabelFor(d.program)),
          type: "bar",
          orientation: "h",
          marker: {{color: "#3a86ff"}},
          hovertemplate: "Program=%{{text}}<br>Genes=%{{x}}<extra></extra>",
        }}],
        {{
          height: sizesHeight,
          margin: {{l: 400, r: 24, t: 16, b: 60}},
          xaxis: {{title: "Gene count", automargin: true}},
          yaxis: {{
            autorange: "reversed",
            automargin: true,
            tickmode: "array",
            tickvals: sizes.map(d => d.program),
            ticktext: sizes.map(d => wrapLabel(axisLabelFor(d.program), 28)),
            tickfont: {{size: 11}},
          }},
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
        }},
        plotConfig
      );
      if (sizesEl) {{
        sizesEl.removeAllListeners?.("plotly_click");
        sizesEl.on("plotly_click", (ev) => {{
          const p = ev?.points?.[0]?.customdata;
          if (p) setProgram(p);
        }});
      }}
    }}

    function renderGseaPlot() {{
      const gsea = payload.gsea || [];
      const gseaHeight = responsiveRowHeight("plot-gsea", gsea.length, 420, 34, 148, 1080);
      const gseaEl = document.getElementById("plot-gsea");
      if (gseaEl) gseaEl.style.height = `${{gseaHeight}}px`;
      Plotly.react(
        "plot-gsea",
        [{{
          x: gsea.map(d => d.nes),
          y: gsea.map(d => d.program),
          customdata: gsea.map(d => d.program),
          mode: "markers",
          type: "scatter",
          marker: {{
            size: gsea.map(d => Math.max(8, -Math.log10(Math.max(d.fdr, 1e-12)) * 5)),
            color: gsea.map(d => d.fdr),
            colorscale: "YlOrRd",
            reversescale: true,
            colorbar: {{title: "FDR"}},
            opacity: 0.85,
          }},
          text: gsea.map(d => `${{fullLabelFor(d.program)}}<br>NES=${{Number(d.nes).toFixed(3)}}<br>FDR=${{Number(d.fdr).toExponential(2)}}`),
          hovertemplate: "%{{text}}<extra></extra>",
        }}],
        {{
          height: gseaHeight,
          margin: {{l: 400, r: 34, t: 16, b: 60}},
          xaxis: {{title: "NES", automargin: true}},
          yaxis: {{
            autorange: "reversed",
            automargin: true,
            tickmode: "array",
            tickvals: gsea.map(d => d.program),
            ticktext: gsea.map(d => wrapLabel(axisLabelFor(d.program), 28)),
            tickfont: {{size: 11}},
          }},
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
        }},
        plotConfig
      );
      if (gseaEl) {{
        gseaEl.removeAllListeners?.("plotly_click");
        gseaEl.on("plotly_click", (ev) => {{
          const p = ev?.points?.[0]?.customdata;
          if (p) setProgram(p);
        }});
      }}
    }}

    function renderHeatmapPlot() {{
      const heat = payload.heatmap || [];
      const heatEl = document.getElementById("plot-heatmap");
      if (heat.length === 0) {{
        if (heatEl) {{
          heatEl.innerHTML = "<div style='padding:20px;color:#5a6b7f;'>No annotation overlap data found. Run with --annotate-programs.</div>";
        }}
        return;
      }}
      const programs = [...new Set(heat.map(d => d.program))];
      const refs = [...new Set(heat.map(d => d.reference_name))];
      const refDisplayMap = new Map(
        heat.map((d) => [d.reference_name, d.reference_display || d.reference_name])
      );
      const heatHeight = responsiveRowHeight("plot-heatmap", programs.length, 390, 26, 160, 1020);
      if (heatEl) heatEl.style.height = `${{heatHeight}}px`;
      const z = programs.map(p => refs.map(r => {{
        const row = heat.find(x => x.program === p && x.reference_name === r);
        return row ? row.jaccard : 0.0;
      }}));
      Plotly.react(
        "plot-heatmap",
        [{{
          z: z,
          x: refs,
          y: programs,
          customdata: programs.map((p) => refs.map((r) => [fullLabelFor(p), refDisplayMap.get(r) || r])),
          type: "heatmap",
          colorscale: "YlOrRd",
          zmin: 0,
          zmax: Math.max(...z.flat()),
          colorbar: {{title: "Jaccard"}},
          hovertemplate: "Program=%{{customdata[0]}}<br>Ref=%{{customdata[1]}}<br>Jaccard=%{{z:.3f}}<extra></extra>",
        }}],
        {{
          height: heatHeight,
          margin: {{l: 360, r: 48, t: 16, b: 150}},
          xaxis: {{
            tickangle: -45,
            tickmode: "array",
            tickvals: refs,
            ticktext: refs.map((r) => wrapLabel(refDisplayMap.get(r) || r, 28)),
            tickfont: {{size: 10}},
            automargin: true,
          }},
          yaxis: {{
            autorange: "reversed",
            automargin: true,
            tickmode: "array",
            tickvals: programs,
            ticktext: programs.map(p => wrapLabel(axisLabelFor(p), 38)),
            tickfont: {{size: 10}},
          }},
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
        }},
        plotConfig
      );
      if (heatEl) {{
        heatEl.removeAllListeners?.("plotly_click");
        heatEl.on("plotly_click", (ev) => {{
          const p = ev?.points?.[0]?.y;
          if (p) setProgram(p);
        }});
      }}
    }}

    function populateProgramOptions(filterText = "") {{
      const selector = document.getElementById("program-selector");
      if (!selector) return;
      const q = String(filterText || "").trim().toLowerCase();
      const visible = (q
        ? programUniverse.filter((program) => {{
            const label = fullLabelFor(program).toLowerCase();
            return String(program).toLowerCase().includes(q) || label.includes(q);
          }})
        : programUniverse
      );
      const options = visible.length > 0 ? visible : programUniverse;
      selector.innerHTML = "";
      options.forEach((program) => {{
        const opt = document.createElement("option");
        opt.value = program;
        opt.textContent = labelFor(program);
        opt.title = fullLabelFor(program);
        selector.appendChild(opt);
      }});
      if (selectedProgram && options.includes(selectedProgram)) {{
        selector.value = selectedProgram;
      }} else if (options.length > 0) {{
        selectedProgram = options[0];
        selector.value = selectedProgram;
      }}
    }}

    function renderProgramSpotlight() {{
      const target = document.getElementById("program-spotlight");
      if (!target) return;
      const row = selectedProgram ? programSummaryMap.get(selectedProgram) : null;
      if (!row) {{
        target.innerHTML = "<div class='spotlight-row'><strong>No program summary available</strong><span>Select a program from the explorer to inspect its evidence chain.</span></div>";
        return;
      }}
      const claimClass = boolLike(row.claim_supported) ? "mini-pill pass" : "mini-pill warn";
      const claimLabel = boolLike(row.claim_supported) ? "Claim supported" : "Claim not fully supported";
      const refText = row.top_reference_name
        ? `${{esc(row.top_reference_name)}} (Jaccard ${{fmtFixed(row.top_reference_jaccard, 2)}})`
        : "No curated anchor saved";
      const contextText = row.top_context_gene
        ? `${{esc(row.top_context_gene)}} | evidence ${{fmtFixed(row.top_context_evidence_abs ?? row.top_context_evidence, 2)}}`
        : "No context driver saved";
      const membershipText = row.top_membership_gene
        ? `${{esc(row.top_membership_gene)}} | membership ${{fmtFixed(row.top_membership, 3)}}`
        : "No membership leader saved";
      target.innerHTML = `
        <div class="spotlight-head">
          <div class="spotlight-title">${{esc(fullLabelFor(row.program))}}</div>
          <div class="spotlight-badges">
            <span class="${{claimClass}}">${{claimLabel}}</span>
            <span class="mini-pill">FDR ${{fmtExp(row.fdr)}}</span>
            <span class="mini-pill">NES ${{fmtFixed(row.nes, 2)}}</span>
          </div>
        </div>
        <div class="spotlight-metrics">
          <div class="spot-stat">
            <span>Program size</span>
            <strong>${{fmtFixed(row.n_genes, 0)}} genes</strong>
          </div>
          <div class="spot-stat">
            <span>Top context shift</span>
            <strong>${{fmtFixed(row.top_context_shift, 2)}}</strong>
          </div>
        </div>
        <div class="spotlight-list">
          <div class="spotlight-row">
            <strong>Top context driver</strong>
            <span>${{contextText}}</span>
          </div>
          <div class="spotlight-row">
            <strong>Top membership gene</strong>
            <span>${{membershipText}}</span>
          </div>
          <div class="spotlight-row">
            <strong>Best curated anchor</strong>
            <span>${{refText}}</span>
          </div>
        </div>
      `;
    }}

    function renderMultiPathwayForSelected() {{
      const plotId = "plot-multi-pathway";
      const tableId = "multi-pathway-table";
      const globalCurveRows = (payload.global_multi_pathway_curves || [])
        .slice()
        .sort((a, b) => Math.abs(numOrZero(b.es)) - Math.abs(numOrZero(a.es)));
      const title = document.getElementById("multi-pathway-title");
      const copy = document.getElementById("multi-pathway-copy");
      const curveRows = (payload.multi_pathway_curves || [])
        .filter((row) => row.program === selectedProgram)
        .sort((a, b) => Math.abs(numOrZero(b.es)) - Math.abs(numOrZero(a.es)) || numOrZero(b.jaccard) - numOrZero(a.jaccard))
        .slice(0, 5);
      const activeCurveRows = curveRows.length > 0 ? curveRows : globalCurveRows.slice(0, 6);
      if (title) {{
        if (curveRows.length > 0) {{
          title.textContent = selectedProgram
            ? "Selected Program: Multi-Pathway Enrichment Curves - " + labelFor(selectedProgram)
            : "Selected Program: Multi-Pathway Enrichment Curves";
        }} else if (activeCurveRows.length > 0) {{
          title.textContent = "Global Ranked-List Multi-Pathway Enrichment Curves";
        }} else {{
          title.textContent = "Selected Program: Multi-Pathway Enrichment Curves";
        }}
      }}
      if (copy) {{
        if (curveRows.length > 0) {{
          copy.textContent = "Classic GSEA-style running enrichment curves for curated pathways linked to the selected dynamic program. Compare peak position, enrichment strength, and leading-edge composition in one place.";
        }} else if (activeCurveRows.length > 0) {{
          copy.textContent = "No program-linked curated curves were saved for this analysis, so this panel falls back to the top curated pathways on the full ranked gene list. This is the correct view for analyses where the curated GMT is global rather than program-specific.";
        }} else {{
          copy.textContent = "Classic GSEA-style running enrichment curves for multiple curated pathways mapped onto the selected dynamic program.";
        }}
      }}
      const rows = (curveRows.length > 0 ? curveRows : (payload.multi_pathway || []))
        .filter((row) => row.program === selectedProgram)
        .sort((a, b) => Math.abs(numOrZero(b.es)) - Math.abs(numOrZero(a.es)) || numOrZero(b.jaccard) - numOrZero(a.jaccard))
        .slice(0, 8);
      if (activeCurveRows.length === 0) {{
        Plotly.react(
          plotId,
          [],
          {{
            margin: {{l: 90, r: 20, t: 10, b: 40}},
            annotations: [{{
              text: "No multi-pathway enrichment curves available for this program",
              showarrow: false,
              xref: "paper",
              yref: "paper",
              x: 0.5,
              y: 0.5,
            }}],
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(0,0,0,0)",
          }},
          plotConfig
        );
        renderInteractiveTable(
          tableId,
          rows.map((row) => ({{
            "Reference pathway": row.reference_display || row.reference_name,
            Jaccard: numOrZero(row.jaccard).toFixed(3),
            "Overlap genes": numOrZero(row.overlap_n).toFixed(0),
            "Program genes": numOrZero(row.program_n).toFixed(0),
            "Novel genes outside reference": numOrZero(row.novel_gene_estimate).toFixed(0),
          }})),
          ["Reference pathway", "Jaccard", "Overlap genes", "Program genes", "Novel genes outside reference"],
          {{
            sortBy: "Jaccard",
            direction: "desc",
            limit: 8,
            limitChoices: [8, 12, 20],
            csvName: `${{selectedProgram || "selected_program"}}_multi_pathway_hits.csv`,
          }}
        );
        return;
      }}
      const plotHeight = responsiveRowHeight(plotId, activeCurveRows.length, 390, 36, 164, 940);
      const plotEl = document.getElementById(plotId);
      if (plotEl) plotEl.style.height = `${{plotHeight}}px`;
      const palette = ["#1f77b4", "#2a9d8f", "#e76f51", "#6a4c93", "#f4a261", "#264653"];
      const ySeries = activeCurveRows.flatMap((row) => Array.isArray(row.y_points) ? row.y_points.map((v) => numOrZero(v)) : []);
      const yMin = ySeries.length ? Math.min(...ySeries) : -0.5;
      const yMax = ySeries.length ? Math.max(...ySeries) : 0.5;
      const span = Math.max(0.25, yMax - yMin);
      const rugBand = Math.max(0.28, span * 0.36);
      const rugStep = rugBand / Math.max(activeCurveRows.length, 1);
      const rugTop = yMin - span * 0.06;
      const rugBottom = rugTop - rugBand;
      const rugShapes = [];
      const traces = activeCurveRows.map((row, idx) => {{
        const color = palette[idx % palette.length];
        const hits = Array.isArray(row.hit_positions) ? row.hit_positions : [];
        const laneTop = rugTop - idx * rugStep;
        const laneBottom = laneTop - Math.max(rugStep * 0.72, 0.03);
        hits.forEach((hit) => {{
          rugShapes.push({{
            type: "line",
            x0: numOrZero(hit),
            x1: numOrZero(hit),
            y0: laneBottom,
            y1: laneTop,
            line: {{color: color, width: 1}},
            opacity: 0.78,
          }});
        }});
        return {{
          x: row.x_points || [],
          y: row.y_points || [],
          mode: "lines",
          type: "scatter",
          name: `${{row.reference_display}} | ES ${{numOrZero(row.es).toFixed(2)}}`,
          line: {{color: color, width: 2.5}},
          hovertemplate:
            "<b>%{{fullData.name}}</b><br>" +
            "Rank=%{{x}}<br>" +
            "Running ES=%{{y:.3f}}<br>" +
            "Jaccard=" + numOrZero(row.jaccard).toFixed(3) + "<br>" +
            "Overlap genes=" + numOrZero(row.overlap_n).toFixed(0) + "<br>" +
            "Leading-edge genes=" + numOrZero(row.leading_edge_n).toFixed(0) + "<br>" +
            "Leading-edge preview=" + esc(String(row.leading_edge_preview || "NA")) + "<br>" +
            "Genes outside reference=" + numOrZero(row.novel_gene_estimate).toFixed(0) +
            "<extra></extra>",
        }};
      }});
      rugShapes.push({{
        type: "line",
        x0: 1,
        x1: Math.max(...activeCurveRows.map((row) => Array.isArray(row.x_points) && row.x_points.length ? numOrZero(row.x_points[row.x_points.length - 1]) : 1)),
        y0: 0,
        y1: 0,
        line: {{color: "rgba(60,72,88,0.65)", width: 1, dash: "dash"}},
      }});
      Plotly.react(
        plotId,
        traces,
        {{
          height: plotHeight,
          margin: {{l: 82, r: 32, t: 20, b: 72}},
          xaxis: {{title: "Rank in preranked gene list", automargin: true}},
          yaxis: {{
            title: "Running enrichment score",
            range: [rugBottom - rugStep * 0.2, yMax + span * 0.14],
            zeroline: false,
            automargin: true,
          }},
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          legend: {{
            orientation: "h",
            x: 0,
            y: 1.12,
            bgcolor: "rgba(255,255,255,0.72)",
          }},
          annotations: activeCurveRows.map((row, idx) => ({{
            xref: "paper",
            yref: "y",
            x: 0,
            y: rugTop - idx * rugStep - rugStep * 0.36,
            xanchor: "left",
            yanchor: "middle",
            showarrow: false,
            text: esc(String(row.reference_display || row.reference_name || "")),
            font: {{size: 10, color: palette[idx % palette.length]}},
          }})),
          shapes: rugShapes,
        }},
        plotConfig
      );
      renderInteractiveTable(
        tableId,
        activeCurveRows.map((row) => ({{
          "Reference pathway": row.reference_display,
          ES: numOrZero(row.es).toFixed(3),
          Jaccard: numOrZero(row.jaccard).toFixed(3),
          "Overlap genes": numOrZero(row.overlap_n).toFixed(0),
          "Reference hits in ranking": numOrZero(row.n_hits_in_ranking).toFixed(0),
          "Leading-edge genes": numOrZero(row.leading_edge_n).toFixed(0),
          "Leading-edge preview": row.leading_edge_preview || "NA",
          "Novel genes outside reference": numOrZero(row.novel_gene_estimate).toFixed(0),
        }})),
        ["Reference pathway", "ES", "Jaccard", "Overlap genes", "Reference hits in ranking", "Leading-edge genes", "Leading-edge preview", "Novel genes outside reference"],
        {{
          sortBy: "ES",
          direction: "desc",
          limit: 8,
          limitChoices: [8, 12, 20],
          csvName: `${{selectedProgram || "selected_program"}}_multi_pathway_hits.csv`,
        }}
      );
    }}

    function setProgram(program) {{
      if (!program) return;
      selectedProgram = program;
      const selector = document.getElementById("program-selector");
      if (selector) selector.value = program;
      const pill = document.getElementById("program-pill");
      if (pill) {{
        pill.textContent = labelFor(program);
        pill.title = fullLabelFor(program);
      }}
      renderProgramSpotlight();
      renderMultiPathwayForSelected();
      renderContextForSelected();
      renderMembershipForSelected();
    }}

    function renderContextForSelected() {{
      const rows = (payload.context_program || [])
        .filter(r => r.program === selectedProgram)
        .sort((a, b) => metricAbsValue(b, selectedContextMetric) - metricAbsValue(a, selectedContextMetric))
        .slice(0, 10);
      const metricLabel = contextMetricLabels[selectedContextMetric] || "Context metric";
      const axisTitle = contextMetricAxis[selectedContextMetric] || metricLabel;
      const title = document.getElementById("context-title");
      if (title) title.textContent = "Context driver genes (top 10) - " + labelFor(selectedProgram);
      const copy = document.getElementById("context-copy");
      if (copy) copy.textContent = "Top 10 genes ranked by how strongly they help explain the case-control difference inside the selected program.";
      const tableTitle = document.getElementById("context-table-title");
      if (tableTitle) tableTitle.textContent = "Context driver genes (top 10) - " + labelFor(selectedProgram);
      const tableCopy = document.getElementById("context-table-copy");
      if (tableCopy) tableCopy.textContent = "Use this to see the top 10 genes driving the selected program's biological interpretation.";
      if (rows.length === 0) {{
        Plotly.react("plot-context", [], {{
          margin: {{l: 90, r: 20, t: 10, b: 40}},
          annotations: [{{text: "No context rows", showarrow: false, xref: "paper", yref: "paper", x: 0.5, y: 0.5}}],
        }}, plotConfig);
        renderInteractiveTable(
          "context-program-table",
          [],
          ["Gene", "Core-weight score", "Context shift", "Signed significance", "Context evidence", "Rank score"],
          {{
            sortBy: "Rank score",
            direction: "desc",
            limit: 10,
            limitChoices: [10, 20, 35],
            csvName: "selected_program_context_table.csv",
          }}
        );
        return;
      }}
      Plotly.react(
        "plot-context",
        [{{
          x: rows.map(d => metricValue(d, selectedContextMetric)),
          y: rows.map(d => d.gene),
          mode: "markers",
          type: "scatter",
          marker: {{
            size: rows.map(d => Math.max(8, (d.base_membership || 0) * 14)),
            color: rows.map(d => metricValue(d, selectedContextMetric)),
            colorscale: "RdBu",
            reversescale: true,
            opacity: 0.9,
            colorbar: {{title: metricLabel}},
          }},
          text: rows.map(
            d => `${{fullLabelFor(d.program)}} | ${{d.gene}}<br>${{metricLabel}}=${{metricValue(d, selectedContextMetric).toFixed(3)}}<br>context_evidence=${{numOrZero(d.context_evidence).toFixed(3)}}<br>context_shift=${{numOrZero(d.context_shift).toFixed(3)}}`
          ),
          hovertemplate: "%{{text}}<extra></extra>",
        }}],
        {{
          height: responsiveRowHeight("plot-context", rows.length, 390, 26, 142, 920),
          margin: {{l: 150, r: 72, t: 18, b: 72}},
          xaxis: {{title: axisTitle, zeroline: true, zerolinecolor: "#4d5d70", zerolinewidth: 1, automargin: true}},
          yaxis: {{autorange: "reversed", automargin: true}},
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
        }},
        plotConfig
      );
      renderInteractiveTable(
        "context-program-table",
        rows.map(r => {{
          return {{
            Gene: r.gene,
            "Core-weight score": Number(r.base_membership).toFixed(3),
            "Context shift": numOrZero(r.context_shift).toFixed(3),
            "Signed significance": numOrZero(r.signed_significance).toFixed(3),
            "Context evidence": numOrZero(r.context_evidence).toFixed(3),
            "Rank score": metricAbsValue(r, selectedContextMetric).toFixed(3),
          }};
        }}),
        ["Gene", "Core-weight score", "Context shift", "Signed significance", "Context evidence", "Rank score"],
        {{
          sortBy: "Rank score",
          direction: "desc",
          limit: 10,
          limitChoices: [10, 20, 35],
          csvName: `${{selectedProgram || "selected_program"}}_context_table.csv`,
        }}
      );
    }}

    function renderMembershipForSelected() {{
      const rows = (payload.membership_program || [])
        .filter(r => r.program === selectedProgram)
        .sort((a, b) => (b.base_membership ?? 0) - (a.base_membership ?? 0))
        .slice(0, 10);
      const title = document.getElementById("membership-title");
      if (title) title.textContent = "Core genes in selected program (top 10) - " + labelFor(selectedProgram);
      const copy = document.getElementById("membership-copy");
      if (copy) copy.textContent = "Top 10 genes ranked by how strongly they belong to the selected program. This is a within-program centrality score, not an effect-size estimate.";
      const tableTitle = document.getElementById("membership-table-title");
      if (tableTitle) tableTitle.textContent = "Core genes for selected program (top 10) - " + labelFor(selectedProgram);
      const tableCopy = document.getElementById("membership-table-copy");
      if (tableCopy) tableCopy.textContent = "These are the top 10 genes most central to the selected program, with context columns shown alongside for quick interpretation.";
      if (rows.length === 0) {{
        Plotly.react("plot-membership", [], {{
          margin: {{l: 90, r: 20, t: 10, b: 40}},
          annotations: [{{text: "No membership rows", showarrow: false, xref: "paper", yref: "paper", x: 0.5, y: 0.5}}],
        }}, plotConfig);
        renderInteractiveTable(
          "membership-program-table",
          [],
          ["Gene", "Core-weight score", "Context shift", "Signed significance", "Context evidence"],
          {{
            sortBy: "Core-weight score",
            direction: "desc",
            limit: 10,
            limitChoices: [10, 20, 35],
            csvName: "selected_program_membership_table.csv",
          }}
        );
        return;
      }}
      Plotly.react(
        "plot-membership",
        [{{
          x: rows.map(d => d.base_membership),
          y: rows.map(d => d.gene),
          type: "bar",
          orientation: "h",
          text: rows.map(
            d => `${{fullLabelFor(d.program)}} | ${{d.gene}}<br>Membership=${{numOrZero(d.base_membership).toFixed(3)}}<br>${{contextMetricLabels[selectedContextMetric]}}=${{metricValue(d, selectedContextMetric).toFixed(3)}}<br>context_evidence=${{numOrZero(d.context_evidence).toFixed(3)}}<br>context_shift=${{numOrZero(d.context_shift).toFixed(3)}}`
          ),
          marker: {{
            color: rows.map(d => metricValue(d, selectedContextMetric)),
            colorscale: "RdBu",
            reversescale: true,
            line: {{color: "rgba(0,0,0,0.12)", width: 0.5}},
          }},
          hovertemplate: "%{{text}}<extra></extra>",
        }}],
        {{
          height: responsiveRowHeight("plot-membership", rows.length, 390, 26, 142, 920),
          margin: {{l: 150, r: 42, t: 18, b: 72}},
          xaxis: {{title: "Base membership", automargin: true}},
          yaxis: {{autorange: "reversed", automargin: true}},
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
        }},
        plotConfig
      );
      renderInteractiveTable(
        "membership-program-table",
        rows.map(r => {{
          return {{
            Gene: r.gene,
            "Core-weight score": Number(r.base_membership).toFixed(3),
            "Context shift": numOrZero(r.context_shift).toFixed(3),
            "Signed significance": numOrZero(r.signed_significance).toFixed(3),
            "Context evidence": numOrZero(r.context_evidence).toFixed(3),
          }};
        }}),
        ["Gene", "Core-weight score", "Context shift", "Signed significance", "Context evidence"],
        {{
          sortBy: "Core-weight score",
          direction: "desc",
          limit: 10,
          limitChoices: [10, 20, 35],
          csvName: `${{selectedProgram || "selected_program"}}_membership_table.csv`,
        }}
      );
    }}

    function renderReferenceLayers() {{
      const tabHost = document.getElementById("reference-source-tabs");
      const summaryHost = document.getElementById("reference-source-summary");
      const tableHost = document.getElementById("reference-source-table");
      if (!tabHost || !summaryHost || !tableHost) return;

      const rows = Array.isArray(payload.reference_source_hits) ? payload.reference_source_hits.slice() : [];
      if (rows.length === 0) {{
        tabHost.innerHTML = "";
        summaryHost.innerHTML = "<div class='summary-item'><span class='summary-label'>No source-grouped reference hits</span><span class='summary-text'>Run with --annotate-programs and at least one reference collection to populate this panel.</span><span class='summary-cta'>No source layers detected</span></div>";
        renderInteractiveTable(
          "reference-source-table",
          [],
          ["Program", "Reference pathway", "Jaccard", "Overlap genes", "Program genes", "Novel genes outside reference"],
          {{
            sortBy: "Jaccard",
            direction: "desc",
            limit: 12,
            limitChoices: [12, 20, 50],
            csvName: "reference_source_hits.csv",
          }}
        );
        return;
      }}

      const grouped = new Map();
      rows.forEach((row) => {{
        const key = String(row.source_display || row.source || "Other");
        if (!grouped.has(key)) grouped.set(key, []);
        grouped.get(key).push(row);
      }});

      const rankedGroups = Array.from(grouped.entries())
        .map(([sourceDisplay, sourceRows]) => {{
          const maxJaccard = Math.max(...sourceRows.map((row) => numOrZero(row.jaccard)), 0);
          const maxInterpretation = Math.max(...sourceRows.map((row) => numOrZero(row.interpretation_score)), 0);
          const maxPriority = Math.max(...sourceRows.map((row) => numOrZero(row.disease_priority_score)), 0);
          const meanJaccard = sourceRows.length > 0
            ? sourceRows.reduce((sum, row) => sum + numOrZero(row.jaccard), 0) / sourceRows.length
            : 0;
          const uniquePrograms = new Set(sourceRows.map((row) => String(row.program || ""))).size;
          const uniqueRefs = new Set(sourceRows.map((row) => String(row.reference_name || ""))).size;
          return {{
            sourceDisplay,
            rows: sourceRows.slice().sort((a, b) =>
              numOrZero(b.interpretation_score) - numOrZero(a.interpretation_score) ||
              numOrZero(b.disease_priority_score) - numOrZero(a.disease_priority_score) ||
              numOrZero(b.jaccard) - numOrZero(a.jaccard) ||
              numOrZero(b.overlap_n) - numOrZero(a.overlap_n)
            ),
            maxJaccard,
            maxInterpretation,
            maxPriority,
            meanJaccard,
            uniquePrograms,
            uniqueRefs,
          }};
        }})
        .sort((a, b) =>
          b.maxInterpretation - a.maxInterpretation ||
          b.maxPriority - a.maxPriority ||
          b.maxJaccard - a.maxJaccard ||
          b.uniquePrograms - a.uniquePrograms ||
          a.sourceDisplay.localeCompare(b.sourceDisplay)
        );

      if (!selectedReferenceSource || !rankedGroups.some((group) => group.sourceDisplay === selectedReferenceSource)) {{
        selectedReferenceSource = rankedGroups[0]?.sourceDisplay || null;
      }}
      const activeGroup = rankedGroups.find((group) => group.sourceDisplay === selectedReferenceSource) || rankedGroups[0];
      selectedReferenceSource = activeGroup?.sourceDisplay || null;

      tabHost.innerHTML = rankedGroups.map((group) => {{
        const active = group.sourceDisplay === selectedReferenceSource ? " active" : "";
        return `<button type="button" class="tab-button${{active}}" data-source-tab="${{esc(group.sourceDisplay)}}">${{esc(group.sourceDisplay)}} <span style="opacity:.7">(${{group.rows.length}})</span></button>`;
      }}).join("");
      tabHost.querySelectorAll("[data-source-tab]").forEach((button) => {{
        button.addEventListener("click", () => {{
          selectedReferenceSource = button.getAttribute("data-source-tab");
          renderReferenceLayers();
        }});
      }});

      if (!activeGroup) return;
      const topRow = activeGroup.rows[0] || null;
      summaryHost.innerHTML = `
        <button type="button" class="summary-item${{topRow?.program ? " summary-item-action" : ""}}" data-program="${{esc(String(topRow?.program || ""))}}">
          <span class="summary-label">Source</span>
          <span class="summary-text">${{esc(activeGroup.sourceDisplay)}} contributes ${{activeGroup.rows.length}} high-overlap rows across ${{activeGroup.uniquePrograms}} programs.</span>
          <span class="summary-cta">Knowledge layer overview</span>
        </button>
        <button type="button" class="summary-item${{topRow?.program ? " summary-item-action" : ""}}" data-program="${{esc(String(topRow?.program || ""))}}">
          <span class="summary-label">Top pathway</span>
          <span class="summary-text">${{esc(String(topRow?.reference_display || topRow?.reference_name || "Not available"))}}</span>
          <span class="summary-cta">${{String(topRow?.priority_band || "Background")}} | score ${{fmtFixed(topRow?.interpretation_score, 2)}}</span>
        </button>
        <div class="summary-item">
          <span class="summary-label">Coverage</span>
          <span class="summary-text">${{activeGroup.uniqueRefs}} distinct pathways are visible from this source in the current dashboard slice.</span>
          <span class="summary-cta">Programs covered: ${{activeGroup.uniquePrograms}}</span>
        </div>
        <div class="summary-item">
          <span class="summary-label">Typical overlap</span>
          <span class="summary-text">Mean Jaccard within this source is ${{fmtFixed(activeGroup.meanJaccard, 3)}}. The dashboard ranks sources by interpretation score first so generic layers do not dominate solely by overlap magnitude.</span>
          <span class="summary-cta">Best source score: ${{fmtFixed(activeGroup.maxInterpretation, 2)}}</span>
        </div>
      `;
      summaryHost.querySelectorAll(".summary-item-action").forEach((el) => {{
        el.addEventListener("click", () => {{
          const program = el.getAttribute("data-program");
          if (program) setProgram(program);
        }});
      }});

      renderInteractiveTable(
        "reference-source-table",
        activeGroup.rows.map((row) => ({{
          program: row.program,
          Program: labelFor(row.program),
          "Reference pathway": row.reference_display || row.reference_name,
          Priority: row.priority_band || "Background",
          "Interpretation score": numOrZero(row.interpretation_score).toFixed(2),
          Jaccard: numOrZero(row.jaccard).toFixed(3),
          "Overlap genes": numOrZero(row.overlap_n).toFixed(0),
          "Program genes": numOrZero(row.program_n).toFixed(0),
          "Reference genes": numOrZero(row.reference_n).toFixed(0),
          "Novel genes outside reference": numOrZero(row.novel_gene_estimate).toFixed(0),
        }})),
        ["Program", "Reference pathway", "Priority", "Interpretation score", "Jaccard", "Overlap genes", "Program genes", "Reference genes", "Novel genes outside reference"],
        {{
          sortBy: "Interpretation score",
          direction: "desc",
          limit: 12,
          limitChoices: [12, 20, 50],
          csvName: `${{activeGroup.sourceDisplay.toLowerCase().replace(/[^a-z0-9]+/g, "_")}}_reference_hits.csv`,
          programKey: "program",
        }}
      );

      const familyRows = Array.isArray(payload.reference_family_hits) ? payload.reference_family_hits.slice() : [];
      renderInteractiveTable(
        "reference-family-table",
        familyRows.map((row) => ({{
          program: row.top_program,
          Family: row.family_display,
          Priority: row.priority_band || "Background",
          "Priority score": numOrZero(row.disease_priority_score).toFixed(1),
          "Interpretation score": numOrZero(row.interpretation_score).toFixed(2),
          Sources: row.sources_display,
          "Best pathway label": row.top_reference_display || row.top_reference_name,
          "Best Jaccard": numOrZero(row.best_jaccard).toFixed(3),
          "Mean Jaccard": numOrZero(row.mean_jaccard).toFixed(3),
          "Programs covered": numOrZero(row.programs_covered).toFixed(0),
          "References merged": numOrZero(row.references_merged).toFixed(0),
        }})),
        ["Family", "Priority", "Priority score", "Interpretation score", "Sources", "Best pathway label", "Best Jaccard", "Mean Jaccard", "Programs covered", "References merged"],
        {{
          sortBy: "Interpretation score",
          direction: "desc",
          limit: 12,
          limitChoices: [12, 20, 50],
          csvName: "reference_family_hits.csv",
          programKey: "program",
        }}
      );

      const calibrationRows = Array.isArray(payload.reference_ranking_calibration)
        ? payload.reference_ranking_calibration.slice()
        : [];
      renderInteractiveTable(
        "reference-calibration-table",
        calibrationRows.map((row) => ({{
          Family: row.family_display,
          Priority: row.priority_band || "Background",
          "Raw rank": finiteOrNull(row.raw_rank) === null ? "NA" : Number(row.raw_rank).toFixed(0),
          "Prioritized rank": finiteOrNull(row.prioritized_rank) === null ? "NA" : Number(row.prioritized_rank).toFixed(0),
          "Interpretation score": numOrZero(row.interpretation_score).toFixed(2),
          "Best Jaccard": numOrZero(row.best_jaccard).toFixed(3),
          "Moved into top set": row.prioritized_top && !row.raw_top ? "Yes" : "No",
          "Dropped from top set": row.raw_top && !row.prioritized_top ? "Yes" : "No",
        }})),
        ["Family", "Priority", "Raw rank", "Prioritized rank", "Interpretation score", "Best Jaccard", "Moved into top set", "Dropped from top set"],
        {{
          sortBy: "Prioritized rank",
          direction: "asc",
          limit: 10,
          limitChoices: [10, 20, 40],
          csvName: "reference_ranking_calibration.csv",
        }}
      );
    }}

    initLayoutEditor();
    initTabCards();

    renderVolcanoPlot();
    renderProgramSizePlot();
    renderGseaPlot();
    renderHeatmapPlot();

    const selector = document.getElementById("program-selector");
    if (selector) {{
      populateProgramOptions();
      selector.addEventListener("change", (ev) => setProgram(ev.target.value));
    }}
    const searchInput = document.getElementById("program-search");
    if (searchInput) {{
      searchInput.addEventListener("input", (ev) => {{
        populateProgramOptions(ev.target.value || "");
      }});
    }}
    const contextMetricSelector = document.getElementById("context-metric-selector");
    if (contextMetricSelector) {{
      contextMetricSelector.value = selectedContextMetric;
      contextMetricSelector.addEventListener("change", (ev) => {{
        selectedContextMetric = ev.target.value || "context_evidence";
        renderContextForSelected();
        renderMembershipForSelected();
      }});
    }}
    document.querySelectorAll(".story-card-action, .metric-card-action").forEach((el) => {{
      el.addEventListener("click", () => {{
        const program = el.getAttribute("data-program");
        if (program) setProgram(program);
      }});
    }});
    const bestProgram = programSummary
      .filter((d) => finiteOrNull(d.fdr) !== null)
      .sort((a, b) => numOrZero(a.fdr) - numOrZero(b.fdr))[0]?.program;
    const largestProgram = programSummary
      .filter((d) => finiteOrNull(d.n_genes) !== null)
      .sort((a, b) => numOrZero(b.n_genes) - numOrZero(a.n_genes))[0]?.program;
    const strongestContextProgram = programSummary
      .filter((d) => finiteOrNull(d.top_context_evidence_abs ?? d.top_context_evidence) !== null)
      .sort((a, b) => numOrZero(b.top_context_evidence_abs ?? b.top_context_evidence) - numOrZero(a.top_context_evidence_abs ?? a.top_context_evidence))[0]?.program;
    const bindJumpButton = (id, program) => {{
      const button = document.getElementById(id);
      if (!button) return;
      if (!program) {{
        button.disabled = true;
        return;
      }}
      button.addEventListener("click", () => setProgram(program));
    }};
    bindJumpButton("jump-best-program", bestProgram);
    bindJumpButton("jump-largest-program", largestProgram);
    bindJumpButton("jump-context-program", strongestContextProgram);
    if (bestProgram) selectedProgram = bestProgram;
    renderInteractiveTable(
      "headline-summary-table",
      payload.headline_summary || [],
      ["metric_display", "value"],
      {{
        sortBy: "value",
        direction: "desc",
        limit: 10,
        limitChoices: [10, 20],
        csvName: "headline_summary.csv",
      }}
    );
    renderInteractiveTable(
      "gate-summary-table",
      payload.gate_summary || [],
      ["gate_display", "n_evaluable", "n_pass", "pass_rate"],
      {{
        sortBy: "pass_rate",
        direction: "desc",
        limit: 10,
        limitChoices: [10, 20],
        csvName: "claim_gate_summary.csv",
      }}
    );
    renderInteractiveTable(
      "top-enriched-table",
      payload.top_enriched_table || [],
      ["Program", "Genes", "NES", "FDR", "Best curated match", "Claim status"],
      {{
        sortBy: "FDR",
        direction: "asc",
        limit: 12,
        limitChoices: [12, 20, 50],
        csvName: "top_enriched_programs.csv",
        programKey: "program",
      }}
    );
    renderInteractiveTable(
      "top-context-table",
      payload.top_context_table || [],
      ["Program", "Context driver gene", "Context evidence", "Context shift", "Best curated match"],
      {{
        sortBy: "Context evidence",
        direction: "desc",
        limit: 12,
        limitChoices: [12, 20, 50],
        csvName: "top_context_evidence_genes.csv",
        programKey: "program",
      }}
    );
    renderInteractiveTable(
      "gene-membership-table",
      payload.gene_membership_table || [],
      ["Program", "Core gene", "Core-weight score", "Genes", "Best curated match"],
      {{
        sortBy: "Core-weight score",
        direction: "desc",
        limit: 12,
        limitChoices: [12, 20, 50],
        csvName: "top_genes_per_program.csv",
        programKey: "program",
      }}
    );
    renderReferenceLayers();
    if (selectedProgram) setProgram(selectedProgram);
  </script>
</body>
</html>"""
        return html


def _records(df: pd.DataFrame, cols: list[str]) -> list[dict[str, Any]]:
    """Convert DataFrame to JSON-serializable list of records."""
    out = df.loc[:, [c for c in cols if c in df.columns]].copy()
    out = out.replace([np.inf, -np.inf], np.nan).fillna(value=np.nan)
    return out.to_dict(orient="records")


def _prepare_overlap_heatmap_long(
    overlap_df: pd.DataFrame | None,
    gsea_df: pd.DataFrame,
    top_k: int = 20,
) -> pd.DataFrame:
    """Prepare compact long-form overlap data for interactive heatmap."""
    if overlap_df is None or overlap_df.empty:
        return pd.DataFrame(columns=["program", "reference_name", "reference_display", "jaccard"])
    if "program" not in overlap_df.columns or "reference_name" not in overlap_df.columns:
        return pd.DataFrame(columns=["program", "reference_name", "reference_display", "jaccard"])

    if "program" in gsea_df.columns and "fdr" in gsea_df.columns:
        gsea_rank = gsea_df.copy()
        gsea_rank["fdr"] = _coerce_numeric_series(gsea_rank["fdr"], fill_value=1.0)
        top_programs = (
            gsea_rank.sort_values("fdr", ascending=True)["program"].astype(str).head(top_k).tolist()
        )
    else:
        top_programs = []
    subset = overlap_df.copy()
    subset["program"] = subset["program"].astype(str)
    subset["reference_name"] = subset["reference_name"].astype(str)
    if "jaccard" in subset.columns:
        subset["jaccard"] = _coerce_numeric_series(subset["jaccard"], fill_value=0.0).clip(
            lower=0.0, upper=1.0
        )
    elif {"overlap_n", "program_n", "reference_n"}.issubset(subset.columns):
        overlap_n = _coerce_numeric_series(subset["overlap_n"], fill_value=0.0)
        program_n = _coerce_numeric_series(subset["program_n"], fill_value=0.0)
        reference_n = _coerce_numeric_series(subset["reference_n"], fill_value=0.0)
        denom = program_n + reference_n - overlap_n
        ratio = np.divide(
            overlap_n.to_numpy(dtype=np.float64),
            denom.to_numpy(dtype=np.float64),
            out=np.zeros(len(subset), dtype=np.float64),
            where=denom.to_numpy(dtype=np.float64) > 0,
        )
        subset["jaccard"] = np.clip(ratio, 0.0, 1.0)
    else:
        subset["jaccard"] = 0.0

    if top_programs:
        subset = subset[subset["program"].isin(top_programs)].copy()

    subset = subset.sort_values("jaccard", ascending=False).copy()
    top_refs = subset.groupby("reference_name", as_index=False)["jaccard"].max()
    keep_refs = top_refs.sort_values("jaccard", ascending=False).head(top_k)["reference_name"].tolist()
    subset = subset[subset["reference_name"].isin(keep_refs)].copy()

    subset["reference_display"] = subset["reference_name"].map(_display_reference_name)
    return subset.loc[:, ["program", "reference_name", "reference_display", "jaccard"]].reset_index(drop=True)


def _prepare_multi_pathway_rows(
    overlap_df: pd.DataFrame | None,
    gsea_df: pd.DataFrame,
    top_k_programs: int = 30,
    top_n_refs: int = 8,
) -> pd.DataFrame:
    """Prepare per-program top reference hits for the selected-program multi-pathway view."""
    if overlap_df is None or overlap_df.empty:
        return pd.DataFrame(
            columns=[
                "program",
                "reference_name",
                "reference_display",
                "jaccard",
                "overlap_n",
                "program_n",
                "reference_n",
                "novel_gene_estimate",
            ]
        )

    subset = overlap_df.copy()
    if "program" not in subset.columns or "reference_name" not in subset.columns:
        return pd.DataFrame(
            columns=[
                "program",
                "reference_name",
                "reference_display",
                "jaccard",
                "overlap_n",
                "program_n",
                "reference_n",
                "novel_gene_estimate",
            ]
        )

    subset["program"] = subset["program"].astype(str)
    subset["reference_name"] = subset["reference_name"].astype(str)

    if "program" in gsea_df.columns and "fdr" in gsea_df.columns:
        gsea_rank = gsea_df.copy()
        gsea_rank["fdr"] = _coerce_numeric_series(gsea_rank["fdr"], fill_value=1.0)
        top_programs = (
            gsea_rank.sort_values("fdr", ascending=True)["program"].astype(str).head(top_k_programs).tolist()
        )
        subset = subset[subset["program"].isin(top_programs)].copy()

    if "jaccard" in subset.columns:
        subset["jaccard"] = _coerce_numeric_series(subset["jaccard"], fill_value=0.0).clip(
            lower=0.0, upper=1.0
        )
    elif {"overlap_n", "program_n", "reference_n"}.issubset(subset.columns):
        overlap_n = _coerce_numeric_series(subset["overlap_n"], fill_value=0.0)
        program_n = _coerce_numeric_series(subset["program_n"], fill_value=0.0)
        reference_n = _coerce_numeric_series(subset["reference_n"], fill_value=0.0)
        denom = program_n + reference_n - overlap_n
        subset["jaccard"] = np.divide(
            overlap_n.to_numpy(dtype=np.float64),
            denom.to_numpy(dtype=np.float64),
            out=np.zeros(len(subset), dtype=np.float64),
            where=denom.to_numpy(dtype=np.float64) > 0,
        )
    else:
        subset["jaccard"] = 0.0

    for column in ("overlap_n", "program_n", "reference_n"):
        if column in subset.columns:
            subset[column] = _coerce_numeric_series(subset[column], fill_value=0.0)
        else:
            subset[column] = 0.0

    subset["reference_display"] = subset["reference_name"].map(_display_reference_name)
    subset["novel_gene_estimate"] = (subset["program_n"] - subset["overlap_n"]).clip(lower=0.0)
    subset = (
        subset.sort_values(["program", "jaccard", "overlap_n"], ascending=[True, False, False])
        .groupby("program", as_index=False, sort=False)
        .head(top_n_refs)
        .reset_index(drop=True)
    )
    return subset.loc[
        :,
        [
            "program",
            "reference_name",
            "reference_display",
            "jaccard",
            "overlap_n",
            "program_n",
            "reference_n",
            "novel_gene_estimate",
        ],
    ]


def _extract_reference_source(reference_name: str) -> str:
    raw = str(reference_name)
    if "::" in raw:
        return raw.split("::", 1)[0]
    if raw.startswith("WP_"):
        return "WP"
    if raw.startswith("KEGG_"):
        return "KEGG"
    if raw.startswith("HALLMARK_"):
        return "HALLMARK"
    if raw.startswith("REACTOME_"):
        return "REACTOME"
    return "OTHER"


def _display_reference_source(source: str) -> str:
    mapping = {
        "WP": "WikiPathways",
        "REACTOME": "Reactome",
        "PATHWAYCOMMONS": "Pathway Commons",
        "GO_BP": "GO BP",
        "GO_CC": "GO CC",
        "GO_MF": "GO MF",
        "HALLMARK": "Hallmark",
        "KEGG": "KEGG",
        "CUSTOM": "Custom GMT",
        "OTHER": "Other",
    }
    return mapping.get(str(source), str(source).replace("_", " "))


def _prepare_reference_source_rows(
    overlap_df: pd.DataFrame | None,
    gsea_df: pd.DataFrame,
    top_k_programs: int = 25,
    top_n_refs_per_source: int = 12,
) -> pd.DataFrame:
    """Prepare source-grouped reference hits for dashboard tabs."""
    rows = _prepare_multi_pathway_rows(
        overlap_df=overlap_df,
        gsea_df=gsea_df,
        top_k_programs=top_k_programs,
        top_n_refs=top_n_refs_per_source * 3,
    )
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "source",
                "source_display",
                "program",
                "reference_name",
                "reference_display",
                "priority_band",
                "disease_priority_score",
                "interpretation_score",
                "jaccard",
                "overlap_n",
                "program_n",
                "reference_n",
                "novel_gene_estimate",
            ]
        )
    rows = rows.copy()
    rows["source"] = rows["reference_name"].astype(str).map(_extract_reference_source)
    rows["source_display"] = rows["source"].map(_display_reference_source)
    rows["disease_priority_score"] = rows["reference_name"].astype(str).map(reference_relevance_score)
    rows["priority_band"] = rows["reference_name"].astype(str).map(reference_relevance_band)
    rows["interpretation_score"] = rows.apply(
        lambda row: source_interpretation_score(
            best_priority_score=float(row["disease_priority_score"]),
            best_jaccard=float(row.get("jaccard", 0.0)),
            programs_covered=1,
            references_covered=1,
        ),
        axis=1,
    )
    rows = (
        rows.sort_values(
            ["source_display", "interpretation_score", "disease_priority_score", "jaccard", "overlap_n"],
            ascending=[True, False, False, False, False],
        )
        .groupby("source_display", as_index=False, sort=False)
        .head(top_n_refs_per_source)
        .reset_index(drop=True)
    )
    return rows


def _reference_family_key(reference_name: str) -> str:
    """Collapse near-duplicate reference labels into conservative family keys."""
    raw = str(reference_name).split("::", 1)[-1]
    raw = re.sub(r"^(GOBP|GOCC|GOMF|WP|KEGG|REACTOME|HALLMARK)_+", "", raw, flags=re.IGNORECASE)
    tokens = re.split(r"[^A-Za-z0-9]+", raw.upper())
    stopwords = {
        "",
        "PATHWAY",
        "PATHWAYS",
        "PROCESS",
        "PROCESSES",
        "SET",
        "MODULE",
        "MODULES",
        "SIGNATURE",
        "SIGNATURES",
        "HOMO",
        "SAPIENS",
        "MUS",
        "MUSCULUS",
    }
    kept = [token for token in tokens if token not in stopwords]
    if not kept:
        kept = [token for token in tokens if token]
    return " ".join(kept)


def _display_reference_family(family_key: str) -> str:
    """Human-readable display label for collapsed reference families."""
    text = str(family_key).strip()
    if not text:
        return "Unlabeled family"
    return text.title()


def _prepare_reference_family_rows(
    overlap_df: pd.DataFrame | None,
    gsea_df: pd.DataFrame,
    top_k_programs: int = 25,
    top_n_families: int = 20,
) -> pd.DataFrame:
    """Collapse source hits into conservative cross-database reference families."""
    rows = _prepare_multi_pathway_rows(
        overlap_df=overlap_df,
        gsea_df=gsea_df,
        top_k_programs=top_k_programs,
        top_n_refs=40,
    )
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "family_key",
                "family_display",
                "top_program",
                "top_reference_name",
                "top_reference_display",
                "disease_priority_score",
                "priority_band",
                "interpretation_score",
                "best_jaccard",
                "mean_jaccard",
                "programs_covered",
                "references_merged",
                "source_count",
                "sources_display",
            ]
        )
    rows = rows.copy()
    rows["family_key"] = rows["reference_name"].astype(str).map(_reference_family_key)
    grouped_rows: list[dict[str, Any]] = []
    for family_key, sub in rows.groupby("family_key", sort=False):
        best = sub.sort_values(["jaccard", "overlap_n"], ascending=[False, False]).iloc[0]
        sources = sorted({_display_reference_source(_extract_reference_source(name)) for name in sub["reference_name"].astype(str)})
        priority_score = float(reference_relevance_score(str(family_key)))
        grouped_rows.append(
            {
                "family_key": str(family_key),
                "family_display": _display_reference_family(str(family_key)),
                "top_program": str(best["program"]),
                "top_reference_name": str(best["reference_name"]),
                "top_reference_display": str(best["reference_display"]),
                "disease_priority_score": priority_score,
                "priority_band": reference_relevance_band(str(family_key)),
                "best_jaccard": float(pd.to_numeric(sub["jaccard"], errors="coerce").fillna(0.0).max()),
                "mean_jaccard": float(pd.to_numeric(sub["jaccard"], errors="coerce").fillna(0.0).mean()),
                "programs_covered": int(sub["program"].astype(str).nunique()),
                "references_merged": int(sub["reference_name"].astype(str).nunique()),
                "source_count": len(sources),
                "sources_display": ", ".join(sources),
            }
        )
    out = pd.DataFrame(grouped_rows)
    if out.empty:
        return out
    out["interpretation_score"] = out.apply(
        lambda row: family_interpretation_score(
            reference_name=str(row["family_key"]),
            best_jaccard=float(row["best_jaccard"]),
            programs_covered=int(row["programs_covered"]),
            source_count=int(row["source_count"]),
        ),
        axis=1,
    )
    out = out.sort_values(
        ["interpretation_score", "disease_priority_score", "best_jaccard", "programs_covered", "references_merged"],
        ascending=[False, False, False, False, False],
    ).head(top_n_families)
    return out.reset_index(drop=True)


def _prepare_reference_ranking_calibration_rows(
    overlap_df: pd.DataFrame | None,
    gsea_df: pd.DataFrame,
    top_k_programs: int = 25,
    top_n_display: int = 10,
) -> pd.DataFrame:
    """Compare raw Jaccard ranking with interpretation-prioritized ranking."""
    family_rows = _prepare_reference_family_rows(
        overlap_df=overlap_df,
        gsea_df=gsea_df,
        top_k_programs=top_k_programs,
        top_n_families=80,
    )
    if family_rows.empty:
        return pd.DataFrame(
            columns=[
                "family_key",
                "family_display",
                "priority_band",
                "disease_priority_score",
                "interpretation_score",
                "best_jaccard",
                "raw_rank",
                "prioritized_rank",
                "raw_top",
                "prioritized_top",
            ]
        )

    raw_ranked = family_rows.sort_values(
        ["best_jaccard", "programs_covered", "references_merged"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    raw_ranked["raw_rank"] = np.arange(1, len(raw_ranked) + 1)

    prioritized = family_rows.sort_values(
        ["interpretation_score", "disease_priority_score", "best_jaccard", "programs_covered"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    prioritized["prioritized_rank"] = np.arange(1, len(prioritized) + 1)

    merged = raw_ranked.merge(
        prioritized.loc[:, ["family_key", "prioritized_rank"]],
        on="family_key",
        how="outer",
    )
    merged["raw_rank"] = pd.to_numeric(merged["raw_rank"], errors="coerce")
    merged["prioritized_rank"] = pd.to_numeric(merged["prioritized_rank"], errors="coerce")
    merged["raw_top"] = merged["raw_rank"].le(top_n_display).fillna(False)
    merged["prioritized_top"] = merged["prioritized_rank"].le(top_n_display).fillna(False)
    merged = merged.loc[merged["raw_top"] | merged["prioritized_top"]].copy()
    merged = merged.sort_values(
        ["prioritized_rank", "raw_rank", "interpretation_score", "best_jaccard"],
        ascending=[True, True, False, False],
        na_position="last",
    ).reset_index(drop=True)
    return merged


def _reference_key_candidates(reference_name: str) -> list[str]:
    """Generate candidate GMT keys for a reference label saved in overlap outputs."""
    raw = str(reference_name)
    candidates = [raw]
    if "::" not in raw:
        return candidates

    source, term = raw.split("::", 1)
    candidates.append(term)
    if source == "WP":
        candidates.append(f"WP_{term}")
    elif source == "KEGG":
        candidates.append(f"KEGG_{term}")
    elif source == "REACTOME":
        candidates.append(f"REACTOME_{term}")
    elif source == "HALLMARK":
        candidates.append(f"HALLMARK_{term}")
    elif source == "GO_BP":
        candidates.append(term if term.startswith("GOBP_") else f"GOBP_{term}")
    elif source == "GO_CC":
        candidates.append(term if term.startswith("GOCC_") else f"GOCC_{term}")
    elif source == "GO_MF":
        candidates.append(term if term.startswith("GOMF_") else f"GOMF_{term}")
    return list(dict.fromkeys(candidates))


def _build_enrichment_curve_record(
    ranked_names: np.ndarray,
    ranked_scores: np.ndarray,
    reference_name: str,
    reference_display: str,
    genes: list[str],
    *,
    program: str | None = None,
    jaccard: float = np.nan,
    overlap_n: float = np.nan,
    program_n: float = np.nan,
    reference_n: float = np.nan,
    novel_gene_estimate: float = np.nan,
    max_points: int = 400,
    max_hits: int = 120,
) -> dict[str, Any] | None:
    """Build one downsampled running-sum curve record for dashboard payloads."""
    gene_set = {str(g) for g in genes}
    mask = np.array([gene in gene_set for gene in ranked_names], dtype=bool)
    if not mask.any():
        return None

    es, running_sum = _compute_enrichment_score(
        ranked_names,
        mask,
        ranked_scores,
        weighted_score_type=1.0,
    )
    if es >= 0:
        edge_idx = int(np.argmax(running_sum))
        leading_edge_mask = mask[: edge_idx + 1]
        leading_edge_genes = ranked_names[: edge_idx + 1][leading_edge_mask]
    else:
        edge_idx = int(np.argmin(running_sum))
        leading_edge_mask = mask[edge_idx:]
        leading_edge_genes = ranked_names[edge_idx:][leading_edge_mask]
    leading_edge_genes = np.asarray(leading_edge_genes, dtype=str)
    preview_genes = leading_edge_genes[:10].tolist()
    preview_text = ", ".join(preview_genes)
    if len(leading_edge_genes) > 10:
        preview_text += ", ..."

    rank_positions_full = np.arange(1, len(ranked_names) + 1, dtype=np.int32)
    if len(running_sum) <= max_points:
        keep_idx = np.arange(len(running_sum), dtype=np.int32)
    else:
        keep_idx = np.linspace(0, len(running_sum) - 1, num=max_points, dtype=np.int32)
        keep_idx = np.unique(keep_idx)
    hit_positions = rank_positions_full[mask]
    if len(hit_positions) > max_hits:
        hit_keep = np.linspace(0, len(hit_positions) - 1, num=max_hits, dtype=np.int32)
        hit_positions = hit_positions[hit_keep]

    return {
        "program": str(program) if program is not None else "__global__",
        "reference_name": str(reference_name),
        "reference_display": str(reference_display),
        "jaccard": float(jaccard) if pd.notna(jaccard) else np.nan,
        "overlap_n": float(overlap_n) if pd.notna(overlap_n) else np.nan,
        "program_n": float(program_n) if pd.notna(program_n) else np.nan,
        "reference_n": float(reference_n) if pd.notna(reference_n) else float(len(gene_set)),
        "novel_gene_estimate": float(novel_gene_estimate) if pd.notna(novel_gene_estimate) else np.nan,
        "es": float(es),
        "n_hits_in_ranking": int(mask.sum()),
        "leading_edge_n": int(len(leading_edge_genes)),
        "leading_edge_genes": leading_edge_genes[:24].tolist(),
        "leading_edge_preview": preview_text,
        "x_points": rank_positions_full[keep_idx].astype(int).tolist(),
        "y_points": running_sum[keep_idx].astype(float).round(6).tolist(),
        "hit_positions": hit_positions.astype(int).tolist(),
    }


def _prepare_multi_pathway_curve_records(
    ranked_genes_df: pd.DataFrame | None,
    reference_gene_sets: dict[str, list[str]] | None,
    overlap_df: pd.DataFrame | None,
    gsea_df: pd.DataFrame,
    top_k_programs: int = 20,
    top_n_refs: int = 4,
    max_points: int = 400,
    max_hits: int = 120,
) -> list[dict[str, Any]]:
    """Precompute downsampled multi-pathway running-sum curves for the dashboard."""
    if (
        ranked_genes_df is None
        or ranked_genes_df.empty
        or reference_gene_sets is None
        or not reference_gene_sets
        or overlap_df is None
        or overlap_df.empty
    ):
        return []
    if not {"gene", "score"}.issubset(ranked_genes_df.columns):
        return []

    ranked_names = ranked_genes_df["gene"].astype(str).to_numpy()
    ranked_scores = _coerce_numeric_series(ranked_genes_df["score"], fill_value=0.0).to_numpy(dtype=np.float64)
    top_rows = _prepare_multi_pathway_rows(
        overlap_df=overlap_df,
        gsea_df=gsea_df,
        top_k_programs=top_k_programs,
        top_n_refs=top_n_refs,
    )
    if top_rows.empty:
        return []

    records: list[dict[str, Any]] = []

    for row in top_rows.itertuples(index=False):
        genes = None
        for candidate in _reference_key_candidates(str(row.reference_name)):
            genes = reference_gene_sets.get(candidate)
            if genes:
                break
        if not genes:
            continue
        record = _build_enrichment_curve_record(
            ranked_names=ranked_names,
            ranked_scores=ranked_scores,
            reference_name=str(row.reference_name),
            reference_display=_display_reference_name(str(row.reference_name)),
            genes=genes,
            program=str(row.program),
            jaccard=float(getattr(row, "jaccard", np.nan)),
            overlap_n=float(getattr(row, "overlap_n", np.nan)),
            program_n=float(getattr(row, "program_n", np.nan)),
            reference_n=float(getattr(row, "reference_n", np.nan)),
            novel_gene_estimate=float(getattr(row, "novel_gene_estimate", np.nan)),
            max_points=max_points,
            max_hits=max_hits,
        )
        if record is not None:
            records.append(record)
    return records


def _prepare_global_gsea_curve_records(
    ranked_genes_df: pd.DataFrame | None,
    reference_gene_sets: dict[str, list[str]] | None,
    reference_score_df: pd.DataFrame | None = None,
    top_n_refs: int = 6,
    max_points: int = 400,
    max_hits: int = 120,
) -> list[dict[str, Any]]:
    """Precompute global multi-pathway GSEA curves from the ranked list and curated GMT."""
    if ranked_genes_df is None or ranked_genes_df.empty or not reference_gene_sets:
        return []
    if not {"gene", "score"}.issubset(ranked_genes_df.columns):
        return []

    ranked_names = ranked_genes_df["gene"].astype(str).to_numpy()
    ranked_scores = _coerce_numeric_series(ranked_genes_df["score"], fill_value=0.0).to_numpy(dtype=np.float64)
    records: list[dict[str, Any]] = []
    ordered_reference_names = list(reference_gene_sets.keys())
    if reference_score_df is not None and not reference_score_df.empty and "program" in reference_score_df.columns:
        score_rank = reference_score_df.copy()
        if "fdr" in score_rank.columns:
            score_rank["fdr"] = _coerce_numeric_series(score_rank["fdr"], fill_value=1.0)
        if "nes" in score_rank.columns:
            score_rank["nes"] = _coerce_numeric_series(score_rank["nes"], fill_value=0.0)
        sort_cols: list[str] = []
        ascending: list[bool] = []
        if "fdr" in score_rank.columns:
            sort_cols.append("fdr")
            ascending.append(True)
        if "nes" in score_rank.columns:
            sort_cols.append("nes")
            ascending.append(False)
        if sort_cols:
            score_rank = score_rank.sort_values(sort_cols, ascending=ascending)
        ordered_reference_names = [
            str(name)
            for name in score_rank["program"].astype(str).tolist()
            if str(name) in reference_gene_sets
        ]
        if ordered_reference_names:
            ordered_reference_names = list(dict.fromkeys(ordered_reference_names))

    for reference_name in ordered_reference_names:
        genes = reference_gene_sets[reference_name]
        record = _build_enrichment_curve_record(
            ranked_names=ranked_names,
            ranked_scores=ranked_scores,
            reference_name=str(reference_name),
            reference_display=_display_reference_name(str(reference_name)),
            genes=list(genes),
            program="__global__",
            max_points=max_points,
            max_hits=max_hits,
        )
        if record is not None:
            records.append(record)
    if not ordered_reference_names:
        records.sort(key=lambda row: abs(float(row["es"])), reverse=True)
    return records[:top_n_refs]


def _escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _chunk_text(text: str, width: int = 28) -> str:
    """Wrap long labels; chunk contiguous strings when no whitespace exists."""
    t = str(text)
    if len(t) <= width:
        return t
    if " " in t:
        words = t.split()
        lines: list[str] = []
        cur = ""
        for w in words:
            nxt = f"{cur} {w}".strip()
            if len(nxt) <= width:
                cur = nxt
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return "<br>".join(lines)
    return "<br>".join(t[i : i + width] for i in range(0, len(t), width))


def _program_label_parts(program: str) -> tuple[str, str, str]:
    """Parse program identifier into stable ID, source namespace, and phrase."""
    raw = str(program)
    pid = raw
    term = raw
    if "__" in raw:
        pid, term = raw.split("__", 1)

    term = term.replace("::", "_").replace("-", "_").replace("/", "_")
    while "__" in term:
        term = term.replace("__", "_")

    prefix_map = {
        "GOBP_": "GO BP",
        "GOBP": "GO BP",
        "KEGG_": "KEGG",
        "KEGG": "KEGG",
        "HALLMARK_": "HALLMARK",
        "HALLMARK": "HALLMARK",
        "REACTOME_": "REACTOME",
        "REACTOME": "REACTOME",
        "CUSTOM_": "CUSTOM",
        "CUSTOM": "CUSTOM",
    }
    source = ""
    for k, v in prefix_map.items():
        if term.startswith(k):
            source = v
            term = term[len(k) :].strip("_")
            break

    tokens = [x for x in term.split("_") if x]
    if tokens:
        phrase = " ".join(tokens)
    else:
        phrase = term
    if not phrase:
        phrase = raw

    head = pid.replace("program_", "P")
    return head, source, phrase


def _display_program_name(program: str) -> str:
    """Full display label for tables and hover text."""
    head, source, phrase = _program_label_parts(program)
    if source:
        return f"{head} - {source}: {phrase}"
    return f"{head} - {phrase}"


def _display_program_name_short(program: str, max_words: int = 5, max_chars: int = 52) -> str:
    """Compact program label for selectors and small UI elements."""
    head, source, phrase = _program_label_parts(program)
    tokens = phrase.split()
    if len(tokens) > max_words:
        phrase = " ".join(tokens[:max_words]) + " ..."
    if len(phrase) > max_chars:
        phrase = phrase[: max_chars - 4].rstrip() + " ..."
    if source:
        return f"{head} - {source}: {phrase}"
    return f"{head} - {phrase}"


def _display_program_name_axis(program: str, max_words: int = 4, max_chars: int = 36) -> str:
    """Axis-focused label emphasizing program ID first to improve scanability."""
    head, source, phrase = _program_label_parts(program)
    tokens = phrase.split()
    if len(tokens) > max_words:
        phrase = " ".join(tokens[:max_words]) + " ..."
    if len(phrase) > max_chars:
        phrase = phrase[: max_chars - 4].rstrip() + " ..."
    if source:
        return f"{head} ({source}) {phrase}"
    return f"{head} {phrase}"


def _display_reference_name(reference_name: str, max_words: int = 6, max_chars: int = 48) -> str:
    """Humanize curated pathway names for axes and tables."""
    raw = str(reference_name)
    source = ""
    term = raw
    if "::" in raw:
        source_raw, term = raw.split("::", 1)
        source_map = {
            "GO_BP": "GO BP",
            "GO_CC": "GO CC",
            "GO_MF": "GO MF",
            "KEGG": "KEGG",
            "WP": "WikiPathways",
            "REACTOME": "Reactome",
            "HALLMARK": "Hallmark",
            "CUSTOM": "Custom",
        }
        source = source_map.get(source_raw, source_raw.replace("_", " "))

    term = term.replace("/", "_").replace("-", "_")
    while "__" in term:
        term = term.replace("__", "_")
    tokens = [token for token in term.split("_") if token]
    phrase = " ".join(tokens) if tokens else raw
    if len(tokens) > max_words:
        phrase = " ".join(tokens[:max_words]) + " ..."
    if len(phrase) > max_chars:
        phrase = phrase[: max_chars - 4].rstrip() + " ..."
    if source:
        return f"{source}: {phrase}"
    return phrase


def _humanize_gate_name(name: str) -> str:
    """Convert gate column names into concise reader-facing labels."""
    raw = str(name).strip()
    if raw.startswith("gate_"):
        raw = raw[len("gate_") :]
    words = [part for part in raw.replace("-", "_").split("_") if part]
    if not words:
        return "Claim gate"

    normalized = " ".join(words).lower()
    mapping = {
        "fdr": "FDR gate",
        "effect": "Effect-size gate",
        "program size": "Program-size gate",
        "size": "Program-size gate",
        "stability": "Stability gate",
        "context": "Context gate",
    }
    if normalized in mapping:
        return mapping[normalized]

    human_words = []
    for word in words:
        upper = word.upper()
        if upper in {"FDR", "NES", "DE"}:
            human_words.append(upper)
        else:
            human_words.append(word.capitalize())
    return " ".join(human_words)


def _coerce_numeric_series(series: pd.Series, fill_value: float) -> pd.Series:
    """Coerce numeric-like series and replace non-finite values."""
    coerced = pd.to_numeric(series, errors="coerce")
    return coerced.replace([np.inf, -np.inf], np.nan).fillna(fill_value)


def _resolve_context_shift(df: pd.DataFrame) -> pd.Series | None:
    """Resolve context shift from native or legacy columns."""
    if "context_shift" in df.columns:
        return _coerce_numeric_series(df["context_shift"], fill_value=0.0)

    prob_cols = sorted(c for c in df.columns if c.startswith("prob_"))
    if len(prob_cols) >= 2:
        p_first = _coerce_numeric_series(df[prob_cols[0]], fill_value=0.5)
        p_second = _coerce_numeric_series(df[prob_cols[1]], fill_value=0.5)
        return (p_first - p_second).clip(lower=-1.0, upper=1.0)

    score_cols = sorted(c for c in df.columns if c.startswith("contextual_score_"))
    if len(score_cols) >= 2 and "base_membership" in df.columns:
        score_first = _coerce_numeric_series(df[score_cols[0]], fill_value=0.0)
        score_second = _coerce_numeric_series(df[score_cols[1]], fill_value=0.0)
        base = _coerce_numeric_series(df["base_membership"], fill_value=0.0).to_numpy(dtype=np.float64)
        denom = np.where(np.abs(base) > 1e-12, base, np.nan)
        shift = np.divide(
            score_first.to_numpy(dtype=np.float64) - score_second.to_numpy(dtype=np.float64),
            denom,
            out=np.zeros(len(df), dtype=np.float64),
            where=np.isfinite(denom),
        )
        return pd.Series(np.clip(shift, -1.0, 1.0), index=df.index)

    return None


def _ensure_context_metrics(context_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure context metric columns exist for robust sorting and plotting."""
    df = context_df.copy()
    context_shift = _resolve_context_shift(df)
    if context_shift is None:
        if df.empty:
            df["context_shift"] = pd.Series(dtype=np.float64)
        else:
            raise ValueError(
                "contextual_membership_scores.csv is missing 'context_shift' and no "
                "compatible fallback columns were found (need >=2 prob_* or contextual_score_*)."
            )
    else:
        df["context_shift"] = context_shift

    if "signed_significance" in df.columns:
        df["signed_significance"] = _coerce_numeric_series(df["signed_significance"], fill_value=0.0)
    else:
        if {"logfc_a_minus_b", "p_value"}.issubset(df.columns):
            logfc = _coerce_numeric_series(df["logfc_a_minus_b"], fill_value=0.0)
            pvals = _coerce_numeric_series(df["p_value"], fill_value=1.0).clip(lower=1e-300, upper=1.0)
            neglog10p = -np.log10(np.clip(pvals.to_numpy(dtype=np.float64), 1e-300, 1.0))
            df["signed_significance"] = np.sign(logfc.to_numpy(dtype=np.float64)) * neglog10p
        else:
            df["signed_significance"] = df["context_shift"]

    if "context_evidence" in df.columns:
        df["context_evidence"] = _coerce_numeric_series(df["context_evidence"], fill_value=0.0)
    else:
        if "neglog10_p_value" in df.columns:
            neglog10p_arr = _coerce_numeric_series(df["neglog10_p_value"], fill_value=0.0).clip(lower=0.0)
            df["context_evidence"] = df["context_shift"] * neglog10p_arr
        elif "p_value" in df.columns:
            pvals = _coerce_numeric_series(df["p_value"], fill_value=1.0).clip(lower=1e-300, upper=1.0)
            neglog10p = -np.log10(np.clip(pvals.to_numpy(dtype=np.float64), 1e-300, 1.0))
            df["context_evidence"] = df["context_shift"].to_numpy(dtype=np.float64) * neglog10p
        else:
            df["context_evidence"] = df["context_shift"]

    if "neglog10_p_value" in df.columns:
        df["neglog10_p_value"] = _coerce_numeric_series(
            df["neglog10_p_value"], fill_value=0.0
        ).clip(lower=0.0)

    df["abs_shift"] = df["context_shift"].abs()
    df["abs_signed_significance"] = df["signed_significance"].abs()
    df["abs_context_evidence"] = df["context_evidence"].abs()
    return df


def _prepare_program_explorer_frames(
    context_df: pd.DataFrame,
    gsea_df: pd.DataFrame,
    top_k_programs: int = 30,
    top_n_per_program: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Prepare compact per-program context/membership tables for interactive drill-down."""
    if context_df.empty:
        return (
            pd.DataFrame(
                columns=[
                    "program",
                    "gene",
                    "base_membership",
                    "context_shift",
                    "signed_significance",
                    "context_evidence",
                    "abs_shift",
                    "abs_signed_significance",
                    "abs_context_evidence",
                ]
            ),
            pd.DataFrame(
                columns=[
                    "program",
                    "gene",
                    "base_membership",
                    "context_shift",
                    "signed_significance",
                    "context_evidence",
                    "abs_shift",
                    "abs_signed_significance",
                    "abs_context_evidence",
                ]
            ),
            [],
        )

    if "program" in gsea_df.columns and "fdr" in gsea_df.columns:
        gsea_rank = gsea_df.copy()
        gsea_rank["fdr"] = _coerce_numeric_series(gsea_rank["fdr"], fill_value=1.0)
        top_programs = (
            gsea_rank.sort_values("fdr", ascending=True)["program"].astype(str).head(top_k_programs).tolist()
        )
    else:
        top_programs = []
    base = _ensure_context_metrics(context_df)
    base["program"] = base["program"].astype(str)
    if top_programs:
        base = base[base["program"].isin(top_programs)].copy()
    if base.empty:
        return (
            pd.DataFrame(
                columns=[
                    "program",
                    "gene",
                    "base_membership",
                    "context_shift",
                    "signed_significance",
                    "context_evidence",
                    "abs_shift",
                    "abs_signed_significance",
                    "abs_context_evidence",
                ]
            ),
            pd.DataFrame(
                columns=[
                    "program",
                    "gene",
                    "base_membership",
                    "context_shift",
                    "signed_significance",
                    "context_evidence",
                    "abs_shift",
                    "abs_signed_significance",
                    "abs_context_evidence",
                ]
            ),
            top_programs,
        )

    context_view = (
        base.sort_values(["program", "abs_context_evidence"], ascending=[True, False])
        .groupby("program", as_index=False, sort=False)
        .head(top_n_per_program)
        .reset_index(drop=True)
    )
    membership_view = (
        base.sort_values(["program", "base_membership"], ascending=[True, False])
        .groupby("program", as_index=False, sort=False)
        .head(top_n_per_program)
        .reset_index(drop=True)
    )
    return context_view, membership_view, top_programs


def build_dynamic_dashboard_package(config: DashboardConfig) -> DashboardArtifacts:
    """Build full dashboard package using three-agent workflow."""
    results_dir = Path(config.results_dir)
    output_dir = Path(config.output_dir)
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"

    ingest_agent = ResultIngestionAgent()
    visual_agent = VisualDesignAgent()
    publishing_agent = DashboardPublishingAgent()

    bundle = ingest_agent.load(results_dir)
    visual_agent.create_static_figures(
        bundle=bundle,
        figure_dir=figure_dir,
        top_k=config.top_k,
        include_pdf=config.include_pdf,
    )
    artifacts = publishing_agent.publish(
        bundle=bundle,
        output_dir=output_dir,
        figure_dir=figure_dir,
        table_dir=table_dir,
        title=config.title,
        top_k=config.top_k,
        include_pdf=config.include_pdf,
    )
    _write_legacy_dashboard_alias(output_dir=output_dir, results_dir=results_dir)
    logger.info("Dashboard generated at %s", artifacts.html_path)
    return artifacts


def _write_legacy_dashboard_alias(output_dir: Path, results_dir: Path) -> None:
    """Mirror root dashboard outputs under results_dir/dashboard for backward compatibility."""
    try:
        if output_dir.resolve() != results_dir.resolve():
            return
    except FileNotFoundError:
        return

    legacy_dir = results_dir / "dashboard"
    legacy_dir.mkdir(parents=True, exist_ok=True)

    html_src = output_dir / "index.html"
    if html_src.exists():
        shutil.copy2(html_src, legacy_dir / "index.html")
    summary_src = output_dir / "summary.md"
    if summary_src.exists():
        shutil.copy2(summary_src, legacy_dir / "summary.md")

    for folder_name in ("figures", "tables"):
        src = output_dir / folder_name
        if src.exists():
            shutil.copytree(src, legacy_dir / folder_name, dirs_exist_ok=True)
