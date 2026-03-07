"""Three-agent dashboard builder for dynamic pathway analysis outputs."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.errors import EmptyDataError

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


class ResultIngestionAgent:
    """Agent 1: Validate and ingest analysis outputs."""

    REQUIRED_FILES: tuple[str, ...] = (
        "de_results.csv",
        "enrichment_gsea_with_claim_gates.csv",
        "dynamic_program_sizes.csv",
        "contextual_membership_scores.csv",
    )

    def load(self, results_dir: Path) -> _ResultBundle:
        """Load required and optional result files."""
        missing = [f for f in self.REQUIRED_FILES if not (results_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                "Missing required files in results_dir: " + ", ".join(missing)
            )

        de_df = pd.read_csv(results_dir / "de_results.csv")
        gsea_df = pd.read_csv(results_dir / "enrichment_gsea_with_claim_gates.csv")
        program_sizes_df = pd.read_csv(results_dir / "dynamic_program_sizes.csv")
        context_df = pd.read_csv(results_dir / "contextual_membership_scores.csv")

        fisher_path = results_dir / "enrichment_fisher.csv"
        fisher_df = self._read_optional_csv(fisher_path)
        program_long_path = results_dir / "dynamic_programs_long.csv"
        program_long_df = self._read_optional_csv(program_long_path)
        annotation_path = results_dir / "program_annotation_matches.csv"
        annotation_df = self._read_optional_csv(annotation_path)
        overlap_path = results_dir / "program_reference_overlap_long.csv"
        overlap_df = self._read_optional_csv(overlap_path)

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
        )

    @staticmethod
    def _read_optional_csv(path: Path) -> pd.DataFrame | None:
        """Read optional CSV and return None for missing/empty files."""
        if not path.exists():
            return None
        try:
            return pd.read_csv(path)
        except EmptyDataError:
            return None

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
            .head(top_k)
            .copy()
        )
        top["program_display_axis"] = top["program"].astype(str).map(_display_program_name_axis)
        fig_height = min(12.0, max(6.2, 0.34 * len(top) + 1.8))
        fig, ax = plt.subplots(figsize=(10, fig_height))
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
        ax.tick_params(axis="y", labelsize=9)
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

        top_enriched.to_csv(table_dir / "top_enriched_programs.csv", index=False)
        top_context_shift.to_csv(table_dir / "top_context_shift_genes.csv", index=False)
        top_context_evidence.to_csv(table_dir / "top_context_evidence_genes.csv", index=False)
        gene_membership_top.to_csv(table_dir / "top_genes_per_program.csv", index=False)
        gate_summary.to_csv(table_dir / "claim_gate_summary.csv", index=False)
        headline_summary.to_csv(table_dir / "headline_summary.csv", index=False)
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
            figure_dir=figure_dir,
            table_dir=table_dir,
            include_pdf=include_pdf,
            overlap_df=bundle.overlap_df,
        )
        html_path.write_text(html, encoding="utf-8")

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
                {"metric": "n_genes_tested", "value": int(len(bundle.de_df))},
                {"metric": "n_significant_de_genes", "value": de_sig},
                {"metric": "n_programs_discovered", "value": n_programs},
                {"metric": "n_claim_supported_programs", "value": n_claim_supported},
            ]
        )

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
                "detail": "FDR <= 0.05",
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
        figure_dir: Path,
        table_dir: Path,
        include_pdf: bool,
        overlap_df: pd.DataFrame | None,
    ) -> str:
        de_for_plot = bundle.de_df.copy()
        de_for_plot["neglog10p"] = -np.log10(np.clip(de_for_plot["p_value"], 1e-300, 1.0))
        de_for_plot["sig"] = de_for_plot["fdr"] <= 0.05

        sizes_for_plot = bundle.program_sizes_df.sort_values("n_genes", ascending=False).head(30).copy()
        gsea_for_plot = bundle.gsea_df.copy()
        if "fdr" in gsea_for_plot.columns:
            gsea_for_plot["fdr"] = _coerce_numeric_series(gsea_for_plot["fdr"], fill_value=1.0)
        gsea_for_plot = gsea_for_plot.sort_values("fdr", ascending=True).head(30).copy()
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
        hero_metrics = self._hero_metrics(bundle, gate_summary)
        story_cards = self._story_cards(program_summary)

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
            "story_cards": story_cards,
            "heatmap": _records(
                _prepare_overlap_heatmap_long(
                    overlap_df=overlap_df,
                    gsea_df=bundle.gsea_df,
                    top_k=20,
                ),
                ["program", "reference_name", "jaccard"],
            ),
        }
        payload_json = json.dumps(payload, ensure_ascii=True)

        summary_html = headline_summary.to_html(index=False, classes="np-table", border=0)
        gate_html = gate_summary.to_html(index=False, classes="np-table", border=0)
        top_enriched_html = top_enriched_view.to_html(index=False, classes="np-table", border=0)
        top_context_html = top_context_view.to_html(index=False, classes="np-table", border=0)
        gene_membership_html = gene_membership_view.to_html(index=False, classes="np-table", border=0)
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

        fig_rel = figure_dir.name
        table_rel = table_dir.name
        download_links: list[tuple[str, str]] = [
            ("Headline summary CSV", f"{table_rel}/headline_summary.csv"),
            ("Claim gate summary CSV", f"{table_rel}/claim_gate_summary.csv"),
            ("Top enriched programs CSV", f"{table_rel}/top_enriched_programs.csv"),
            ("Top context evidence CSV", f"{table_rel}/top_context_evidence_genes.csv"),
            ("Top genes per program CSV", f"{table_rel}/top_genes_per_program.csv"),
            ("Figure 1 PNG", f"{fig_rel}/figure_1_volcano.png"),
            ("Figure 2 PNG", f"{fig_rel}/figure_2_program_sizes.png"),
            ("Figure 3 PNG", f"{fig_rel}/figure_3_claim_gates.png"),
            ("Figure 4 PNG", f"{fig_rel}/figure_4_context_shift.png"),
        ]
        if include_pdf:
            download_links.extend(
                [
                    ("Figure 1 PDF", f"{fig_rel}/figure_1_volcano.pdf"),
                    ("Figure 2 PDF", f"{fig_rel}/figure_2_program_sizes.pdf"),
                    ("Figure 3 PDF", f"{fig_rel}/figure_3_claim_gates.pdf"),
                    ("Figure 4 PDF", f"{fig_rel}/figure_4_context_shift.pdf"),
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
    .fig-grid img {{
      width: 100%;
      border: 1px solid rgba(20, 33, 61, 0.08);
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.88);
      box-shadow: var(--shadow-soft);
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
      .toolbar-actions {{
        justify-content: flex-start;
      }}
      .span-7, .span-6, .span-5, .span-4, .span-3 {{ grid-column: span 12; }}
      .plot, .plot-sm {{ height: 360px; }}
    }}
    @media (max-width: 760px) {{
      .wrap {{ padding: 14px 14px 28px; }}
      .hero {{ padding: 20px; border-radius: 24px; }}
      .section-nav {{ top: 0; }}
      .hero-stats,
      .story-grid,
      .spotlight-metrics,
      .fig-grid {{
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
    </section>

    <nav class="section-nav">
      <a class="nav-chip" href="#overview">Overview</a>
      <a class="nav-chip" href="#explorer">Explorer</a>
      <a class="nav-chip" href="#atlas">Signal Atlas</a>
      <a class="nav-chip" href="#evidence">Evidence Maps</a>
      <a class="nav-chip" href="#tables">Tables</a>
      <a class="nav-chip" href="#assets">Assets</a>
    </nav>

    <section class="grid">
      <article class="card card-strong span-8" id="overview">
        <div class="section-head">
          <div>
            <h2>Reviewer Checklist</h2>
            <p class="section-copy">These cards surface the first programs a reviewer will inspect: the strongest enrichment signal, the clearest curated anchor, the most defensible frontier program, and the sharpest context shift.</p>
          </div>
        </div>
        <div class="story-grid">{story_cards_html}</div>
      </article>

      <article class="card span-4">
        <div class="section-head">
          <div>
            <h2>Download Center</h2>
            <p class="section-copy">Direct access to the summary tables, claim dossiers, and figure exports used for manuscript assembly.</p>
          </div>
        </div>
        <div class="download-list">{download_html}</div>
      </article>

      <article class="card card-strong span-12" id="explorer">
        <div class="section-head">
          <div>
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
        <p class="hint">Click a bar, point, or heatmap tile to sync all program views. Search narrows the selector live. The spotlight card below exposes the selected program's evidence chain.</p>
      </article>

      <article class="card span-4">
        <div class="section-head">
          <div>
            <h2>Program Spotlight</h2>
            <p class="section-copy">Compact per-program evidence summary with claim support, context driver, and best curated anchor.</p>
          </div>
        </div>
        <div id="program-spotlight" class="spotlight"></div>
      </article>
      <article class="card span-4">
        <div class="section-head">
          <div>
            <h2>Headline Metrics</h2>
            <p class="section-copy">Study-level counts exported as the canonical headline summary.</p>
          </div>
        </div>
        <div class="table-wrap">{summary_html}</div>
      </article>
      <article class="card span-4">
        <div class="section-head">
          <div>
            <h2>Claim Gate Summary</h2>
            <p class="section-copy">Gate-level pass rates for manuscript-safe interpretation.</p>
          </div>
        </div>
        <div class="table-wrap">{gate_html}</div>
      </article>

      <article class="card span-6" id="atlas">
        <div class="section-head">
          <div>
            <h2>Differential Signal Volcano</h2>
            <p class="section-copy">Global gene-level effect and significance landscape.</p>
          </div>
        </div>
        <div id="plot-volcano" class="plot-sm"></div>
      </article>
      <article class="card span-6">
        <div class="section-head">
          <div>
            <h2>Program Size Landscape</h2>
            <p class="section-copy">Dynamic program footprint ranked by recovered gene membership.</p>
          </div>
        </div>
        <div id="plot-sizes" class="plot-sm"></div>
      </article>

      <article class="card span-7">
        <div class="section-head">
          <div>
            <h2>Enriched Programs</h2>
            <p class="section-copy">Programs prioritized by enrichment score and significance.</p>
          </div>
        </div>
        <div id="plot-gsea" class="plot"></div>
      </article>
      <article class="card span-5" id="evidence">
        <div class="section-head">
          <div>
            <h2>Program vs Reference Overlap</h2>
            <p class="section-copy">Curated anchor map for the top dynamic programs.</p>
          </div>
        </div>
        <div id="plot-heatmap" class="plot"></div>
      </article>

      <article class="card span-7">
        <div class="section-head">
          <div>
            <h2 id="context-title">Context evidence by gene</h2>
            <p class="section-copy">Genes ranked within the selected program by context-aware evidence.</p>
          </div>
        </div>
        <div id="plot-context" class="plot"></div>
      </article>
      <article class="card span-5">
        <div class="section-head">
          <div>
            <h2 id="context-table-title">Top Context-Evidence Genes</h2>
            <p class="section-copy">Reviewer-facing table for the currently selected program.</p>
          </div>
        </div>
        <div id="context-program-table" class="table-wrap"></div>
      </article>

      <article class="card span-7">
        <div class="section-head">
          <div>
            <h2 id="membership-title">Membership weights by gene</h2>
            <p class="section-copy">Base membership structure for the currently selected program.</p>
          </div>
        </div>
        <div id="plot-membership" class="plot"></div>
      </article>
      <article class="card span-5">
        <div class="section-head">
          <div>
            <h2 id="membership-table-title">Top Membership Genes</h2>
            <p class="section-copy">Top genes by base membership, color-coded by the active context metric.</p>
          </div>
        </div>
        <div id="membership-program-table" class="table-wrap"></div>
      </article>

      <article class="card span-4" id="tables">
        <div class="section-head">
          <div>
            <h2>Top Enriched Programs Table</h2>
            <p class="section-copy">Static export for the best enrichment hits.</p>
          </div>
        </div>
        <div class="table-wrap">{top_enriched_html}</div>
      </article>
      <article class="card span-4">
        <div class="section-head">
          <div>
            <h2>Global Context Table</h2>
            <p class="section-copy">Genes with the strongest context evidence across all programs.</p>
          </div>
        </div>
        <div class="table-wrap">{top_context_html}</div>
      </article>
      <article class="card span-4">
        <div class="section-head">
          <div>
            <h2>Global Membership Table</h2>
            <p class="section-copy">Top genes per dynamic program by base membership.</p>
          </div>
        </div>
        <div class="table-wrap">{gene_membership_html}</div>
      </article>

      <article class="card span-12" id="assets">
        <div class="section-head">
          <div>
            <h2>Figure Vault</h2>
            <p class="section-copy">Static figure exports kept in the same package as the interactive dashboard.</p>
          </div>
        </div>
        <div class="fig-grid">
          <img alt="Figure 1 Volcano" src="{fig_rel}/figure_1_volcano.png" />
          <img alt="Figure 2 Program Sizes" src="{fig_rel}/figure_2_program_sizes.png" />
          <img alt="Figure 3 Claim Gates" src="{fig_rel}/figure_3_claim_gates.png" />
          <img alt="Figure 4 Context Evidence" src="{fig_rel}/figure_4_context_shift.png" />
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

    function renderSimpleTable(targetId, rows, cols) {{
      const target = document.getElementById(targetId);
      if (!target) return;
      if (!rows || rows.length === 0) {{
        target.innerHTML = "<div style='padding:14px;color:#5a6b7f;'>No rows for selected program.</div>";
        return;
      }}
      const head = "<thead><tr>" + cols.map(c => `<th>${{esc(c)}}</th>`).join("") + "</tr></thead>";
      const body = "<tbody>" + rows.map(r => "<tr>" + cols.map(c => `<td>${{esc(r[c])}}</td>`).join("") + "</tr>").join("") + "</tbody>";
      target.innerHTML = `<table class="np-table">${{head}}${{body}}</table>`;
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
      renderContextForSelected();
      renderMembershipForSelected();
    }}

    function renderContextForSelected() {{
      const rows = (payload.context_program || [])
        .filter(r => r.program === selectedProgram)
        .sort((a, b) => metricAbsValue(b, selectedContextMetric) - metricAbsValue(a, selectedContextMetric))
        .slice(0, 35);
      const metricLabel = contextMetricLabels[selectedContextMetric] || "Context metric";
      const axisTitle = contextMetricAxis[selectedContextMetric] || metricLabel;
      const title = document.getElementById("context-title");
      if (title) title.textContent = metricLabel + " by gene - " + labelFor(selectedProgram);
      const tableTitle = document.getElementById("context-table-title");
      if (tableTitle) tableTitle.textContent = "Top " + metricLabel + " genes - " + labelFor(selectedProgram);
      if (rows.length === 0) {{
        Plotly.react("plot-context", [], {{
          margin: {{l: 90, r: 20, t: 10, b: 40}},
          annotations: [{{text: "No context rows", showarrow: false, xref: "paper", yref: "paper", x: 0.5, y: 0.5}}],
        }}, plotConfig);
        renderSimpleTable(
          "context-program-table",
          [],
          ["gene", "base_membership", "context_shift", "signed_significance", "context_evidence", "abs_rank_metric"]
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
          margin: {{l: 120, r: 30, t: 8, b: 40}},
          xaxis: {{title: axisTitle, zeroline: true, zerolinecolor: "#4d5d70", zerolinewidth: 1}},
          yaxis: {{autorange: "reversed", automargin: true}},
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
        }},
        plotConfig
      );
      renderSimpleTable(
        "context-program-table",
        rows.map(r => {{
          return {{
            gene: r.gene,
            base_membership: Number(r.base_membership).toFixed(3),
            context_shift: numOrZero(r.context_shift).toFixed(3),
            signed_significance: numOrZero(r.signed_significance).toFixed(3),
            context_evidence: numOrZero(r.context_evidence).toFixed(3),
            abs_rank_metric: metricAbsValue(r, selectedContextMetric).toFixed(3),
          }};
        }}),
        ["gene", "base_membership", "context_shift", "signed_significance", "context_evidence", "abs_rank_metric"]
      );
    }}

    function renderMembershipForSelected() {{
      const rows = (payload.membership_program || [])
        .filter(r => r.program === selectedProgram)
        .sort((a, b) => (b.base_membership ?? 0) - (a.base_membership ?? 0))
        .slice(0, 35);
      const title = document.getElementById("membership-title");
      if (title) title.textContent = "Membership weights by gene - " + labelFor(selectedProgram);
      const tableTitle = document.getElementById("membership-table-title");
      if (tableTitle) tableTitle.textContent = "Top Membership Genes - " + labelFor(selectedProgram);
      if (rows.length === 0) {{
        Plotly.react("plot-membership", [], {{
          margin: {{l: 90, r: 20, t: 10, b: 40}},
          annotations: [{{text: "No membership rows", showarrow: false, xref: "paper", yref: "paper", x: 0.5, y: 0.5}}],
        }}, plotConfig);
        renderSimpleTable(
          "membership-program-table",
          [],
          ["gene", "base_membership", "context_shift", "signed_significance", "context_evidence"]
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
          margin: {{l: 120, r: 30, t: 8, b: 40}},
          xaxis: {{title: "Base membership"}},
          yaxis: {{autorange: "reversed", automargin: true}},
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
        }},
        plotConfig
      );
      renderSimpleTable(
        "membership-program-table",
        rows.map(r => {{
          return {{
            gene: r.gene,
            base_membership: Number(r.base_membership).toFixed(3),
            context_shift: numOrZero(r.context_shift).toFixed(3),
            signed_significance: numOrZero(r.signed_significance).toFixed(3),
            context_evidence: numOrZero(r.context_evidence).toFixed(3),
          }};
        }}),
        ["gene", "base_membership", "context_shift", "signed_significance", "context_evidence"]
      );
    }}

    const volcano = payload.de;
    Plotly.newPlot(
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
        margin: {{l: 50, r: 20, t: 10, b: 45}},
        xaxis: {{title: "logFC (A - B)"}},
        yaxis: {{title: "-log10(p-value)"}},
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
      }},
      plotConfig
    );

    const sizes = payload.sizes || [];
    Plotly.newPlot(
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
        margin: {{l: 320, r: 20, t: 10, b: 45}},
        xaxis: {{title: "Gene count"}},
        yaxis: {{
          autorange: "reversed",
          automargin: true,
          tickmode: "array",
          tickvals: sizes.map(d => d.program),
          ticktext: sizes.map(d => wrapLabel(axisLabelFor(d.program), 42)),
          tickfont: {{size: 11}},
        }},
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
      }},
      plotConfig
    );
    const sizesEl = document.getElementById("plot-sizes");
    if (sizesEl) {{
      sizesEl.on("plotly_click", (ev) => {{
        const p = ev?.points?.[0]?.customdata;
        if (p) setProgram(p);
      }});
    }}

    const gsea = payload.gsea || [];
    Plotly.newPlot(
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
        margin: {{l: 320, r: 20, t: 10, b: 45}},
        xaxis: {{title: "NES"}},
        yaxis: {{
          autorange: "reversed",
          automargin: true,
          tickmode: "array",
          tickvals: gsea.map(d => d.program),
          ticktext: gsea.map(d => wrapLabel(axisLabelFor(d.program), 42)),
          tickfont: {{size: 11}},
        }},
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
      }},
      plotConfig
    );
    const gseaEl = document.getElementById("plot-gsea");
    if (gseaEl) {{
      gseaEl.on("plotly_click", (ev) => {{
        const p = ev?.points?.[0]?.customdata;
        if (p) setProgram(p);
      }});
    }}

    const heat = payload.heatmap || [];
    if (heat.length === 0) {{
      const el = document.getElementById("plot-heatmap");
      if (el) {{
        el.innerHTML = "<div style='padding:20px;color:#5a6b7f;'>No annotation overlap data found. Run with --annotate-programs.</div>";
      }}
    }} else {{
      const programs = [...new Set(heat.map(d => d.program))];
      const refs = [...new Set(heat.map(d => d.reference_name))];
      const z = programs.map(p => refs.map(r => {{
        const row = heat.find(x => x.program === p && x.reference_name === r);
        return row ? row.jaccard : 0.0;
      }}));
      const fullProgramGrid = programs.map(p => refs.map(() => fullLabelFor(p)));
      Plotly.newPlot(
        "plot-heatmap",
        [{{
          z: z,
          x: refs,
          y: programs,
          customdata: fullProgramGrid,
          type: "heatmap",
          colorscale: "YlOrRd",
          zmin: 0,
          zmax: Math.max(...z.flat()),
          colorbar: {{title: "Jaccard"}},
          hovertemplate: "Program=%{{customdata}}<br>Ref=%{{x}}<br>Jaccard=%{{z:.3f}}<extra></extra>",
        }}],
        {{
          margin: {{l: 360, r: 40, t: 10, b: 140}},
          xaxis: {{tickangle: -45}},
          yaxis: {{
            autorange: "reversed",
            automargin: true,
            tickmode: "array",
            tickvals: programs,
            ticktext: programs.map(p => wrapLabel(axisLabelFor(p), 50)),
            tickfont: {{size: 11}},
          }},
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
        }},
        plotConfig
      );
      const heatEl = document.getElementById("plot-heatmap");
      if (heatEl) {{
        heatEl.on("plotly_click", (ev) => {{
          const p = ev?.points?.[0]?.y;
          if (p) setProgram(p);
        }});
      }}
    }}

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
        return pd.DataFrame(columns=["program", "reference_name", "jaccard"])
    if "program" not in overlap_df.columns or "reference_name" not in overlap_df.columns:
        return pd.DataFrame(columns=["program", "reference_name", "jaccard"])

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

    return subset.loc[:, ["program", "reference_name", "jaccard"]].reset_index(drop=True)


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
    logger.info("Dashboard generated at %s", artifacts.html_path)
    return artifacts
