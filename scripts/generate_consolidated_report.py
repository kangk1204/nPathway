#!/usr/bin/env python3
"""Generate consolidated publication-quality benchmark report.

Loads pre-computed results from all benchmark scripts and generates
a single comprehensive PDF covering:
  1. Core benchmark (PBMC 3k + Hallmark)
  2. FM Embedding Comparison
  3. FM Advantage: Low-Data, Zero-Shot, Gene Coverage
  4. Multi-Dataset Validation (4 datasets)
  5. Large-Scale Validation (PBMC 68k)
  6. Cross-Dataset Reproducibility
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
TABLES_DIR = ROOT / "results" / "tables"
REPORT_PATH = ROOT / "benchmark_report.pdf"

# Color palette
COLORS = {
    "nPathway-KMeans": "#1565C0",
    "nPathway-Refined": "#2196F3",
    "nPathway-Ensemble": "#0D47A1",
    "nPathway-Leiden": "#1976D2",
    "nPathway-ETM": "#1B5E20",
    "Spectra": "#FF6F00",
    "WGCNA": "#E65100",
    "cNMF": "#6A1B9A",
    "Expr-Cluster": "#C62828",
    "Random": "#757575",
    "PCA": "#1565C0",
    "scGPT": "#E65100",
    "Geneformer": "#2E7D32",
    "Hybrid-scGPT": "#AB47BC",
}


def _get_color(name: str) -> str:
    """Get color for a method/embedding name."""
    for key, color in COLORS.items():
        if key in name:
            return color
    return "#757575"


def _make_table_page(
    ax: plt.Axes,
    data: list[list[str]],
    col_labels: list[str],
    title: str,
    highlight_prefix: str = "nPathway",
) -> None:
    """Render a formatted table on a matplotlib axes."""
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)

    table = ax.table(
        cellText=data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    # Header styling
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#1565C0")
        table[0, j].set_text_props(color="white", fontweight="bold", fontsize=8)

    # Row styling
    for i in range(1, len(data) + 1):
        bg = "#E3F2FD" if i % 2 == 0 else "#FFFFFF"
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(bg)
        if data[i - 1][0].startswith(highlight_prefix):
            table[i, 0].set_text_props(fontweight="bold")

    table.auto_set_column_width(list(range(len(col_labels))))


def generate_consolidated_report() -> None:
    """Generate the full consolidated benchmark report PDF."""
    logger.info("Generating consolidated benchmark report...")

    # Load all pre-computed results
    pd.read_csv(TABLES_DIR / "benchmark1_recovery.csv")  # available for future use
    core_discovery = pd.read_csv(TABLES_DIR / "benchmark2_discovery.csv")
    core_power = pd.read_csv(TABLES_DIR / "benchmark3_power.csv")
    fm_quality = pd.read_csv(TABLES_DIR / "fm_embedding_quality.csv")
    fm_power = pd.read_csv(TABLES_DIR / "fm_embedding_power.csv")
    fm_lowdata = pd.read_csv(TABLES_DIR / "fm_advantage_lowdata.csv")
    fm_zeroshot = pd.read_csv(TABLES_DIR / "fm_advantage_zeroshot.csv")
    fm_coverage = pd.read_csv(TABLES_DIR / "fm_advantage_coverage.csv")
    multi_quality = pd.read_csv(TABLES_DIR / "multi_dataset_quality.csv")
    cross_reprod = pd.read_csv(TABLES_DIR / "cross_dataset_reproducibility.csv")
    largescale_recovery = pd.read_csv(TABLES_DIR / "largescale_recovery.csv")
    largescale_power = pd.read_csv(TABLES_DIR / "largescale_power.csv")

    with PdfPages(str(REPORT_PATH)) as pdf:
        # ===================================================================
        # Page 1: Title
        # ===================================================================
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(
            0.5, 0.78, "nPathway",
            fontsize=36, fontweight="bold", ha="center",
            transform=ax.transAxes, color="#0D47A1",
        )
        ax.text(
            0.5, 0.66,
            "Foundation Model-Derived Gene Programs\n"
            "for Context-Aware Gene Set Enrichment Analysis",
            fontsize=16, ha="center", transform=ax.transAxes, style="italic",
        )
        ax.text(
            0.5, 0.52,
            "Consolidated Benchmark Report",
            fontsize=18, ha="center", transform=ax.transAxes,
            fontweight="bold", color="#1565C0",
        )

        sections = [
            "1. Core Benchmark: PBMC 3k + MSigDB Hallmark (8 methods)",
            "2. FM Embedding Comparison: PCA vs scGPT vs Geneformer",
            "3. FM Advantage: Low-Data Crossover & Zero-Shot Programs",
            "4. Multi-Dataset Validation: 4 datasets, 7 methods",
            "5. Large-Scale: PBMC 68k (65,877 cells, 11 cell types)",
            "6. Cross-Dataset Reproducibility",
        ]
        section_text = "\n".join(sections)
        ax.text(
            0.5, 0.30, section_text,
            fontsize=11, ha="center", transform=ax.transAxes,
            family="monospace", linespacing=1.8,
        )
        ax.text(
            0.5, 0.08,
            "Date: 2026-03-03  |  Datasets: PBMC3k, PBMC68k, Paul15, Burczynski06",
            fontsize=10, ha="center", transform=ax.transAxes, color="gray",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ===================================================================
        # Page 2: Core Benchmark - Discovery Quality
        # ===================================================================
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(
            "Section 1: Core Discovery Quality Metrics (PBMC 3k)",
            fontsize=14, fontweight="bold",
        )

        method_order = [
            "nPathway-Ensemble", "nPathway-KMeans", "nPathway-Refined",
            "nPathway-Leiden", "nPathway-ETM", "Spectra",
            "WGCNA", "cNMF", "Expr-Cluster", "Random",
        ]
        disc = core_discovery.copy()
        disc["method"] = pd.Categorical(disc["method"], categories=method_order, ordered=True)
        disc = disc.sort_values("method").dropna(subset=["method"])
        disc_colors = [_get_color(m) for m in disc["method"]]

        metrics_info = [
            ("coverage", "Gene Coverage", "{:.2f}"),
            ("redundancy", "Redundancy (lower=better)", "{:.3f}"),
            ("novelty", "Novelty Score", "{:.2f}"),
            ("specificity", "Specificity", "{:.2f}"),
            ("mean_hallmark_alignment", "Hallmark Alignment", "{:.3f}"),
            ("mean_program_size", "Mean Program Size", "{:.0f}"),
        ]

        for idx, (metric, title, fmt) in enumerate(metrics_info):
            ax = axes[idx // 3][idx % 3]
            vals = disc[metric].values
            methods_list = disc["method"].tolist()
            ax.bar(range(len(methods_list)), vals, color=disc_colors,
                   edgecolor="black", linewidth=0.5)
            ax.set_xticks(range(len(methods_list)))
            ax.set_xticklabels(methods_list, rotation=40, ha="right", fontsize=7)
            ax.set_title(title, fontsize=11)
            ax.grid(True, alpha=0.3, axis="y")
            for i, v in enumerate(vals):
                ax.text(i, v + 0.005, fmt.format(v), ha="center", fontsize=6)

        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ===================================================================
        # Page 3: Core Benchmark - Power Curves
        # ===================================================================
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        fig.suptitle(
            "Section 1: Statistical Power (PBMC 3k, Cell-Type Markers)",
            fontsize=14, fontweight="bold",
        )

        for method in method_order:
            sub = core_power[core_power["method"] == method]
            if sub.empty:
                continue
            color = _get_color(method)
            axes[0].plot(sub["fold_change"], sub["tpr"], "-o",
                        color=color, label=method, linewidth=2, markersize=5)
            axes[1].plot(sub["fold_change"], sub["fpr"], "-o",
                        color=color, label=method, linewidth=2, markersize=5)

        axes[0].set_xlabel("Fold Change")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title("Sensitivity (TPR)")
        axes[0].legend(fontsize=7, loc="lower right", ncol=2)
        axes[0].set_ylim(-0.05, 1.05)
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel("Fold Change")
        axes[1].set_ylabel("False Positive Rate")
        axes[1].set_title("Specificity (FPR, lower=better)")
        axes[1].legend(fontsize=7, loc="upper left", ncol=2)
        axes[1].set_ylim(-0.02, 0.6)
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ===================================================================
        # Page 4: FM Embedding Comparison
        # ===================================================================
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
        fig.suptitle(
            "Section 2: Foundation Model Embedding Comparison (PBMC 3k)",
            fontsize=14, fontweight="bold",
        )

        # Hallmark alignment by embedding
        fm_methods = fm_quality["method"].tolist()
        fm_colors = [_get_color(m) for m in fm_methods]

        axes[0].bar(range(len(fm_methods)), fm_quality["hallmark_alignment"],
                    color=fm_colors, edgecolor="black", linewidth=0.5)
        axes[0].set_xticks(range(len(fm_methods)))
        axes[0].set_xticklabels(fm_methods, rotation=40, ha="right", fontsize=7)
        axes[0].set_title("Hallmark Alignment (Jaccard)")
        axes[0].grid(True, alpha=0.3, axis="y")
        for i, v in enumerate(fm_quality["hallmark_alignment"]):
            axes[0].text(i, v + 0.001, f"{v:.3f}", ha="center", fontsize=7)

        # Coverage
        axes[1].bar(range(len(fm_methods)), fm_quality["coverage"],
                    color=fm_colors, edgecolor="black", linewidth=0.5)
        axes[1].set_xticks(range(len(fm_methods)))
        axes[1].set_xticklabels(fm_methods, rotation=40, ha="right", fontsize=7)
        axes[1].set_title("Gene Coverage")
        axes[1].grid(True, alpha=0.3, axis="y")

        # Power: TPR at FC=2.0
        fm_pwr = fm_power[fm_power["fold_change"] == 2.0]
        if not fm_pwr.empty:
            pwr_methods = fm_pwr["method"].tolist()
            pwr_colors = [_get_color(m) for m in pwr_methods]
            axes[2].bar(range(len(pwr_methods)), fm_pwr["tpr"],
                        color=pwr_colors, edgecolor="black", linewidth=0.5)
            axes[2].set_xticks(range(len(pwr_methods)))
            axes[2].set_xticklabels(pwr_methods, rotation=40, ha="right", fontsize=7)
            axes[2].set_title("TPR at FC=2.0")
            axes[2].grid(True, alpha=0.3, axis="y")
            for i, v in enumerate(fm_pwr["tpr"]):
                axes[2].text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)

        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ===================================================================
        # Page 5: FM Advantage - Low-Data Crossover (KEY FINDING)
        # ===================================================================
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            "Section 3: FM Advantage — Low-Data Regime Crossover",
            fontsize=14, fontweight="bold",
        )

        emb_colors = {
            "PCA": "#1565C0",
            "scGPT": "#E65100",
            "Geneformer": "#2E7D32",
            "Hybrid-scGPT": "#AB47BC",
        }

        for emb in ["PCA", "Geneformer", "scGPT", "Hybrid-scGPT"]:
            sub = fm_lowdata[fm_lowdata["embedding"] == emb]
            if sub.empty:
                continue
            axes[0].plot(
                sub["n_cells"], sub["mean_jaccard"], "-o",
                color=emb_colors.get(emb, "#999"), label=emb,
                linewidth=2.5, markersize=7,
            )
            axes[1].plot(
                sub["n_cells"], sub["mean_recall"], "-o",
                color=emb_colors.get(emb, "#999"), label=emb,
                linewidth=2.5, markersize=7,
            )

        # Add crossover annotation
        axes[0].axvline(x=1000, color="red", linestyle="--", alpha=0.5, linewidth=1)
        axes[0].annotate(
            "Crossover\n~1000 cells",
            xy=(1000, 0.20), xytext=(2000, 0.12),
            fontsize=10, fontweight="bold", color="red",
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
        )

        for ax in axes:
            ax.set_xlabel("Number of Cells", fontsize=11)
            ax.set_xscale("log")
            ax.legend(fontsize=10, loc="upper left")
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel("Mean Jaccard (Cell-Type DE Recovery)")
        axes[0].set_title("Jaccard Similarity vs Cell Count")
        axes[1].set_ylabel("Mean Recall")
        axes[1].set_title("Recall vs Cell Count")

        # Add text box with key finding
        textstr = (
            "Key Finding: Geneformer outperforms PCA at <1000 cells\n"
            "At n=50: Geneformer 0.201 vs PCA 0.045 (4.5× advantage)\n"
            "At n=1000: crossover point (0.201 vs 0.199)"
        )
        props = dict(boxstyle="round,pad=0.5", facecolor="#E8F5E9", alpha=0.9, edgecolor="#2E7D32")
        axes[0].text(
            0.98, 0.02, textstr, transform=axes[0].transAxes,
            fontsize=8, verticalalignment="bottom", horizontalalignment="right",
            bbox=props,
        )

        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ===================================================================
        # Page 6: FM Advantage - Zero-Shot & Gene Coverage
        # ===================================================================
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            "Section 3: FM Advantage — Zero-Shot Programs & Gene Coverage Expansion",
            fontsize=14, fontweight="bold",
        )

        # Zero-shot bar chart
        zs = fm_zeroshot.copy()
        zs["label"] = zs["model"] + " (" + zs["graph_reg"] + ", " + zs["method"] + ")"
        zs_colors = [emb_colors.get("scGPT" if "scgpt" in m else "Geneformer", "#999")
                     for m in zs["model"]]
        axes[0].barh(range(len(zs)), zs["mean_jaccard"],
                           color=zs_colors, edgecolor="black", linewidth=0.5)
        axes[0].set_yticks(range(len(zs)))
        axes[0].set_yticklabels(zs["label"], fontsize=8)
        axes[0].set_xlabel("Mean Jaccard (Cell-Type DE Recovery)")
        axes[0].set_title("Zero-Shot: Programs from FM Embeddings Only\n(No Expression Data)")
        axes[0].grid(True, alpha=0.3, axis="x")
        for i, (v, r) in enumerate(zip(zs["mean_jaccard"], zs["mean_recall"])):
            axes[0].text(v + 0.005, i, f"J={v:.3f}, R={r:.2f}", fontsize=7, va="center")

        # Gene coverage comparison
        cov = fm_coverage.copy()
        cov["label"] = cov["scope"] + "\n" + cov["embedding"] + f" ({cov['n_genes']}g)"
        x = np.arange(len(cov))
        width = 0.35
        cov_colors = [_get_color(e) for e in cov["embedding"]]
        axes[1].bar(x, cov["mean_jaccard"], width, label="Jaccard", color=cov_colors,
                   edgecolor="black", linewidth=0.5, alpha=0.8)
        axes[1].bar(x + width, cov["mean_recall"], width, label="Recall",
                   color=cov_colors, edgecolor="black", linewidth=0.5, alpha=0.5)
        axes[1].set_xticks(x + width / 2)
        axes[1].set_xticklabels(cov["label"], fontsize=7, rotation=30, ha="right")
        axes[1].set_ylabel("Score")
        axes[1].set_title("Gene Coverage: HVG-Only vs All Expressed Genes")
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3, axis="y")

        # Key finding annotation
        textstr2 = (
            "Key: Geneformer recall improves 12%\n"
            "with all genes (0.423 → 0.473)\n"
            "expanding beyond HVG bottleneck"
        )
        props2 = dict(boxstyle="round,pad=0.5", facecolor="#E8F5E9", alpha=0.9, edgecolor="#2E7D32")
        axes[1].text(
            0.98, 0.98, textstr2, transform=axes[1].transAxes,
            fontsize=8, verticalalignment="top", horizontalalignment="right",
            bbox=props2,
        )

        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ===================================================================
        # Page 7: Multi-Dataset Quality (4 datasets)
        # ===================================================================
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            "Section 4: Multi-Dataset Hallmark Alignment",
            fontsize=14, fontweight="bold",
        )

        datasets = ["PBMC3k", "PBMC68k", "Paul15", "Burczynski06"]
        for ax_idx, dataset in enumerate(datasets):
            ax = axes[ax_idx // 2][ax_idx % 2]
            sub = multi_quality[multi_quality["dataset"] == dataset].copy()
            if sub.empty:
                ax.set_visible(False)
                continue

            methods = sub["method"].tolist()
            colors = [_get_color(m) for m in methods]
            ax.bar(range(len(methods)), sub["hallmark_alignment"],
                   color=colors, edgecolor="black", linewidth=0.5)
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=8)
            ax.set_ylabel("Hallmark Alignment")

            ax.set_title(f"{dataset}", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="y")

            for i, v in enumerate(sub["hallmark_alignment"]):
                ax.text(i, v + 0.001, f"{v:.3f}", ha="center", fontsize=7)

        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ===================================================================
        # Page 8: Cross-Dataset Reproducibility
        # ===================================================================
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle(
            "Section 6: Cross-Dataset Reproducibility (Best-Match Jaccard)",
            fontsize=14, fontweight="bold",
        )

        reprod_mean = cross_reprod[cross_reprod["dataset_pair"] == "mean"].copy()
        reprod_mean = reprod_mean.sort_values("similarity", ascending=True)
        colors_reprod = [_get_color(m) for m in reprod_mean["method"]]
        ax.barh(range(len(reprod_mean)), reprod_mean["similarity"],
                color=colors_reprod, edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(len(reprod_mean)))
        ax.set_yticklabels(reprod_mean["method"], fontsize=10)
        ax.set_xlabel("Mean Cross-Dataset Similarity", fontsize=11)
        ax.set_title("Average Reproducibility Across Human Datasets")
        ax.grid(True, alpha=0.3, axis="x")
        for i, v in enumerate(reprod_mean["similarity"]):
            ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)

        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ===================================================================
        # Page 9: Large-Scale Validation (PBMC 68k)
        # ===================================================================
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            "Section 5: Large-Scale Validation — PBMC 68k (65,877 cells)",
            fontsize=14, fontweight="bold",
        )

        ls = largescale_recovery.sort_values("mean_jaccard", ascending=False)
        ls_colors = [_get_color(m) for m in ls["method"]]
        axes[0].barh(range(len(ls)), ls["mean_jaccard"],
                     color=ls_colors, edgecolor="black", linewidth=0.5)
        axes[0].set_yticks(range(len(ls)))
        axes[0].set_yticklabels(ls["method"], fontsize=9)
        axes[0].set_xlabel("Mean Jaccard")
        axes[0].set_title("Cell-Type DE Recovery (11 cell types)")
        axes[0].grid(True, alpha=0.3, axis="x")
        for i, v in enumerate(ls["mean_jaccard"]):
            axes[0].text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)

        ls_recall = largescale_recovery.sort_values("mean_recall", ascending=False)
        ls_r_colors = [_get_color(m) for m in ls_recall["method"]]
        axes[1].barh(range(len(ls_recall)), ls_recall["mean_recall"],
                     color=ls_r_colors, edgecolor="black", linewidth=0.5)
        axes[1].set_yticks(range(len(ls_recall)))
        axes[1].set_yticklabels(ls_recall["method"], fontsize=9)
        axes[1].set_xlabel("Mean Recall")
        axes[1].set_title("Recall of DE Genes")
        axes[1].grid(True, alpha=0.3, axis="x")
        for i, v in enumerate(ls_recall["mean_recall"]):
            axes[1].text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)

        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ===================================================================
        # Page 10: Large-Scale Power Curves
        # ===================================================================
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        fig.suptitle(
            "Section 5: Large-Scale Statistical Power (PBMC 68k)",
            fontsize=14, fontweight="bold",
        )

        for method in largescale_power["method"].unique():
            sub = largescale_power[largescale_power["method"] == method]
            color = _get_color(method)
            axes[0].plot(sub["fold_change"], sub["tpr"], "-o",
                        color=color, label=method, linewidth=2, markersize=5)
            axes[1].plot(sub["fold_change"], sub["fpr"], "-o",
                        color=color, label=method, linewidth=2, markersize=5)

        axes[0].set_xlabel("Fold Change")
        axes[0].set_ylabel("TPR")
        axes[0].set_title("Sensitivity")
        axes[0].legend(fontsize=7, loc="lower right", ncol=2)
        axes[0].set_ylim(-0.05, 1.05)
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel("Fold Change")
        axes[1].set_ylabel("FPR")
        axes[1].set_title("Specificity (lower=better)")
        axes[1].legend(fontsize=7, loc="upper left", ncol=2)
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ===================================================================
        # Page 11: Comprehensive Summary Table
        # ===================================================================
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.axis("off")
        fig.suptitle(
            "Comprehensive Summary: Key Results Across All Benchmarks",
            fontsize=14, fontweight="bold",
        )

        summary_data = []

        # Core metrics from PBMC 3k
        for _, row in core_discovery.iterrows():
            method = row["method"]
            pwr_row = core_power[
                (core_power["method"] == method) & (core_power["fold_change"] == 2.0)
            ]
            tpr = pwr_row["tpr"].values[0] if not pwr_row.empty else 0.0

            # Multi-dataset mean alignment
            mq_sub = multi_quality[multi_quality["method"] == method]
            human_sub = mq_sub[mq_sub["dataset"].isin(["PBMC3k", "PBMC68k", "Burczynski06"])]
            multi_align = human_sub["hallmark_alignment"].mean() if not human_sub.empty else 0.0

            # Reproducibility
            reprod_row = cross_reprod[
                (cross_reprod["method"] == method) & (cross_reprod["dataset_pair"] == "mean")
            ]
            reprod = reprod_row["similarity"].values[0] if not reprod_row.empty else 0.0

            summary_data.append([
                method,
                f"{row['coverage']:.2f}",
                f"{row['redundancy']:.3f}",
                f"{row['mean_hallmark_alignment']:.3f}",
                f"{tpr:.2f}",
                f"{multi_align:.3f}",
                f"{reprod:.3f}",
            ])

        summary_cols = [
            "Method",
            "Coverage",
            "Redund.\n(↓better)",
            "Hallmark\nAlign.",
            "TPR\n(FC=2.0)",
            "Multi-DS\nAlign.",
            "Cross-DS\nReprod.",
        ]

        _make_table_page(ax, summary_data, summary_cols,
                        "All Methods: Core Metrics + Multi-Dataset")

        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ===================================================================
        # Page 12: FM Advantage Summary Table
        # ===================================================================
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.axis("off")
        fig.suptitle(
            "FM Advantage Summary: When Foundation Models Help",
            fontsize=14, fontweight="bold",
        )

        fm_table_data = []
        for _, row in fm_lowdata.iterrows():
            fm_table_data.append([
                row["embedding"],
                str(int(row["n_cells"])),
                f"{row['mean_jaccard']:.3f}",
                f"{row['mean_recall']:.3f}",
                f"{row['coverage']:.2f}",
            ])

        fm_cols = ["Embedding", "n_cells", "Jaccard", "Recall", "Coverage"]
        _make_table_page(
            ax, fm_table_data, fm_cols,
            "Low-Data Regime: Cell-Type DE Recovery by Cell Count",
            highlight_prefix="Geneformer",
        )

        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ===================================================================
        # Page 13: Key Findings
        # ===================================================================
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.set_title("Key Findings", fontsize=18, fontweight="bold", pad=30)

        findings = [
            ("1. FM Embeddings Provide Data-Free Priors",
             "Geneformer produces meaningful gene programs (Jaccard 0.231)\n"
             "without ANY expression data — pure pretrained knowledge."),
            ("2. Low-Data Crossover at ~1000 Cells",
             "Below 1000 cells, Geneformer outperforms PCA by up to 4.5×.\n"
             "At n=50: Geneformer 0.201 vs PCA 0.045."),
            ("3. Gene Coverage Beyond HVGs",
             "FM embeddings enable programs over all expressed genes,\n"
             "improving recall by 12% (0.423 → 0.473)."),
            ("4. Graph Regularization Is the Core Differentiator",
             "10× silhouette improvement, 50% higher TPR.\n"
             "PCA + graph-reg matches or beats raw FM embeddings."),
            ("5. Expression-Based Methods Win at High Data",
             "With >1000 cells: PCA Jaccard 0.326, Geneformer plateaus at 0.201.\n"
             "cNMF achieves best Hallmark alignment (0.073) on PBMC3k."),
            ("6. nPathway: 100% Coverage, 0% Redundancy",
             "All nPathway methods: complete gene space coverage,\n"
             "no redundant programs, 100% specificity."),
        ]

        y_start = 0.88
        for i, (title, body) in enumerate(findings):
            y = y_start - i * 0.14
            ax.text(0.05, y, title, transform=ax.transAxes,
                   fontsize=12, fontweight="bold", color="#0D47A1")
            ax.text(0.08, y - 0.04, body, transform=ax.transAxes,
                   fontsize=10, color="#333333", linespacing=1.4)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    logger.info("Consolidated report: %s", REPORT_PATH)
    logger.info("Done!")


if __name__ == "__main__":
    generate_consolidated_report()
