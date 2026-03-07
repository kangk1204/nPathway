#!/usr/bin/env python3
"""Multi-dataset benchmark for nPathway.

Evaluates nPathway vs baselines across multiple scRNA-seq datasets:
  - PBMC 3k (10x, immune cells)
  - PBMC 68k reduced (10x, immune cells, larger)
  - Paul15 (myeloid progenitor differentiation)
  - Burczynski06 (ulcerative colitis, bulk microarray)

Also computes cross-dataset reproducibility (Task #22):
  Programs discovered on one dataset are evaluated on another
  to test generalization.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from npathway.data.datasets import (
    filter_gene_sets_to_adata,
    load_burczynski06,
    load_msigdb_gene_sets,
    load_paul15,
    load_pbmc3k,
    load_pbmc68k_reduced,
)
from npathway.data.preprocessing import (
    _safe_toarray,
    build_gene_embeddings_from_expression,
    build_graph_regularized_embeddings,
    preprocess_adata,
)
from npathway.discovery.baselines import (
    CNMFProgramDiscovery,
    ExpressionClusteringBaseline,
    RandomProgramDiscovery,
    WGCNAProgramDiscovery,
)
from npathway.discovery.clustering import ClusteringProgramDiscovery
from npathway.discovery.ensemble import EnsembleProgramDiscovery
from npathway.evaluation.metrics import (
    compute_overlap_matrix,
    coverage,
    jaccard_similarity,
    novelty_score,
    program_redundancy,
    program_specificity,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "results"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = Path(__file__).parent.parent / "benchmark_multi_dataset.pdf"

N_PROGRAMS = 20
TOP_N_GENES = 50
SEED = 42

METHOD_COLORS: dict[str, str] = {
    "nPathway-KMeans": "#1565C0",
    "nPathway-Refined": "#2196F3",
    "nPathway-Ensemble": "#0D47A1",
    "WGCNA": "#E65100",
    "cNMF": "#6A1B9A",
    "Expr-Cluster": "#C62828",
    "Random": "#757575",
}


# ===========================================================================
# Dataset loading
# ===========================================================================

def load_datasets() -> dict[str, tuple]:
    """Load and preprocess all benchmark datasets.

    Returns dict mapping dataset_name -> (adata, graph_emb, gene_names, hallmark).
    """
    datasets: dict[str, tuple] = {}
    hallmark_raw = load_msigdb_gene_sets(collection="hallmark")

    # 1. PBMC 3k
    logger.info("Loading PBMC 3k...")
    adata_pbmc3k = load_pbmc3k(preprocessed=True)
    graph_emb_pbmc3k, gn_pbmc3k = build_graph_regularized_embeddings(
        adata_pbmc3k, n_components=50, k_neighbors=15, n_diffusion_steps=3, alpha=0.5,
    )
    hallmark_pbmc3k = filter_gene_sets_to_adata(hallmark_raw, adata_pbmc3k, min_genes=3)
    datasets["PBMC3k"] = (adata_pbmc3k, graph_emb_pbmc3k, gn_pbmc3k, hallmark_pbmc3k)
    logger.info("  PBMC3k: %d cells x %d genes, %d Hallmark sets",
                adata_pbmc3k.n_obs, adata_pbmc3k.n_vars, len(hallmark_pbmc3k))

    # 2. PBMC 68k reduced (already preprocessed + scaled in scanpy)
    logger.info("Loading PBMC 68k reduced...")
    adata_68k = load_pbmc68k_reduced()
    # Already preprocessed: has HVGs, scaled .X, louvain labels
    graph_emb_68k, gn_68k = build_graph_regularized_embeddings(
        adata_68k, n_components=min(50, adata_68k.n_vars - 1),
        k_neighbors=15, n_diffusion_steps=3, alpha=0.5,
    )
    hallmark_68k = filter_gene_sets_to_adata(hallmark_raw, adata_68k, min_genes=3)
    datasets["PBMC68k"] = (adata_68k, graph_emb_68k, gn_68k, hallmark_68k)
    logger.info("  PBMC68k: %d cells x %d genes, %d Hallmark sets",
                adata_68k.n_obs, adata_68k.n_vars, len(hallmark_68k))

    # 3. Paul15 (myeloid differentiation)
    logger.info("Loading Paul15...")
    adata_paul = load_paul15()
    adata_paul = preprocess_adata(adata_paul, n_top_genes=min(2000, adata_paul.n_vars))
    graph_emb_paul, gn_paul = build_graph_regularized_embeddings(
        adata_paul, n_components=min(50, adata_paul.n_vars - 1),
        k_neighbors=15, n_diffusion_steps=3, alpha=0.5,
    )
    hallmark_paul = filter_gene_sets_to_adata(hallmark_raw, adata_paul, min_genes=3)
    datasets["Paul15"] = (adata_paul, graph_emb_paul, gn_paul, hallmark_paul)
    logger.info("  Paul15: %d cells x %d genes, %d Hallmark sets",
                adata_paul.n_obs, adata_paul.n_vars, len(hallmark_paul))

    # 4. Burczynski06 (ulcerative colitis, bulk)
    logger.info("Loading Burczynski06...")
    adata_burc = load_burczynski06(max_genes=3000)
    adata_burc = preprocess_adata(adata_burc, n_top_genes=min(2000, adata_burc.n_vars))
    graph_emb_burc, gn_burc = build_graph_regularized_embeddings(
        adata_burc, n_components=min(50, adata_burc.n_vars - 1),
        k_neighbors=min(15, adata_burc.n_obs - 1), n_diffusion_steps=3, alpha=0.5,
    )
    hallmark_burc = filter_gene_sets_to_adata(hallmark_raw, adata_burc, min_genes=3)
    datasets["Burczynski06"] = (adata_burc, graph_emb_burc, gn_burc, hallmark_burc)
    logger.info("  Burczynski06: %d cells x %d genes, %d Hallmark sets",
                adata_burc.n_obs, adata_burc.n_vars, len(hallmark_burc))

    return datasets


# ===========================================================================
# Run discovery methods on a single dataset
# ===========================================================================

def discover_methods_for_dataset(
    adata: object,
    graph_reg_embeddings: np.ndarray,
    gene_names: list[str],
) -> dict[str, dict[str, list[str]]]:
    """Run core discovery methods on a single dataset."""
    all_methods: dict[str, dict[str, list[str]]] = {}
    X_expr = _safe_toarray(adata.X).astype(np.float64)  # type: ignore[union-attr]
    X_nonneg = np.clip(X_expr, 0, None) + 1e-6

    # nPathway-KMeans
    km = ClusteringProgramDiscovery(
        method="kmeans", n_programs=N_PROGRAMS, random_state=SEED
    )
    km.fit(graph_reg_embeddings, gene_names)
    all_methods["nPathway-KMeans"] = km.get_programs()

    # nPathway-Refined (top N genes by combined confidence)
    km_scores = km.get_program_scores()
    km_confidence = km.get_gene_confidence()
    refined: dict[str, list[str]] = {}
    for prog_name, scored_genes in km_scores.items():
        conf = km_confidence.get(prog_name, {})
        combined: list[tuple[str, float]] = []
        for gene, cosine_score in scored_genes:
            conf_score = conf.get(gene, 0.5)
            combined.append((gene, (cosine_score * conf_score) ** 0.5))
        combined.sort(key=lambda x: x[1], reverse=True)
        refined[prog_name] = [g for g, _s in combined[:TOP_N_GENES]]
    all_methods["nPathway-Refined"] = refined

    # nPathway-Ensemble
    try:
        ensemble = EnsembleProgramDiscovery(
            methods=[
                ClusteringProgramDiscovery(
                    method="kmeans", n_programs=N_PROGRAMS, random_state=SEED
                ),
                ClusteringProgramDiscovery(
                    method="leiden", random_state=SEED + 1
                ),
                ClusteringProgramDiscovery(
                    method="spectral", n_programs=N_PROGRAMS, random_state=SEED + 2
                ),
            ],
            consensus_method="leiden",
            resolution=1.0,
            threshold_quantile=0.3,
            min_program_size=5,
            random_state=SEED,
        )
        ensemble.fit(graph_reg_embeddings, gene_names)
        all_methods["nPathway-Ensemble"] = ensemble.get_programs()
    except Exception as exc:
        logger.warning("Ensemble failed: %s", exc)

    # WGCNA
    wgcna = WGCNAProgramDiscovery(
        n_programs=N_PROGRAMS, soft_power=6, min_module_size=5, random_state=SEED,
    )
    wgcna.fit(X_expr, gene_names)
    all_methods["WGCNA"] = wgcna.get_programs()

    # cNMF
    cnmf = CNMFProgramDiscovery(
        n_programs=N_PROGRAMS, n_iter=5, top_n_genes=TOP_N_GENES, random_state=SEED,
    )
    cnmf.fit(X_nonneg, gene_names)
    all_methods["cNMF"] = cnmf.get_programs()

    # Expression Clustering
    expr_cl = ExpressionClusteringBaseline(
        n_programs=N_PROGRAMS, random_state=SEED,
    )
    expr_cl.fit(X_expr, gene_names)
    all_methods["Expr-Cluster"] = expr_cl.get_programs()

    # Random
    rand = RandomProgramDiscovery(
        n_programs=N_PROGRAMS, genes_per_program=TOP_N_GENES, random_state=SEED,
    )
    pca_emb, _ = build_gene_embeddings_from_expression(
        adata, n_components=min(50, len(gene_names) - 1)  # type: ignore[arg-type]
    )
    rand.fit(pca_emb, gene_names)
    all_methods["Random"] = rand.get_programs()

    return all_methods


# ===========================================================================
# Evaluate discovery quality
# ===========================================================================

def evaluate_quality(
    programs: dict[str, list[str]],
    hallmark: dict[str, list[str]],
    gene_names: list[str],
) -> dict[str, float]:
    """Compute quality metrics for a single method on one dataset."""
    cov = coverage(programs, gene_names)
    red = program_redundancy(programs)
    nov = novelty_score(programs, hallmark)
    spec = program_specificity(programs)
    mean_size = float(np.mean([len(v) for v in programs.values()])) if programs else 0.0

    alignment_scores = []
    for prog_genes in programs.values():
        prog_set = set(prog_genes)
        best_jac = 0.0
        for ref_genes in hallmark.values():
            jac = jaccard_similarity(prog_set, set(ref_genes))
            if jac > best_jac:
                best_jac = jac
        alignment_scores.append(best_jac)
    mean_alignment = float(np.mean(alignment_scores)) if alignment_scores else 0.0

    return {
        "coverage": cov,
        "redundancy": red,
        "novelty": nov,
        "specificity": spec,
        "mean_program_size": mean_size,
        "hallmark_alignment": mean_alignment,
        "n_programs": len(programs),
    }


# ===========================================================================
# Cross-dataset reproducibility (Task #22)
# ===========================================================================

def cross_dataset_reproducibility(
    all_dataset_programs: dict[str, dict[str, dict[str, list[str]]]],
) -> pd.DataFrame:
    """Compute cross-dataset program overlap for each method.

    For each method, compute pairwise similarity between programs
    discovered on different datasets. Higher = more reproducible.
    """
    logger.info("=== Cross-Dataset Reproducibility ===")
    dataset_names = list(all_dataset_programs.keys())
    rows = []

    for method_name in METHOD_COLORS:
        pair_scores = []
        for i, ds_i in enumerate(dataset_names):
            progs_i = all_dataset_programs[ds_i].get(method_name)
            if not progs_i:
                continue
            for j, ds_j in enumerate(dataset_names):
                if i >= j:
                    continue
                progs_j = all_dataset_programs[ds_j].get(method_name)
                if not progs_j:
                    continue

                # Find shared genes between datasets
                genes_i = {g for gl in progs_i.values() for g in gl}
                genes_j = {g for gl in progs_j.values() for g in gl}
                shared = genes_i & genes_j

                if len(shared) < 10:
                    # Skip pairs with insufficient shared genes (e.g., cross-species)
                    continue

                # Filter programs to shared genes only
                progs_i_filtered = {
                    k: [g for g in v if g in shared] for k, v in progs_i.items()
                }
                progs_j_filtered = {
                    k: [g for g in v if g in shared] for k, v in progs_j.items()
                }
                # Remove empty programs
                progs_i_filtered = {k: v for k, v in progs_i_filtered.items() if v}
                progs_j_filtered = {k: v for k, v in progs_j_filtered.items() if v}

                if not progs_i_filtered or not progs_j_filtered:
                    pair_scores.append(0.0)
                    continue

                ov = compute_overlap_matrix(progs_i_filtered, progs_j_filtered)
                # Mean of best-match Jaccard (symmetric)
                sim_ij = float(ov.max(axis=1).mean())
                sim_ji = float(ov.max(axis=0).mean())
                pair_scores.append((sim_ij + sim_ji) / 2)

                rows.append({
                    "method": method_name,
                    "dataset_pair": f"{ds_i}-{ds_j}",
                    "similarity": pair_scores[-1],
                    "shared_genes": len(shared),
                })

        if pair_scores:
            rows.append({
                "method": method_name,
                "dataset_pair": "mean",
                "similarity": float(np.mean(pair_scores)),
                "shared_genes": 0,
            })

    return pd.DataFrame(rows)


# ===========================================================================
# Report generation
# ===========================================================================

def generate_report(
    quality_df: pd.DataFrame,
    reprod_df: pd.DataFrame,
    dataset_info: dict[str, dict],
) -> None:
    """Generate multi-dataset benchmark PDF."""
    logger.info("Generating multi-dataset benchmark PDF...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    with PdfPages(str(REPORT_PATH)) as pdf:
        # Title page
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.75, "nPathway Multi-Dataset Benchmark", fontsize=26,
                fontweight="bold", ha="center", transform=ax.transAxes)
        ax.text(0.5, 0.62, "Cross-Dataset Evaluation & Reproducibility Analysis",
                fontsize=16, ha="center", transform=ax.transAxes, style="italic")
        ds_info_str = "\n".join(
            f"  {name}: {info['n_cells']} cells x {info['n_genes']} genes, "
            f"{info['n_hallmark']} Hallmark sets"
            for name, info in dataset_info.items()
        )
        ax.text(0.5, 0.35, f"Datasets:\n{ds_info_str}",
                fontsize=11, ha="center", transform=ax.transAxes,
                family="monospace", color="#1565C0")
        ax.text(0.5, 0.15, "Date: 2026-03-03", fontsize=10, ha="center",
                transform=ax.transAxes, color="gray")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Per-metric comparison across datasets
        metrics = ["coverage", "redundancy", "specificity", "hallmark_alignment"]
        metric_titles = ["Gene Coverage", "Redundancy (lower=better)",
                        "Specificity", "Hallmark Alignment"]
        dataset_names = quality_df["dataset"].unique()

        for metric, title in zip(metrics, metric_titles):
            fig, axes = plt.subplots(1, len(dataset_names), figsize=(5 * len(dataset_names), 5))
            if len(dataset_names) == 1:
                axes = [axes]
            fig.suptitle(f"{title} Across Datasets", fontsize=14, fontweight="bold")

            for ax_idx, ds_name in enumerate(dataset_names):
                sub = quality_df[quality_df["dataset"] == ds_name]
                methods = sub["method"].tolist()
                vals = sub[metric].values
                colors = [METHOD_COLORS.get(m, "#999") for m in methods]
                axes[ax_idx].bar(range(len(methods)), vals, color=colors,
                               edgecolor="black", linewidth=0.5)
                axes[ax_idx].set_xticks(range(len(methods)))
                axes[ax_idx].set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
                axes[ax_idx].set_title(ds_name, fontsize=11)
                axes[ax_idx].grid(True, alpha=0.3, axis="y")
                for i, v in enumerate(vals):
                    axes[ax_idx].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)

            fig.tight_layout()
            fig.savefig(str(FIGURES_DIR / f"multi_{metric}.png"), dpi=150, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Aggregate comparison: mean rank across datasets
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle("Mean Rank Across Datasets (lower=better)", fontsize=14, fontweight="bold")

        rank_metrics = ["coverage", "specificity", "hallmark_alignment"]
        rank_data = {}
        for method in METHOD_COLORS:
            method_sub = quality_df[quality_df["method"] == method]
            if method_sub.empty:
                continue
            ranks = []
            for metric in rank_metrics:
                for ds_name in dataset_names:
                    ds_sub = quality_df[quality_df["dataset"] == ds_name]
                    if method in ds_sub["method"].values:
                        all_vals = ds_sub.set_index("method")[metric]
                        sorted_vals = all_vals.sort_values(ascending=False)
                        rank = list(sorted_vals.index).index(method) + 1
                        ranks.append(rank)
            # Inverse rank for redundancy (lower is better)
            for ds_name in dataset_names:
                ds_sub = quality_df[quality_df["dataset"] == ds_name]
                if method in ds_sub["method"].values:
                    all_vals = ds_sub.set_index("method")["redundancy"]
                    sorted_vals = all_vals.sort_values(ascending=True)
                    rank = list(sorted_vals.index).index(method) + 1
                    ranks.append(rank)
            if ranks:
                rank_data[method] = np.mean(ranks)

        if rank_data:
            sorted_methods = sorted(rank_data.keys(), key=lambda m: rank_data[m])
            colors = [METHOD_COLORS.get(m, "#999") for m in sorted_methods]
            vals = [rank_data[m] for m in sorted_methods]
            ax.barh(range(len(sorted_methods)), vals, color=colors,
                   edgecolor="black", linewidth=0.5)
            ax.set_yticks(range(len(sorted_methods)))
            ax.set_yticklabels(sorted_methods, fontsize=10)
            ax.set_xlabel("Mean Rank (lower=better)")
            ax.invert_yaxis()
            for i, v in enumerate(vals):
                ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=10, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="x")

        fig.tight_layout()
        fig.savefig(str(FIGURES_DIR / "multi_rank.png"), dpi=150, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Cross-dataset reproducibility
        if not reprod_df.empty:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
            fig.suptitle("Cross-Dataset Reproducibility", fontsize=14, fontweight="bold")

            # Mean reproducibility per method
            mean_reprod = reprod_df[reprod_df["dataset_pair"] == "mean"]
            if not mean_reprod.empty:
                methods = mean_reprod["method"].tolist()
                vals = mean_reprod["similarity"].values
                colors = [METHOD_COLORS.get(m, "#999") for m in methods]
                axes[0].bar(range(len(methods)), vals, color=colors,
                           edgecolor="black", linewidth=0.5)
                axes[0].set_xticks(range(len(methods)))
                axes[0].set_xticklabels(methods, rotation=35, ha="right", fontsize=9)
                axes[0].set_ylabel("Mean Best-Match Jaccard")
                axes[0].set_title("Mean Cross-Dataset Program Similarity")
                for i, v in enumerate(vals):
                    axes[0].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
                axes[0].grid(True, alpha=0.3, axis="y")

            # Per-pair heatmap
            pair_reprod = reprod_df[reprod_df["dataset_pair"] != "mean"]
            if not pair_reprod.empty:
                pivot = pair_reprod.pivot_table(
                    index="method", columns="dataset_pair", values="similarity",
                    aggfunc="mean"
                )
                ordered = [m for m in METHOD_COLORS if m in pivot.index]
                pivot = pivot.reindex(ordered)
                sns.heatmap(pivot, cmap="YlGnBu", ax=axes[1], annot=True, fmt=".3f",
                           linewidths=0.5, linecolor="white", vmin=0, vmax=0.5)
                axes[1].set_title("Reproducibility per Dataset Pair")
                axes[1].set_ylabel("")

            fig.tight_layout()
            fig.savefig(str(FIGURES_DIR / "multi_reproducibility.png"), dpi=150,
                       bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Summary table
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.axis("off")
        fig.suptitle("Multi-Dataset Summary (Mean Across Datasets)",
                     fontsize=14, fontweight="bold")

        summary_rows = []
        for method in METHOD_COLORS:
            msub = quality_df[quality_df["method"] == method]
            if msub.empty:
                continue
            reprod_val = 0.0
            if not reprod_df.empty:
                mr = reprod_df[(reprod_df["method"] == method) &
                              (reprod_df["dataset_pair"] == "mean")]
                if not mr.empty:
                    reprod_val = mr["similarity"].values[0]

            summary_rows.append([
                method,
                f"{msub['coverage'].mean():.3f}",
                f"{msub['redundancy'].mean():.3f}",
                f"{msub['specificity'].mean():.3f}",
                f"{msub['hallmark_alignment'].mean():.4f}",
                f"{msub['mean_program_size'].mean():.0f}",
                f"{reprod_val:.3f}",
            ])

        col_labels = ["Method", "Coverage", "Redundancy\n(lower=better)",
                     "Specificity", "Hallmark\nAlignment", "Mean Size",
                     "Cross-Dataset\nReproducibility"]
        table = ax.table(cellText=summary_rows, colLabels=col_labels,
                        cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.0)
        for j in range(len(col_labels)):
            table[0, j].set_facecolor("#1565C0")
            table[0, j].set_text_props(color="white", fontweight="bold", fontsize=9)
        for i in range(1, len(summary_rows) + 1):
            bg = "#E3F2FD" if i % 2 == 0 else "#FFFFFF"
            for j in range(len(col_labels)):
                table[i, j].set_facecolor(bg)
            if summary_rows[i - 1][0].startswith("nPathway"):
                table[i, 0].set_text_props(fontweight="bold")

        fig.tight_layout()
        fig.savefig(str(FIGURES_DIR / "multi_summary.png"), dpi=150, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # Save CSVs
    quality_df.to_csv(str(TABLES_DIR / "multi_dataset_quality.csv"), index=False)
    if not reprod_df.empty:
        reprod_df.to_csv(str(TABLES_DIR / "cross_dataset_reproducibility.csv"), index=False)

    logger.info("Multi-dataset report: %s", REPORT_PATH)


def main() -> None:
    """Run multi-dataset benchmark."""
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("nPathway Multi-Dataset Benchmark")
    logger.info("=" * 60)

    datasets = load_datasets()

    # Collect dataset info for report
    dataset_info: dict[str, dict] = {}
    for name, (adata, _, gene_names, hallmark) in datasets.items():
        dataset_info[name] = {
            "n_cells": adata.n_obs,
            "n_genes": len(gene_names),
            "n_hallmark": len(hallmark),
        }

    # Run discovery on each dataset
    all_dataset_programs: dict[str, dict[str, dict[str, list[str]]]] = {}
    quality_rows: list[dict] = []

    for ds_name, (adata, graph_emb, gene_names, hallmark) in datasets.items():
        logger.info("Running methods on %s...", ds_name)
        methods = discover_methods_for_dataset(adata, graph_emb, gene_names)
        all_dataset_programs[ds_name] = methods

        for method_name, programs in methods.items():
            metrics = evaluate_quality(programs, hallmark, gene_names)
            metrics["method"] = method_name
            metrics["dataset"] = ds_name
            quality_rows.append(metrics)

    quality_df = pd.DataFrame(quality_rows)

    # Cross-dataset reproducibility
    reprod_df = cross_dataset_reproducibility(all_dataset_programs)

    generate_report(quality_df, reprod_df, dataset_info)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("MULTI-DATASET BENCHMARK COMPLETE (%.1f seconds)", elapsed)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
