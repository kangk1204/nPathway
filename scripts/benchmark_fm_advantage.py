#!/usr/bin/env python3
"""FM Advantage Benchmark: Find scenarios where FM embeddings outperform PCA.

Experiments:
1. Low-data regime: subsample cells, compare PCA vs FM at each sample size
2. Zero-shot programs: FM embeddings without ANY expression data
3. Gene coverage expansion: HVGs vs all expressed genes
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
import scanpy as sc
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from npathway.data.preprocessing import _safe_toarray
from npathway.discovery.clustering import ClusteringProgramDiscovery
from npathway.discovery.ensemble import EnsembleProgramDiscovery
from npathway.evaluation.metrics import jaccard_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "results"
TABLES_DIR = OUTPUT_DIR / "tables"
REPORT_PATH = Path(__file__).parent.parent / "benchmark_fm_advantage.pdf"
MODEL_DIR = Path(__file__).parent.parent / "models"

N_PROGRAMS = 20
TOP_N_GENES = 50
SEED = 42


# ===========================================================================
# Shared utilities
# ===========================================================================

def load_fm_embeddings(gene_names: list[str], model_name: str) -> tuple[np.ndarray, list[str]] | None:
    """Load FM gene embeddings and map to dataset genes."""
    emb_path = MODEL_DIR / f"{model_name}_gene_embeddings.npz"
    if not emb_path.exists():
        return None

    data = np.load(emb_path, allow_pickle=True)
    all_emb = data["embeddings"]
    all_names = list(data["gene_names"])

    name_to_idx = {n: i for i, n in enumerate(all_names)}
    name_upper = {n.upper(): i for i, n in enumerate(all_names)}

    matched_idx = []
    matched_genes = []
    for g in gene_names:
        idx = name_to_idx.get(g) or name_upper.get(g.upper())
        if idx is not None:
            matched_idx.append(idx)
            matched_genes.append(g)

    if len(matched_idx) < 50:
        return None

    embeddings = all_emb[matched_idx].astype(np.float64)
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    return embeddings, matched_genes


def apply_graph_reg(
    emb: np.ndarray,
    n_components: int = 50,
    k_neighbors: int = 15,
    n_steps: int = 3,
    alpha: float = 0.5,
) -> np.ndarray:
    """Apply PCA reduction + kNN graph-regularized diffusion."""
    emb = emb.astype(np.float64)
    emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)

    n, dim = emb.shape
    if dim > n_components and n > n_components:
        pca = PCA(n_components=n_components, random_state=SEED, svd_solver="full")
        emb = pca.fit_transform(emb)

    k = min(k_neighbors, n - 1)
    if k < 2:
        return emb.astype(np.float32)

    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    nn.fit(emb)
    distances, indices = nn.kneighbors(emb)

    adj = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j_idx, j in enumerate(indices[i]):
            if i != j:
                adj[i, j] = max(0.0, 1.0 - distances[i, j_idx])
    adj = 0.5 * (adj + adj.T)
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    adj_norm = adj / row_sums

    result = emb.copy()
    for _ in range(n_steps):
        result = alpha * result + (1 - alpha) * (adj_norm @ result)
    return result.astype(np.float32)


def discover_and_evaluate(
    emb: np.ndarray,
    gene_names: list[str],
    de_gene_sets: dict[str, list[str]],
    all_gene_names: list[str],
    method: str = "kmeans",
) -> dict:
    """Discover programs and evaluate against DE gene sets."""
    gene_set = set(all_gene_names)

    if method == "ensemble":
        try:
            disc = EnsembleProgramDiscovery(
                methods=[
                    ClusteringProgramDiscovery(method="kmeans", n_programs=N_PROGRAMS, random_state=SEED),
                    ClusteringProgramDiscovery(method="leiden", random_state=SEED + 1),
                    ClusteringProgramDiscovery(method="spectral", n_programs=N_PROGRAMS, random_state=SEED + 2),
                ],
                consensus_method="leiden", resolution=1.0, threshold_quantile=0.3,
                min_program_size=5, random_state=SEED,
            )
            disc.fit(emb, gene_names)
            programs = disc.get_programs()
        except Exception:
            return {"mean_jaccard": 0, "mean_recall": 0, "n_programs": 0, "coverage": 0}
    else:
        disc = ClusteringProgramDiscovery(method=method, n_programs=N_PROGRAMS, random_state=SEED)
        disc.fit(emb, gene_names)
        programs = disc.get_programs()

    # Evaluate
    jaccards = []
    recalls = []
    for de_genes in de_gene_sets.values():
        de_set = set(de_genes) & gene_set
        if len(de_set) < 5:
            continue
        best_j = 0.0
        best_r = 0.0
        for prog_genes in programs.values():
            prog_set = set(prog_genes) & gene_set
            if not prog_set:
                continue
            j = jaccard_similarity(de_set, prog_set)
            r = len(de_set & prog_set) / len(de_set)
            if j > best_j:
                best_j = j
                best_r = r
        jaccards.append(best_j)
        recalls.append(best_r)

    n_covered = len({g for gl in programs.values() for g in gl})
    return {
        "mean_jaccard": np.mean(jaccards) if jaccards else 0,
        "mean_recall": np.mean(recalls) if recalls else 0,
        "n_programs": len(programs),
        "coverage": n_covered / len(all_gene_names) if all_gene_names else 0,
    }


# ===========================================================================
# Experiment 1: Low-data regime
# ===========================================================================

def experiment_low_data(adata_full: sc.AnnData, de_gene_sets: dict[str, list[str]]) -> pd.DataFrame:
    """Subsample cells and compare PCA vs FM at each sample size."""
    logger.info("=" * 50)
    logger.info("Experiment 1: Low-data regime")
    logger.info("=" * 50)

    gene_names = list(adata_full.var_names)
    sample_sizes = [50, 100, 200, 500, 1000, 2000, 5000]
    # Filter to sizes smaller than dataset
    sample_sizes = [s for s in sample_sizes if s < adata_full.n_obs]

    rng = np.random.RandomState(SEED)
    rows = []

    for n_cells in sample_sizes:
        logger.info("  n_cells = %d", n_cells)
        idx = rng.choice(adata_full.n_obs, n_cells, replace=False)
        adata_sub = adata_full[idx].copy()

        # PCA embeddings from subsampled expression
        X_sub = _safe_toarray(adata_sub.X).astype(np.float64)
        n_comp = min(50, min(X_sub.shape) - 1)
        if n_comp < 5:
            logger.warning("    Too few components for PCA at n=%d, skipping", n_cells)
            continue

        pca = PCA(n_components=n_comp, random_state=SEED, svd_solver="full")
        pca_emb = pca.fit_transform(X_sub.T)  # genes x components
        pca_graph = apply_graph_reg(pca_emb, n_components=n_comp)

        result_pca = discover_and_evaluate(pca_graph, gene_names, de_gene_sets, gene_names)
        rows.append({"n_cells": n_cells, "embedding": "PCA", **result_pca})

        # scGPT embeddings (fixed, independent of n_cells)
        scgpt = load_fm_embeddings(gene_names, "scgpt")
        if scgpt is not None:
            scgpt_emb, scgpt_genes = scgpt
            scgpt_graph = apply_graph_reg(scgpt_emb)
            result_scgpt = discover_and_evaluate(scgpt_graph, scgpt_genes, de_gene_sets, gene_names)
            rows.append({"n_cells": n_cells, "embedding": "scGPT", **result_scgpt})

        # Geneformer embeddings (fixed)
        gf = load_fm_embeddings(gene_names, "geneformer")
        if gf is not None:
            gf_emb, gf_genes = gf
            gf_graph = apply_graph_reg(gf_emb)
            result_gf = discover_and_evaluate(gf_graph, gf_genes, de_gene_sets, gene_names)
            rows.append({"n_cells": n_cells, "embedding": "Geneformer", **result_gf})

        # Hybrid: FM embedding + expression-based graph (use expression kNN, FM features)
        if scgpt is not None:
            # Build expression-based kNN graph, diffuse FM embeddings on it
            hybrid_emb = _hybrid_fm_expression(scgpt_emb, X_sub, scgpt_genes, gene_names)
            if hybrid_emb is not None:
                result_hybrid = discover_and_evaluate(hybrid_emb, scgpt_genes, de_gene_sets, gene_names)
                rows.append({"n_cells": n_cells, "embedding": "Hybrid-scGPT", **result_hybrid})

    df = pd.DataFrame(rows)
    logger.info("Low-data results:\n%s", df.to_string(index=False))
    return df


def _hybrid_fm_expression(
    fm_emb: np.ndarray,
    X_expr: np.ndarray,
    fm_genes: list[str],
    all_genes: list[str],
    n_components: int = 50,
    k: int = 15,
    n_steps: int = 3,
    alpha: float = 0.5,
) -> np.ndarray | None:
    """Hybrid approach: use expression-derived kNN graph to diffuse FM embeddings.

    The graph captures dataset-specific co-expression, while FM embeddings
    provide the feature space. Best of both worlds.
    """
    gene_idx_map = {g: i for i, g in enumerate(all_genes)}
    fm_gene_indices = [gene_idx_map[g] for g in fm_genes if g in gene_idx_map]
    if len(fm_gene_indices) < 50:
        return None

    # Build expression-based gene embeddings for graph construction only
    X_genes = X_expr[:, fm_gene_indices].T  # fm_genes x cells
    n_comp = min(n_components, min(X_genes.shape) - 1)
    if n_comp < 5:
        return None

    pca = PCA(n_components=n_comp, random_state=SEED, svd_solver="full")
    expr_emb = pca.fit_transform(X_genes.astype(np.float64))

    # Build kNN graph from expression
    n = expr_emb.shape[0]
    k_actual = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k_actual, metric="cosine", algorithm="brute")
    nn.fit(expr_emb)
    distances, indices = nn.kneighbors(expr_emb)

    adj = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j_idx, j in enumerate(indices[i]):
            if i != j:
                adj[i, j] = max(0.0, 1.0 - distances[i, j_idx])
    adj = 0.5 * (adj + adj.T)
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    adj_norm = adj / row_sums

    # Diffuse FM embeddings on expression graph
    fm_reduced = fm_emb.astype(np.float64)
    if fm_reduced.shape[1] > n_components:
        pca2 = PCA(n_components=n_components, random_state=SEED, svd_solver="full")
        fm_reduced = pca2.fit_transform(fm_reduced)

    result = fm_reduced.copy()
    for _ in range(n_steps):
        result = alpha * result + (1 - alpha) * (adj_norm @ result)

    return result.astype(np.float32)


# ===========================================================================
# Experiment 2: Zero-shot gene programs
# ===========================================================================

def experiment_zero_shot(
    gene_names: list[str],
    de_gene_sets: dict[str, list[str]],
) -> pd.DataFrame:
    """Generate programs from FM embeddings only, without any expression data."""
    logger.info("=" * 50)
    logger.info("Experiment 2: Zero-shot gene programs (no expression data)")
    logger.info("=" * 50)

    rows = []

    for model_name in ["scgpt", "geneformer"]:
        result = load_fm_embeddings(gene_names, model_name)
        if result is None:
            continue

        emb, matched_genes = result
        logger.info("  %s: %d genes matched", model_name, len(matched_genes))

        # Pure FM embeddings (no graph reg from expression)
        emb_reduced = emb.astype(np.float64)
        n_comp = min(50, min(emb_reduced.shape) - 1)
        if emb_reduced.shape[1] > n_comp:
            pca = PCA(n_components=n_comp, random_state=SEED, svd_solver="full")
            emb_reduced = pca.fit_transform(emb_reduced)

        # No graph reg - pure FM
        for method in ["kmeans", "ensemble"]:
            res = discover_and_evaluate(
                emb_reduced.astype(np.float32), matched_genes, de_gene_sets, gene_names, method
            )
            rows.append({"model": model_name, "graph_reg": "none", "method": method, **res})

        # FM with self-graph-reg (kNN from FM embeddings themselves)
        fm_graph = apply_graph_reg(emb)
        for method in ["kmeans", "ensemble"]:
            res = discover_and_evaluate(fm_graph, matched_genes, de_gene_sets, gene_names, method)
            rows.append({"model": model_name, "graph_reg": "fm_self", "method": method, **res})

    df = pd.DataFrame(rows)
    logger.info("Zero-shot results:\n%s", df.to_string(index=False))
    return df


# ===========================================================================
# Experiment 3: Gene coverage expansion
# ===========================================================================

def experiment_gene_coverage(
    adata_full: sc.AnnData,
    adata_hvg: sc.AnnData,
    de_gene_sets_full: dict[str, list[str]],
    de_gene_sets_hvg: dict[str, list[str]],
) -> pd.DataFrame:
    """Compare HVG-only vs all-gene programs.

    FM can cover all genes; PCA limited to HVGs.
    """
    logger.info("=" * 50)
    logger.info("Experiment 3: Gene coverage expansion")
    logger.info("=" * 50)

    rows = []

    # HVG genes
    hvg_genes = list(adata_hvg.var_names)
    logger.info("  HVG genes: %d", len(hvg_genes))

    # All expressed genes (top 10k by mean expression)
    all_genes = list(adata_full.var_names)
    logger.info("  All genes: %d", len(all_genes))

    # PCA on HVGs (standard)
    rng = np.random.RandomState(SEED)
    idx = rng.choice(adata_hvg.n_obs, min(5000, adata_hvg.n_obs), replace=False)
    X_hvg = _safe_toarray(adata_hvg[idx].X).astype(np.float64)
    n_comp = min(50, min(X_hvg.shape) - 1)
    pca = PCA(n_components=n_comp, random_state=SEED, svd_solver="full")
    pca_emb = pca.fit_transform(X_hvg.T)
    pca_graph = apply_graph_reg(pca_emb, n_components=n_comp)
    res = discover_and_evaluate(pca_graph, hvg_genes, de_gene_sets_hvg, hvg_genes)
    rows.append({"scope": "HVG_only", "embedding": "PCA", "n_genes": len(hvg_genes), **res})

    # FM on HVGs only
    for model_name in ["scgpt", "geneformer"]:
        fm = load_fm_embeddings(hvg_genes, model_name)
        if fm is None:
            continue
        fm_emb, fm_genes = fm
        fm_graph = apply_graph_reg(fm_emb)
        res = discover_and_evaluate(fm_graph, fm_genes, de_gene_sets_hvg, hvg_genes)
        rows.append({"scope": "HVG_only", "embedding": model_name, "n_genes": len(fm_genes), **res})

    # FM on ALL genes (PCA can't do this efficiently)
    for model_name in ["scgpt", "geneformer"]:
        fm = load_fm_embeddings(all_genes, model_name)
        if fm is None:
            continue
        fm_emb, fm_genes = fm
        logger.info("  %s all-gene: %d/%d genes matched", model_name, len(fm_genes), len(all_genes))
        fm_graph = apply_graph_reg(fm_emb)
        res = discover_and_evaluate(fm_graph, fm_genes, de_gene_sets_full, all_genes)
        rows.append({"scope": "all_genes", "embedding": model_name, "n_genes": len(fm_genes), **res})

    df = pd.DataFrame(rows)
    logger.info("Gene coverage results:\n%s", df.to_string(index=False))
    return df


# ===========================================================================
# Report generation
# ===========================================================================

def generate_report(
    lowdata_df: pd.DataFrame,
    zeroshot_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
):
    """Generate comprehensive PDF report."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    lowdata_df.to_csv(TABLES_DIR / "fm_advantage_lowdata.csv", index=False)
    zeroshot_df.to_csv(TABLES_DIR / "fm_advantage_zeroshot.csv", index=False)
    coverage_df.to_csv(TABLES_DIR / "fm_advantage_coverage.csv", index=False)

    emb_colors = {
        "PCA": "#1565C0",
        "scGPT": "#E65100",
        "Geneformer": "#2E7D32",
        "scgpt": "#E65100",
        "geneformer": "#2E7D32",
        "Hybrid-scGPT": "#AD1457",
    }

    def get_color(name: str) -> str:
        for key, c in emb_colors.items():
            if key.lower() in name.lower():
                return c
        return "#757575"

    with PdfPages(REPORT_PATH) as pdf:
        # Title page
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.7, "FM Advantage Benchmark",
                ha="center", va="center", fontsize=24, fontweight="bold")
        ax.text(0.5, 0.55, "Scenarios Where Foundation Model Embeddings Outperform PCA",
                ha="center", va="center", fontsize=16)
        ax.text(0.5, 0.4, "Low-data regime | Zero-shot programs | Gene coverage expansion",
                ha="center", va="center", fontsize=13, color="gray")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Experiment 1: Low-data regime
        if not lowdata_df.empty:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            for metric, ax, title in [
                ("mean_jaccard", axes[0], "Cell-Type DE Recovery (Jaccard)"),
                ("mean_recall", axes[1], "Cell-Type DE Recovery (Recall)"),
            ]:
                for emb_type in lowdata_df["embedding"].unique():
                    edf = lowdata_df[lowdata_df["embedding"] == emb_type]
                    color = get_color(emb_type)
                    ax.plot(edf["n_cells"], edf[metric], "o-", color=color,
                            label=emb_type, linewidth=2, markersize=6)
                ax.set_xlabel("Number of Cells (log scale)")
                ax.set_ylabel(title.split("(")[1].rstrip(")"))
                ax.set_title(title)
                ax.set_xscale("log")
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)

            plt.suptitle("Experiment 1: Low-Data Regime", fontsize=14, fontweight="bold")
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Experiment 2: Zero-shot
        if not zeroshot_df.empty:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            for metric, ax, title in [
                ("mean_jaccard", axes[0], "Zero-Shot Jaccard"),
                ("mean_recall", axes[1], "Zero-Shot Recall"),
            ]:
                zdf = zeroshot_df.copy()
                zdf["label"] = zdf["model"] + "\n" + zdf["graph_reg"] + "\n" + zdf["method"]
                colors = [get_color(m) for m in zdf["model"]]
                ax.barh(range(len(zdf)), zdf[metric], color=colors, edgecolor="white")
                ax.set_yticks(range(len(zdf)))
                ax.set_yticklabels(zdf["label"], fontsize=7)
                ax.set_xlabel(metric.replace("_", " ").title())
                ax.set_title(title)

            plt.suptitle("Experiment 2: Zero-Shot Gene Programs (No Expression Data)",
                         fontsize=14, fontweight="bold")
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Experiment 3: Gene coverage
        if not coverage_df.empty:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            for metric, ax, title in [
                ("mean_jaccard", axes[0], "Jaccard by Gene Scope"),
                ("mean_recall", axes[1], "Recall by Gene Scope"),
            ]:
                cdf = coverage_df.copy()
                cdf["label"] = cdf["embedding"] + " (" + cdf["scope"] + ")"
                colors = [get_color(e) for e in cdf["embedding"]]
                ax.barh(range(len(cdf)), cdf[metric], color=colors, edgecolor="white")
                ax.set_yticks(range(len(cdf)))
                ax.set_yticklabels(cdf["label"], fontsize=8)
                ax.set_xlabel(metric.replace("_", " ").title())
                ax.set_title(title)

            plt.suptitle("Experiment 3: Gene Coverage Expansion",
                         fontsize=14, fontweight="bold")
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Summary table
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis("off")
        summary_rows = []

        if not lowdata_df.empty:
            # Find crossover point
            for emb in ["scGPT", "Geneformer", "Hybrid-scGPT"]:
                edf = lowdata_df[lowdata_df["embedding"] == emb]
                pdf_pca = lowdata_df[lowdata_df["embedding"] == "PCA"]
                if edf.empty or pdf_pca.empty:
                    continue
                for _, row in edf.iterrows():
                    n = row["n_cells"]
                    pca_row = pdf_pca[pdf_pca["n_cells"] == n]
                    if pca_row.empty:
                        continue
                    pca_j = pca_row.iloc[0]["mean_jaccard"]
                    fm_j = row["mean_jaccard"]
                    if fm_j > pca_j and pca_j > 0:
                        summary_rows.append({
                            "Finding": f"{emb} > PCA at n={int(n)}",
                            "FM Jaccard": f"{fm_j:.4f}",
                            "PCA Jaccard": f"{pca_j:.4f}",
                            "Improvement": f"{(fm_j/pca_j - 1)*100:.1f}%",
                        })

        if not zeroshot_df.empty:
            best = zeroshot_df.loc[zeroshot_df["mean_jaccard"].idxmax()]
            summary_rows.append({
                "Finding": f"Best zero-shot: {best['model']}",
                "FM Jaccard": f"{best['mean_jaccard']:.4f}",
                "PCA Jaccard": "N/A (requires data)",
                "Improvement": "Unique FM capability",
            })

        if not coverage_df.empty:
            for model in ["scgpt", "geneformer"]:
                hvg = coverage_df[(coverage_df["scope"] == "HVG_only") & (coverage_df["embedding"] == model)]
                allg = coverage_df[(coverage_df["scope"] == "all_genes") & (coverage_df["embedding"] == model)]
                if not hvg.empty and not allg.empty:
                    hvg_j = hvg.iloc[0]["mean_jaccard"]
                    all_j = allg.iloc[0]["mean_jaccard"]
                    if all_j > hvg_j and hvg_j > 0:
                        summary_rows.append({
                            "Finding": f"{model} all-genes > HVGs",
                            "FM Jaccard": f"{all_j:.4f}",
                            "PCA Jaccard": f"{hvg_j:.4f} (HVG)",
                            "Improvement": f"{(all_j/hvg_j - 1)*100:.1f}%",
                        })

        if summary_rows:
            sdf = pd.DataFrame(summary_rows)
            table = ax.table(
                cellText=sdf.values, colLabels=list(sdf.columns),
                loc="center", cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.0, 1.8)
            for j in range(len(sdf.columns)):
                table[0, j].set_facecolor("#1565C0")
                table[0, j].set_text_props(color="white", fontweight="bold")
        else:
            ax.text(0.5, 0.5, "No FM advantage scenarios found in this benchmark",
                    ha="center", va="center", fontsize=14, color="gray")

        ax.set_title("FM Advantage Summary", fontsize=14, fontweight="bold", pad=20)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    logger.info("Report saved: %s", REPORT_PATH)


# ===========================================================================
# Main
# ===========================================================================

def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("FM Advantage Benchmark")
    logger.info("=" * 60)

    # Load PBMC 68k
    import scvelo as scv
    logger.info("Loading PBMC 68k...")
    adata_full = scv.datasets.pbmc68k()
    logger.info("  %d cells x %d genes", adata_full.n_obs, adata_full.n_vars)

    # Preprocess for HVG version
    logger.info("Preprocessing...")
    sc.pp.filter_genes(adata_full, min_cells=10)
    adata_full.layers["counts"] = adata_full.X.copy()
    sc.pp.normalize_total(adata_full, target_sum=1e4)
    sc.pp.log1p(adata_full)

    # Compute DE on full gene set (before HVG filtering)
    logger.info("Computing cell-type DE genes (full gene set)...")
    sc.tl.rank_genes_groups(adata_full, groupby="celltype", method="wilcoxon",
                            n_genes=100, use_raw=False)
    de_gene_sets_full: dict[str, list[str]] = {}
    gene_set_full = set(adata_full.var_names)
    for ct in adata_full.obs["celltype"].unique():
        try:
            de = [str(g) for g in adata_full.uns["rank_genes_groups"]["names"][ct]
                  if str(g) in gene_set_full]
            if len(de) >= 10:
                de_gene_sets_full[ct] = de
        except (KeyError, IndexError):
            continue
    logger.info("  %d cell types with DE genes (full)", len(de_gene_sets_full))

    # HVG version
    sc.pp.highly_variable_genes(adata_full, n_top_genes=3000, layer="counts")
    adata_hvg = adata_full[:, adata_full.var["highly_variable"]].copy()
    sc.pp.scale(adata_hvg, max_value=10)
    hvg_genes = list(adata_hvg.var_names)

    # DE on HVGs
    de_gene_sets_hvg: dict[str, list[str]] = {}
    for ct, genes in de_gene_sets_full.items():
        hvg_de = [g for g in genes if g in set(hvg_genes)]
        if len(hvg_de) >= 10:
            de_gene_sets_hvg[ct] = hvg_de
    logger.info("  %d cell types with DE genes (HVG)", len(de_gene_sets_hvg))

    # Scale full data too (for PCA on subsets)
    sc.pp.scale(adata_full, max_value=10)

    # Experiment 1: Low-data regime
    lowdata_df = experiment_low_data(adata_hvg, de_gene_sets_hvg)

    # Experiment 2: Zero-shot programs
    zeroshot_df = experiment_zero_shot(hvg_genes, de_gene_sets_hvg)

    # Experiment 3: Gene coverage expansion
    coverage_df = experiment_gene_coverage(adata_full, adata_hvg, de_gene_sets_full, de_gene_sets_hvg)

    # Generate report
    generate_report(lowdata_df, zeroshot_df, coverage_df)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("FM ADVANTAGE BENCHMARK COMPLETE (%.1f seconds)", elapsed)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
