#!/usr/bin/env python3
"""Large-Scale Benchmark: PBMC 68k with Cell-Type DE Recovery.

Combines large-scale validation (65,877 cells) with perturbation-style
recovery evaluation using cell-type-specific DE genes as ground truth.

Tests PCA, scGPT, and Geneformer embeddings with graph regularization
on a dataset ~25x larger than PBMC 3k.
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
from npathway.discovery.baselines import (
    CNMFProgramDiscovery,
    ExpressionClusteringBaseline,
    RandomProgramDiscovery,
    WGCNAProgramDiscovery,
)
from npathway.discovery.clustering import ClusteringProgramDiscovery
from npathway.discovery.ensemble import EnsembleProgramDiscovery
from npathway.evaluation.metrics import jaccard_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "results"
TABLES_DIR = OUTPUT_DIR / "tables"
REPORT_PATH = Path(__file__).parent.parent / "benchmark_largescale.pdf"
MODEL_DIR = Path(__file__).parent.parent / "models"

N_PROGRAMS = 20
TOP_N_GENES = 50
SEED = 42


# ===========================================================================
# Data loading
# ===========================================================================

def load_pbmc68k() -> sc.AnnData:
    """Load PBMC 68k dataset from scvelo."""
    import scvelo as scv
    logger.info("Loading PBMC 68k from scvelo...")
    adata = scv.datasets.pbmc68k()
    logger.info("  Loaded: %d cells x %d genes, %d cell types",
                adata.n_obs, adata.n_vars, adata.obs["celltype"].nunique())
    return adata


def preprocess_pbmc68k(adata: sc.AnnData, n_top_genes: int = 3000) -> sc.AnnData:
    """Preprocess PBMC 68k: filter, normalize, log-transform, select HVGs."""
    logger.info("Preprocessing PBMC 68k...")

    # Filter genes present in at least 10 cells
    sc.pp.filter_genes(adata, min_cells=10)
    logger.info("  After gene filter: %d genes", adata.n_vars)

    # Store raw counts for DE later
    adata.layers["counts"] = adata.X.copy()

    # Normalize + log
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # HVG selection
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat_v3",
                                layer="counts")
    n_hvg = adata.var["highly_variable"].sum()
    logger.info("  %d highly variable genes selected", n_hvg)

    # Subset to HVGs
    adata = adata[:, adata.var["highly_variable"]].copy()

    # Scale
    sc.pp.scale(adata, max_value=10)
    logger.info("  Final: %d cells x %d genes", adata.n_obs, adata.n_vars)
    return adata


# ===========================================================================
# Cell-type DE genes (perturbation ground truth)
# ===========================================================================

def compute_celltype_de_genes(
    adata: sc.AnnData,
    n_top_genes: int = 100,
    min_cells: int = 50,
) -> dict[str, list[str]]:
    """Compute DE genes for each cell type vs rest (Wilcoxon).

    Uses cell-type DE genes as ground truth perturbation signatures.
    """
    logger.info("Computing cell-type DE genes...")
    celltypes = adata.obs["celltype"].value_counts()
    valid_types = celltypes[celltypes >= min_cells].index.tolist()
    logger.info("  %d cell types with >= %d cells", len(valid_types), min_cells)

    # Run rank_genes_groups
    sc.tl.rank_genes_groups(adata, groupby="celltype", method="wilcoxon",
                            n_genes=n_top_genes, use_raw=False)

    de_gene_sets: dict[str, list[str]] = {}
    gene_set = set(adata.var_names)

    for ct in valid_types:
        try:
            de_genes = [
                str(g) for g in adata.uns["rank_genes_groups"]["names"][ct]
                if str(g) in gene_set
            ]
            if len(de_genes) >= 10:
                de_gene_sets[ct] = de_genes
        except (KeyError, IndexError):
            continue

    logger.info("  DE gene sets for %d cell types (avg %.1f genes each)",
                len(de_gene_sets),
                np.mean([len(v) for v in de_gene_sets.values()]) if de_gene_sets else 0)
    return de_gene_sets


# ===========================================================================
# Embeddings
# ===========================================================================

def build_pca_graph_embeddings(
    adata: sc.AnnData,
    n_components: int = 50,
    k_neighbors: int = 15,
    n_steps: int = 3,
    alpha: float = 0.5,
) -> tuple[np.ndarray, list[str]]:
    """Build PCA graph-regularized gene embeddings from expression."""
    X = _safe_toarray(adata.X).astype(np.float64)
    gene_names = list(adata.var_names)

    # PCA on transposed expression (genes x components)
    pca = PCA(n_components=n_components, random_state=SEED, svd_solver="full")
    emb = pca.fit_transform(X.T)
    logger.info("  PCA: %s, explained var: %.1f%%",
                emb.shape, pca.explained_variance_ratio_.sum() * 100)

    # Graph regularization
    emb = _apply_graph_reg(emb, k_neighbors, n_steps, alpha)
    return emb.astype(np.float32), gene_names


def load_fm_embeddings(
    gene_names: list[str],
    model_name: str,
    n_components: int = 50,
    k_neighbors: int = 15,
    n_steps: int = 3,
    alpha: float = 0.5,
) -> tuple[np.ndarray, list[str]] | None:
    """Load FM embeddings, map to dataset genes, apply graph reg."""
    emb_path = MODEL_DIR / f"{model_name}_gene_embeddings.npz"
    if not emb_path.exists():
        logger.warning("  %s embeddings not found at %s", model_name, emb_path)
        return None

    data = np.load(emb_path, allow_pickle=True)
    all_emb = data["embeddings"]
    all_names = list(data["gene_names"])

    # Build lookup
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    name_upper = {n.upper(): i for i, n in enumerate(all_names)}

    matched_idx = []
    matched_genes = []
    for g in gene_names:
        idx = name_to_idx.get(g) or name_upper.get(g.upper())
        if idx is not None:
            matched_idx.append(idx)
            matched_genes.append(g)

    if len(matched_idx) < 100:
        logger.warning("  %s: only %d genes matched, skipping", model_name, len(matched_idx))
        return None

    embeddings = all_emb[matched_idx].astype(np.float64)
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    logger.info("  %s: %d/%d genes matched (%.1f%%), dim=%d",
                model_name, len(matched_genes), len(gene_names),
                100.0 * len(matched_genes) / len(gene_names), embeddings.shape[1])

    # PCA reduction if dim > n_components
    if embeddings.shape[1] > n_components:
        pca = PCA(n_components=n_components, random_state=SEED, svd_solver="full")
        embeddings = pca.fit_transform(embeddings)
        logger.info("    PCA reduction: %d -> %d (var: %.1f%%)",
                    all_emb.shape[1], n_components,
                    pca.explained_variance_ratio_.sum() * 100)

    # Graph regularization
    embeddings = _apply_graph_reg(embeddings, k_neighbors, n_steps, alpha)
    return embeddings.astype(np.float32), matched_genes


def _apply_graph_reg(
    emb: np.ndarray,
    k_neighbors: int = 15,
    n_steps: int = 3,
    alpha: float = 0.5,
) -> np.ndarray:
    """Apply kNN graph-regularized diffusion."""
    n = emb.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k_neighbors, n - 1), metric="cosine", algorithm="brute")
    nn.fit(emb)
    distances, indices = nn.kneighbors(emb)

    # Build adjacency
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
    return result


# ===========================================================================
# Discovery
# ===========================================================================

def discover_programs(
    emb: np.ndarray,
    gene_names: list[str],
    prefix: str,
) -> dict[str, dict[str, list[str]]]:
    """Run discovery methods on embeddings."""
    methods: dict[str, dict[str, list[str]]] = {}

    # KMeans
    km = ClusteringProgramDiscovery(method="kmeans", n_programs=N_PROGRAMS, random_state=SEED)
    km.fit(emb, gene_names)
    methods[f"{prefix}-KMeans"] = km.get_programs()

    # Refined (confidence-weighted top genes)
    km_conf = km.get_gene_confidence()
    km_scores = km.get_program_scores()
    refined: dict[str, list[str]] = {}
    for pn, sg in km_scores.items():
        conf = km_conf.get(pn, {})
        combined = [(g, (cs * conf.get(g, 0.5)) ** 0.5) for g, cs in sg]
        combined.sort(key=lambda x: x[1], reverse=True)
        refined[pn] = [g for g, _ in combined[:TOP_N_GENES]]
    methods[f"{prefix}-Refined"] = refined

    # Ensemble
    try:
        ens = EnsembleProgramDiscovery(
            methods=[
                ClusteringProgramDiscovery(method="kmeans", n_programs=N_PROGRAMS, random_state=SEED),
                ClusteringProgramDiscovery(method="leiden", random_state=SEED + 1),
                ClusteringProgramDiscovery(method="spectral", n_programs=N_PROGRAMS, random_state=SEED + 2),
            ],
            consensus_method="leiden", resolution=1.0, threshold_quantile=0.3,
            min_program_size=5, random_state=SEED,
        )
        ens.fit(emb, gene_names)
        methods[f"{prefix}-Ensemble"] = ens.get_programs()
    except Exception as exc:
        logger.warning("  %s-Ensemble failed: %s", prefix, exc)

    return methods


# ===========================================================================
# Evaluation
# ===========================================================================

def evaluate_de_recovery(
    all_methods: dict[str, dict[str, list[str]]],
    de_gene_sets: dict[str, list[str]],
    gene_names: list[str],
) -> pd.DataFrame:
    """Evaluate how well programs recover cell-type DE gene signatures."""
    gene_set = set(gene_names)
    rows = []

    for method_name, programs in all_methods.items():
        jaccards = []
        recalls = []
        precisions = []

        for ct_name, de_genes in de_gene_sets.items():
            de_set = set(de_genes) & gene_set
            if len(de_set) < 5:
                continue

            best_jaccard = 0.0
            best_recall = 0.0
            best_precision = 0.0
            for prog_genes in programs.values():
                prog_set = set(prog_genes) & gene_set
                if not prog_set:
                    continue
                j = jaccard_similarity(de_set, prog_set)
                overlap = len(de_set & prog_set)
                recall = overlap / len(de_set) if de_set else 0
                precision = overlap / len(prog_set) if prog_set else 0
                if j > best_jaccard:
                    best_jaccard = j
                    best_recall = recall
                    best_precision = precision

            jaccards.append(best_jaccard)
            recalls.append(best_recall)
            precisions.append(best_precision)

        if jaccards:
            rows.append({
                "method": method_name,
                "mean_jaccard": np.mean(jaccards),
                "median_jaccard": np.median(jaccards),
                "mean_recall": np.mean(recalls),
                "mean_precision": np.mean(precisions),
                "n_celltypes": len(jaccards),
                "top_jaccard": max(jaccards),
            })

    return pd.DataFrame(rows).sort_values("mean_jaccard", ascending=False)


def compute_power_analysis(
    adata: sc.AnnData,
    all_methods: dict[str, dict[str, list[str]]],
    n_trials: int = 20,
    fold_changes: list[float] | None = None,
) -> pd.DataFrame:
    """Power analysis: inject signal into a cell-type subgroup, test detection."""
    if fold_changes is None:
        fold_changes = [1.5, 2.0, 3.0]

    rng = np.random.RandomState(SEED)
    gene_names = list(adata.var_names)

    # Get marker genes for one cell type
    sc.tl.rank_genes_groups(adata, groupby="celltype", method="wilcoxon",
                            n_genes=50, use_raw=False)
    celltypes = list(adata.obs["celltype"].value_counts().index)

    # Use a mid-frequency cell type for power analysis
    target_ct = celltypes[len(celltypes) // 2]
    markers = [str(g) for g in adata.uns["rank_genes_groups"]["names"][target_ct]][:20]
    marker_idx = [gene_names.index(g) for g in markers if g in gene_names]
    logger.info("  Power analysis target: '%s', %d markers", target_ct, len(marker_idx))

    rows = []
    for method_name, programs in all_methods.items():
        for fc in fold_changes:
            tps, fps = 0, 0
            for trial in range(n_trials):
                # Sample subset
                ct_mask = adata.obs["celltype"] == target_ct
                ct_idx = np.where(ct_mask)[0]
                n_sub = min(100, len(ct_idx))
                sub_idx = rng.choice(ct_idx, n_sub, replace=False)

                # Inject fold change
                X_mod = _safe_toarray(adata.X).astype(np.float64).copy()
                for gi in marker_idx:
                    X_mod[sub_idx, gi] *= fc

                # Test each marker
                for gi in marker_idx:
                    vals_perturbed = X_mod[sub_idx, gi]
                    vals_ctrl = X_mod[~ct_mask, gi]
                    mean_diff = np.mean(vals_perturbed) - np.mean(vals_ctrl)

                    g = gene_names[gi]
                    in_program = any(g in genes for genes in programs.values())
                    if mean_diff > 0 and in_program:
                        tps += 1

                # False positives: random genes that shouldn't be affected
                non_markers = [i for i in range(len(gene_names)) if i not in marker_idx]
                for _ in range(len(marker_idx)):
                    gi = rng.choice(non_markers)
                    g = gene_names[gi]
                    in_program = any(g in genes for genes in programs.values())
                    if in_program:
                        fps += 1

            total = n_trials * len(marker_idx)
            rows.append({
                "method": method_name,
                "fold_change": fc,
                "tpr": tps / total if total > 0 else 0,
                "fpr": fps / total if total > 0 else 0,
            })

    return pd.DataFrame(rows)


# ===========================================================================
# Report
# ===========================================================================

def generate_report(
    recovery_df: pd.DataFrame,
    power_df: pd.DataFrame,
    all_methods: dict[str, dict[str, list[str]]],
    de_gene_sets: dict[str, list[str]],
    n_cells: int,
    n_genes: int,
):
    """Generate comprehensive benchmark PDF."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    recovery_df.to_csv(TABLES_DIR / "largescale_recovery.csv", index=False)
    power_df.to_csv(TABLES_DIR / "largescale_power.csv", index=False)

    emb_colors = {"PCA": "#1565C0", "scGPT": "#E65100", "Geneformer": "#2E7D32"}
    baseline_colors = {"cNMF": "#6A1B9A", "WGCNA": "#00838F", "ExprCluster": "#BF360C",
                       "Random": "#9E9E9E"}

    def get_color(name: str) -> str:
        for emb, c in emb_colors.items():
            if name.startswith(emb):
                return c
        for bl, c in baseline_colors.items():
            if bl in name:
                return c
        return "#757575"

    with PdfPages(REPORT_PATH) as pdf:
        # Title page
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.7, "Large-Scale Benchmark: PBMC 68k",
                ha="center", va="center", fontsize=24, fontweight="bold")
        ax.text(0.5, 0.55, f"{n_cells:,} cells x {n_genes:,} genes",
                ha="center", va="center", fontsize=16)
        ax.text(0.5, 0.45, f"Cell-type DE recovery ({len(de_gene_sets)} cell types)",
                ha="center", va="center", fontsize=14, color="gray")
        ax.text(0.5, 0.35, "PCA vs scGPT vs Geneformer + Graph Regularization",
                ha="center", va="center", fontsize=14, color="gray")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Recovery bar charts
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        df_s = recovery_df.sort_values("mean_jaccard", ascending=True)
        colors = [get_color(m) for m in df_s["method"]]

        ax = axes[0]
        ax.barh(range(len(df_s)), df_s["mean_jaccard"], color=colors, edgecolor="white")
        ax.set_yticks(range(len(df_s)))
        ax.set_yticklabels(df_s["method"], fontsize=8)
        ax.set_xlabel("Mean Best-Match Jaccard")
        ax.set_title("Cell-Type DE Recovery (Jaccard)")

        df_s2 = recovery_df.sort_values("mean_recall", ascending=True)
        colors2 = [get_color(m) for m in df_s2["method"]]
        ax = axes[1]
        ax.barh(range(len(df_s2)), df_s2["mean_recall"], color=colors2, edgecolor="white")
        ax.set_yticks(range(len(df_s2)))
        ax.set_yticklabels(df_s2["method"], fontsize=8)
        ax.set_xlabel("Mean Best-Match Recall")
        ax.set_title("Cell-Type DE Recovery (Recall)")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Power analysis
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax_idx, metric, title in [(0, "tpr", "True Positive Rate"),
                                       (1, "fpr", "False Positive Rate")]:
            ax = axes[ax_idx]
            for method in power_df["method"].unique():
                mdf = power_df[power_df["method"] == method]
                color = get_color(method)
                ax.plot(mdf["fold_change"], mdf[metric], "o-", color=color,
                        label=method, linewidth=1.5, markersize=4)
            ax.set_xlabel("Fold Change")
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.legend(fontsize=6, ncol=2)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Summary table
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis("off")
        cols = ["method", "mean_jaccard", "mean_recall", "mean_precision", "top_jaccard"]
        display_df = recovery_df[cols].copy()
        for c in cols[1:]:
            display_df[c] = display_df[c].map(lambda x: f"{x:.4f}")
        table = ax.table(
            cellText=display_df.values,
            colLabels=["Method", "Mean Jaccard", "Mean Recall", "Mean Precision", "Top Jaccard"],
            loc="center", cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.5)
        for j in range(len(cols)):
            table[0, j].set_facecolor("#1565C0")
            table[0, j].set_text_props(color="white", fontweight="bold")
        ax.set_title("Cell-Type DE Recovery Summary (PBMC 68k)", fontsize=14,
                     fontweight="bold", pad=20)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Method program counts
        fig, ax = plt.subplots(figsize=(12, 6))
        method_names = list(all_methods.keys())
        n_progs = [len(p) for p in all_methods.values()]
        n_genes_covered = [len({g for gl in p.values() for g in gl}) for p in all_methods.values()]
        x = range(len(method_names))
        ax.bar(x, n_progs, color=[get_color(m) for m in method_names], edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Number of Programs")
        ax.set_title("Programs Discovered per Method")
        # Add gene count labels
        for i, (np_, ng_) in enumerate(zip(n_progs, n_genes_covered)):
            ax.text(i, np_ + 0.5, f"{ng_}g", ha="center", fontsize=7)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    logger.info("Report saved: %s", REPORT_PATH)


# ===========================================================================
# Main
# ===========================================================================

def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("Large-Scale Benchmark: PBMC 68k")
    logger.info("=" * 60)

    # 1. Load and preprocess
    adata = load_pbmc68k()
    adata = preprocess_pbmc68k(adata, n_top_genes=3000)

    gene_names = list(adata.var_names)
    n_cells = adata.n_obs
    n_genes = adata.n_vars

    # 2. Compute cell-type DE genes (ground truth)
    de_gene_sets = compute_celltype_de_genes(adata, n_top_genes=100)

    # 3. Subsample for program discovery (use 5000 cells for speed)
    logger.info("Subsampling for program discovery...")
    rng = np.random.RandomState(SEED)
    n_sample = min(5000, adata.n_obs)
    idx = rng.choice(adata.n_obs, n_sample, replace=False)
    adata_sub = adata[idx].copy()
    logger.info("  Subsampled: %d cells x %d genes", adata_sub.n_obs, adata_sub.n_vars)

    # 4. Build embeddings
    all_methods: dict[str, dict[str, list[str]]] = {}

    # PCA graph-reg
    logger.info("Building PCA graph-regularized embeddings...")
    pca_emb, pca_genes = build_pca_graph_embeddings(adata_sub)
    pca_methods = discover_programs(pca_emb, pca_genes, "PCA")
    all_methods.update(pca_methods)

    # scGPT
    logger.info("Loading scGPT embeddings...")
    scgpt_result = load_fm_embeddings(gene_names, "scgpt")
    if scgpt_result is not None:
        scgpt_emb, scgpt_genes = scgpt_result
        scgpt_methods = discover_programs(scgpt_emb, scgpt_genes, "scGPT")
        all_methods.update(scgpt_methods)

    # Geneformer
    logger.info("Loading Geneformer embeddings...")
    gf_result = load_fm_embeddings(gene_names, "geneformer")
    if gf_result is not None:
        gf_emb, gf_genes = gf_result
        gf_methods = discover_programs(gf_emb, gf_genes, "Geneformer")
        all_methods.update(gf_methods)

    # Baselines on PCA embeddings
    logger.info("Running baselines...")
    X_expr = _safe_toarray(adata_sub.X).astype(np.float64)
    X_nonneg = np.clip(X_expr, 0, None) + 1e-6

    cnmf = CNMFProgramDiscovery(n_programs=N_PROGRAMS, n_iter=5,
                                top_n_genes=TOP_N_GENES, random_state=SEED)
    cnmf.fit(X_nonneg, gene_names)
    all_methods["cNMF"] = cnmf.get_programs()

    wgcna = WGCNAProgramDiscovery(n_programs=N_PROGRAMS, soft_power=6,
                                  min_module_size=5, random_state=SEED)
    wgcna.fit(X_expr, gene_names)
    all_methods["WGCNA"] = wgcna.get_programs()

    expr_cl = ExpressionClusteringBaseline(n_programs=N_PROGRAMS, random_state=SEED)
    expr_cl.fit(X_expr, gene_names)
    all_methods["ExprCluster"] = expr_cl.get_programs()

    rand = RandomProgramDiscovery(n_programs=N_PROGRAMS, genes_per_program=TOP_N_GENES,
                                  random_state=SEED)
    rand.fit(pca_emb, gene_names)
    all_methods["Random"] = rand.get_programs()

    for name, progs in all_methods.items():
        n_covered = len({g for gl in progs.values() for g in gl})
        logger.info("  %s: %d programs, %d genes", name, len(progs), n_covered)

    # 5. Evaluate cell-type DE recovery
    logger.info("Evaluating cell-type DE recovery...")
    recovery_df = evaluate_de_recovery(all_methods, de_gene_sets, gene_names)
    logger.info("\n%s", recovery_df.to_string(index=False))

    # 6. Power analysis (on subsampled data for speed)
    logger.info("Running power analysis (20 trials)...")
    power_df = compute_power_analysis(adata_sub, all_methods, n_trials=20)

    # 7. Generate report
    logger.info("Generating report...")
    generate_report(recovery_df, power_df, all_methods, de_gene_sets, n_cells, n_genes)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("LARGE-SCALE BENCHMARK COMPLETE (%.1f seconds)", elapsed)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
