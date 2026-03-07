#!/usr/bin/env python3
"""Benchmark: PCA vs Foundation Model Gene Embeddings for Program Discovery.

Compares graph-regularized embeddings derived from:
1. PCA (expression-based, current default)
2. scGPT (whole-human pretrained, 512-d token embeddings)
3. Geneformer (V1-10M pretrained, 256-d token embeddings)

Each embedding type is used with graph regularization (kNN diffusion),
then fed to the same discovery algorithms (KMeans, Leiden, Ensemble).
Evaluated on PBMC 3k against MSigDB Hallmark gene sets.
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
    load_msigdb_gene_sets,
    load_pbmc3k,
)
from npathway.data.preprocessing import (
    _safe_toarray,
    build_gene_embeddings_from_expression,
    build_graph_regularized_embeddings,
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
    jaccard_similarity,
    novelty_score,
    program_redundancy,
    program_specificity,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
REPORT_PATH = Path(__file__).parent.parent / "benchmark_fm_embeddings.pdf"
MODEL_DIR = Path(__file__).parent.parent / "models"

N_PROGRAMS = 20
TOP_N_GENES = 50
SEED = 42


# ===========================================================================
# Load FM embeddings
# ===========================================================================

def load_scgpt_embeddings(gene_names: list[str]) -> np.ndarray | None:
    """Load scGPT pretrained gene embeddings and map to dataset genes."""
    emb_path = MODEL_DIR / "scgpt_gene_embeddings.npz"
    if not emb_path.exists():
        logger.warning("scGPT embeddings not found at %s", emb_path)
        return None

    data = np.load(emb_path, allow_pickle=True)
    all_emb = data["embeddings"]  # (60697, 512)
    all_names = list(data["gene_names"])

    # Build lookup
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    name_upper_to_idx = {n.upper(): i for i, n in enumerate(all_names)}

    # Map dataset genes
    matched_idx = []
    matched_genes = []
    for g in gene_names:
        idx = name_to_idx.get(g) or name_upper_to_idx.get(g.upper())
        if idx is not None:
            matched_idx.append(idx)
            matched_genes.append(g)

    if not matched_idx:
        logger.warning("No scGPT genes matched dataset")
        return None

    embeddings = all_emb[matched_idx]
    logger.info("scGPT: matched %d/%d genes (%.1f%%), dim=%d",
                len(matched_genes), len(gene_names),
                100 * len(matched_genes) / len(gene_names), embeddings.shape[1])
    return embeddings, matched_genes


def load_geneformer_embeddings(gene_names: list[str]) -> np.ndarray | None:
    """Load Geneformer pretrained gene embeddings and map to dataset genes."""
    emb_path = MODEL_DIR / "geneformer_gene_embeddings.npz"
    if not emb_path.exists():
        logger.warning("Geneformer embeddings not found at %s", emb_path)
        return None

    data = np.load(emb_path, allow_pickle=True)
    all_emb = data["embeddings"]
    all_names = list(data["gene_names"])

    name_to_idx = {n: i for i, n in enumerate(all_names)}
    name_upper_to_idx = {n.upper(): i for i, n in enumerate(all_names)}

    matched_idx = []
    matched_genes = []
    for g in gene_names:
        idx = name_to_idx.get(g) or name_upper_to_idx.get(g.upper())
        if idx is not None:
            matched_idx.append(idx)
            matched_genes.append(g)

    if not matched_idx:
        logger.warning("No Geneformer genes matched dataset")
        return None

    embeddings = all_emb[matched_idx]
    logger.info("Geneformer: matched %d/%d genes (%.1f%%), dim=%d",
                len(matched_genes), len(gene_names),
                100 * len(matched_genes) / len(gene_names), embeddings.shape[1])
    return embeddings, matched_genes


def apply_graph_regularization(
    embeddings: np.ndarray,
    n_components: int = 50,
    k_neighbors: int = 15,
    n_diffusion_steps: int = 3,
    alpha: float = 0.5,
) -> np.ndarray:
    """Apply graph regularization (kNN diffusion) to gene embeddings."""
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    n_genes, dim = embeddings.shape

    # Clean NaN/inf values (can occur in pretrained token embeddings for rare genes)
    nan_mask = ~np.isfinite(embeddings).all(axis=1)
    if nan_mask.any():
        logger.warning("  Replacing %d genes with NaN/inf embeddings with row mean", nan_mask.sum())
        row_mean = np.nanmean(embeddings[~nan_mask], axis=0)
        embeddings[nan_mask] = row_mean
    # Final safety: replace any remaining NaN with 0
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

    # If dim > n_components, reduce first
    if dim > n_components:
        pca = PCA(n_components=n_components, random_state=SEED, svd_solver="full")
        embeddings = pca.fit_transform(embeddings.astype(np.float64))
        logger.info("  PCA reduction: %d -> %d dims (var explained: %.1f%%)",
                    dim, n_components, 100 * pca.explained_variance_ratio_.sum())

    # Build kNN graph
    nn = NearestNeighbors(n_neighbors=k_neighbors, metric="cosine", algorithm="brute")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    # Build adjacency matrix (row-normalized)
    n = embeddings.shape[0]
    adj = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j_idx, j in enumerate(indices[i]):
            if i != j:
                sim = max(0.0, 1.0 - distances[i, j_idx])
                adj[i, j] = sim
    # Symmetrize
    adj = 0.5 * (adj + adj.T)
    # Row-normalize
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    adj_norm = adj / row_sums

    # Diffusion
    result = embeddings.copy().astype(np.float64)
    for step in range(n_diffusion_steps):
        result = alpha * result + (1 - alpha) * (adj_norm @ result)

    logger.info("  Graph-reg applied: k=%d, steps=%d, alpha=%.1f",
                k_neighbors, n_diffusion_steps, alpha)
    return result.astype(np.float32)


# ===========================================================================
# Discovery methods
# ===========================================================================

def discover_programs(
    embeddings: np.ndarray,
    gene_names: list[str],
    emb_label: str,
) -> dict[str, dict[str, list[str]]]:
    """Run KMeans, Refined, Leiden, Ensemble on given embeddings."""
    methods: dict[str, dict[str, list[str]]] = {}

    # KMeans
    km = ClusteringProgramDiscovery(
        method="kmeans", n_programs=N_PROGRAMS, random_state=SEED
    )
    km.fit(embeddings, gene_names)
    methods[f"{emb_label}-KMeans"] = km.get_programs()

    # Refined
    km_confidence = km.get_gene_confidence()
    km_scores = km.get_program_scores()
    refined: dict[str, list[str]] = {}
    for prog_name, scored_genes in km_scores.items():
        conf = km_confidence.get(prog_name, {})
        combined = []
        for gene, cosine_score in scored_genes:
            conf_score = conf.get(gene, 0.5)
            combined.append((gene, (cosine_score * conf_score) ** 0.5))
        combined.sort(key=lambda x: x[1], reverse=True)
        refined[prog_name] = [g for g, _ in combined[:TOP_N_GENES]]
    methods[f"{emb_label}-Refined"] = refined

    # Leiden
    try:
        leiden = ClusteringProgramDiscovery(method="leiden", random_state=SEED)
        leiden.fit(embeddings, gene_names)
        methods[f"{emb_label}-Leiden"] = leiden.get_programs()
    except Exception as exc:
        logger.warning("Leiden failed for %s: %s", emb_label, exc)

    # Ensemble
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
        ensemble.fit(embeddings, gene_names)
        methods[f"{emb_label}-Ensemble"] = ensemble.get_programs()
    except Exception as exc:
        logger.warning("Ensemble failed for %s: %s", emb_label, exc)

    return methods


# ===========================================================================
# Evaluation
# ===========================================================================

def evaluate_programs(
    all_methods: dict[str, dict[str, list[str]]],
    hallmark: dict[str, list[str]],
    all_genes: list[str],
) -> pd.DataFrame:
    """Compute discovery quality metrics for all methods."""
    rows = []
    hallmark_all = hallmark
    for method_name, programs in all_methods.items():
        prog_genes_flat = {g for gl in programs.values() for g in gl}
        cov = len(prog_genes_flat & set(all_genes)) / max(len(all_genes), 1)
        red = program_redundancy(programs)
        spec = program_specificity(programs)
        nov = novelty_score(programs, hallmark_all)

        # Hallmark alignment (mean best-match Jaccard)
        if hallmark_all:
            overlaps = compute_overlap_matrix(programs, hallmark_all)
            align = float(overlaps.max(axis=1).mean()) if overlaps.size > 0 else 0.0
        else:
            align = 0.0

        sizes = [len(g) for g in programs.values()]
        rows.append({
            "method": method_name,
            "n_programs": len(programs),
            "coverage": cov,
            "redundancy": red,
            "novelty": nov,
            "specificity": spec,
            "mean_program_size": np.mean(sizes) if sizes else 0,
            "hallmark_alignment": align,
        })

    return pd.DataFrame(rows)


def compute_power_analysis(
    all_methods: dict[str, dict[str, list[str]]],
    adata,
    hallmark: dict[str, list[str]],
    gene_names: list[str],
    n_trials: int = 30,
) -> pd.DataFrame:
    """Statistical power analysis: can programs recover injected markers?"""
    import scanpy as sc

    from npathway.evaluation.enrichment import preranked_gsea

    # Extract cell type markers
    adata_work = adata.copy()
    if hasattr(adata_work, "raw") and adata_work.raw is not None:
        adata_de = adata_work.raw.to_adata()
        adata_de.obs = adata_work.obs
    else:
        adata_de = adata_work

    x_matrix = adata_de.X
    if hasattr(x_matrix, "toarray"):
        if np.issubdtype(x_matrix.dtype, np.number):
            pass
        else:
            adata_de.X = x_matrix.toarray().astype(np.float64)
    elif not np.issubdtype(x_matrix.dtype, np.number):
        adata_de.X = x_matrix.astype(np.float64)

    cell_type_key = None
    for key in ("louvain", "leiden", "cell_type", "celltype"):
        if key in adata_de.obs.columns:
            cell_type_key = key
            break
    if cell_type_key is None:
        logger.warning("No cell type annotation found for power analysis")
        return pd.DataFrame()

    logger.info("Extracting markers for '%s' groups using wilcoxon test ...", cell_type_key)
    sc.tl.rank_genes_groups(adata_de, groupby=cell_type_key, method="wilcoxon",
                           n_genes=30, use_raw=False)
    result = adata_de.uns["rank_genes_groups"]
    groups = list(result["names"].dtype.names)
    marker_sets: dict[str, list[str]] = {}
    for grp in groups:
        markers = [str(g) for g in result["names"][grp] if str(g) in set(gene_names)]
        if markers:
            marker_sets[grp] = markers
    all_markers = [g for gl in marker_sets.values() for g in gl]
    logger.info("Extracted markers for %d groups (%d total marker genes).",
                len(marker_sets), len(all_markers))

    # Pick a target group (Megakaryocytes if available, else smallest)
    target = None
    for candidate in ["Megakaryocytes", "Dendritic cells"]:
        if candidate in marker_sets:
            target = candidate
            break
    if target is None:
        target = min(marker_sets, key=lambda k: len(marker_sets[k]))
    target_markers = set(marker_sets[target])
    logger.info("  Target: %s markers (%d genes)", target, len(target_markers))

    fold_changes = [1.1, 1.3, 1.5, 2.0, 3.0]
    rng = np.random.RandomState(SEED)
    gene_set = set(gene_names)
    rows = []

    for fc in fold_changes:
        for method_name, programs in all_methods.items():
            tp_total = 0
            fp_total = 0
            n_pos = 0
            n_neg = 0
            for trial in range(n_trials):
                # Simulate ranked list: markers boosted by fc, rest ~N(0,1)
                scores = {g: rng.normal(0, 1) for g in gene_names}
                for g in target_markers:
                    if g in scores:
                        scores[g] = rng.normal(0, 1) + np.log2(fc)
                ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

                # Test enrichment
                for prog_name, prog_genes in programs.items():
                    valid = [g for g in prog_genes if g in gene_set]
                    if len(valid) < 5:
                        continue
                    res = preranked_gsea(ranked, {prog_name: valid}, n_perm=200)
                    sig = (res["fdr"].iloc[0] < 0.05) if len(res) > 0 else False
                    has_marker = len(set(valid) & target_markers) >= 2
                    if has_marker:
                        n_pos += 1
                        if sig:
                            tp_total += 1
                    else:
                        n_neg += 1
                        if sig:
                            fp_total += 1

            tpr = tp_total / max(n_pos, 1)
            fpr = fp_total / max(n_neg, 1)
            rows.append({"method": method_name, "fold_change": fc, "tpr": tpr, "fpr": fpr})

    return pd.DataFrame(rows)


# ===========================================================================
# Cross-embedding agreement analysis
# ===========================================================================

def cross_embedding_agreement(
    all_methods: dict[str, dict[str, list[str]]],
    emb_types: list[str],
) -> pd.DataFrame:
    """Compute agreement between same method using different embeddings."""
    rows = []
    discovery_methods = ["KMeans", "Refined", "Leiden", "Ensemble"]
    for dm in discovery_methods:
        for i, emb_a in enumerate(emb_types):
            for emb_b in emb_types[i + 1:]:
                key_a = f"{emb_a}-{dm}"
                key_b = f"{emb_b}-{dm}"
                if key_a in all_methods and key_b in all_methods:
                    progs_a = all_methods[key_a]
                    progs_b = all_methods[key_b]
                    # Compute best-match Jaccard
                    jaccards = []
                    for pa_genes in progs_a.values():
                        best_j = 0.0
                        for pb_genes in progs_b.values():
                            j = jaccard_similarity(set(pa_genes), set(pb_genes))
                            if j > best_j:
                                best_j = j
                        jaccards.append(best_j)
                    mean_j = float(np.mean(jaccards)) if jaccards else 0.0
                    rows.append({
                        "discovery_method": dm,
                        "embedding_a": emb_a,
                        "embedding_b": emb_b,
                        "mean_best_jaccard": mean_j,
                        "n_programs_a": len(progs_a),
                        "n_programs_b": len(progs_b),
                    })
    return pd.DataFrame(rows)


# ===========================================================================
# PDF generation
# ===========================================================================

def generate_report(
    quality_df: pd.DataFrame,
    power_df: pd.DataFrame,
    agreement_df: pd.DataFrame,
    emb_types: list[str],
):
    """Generate benchmark PDF with comparison plots."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Save tables
    quality_df.to_csv(TABLES_DIR / "fm_embedding_quality.csv", index=False)
    if not power_df.empty:
        power_df.to_csv(TABLES_DIR / "fm_embedding_power.csv", index=False)
    if not agreement_df.empty:
        agreement_df.to_csv(TABLES_DIR / "fm_embedding_agreement.csv", index=False)

    # Color mapping by embedding source
    emb_colors = {
        "PCA": "#1565C0",
        "scGPT": "#E65100",
        "Geneformer": "#2E7D32",
    }
    # Method color: derive from embedding + discovery method
    def get_color(method_name: str) -> str:
        for emb_name, color in emb_colors.items():
            if method_name.startswith(emb_name):
                return color
        if method_name in ("cNMF", "WGCNA", "Expr-Cluster", "Random"):
            return "#757575"
        return "#000000"

    with PdfPages(REPORT_PATH) as pdf:
        # --- Page 1: Title ---
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.7, "nPathway: FM Embedding Benchmark",
                ha="center", va="center", fontsize=24, fontweight="bold")
        ax.text(0.5, 0.55, "PCA vs scGPT vs Geneformer Gene Embeddings",
                ha="center", va="center", fontsize=16)
        ax.text(0.5, 0.4, f"PBMC 3k | MSigDB Hallmark | {len(quality_df)} methods",
                ha="center", va="center", fontsize=14, color="gray")
        ax.text(0.5, 0.25, "2026-03-03", ha="center", va="center", fontsize=12)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # --- Page 2: Hallmark Alignment comparison ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Bar chart: Hallmark alignment by method
        ax = axes[0]
        df_sorted = quality_df.sort_values("hallmark_alignment", ascending=True)
        colors = [get_color(m) for m in df_sorted["method"]]
        ax.barh(range(len(df_sorted)), df_sorted["hallmark_alignment"], color=colors, edgecolor="white")
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted["method"], fontsize=8)
        ax.set_xlabel("Mean Hallmark Alignment (best-match Jaccard)")
        ax.set_title("Hallmark Alignment by Method")
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

        # Bar chart: Coverage
        ax = axes[1]
        colors = [get_color(m) for m in df_sorted["method"]]
        ax.barh(range(len(df_sorted)), df_sorted["coverage"], color=colors, edgecolor="white")
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted["method"], fontsize=8)
        ax.set_xlabel("Gene Coverage")
        ax.set_title("Gene Coverage by Method")

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # --- Page 3: Grouped comparison (same discovery method, different embeddings) ---
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        metrics = ["hallmark_alignment", "coverage", "redundancy", "specificity"]
        titles = ["Hallmark Alignment", "Coverage", "Redundancy", "Specificity"]

        for ax, metric, title in zip(axes.flat, metrics, titles):
            # Group by discovery method suffix
            groups = {}
            for _, row in quality_df.iterrows():
                name = row["method"]
                for dm in ["KMeans", "Refined", "Leiden", "Ensemble"]:
                    if name.endswith(dm):
                        if dm not in groups:
                            groups[dm] = {}
                        emb = name.replace(f"-{dm}", "")
                        groups[dm][emb] = row[metric]

            # Add baselines
            for _, row in quality_df.iterrows():
                if row["method"] in ("cNMF", "WGCNA", "Expr-Cluster", "Random"):
                    groups.setdefault("Baselines", {})[row["method"]] = row[metric]

            # Plot grouped bars
            x_labels = list(groups.keys())
            x = np.arange(len(x_labels))
            width = 0.15
            for i, emb in enumerate(emb_types):
                vals = [groups.get(dm, {}).get(emb, 0) for dm in x_labels]
                if any(v > 0 for v in vals):
                    ax.bar(x + i * width, vals, width, label=emb,
                           color=emb_colors.get(emb, "#757575"), alpha=0.85)

            # Baselines
            if "Baselines" in groups:
                bl_idx = x_labels.index("Baselines")
                for j, bl_name in enumerate(["cNMF", "WGCNA", "Expr-Cluster", "Random"]):
                    val = groups["Baselines"].get(bl_name, 0)
                    if val > 0:
                        ax.bar(bl_idx + j * width, val, width, label=bl_name,
                               color="#757575", alpha=0.5 + 0.1 * j)

            ax.set_xticks(x + width)
            ax.set_xticklabels(x_labels, fontsize=9)
            ax.set_title(title)
            if ax == axes[0, 0]:
                ax.legend(fontsize=7, loc="upper left")

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # --- Page 4: Power analysis ---
        if not power_df.empty:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # TPR curves
            ax = axes[0]
            for method_name in power_df["method"].unique():
                mdf = power_df[power_df["method"] == method_name]
                color = get_color(method_name)
                ls = "-" if any(method_name.startswith(e) for e in emb_types) else "--"
                ax.plot(mdf["fold_change"], mdf["tpr"], "o-", color=color,
                        label=method_name, markersize=4, linewidth=1.5, linestyle=ls)
            ax.set_xlabel("Fold Change")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("Statistical Power: Marker Recovery")
            ax.legend(fontsize=6, ncol=2, loc="upper left")
            ax.grid(alpha=0.3)

            # FPR curves
            ax = axes[1]
            for method_name in power_df["method"].unique():
                mdf = power_df[power_df["method"] == method_name]
                color = get_color(method_name)
                ls = "-" if any(method_name.startswith(e) for e in emb_types) else "--"
                ax.plot(mdf["fold_change"], mdf["fpr"], "o-", color=color,
                        label=method_name, markersize=4, linewidth=1.5, linestyle=ls)
            ax.set_xlabel("Fold Change")
            ax.set_ylabel("False Positive Rate")
            ax.set_title("False Positive Rate")
            ax.legend(fontsize=6, ncol=2, loc="upper left")
            ax.grid(alpha=0.3)

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # --- Page 5: Cross-embedding agreement ---
        if not agreement_df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            pivot = agreement_df.pivot_table(
                index="discovery_method",
                columns=agreement_df.apply(
                    lambda r: f"{r['embedding_a']} vs {r['embedding_b']}", axis=1
                ),
                values="mean_best_jaccard",
            )
            if not pivot.empty:
                sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd",
                            ax=ax, vmin=0, vmax=0.5)
                ax.set_title("Cross-Embedding Agreement (Mean Best-Match Jaccard)")
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # --- Page 6: Summary table ---
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis("off")
        cols = ["method", "n_programs", "coverage", "redundancy",
                "specificity", "hallmark_alignment"]
        display_df = quality_df[cols].copy()
        for c in ["coverage", "redundancy", "specificity", "hallmark_alignment"]:
            display_df[c] = display_df[c].map(lambda x: f"{x:.3f}")
        table = ax.table(
            cellText=display_df.values,
            colLabels=display_df.columns,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.4)

        # Color header
        for j in range(len(display_df.columns)):
            table[0, j].set_facecolor("#1565C0")
            table[0, j].set_text_props(color="white", fontweight="bold")

        # Highlight best in hallmark_alignment column
        best_val = quality_df["hallmark_alignment"].max()
        for i, val in enumerate(quality_df["hallmark_alignment"]):
            if abs(val - best_val) < 1e-6:
                table[i + 1, 5].set_facecolor("#FFECB3")

        ax.set_title("FM Embedding Benchmark: Discovery Quality Summary",
                     fontsize=14, fontweight="bold", pad=20)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    logger.info("Report saved: %s", REPORT_PATH)


# ===========================================================================
# Main
# ===========================================================================

def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("nPathway FM Embedding Benchmark")
    logger.info("=" * 60)

    # 1. Load PBMC 3k
    logger.info("Loading PBMC 3k dataset...")
    adata = load_pbmc3k(preprocessed=True)
    gene_names = list(adata.var_names)
    logger.info("  %d cells x %d genes", adata.n_obs, adata.n_vars)

    X_expr = _safe_toarray(adata.X).astype(np.float64)
    X_nonneg = np.clip(X_expr, 0, None) + 1e-6

    # 2. Load Hallmark
    hallmark_raw = load_msigdb_gene_sets(collection="hallmark")
    hallmark = filter_gene_sets_to_adata(hallmark_raw, adata, min_genes=3)
    logger.info("  %d Hallmark sets retained", len(hallmark))

    # 3. Build embeddings from each source
    all_methods: dict[str, dict[str, list[str]]] = {}
    emb_types_available: list[str] = []

    # --- PCA (baseline) ---
    logger.info("Building PCA graph-regularized embeddings...")
    pca_emb, pca_genes = build_gene_embeddings_from_expression(adata, n_components=50)
    pca_graph, _ = build_graph_regularized_embeddings(
        adata, n_components=50, k_neighbors=15, n_diffusion_steps=3, alpha=0.5,
    )
    logger.info("  PCA graph-reg: %s", pca_graph.shape)
    pca_methods = discover_programs(pca_graph, gene_names, "PCA")
    all_methods.update(pca_methods)
    emb_types_available.append("PCA")

    # --- scGPT ---
    logger.info("Loading scGPT embeddings...")
    scgpt_result = load_scgpt_embeddings(gene_names)
    if scgpt_result is not None:
        scgpt_raw, scgpt_genes = scgpt_result
        logger.info("  Applying graph regularization to scGPT embeddings...")
        scgpt_graph = apply_graph_regularization(scgpt_raw, n_components=50)
        scgpt_methods = discover_programs(scgpt_graph, scgpt_genes, "scGPT")
        all_methods.update(scgpt_methods)
        emb_types_available.append("scGPT")

    # --- Geneformer ---
    logger.info("Loading Geneformer embeddings...")
    gf_result = load_geneformer_embeddings(gene_names)
    if gf_result is not None:
        gf_raw, gf_genes = gf_result
        logger.info("  Applying graph regularization to Geneformer embeddings...")
        gf_graph = apply_graph_regularization(gf_raw, n_components=50)
        gf_methods = discover_programs(gf_graph, gf_genes, "Geneformer")
        all_methods.update(gf_methods)
        emb_types_available.append("Geneformer")

    # --- Baselines (expression-based, no graph regularization) ---
    logger.info("Running baselines...")

    # cNMF
    cnmf = CNMFProgramDiscovery(
        n_programs=N_PROGRAMS, n_iter=5, top_n_genes=TOP_N_GENES, random_state=SEED,
    )
    cnmf.fit(X_nonneg, gene_names)
    all_methods["cNMF"] = cnmf.get_programs()

    # WGCNA
    wgcna = WGCNAProgramDiscovery(
        n_programs=N_PROGRAMS, soft_power=6, min_module_size=5, random_state=SEED,
    )
    wgcna.fit(X_expr, gene_names)
    all_methods["WGCNA"] = wgcna.get_programs()

    # Expr-Cluster
    expr_cl = ExpressionClusteringBaseline(n_programs=N_PROGRAMS, random_state=SEED)
    expr_cl.fit(X_expr, gene_names)
    all_methods["Expr-Cluster"] = expr_cl.get_programs()

    # Random
    rand = RandomProgramDiscovery(
        n_programs=N_PROGRAMS, genes_per_program=TOP_N_GENES, random_state=SEED,
    )
    rand.fit(pca_emb, gene_names)
    all_methods["Random"] = rand.get_programs()

    # Log summary
    for name, progs in all_methods.items():
        n_covered = len({g for gl in progs.values() for g in gl})
        logger.info("  %s: %d programs, %d genes", name, len(progs), n_covered)

    # 4. Evaluate
    logger.info("Evaluating discovery quality...")
    quality_df = evaluate_programs(all_methods, hallmark, gene_names)
    logger.info("\n%s", quality_df.to_string(index=False))

    # 5. Power analysis
    logger.info("Running power analysis (30 trials)...")
    power_df = compute_power_analysis(all_methods, adata, hallmark, gene_names, n_trials=30)

    # 6. Cross-embedding agreement
    logger.info("Computing cross-embedding agreement...")
    agreement_df = cross_embedding_agreement(all_methods, emb_types_available)
    if not agreement_df.empty:
        logger.info("\n%s", agreement_df.to_string(index=False))

    # 7. Generate report
    logger.info("Generating report...")
    generate_report(quality_df, power_df, agreement_df, emb_types_available)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("FM EMBEDDING BENCHMARK COMPLETE (%.1f seconds)", elapsed)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
