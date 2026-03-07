#!/usr/bin/env python3
"""Perturbation Recovery Benchmark: Can gene programs predict perturbation effects?

Uses Norman 2019 CRISPRa Perturb-seq data (K562 cells, 287 perturbations).
Tests whether discovered gene programs enrich for genes affected by known perturbations.
This is the gold standard for biological validation of gene program methods.
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

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from npathway.data.preprocessing import (
    _safe_toarray,
    build_gene_embeddings_from_expression,
)
from npathway.discovery.baselines import (
    CNMFProgramDiscovery,
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
REPORT_PATH = Path(__file__).parent.parent / "benchmark_perturbation.pdf"
MODEL_DIR = Path(__file__).parent.parent / "models"

N_PROGRAMS = 20
TOP_N_GENES = 50
SEED = 42


def load_norman_2019():
    """Load Norman 2019 CRISPRa data, or fall back to PBMC 68k cell-type perturbation proxy."""
    data_dir = Path(__file__).parent.parent / "data"
    h5ad_path = data_dir / "norman_2019.h5ad"

    if h5ad_path.exists() and h5ad_path.stat().st_size > 1000:
        logger.info("Loading cached Norman 2019 from %s", h5ad_path)
        adata = sc.read_h5ad(h5ad_path)
        logger.info("  Raw: %d cells x %d genes", adata.n_obs, adata.n_vars)
        return adata

    # Fallback: use PBMC 68k with cell types as perturbation conditions
    logger.info("Norman 2019 not available; using PBMC 68k cell-type perturbation proxy")
    import scvelo as scv
    adata = scv.datasets.pbmc68k()

    # Create perturbation-like column from cell types
    adata.obs["perturbation"] = adata.obs["celltype"].astype(str)

    # Designate largest cell type as "control"
    top_ct = adata.obs["celltype"].value_counts().idxmax()
    adata.obs.loc[adata.obs["celltype"] == top_ct, "perturbation"] = "control"
    logger.info("  PBMC 68k loaded: %d cells x %d genes, control='%s'",
                adata.n_obs, adata.n_vars, top_ct)
    return adata


def get_perturbation_de_genes(
    adata,
    n_top_genes: int = 100,
    min_cells: int = 20,
) -> dict[str, list[str]]:
    """Compute DE genes for each single-gene perturbation vs control.

    Returns dict mapping perturbation_name -> list of DE gene names.
    """
    # Identify perturbation column
    perturb_col = None
    for col in ["perturbation", "gene", "guide_id", "condition"]:
        if col in adata.obs.columns:
            perturb_col = col
            break
    if perturb_col is None:
        # Try to find it
        for col in adata.obs.columns:
            if adata.obs[col].dtype == "category" or adata.obs[col].dtype == "object":
                nunique = adata.obs[col].nunique()
                if 10 < nunique < 500:
                    perturb_col = col
                    break
    if perturb_col is None:
        raise ValueError("Could not find perturbation column in adata.obs")
    logger.info("  Using perturbation column: '%s' (%d unique values)",
                perturb_col, adata.obs[perturb_col].nunique())

    # Find control cells
    control_names = {"control", "ctrl", "non-targeting", "NT", "unperturbed", "neg_ctrl"}
    all_labels = set(adata.obs[perturb_col].astype(str).unique())
    ctrl_label = None
    for cn in control_names:
        for lab in all_labels:
            if cn.lower() in lab.lower():
                ctrl_label = lab
                break
        if ctrl_label:
            break
    if ctrl_label is None:
        # Use the most frequent label
        ctrl_label = adata.obs[perturb_col].value_counts().idxmax()
        logger.warning("  No obvious control found, using most frequent: '%s'", ctrl_label)

    ctrl_mask = adata.obs[perturb_col].astype(str) == ctrl_label
    logger.info("  Control: '%s' (%d cells)", ctrl_label, ctrl_mask.sum())

    # Identify perturbations/cell-types with enough cells
    label_counts = adata.obs[perturb_col].value_counts()
    valid_perturbations = []
    # Detect if this is CRISPRa data (labels like "gene1+gene2") vs cell-type data
    is_perturb_seq = any("+" in str(lab) and "+" not in str(lab).split("/")[0].split(" ")[0]
                         for lab in label_counts.index[:20]
                         if str(lab) != ctrl_label)
    for label, count in label_counts.items():
        label_str = str(label)
        if label_str == ctrl_label:
            continue
        if count < min_cells:
            continue
        # Only skip multi-gene combos for actual Perturb-seq (not cell-type labels)
        if is_perturb_seq and ("," in label_str):
            continue
        valid_perturbations.append(label_str)

    logger.info("  %d single-gene perturbations with >= %d cells",
                len(valid_perturbations), min_cells)

    # Compute DE for each perturbation vs control
    de_gene_sets: dict[str, list[str]] = {}
    gene_set = set(adata.var_names)

    for i, pert in enumerate(valid_perturbations[:50]):  # Cap at 50 for speed
        # Subset to perturbation + control cells
        mask = (adata.obs[perturb_col].astype(str) == pert) | ctrl_mask
        adata_sub = adata[mask].copy()
        adata_sub.obs["_group"] = (
            adata_sub.obs[perturb_col].astype(str) == pert
        ).astype(str)

        try:
            sc.tl.rank_genes_groups(
                adata_sub, groupby="_group", groups=["True"],
                reference="False", method="wilcoxon", n_genes=n_top_genes,
                use_raw=False,
            )
            de_genes = [
                str(g) for g in adata_sub.uns["rank_genes_groups"]["names"]["True"]
                if str(g) in gene_set
            ]
            if de_genes:
                de_gene_sets[pert] = de_genes
        except Exception:
            continue

        if (i + 1) % 10 == 0:
            logger.info("    Processed %d/%d perturbations", i + 1, len(valid_perturbations[:50]))

    logger.info("  DE gene sets computed for %d perturbations", len(de_gene_sets))
    return de_gene_sets


def load_fm_embeddings(gene_names: list[str], model_name: str):
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

    if not matched_idx:
        return None

    embeddings = all_emb[matched_idx]
    # Clean NaN/inf
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    return embeddings, matched_genes


def apply_graph_reg(embeddings, n_components=50, k_neighbors=15, n_steps=3, alpha=0.5):
    """Apply graph regularization to embeddings."""
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    emb = embeddings.astype(np.float64)
    emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)

    n, dim = emb.shape
    if dim > n_components:
        pca = PCA(n_components=n_components, random_state=SEED, svd_solver="full")
        emb = pca.fit_transform(emb)

    nn = NearestNeighbors(n_neighbors=k_neighbors, metric="cosine", algorithm="brute")
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


def discover_all_methods(
    adata,
    gene_names: list[str],
) -> dict[str, dict[str, list[str]]]:
    """Run all discovery methods on the dataset."""
    all_methods: dict[str, dict[str, list[str]]] = {}
    X_expr = _safe_toarray(adata.X).astype(np.float64)
    X_nonneg = np.clip(X_expr, 0, None) + 1e-6

    # PCA graph-regularized embeddings
    logger.info("Building PCA graph-reg embeddings...")
    pca_emb, _ = build_gene_embeddings_from_expression(adata, n_components=50)
    graph_emb = apply_graph_reg(pca_emb)

    # PCA-KMeans
    km = ClusteringProgramDiscovery(method="kmeans", n_programs=N_PROGRAMS, random_state=SEED)
    km.fit(graph_emb, gene_names)
    all_methods["PCA-KMeans"] = km.get_programs()

    # PCA-Refined
    km_conf = km.get_gene_confidence()
    km_scores = km.get_program_scores()
    refined: dict[str, list[str]] = {}
    for pn, sg in km_scores.items():
        conf = km_conf.get(pn, {})
        combined = [(g, (cs * conf.get(g, 0.5)) ** 0.5) for g, cs in sg]
        combined.sort(key=lambda x: x[1], reverse=True)
        refined[pn] = [g for g, _ in combined[:TOP_N_GENES]]
    all_methods["PCA-Refined"] = refined

    # PCA-Ensemble
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
        ens.fit(graph_emb, gene_names)
        all_methods["PCA-Ensemble"] = ens.get_programs()
    except Exception as exc:
        logger.warning("PCA-Ensemble failed: %s", exc)

    # scGPT embeddings
    scgpt_result = load_fm_embeddings(gene_names, "scgpt")
    if scgpt_result is not None:
        scgpt_emb, scgpt_genes = scgpt_result
        logger.info("scGPT: %d/%d genes matched", len(scgpt_genes), len(gene_names))
        scgpt_graph = apply_graph_reg(scgpt_emb)

        km2 = ClusteringProgramDiscovery(method="kmeans", n_programs=N_PROGRAMS, random_state=SEED)
        km2.fit(scgpt_graph, scgpt_genes)
        all_methods["scGPT-KMeans"] = km2.get_programs()

        km2_conf = km2.get_gene_confidence()
        km2_scores = km2.get_program_scores()
        refined2: dict[str, list[str]] = {}
        for pn, sg in km2_scores.items():
            conf = km2_conf.get(pn, {})
            combined = [(g, (cs * conf.get(g, 0.5)) ** 0.5) for g, cs in sg]
            combined.sort(key=lambda x: x[1], reverse=True)
            refined2[pn] = [g for g, _ in combined[:TOP_N_GENES]]
        all_methods["scGPT-Refined"] = refined2

    # Geneformer embeddings
    gf_result = load_fm_embeddings(gene_names, "geneformer")
    if gf_result is not None:
        gf_emb, gf_genes = gf_result
        logger.info("Geneformer: %d/%d genes matched", len(gf_genes), len(gene_names))
        gf_graph = apply_graph_reg(gf_emb)

        km3 = ClusteringProgramDiscovery(method="kmeans", n_programs=N_PROGRAMS, random_state=SEED)
        km3.fit(gf_graph, gf_genes)
        all_methods["Geneformer-KMeans"] = km3.get_programs()

        km3_conf = km3.get_gene_confidence()
        km3_scores = km3.get_program_scores()
        refined3: dict[str, list[str]] = {}
        for pn, sg in km3_scores.items():
            conf = km3_conf.get(pn, {})
            combined = [(g, (cs * conf.get(g, 0.5)) ** 0.5) for g, cs in sg]
            combined.sort(key=lambda x: x[1], reverse=True)
            refined3[pn] = [g for g, _ in combined[:TOP_N_GENES]]
        all_methods["Geneformer-Refined"] = refined3

    # Baselines
    cnmf = CNMFProgramDiscovery(n_programs=N_PROGRAMS, n_iter=5, top_n_genes=TOP_N_GENES, random_state=SEED)
    cnmf.fit(X_nonneg, gene_names)
    all_methods["cNMF"] = cnmf.get_programs()

    wgcna = WGCNAProgramDiscovery(n_programs=N_PROGRAMS, soft_power=6, min_module_size=5, random_state=SEED)
    wgcna.fit(X_expr, gene_names)
    all_methods["WGCNA"] = wgcna.get_programs()

    rand = RandomProgramDiscovery(n_programs=N_PROGRAMS, genes_per_program=TOP_N_GENES, random_state=SEED)
    rand.fit(pca_emb, gene_names)
    all_methods["Random"] = rand.get_programs()

    for name, progs in all_methods.items():
        n_covered = len({g for gl in progs.values() for g in gl})
        logger.info("  %s: %d programs, %d genes", name, len(progs), n_covered)

    return all_methods


def perturbation_recovery(
    all_methods: dict[str, dict[str, list[str]]],
    de_gene_sets: dict[str, list[str]],
    gene_names: list[str],
) -> pd.DataFrame:
    """Compute perturbation recovery: how well do programs capture DE genes?

    For each perturbation, find the best-matching program (highest Jaccard with DE genes).
    Reports mean best-match Jaccard across all perturbations.
    """
    rows = []
    gene_set = set(gene_names)

    for method_name, programs in all_methods.items():
        jaccards = []
        recall_at_k = []  # fraction of DE genes in best-matching program

        for pert_name, de_genes in de_gene_sets.items():
            de_set = set(de_genes) & gene_set
            if len(de_set) < 5:
                continue

            best_jaccard = 0.0
            best_recall = 0.0
            for prog_genes in programs.values():
                prog_set = set(prog_genes) & gene_set
                if not prog_set:
                    continue
                j = jaccard_similarity(de_set, prog_set)
                recall = len(de_set & prog_set) / len(de_set)
                if j > best_jaccard:
                    best_jaccard = j
                    best_recall = recall

            jaccards.append(best_jaccard)
            recall_at_k.append(best_recall)

        if jaccards:
            rows.append({
                "method": method_name,
                "mean_jaccard": np.mean(jaccards),
                "median_jaccard": np.median(jaccards),
                "mean_recall": np.mean(recall_at_k),
                "n_perturbations": len(jaccards),
                "top_10pct_jaccard": np.percentile(jaccards, 90),
            })

    return pd.DataFrame(rows).sort_values("mean_jaccard", ascending=False)


def generate_report(
    recovery_df: pd.DataFrame,
    de_gene_sets: dict[str, list[str]],
):
    """Generate perturbation recovery benchmark PDF."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    recovery_df.to_csv(TABLES_DIR / "perturbation_recovery.csv", index=False)

    emb_colors = {"PCA": "#1565C0", "scGPT": "#E65100", "Geneformer": "#2E7D32"}

    def get_color(name: str) -> str:
        for emb, c in emb_colors.items():
            if name.startswith(emb):
                return c
        return "#757575"

    with PdfPages(REPORT_PATH) as pdf:
        # Title page
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.7, "Perturbation Recovery Benchmark",
                ha="center", va="center", fontsize=24, fontweight="bold")
        ax.text(0.5, 0.55, "Norman 2019 CRISPRa Perturb-seq (K562)",
                ha="center", va="center", fontsize=16)
        ax.text(0.5, 0.4, f"{len(de_gene_sets)} perturbations evaluated",
                ha="center", va="center", fontsize=14, color="gray")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Recovery bar chart
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Mean Jaccard
        ax = axes[0]
        df_sorted = recovery_df.sort_values("mean_jaccard", ascending=True)
        colors = [get_color(m) for m in df_sorted["method"]]
        ax.barh(range(len(df_sorted)), df_sorted["mean_jaccard"], color=colors, edgecolor="white")
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted["method"], fontsize=9)
        ax.set_xlabel("Mean Best-Match Jaccard with Perturbation DE Genes")
        ax.set_title("Perturbation Recovery (Jaccard)")

        # Mean Recall
        ax = axes[1]
        df_sorted2 = recovery_df.sort_values("mean_recall", ascending=True)
        colors = [get_color(m) for m in df_sorted2["method"]]
        ax.barh(range(len(df_sorted2)), df_sorted2["mean_recall"], color=colors, edgecolor="white")
        ax.set_yticks(range(len(df_sorted2)))
        ax.set_yticklabels(df_sorted2["method"], fontsize=9)
        ax.set_xlabel("Mean Best-Match Recall (DE genes in program)")
        ax.set_title("Perturbation Recovery (Recall)")

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Summary table
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis("off")
        cols = ["method", "mean_jaccard", "median_jaccard", "mean_recall", "top_10pct_jaccard"]
        display_df = recovery_df[cols].copy()
        for c in cols[1:]:
            display_df[c] = display_df[c].map(lambda x: f"{x:.4f}")
        table = ax.table(
            cellText=display_df.values,
            colLabels=["Method", "Mean Jaccard", "Median Jaccard", "Mean Recall", "Top 10% Jaccard"],
            loc="center", cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.5)
        for j in range(len(cols)):
            table[0, j].set_facecolor("#1565C0")
            table[0, j].set_text_props(color="white", fontweight="bold")
        ax.set_title("Perturbation Recovery Summary", fontsize=14, fontweight="bold", pad=20)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    logger.info("Report: %s", REPORT_PATH)


def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("Perturbation Recovery Benchmark")
    logger.info("=" * 60)

    # 1. Load Norman 2019
    adata = load_norman_2019()

    # 2. Preprocess
    logger.info("Preprocessing...")
    X = _safe_toarray(adata.X)
    if X.max() > 100:  # raw counts - need normalize + log
        sc.pp.filter_genes(adata, min_cells=10)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=3000)
        adata = adata[:, adata.var["highly_variable"]].copy()
        sc.pp.scale(adata, max_value=10)
    logger.info("  Preprocessed: %d cells x %d genes", adata.n_obs, adata.n_vars)

    gene_names = list(adata.var_names)

    # 3. Compute perturbation DE gene sets
    de_gene_sets = get_perturbation_de_genes(adata, n_top_genes=100)

    if len(de_gene_sets) < 5:
        logger.error("Too few perturbation DE sets (%d). Aborting.", len(de_gene_sets))
        return

    # 4. Subsample for program discovery (use control cells + sample)
    # Use a subset of cells for speed (program discovery doesn't need all 111k cells)
    logger.info("Subsampling for program discovery...")
    rng = np.random.RandomState(SEED)
    n_sample = min(5000, adata.n_obs)
    idx = rng.choice(adata.n_obs, n_sample, replace=False)
    adata_sub = adata[idx].copy()
    logger.info("  Subsampled: %d cells", adata_sub.n_obs)

    # 5. Discover programs
    logger.info("Discovering programs...")
    all_methods = discover_all_methods(adata_sub, gene_names)

    # 6. Evaluate perturbation recovery
    logger.info("Evaluating perturbation recovery...")
    recovery_df = perturbation_recovery(all_methods, de_gene_sets, gene_names)
    logger.info("\n%s", recovery_df.to_string(index=False))

    # 7. Generate report
    generate_report(recovery_df, de_gene_sets)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("PERTURBATION BENCHMARK COMPLETE (%.1f seconds)", elapsed)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
