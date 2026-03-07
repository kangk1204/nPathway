"""Preprocessing utilities for scRNA-seq data.

This module implements a standard scanpy-based preprocessing pipeline,
gene-gene correlation computation, differential-expression-based marker
gene extraction, pseudobulk aggregation, and expression-based gene
embedding construction.

All functions accept ``AnnData`` objects and are designed to work
seamlessly with the dataset loaders in :mod:`npathway.data.datasets`.
"""

from __future__ import annotations

import logging
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

logger = logging.getLogger(__name__)


# ======================================================================
# Standard preprocessing pipeline
# ======================================================================


def preprocess_adata(
    adata: ad.AnnData,
    *,
    min_genes: int = 200,
    min_cells: int = 3,
    max_pct_mito: float = 20.0,
    n_top_genes: int = 2000,
    n_pcs: int = 50,
    n_neighbors: int = 15,
    copy: bool = True,
) -> ad.AnnData:
    """Run a standard scanpy preprocessing pipeline.

    Steps:
    1. Filter cells with fewer than *min_genes* detected genes.
    2. Filter genes detected in fewer than *min_cells*.
    3. (Optional) Filter cells with high mitochondrial content.
    4. Normalize total counts per cell to 10 000.
    5. Log-transform (``log1p``).
    6. Identify *n_top_genes* highly variable genes.
    7. Subset to HVGs.
    8. Scale to unit variance and zero mean.
    9. PCA (first *n_pcs* components).
    10. Compute *n_neighbors* nearest-neighbor graph.

    Args:
        adata: Raw AnnData object (counts in ``adata.X``).
        min_genes: Minimum number of genes per cell.
        min_cells: Minimum number of cells per gene.
        max_pct_mito: Maximum percentage of mitochondrial reads per cell.
            Set to ``100.0`` to disable filtering.
        n_top_genes: Number of highly variable genes to retain.
        n_pcs: Number of principal components.
        n_neighbors: Number of neighbors for the kNN graph.
        copy: If ``True``, operate on a copy; otherwise modify in place.

    Returns:
        Preprocessed AnnData with PCA embeddings in ``adata.obsm["X_pca"]``
        and a neighbor graph in ``adata.obsp``.
    """
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError(
            "scanpy is required for preprocessing. "
            "Install with: pip install scanpy"
        ) from exc

    if copy:
        adata = adata.copy()

    logger.info(
        "Preprocessing: starting with %d cells x %d genes.",
        adata.n_obs,
        adata.n_vars,
    )

    # Store raw counts before filtering
    adata.var_names_make_unique()

    # Basic QC filtering
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    if adata.n_obs < 3:
        raise ValueError(
            "Preprocessing requires at least 3 cells after filtering to build "
            "a stable neighbor graph."
        )
    if adata.n_vars < 2:
        raise ValueError(
            "Preprocessing requires at least 2 genes after filtering."
        )

    # Mitochondrial gene QC
    mito_prefixes = ("MT-", "mt-", "Mt-")
    mito_mask = np.array([
        any(name.startswith(p) for p in mito_prefixes)
        for name in adata.var_names
    ])
    if mito_mask.any():
        adata.obs["pct_mito"] = np.array(
            _safe_toarray(adata[:, mito_mask].X).sum(axis=1)
        ).flatten() / np.array(
            _safe_toarray(adata.X).sum(axis=1)
        ).flatten() * 100

        if max_pct_mito < 100.0:
            n_before = adata.n_obs
            adata = adata[adata.obs["pct_mito"] < max_pct_mito, :].copy()
            n_removed = n_before - adata.n_obs
            if n_removed > 0:
                logger.info(
                    "Removed %d cells with > %.1f%% mitochondrial reads.",
                    n_removed,
                    max_pct_mito,
                )
            if adata.n_obs < 3:
                raise ValueError(
                    "Preprocessing requires at least 3 cells after mitochondrial filtering."
                )

    # Normalize and log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Store the full normalized data before HVG subsetting
    adata.raw = adata.copy()

    # Highly variable genes
    n_top_genes = min(n_top_genes, adata.n_vars)
    sc.pp.highly_variable_genes(
        adata, n_top_genes=n_top_genes, flavor="seurat_v3" if _has_raw_counts(adata) else "seurat"
    )
    adata = adata[:, adata.var["highly_variable"]].copy()
    if adata.n_vars < 2:
        raise ValueError(
            "Highly variable gene selection retained fewer than 2 genes; "
            "increase input size or relax filtering thresholds."
        )

    # Scale
    sc.pp.scale(adata, max_value=10)

    # PCA
    n_pcs = min(n_pcs, adata.n_vars - 1, adata.n_obs - 1)
    if n_pcs < 1:
        raise ValueError(
            "PCA requires at least 2 cells and 2 genes after preprocessing."
        )
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack")

    # Neighbors
    n_neighbors = min(max(2, n_neighbors), adata.n_obs - 1)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

    logger.info(
        "Preprocessing complete: %d cells x %d HVGs, %d PCs.",
        adata.n_obs,
        adata.n_vars,
        n_pcs,
    )
    return adata


def _has_raw_counts(adata: ad.AnnData) -> bool:
    """Check whether adata.X contains raw integer counts.

    Args:
        adata: AnnData to check.

    Returns:
        True if the data appears to be raw counts.
    """
    X_sample = _safe_toarray(adata.X[:100, :100] if adata.n_obs > 100 else adata.X)
    return bool(np.allclose(X_sample, X_sample.astype(int)))


def _safe_toarray(X: Any) -> np.ndarray:
    """Convert sparse or dense matrix to dense numpy array.

    Args:
        X: Input matrix (sparse or dense).

    Returns:
        Dense numpy ndarray.
    """
    if sparse.issparse(X):
        return np.asarray(X.toarray())
    return np.asarray(X)


# ======================================================================
# Gene-gene correlation
# ======================================================================


def compute_gene_correlations(
    adata: ad.AnnData,
    method: str = "pearson",
    use_raw: bool = False,
) -> pd.DataFrame:
    """Compute the gene-gene correlation matrix from expression data.

    This function computes pairwise Pearson or Spearman correlations
    between genes using the expression matrix in ``adata.X`` (or
    ``adata.raw.X`` if *use_raw* is True).

    For memory efficiency, correlations are computed on a dense matrix.
    If the dataset has many genes, consider subsetting to HVGs first.

    Args:
        adata: AnnData object with expression data.
        method: Correlation method: ``"pearson"`` or ``"spearman"``.
        use_raw: If ``True``, use ``adata.raw.X`` instead of ``adata.X``.

    Returns:
        DataFrame of shape ``(n_genes, n_genes)`` with correlation
        coefficients.  Index and columns are gene names.

    Raises:
        ValueError: If an unsupported correlation method is specified.
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(
            f"Unsupported correlation method '{method}'. "
            "Choose 'pearson' or 'spearman'."
        )

    if use_raw and adata.raw is not None:
        X = _safe_toarray(adata.raw.X)
        gene_names = list(adata.raw.var_names)
    else:
        X = _safe_toarray(adata.X)
        gene_names = list(adata.var_names)

    logger.info(
        "Computing %s gene correlations for %d genes x %d cells ...",
        method,
        X.shape[1],
        X.shape[0],
    )

    if method == "spearman":
        from scipy.stats import rankdata

        X = np.apply_along_axis(rankdata, 0, X)

    # Correlation: center, normalize, dot product
    X_centered = X - X.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(X_centered, axis=0, keepdims=True)
    norms[norms < 1e-12] = 1.0
    X_normed = X_centered / norms

    # Pearson correlation equals cosine similarity between mean-centered
    # vectors; dividing again by (n-1) would underestimate correlations.
    cor_matrix = X_normed.T @ X_normed
    np.fill_diagonal(cor_matrix, 1.0)

    # Handle NaN from constant genes
    cor_matrix = np.nan_to_num(cor_matrix, nan=0.0)

    logger.info("Gene correlation matrix computed: shape %s.", cor_matrix.shape)

    return pd.DataFrame(cor_matrix, index=gene_names, columns=gene_names)


# ======================================================================
# Cell-type marker extraction
# ======================================================================


def extract_cell_type_markers(
    adata: ad.AnnData,
    groupby: str = "louvain",
    n_genes: int = 50,
    method: str = "wilcoxon",
) -> dict[str, list[str]]:
    """Extract marker genes per cell type using differential expression.

    Runs ``scanpy.tl.rank_genes_groups`` to identify differentially
    expressed genes for each group defined by *groupby*, then returns
    the top *n_genes* per group.

    Args:
        adata: AnnData with cell-type annotations in ``adata.obs[groupby]``.
        groupby: Column in ``adata.obs`` for cell-type grouping.
        n_genes: Number of top marker genes to extract per group.
        method: DE test method (``"wilcoxon"``, ``"t-test"``,
            ``"t-test_overestim_var"``, or ``"logreg"``).

    Returns:
        Dictionary mapping group names (cell types) to lists of marker
        gene symbols.

    Raises:
        KeyError: If *groupby* is not a column in ``adata.obs``.
        ImportError: If scanpy is not installed.
    """
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError(
            "scanpy is required for marker extraction. "
            "Install with: pip install scanpy"
        ) from exc

    if groupby not in adata.obs.columns:
        raise KeyError(
            f"Column '{groupby}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    logger.info(
        "Extracting markers for '%s' groups using %s test ...",
        groupby,
        method,
    )

    # Work on a copy to avoid modifying the original
    adata_work = adata.copy()

    # Ensure the expression matrix is suitable for DE
    # If already preprocessed (scaled), use raw if available
    if adata_work.raw is not None:
        adata_de = adata_work.raw.to_adata()
        adata_de.obs = adata_work.obs
    else:
        adata_de = adata_work

    # Guard against object-typed matrices from some external datasets.
    x_matrix = adata_de.X
    if sparse.issparse(x_matrix):
        if np.issubdtype(x_matrix.dtype, np.number):
            adata_de.X = x_matrix.astype(np.float32)
        else:
            dense = np.asarray(x_matrix.toarray())
            dense = pd.DataFrame(dense).apply(
                pd.to_numeric, errors="coerce"
            ).fillna(0.0).to_numpy(dtype=np.float32)
            adata_de.X = dense
    else:
        dense = np.asarray(x_matrix)
        if dense.dtype.kind not in {"f", "i", "u", "b"}:
            dense = pd.DataFrame(dense).apply(
                pd.to_numeric, errors="coerce"
            ).fillna(0.0).to_numpy(dtype=np.float32)
        else:
            dense = dense.astype(np.float32, copy=False)
        adata_de.X = dense

    sc.tl.rank_genes_groups(
        adata_de,
        groupby=groupby,
        method=method,
        n_genes=n_genes,
    )

    markers: dict[str, list[str]] = {}
    groups = adata_de.obs[groupby].unique().tolist()

    for group in groups:
        group_str = str(group)
        gene_names_result = sc.get.rank_genes_groups_df(
            adata_de, group=group_str
        )
        top_genes = gene_names_result["names"].head(n_genes).tolist()
        markers[group_str] = top_genes

    total_markers = sum(len(v) for v in markers.values())
    logger.info(
        "Extracted markers for %d groups (%d total marker genes).",
        len(markers),
        total_markers,
    )
    return markers


# ======================================================================
# Pseudobulk aggregation
# ======================================================================


def compute_pseudobulk(
    adata: ad.AnnData,
    groupby: str = "louvain",
    layer: str | None = None,
    use_raw: bool = True,
) -> ad.AnnData:
    """Aggregate single-cell data into pseudobulk profiles per group.

    For each group in ``adata.obs[groupby]``, sums the expression
    across all cells in that group to produce a pseudobulk expression
    profile.  This is useful for bulk-style analyses on scRNA-seq data.

    Args:
        adata: AnnData with cell-group annotations.
        groupby: Column in ``adata.obs`` for grouping.
        layer: AnnData layer to aggregate.  If ``None``, uses ``adata.X``
            or ``adata.raw.X`` depending on *use_raw*.
        use_raw: If ``True`` and ``adata.raw`` is not None, aggregate
            from raw counts.

    Returns:
        AnnData of shape ``(n_groups, n_genes)`` with summed expression.

    Raises:
        KeyError: If *groupby* is not in ``adata.obs``.
    """
    if groupby not in adata.obs.columns:
        raise KeyError(
            f"Column '{groupby}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    if layer is not None:
        X = _safe_toarray(adata.layers[layer])
        gene_names = list(adata.var_names)
    elif use_raw and adata.raw is not None:
        X = _safe_toarray(adata.raw.X)
        gene_names = list(adata.raw.var_names)
    else:
        X = _safe_toarray(adata.X)
        gene_names = list(adata.var_names)

    groups = adata.obs[groupby].values
    unique_groups = sorted(set(str(g) for g in groups))

    logger.info(
        "Computing pseudobulk for %d groups (%s) ...",
        len(unique_groups),
        groupby,
    )

    pseudobulk_matrix = np.zeros(
        (len(unique_groups), X.shape[1]), dtype=np.float64
    )
    n_cells_per_group: list[int] = []

    for i, group in enumerate(unique_groups):
        mask = np.array([str(g) == group for g in groups])
        pseudobulk_matrix[i, :] = X[mask, :].sum(axis=0)
        n_cells_per_group.append(int(mask.sum()))

    obs_df = pd.DataFrame(
        {
            groupby: unique_groups,
            "n_cells": n_cells_per_group,
        },
        index=unique_groups,
    )
    var_df = pd.DataFrame(index=gene_names)

    pb_adata = ad.AnnData(
        X=pseudobulk_matrix,
        obs=obs_df,
        var=var_df,
    )

    logger.info(
        "Pseudobulk computed: %d groups x %d genes (cells per group: %s).",
        pb_adata.n_obs,
        pb_adata.n_vars,
        n_cells_per_group,
    )
    return pb_adata


# ======================================================================
# Expression-based gene embeddings
# ======================================================================


def build_gene_embeddings_from_expression(
    adata: ad.AnnData,
    n_components: int = 50,
    use_raw: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Build gene embeddings by applying PCA to the transposed expression matrix.

    Each gene is represented as a vector in cell space.  PCA is applied
    to this (n_genes x n_cells) matrix to obtain compact gene embeddings.
    This provides a simple expression-based baseline for gene program
    discovery when foundation-model embeddings are not available.

    Args:
        adata: AnnData with expression data.
        n_components: Number of PCA components for the gene embeddings.
        use_raw: If ``True``, use ``adata.raw.X``.

    Returns:
        Tuple of ``(embeddings, gene_names)`` where embeddings is a
        numpy array of shape ``(n_genes, n_components)`` and gene_names
        is the corresponding list of gene symbols.
    """
    from sklearn.decomposition import PCA

    if use_raw and adata.raw is not None:
        X = _safe_toarray(adata.raw.X)
        gene_names = list(adata.raw.var_names)
    else:
        X = _safe_toarray(adata.X)
        gene_names = list(adata.var_names)

    n_genes = X.shape[1]
    n_components = min(n_components, n_genes - 1, X.shape[0] - 1)

    logger.info(
        "Building gene embeddings via PCA on transposed expression: "
        "%d genes, %d components ...",
        n_genes,
        n_components,
    )

    # Transpose: genes become samples, cells become features
    # Use float64 for numerical stability in PCA
    gene_profiles = X.T.astype(np.float64)  # (n_genes, n_cells)

    # Handle NaN/Inf from constant genes or extreme values
    gene_profiles = np.nan_to_num(
        gene_profiles, nan=0.0, posinf=0.0, neginf=0.0
    )

    # PCA (use arpack solver for numerical stability with large matrices)
    pca = PCA(n_components=n_components, random_state=42, svd_solver="arpack")
    embeddings = pca.fit_transform(gene_profiles)

    explained_var = pca.explained_variance_ratio_.sum()
    logger.info(
        "Gene embeddings built: shape %s, explained variance: %.2f%%.",
        embeddings.shape,
        explained_var * 100,
    )

    return embeddings.astype(np.float32), gene_names


def build_graph_regularized_embeddings(
    adata: ad.AnnData,
    n_components: int = 50,
    k_neighbors: int = 15,
    n_diffusion_steps: int = 3,
    alpha: float = 0.5,
    use_raw: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Build gene embeddings with kNN-graph diffusion regularization.

    Constructs PCA-based gene embeddings, then smooths them by propagating
    information along a gene-gene coexpression kNN graph. This captures
    higher-order co-expression relationships that raw PCA misses, similar
    to the graph-based regularization used in Spectra (Dann et al., 2023).

    The diffusion process iteratively replaces each gene's embedding with
    a weighted average of its own embedding and its kNN neighbors':

        ``emb_new = alpha * emb + (1 - alpha) * neighbor_avg``

    This encourages co-expressed genes to have similar embeddings, improving
    downstream clustering into biologically coherent gene programs.

    Args:
        adata: AnnData with expression data.
        n_components: Number of PCA components.
        k_neighbors: Number of nearest neighbors for the gene coexpression
            kNN graph.
        n_diffusion_steps: Number of diffusion iterations. More steps
            produce smoother embeddings. Typical range: 1-5.
        alpha: Self-retention weight in ``[0, 1]``. Higher values preserve
            more of the original embedding signal.
        use_raw: If ``True``, use ``adata.raw.X``.

    Returns:
        Tuple of ``(embeddings, gene_names)`` where embeddings is a numpy
        array of shape ``(n_genes, n_components)``.
    """
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    if use_raw and adata.raw is not None:
        X = _safe_toarray(adata.raw.X)
        gene_names = list(adata.raw.var_names)
    else:
        X = _safe_toarray(adata.X)
        gene_names = list(adata.var_names)

    n_genes = X.shape[1]
    n_components = min(n_components, n_genes - 1, X.shape[0] - 1)
    k_neighbors = min(k_neighbors, n_genes - 1)

    logger.info(
        "Building graph-regularized gene embeddings: %d genes, "
        "%d components, k=%d, %d diffusion steps ...",
        n_genes, n_components, k_neighbors, n_diffusion_steps,
    )

    # Step 1: PCA embeddings (same as build_gene_embeddings_from_expression)
    gene_profiles = X.T.astype(np.float64)
    gene_profiles = np.nan_to_num(gene_profiles, nan=0.0, posinf=0.0, neginf=0.0)
    pca = PCA(n_components=n_components, random_state=42, svd_solver="arpack")
    embeddings = pca.fit_transform(gene_profiles)

    # Step 2: Build gene-gene coexpression kNN graph
    # Compute correlations from expression (genes as rows, cells as columns)
    gene_centered = gene_profiles - gene_profiles.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(gene_centered, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    gene_normed = gene_centered / norms

    # Use cosine distance on correlation profiles
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="cosine")
    nn.fit(gene_normed)
    distances, indices = nn.kneighbors(gene_normed)

    # Build sparse transition matrix (row-normalized)
    # Each gene's neighbors get equal weight (excluding self)
    transition = np.zeros((n_genes, n_genes), dtype=np.float64)
    for i in range(n_genes):
        neighbor_idx = indices[i, 1:]  # skip self at position 0
        neighbor_sim = 1.0 - distances[i, 1:]
        neighbor_sim = np.clip(neighbor_sim, 0.0, None)
        total = neighbor_sim.sum()
        if total > 0:
            transition[i, neighbor_idx] = neighbor_sim / total
        else:
            transition[i, neighbor_idx] = 1.0 / k_neighbors

    # Step 3: Diffusion -- iteratively smooth embeddings
    for step in range(n_diffusion_steps):
        neighbor_emb = transition @ embeddings
        embeddings = alpha * embeddings + (1.0 - alpha) * neighbor_emb

    explained_var = pca.explained_variance_ratio_.sum()
    logger.info(
        "Graph-regularized gene embeddings built: shape %s, "
        "PCA explained variance: %.2f%%.",
        embeddings.shape,
        explained_var * 100,
    )

    return embeddings.astype(np.float32), gene_names
