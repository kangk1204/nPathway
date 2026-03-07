"""Data loading and preprocessing subpackage.

This subpackage provides utilities for downloading, caching, and
preprocessing publicly available scRNA-seq datasets and curated
pathway gene-set collections for benchmarking gene program discovery.

Modules
-------
datasets
    Dataset download and caching (PBMC 3k, Tabula Muris, MSigDB gene sets).
preprocessing
    Standard preprocessing pipelines, correlation computation, marker
    gene extraction, pseudobulk aggregation, and expression-based
    gene embedding construction.
"""

from __future__ import annotations

from npathway.data.datasets import (
    convert_ensembl_to_symbol,
    download_msigdb_gmt,
    filter_gene_sets_to_adata,
    load_all_msigdb_collections,
    load_burczynski06,
    load_moignard15,
    load_msigdb_gene_sets,
    load_paul15,
    load_pbmc3k,
    load_pbmc68k_reduced,
    load_tabula_muris,
)
from npathway.data.preprocessing import (
    build_gene_embeddings_from_expression,
    build_graph_regularized_embeddings,
    compute_gene_correlations,
    compute_pseudobulk,
    extract_cell_type_markers,
    preprocess_adata,
)

__all__ = [
    # Dataset loaders
    "load_pbmc3k",
    "load_pbmc68k_reduced",
    "load_paul15",
    "load_moignard15",
    "load_burczynski06",
    "load_tabula_muris",
    "load_msigdb_gene_sets",
    "load_all_msigdb_collections",
    "download_msigdb_gmt",
    "filter_gene_sets_to_adata",
    "convert_ensembl_to_symbol",
    # Preprocessing
    "preprocess_adata",
    "compute_gene_correlations",
    "extract_cell_type_markers",
    "compute_pseudobulk",
    "build_gene_embeddings_from_expression",
    "build_graph_regularized_embeddings",
]
