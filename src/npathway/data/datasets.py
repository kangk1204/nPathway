"""Public dataset loaders for real-data benchmarking.

This module provides functions to download, cache, and return standard
single-cell RNA-seq datasets and curated pathway gene-set collections
used for benchmarking gene program discovery methods.

Supported scRNA-seq datasets
-----------------------------
* **PBMC 3k** -- ~2 700 peripheral blood mononuclear cells from 10x Genomics,
  pre-processed and annotated with cell types (via ``scanpy.datasets``).
* **PBMC68k reduced** -- curated PBMC subset from 10x Genomics
  (``scanpy.datasets.pbmc68k_reduced``).
* **Paul15** -- myeloid progenitor differentiation trajectories
  (``scanpy.datasets.paul15``).
* **Moignard15** -- blood development qPCR dataset
  (``scanpy.datasets.moignard15``).
* **Burczynski06** -- ulcerative colitis expression cohort
  (``scanpy.datasets.burczynski06``).
* **Tabula Muris** -- Mouse atlas data from a local ``.h5ad`` file or
  scanpy proxy fallback.

Supported gene-set collections
-------------------------------
* **MSigDB Hallmark** -- 50 well-defined biological-state gene sets.
* **MSigDB C2:KEGG** -- KEGG pathway gene sets.
* **MSigDB C5:GO_BP** -- Gene Ontology Biological Process gene sets.

All downloaded files are cached under ``~/.npathway/data/`` to avoid
repeated network transfers.
"""

from __future__ import annotations

import gzip
import logging
import shutil
import urllib.request
from pathlib import Path

import anndata as ad
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache directory
# ---------------------------------------------------------------------------

_CACHE_DIR = Path.home() / ".npathway" / "data"

# MSigDB release base URLs (v2024.1)
_MSIGDB_BASE_URLS: dict[str, str] = {
    "human": "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Hs/",
    "mouse": "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Mm/",
}

# MSigDB collection file names (symbols version) by species
_MSIGDB_FILES: dict[str, dict[str, str]] = {
    "human": {
        "hallmark": "h.all.v2024.1.Hs.symbols.gmt",
        "kegg": "c2.cp.kegg_legacy.v2024.1.Hs.symbols.gmt",
        "go_bp": "c5.go.bp.v2024.1.Hs.symbols.gmt",
    },
    "mouse": {
        "hallmark": "mh.all.v2024.1.Mm.symbols.gmt",
        "kegg": "m2.cp.v2024.1.Mm.symbols.gmt",
        "go_bp": "m5.go.bp.v2024.1.Mm.symbols.gmt",
    },
}


def _annotate_dataset_provenance(
    adata: ad.AnnData,
    *,
    dataset_name: str,
    source: str,
    is_proxy: bool,
) -> ad.AnnData:
    """Attach standardized dataset provenance metadata to ``adata.uns``."""
    adata.uns["npathway_dataset"] = {
        "dataset_name": dataset_name,
        "source": source,
        "is_proxy": bool(is_proxy),
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
    }
    return adata


def _ensure_cache_dir() -> Path:
    """Create the cache directory if it does not exist.

    Returns:
        Path to the cache directory.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def _download_file(url: str, dest: Path, description: str = "") -> Path:
    """Download a file from *url* to *dest* with progress logging.

    If the destination already exists the download is skipped.

    Args:
        url: Remote URL to fetch.
        dest: Local destination path.
        description: Human-readable label for log messages.

    Returns:
        Path to the downloaded file.

    Raises:
        RuntimeError: If the download fails after exhausting retries.
    """
    if dest.exists():
        logger.info("Using cached %s at %s", description, dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s from %s ...", description, url)

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "nPathway/0.1"})
            with urllib.request.urlopen(req, timeout=120) as response:
                with open(dest, "wb") as fh:
                    shutil.copyfileobj(response, fh)
            logger.info("Downloaded %s (%s bytes)", description, dest.stat().st_size)
            return dest
        except Exception as exc:
            logger.warning(
                "Download attempt %d/%d for %s failed: %s",
                attempt,
                max_retries,
                description,
                exc,
            )
            if dest.exists():
                dest.unlink()
            if attempt == max_retries:
                raise RuntimeError(
                    f"Failed to download {description} from {url} "
                    f"after {max_retries} attempts: {exc}"
                ) from exc
    return dest  # unreachable, satisfies type checker


# ======================================================================
# scRNA-seq dataset loaders
# ======================================================================


def load_pbmc3k(preprocessed: bool = True) -> ad.AnnData:
    """Download and return the PBMC 3k dataset from 10x Genomics.

    Uses ``scanpy.datasets.pbmc3k_processed()`` which returns an AnnData
    with ~2 700 cells, gene expression, PCA, UMAP, and cell-type
    annotations in ``adata.obs["louvain"]``.

    When *preprocessed* is ``False``, the raw (unprocessed) version is
    returned via ``scanpy.datasets.pbmc3k()``.

    Args:
        preprocessed: If ``True`` (default), return the fully processed
            version with embeddings and annotations.  If ``False``,
            return the raw counts.

    Returns:
        AnnData object with PBMC 3k data.

    Raises:
        ImportError: If scanpy is not installed.
        RuntimeError: If the download fails.
    """
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError(
            "scanpy is required for dataset loading. "
            "Install with: pip install scanpy"
        ) from exc

    logger.info("Loading PBMC 3k dataset (preprocessed=%s) ...", preprocessed)

    if preprocessed:
        adata = sc.datasets.pbmc3k_processed()
    else:
        adata = sc.datasets.pbmc3k()

    logger.info(
        "PBMC 3k loaded: %d cells, %d genes.",
        adata.n_obs,
        adata.n_vars,
    )
    source = (
        "scanpy.datasets.pbmc3k_processed"
        if preprocessed
        else "scanpy.datasets.pbmc3k"
    )
    return _annotate_dataset_provenance(
        adata,
        dataset_name="pbmc3k",
        source=source,
        is_proxy=False,
    )


def _load_scanpy_dataset(
    *,
    dataset_fn_name: str,
    dataset_name: str,
) -> ad.AnnData:
    """Load a scanpy bundled dataset and attach provenance metadata."""
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError(
            "scanpy is required for dataset loading. "
            "Install with: pip install scanpy"
        ) from exc

    fn = getattr(sc.datasets, dataset_fn_name, None)
    if fn is None:
        raise RuntimeError(
            f"scanpy.datasets has no loader '{dataset_fn_name}'. "
            "Please update scanpy."
        )

    logger.info(
        "Loading %s via scanpy.datasets.%s ...",
        dataset_name,
        dataset_fn_name,
    )
    adata = fn()
    if adata.var_names.has_duplicates:
        adata.var_names_make_unique()
    logger.info(
        "%s loaded: %d cells, %d genes.",
        dataset_name,
        adata.n_obs,
        adata.n_vars,
    )
    return _annotate_dataset_provenance(
        adata,
        dataset_name=dataset_name,
        source=f"scanpy.datasets.{dataset_fn_name}",
        is_proxy=False,
    )


def load_pbmc68k_reduced() -> ad.AnnData:
    """Load scanpy PBMC68k reduced dataset."""
    return _load_scanpy_dataset(
        dataset_fn_name="pbmc68k_reduced",
        dataset_name="pbmc68k_reduced",
    )


def load_paul15() -> ad.AnnData:
    """Load scanpy Paul15 differentiation dataset."""
    return _load_scanpy_dataset(
        dataset_fn_name="paul15",
        dataset_name="paul15",
    )


def load_moignard15() -> ad.AnnData:
    """Load scanpy Moignard15 blood development dataset.

    Notes
    -----
    This loader requires ``openpyxl`` because the upstream source is an
    Excel workbook.
    """
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "moignard15 dataset requires optional dependency `openpyxl`."
        ) from exc
    return _load_scanpy_dataset(
        dataset_fn_name="moignard15",
        dataset_name="moignard15",
    )


def _cap_genes_by_variance(
    adata: ad.AnnData,
    *,
    max_genes: int,
    dataset_name: str,
) -> ad.AnnData:
    """Keep top-variance genes to bound runtime/memory on very wide matrices."""
    if adata.n_vars <= max_genes:
        return adata

    x = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    x = np.asarray(x, dtype=np.float32)
    variances = np.nan_to_num(np.var(x, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    top_idx = np.argsort(variances)[-max_genes:]
    adata_capped = adata[:, np.sort(top_idx)].copy()

    logger.info(
        "%s gene cap applied: retained %d / %d genes by variance.",
        dataset_name,
        adata_capped.n_vars,
        adata.n_vars,
    )
    meta = adata_capped.uns.get("npathway_dataset", {})
    meta["original_n_vars"] = int(adata.n_vars)
    meta["n_vars"] = int(adata_capped.n_vars)
    meta["gene_cap"] = int(max_genes)
    adata_capped.uns["npathway_dataset"] = meta
    return adata_capped


def load_burczynski06(max_genes: int = 5000) -> ad.AnnData:
    """Load scanpy Burczynski06 cohort dataset.

    Notes
    -----
    This dataset is extremely wide (22k+ genes). To keep expanded benchmark
    runtime/memory stable across machines, the loader keeps the top-variance
    ``max_genes`` features by default.
    """
    adata = _load_scanpy_dataset(
        dataset_fn_name="burczynski06",
        dataset_name="burczynski06",
    )
    return _cap_genes_by_variance(
        adata,
        max_genes=max_genes,
        dataset_name="burczynski06",
    )


def load_tabula_muris(
    method: str = "droplet",
    dataset_path: str | Path | None = None,
    allow_proxy: bool = True,
) -> ad.AnnData:
    """Download and return a Tabula Muris dataset.

    The Tabula Muris dataset contains single-cell transcriptomes from
    20 mouse organs profiled with both FACS-sorted plate-based (Smart-seq2)
    and microfluidic droplet-based (10x Chromium) protocols.

    This function supports two modes:
    1) load a user-provided local Tabula Muris ``.h5ad`` file via
       ``dataset_path`` (recommended for publication-grade benchmarks), or
    2) use ``scanpy.datasets.pbmc68k_reduced`` as a lightweight proxy
       when ``allow_proxy=True``.

    Args:
        method: Protocol variant. Currently only ``"droplet"`` is supported.
        dataset_path: Optional local path to a real Tabula Muris ``.h5ad`` file.
            If provided, proxy loading is skipped and this file is loaded.
        allow_proxy: Whether fallback to the scanpy proxy dataset is allowed
            when ``dataset_path`` is not provided.

    Returns:
        AnnData object with Tabula Muris data.

    Raises:
        FileNotFoundError: If ``dataset_path`` is provided but does not exist.
        RuntimeError: If no local dataset is provided and proxy loading is
            disabled.
        ImportError: If scanpy is required for proxy loading but unavailable.
        ValueError: If an unsupported method is specified.
    """
    if method not in ("droplet",):
        raise ValueError(
            f"Unsupported Tabula Muris method '{method}'. "
            "Currently supported: 'droplet'."
        )

    if dataset_path is not None:
        path = Path(dataset_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Tabula Muris dataset path not found: {path}")
        logger.info("Loading Tabula Muris (%s) from local file: %s", method, path)
        adata = ad.read_h5ad(path)
        logger.info(
            "Tabula Muris local loaded: %d cells, %d genes.",
            adata.n_obs,
            adata.n_vars,
        )
        return _annotate_dataset_provenance(
            adata,
            dataset_name="tabula_muris",
            source=f"local_h5ad:{path}",
            is_proxy=False,
        )

    if not allow_proxy:
        raise RuntimeError(
            "No local Tabula Muris dataset_path provided and proxy loading is "
            "disabled (allow_proxy=False)."
        )

    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError(
            "scanpy is required for dataset loading. "
            "Install with: pip install scanpy"
        ) from exc

    logger.info("Loading Tabula Muris (%s) via scanpy ...", method)

    # scanpy provides pbmc68k_reduced as a compact multi-cell-type dataset
    # that serves as a practical proxy for benchmarking workflows
    adata = sc.datasets.pbmc68k_reduced()

    logger.info(
        "Tabula Muris proxy loaded: %d cells, %d genes.",
        adata.n_obs,
        adata.n_vars,
    )
    return _annotate_dataset_provenance(
        adata,
        dataset_name="tabula_muris",
        source="scanpy.datasets.pbmc68k_reduced",
        is_proxy=True,
    )


# ======================================================================
# MSigDB gene-set loaders
# ======================================================================


def _read_gmt_file(filepath: Path) -> dict[str, list[str]]:
    """Parse a GMT file into a gene-set dictionary.

    Handles both plain-text and gzip-compressed GMT files.

    Args:
        filepath: Path to the GMT file (optionally gzipped).

    Returns:
        Dictionary mapping gene-set names to lists of gene symbols.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"GMT file not found: {filepath}")

    opener = gzip.open if filepath.suffix == ".gz" else open
    gene_sets: dict[str, list[str]] = {}

    with opener(filepath, "rt", encoding="utf-8") as fh:  # type: ignore[call-overload]
        for line in fh:
            line = line.rstrip("\n\r")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            name = parts[0].strip()
            genes = [g.strip() for g in parts[2:] if g.strip()]
            gene_sets[name] = genes

    logger.info(
        "Parsed %d gene sets from %s (%d unique genes).",
        len(gene_sets),
        filepath.name,
        len({g for gs in gene_sets.values() for g in gs}),
    )
    return gene_sets


def download_msigdb_gmt(
    collection: str = "hallmark",
    species: str = "human",
    cache_dir: Path | None = None,
) -> Path:
    """Download an MSigDB GMT file and return its local path.

    Args:
        collection: One of ``"hallmark"``, ``"kegg"``, or ``"go_bp"``.
        species: ``"human"`` or ``"mouse"``.
        cache_dir: Override cache directory.  Defaults to
            ``~/.npathway/data/msigdb/``.

    Returns:
        Path to the downloaded GMT file.

    Raises:
        ValueError: If the collection name is not recognized.
        RuntimeError: If the download fails.
    """
    if species not in _MSIGDB_FILES:
        raise ValueError(
            f"Unknown MSigDB species '{species}'. "
            f"Choose from: {sorted(_MSIGDB_FILES.keys())}"
        )
    if collection not in _MSIGDB_FILES[species]:
        raise ValueError(
            f"Unknown MSigDB collection '{collection}'. "
            f"Choose from: {sorted(_MSIGDB_FILES[species].keys())}"
        )

    filename = _MSIGDB_FILES[species][collection]
    url = _MSIGDB_BASE_URLS[species] + filename

    if cache_dir is None:
        cache_dir = _ensure_cache_dir() / "msigdb" / species
    cache_dir.mkdir(parents=True, exist_ok=True)

    dest = cache_dir / filename
    return _download_file(url, dest, description=f"MSigDB {species}:{collection}")


def load_msigdb_gene_sets(
    collection: str = "hallmark",
    species: str = "human",
    cache_dir: Path | None = None,
) -> dict[str, list[str]]:
    """Download and parse an MSigDB gene-set collection.

    Args:
        collection: One of ``"hallmark"``, ``"kegg"``, or ``"go_bp"``.
        species: ``"human"`` or ``"mouse"``.
        cache_dir: Override cache directory.

    Returns:
        Dictionary mapping gene-set names to lists of gene symbols.

    Raises:
        ValueError: If the collection name is not recognized.
        RuntimeError: If the download fails.
    """
    gmt_path = download_msigdb_gmt(
        collection=collection,
        species=species,
        cache_dir=cache_dir,
    )
    return _read_gmt_file(gmt_path)


def load_all_msigdb_collections(
    species: str = "human",
    cache_dir: Path | None = None,
) -> dict[str, dict[str, list[str]]]:
    """Download and parse all supported MSigDB collections.

    Returns a nested dictionary keyed by collection name (``"hallmark"``,
    ``"kegg"``, ``"go_bp"``), each mapping gene-set names to gene lists.

    Args:
        cache_dir: Override cache directory.

    Returns:
        Nested dictionary ``{collection: {gene_set_name: [genes]}}``.
    """
    if species not in _MSIGDB_FILES:
        raise ValueError(
            f"Unknown MSigDB species '{species}'. "
            f"Choose from: {sorted(_MSIGDB_FILES.keys())}"
        )
    result: dict[str, dict[str, list[str]]] = {}
    for collection in _MSIGDB_FILES[species]:
        result[collection] = load_msigdb_gene_sets(
            collection=collection,
            species=species,
            cache_dir=cache_dir,
        )
    return result


# ======================================================================
# Gene name utilities
# ======================================================================


def filter_gene_sets_to_adata(
    gene_sets: dict[str, list[str]],
    adata: ad.AnnData,
    min_genes: int = 5,
) -> dict[str, list[str]]:
    """Filter gene sets to only include genes present in an AnnData object.

    This handles the common case where MSigDB gene symbols do not fully
    overlap with the genes measured in a particular dataset.  Gene sets
    with fewer than *min_genes* remaining members after filtering are
    discarded.

    Args:
        gene_sets: Dictionary mapping gene-set names to gene lists.
        adata: AnnData object whose ``.var_names`` define the measured
            gene universe.
        min_genes: Minimum number of genes a set must retain to be kept.

    Returns:
        Filtered gene-set dictionary.
    """
    measured_genes = set(adata.var_names)
    filtered: dict[str, list[str]] = {}

    for name, genes in gene_sets.items():
        kept = [g for g in genes if g in measured_genes]
        if len(kept) >= min_genes:
            filtered[name] = kept

    n_removed = len(gene_sets) - len(filtered)
    if n_removed > 0:
        logger.info(
            "Filtered gene sets: kept %d / %d (removed %d with < %d genes in dataset).",
            len(filtered),
            len(gene_sets),
            n_removed,
            min_genes,
        )
    return filtered


def convert_ensembl_to_symbol(
    adata: ad.AnnData,
    gene_column: str | None = None,
) -> ad.AnnData:
    """Convert Ensembl gene IDs to gene symbols in an AnnData object.

    If the AnnData ``var_names`` look like Ensembl IDs (starting with
    ``ENSG`` or ``ENSMUSG``), this function attempts to map them to
    gene symbols using an annotation column in ``adata.var``.

    Args:
        adata: AnnData object, potentially with Ensembl var_names.
        gene_column: Column in ``adata.var`` containing gene symbols.
            If ``None``, common column names are tried:
            ``"gene_symbols"``, ``"gene_short_name"``, ``"feature_name"``.

    Returns:
        AnnData with var_names set to gene symbols.  Genes without a
        valid symbol mapping are removed.
    """
    current_names = adata.var_names.tolist()
    looks_ensembl = any(
        n.startswith(("ENSG", "ENSMUSG")) for n in current_names[:20]
    )

    if not looks_ensembl:
        logger.info("Gene names do not look like Ensembl IDs; no conversion needed.")
        return adata

    # Try to find symbol column
    candidate_columns = ["gene_symbols", "gene_short_name", "feature_name", "gene_name"]
    if gene_column is not None:
        candidate_columns = [gene_column] + candidate_columns

    symbol_col: str | None = None
    for col in candidate_columns:
        if col in adata.var.columns:
            symbol_col = col
            break

    if symbol_col is None:
        logger.warning(
            "Could not find a gene symbol column in adata.var (tried %s). "
            "Returning original AnnData.",
            candidate_columns,
        )
        return adata

    symbols = adata.var[symbol_col].values
    valid_mask = np.array([
        isinstance(s, str) and len(s) > 0 and s != "nan"
        for s in symbols
    ])

    adata_filtered = adata[:, valid_mask].copy()
    adata_filtered.var_names = adata_filtered.var[symbol_col].values[
        : adata_filtered.n_vars
    ].tolist()
    adata_filtered.var_names_make_unique()

    logger.info(
        "Converted Ensembl IDs to symbols: %d / %d genes retained.",
        adata_filtered.n_vars,
        adata.n_vars,
    )
    return adata_filtered
