"""Tests for the real-data benchmark infrastructure.

These tests exercise the dataset loaders, preprocessing pipeline,
pseudobulk computation, and benchmark classes using mock data so
that they run without network access.  Tests that require actual
downloads are guarded by ``pytest.mark.skipif`` checks.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from npathway.data.datasets import (
    _read_gmt_file,
    convert_ensembl_to_symbol,
    filter_gene_sets_to_adata,
)
from npathway.data.preprocessing import (
    _safe_toarray,
    build_gene_embeddings_from_expression,
    build_graph_regularized_embeddings,
    compute_gene_correlations,
    compute_pseudobulk,
    extract_cell_type_markers,
    preprocess_adata,
)
from npathway.evaluation.benchmark_real import (
    CellTypeMarkerBenchmark,
    RealDataBenchmark,
    _pathway_recovery,
)

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def mock_adata() -> ad.AnnData:
    """Create a small AnnData for testing (200 cells x 100 genes)."""
    rng = np.random.default_rng(42)
    n_cells = 200
    n_genes = 100

    # Simulate counts with structure
    X = rng.poisson(5, size=(n_cells, n_genes)).astype(np.float32)

    # Make some genes mitochondrial
    gene_names = [f"GENE_{i}" for i in range(n_genes - 3)] + [
        "MT-CO1",
        "MT-CO2",
        "MT-ND1",
    ]

    # Assign cell types with some structure
    cell_types = np.array(
        ["T_cell"] * 70 + ["B_cell"] * 60 + ["Monocyte"] * 40 + ["NK"] * 30
    )
    rng.shuffle(cell_types)

    obs = pd.DataFrame(
        {"louvain": cell_types, "cell_type": cell_types},
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=gene_names)

    adata = ad.AnnData(X=X, obs=obs, var=var)
    return adata


@pytest.fixture
def mock_gene_programs() -> dict[str, list[str]]:
    """Gene programs using GENE_* names matching mock_adata."""
    return {
        "program_A": [f"GENE_{i}" for i in range(0, 15)],
        "program_B": [f"GENE_{i}" for i in range(10, 30)],
        "program_C": [f"GENE_{i}" for i in range(25, 45)],
        "program_D": [f"GENE_{i}" for i in range(40, 60)],
        "program_E": [f"GENE_{i}" for i in range(55, 75)],
    }


@pytest.fixture
def mock_reference_gene_sets() -> dict[str, list[str]]:
    """Reference gene sets overlapping with mock gene programs."""
    return {
        "HALLMARK_ALPHA": [f"GENE_{i}" for i in range(0, 20)],
        "HALLMARK_BETA": [f"GENE_{i}" for i in range(15, 35)],
        "HALLMARK_GAMMA": [f"GENE_{i}" for i in range(30, 50)],
        "HALLMARK_DELTA": [f"GENE_{i}" for i in range(45, 65)],
        "HALLMARK_EPSILON": [f"GENE_{i}" for i in range(60, 80)],
    }


@pytest.fixture
def mock_gmt_file(tmp_path: Path) -> Path:
    """Create a temporary GMT file for testing."""
    gmt_path = tmp_path / "test.gmt"
    lines = [
        "SET_A\tdescription\tGENE_0\tGENE_1\tGENE_2\tGENE_3\n",
        "SET_B\tdescription\tGENE_4\tGENE_5\tGENE_6\n",
        "SET_C\tdescription\tGENE_7\tGENE_8\tGENE_9\tGENE_10\tGENE_11\n",
    ]
    gmt_path.write_text("".join(lines))
    return gmt_path


# ======================================================================
# Tests: GMT file parsing
# ======================================================================


def test_read_gmt_file(mock_gmt_file: Path) -> None:
    """Parsing a GMT file should return correct gene sets."""
    gene_sets = _read_gmt_file(mock_gmt_file)

    assert len(gene_sets) == 3
    assert "SET_A" in gene_sets
    assert gene_sets["SET_A"] == ["GENE_0", "GENE_1", "GENE_2", "GENE_3"]
    assert gene_sets["SET_B"] == ["GENE_4", "GENE_5", "GENE_6"]
    assert len(gene_sets["SET_C"]) == 5


def test_read_gmt_file_not_found() -> None:
    """Reading a non-existent GMT file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        _read_gmt_file(Path("/nonexistent/file.gmt"))


# ======================================================================
# Tests: Gene set filtering
# ======================================================================


def test_filter_gene_sets_to_adata(mock_adata: ad.AnnData) -> None:
    """Filtering gene sets should remove sets with too few overlapping genes."""
    gene_sets = {
        "good_set": [f"GENE_{i}" for i in range(10)],  # all in adata
        "bad_set": ["NOT_IN_DATA_1", "NOT_IN_DATA_2"],  # none in adata
        "borderline": [f"GENE_{i}" for i in range(3)] + ["MISSING"],  # 3 in adata
    }

    filtered = filter_gene_sets_to_adata(gene_sets, mock_adata, min_genes=3)

    assert "good_set" in filtered
    assert "bad_set" not in filtered
    assert "borderline" in filtered  # has exactly 3 genes
    assert len(filtered["good_set"]) == 10
    assert len(filtered["borderline"]) == 3


def test_filter_gene_sets_strict_threshold(mock_adata: ad.AnnData) -> None:
    """Strict min_genes threshold should filter more gene sets."""
    gene_sets = {
        "small_set": [f"GENE_{i}" for i in range(3)],
        "large_set": [f"GENE_{i}" for i in range(20)],
    }

    filtered = filter_gene_sets_to_adata(gene_sets, mock_adata, min_genes=10)

    assert "small_set" not in filtered
    assert "large_set" in filtered


# ======================================================================
# Tests: Ensembl conversion
# ======================================================================


def test_convert_ensembl_to_symbol_noop(mock_adata: ad.AnnData) -> None:
    """Non-Ensembl names should pass through without modification."""
    result = convert_ensembl_to_symbol(mock_adata)
    # Should return same object (or copy with same var_names)
    assert list(result.var_names) == list(mock_adata.var_names)


def test_convert_ensembl_to_symbol_with_ensembl() -> None:
    """Ensembl IDs should be converted to symbols using var column."""
    rng = np.random.default_rng(42)
    n_cells, n_genes = 10, 5
    X = rng.random((n_cells, n_genes)).astype(np.float32)

    ensembl_ids = [f"ENSG000000{i:02d}" for i in range(n_genes)]
    symbols = [f"SYM_{i}" for i in range(n_genes)]

    var = pd.DataFrame(
        {"gene_symbols": symbols},
        index=ensembl_ids,
    )
    adata = ad.AnnData(X=X, var=var)

    result = convert_ensembl_to_symbol(adata)
    assert all(name.startswith("SYM_") for name in result.var_names)


# ======================================================================
# Tests: Preprocessing pipeline
# ======================================================================


def test_preprocess_adata(mock_adata: ad.AnnData) -> None:
    """Preprocessing should produce PCA and neighbors."""
    processed = preprocess_adata(
        mock_adata,
        min_genes=5,
        min_cells=2,
        n_top_genes=50,
        n_pcs=10,
        n_neighbors=10,
        copy=True,
    )

    assert "X_pca" in processed.obsm
    assert processed.obsm["X_pca"].shape[1] <= 10
    assert processed.n_obs > 0
    assert processed.n_vars > 0
    assert processed.n_vars <= 50  # HVG subset


def test_preprocess_adata_preserves_original(mock_adata: ad.AnnData) -> None:
    """Preprocessing with copy=True should not modify the original."""
    original_shape = mock_adata.shape
    _ = preprocess_adata(mock_adata, min_genes=5, min_cells=2, n_top_genes=50, copy=True)
    assert mock_adata.shape == original_shape


def test_preprocess_adata_rejects_tiny_filtered_dataset() -> None:
    """Tiny datasets should raise a clear validation error before Scanpy internals fail."""
    tiny = ad.AnnData(
        X=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    )

    with pytest.raises(ValueError, match="at least 3 cells"):
        preprocess_adata(tiny, min_genes=1, min_cells=1, n_top_genes=2, n_pcs=5, n_neighbors=1)


# ======================================================================
# Tests: Gene correlations
# ======================================================================


def test_compute_gene_correlations(mock_adata: ad.AnnData) -> None:
    """Gene correlation matrix should be square with gene-name index."""
    # Use a small subset to speed up
    small_adata = mock_adata[:, :20].copy()
    corr_df = compute_gene_correlations(small_adata, method="pearson")

    assert isinstance(corr_df, pd.DataFrame)
    assert corr_df.shape == (20, 20)
    # Diagonal should be 1.0
    np.testing.assert_allclose(np.diag(corr_df.values), 1.0, atol=1e-6)
    # Should be symmetric
    np.testing.assert_allclose(corr_df.values, corr_df.values.T, atol=1e-6)


def test_compute_gene_correlations_spearman(mock_adata: ad.AnnData) -> None:
    """Spearman correlations should also work."""
    small_adata = mock_adata[:, :15].copy()
    corr_df = compute_gene_correlations(small_adata, method="spearman")

    assert isinstance(corr_df, pd.DataFrame)
    assert corr_df.shape == (15, 15)


def test_compute_gene_correlations_perfect_linear_relation() -> None:
    """Perfectly linear genes should have Pearson correlation of 1."""
    X = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
            [4.0, 8.0, 12.0],
        ],
        dtype=np.float64,
    )
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(X.shape[0])]),
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    )
    corr_df = compute_gene_correlations(adata, method="pearson")
    np.testing.assert_allclose(corr_df.values, np.ones((3, 3)), atol=1e-8)


def test_compute_gene_correlations_invalid_method(mock_adata: ad.AnnData) -> None:
    """Invalid correlation method should raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported correlation method"):
        compute_gene_correlations(mock_adata[:, :10], method="invalid")


# ======================================================================
# Tests: Cell-type marker extraction
# ======================================================================


def test_extract_cell_type_markers(mock_adata: ad.AnnData) -> None:
    """Marker extraction should return genes per cell type."""
    markers = extract_cell_type_markers(
        mock_adata, groupby="louvain", n_genes=10, method="wilcoxon"
    )

    assert isinstance(markers, dict)
    assert len(markers) > 0
    # Each cell type should have markers
    for ct, genes in markers.items():
        assert len(genes) > 0
        assert len(genes) <= 10


def test_extract_markers_missing_column(mock_adata: ad.AnnData) -> None:
    """Extracting markers with a missing column should raise KeyError."""
    with pytest.raises(KeyError, match="not found"):
        extract_cell_type_markers(mock_adata, groupby="nonexistent")


# ======================================================================
# Tests: Pseudobulk computation
# ======================================================================


def test_compute_pseudobulk(mock_adata: ad.AnnData) -> None:
    """Pseudobulk should aggregate cells by group."""
    pb = compute_pseudobulk(mock_adata, groupby="louvain", use_raw=False)

    assert isinstance(pb, ad.AnnData)
    # Should have one row per unique cell type
    n_types = len(mock_adata.obs["louvain"].unique())
    assert pb.n_obs == n_types
    assert pb.n_vars == mock_adata.n_vars
    assert "n_cells" in pb.obs.columns

    # Sum of cells should equal total
    assert pb.obs["n_cells"].sum() == mock_adata.n_obs


def test_compute_pseudobulk_missing_column(mock_adata: ad.AnnData) -> None:
    """Pseudobulk with missing column should raise KeyError."""
    with pytest.raises(KeyError, match="not found"):
        compute_pseudobulk(mock_adata, groupby="nonexistent")


# ======================================================================
# Tests: Gene embeddings from expression
# ======================================================================


def test_build_gene_embeddings(mock_adata: ad.AnnData) -> None:
    """Expression-based gene embeddings should have correct shape."""
    embeddings, gene_names = build_gene_embeddings_from_expression(
        mock_adata, n_components=10
    )

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (mock_adata.n_vars, 10)
    assert len(gene_names) == mock_adata.n_vars
    assert embeddings.dtype == np.float32

    # Should not contain NaN
    assert not np.isnan(embeddings).any()


def test_build_gene_embeddings_auto_components(mock_adata: ad.AnnData) -> None:
    """Components should be capped at min(n_genes-1, n_cells-1)."""
    small_adata = mock_adata[:10, :5].copy()  # 10 cells, 5 genes
    embeddings, gene_names = build_gene_embeddings_from_expression(
        small_adata, n_components=100
    )

    # Should be capped at min(4, 9) = 4
    assert embeddings.shape[1] <= 4
    assert len(gene_names) == 5


def test_build_graph_regularized_embeddings(mock_adata: ad.AnnData) -> None:
    """Graph-regularized embeddings should have correct shape and differ from PCA."""
    pca_emb, pca_names = build_gene_embeddings_from_expression(
        mock_adata, n_components=10
    )
    graph_emb, graph_names = build_graph_regularized_embeddings(
        mock_adata, n_components=10, k_neighbors=5, n_diffusion_steps=2,
    )

    assert isinstance(graph_emb, np.ndarray)
    assert graph_emb.shape == pca_emb.shape
    assert graph_names == pca_names
    assert graph_emb.dtype == np.float32
    assert not np.isnan(graph_emb).any()
    # Diffusion should smooth embeddings (not identical to raw PCA)
    assert not np.allclose(pca_emb, graph_emb, atol=1e-3)


# ======================================================================
# Tests: Pathway recovery helper
# ======================================================================


def test_pathway_recovery(
    mock_gene_programs: dict[str, list[str]],
    mock_reference_gene_sets: dict[str, list[str]],
) -> None:
    """Pathway recovery should detect overlapping programs."""
    recovery = _pathway_recovery(
        mock_gene_programs,
        mock_reference_gene_sets,
        jaccard_threshold=0.1,
    )

    assert "recovery_rate" in recovery
    assert "mean_best_jaccard" in recovery
    assert "per_pathway" in recovery
    assert 0.0 <= recovery["recovery_rate"] <= 1.0
    assert 0.0 <= recovery["mean_best_jaccard"] <= 1.0
    assert len(recovery["per_pathway"]) == len(mock_reference_gene_sets)

    # With high overlap, some pathways should be recovered
    assert recovery["n_recovered"] > 0


def test_pathway_recovery_no_overlap() -> None:
    """Recovery rate should be 0 when there is no overlap."""
    discovered = {"prog_0": ["A", "B", "C"]}
    reference = {"ref_0": ["X", "Y", "Z"]}

    recovery = _pathway_recovery(discovered, reference, jaccard_threshold=0.1)
    assert recovery["recovery_rate"] == 0.0
    assert recovery["mean_best_jaccard"] == 0.0


# ======================================================================
# Tests: RealDataBenchmark (mock data)
# ======================================================================


def test_real_data_benchmark_with_mock(
    mock_adata: ad.AnnData,
    mock_gene_programs: dict[str, list[str]],
    mock_reference_gene_sets: dict[str, list[str]],
) -> None:
    """RealDataBenchmark should run with pre-loaded mock data."""
    benchmark = RealDataBenchmark(
        n_programs=5,
        top_n_genes=15,
        recovery_threshold=0.05,
        seed=42,
    )

    results = benchmark.run(
        mock_gene_programs,
        adata=mock_adata,
        reference_gene_sets=mock_reference_gene_sets,
        run_discovery=False,  # Skip discovery for speed
    )

    assert isinstance(results, dict)
    assert "per_method" in results
    assert "n_methods" in results
    assert results["n_methods"] >= 1

    # Check results DataFrame
    df = benchmark.get_results_df()
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 1
    assert "method" in df.columns
    assert "recovery_rate" in df.columns
    assert "coverage" in df.columns
    assert "redundancy" in df.columns
    assert "novelty" in df.columns


def test_real_data_benchmark_with_discovery(
    mock_adata: ad.AnnData,
    mock_reference_gene_sets: dict[str, list[str]],
) -> None:
    """RealDataBenchmark with discovery should run multiple methods."""
    # Use a very small setup for speed
    small_adata = mock_adata[:50, :30].copy()
    gene_names = list(small_adata.var_names)

    # Simple programs matching the genes
    programs = {
        "prog_0": gene_names[:10],
        "prog_1": gene_names[10:20],
    }

    # Filter reference to genes in small adata
    ref_filtered = {}
    gene_set = set(gene_names)
    for name, genes in mock_reference_gene_sets.items():
        filtered = [g for g in genes if g in gene_set]
        if len(filtered) >= 3:
            ref_filtered[name] = filtered

    benchmark = RealDataBenchmark(
        n_programs=3,
        top_n_genes=10,
        recovery_threshold=0.05,
        seed=42,
    )

    results = benchmark.run(
        programs,
        adata=small_adata,
        reference_gene_sets=ref_filtered if ref_filtered else mock_reference_gene_sets,
        run_discovery=True,
    )

    assert results["n_methods"] > 1  # At least provided + some discovery methods

    df = benchmark.get_results_df()
    assert len(df) > 1


def test_real_data_benchmark_plot(
    mock_adata: ad.AnnData,
    mock_gene_programs: dict[str, list[str]],
    mock_reference_gene_sets: dict[str, list[str]],
    tmp_path: Path,
) -> None:
    """RealDataBenchmark should generate a plot."""
    benchmark = RealDataBenchmark(n_programs=5, seed=42)
    benchmark.run(
        mock_gene_programs,
        adata=mock_adata,
        reference_gene_sets=mock_reference_gene_sets,
        run_discovery=False,
    )

    save_path = str(tmp_path / "benchmark_plot.png")
    fig = benchmark.plot_results(save_path=save_path)

    import matplotlib.pyplot as plt

    assert isinstance(fig, plt.Figure)
    assert Path(save_path).exists()
    plt.close(fig)


# ======================================================================
# Tests: CellTypeMarkerBenchmark
# ======================================================================


def test_cell_type_marker_benchmark(mock_adata: ad.AnnData) -> None:
    """CellTypeMarkerBenchmark should evaluate against markers."""
    gene_names = list(mock_adata.var_names)

    # Create programs that partially overlap with marker genes
    programs = {
        "prog_0": gene_names[:20],
        "prog_1": gene_names[20:40],
        "prog_2": gene_names[40:60],
        "prog_3": gene_names[60:80],
    }

    benchmark = CellTypeMarkerBenchmark(
        groupby="louvain",
        n_marker_genes=15,
        recovery_threshold=0.05,
    )

    results = benchmark.run(
        programs,
        adata=mock_adata,
    )

    assert isinstance(results, dict)
    assert "ari" in results
    assert "nmi" in results
    assert "mean_best_jaccard" in results
    assert "recovery_rate" in results
    assert "n_cell_types" in results
    assert results["n_cell_types"] > 0

    # Check per-celltype DataFrame
    df = benchmark.get_results_df()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == results["n_cell_types"]
    assert "cell_type" in df.columns
    assert "best_jaccard" in df.columns


def test_cell_type_marker_benchmark_with_precomputed_markers() -> None:
    """CellTypeMarkerBenchmark should work with pre-computed markers."""
    markers = {
        "T_cell": ["GENE_0", "GENE_1", "GENE_2", "GENE_3", "GENE_4"],
        "B_cell": ["GENE_5", "GENE_6", "GENE_7", "GENE_8", "GENE_9"],
        "Monocyte": ["GENE_10", "GENE_11", "GENE_12", "GENE_13", "GENE_14"],
    }

    programs = {
        "prog_0": ["GENE_0", "GENE_1", "GENE_2", "GENE_3", "GENE_4"],  # matches T_cell
        "prog_1": ["GENE_5", "GENE_6", "GENE_20", "GENE_21", "GENE_22"],  # partial B_cell
        "prog_2": ["GENE_30", "GENE_31", "GENE_32", "GENE_33", "GENE_34"],  # no match
    }

    benchmark = CellTypeMarkerBenchmark(recovery_threshold=0.1)
    results = benchmark.run(programs, marker_programs=markers)

    assert results["n_cell_types"] == 3
    # T_cell should be perfectly recovered
    df = benchmark.get_results_df()
    t_cell_row = df[df["cell_type"] == "T_cell"]
    assert t_cell_row["best_jaccard"].values[0] == 1.0
    assert bool(t_cell_row["recovered"].values[0]) is True


def test_cell_type_marker_benchmark_requires_input() -> None:
    """CellTypeMarkerBenchmark should raise without adata or markers."""
    benchmark = CellTypeMarkerBenchmark()
    with pytest.raises(ValueError, match="Either 'adata' or 'marker_programs'"):
        benchmark.run({"prog_0": ["G1", "G2"]})


def test_cell_type_marker_benchmark_plot(
    mock_adata: ad.AnnData,
    tmp_path: Path,
) -> None:
    """CellTypeMarkerBenchmark should generate a plot."""
    gene_names = list(mock_adata.var_names)
    programs = {
        "prog_0": gene_names[:15],
        "prog_1": gene_names[15:30],
    }

    benchmark = CellTypeMarkerBenchmark(
        groupby="louvain",
        n_marker_genes=10,
        recovery_threshold=0.05,
    )
    benchmark.run(programs, adata=mock_adata)

    save_path = str(tmp_path / "marker_plot.png")
    fig = benchmark.plot_results(save_path=save_path)

    import matplotlib.pyplot as plt

    assert isinstance(fig, plt.Figure)
    assert Path(save_path).exists()
    plt.close(fig)


# ======================================================================
# Tests: Dataset download (skipped without network)
# ======================================================================


def _network_available() -> bool:
    """Check if network access is available."""
    import urllib.request

    try:
        urllib.request.urlopen("https://www.google.com", timeout=3)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _network_available(), reason="No network access")
def test_download_pbmc3k() -> None:
    """PBMC 3k download should return a valid AnnData."""
    from npathway.data.datasets import load_pbmc3k

    adata = load_pbmc3k(preprocessed=True)
    assert isinstance(adata, ad.AnnData)
    assert adata.n_obs > 2000
    assert adata.n_vars > 100


@pytest.mark.skipif(not _network_available(), reason="No network access")
def test_download_msigdb_hallmark() -> None:
    """MSigDB Hallmark download should return gene sets."""
    from npathway.data.datasets import load_msigdb_gene_sets

    hallmark = load_msigdb_gene_sets(collection="hallmark")
    assert isinstance(hallmark, dict)
    assert len(hallmark) >= 40  # Hallmark has 50 gene sets
    # Check a well-known set
    for name in hallmark:
        assert len(hallmark[name]) > 5


def test_load_tabula_muris_local_path(tmp_path: Path) -> None:
    """Local Tabula Muris path should load without proxy fallback."""
    from npathway.data.datasets import load_tabula_muris

    X = np.ones((6, 4), dtype=np.float32)
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(6)])
    var = pd.DataFrame(index=[f"GENE_{i}" for i in range(4)])
    adata = ad.AnnData(X=X, obs=obs, var=var)

    local_path = tmp_path / "tabula_local.h5ad"
    adata.write_h5ad(local_path)

    loaded = load_tabula_muris(
        method="droplet",
        dataset_path=local_path,
        allow_proxy=False,
    )
    assert isinstance(loaded, ad.AnnData)
    assert loaded.n_obs == 6
    assert loaded.n_vars == 4
    assert "npathway_dataset" in loaded.uns
    meta = loaded.uns["npathway_dataset"]
    assert meta["dataset_name"] == "tabula_muris"
    assert meta["is_proxy"] is False
    assert str(meta["source"]).startswith("local_h5ad:")


def test_load_tabula_muris_disallow_proxy_without_path() -> None:
    """Proxy-disallowed mode should fail if no local dataset path is provided."""
    from npathway.data.datasets import load_tabula_muris

    with pytest.raises(RuntimeError, match="proxy"):
        load_tabula_muris(method="droplet", dataset_path=None, allow_proxy=False)


# ======================================================================
# Tests: _safe_toarray helper
# ======================================================================


def test_safe_toarray_dense() -> None:
    """_safe_toarray should pass through dense arrays."""
    arr = np.array([[1, 2], [3, 4]])
    result = _safe_toarray(arr)
    np.testing.assert_array_equal(result, arr)


def test_safe_toarray_sparse() -> None:
    """_safe_toarray should convert sparse to dense."""
    from scipy import sparse

    arr = sparse.csr_matrix(np.array([[1, 0], [0, 4]]))
    result = _safe_toarray(arr)
    expected = np.array([[1, 0], [0, 4]])
    np.testing.assert_array_equal(result, expected)
