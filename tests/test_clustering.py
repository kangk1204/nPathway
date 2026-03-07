"""Tests for clustering-based gene program discovery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from npathway.discovery.clustering import ClusteringProgramDiscovery


def _make_clustered_data(
    n_clusters: int = 5,
    genes_per_cluster: int = 40,
    n_dims: int = 32,
    seed: int = 42,
) -> tuple[np.ndarray, list[str]]:
    """Generate synthetic embeddings with clear cluster structure."""
    rng = np.random.default_rng(seed)
    embeddings_list = []
    gene_names = []
    for c in range(n_clusters):
        center = rng.standard_normal(n_dims) * 5.0
        cluster_emb = center + rng.standard_normal((genes_per_cluster, n_dims)) * 0.3
        embeddings_list.append(cluster_emb)
        gene_names.extend([f"C{c}_GENE_{i}" for i in range(genes_per_cluster)])
    embeddings = np.vstack(embeddings_list).astype(np.float64)
    return embeddings, gene_names


def test_clustering_kmeans() -> None:
    """KMeans clustering should discover the correct number of programs."""
    embeddings, gene_names = _make_clustered_data(n_clusters=5, genes_per_cluster=30)
    model = ClusteringProgramDiscovery(method="kmeans", n_programs=5, random_state=42)
    model.fit(embeddings, gene_names)

    programs = model.get_programs()
    assert isinstance(programs, dict)
    assert len(programs) == 5
    total_genes = sum(len(g) for g in programs.values())
    assert total_genes == len(gene_names)


def test_clustering_spectral() -> None:
    """Spectral clustering should discover programs."""
    embeddings, gene_names = _make_clustered_data(n_clusters=4, genes_per_cluster=30)
    model = ClusteringProgramDiscovery(
        method="spectral", n_programs=4, k_neighbors=10, random_state=42
    )
    model.fit(embeddings, gene_names)

    programs = model.get_programs()
    assert isinstance(programs, dict)
    assert len(programs) == 4


def test_clustering_leiden() -> None:
    """Leiden clustering should discover programs (if igraph+leidenalg available)."""
    try:
        import igraph  # noqa: F401
        import leidenalg  # noqa: F401
    except ImportError:
        pytest.skip("igraph and/or leidenalg not installed")

    embeddings, gene_names = _make_clustered_data(n_clusters=5, genes_per_cluster=30)
    model = ClusteringProgramDiscovery(
        method="leiden", resolution=1.0, k_neighbors=10, random_state=42
    )
    model.fit(embeddings, gene_names)

    programs = model.get_programs()
    assert isinstance(programs, dict)
    assert len(programs) >= 2


def test_clustering_hdbscan() -> None:
    """HDBSCAN clustering should discover programs (if hdbscan available)."""
    try:
        import hdbscan  # noqa: F401
    except ImportError:
        pytest.skip("hdbscan not installed")

    embeddings, gene_names = _make_clustered_data(n_clusters=5, genes_per_cluster=30)
    model = ClusteringProgramDiscovery(
        method="hdbscan", min_cluster_size=5, random_state=42
    )
    model.fit(embeddings, gene_names)

    programs = model.get_programs()
    assert isinstance(programs, dict)
    assert len(programs) >= 1


def test_auto_k_selection() -> None:
    """Auto-K selection via silhouette should pick a reasonable K."""
    embeddings, gene_names = _make_clustered_data(n_clusters=4, genes_per_cluster=40)
    model = ClusteringProgramDiscovery(
        method="kmeans",
        n_programs=None,  # auto-select
        k_range=(2, 8),
        random_state=42,
    )
    model.fit(embeddings, gene_names)

    programs = model.get_programs()
    # Should find between 2 and 8 programs
    assert 2 <= len(programs) <= 8


def test_get_programs_returns_dict() -> None:
    """get_programs should return a dict of str -> list[str]."""
    embeddings, gene_names = _make_clustered_data(n_clusters=3, genes_per_cluster=20)
    model = ClusteringProgramDiscovery(method="kmeans", n_programs=3, random_state=42)
    model.fit(embeddings, gene_names)

    programs = model.get_programs()
    assert isinstance(programs, dict)
    for key, val in programs.items():
        assert isinstance(key, str)
        assert isinstance(val, list)
        for g in val:
            assert isinstance(g, str)


def test_get_program_scores_returns_weighted() -> None:
    """get_program_scores should return gene-weight tuples."""
    embeddings, gene_names = _make_clustered_data(n_clusters=3, genes_per_cluster=20)
    model = ClusteringProgramDiscovery(method="kmeans", n_programs=3, random_state=42)
    model.fit(embeddings, gene_names)

    scores = model.get_program_scores()
    assert isinstance(scores, dict)
    for prog_name, gene_score_list in scores.items():
        assert isinstance(gene_score_list, list)
        for gene, score in gene_score_list:
            assert isinstance(gene, str)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0


def test_to_gmt_export(tmp_output_dir: Path) -> None:
    """to_gmt should write a valid GMT file."""
    embeddings, gene_names = _make_clustered_data(n_clusters=3, genes_per_cluster=20)
    model = ClusteringProgramDiscovery(method="kmeans", n_programs=3, random_state=42)
    model.fit(embeddings, gene_names)

    filepath = str(tmp_output_dir / "programs.gmt")
    model.to_gmt(filepath)
    assert Path(filepath).exists()

    # Read back and verify
    from npathway.utils.gmt_io import read_gmt

    loaded = read_gmt(filepath)
    programs = model.get_programs()
    assert set(loaded.keys()) == set(programs.keys())


# ------------------------------------------------------------------
# Tests for new SOTA capabilities
# ------------------------------------------------------------------


def test_quality_metrics_include_davies_bouldin() -> None:
    """get_quality_metrics should include DB index after fitting."""
    embeddings, gene_names = _make_clustered_data(n_clusters=4, genes_per_cluster=25)
    model = ClusteringProgramDiscovery(method="kmeans", n_programs=4, random_state=42)
    model.fit(embeddings, gene_names)

    qm = model.get_quality_metrics()
    assert "silhouette_score" in qm
    assert "calinski_harabasz_score" in qm
    assert "davies_bouldin_score" in qm
    assert "n_programs" in qm
    assert "n_noise" in qm
    # DB index should be finite and positive for well-separated clusters
    assert np.isfinite(qm["davies_bouldin_score"])
    assert qm["davies_bouldin_score"] >= 0.0


def test_gene_confidence_after_fit() -> None:
    """Gene confidence scores should be computed after fit."""
    embeddings, gene_names = _make_clustered_data(n_clusters=3, genes_per_cluster=20)
    model = ClusteringProgramDiscovery(method="kmeans", n_programs=3, random_state=42)
    model.fit(embeddings, gene_names)

    confidence = model.get_gene_confidence()
    assert isinstance(confidence, dict)
    assert len(confidence) == 3  # 3 programs

    all_genes_in_confidence = set()
    for prog_name, gene_scores in confidence.items():
        assert isinstance(gene_scores, dict)
        for gene, score in gene_scores.items():
            assert isinstance(gene, str)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
            all_genes_in_confidence.add(gene)

    # Every gene should have a confidence score
    assert all_genes_in_confidence == set(gene_names)


def test_consensus_k_selection() -> None:
    """Consensus K selection should use multiple criteria."""
    embeddings, gene_names = _make_clustered_data(n_clusters=4, genes_per_cluster=30)
    model = ClusteringProgramDiscovery(
        method="kmeans",
        n_programs=None,
        k_range=(2, 8),
        random_state=42,
    )
    model.fit(embeddings, gene_names)

    programs = model.get_programs()
    # The consensus should find something in the valid range
    assert 2 <= len(programs) <= 8
    # Quality should be computed
    qm = model.get_quality_metrics()
    assert np.isfinite(qm["silhouette_score"])


def test_multi_resolution_leiden() -> None:
    """Multi-resolution Leiden should sweep and return best partition."""
    try:
        import igraph  # noqa: F401
        import leidenalg  # noqa: F401
    except ImportError:
        pytest.skip("igraph and/or leidenalg not installed")

    embeddings, gene_names = _make_clustered_data(n_clusters=5, genes_per_cluster=30)
    model = ClusteringProgramDiscovery(
        method="leiden", k_neighbors=10, random_state=42
    )
    result = model.multi_resolution_leiden(
        embeddings,
        gene_names,
        n_resolutions=10,
        resolution_range=(0.2, 2.0),
    )

    # Check return structure
    assert "best_resolution" in result
    assert "best_modularity" in result
    assert "best_coherence" in result
    assert "best_combined_score" in result
    assert "n_communities" in result
    assert "resolution_sweep" in result
    assert "labels" in result

    assert isinstance(result["best_resolution"], float)
    assert result["best_resolution"] > 0
    assert result["n_communities"] >= 2
    assert len(result["resolution_sweep"]) == 10
    assert isinstance(result["labels"], np.ndarray)
    assert len(result["labels"]) == len(gene_names)

    # Model should now be fitted
    programs = model.get_programs()
    assert isinstance(programs, dict)
    assert len(programs) >= 2


def test_multi_resolution_leiden_explicit_resolutions() -> None:
    """Multi-resolution Leiden should accept explicit resolution list."""
    try:
        import igraph  # noqa: F401
        import leidenalg  # noqa: F401
    except ImportError:
        pytest.skip("igraph and/or leidenalg not installed")

    embeddings, gene_names = _make_clustered_data(n_clusters=3, genes_per_cluster=25)
    model = ClusteringProgramDiscovery(
        method="leiden", k_neighbors=10, random_state=42
    )
    result = model.multi_resolution_leiden(
        embeddings,
        gene_names,
        resolutions=[0.5, 1.0, 1.5, 2.0],
    )

    assert len(result["resolution_sweep"]) == 4
    assert result["best_resolution"] in [0.5, 1.0, 1.5, 2.0]


def test_stability_analysis() -> None:
    """Stability analysis should return meaningful Jaccard scores."""
    embeddings, gene_names = _make_clustered_data(n_clusters=4, genes_per_cluster=30)
    model = ClusteringProgramDiscovery(
        method="kmeans", n_programs=4, random_state=42
    )
    model.fit(embeddings, gene_names)

    stability = model.compute_stability(
        embeddings, gene_names, n_iterations=5, subsample_fraction=0.8
    )

    assert "mean_jaccard" in stability
    assert "std_jaccard" in stability
    assert "per_iteration_jaccard" in stability
    assert "stability_score" in stability

    # Jaccard should be between 0 and 1
    assert 0.0 <= stability["mean_jaccard"] <= 1.0
    assert 0.0 <= stability["stability_score"] <= 1.0
    assert len(stability["per_iteration_jaccard"]) == 5

    # For well-separated data, stability should be reasonable (> 0.3)
    assert stability["mean_jaccard"] > 0.3

    # Stability should be added to quality metrics
    qm = model.get_quality_metrics()
    assert "stability_score" in qm


def test_stability_without_prior_fit() -> None:
    """Stability analysis should auto-fit if model is not yet fitted."""
    embeddings, gene_names = _make_clustered_data(n_clusters=3, genes_per_cluster=25)
    model = ClusteringProgramDiscovery(
        method="kmeans", n_programs=3, random_state=42
    )

    # Should not raise -- will auto-fit
    stability = model.compute_stability(
        embeddings, gene_names, n_iterations=3, subsample_fraction=0.8
    )
    assert stability["mean_jaccard"] >= 0.0


def test_umap_preprocessing_kmeans() -> None:
    """UMAP pre-processing should reduce dimensionality before clustering."""
    embeddings, gene_names = _make_clustered_data(
        n_clusters=4, genes_per_cluster=30, n_dims=64
    )
    model = ClusteringProgramDiscovery(
        method="kmeans",
        n_programs=4,
        random_state=42,
        use_umap=True,
        umap_n_components=10,
        umap_n_neighbors=15,
        umap_min_dist=0.0,
    )
    model.fit(embeddings, gene_names)

    programs = model.get_programs()
    assert isinstance(programs, dict)
    assert len(programs) == 4

    # UMAP embeddings should be stored
    assert model.umap_embeddings_ is not None
    assert model.umap_embeddings_.shape == (len(gene_names), 10)

    # Quality should still be computed
    qm = model.get_quality_metrics()
    assert np.isfinite(qm["silhouette_score"])


def test_umap_preprocessing_leiden() -> None:
    """UMAP pre-processing should work with Leiden clustering."""
    try:
        import igraph  # noqa: F401
        import leidenalg  # noqa: F401
    except ImportError:
        pytest.skip("igraph and/or leidenalg not installed")

    embeddings, gene_names = _make_clustered_data(
        n_clusters=4, genes_per_cluster=30, n_dims=64
    )
    model = ClusteringProgramDiscovery(
        method="leiden",
        k_neighbors=10,
        random_state=42,
        use_umap=True,
        umap_n_components=8,
    )
    model.fit(embeddings, gene_names)

    programs = model.get_programs()
    assert isinstance(programs, dict)
    assert len(programs) >= 2
    assert model.umap_embeddings_ is not None


def test_umap_with_multi_resolution_leiden() -> None:
    """UMAP + multi-resolution Leiden should work together."""
    try:
        import igraph  # noqa: F401
        import leidenalg  # noqa: F401
    except ImportError:
        pytest.skip("igraph and/or leidenalg not installed")

    embeddings, gene_names = _make_clustered_data(
        n_clusters=4, genes_per_cluster=25, n_dims=64
    )
    model = ClusteringProgramDiscovery(
        method="leiden",
        k_neighbors=10,
        random_state=42,
        use_umap=True,
        umap_n_components=8,
    )
    result = model.multi_resolution_leiden(
        embeddings,
        gene_names,
        n_resolutions=5,
        resolution_range=(0.5, 2.0),
    )

    assert result["n_communities"] >= 2
    assert model.umap_embeddings_ is not None
    programs = model.get_programs()
    assert len(programs) >= 2


def test_pairwise_jaccard_static_method() -> None:
    """Static Jaccard helper should compute correct values."""
    ref = {"A": ["g1", "g2", "g3"], "B": ["g4", "g5", "g6"]}
    sub = {"X": ["g1", "g2", "g3"], "Y": ["g4", "g5", "g7"]}

    j = ClusteringProgramDiscovery._compute_pairwise_jaccard(ref, sub)
    # A best-matches X: J(A,X) = 3/3 = 1.0
    # B best-matches Y: J(B,Y) = 2/4 = 0.5
    # Mean = 0.75
    assert abs(j - 0.75) < 1e-6


def test_pairwise_jaccard_empty() -> None:
    """Jaccard with empty programs should return 0."""
    assert ClusteringProgramDiscovery._compute_pairwise_jaccard({}, {"A": ["g1"]}) == 0.0
    assert ClusteringProgramDiscovery._compute_pairwise_jaccard({"A": ["g1"]}, {}) == 0.0


def test_gene_confidence_not_fitted() -> None:
    """get_gene_confidence before fit should raise RuntimeError."""
    model = ClusteringProgramDiscovery(method="kmeans", n_programs=3)
    with pytest.raises(RuntimeError):
        model.get_gene_confidence()


def test_quality_metrics_not_fitted() -> None:
    """get_quality_metrics before fit should raise RuntimeError."""
    model = ClusteringProgramDiscovery(method="kmeans", n_programs=3)
    with pytest.raises(RuntimeError):
        model.get_quality_metrics()


def test_discriminative_programs_basic() -> None:
    """get_discriminative_programs returns top-N genes per program with margin scores."""
    rng = np.random.default_rng(0)
    embeddings = rng.standard_normal((60, 10))
    gene_names = [f"gene_{i}" for i in range(60)]
    model = ClusteringProgramDiscovery(method="kmeans", n_programs=3, random_state=0)
    model.fit(embeddings, gene_names)
    disc = model.get_discriminative_programs(top_n=5)
    # Should return one entry per program
    assert len(disc) == 3
    # Each program has at most top_n genes
    for prog, genes in disc.items():
        assert len(genes) <= 5
        assert all(isinstance(g, str) for g in genes)
    # All gene names come from the original list
    all_disc_genes = set(g for genes in disc.values() for g in genes)
    assert all_disc_genes.issubset(set(gene_names))


def test_discriminative_programs_no_duplicates_within_program() -> None:
    """Discriminative programs should have no duplicate genes within a program."""
    rng = np.random.default_rng(1)
    embeddings = rng.standard_normal((50, 8))
    gene_names = [f"g{i}" for i in range(50)]
    model = ClusteringProgramDiscovery(method="kmeans", n_programs=4, random_state=1)
    model.fit(embeddings, gene_names)
    disc = model.get_discriminative_programs(top_n=10)
    for prog, genes in disc.items():
        assert len(genes) == len(set(genes)), f"Duplicates found in {prog}"


def test_discriminative_programs_not_fitted() -> None:
    """get_discriminative_programs before fit should raise RuntimeError."""
    model = ClusteringProgramDiscovery(method="kmeans", n_programs=3)
    with pytest.raises(RuntimeError):
        model.get_discriminative_programs()


def test_auto_k_small_input_clamps_to_valid_cluster_count() -> None:
    """Auto-K should not pick more clusters than samples."""
    embeddings = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    gene_names = ["g1", "g2"]
    model = ClusteringProgramDiscovery(
        method="kmeans",
        n_programs=None,
        k_range=(5, 30),
        random_state=0,
    )
    model.fit(embeddings, gene_names)

    assert len(model.get_programs()) == 2


def test_compute_stability_single_gene_returns_trivial_stability() -> None:
    """Single-gene inputs should not try to subsample more than the population."""
    embeddings = np.array([[1.0, 0.0]], dtype=np.float64)
    gene_names = ["g1"]
    model = ClusteringProgramDiscovery(method="kmeans", n_programs=1, random_state=0)
    model.fit(embeddings, gene_names)

    stability = model.compute_stability(
        embeddings,
        gene_names,
        n_iterations=2,
        subsample_fraction=0.5,
    )

    assert stability["stability_score"] == 1.0


def test_gene_confidence_handles_all_singleton_clusters() -> None:
    """Gene confidence should not crash when every sample forms its own cluster."""
    embeddings = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    gene_names = ["g1", "g2"]
    model = ClusteringProgramDiscovery(method="kmeans", n_programs=2, random_state=0)
    model.fit(embeddings, gene_names)

    confidence = model.get_gene_confidence()
    assert set(confidence.keys()) == {"program_0", "program_1"}
