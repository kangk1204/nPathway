"""Tests for ensemble/consensus discovery improvements."""

from __future__ import annotations

import numpy as np

from npathway.discovery.baselines import RandomProgramDiscovery
from npathway.discovery.clustering import ClusteringProgramDiscovery
from npathway.discovery.ensemble import EnsembleProgramDiscovery


def _make_clustered_data(
    n_clusters: int = 4,
    genes_per_cluster: int = 30,
    n_dims: int = 24,
    seed: int = 42,
) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    blocks: list[np.ndarray] = []
    names: list[str] = []
    for c in range(n_clusters):
        center = rng.standard_normal(n_dims) * 5.0
        block = center + 0.25 * rng.standard_normal((genes_per_cluster, n_dims))
        blocks.append(block)
        names.extend([f"C{c}_GENE_{i}" for i in range(genes_per_cluster)])
    return np.vstack(blocks).astype(np.float64), names


def _program_purity(programs: dict[str, list[str]]) -> float:
    if not programs:
        return 0.0
    per_program = []
    for genes in programs.values():
        if not genes:
            continue
        counts: dict[str, int] = {}
        for g in genes:
            prefix = g.split("_", 1)[0]
            counts[prefix] = counts.get(prefix, 0) + 1
        per_program.append(max(counts.values()) / len(genes))
    return float(np.mean(per_program)) if per_program else 0.0


def test_ensemble_coherence_weighting_improves_purity() -> None:
    embeddings, gene_names = _make_clustered_data()

    uniform = EnsembleProgramDiscovery(
        methods=[
            ClusteringProgramDiscovery(method="kmeans", n_programs=4, random_state=7),
            RandomProgramDiscovery(n_programs=4, genes_per_program=30, random_state=13),
        ],
        consensus_method="hierarchical",
        n_programs=4,
        method_weighting="uniform",
        random_state=7,
    )
    uniform.fit(embeddings, gene_names)
    purity_uniform = _program_purity(uniform.get_programs())

    coherence = EnsembleProgramDiscovery(
        methods=[
            ClusteringProgramDiscovery(method="kmeans", n_programs=4, random_state=7),
            RandomProgramDiscovery(n_programs=4, genes_per_program=30, random_state=13),
        ],
        consensus_method="hierarchical",
        n_programs=4,
        method_weighting="coherence",
        random_state=7,
    )
    coherence.fit(embeddings, gene_names)
    purity_coherence = _program_purity(coherence.get_programs())
    weights = coherence.get_method_weights()

    # The coherent clustering method should receive higher weight than random.
    assert float(weights[0]) > float(weights[1])
    # Coherence-weighted consensus should be at least as pure in this setup.
    assert purity_coherence >= purity_uniform
    assert purity_coherence > 0.80


def test_ensemble_method_weights_are_normalized() -> None:
    embeddings, gene_names = _make_clustered_data()
    model = EnsembleProgramDiscovery(
        methods=[
            ClusteringProgramDiscovery(method="kmeans", n_programs=4, random_state=1),
            ClusteringProgramDiscovery(method="kmeans", n_programs=4, random_state=2),
        ],
        consensus_method="hierarchical",
        n_programs=4,
        method_weighting="coherence",
        random_state=1,
    )
    model.fit(embeddings, gene_names)
    weights = model.get_method_weights()
    coherence = model.get_method_coherence()

    assert np.isfinite(weights).all()
    assert np.isfinite(coherence).all()
    assert np.all(weights > 0)
    assert np.isclose(float(np.sum(weights)), 1.0, atol=1e-10)


def test_hierarchical_consensus_single_gene_returns_single_program() -> None:
    """Single-gene consensus should not call agglomerative clustering with n=1."""
    embeddings = np.array([[1.0, 2.0]], dtype=np.float64)
    gene_names = ["g1"]
    model = EnsembleProgramDiscovery(
        methods=[ClusteringProgramDiscovery(method="kmeans", n_programs=1, random_state=0)],
        consensus_method="hierarchical",
        n_programs=1,
        random_state=0,
    )

    model.fit(embeddings, gene_names)
    assert model.get_programs() == {"consensus_0": ["g1"]}
