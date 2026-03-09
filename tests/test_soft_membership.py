"""Tests for soft (weighted) gene-program membership.

Covers:
- Base class fallback ``get_soft_programs()``
- ETM native soft output
- Clustering distance-based soft membership
- Ensemble co-occurrence soft membership
- Soft metrics (coverage, redundancy, specificity, entropy)
- Weighted GMT I/O round-trip
- Weighted Fisher enrichment
- Backward compatibility (hard programs unchanged)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from npathway.discovery.clustering import ClusteringProgramDiscovery
from npathway.discovery.topic_model import TopicModelProgramDiscovery
from npathway.evaluation.enrichment import weighted_fisher_enrichment
from npathway.evaluation.metrics import (
    membership_entropy,
    soft_comprehensive_evaluation,
    soft_coverage,
    soft_redundancy,
    soft_specificity,
)
from npathway.utils.gmt_io import read_weighted_gmt, weighted_programs_to_gmt


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture()
def simple_embeddings() -> tuple[np.ndarray, list[str]]:
    """Return small embeddings with clear cluster structure."""
    rng = np.random.RandomState(42)
    n_genes = 60
    dim = 10
    # 3 clusters: genes 0-19, 20-39, 40-59
    emb = rng.randn(n_genes, dim).astype(np.float32)
    emb[:20] += 3.0
    emb[20:40] -= 3.0
    emb[40:] += np.array([3.0, -3.0, 0, 0, 0, 0, 0, 0, 0, 0])
    gene_names = [f"GENE{i}" for i in range(n_genes)]
    return emb, gene_names


@pytest.fixture()
def fitted_clustering(
    simple_embeddings: tuple[np.ndarray, list[str]],
) -> ClusteringProgramDiscovery:
    """Return a fitted ClusteringProgramDiscovery instance."""
    emb, names = simple_embeddings
    disc = ClusteringProgramDiscovery(method="kmeans", n_programs=3, random_state=42)
    disc.fit(emb, names)
    return disc


@pytest.fixture()
def fitted_etm(
    simple_embeddings: tuple[np.ndarray, list[str]],
) -> TopicModelProgramDiscovery:
    """Return a fitted TopicModelProgramDiscovery instance."""
    emb, names = simple_embeddings
    disc = TopicModelProgramDiscovery(
        n_topics=3,
        n_epochs=5,
        top_n_genes=20,
        early_stopping_patience=0,
        random_state=42,
        device="cpu",
    )
    disc.fit(emb, names)
    return disc


# ------------------------------------------------------------------ #
# Base class fallback
# ------------------------------------------------------------------ #

class TestBaseFallback:
    """Test get_soft_programs fallback from hard program_scores."""

    def test_soft_programs_keys_match_hard(
        self, fitted_clustering: ClusteringProgramDiscovery
    ) -> None:
        hard = fitted_clustering.get_programs()
        soft = fitted_clustering.get_soft_programs(threshold=0.0)
        assert set(soft.keys()) == set(hard.keys())

    def test_soft_programs_contain_genes(
        self, fitted_clustering: ClusteringProgramDiscovery
    ) -> None:
        soft = fitted_clustering.get_soft_programs(threshold=0.0)
        for prog, gw in soft.items():
            assert len(gw) > 0
            for gene, weight in gw.items():
                assert isinstance(gene, str)
                assert isinstance(weight, float)
                assert weight >= 0.0

    def test_gene_memberships_transpose(
        self, fitted_clustering: ClusteringProgramDiscovery
    ) -> None:
        soft = fitted_clustering.get_soft_programs(threshold=0.0)
        memberships = fitted_clustering.get_gene_memberships()
        # Every gene in soft should appear in memberships
        for prog, gw in soft.items():
            for gene, weight in gw.items():
                assert gene in memberships
                assert prog in memberships[gene]
                assert abs(memberships[gene][prog] - weight) < 1e-6


# ------------------------------------------------------------------ #
# Clustering soft membership
# ------------------------------------------------------------------ #

class TestClusteringSoft:
    """Test distance-based soft membership from clustering."""

    def test_soft_programs_populated(
        self, fitted_clustering: ClusteringProgramDiscovery
    ) -> None:
        assert fitted_clustering.soft_programs_ is not None

    def test_all_genes_in_all_programs(
        self,
        fitted_clustering: ClusteringProgramDiscovery,
        simple_embeddings: tuple[np.ndarray, list[str]],
    ) -> None:
        """Every gene should have a weight in every program (before threshold)."""
        _, names = simple_embeddings
        soft = fitted_clustering.get_soft_programs(threshold=0.0)
        for prog, gw in soft.items():
            assert len(gw) == len(names)

    def test_weights_sum_to_one(
        self, fitted_clustering: ClusteringProgramDiscovery
    ) -> None:
        """Per-gene weights across programs should sum to ~1."""
        soft = fitted_clustering.get_soft_programs(threshold=0.0)
        prog_names = list(soft.keys())
        # Pick a sample gene
        gene = list(soft[prog_names[0]].keys())[0]
        total = sum(soft[p].get(gene, 0.0) for p in prog_names)
        assert abs(total - 1.0) < 1e-5

    def test_hard_programs_unchanged(
        self,
        fitted_clustering: ClusteringProgramDiscovery,
    ) -> None:
        """Hard programs should still work as before."""
        hard = fitted_clustering.get_programs()
        assert len(hard) == 3
        for prog, genes in hard.items():
            assert len(genes) > 0

    def test_gene_multi_membership(
        self, fitted_clustering: ClusteringProgramDiscovery
    ) -> None:
        """With a low threshold, genes should appear in multiple programs."""
        soft = fitted_clustering.get_soft_programs(threshold=0.01)
        memberships = fitted_clustering.get_gene_memberships()
        multi_count = sum(1 for g, progs in memberships.items() if len(progs) > 1)
        # With softmax, most genes should have non-zero weight in all programs
        assert multi_count > 0


# ------------------------------------------------------------------ #
# ETM soft membership
# ------------------------------------------------------------------ #

class TestETMSoft:
    """Test native soft output from ETM."""

    def test_soft_programs_populated(
        self, fitted_etm: TopicModelProgramDiscovery
    ) -> None:
        assert fitted_etm.soft_programs_ is not None

    def test_soft_has_more_genes_than_hard(
        self, fitted_etm: TopicModelProgramDiscovery
    ) -> None:
        """Soft programs contain all genes, not just top-N."""
        hard = fitted_etm.get_programs()
        soft = fitted_etm.get_soft_programs(threshold=0.0)
        for prog in hard:
            if prog in soft:
                assert len(soft[prog]) >= len(hard[prog])

    def test_soft_threshold_filtering(
        self, fitted_etm: TopicModelProgramDiscovery
    ) -> None:
        """Higher threshold should yield fewer genes."""
        soft_low = fitted_etm.get_soft_programs(threshold=0.001)
        soft_high = fitted_etm.get_soft_programs(threshold=0.1)
        for prog in soft_low:
            if prog in soft_high:
                assert len(soft_high[prog]) <= len(soft_low[prog])


# ------------------------------------------------------------------ #
# Soft metrics
# ------------------------------------------------------------------ #

class TestSoftMetrics:
    """Test soft coverage, redundancy, specificity, entropy."""

    @pytest.fixture()
    def sample_soft(self) -> dict[str, dict[str, float]]:
        return {
            "prog_A": {"G1": 0.8, "G2": 0.6, "G3": 0.1, "G4": 0.05},
            "prog_B": {"G1": 0.2, "G2": 0.4, "G3": 0.9, "G4": 0.95},
        }

    def test_soft_coverage(self, sample_soft: dict[str, dict[str, float]]) -> None:
        ref = ["G1", "G2", "G3", "G4", "G5"]
        # G5 not in any program → coverage < 1.0
        cov = soft_coverage(sample_soft, ref, threshold=0.1)
        assert 0.0 < cov < 1.0
        # G1-G3 pass threshold for at least one program; G4 passes in prog_B
        assert cov == pytest.approx(4 / 5)

    def test_soft_redundancy(self, sample_soft: dict[str, dict[str, float]]) -> None:
        # G1, G2, G3 all pass threshold=0.1 in both programs
        red = soft_redundancy(sample_soft, threshold=0.1)
        # Some genes in 2 programs → redundancy > 0
        assert red > 0.0

    def test_soft_specificity(self, sample_soft: dict[str, dict[str, float]]) -> None:
        spec = soft_specificity(sample_soft)
        assert 0.0 <= spec <= 1.0

    def test_membership_entropy(self, sample_soft: dict[str, dict[str, float]]) -> None:
        ent = membership_entropy(sample_soft)
        assert "G1" in ent
        # G1 has weights 0.8 and 0.2 → more specific than G2 (0.6, 0.4)
        assert ent["G1"] < ent["G2"]

    def test_comprehensive(self, sample_soft: dict[str, dict[str, float]]) -> None:
        ref = ["G1", "G2", "G3", "G4"]
        result = soft_comprehensive_evaluation(sample_soft, ref, threshold=0.1)
        assert "soft_coverage" in result
        assert "soft_redundancy" in result
        assert "soft_specificity" in result


# ------------------------------------------------------------------ #
# Weighted GMT I/O round-trip
# ------------------------------------------------------------------ #

class TestWeightedGMT:
    """Test weighted GMT write and read back."""

    def test_roundtrip(self) -> None:
        original: dict[str, list[tuple[str, float]]] = {
            "prog_A": [("GENE1", 0.95), ("GENE2", 0.7), ("GENE3", 0.1)],
            "prog_B": [("GENE4", 0.88), ("GENE5", 0.65)],
        }
        with tempfile.NamedTemporaryFile(suffix=".gmt", delete=False) as f:
            path = f.name

        weighted_programs_to_gmt(original, path)
        loaded = read_weighted_gmt(path)

        assert set(loaded.keys()) == set(original.keys())
        for prog in original:
            orig_genes = {g for g, _ in original[prog]}
            load_genes = {g for g, _ in loaded[prog]}
            assert orig_genes == load_genes
            # Check weights are preserved
            for g_orig, w_orig in original[prog]:
                for g_load, w_load in loaded[prog]:
                    if g_orig == g_load:
                        assert abs(w_orig - w_load) < 1e-4

        Path(path).unlink(missing_ok=True)


# ------------------------------------------------------------------ #
# Weighted Fisher enrichment
# ------------------------------------------------------------------ #

class TestWeightedFisher:
    """Test weighted Fisher enrichment."""

    def test_basic_weighted_fisher(self) -> None:
        gene_list = ["G1", "G2", "G3"]
        weighted_progs = {
            "prog_A": {"G1": 0.9, "G2": 0.8, "G4": 0.7, "G5": 0.1},
            "prog_B": {"G3": 0.95, "G6": 0.85, "G7": 0.6},
        }
        background = [f"G{i}" for i in range(1, 20)]
        df = weighted_fisher_enrichment(
            gene_list, weighted_progs, background=background, weight_threshold=0.1
        )
        assert len(df) > 0
        assert "weighted_overlap" in df.columns
        assert "p_value" in df.columns

    def test_weighted_overlap_score(self) -> None:
        gene_list = ["G1", "G2"]
        weighted_progs = {
            "prog_A": {"G1": 0.9, "G2": 0.3},
        }
        background = ["G1", "G2", "G3", "G4", "G5"]
        df = weighted_fisher_enrichment(
            gene_list, weighted_progs, background=background, weight_threshold=0.1
        )
        # Weighted overlap should be 0.9 + 0.3 = 1.2
        assert df.iloc[0]["weighted_overlap"] == pytest.approx(1.2, abs=1e-4)


# ------------------------------------------------------------------ #
# to_weighted_gmt via base class
# ------------------------------------------------------------------ #

class TestToWeightedGMT:
    """Test the base class to_weighted_gmt method."""

    def test_export(self, fitted_clustering: ClusteringProgramDiscovery) -> None:
        with tempfile.NamedTemporaryFile(suffix=".gmt", delete=False) as f:
            path = f.name
        fitted_clustering.to_weighted_gmt(path, threshold=0.01)
        loaded = read_weighted_gmt(path)
        assert len(loaded) > 0
        Path(path).unlink(missing_ok=True)
