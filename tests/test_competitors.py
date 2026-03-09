"""Tests for competitor baseline wrappers (scSpectra, scETM).

Tests cover:
- Graceful import fallback when packages are not installed.
- ``is_available()`` classmethod correctness.
- Basic interface conformance (unfitted raises, correct return types).
- Full smoke tests when the external packages are available (skipped otherwise).
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from npathway.discovery.competitors import (
    ScETMProgramDiscovery,
    SpectraProgramDiscovery,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _has_module(name: str) -> bool:
    """Return True if *name* is an importable Python package."""
    return importlib.util.find_spec(name) is not None


# ------------------------------------------------------------------
# Import fallback tests (always run)
# ------------------------------------------------------------------


class TestSpectraFallback:
    """Verify SpectraProgramDiscovery behaves correctly when scSpectra is absent."""

    def test_is_available_matches_import(self) -> None:
        """``is_available()`` should match actual importability of Spectra."""
        expected = _has_module("Spectra")
        assert SpectraProgramDiscovery.is_available() == expected

    def test_instantiation_does_not_raise(self) -> None:
        """Creating the wrapper should never fail, even without scSpectra."""
        model = SpectraProgramDiscovery(n_programs=5)
        assert model.programs_ is None

    def test_fit_without_package_raises_import_error(self) -> None:
        """If scSpectra is missing, ``fit()`` should raise ImportError."""
        if SpectraProgramDiscovery.is_available():
            pytest.skip("scSpectra is installed; fallback path not testable.")

        model = SpectraProgramDiscovery(n_programs=5)
        emb = np.random.default_rng(0).standard_normal((50, 10)).astype(np.float32)
        genes = [f"G{i}" for i in range(50)]

        with pytest.raises(ImportError, match="scSpectra is not installed"):
            model.fit(emb, genes, expression=np.zeros((20, 50)))

    def test_unfitted_get_programs_raises(self) -> None:
        """Accessing programs before fit should raise RuntimeError."""
        model = SpectraProgramDiscovery()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.get_programs()

    def test_unfitted_get_program_scores_raises(self) -> None:
        """Accessing program scores before fit should raise RuntimeError."""
        model = SpectraProgramDiscovery()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.get_program_scores()

    def test_unfitted_get_soft_programs_raises(self) -> None:
        """Accessing soft programs before fit should raise RuntimeError."""
        model = SpectraProgramDiscovery()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.get_soft_programs()

    def test_repr_unfitted(self) -> None:
        """repr should work for unfitted instances."""
        model = SpectraProgramDiscovery(n_programs=7)
        r = repr(model)
        assert "SpectraProgramDiscovery" in r
        assert "fitted=False" in r


class TestScETMFallback:
    """Verify ScETMProgramDiscovery behaves correctly when scETM is absent."""

    def test_is_available_matches_import(self) -> None:
        """``is_available()`` should match actual importability of scETM."""
        expected = _has_module("scETM")
        assert ScETMProgramDiscovery.is_available() == expected

    def test_instantiation_does_not_raise(self) -> None:
        """Creating the wrapper should never fail, even without scETM."""
        model = ScETMProgramDiscovery(n_programs=10)
        assert model.programs_ is None

    def test_fit_without_package_raises_import_error(self) -> None:
        """If scETM is missing, ``fit()`` should raise ImportError."""
        if ScETMProgramDiscovery.is_available():
            pytest.skip("scETM is installed; fallback path not testable.")

        model = ScETMProgramDiscovery(n_programs=5)
        emb = np.random.default_rng(0).standard_normal((50, 10)).astype(np.float32)
        genes = [f"G{i}" for i in range(50)]

        with pytest.raises(ImportError, match="scETM is not installed"):
            model.fit(emb, genes, expression=np.zeros((20, 50)))

    def test_unfitted_get_programs_raises(self) -> None:
        """Accessing programs before fit should raise RuntimeError."""
        model = ScETMProgramDiscovery()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.get_programs()

    def test_unfitted_get_program_scores_raises(self) -> None:
        """Accessing program scores before fit should raise RuntimeError."""
        model = ScETMProgramDiscovery()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.get_program_scores()

    def test_unfitted_get_soft_programs_raises(self) -> None:
        """Accessing soft programs before fit should raise RuntimeError."""
        model = ScETMProgramDiscovery()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.get_soft_programs()

    def test_repr_unfitted(self) -> None:
        """repr should work for unfitted instances."""
        model = ScETMProgramDiscovery(n_programs=12)
        r = repr(model)
        assert "ScETMProgramDiscovery" in r
        assert "fitted=False" in r


# ------------------------------------------------------------------
# Validation tests (always run, no external packages needed)
# ------------------------------------------------------------------


class TestSpectraInputValidation:
    """Input validation for SpectraProgramDiscovery (does not need scSpectra)."""

    @pytest.mark.skipif(
        not _has_module("Spectra"),
        reason="scSpectra not installed",
    )
    def test_missing_expression_raises(self) -> None:
        """fit() without expression= should raise ValueError."""
        model = SpectraProgramDiscovery(n_programs=3)
        emb = np.random.default_rng(1).standard_normal((20, 5)).astype(np.float32)
        genes = [f"G{i}" for i in range(20)]

        with pytest.raises(ValueError, match="requires an expression matrix"):
            model.fit(emb, genes)

    @pytest.mark.skipif(
        not _has_module("Spectra"),
        reason="scSpectra not installed",
    )
    def test_shape_mismatch_raises(self) -> None:
        """Expression column count must match gene_names length."""
        model = SpectraProgramDiscovery(n_programs=3)
        emb = np.random.default_rng(1).standard_normal((20, 5)).astype(np.float32)
        genes = [f"G{i}" for i in range(20)]
        wrong_expr = np.zeros((10, 15), dtype=np.float32)  # 15 != 20

        with pytest.raises(ValueError, match="columns"):
            model.fit(emb, genes, expression=wrong_expr)


class TestScETMInputValidation:
    """Input validation for ScETMProgramDiscovery (does not need scETM)."""

    @pytest.mark.skipif(
        not _has_module("scETM"),
        reason="scETM not installed",
    )
    def test_missing_expression_raises(self) -> None:
        """fit() without expression= should raise ValueError."""
        model = ScETMProgramDiscovery(n_programs=3)
        emb = np.random.default_rng(1).standard_normal((20, 5)).astype(np.float32)
        genes = [f"G{i}" for i in range(20)]

        with pytest.raises(ValueError, match="requires an expression matrix"):
            model.fit(emb, genes)

    @pytest.mark.skipif(
        not _has_module("scETM"),
        reason="scETM not installed",
    )
    def test_shape_mismatch_expression_raises(self) -> None:
        """Expression column count must match gene_names length."""
        model = ScETMProgramDiscovery(n_programs=3)
        emb = np.random.default_rng(1).standard_normal((20, 5)).astype(np.float32)
        genes = [f"G{i}" for i in range(20)]
        wrong_expr = np.zeros((10, 15), dtype=np.float32)

        with pytest.raises(ValueError, match="columns"):
            model.fit(emb, genes, expression=wrong_expr)

    @pytest.mark.skipif(
        not _has_module("scETM"),
        reason="scETM not installed",
    )
    def test_shape_mismatch_embeddings_raises(self) -> None:
        """Embedding row count must match gene_names length."""
        model = ScETMProgramDiscovery(n_programs=3)
        emb = np.random.default_rng(1).standard_normal((15, 5)).astype(np.float32)
        genes = [f"G{i}" for i in range(20)]
        expr = np.zeros((10, 20), dtype=np.float32)

        with pytest.raises(ValueError, match="rows"):
            model.fit(emb, genes, expression=expr)


# ------------------------------------------------------------------
# Full smoke tests (require external packages)
# ------------------------------------------------------------------


def _spectra_functional() -> bool:
    """Return True if scSpectra can run end-to-end (no torch/numpy ABI mismatch)."""
    if not _has_module("Spectra"):
        return False
    try:
        import torch  # type: ignore[import-untyped]

        # Spectra calls tensor.detach().numpy() internally; this will fail
        # when torch was compiled against numpy 1.x and numpy 2.x is installed.
        torch.tensor([1.0]).numpy()
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _spectra_functional(),
    reason="scSpectra not installed or torch/numpy ABI mismatch",
)
class TestSpectraSmokeIfInstalled:
    """End-to-end tests for SpectraProgramDiscovery (only run if scSpectra is available)."""

    def test_fit_returns_self(self) -> None:
        """fit() should return the instance itself."""
        rng = np.random.default_rng(42)
        n_genes, n_cells, n_dims = 60, 40, 16
        emb = rng.standard_normal((n_genes, n_dims)).astype(np.float32)
        expr = rng.gamma(2.0, 1.0, size=(n_cells, n_genes)).astype(np.float32)
        genes = [f"G{i}" for i in range(n_genes)]

        model = SpectraProgramDiscovery(n_programs=5, n_epochs=50, top_n_genes=10)
        result = model.fit(emb, genes, expression=expr)
        assert result is model

    def test_programs_nonempty(self) -> None:
        """Fitted model should have non-empty programs."""
        rng = np.random.default_rng(42)
        n_genes, n_cells, n_dims = 60, 40, 16
        emb = rng.standard_normal((n_genes, n_dims)).astype(np.float32)
        expr = rng.gamma(2.0, 1.0, size=(n_cells, n_genes)).astype(np.float32)
        genes = [f"G{i}" for i in range(n_genes)]

        model = SpectraProgramDiscovery(n_programs=5, n_epochs=50, top_n_genes=10)
        model.fit(emb, genes, expression=expr)

        programs = model.get_programs()
        assert len(programs) >= 1
        assert all(len(g) >= 1 for g in programs.values())

    def test_soft_programs_native(self) -> None:
        """Spectra should populate soft_programs_ natively from factors."""
        rng = np.random.default_rng(42)
        n_genes, n_cells, n_dims = 60, 40, 16
        emb = rng.standard_normal((n_genes, n_dims)).astype(np.float32)
        expr = rng.gamma(2.0, 1.0, size=(n_cells, n_genes)).astype(np.float32)
        genes = [f"G{i}" for i in range(n_genes)]

        model = SpectraProgramDiscovery(n_programs=5, n_epochs=50, top_n_genes=10)
        model.fit(emb, genes, expression=expr)

        assert model.soft_programs_ is not None
        soft = model.get_soft_programs(threshold=0.0)
        assert len(soft) >= 1
        # Every soft program should have gene->weight entries
        for prog, gw in soft.items():
            assert isinstance(gw, dict)
            for gene, weight in gw.items():
                assert isinstance(gene, str)
                assert isinstance(weight, float)
                assert weight >= 0.0

    def test_scores_match_programs(self) -> None:
        """Program scores keys should match program keys."""
        rng = np.random.default_rng(42)
        n_genes, n_cells, n_dims = 60, 40, 16
        emb = rng.standard_normal((n_genes, n_dims)).astype(np.float32)
        expr = rng.gamma(2.0, 1.0, size=(n_cells, n_genes)).astype(np.float32)
        genes = [f"G{i}" for i in range(n_genes)]

        model = SpectraProgramDiscovery(n_programs=5, n_epochs=50, top_n_genes=10)
        model.fit(emb, genes, expression=expr)

        programs = model.get_programs()
        scores = model.get_program_scores()
        assert set(programs.keys()) == set(scores.keys())

    def test_get_factors_shape(self) -> None:
        """get_factors() should return matrix with correct shape."""
        rng = np.random.default_rng(42)
        n_genes, n_cells, n_dims = 60, 40, 16
        emb = rng.standard_normal((n_genes, n_dims)).astype(np.float32)
        expr = rng.gamma(2.0, 1.0, size=(n_cells, n_genes)).astype(np.float32)
        genes = [f"G{i}" for i in range(n_genes)]

        model = SpectraProgramDiscovery(n_programs=5, n_epochs=50, top_n_genes=10)
        model.fit(emb, genes, expression=expr)

        factors = model.get_factors()
        assert factors.ndim == 2
        assert factors.shape[1] == n_genes


@pytest.mark.skipif(
    not _has_module("scETM"),
    reason="scETM not installed",
)
class TestScETMSmokeIfInstalled:
    """End-to-end tests for ScETMProgramDiscovery (only run if scETM is available)."""

    def test_fit_returns_self(self) -> None:
        """fit() should return the instance itself."""
        rng = np.random.default_rng(42)
        n_genes, n_cells, n_dims = 60, 40, 16
        emb = rng.standard_normal((n_genes, n_dims)).astype(np.float32)
        expr = rng.gamma(2.0, 1.0, size=(n_cells, n_genes)).astype(np.float32)
        genes = [f"G{i}" for i in range(n_genes)]

        model = ScETMProgramDiscovery(n_programs=5, n_epochs=5, top_n_genes=10)
        result = model.fit(emb, genes, expression=expr)
        assert result is model

    def test_programs_nonempty(self) -> None:
        """Fitted model should have non-empty programs."""
        rng = np.random.default_rng(42)
        n_genes, n_cells, n_dims = 60, 40, 16
        emb = rng.standard_normal((n_genes, n_dims)).astype(np.float32)
        expr = rng.gamma(2.0, 1.0, size=(n_cells, n_genes)).astype(np.float32)
        genes = [f"G{i}" for i in range(n_genes)]

        model = ScETMProgramDiscovery(n_programs=5, n_epochs=5, top_n_genes=10)
        model.fit(emb, genes, expression=expr)

        programs = model.get_programs()
        assert len(programs) >= 1
        assert all(len(g) >= 1 for g in programs.values())

    def test_soft_programs_native(self) -> None:
        """scETM should populate soft_programs_ natively from topic distributions."""
        rng = np.random.default_rng(42)
        n_genes, n_cells, n_dims = 60, 40, 16
        emb = rng.standard_normal((n_genes, n_dims)).astype(np.float32)
        expr = rng.gamma(2.0, 1.0, size=(n_cells, n_genes)).astype(np.float32)
        genes = [f"G{i}" for i in range(n_genes)]

        model = ScETMProgramDiscovery(n_programs=5, n_epochs=5, top_n_genes=10)
        model.fit(emb, genes, expression=expr)

        assert model.soft_programs_ is not None
        soft = model.get_soft_programs(threshold=0.0)
        assert len(soft) >= 1
        for prog, gw in soft.items():
            assert isinstance(gw, dict)
            for gene, weight in gw.items():
                assert isinstance(gene, str)
                assert isinstance(weight, float)
                assert weight >= 0.0

    def test_scores_match_programs(self) -> None:
        """Program scores keys should match program keys."""
        rng = np.random.default_rng(42)
        n_genes, n_cells, n_dims = 60, 40, 16
        emb = rng.standard_normal((n_genes, n_dims)).astype(np.float32)
        expr = rng.gamma(2.0, 1.0, size=(n_cells, n_genes)).astype(np.float32)
        genes = [f"G{i}" for i in range(n_genes)]

        model = ScETMProgramDiscovery(n_programs=5, n_epochs=5, top_n_genes=10)
        model.fit(emb, genes, expression=expr)

        programs = model.get_programs()
        scores = model.get_program_scores()
        assert set(programs.keys()) == set(scores.keys())

    def test_get_topic_gene_matrix_shape(self) -> None:
        """get_topic_gene_matrix() should return matrix with correct shape."""
        rng = np.random.default_rng(42)
        n_genes, n_cells, n_dims = 60, 40, 16
        emb = rng.standard_normal((n_genes, n_dims)).astype(np.float32)
        expr = rng.gamma(2.0, 1.0, size=(n_cells, n_genes)).astype(np.float32)
        genes = [f"G{i}" for i in range(n_genes)]

        model = ScETMProgramDiscovery(n_programs=5, n_epochs=5, top_n_genes=10)
        model.fit(emb, genes, expression=expr)

        mat = model.get_topic_gene_matrix()
        assert mat.ndim == 2
        assert mat.shape == (5, n_genes)

    def test_gene_memberships(self) -> None:
        """get_gene_memberships() should return per-gene dict."""
        rng = np.random.default_rng(42)
        n_genes, n_cells, n_dims = 60, 40, 16
        emb = rng.standard_normal((n_genes, n_dims)).astype(np.float32)
        expr = rng.gamma(2.0, 1.0, size=(n_cells, n_genes)).astype(np.float32)
        genes = [f"G{i}" for i in range(n_genes)]

        model = ScETMProgramDiscovery(n_programs=5, n_epochs=5, top_n_genes=10)
        model.fit(emb, genes, expression=expr)

        memberships = model.get_gene_memberships()
        assert isinstance(memberships, dict)
        # At least some genes should have memberships
        assert len(memberships) > 0
        for gene, prog_weights in memberships.items():
            assert isinstance(gene, str)
            assert isinstance(prog_weights, dict)
