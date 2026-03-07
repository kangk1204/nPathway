"""Tests for embedding extraction module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from npathway.embedding.base import BaseEmbeddingExtractor
from npathway.embedding.factory import get_extractor


def test_factory_invalid_model() -> None:
    """get_extractor should raise ValueError for an unrecognised model name."""
    with pytest.raises(ValueError, match="Unknown model name"):
        get_extractor("nonexistent_model_xyz")


class _ConcreteExtractor(BaseEmbeddingExtractor):
    """Minimal concrete subclass for testing the base class save/load."""

    def load_model(self, model_path: str) -> None:
        self.is_loaded = True

    def extract_gene_embeddings(self, adata, layer=-1):
        return np.zeros((0, 0))

    def extract_context_embeddings(self, adata, cell_type_key, layer=-1):
        return {}


def test_base_save_load_embeddings(tmp_output_dir: Path) -> None:
    """BaseEmbeddingExtractor.save_embeddings / load_embeddings roundtrip."""
    extractor = _ConcreteExtractor(model_name="test_model")

    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((50, 32)).astype(np.float32)
    gene_names = [f"GENE_{i}" for i in range(50)]

    base_path = str(tmp_output_dir / "test_emb")
    extractor.save_embeddings(embeddings, gene_names, base_path)

    # Verify files exist
    assert (tmp_output_dir / "test_emb.npz").exists()
    assert (tmp_output_dir / "test_emb.json").exists()

    # Verify metadata
    with open(tmp_output_dir / "test_emb.json", "r") as fh:
        metadata = json.load(fh)
    assert metadata["n_genes"] == 50
    assert metadata["embedding_dim"] == 32
    assert metadata["model_name"] == "test_model"

    # Load back
    loaded_emb, loaded_names = extractor.load_embeddings(base_path)
    np.testing.assert_array_almost_equal(loaded_emb, embeddings)
    assert loaded_names == gene_names


def test_scgpt_import_error_handling() -> None:
    """ScGPTEmbeddingExtractor.load_model should raise ImportError if scgpt is missing."""
    # Patch the import to simulate scgpt not being installed
    with patch.dict("sys.modules", {"scgpt": None, "scgpt.model": None, "scgpt.tokenizer": None}):
        try:
            from npathway.embedding.extract_scgpt import ScGPTEmbeddingExtractor

            extractor = ScGPTEmbeddingExtractor()
            with pytest.raises(ImportError):
                extractor.load_model("/nonexistent/path")
        except ImportError:
            # If the module itself can't import (e.g., torch missing), that's fine too
            pass


def test_geneformer_import_error_handling() -> None:
    """GeneformerEmbeddingExtractor.load_model should raise if dependencies missing."""
    with patch.dict("sys.modules", {"transformers": None}):
        try:
            from npathway.embedding.extract_geneformer import GeneformerEmbeddingExtractor

            extractor = GeneformerEmbeddingExtractor()
            with pytest.raises((ImportError, FileNotFoundError, RuntimeError)):
                extractor.load_model("/nonexistent/path")
        except (ImportError, RuntimeError):
            # Module can't import at all (e.g., torch incompatible with
            # Python version) -- acceptable
            pass


def test_scgpt_empty_adata_raises_value_error() -> None:
    """Empty AnnData should raise a clear error instead of returning NaNs."""
    from npathway.embedding.extract_scgpt import ScGPTEmbeddingExtractor

    extractor = ScGPTEmbeddingExtractor(device="cpu")
    extractor.model = object()
    extractor.is_loaded = True
    extractor.embedding_dim = 4
    extractor._n_layers = 1
    extractor.vocab = {"G1": 0, "G2": 1}

    adata = ad.AnnData(
        X=np.zeros((0, 2), dtype=np.float32),
        obs=pd.DataFrame(index=[]),
        var=pd.DataFrame(index=["G1", "G2"]),
    )

    with pytest.raises(ValueError, match="0 cells"):
        extractor.extract_gene_embeddings(adata)
