"""Common fixtures for nPathway tests."""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def sample_gene_programs() -> dict[str, list[str]]:
    """Five gene programs with known gene names."""
    return {
        "apoptosis": ["BCL2", "BAX", "CASP3", "CASP9", "TP53", "CYCS", "APAF1"],
        "cell_cycle": ["CDK1", "CDK2", "CCND1", "CCNE1", "RB1", "E2F1", "MYC"],
        "immune_response": ["TNF", "IL6", "IFNG", "CD8A", "CD4", "STAT1", "JAK2"],
        "metabolism": ["HK1", "PFKL", "PKM", "LDHA", "PDK1", "IDH1", "OGDH"],
        "signaling": ["EGFR", "KRAS", "BRAF", "MEK1", "ERK1", "AKT1", "MTOR"],
    }


@pytest.fixture
def sample_embeddings() -> tuple[np.ndarray, list[str]]:
    """200 gene embeddings of dimension 64, plus gene names."""
    rng = np.random.default_rng(42)
    n_genes = 200
    n_dims = 64
    embeddings = rng.standard_normal((n_genes, n_dims)).astype(np.float32)
    gene_names = [f"GENE_{i}" for i in range(n_genes)]
    return embeddings, gene_names


@pytest.fixture
def sample_adata():
    """Small AnnData: 100 cells x 50 genes with cell_type annotation."""
    import anndata

    rng = np.random.default_rng(42)
    n_cells = 100
    n_genes = 50
    X = rng.poisson(5, size=(n_cells, n_genes)).astype(np.float32)
    gene_names = [f"GENE_{i}" for i in range(n_genes)]
    cell_types = rng.choice(["T_cell", "B_cell", "Monocyte"], size=n_cells)

    import pandas as pd

    obs = pd.DataFrame({"cell_type": cell_types}, index=[f"cell_{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=gene_names)
    adata = anndata.AnnData(X=X, obs=obs, var=var)
    return adata


@pytest.fixture
def sample_curated_pathways() -> dict[str, list[str]]:
    """KEGG-like pathway dict with some overlap to sample_gene_programs."""
    return {
        "KEGG_APOPTOSIS": ["BCL2", "BAX", "CASP3", "CASP9", "TP53", "FAS", "BID"],
        "KEGG_CELL_CYCLE": ["CDK1", "CDK2", "CCND1", "CCNE1", "RB1", "CDC25A"],
        "KEGG_JAK_STAT": ["TNF", "IL6", "IFNG", "STAT1", "JAK2", "STAT3", "SOCS1"],
        "KEGG_GLYCOLYSIS": ["HK1", "PFKL", "PKM", "LDHA", "ENO1", "GAPDH"],
        "KEGG_MAPK": ["EGFR", "KRAS", "BRAF", "MAP2K1", "MAPK1", "RAF1"],
    }


@pytest.fixture
def tmp_output_dir() -> Path:
    """Temporary directory for file outputs, cleaned up afterwards."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_expression_matrix() -> np.ndarray:
    """Numpy array 100 x 50 with synthetic expression data."""
    rng = np.random.default_rng(42)
    return rng.lognormal(mean=2.0, sigma=1.0, size=(100, 50)).astype(np.float32)
