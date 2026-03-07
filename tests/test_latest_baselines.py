"""Tests for latest external baseline adapters."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from npathway.discovery.baselines import (
    ExpressionClusteringBaseline,
    OfficialCNMFProgramDiscovery,
    StarCATReferenceProgramDiscovery,
)


def _has_module(module_name: str) -> bool:
    """Return True if an optional dependency is importable."""
    return importlib.util.find_spec(module_name) is not None


@pytest.mark.skipif(not _has_module("cnmf"), reason="cnmf not installed")
def test_official_cnmf_discovery_smoke() -> None:
    """Official cNMF wrapper should return non-empty scored programs."""
    rng = np.random.default_rng(7)
    expression = rng.gamma(shape=2.0, scale=1.0, size=(48, 90)).astype(np.float64)
    gene_names = [f"G{i}" for i in range(expression.shape[1])]

    model = OfficialCNMFProgramDiscovery(
        n_programs=5,
        n_iter=2,
        top_n_genes=12,
        density_threshold=0.5,
        max_nmf_iter=120,
        num_highvar_genes=90,
        random_state=7,
    )
    model.fit(expression, gene_names)

    programs = model.get_programs()
    scores = model.get_program_scores()

    assert 1 <= len(programs) <= 5
    assert set(programs) == set(scores)
    assert all(len(genes) >= 3 for genes in programs.values())
    assert all(len(scored) == len(programs[name]) for name, scored in scores.items())


def test_starcat_reference_mapping_case_insensitive() -> None:
    """Case-insensitive gene mapping should map uppercase reference genes."""
    ref = pd.DataFrame(
        data=[
            [1.0, 0.7, 0.1, 0.0, 0.0],
            [0.0, 0.1, 0.9, 0.8, 0.0],
        ],
        index=["Tcell", "Myeloid"],
        columns=["CD3D", "TRAC", "LST1", "S100A8", "MKI67"],
    )
    gene_names = ["Cd3d", "Trac", "Lst1", "S100a8", "Mki67", "H2-Ab1"]
    emb = np.arange(len(gene_names) * 4, dtype=np.float64).reshape(len(gene_names), 4)

    model = StarCATReferenceProgramDiscovery(
        reference_table=ref,
        top_n_genes=3,
        case_insensitive=True,
    )
    model.fit(emb, gene_names)
    programs = model.get_programs()

    assert "starcat_Tcell" in programs
    assert "starcat_Myeloid" in programs
    assert {"Cd3d", "Trac"}.issubset(set(programs["starcat_Tcell"]))
    assert {"Lst1", "S100a8"}.issubset(set(programs["starcat_Myeloid"]))


def test_starcat_reference_mapping_no_overlap_raises() -> None:
    """No overlap between reference and dataset genes should raise."""
    ref = pd.DataFrame(
        data=[[1.0, 0.8], [0.7, 0.6]],
        index=["ProgramA", "ProgramB"],
        columns=["CD3D", "TRAC"],
    )
    gene_names = ["GeneX", "GeneY", "GeneZ"]
    emb = np.zeros((len(gene_names), 3), dtype=np.float64)

    model = StarCATReferenceProgramDiscovery(
        reference_table=ref,
        top_n_genes=2,
        case_insensitive=True,
    )
    with pytest.raises(RuntimeError, match="produced no programs"):
        model.fit(emb, gene_names)


def test_expression_clustering_single_gene_returns_single_program() -> None:
    """A one-gene input should yield a trivial single-gene program."""
    model = ExpressionClusteringBaseline(n_programs=5, random_state=0)
    expression = np.array([[1.0], [2.0]], dtype=np.float64)
    model.fit(expression, ["g1"])

    assert model.get_programs() == {"expr_cluster_0": ["g1"]}
