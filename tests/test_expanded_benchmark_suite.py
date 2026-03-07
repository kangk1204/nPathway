from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
MODULE_PATH = SCRIPTS_DIR / "run_expanded_benchmark_suite.py"


def _load_suite_module():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location("expanded_suite_module", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fuse_program_scores_combines_primary_and_secondary():
    mod = _load_suite_module()
    primary = {"p0": [("A", 1.0), ("B", 0.8), ("C", 0.6)]}
    secondary = {"s0": [("B", 1.0), ("D", 0.9), ("E", 0.7)]}
    fused = mod._fuse_program_scores(
        primary_scores=primary,
        secondary_scores=secondary,
        top_n_genes=4,
        secondary_weight=0.5,
    )
    assert "hybrid_0" in fused
    assert fused["hybrid_0"][:2] == ["B", "A"]
    assert "D" in fused["hybrid_0"]


def test_fuse_program_scores_fallback_to_primary_when_secondary_empty():
    mod = _load_suite_module()
    primary = {"p0": [("G1", 0.9), ("G2", 0.5), ("G3", 0.2)]}
    fused = mod._fuse_program_scores(
        primary_scores=primary,
        secondary_scores={},
        top_n_genes=3,
        secondary_weight=0.35,
    )
    assert fused == {"hybrid_0": ["G1", "G2", "G3"]}


def test_split_gene_sets_for_leakage_control_disjoint_cover():
    mod = _load_suite_module()
    gs = {f"S{i}": [f"G{i}", f"G{i+1}", f"G{i+2}"] for i in range(10)}
    tune, eval_ = mod._split_gene_sets_for_leakage_control(
        gs,
        tune_fraction=0.6,
        split_seed=42,
    )
    assert set(tune).isdisjoint(set(eval_))
    assert set(tune) | set(eval_) == set(gs)
    assert len(tune) > 0
    assert len(eval_) > 0


def test_build_primary_leaderboard_orders_by_primary_metrics():
    mod = _load_suite_module()
    method_summary = pd.DataFrame(
        {
            "dataset": ["d1", "d1", "d1"],
            "track": ["de_novo", "de_novo", "de_novo"],
            "method": ["A", "B", "C"],
            "discovery_mean_hallmark_alignment_mean": [0.40, 0.60, 0.50],
            "power_mean_fpr_mean": [0.08, 0.03, 0.05],
        }
    )
    out = mod.build_primary_leaderboard(
        method_summary,
        primary_metrics=[
            "discovery_mean_hallmark_alignment",
            "power_mean_fpr",
        ],
    )
    assert not out.empty
    assert out.iloc[0]["method"] == "B"


def test_fuse_program_scores_multi_combines_multiple_sources():
    mod = _load_suite_module()
    primary = {"p0": [("A", 1.0), ("B", 0.7), ("C", 0.5)]}
    sec1 = {"s1": [("B", 1.0), ("D", 0.9), ("E", 0.2)]}
    sec2 = {"s2": [("A", 0.8), ("E", 1.0), ("F", 0.9)]}
    fused = mod._fuse_program_scores_multi(
        primary_scores=primary,
        secondary_sources=[
            ("sec1", sec1, 0.4),
            ("sec2", sec2, 0.4),
        ],
        top_n_genes=6,
        fusion_mode="score_sum",
    )
    genes = fused["hybrid_0"]
    assert "D" in genes
    assert "E" in genes


def test_fuse_program_scores_multi_overlap_gate_blocks_unmatched_secondary():
    mod = _load_suite_module()
    primary = {"p0": [("G1", 1.0), ("G2", 0.6), ("G3", 0.3)]}
    sec = {"s0": [("X1", 1.0), ("X2", 0.8), ("X3", 0.7)]}
    fused = mod._fuse_program_scores_multi(
        primary_scores=primary,
        secondary_sources=[("sec", sec, 0.5)],
        top_n_genes=3,
        min_overlap_jaccard=0.2,
        fusion_mode="score_sum",
    )
    assert fused == {"hybrid_0": ["G1", "G2", "G3"]}


def test_program_internal_diversity_lower_for_overlapping_programs():
    mod = _load_suite_module()
    high_overlap = {
        "p0": ["A", "B", "C"],
        "p1": ["A", "B", "D"],
    }
    low_overlap = {
        "p0": ["A", "B", "C"],
        "p1": ["D", "E", "F"],
    }
    assert mod._program_internal_diversity(low_overlap) > mod._program_internal_diversity(
        high_overlap
    )


def test_robust_program_score_for_tuning_uses_worst_case_weight(monkeypatch: pytest.MonkeyPatch):
    mod = _load_suite_module()

    def _fake_score(programs, tuning_sets, gene_names):
        if "aux_marker" in tuning_sets:
            return 0.0
        return 1.0

    monkeypatch.setattr(mod, "_program_score_for_tuning", _fake_score)
    robust, components = mod._robust_program_score_for_tuning(
        programs={"p0": ["A", "B", "C"]},
        tuning_sets={"primary_marker": ["A", "B", "C"]},
        gene_names=["A", "B", "C"],
        auxiliary_tuning_sets=[("aux", {"aux_marker": ["A", "B", "C"]})],
        worst_case_weight=0.5,
    )
    assert robust == pytest.approx(0.25)
    assert components["primary"] == 1.0
    assert components["aux"] == 0.0


def test_prepare_group_labels_strict_rejects_index_parity_fallback():
    mod = _load_suite_module()
    adata = ad.AnnData(
        X=np.ones((4, 3), dtype=np.float32),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(4)]),
        var=pd.DataFrame(index=[f"G{i}" for i in range(3)]),
    )
    with pytest.raises(RuntimeError, match="Strict publication mode disallows index-parity"):
        mod._prepare_group_labels(adata, allow_index_parity_fallback=False)


def test_collection_with_fallback_strict_rejects_species_switch(monkeypatch: pytest.MonkeyPatch):
    mod = _load_suite_module()

    def fake_load_msigdb_gene_sets(*, collection: str, species: str):
        return {f"{species}_set": ["G1", "G2", "G3"]}

    def fake_filter_gene_sets_to_adata(raw, adata, min_genes=3):
        key = next(iter(raw))
        return raw if str(key).startswith("mouse_") else {}

    monkeypatch.setattr(mod, "load_msigdb_gene_sets", fake_load_msigdb_gene_sets)
    monkeypatch.setattr(mod, "filter_gene_sets_to_adata", fake_filter_gene_sets_to_adata)

    with pytest.raises(RuntimeError, match="disallows automatic fallback"):
        mod._collection_with_fallback(
            collection="hallmark",
            species="human",
            adata=object(),
            msigdb_species_mode="auto",
            allow_species_fallback=False,
        )


def test_load_dataset_bundle_strict_rejects_tuning_eval_reuse(monkeypatch: pytest.MonkeyPatch):
    mod = _load_suite_module()
    adata = ad.AnnData(
        X=np.ones((6, 4), dtype=np.float32),
        obs=pd.DataFrame({"louvain": ["0", "0", "1", "1", "0", "1"]}, index=[f"cell_{i}" for i in range(6)]),
        var=pd.DataFrame(index=[f"G{i}" for i in range(4)]),
    )

    monkeypatch.setattr(mod, "load_pbmc3k", lambda preprocessed=True: adata.copy())
    monkeypatch.setattr(
        mod,
        "build_gene_embeddings_from_expression",
        lambda adata, n_components=50: (np.ones((adata.n_vars, 2), dtype=np.float32), list(adata.var_names)),
    )
    monkeypatch.setattr(
        mod,
        "build_graph_regularized_embeddings",
        lambda adata, n_components=50, k_neighbors=15, n_diffusion_steps=3, alpha=0.5: (
            np.ones((adata.n_vars, 2), dtype=np.float32),
            list(adata.var_names),
        ),
    )

    def fake_collection_with_fallback(
        *,
        collection: str,
        species: str,
        adata,
        msigdb_species_mode: str,
        allow_species_fallback: bool = True,
    ):
        if collection == "hallmark":
            return {"HALLMARK_A": ["G0", "G1", "G2"]}, species, False
        if collection == "kegg":
            return {}, species, False
        if collection == "go_bp":
            return {"GO_A": ["G0", "G1", "G2"]}, species, False
        return {}, species, False

    monkeypatch.setattr(mod, "_collection_with_fallback", fake_collection_with_fallback)

    with pytest.raises(RuntimeError, match="forbids reusing evaluation gene sets"):
        mod.load_dataset_bundle(
            "pbmc3k",
            allow_proxy_datasets=False,
            strict_publication=True,
            tuning_collection="kegg",
            evaluation_collection="hallmark",
        )
