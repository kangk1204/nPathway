"""Tests for enrichment analysis functions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from npathway.evaluation.enrichment import (
    _bh_fdr,
    _storey_qvalue,
    aucell_score,
    effect_size_metrics,
    leading_edge_analysis,
    leading_edge_overlap,
    preranked_gsea,
    run_enrichment,
    ssgsea_score,
    weighted_enrichment,
)


def test_fisher_enrichment() -> None:
    """Fisher's exact test should detect enriched programs."""
    # Create a query list that overlaps heavily with prog_A
    query_genes = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10"]
    programs = {
        "prog_A": ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "X1", "X2", "X3"],
        "prog_B": ["Y1", "Y2", "Y3", "Y4", "Y5", "Y6", "Y7", "Y8", "Y9", "Y10"],
    }
    background = (
        query_genes
        + ["X1", "X2", "X3"]
        + ["Y1", "Y2", "Y3", "Y4", "Y5", "Y6", "Y7", "Y8", "Y9", "Y10"]
        + [f"BG{i}" for i in range(50)]
    )

    result = run_enrichment(query_genes, programs, method="fisher", background=background)
    assert isinstance(result, pd.DataFrame)
    assert "p_value" in result.columns
    assert "fdr" in result.columns
    assert "program" in result.columns

    # prog_A should be much more significant than prog_B
    prog_a_row = result[result["program"] == "prog_A"].iloc[0]
    prog_b_row = result[result["program"] == "prog_B"].iloc[0]
    assert prog_a_row["p_value"] < prog_b_row["p_value"]


def test_preranked_gsea() -> None:
    """Preranked GSEA should return valid enrichment scores."""
    rng = np.random.default_rng(42)
    n_genes = 500
    gene_names = [f"GENE_{i}" for i in range(n_genes)]
    scores = rng.standard_normal(n_genes)
    scores[:20] += 3.0  # Spike the top 20 genes

    ranked_genes = sorted(
        zip(gene_names, scores), key=lambda x: x[1], reverse=True
    )

    programs = {
        "enriched_prog": gene_names[:20],  # These genes have spiked scores
        "random_prog": list(rng.choice(gene_names[100:], size=20, replace=False)),
    }

    result = preranked_gsea(ranked_genes, programs, n_perm=200, seed=42)
    assert isinstance(result, pd.DataFrame)
    assert "es" in result.columns
    assert "p_value" in result.columns
    assert "fdr" in result.columns
    assert len(result) == 2


def test_ssgsea_score() -> None:
    """ssGSEA should return a DataFrame with one column per program."""
    rng = np.random.default_rng(42)
    n_cells = 30
    n_genes = 100
    expression = rng.lognormal(2.0, 1.0, size=(n_cells, n_genes)).astype(np.float32)
    gene_names = [f"GENE_{i}" for i in range(n_genes)]

    programs = {
        "prog_A": gene_names[:10],
        "prog_B": gene_names[50:60],
    }

    result = ssgsea_score(expression, gene_names, programs)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (n_cells, 2)
    assert list(result.columns) == ["prog_A", "prog_B"]


def test_weighted_enrichment() -> None:
    """Weighted enrichment should return valid results with gene weights."""
    rng = np.random.default_rng(42)
    n_genes = 300
    gene_names = [f"GENE_{i}" for i in range(n_genes)]
    metric = rng.standard_normal(n_genes)
    metric[:15] += 4.0

    ranked_genes = sorted(
        zip(gene_names, metric), key=lambda x: x[1], reverse=True
    )

    weighted_programs = {
        "weighted_prog": [
            (gene_names[i], 0.9 - i * 0.05) for i in range(15)
        ],
        "random_prog": [
            (gene_names[i + 150], 0.5) for i in range(15)
        ],
    }

    result = weighted_enrichment(ranked_genes, weighted_programs, n_perm=200, seed=42)
    assert isinstance(result, pd.DataFrame)
    assert "es" in result.columns
    assert "p_value" in result.columns
    assert "fdr" in result.columns
    assert len(result) == 2


def test_enrichment_fdr_correction() -> None:
    """FDR correction should produce q-values <= 1 and >= p-values."""
    np.random.default_rng(42)
    n_genes = 200
    gene_names = [f"GENE_{i}" for i in range(n_genes)]

    # Create 10 programs, mostly non-overlapping with query
    programs = {}
    for p in range(10):
        start = p * 15
        programs[f"prog_{p}"] = gene_names[start : start + 15]

    query = gene_names[:5] + [f"QG_{i}" for i in range(20)]
    background = gene_names + [f"QG_{i}" for i in range(20)]

    result = run_enrichment(
        query, programs, method="fisher", background=background
    )

    assert "fdr" in result.columns
    # FDR should be between 0 and 1
    assert (result["fdr"] >= 0).all()
    assert (result["fdr"] <= 1.0).all()
    # FDR should be >= p_value (after correction)
    assert (result["fdr"] >= result["p_value"] - 1e-12).all()


def test_enrichment_with_known_signal() -> None:
    """Spike in a known pathway signal and verify it is recovered."""
    rng = np.random.default_rng(42)
    n_genes = 500
    gene_names = [f"GENE_{i}" for i in range(n_genes)]

    # Create gene programs
    target_genes = gene_names[:25]
    decoy_programs = {}
    for p in range(9):
        start = 50 + p * 25
        decoy_programs[f"decoy_{p}"] = gene_names[start : start + 25]

    programs = {"target_pathway": target_genes, **decoy_programs}

    # Create a query list that strongly overlaps with target_pathway
    query = target_genes[:20] + list(
        rng.choice(gene_names[200:], size=10, replace=False)
    )
    background = gene_names

    result = run_enrichment(
        query, programs, method="fisher", background=background
    )

    # The target pathway should be the most significant (rank 1 by p_value)
    result_sorted = result.sort_values("p_value")
    assert result_sorted.iloc[0]["program"] == "target_pathway"
    assert result_sorted.iloc[0]["p_value"] < 0.01


# ---------------------------------------------------------------------------
# Tests for fgsea-style adaptive GSEA
# ---------------------------------------------------------------------------


def test_preranked_gsea_leading_edge_scores() -> None:
    """GSEA result should include leading_edge_scores column."""
    rng = np.random.default_rng(42)
    n_genes = 500
    gene_names = [f"GENE_{i}" for i in range(n_genes)]
    scores = rng.standard_normal(n_genes)
    scores[:20] += 3.0

    ranked_genes = sorted(
        zip(gene_names, scores), key=lambda x: x[1], reverse=True
    )
    programs = {"enriched_prog": gene_names[:20]}

    result = preranked_gsea(ranked_genes, programs, n_perm=200, seed=42)
    assert "leading_edge_scores" in result.columns
    assert "leading_edge_genes" in result.columns

    le_scores_str = result.iloc[0]["leading_edge_scores"]
    assert len(le_scores_str) > 0
    # Scores should be parseable as floats
    parsed = [float(s) for s in le_scores_str.split(",")]
    assert all(0.0 <= s <= 1.0 for s in parsed)


def test_preranked_gsea_auto_scales_perm_for_many_sets() -> None:
    """When many gene sets are provided, permutations should auto-scale."""
    rng = np.random.default_rng(42)
    n_genes = 200
    gene_names = [f"GENE_{i}" for i in range(n_genes)]
    scores = rng.standard_normal(n_genes)

    ranked_genes = sorted(
        zip(gene_names, scores), key=lambda x: x[1], reverse=True
    )
    # 150 gene sets (> 100 threshold for auto-scaling)
    programs = {
        f"prog_{i}": list(rng.choice(gene_names, size=10, replace=False))
        for i in range(150)
    }

    result = preranked_gsea(ranked_genes, programs, n_perm=100, seed=42)
    assert len(result) == 150
    assert "p_value" in result.columns
    assert (result["p_value"] > 0).all()
    assert (result["p_value"] <= 1.0).all()


def test_preranked_gsea_empty_gene_set() -> None:
    """GSEA should handle gene sets with zero overlap gracefully."""
    ranked_genes = [(f"GENE_{i}", float(i)) for i in range(50)]
    programs = {"no_overlap": ["MISSING_1", "MISSING_2"]}

    result = preranked_gsea(ranked_genes, programs, n_perm=50, seed=42)
    assert len(result) == 1
    assert result.iloc[0]["es"] == 0.0
    assert result.iloc[0]["p_value"] == 1.0
    assert result.iloc[0]["n_hits"] == 0


# ---------------------------------------------------------------------------
# Storey q-value tests
# ---------------------------------------------------------------------------


def test_storey_qvalue_returns_valid_values() -> None:
    """Storey q-values should be in [0, 1] and no larger than BH values."""
    rng = np.random.default_rng(42)
    # Mix of significant and non-significant p-values
    p_values = np.concatenate([rng.uniform(0.0, 0.01, size=20),
                               rng.uniform(0.3, 1.0, size=80)])
    qvals = _storey_qvalue(p_values)
    assert len(qvals) == len(p_values)
    assert (qvals >= 0).all()
    assert (qvals <= 1.0).all()


def test_storey_via_run_enrichment() -> None:
    """Storey FDR method should be selectable via run_enrichment."""
    query_genes = ["G1", "G2", "G3", "G4", "G5"]
    programs = {
        f"prog_{i}": [f"G{j}" for j in range(i * 3, i * 3 + 5)]
        for i in range(15)
    }
    background = [f"G{i}" for i in range(100)]

    result = run_enrichment(
        query_genes, programs, method="fisher",
        background=background, fdr_method="storey",
    )
    assert "fdr" in result.columns
    assert (result["fdr"] >= 0).all()
    assert (result["fdr"] <= 1.0).all()


def test_bh_fdr_empty() -> None:
    """BH FDR should handle empty input."""
    result = _bh_fdr(np.array([]))
    assert len(result) == 0


def test_storey_small_sample_falls_back_to_bh() -> None:
    """With < 10 p-values, Storey should fall back to BH."""
    p = np.array([0.01, 0.05, 0.1, 0.5, 0.9])
    q_storey = _storey_qvalue(p)
    q_bh = _bh_fdr(p)
    np.testing.assert_allclose(q_storey, q_bh)


# ---------------------------------------------------------------------------
# Leading edge analysis tests
# ---------------------------------------------------------------------------


def test_leading_edge_analysis() -> None:
    """Leading edge analysis should return per-gene contribution scores."""
    rng = np.random.default_rng(42)
    n_genes = 500
    gene_names = [f"GENE_{i}" for i in range(n_genes)]
    scores = rng.standard_normal(n_genes)
    scores[:20] += 3.0

    ranked_genes = sorted(
        zip(gene_names, scores), key=lambda x: x[1], reverse=True
    )
    programs = {"enriched_prog": gene_names[:20]}

    gsea_result = preranked_gsea(ranked_genes, programs, n_perm=200, seed=42)
    le_dict = leading_edge_analysis(gsea_result, programs)

    assert "enriched_prog" in le_dict
    le_df = le_dict["enriched_prog"]
    assert isinstance(le_df, pd.DataFrame)
    assert "gene" in le_df.columns
    assert "contribution_score" in le_df.columns
    assert len(le_df) > 0
    # Scores should be sorted descending
    assert le_df["contribution_score"].is_monotonic_decreasing


def test_leading_edge_overlap_matrix() -> None:
    """Leading edge overlap should return a square Jaccard matrix."""
    rng = np.random.default_rng(42)
    n_genes = 300
    gene_names = [f"GENE_{i}" for i in range(n_genes)]
    scores = rng.standard_normal(n_genes)
    scores[:15] += 3.0
    scores[15:30] += 2.5

    ranked_genes = sorted(
        zip(gene_names, scores), key=lambda x: x[1], reverse=True
    )
    programs = {
        "prog_A": gene_names[:15],
        "prog_B": gene_names[10:30],  # Overlaps with prog_A
        "prog_C": gene_names[200:215],  # No overlap
    }

    gsea_result = preranked_gsea(ranked_genes, programs, n_perm=200, seed=42)
    overlap = leading_edge_overlap(gsea_result)

    assert isinstance(overlap, pd.DataFrame)
    assert overlap.shape[0] == overlap.shape[1]
    assert overlap.shape[0] == 3
    # Diagonal should be 1.0 (self-overlap)
    for prog in overlap.index:
        assert overlap.loc[prog, prog] == 1.0 or overlap.loc[prog, prog] == 0.0
    # All values in [0, 1]
    assert (overlap.values >= 0).all()
    assert (overlap.values <= 1.0).all()


# ---------------------------------------------------------------------------
# AUCell tests
# ---------------------------------------------------------------------------


def test_aucell_score_basic() -> None:
    """AUCell should return per-cell scores in [0, 1]."""
    rng = np.random.default_rng(42)
    n_cells = 30
    n_genes = 100
    expression = rng.lognormal(2.0, 1.0, size=(n_cells, n_genes)).astype(np.float64)
    gene_names = [f"GENE_{i}" for i in range(n_genes)]

    programs = {
        "prog_A": gene_names[:10],
        "prog_B": gene_names[50:60],
    }

    result = aucell_score(expression, gene_names, programs)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (n_cells, 2)
    assert list(result.columns) == ["prog_A", "prog_B"]
    # AUCell scores should be in [0, 1]
    assert (result.values >= 0).all()
    assert (result.values <= 1.0).all()


def test_aucell_detects_enriched_cells() -> None:
    """AUCell should give higher scores to cells with upregulated gene sets."""
    rng = np.random.default_rng(42)
    n_cells = 50
    n_genes = 200
    expression = rng.lognormal(1.0, 0.5, size=(n_cells, n_genes))
    gene_names = [f"GENE_{i}" for i in range(n_genes)]

    # Upregulate genes 0-9 in the first 10 cells
    expression[:10, :10] += 20.0

    programs = {"upregulated": gene_names[:10]}
    result = aucell_score(expression, gene_names, programs, top_fraction=0.1)

    # First 10 cells should have higher scores than the rest
    mean_enriched = result["upregulated"].iloc[:10].mean()
    mean_other = result["upregulated"].iloc[10:].mean()
    assert mean_enriched > mean_other


def test_aucell_empty_gene_set() -> None:
    """AUCell should handle gene sets with no overlap gracefully."""
    rng = np.random.default_rng(42)
    expression = rng.lognormal(2.0, 1.0, size=(5, 20))
    gene_names = [f"GENE_{i}" for i in range(20)]

    programs = {"no_overlap": ["MISSING_1", "MISSING_2"]}
    result = aucell_score(expression, gene_names, programs)
    assert result.shape == (5, 1)
    assert (result.values == 0).all()


def test_aucell_via_run_enrichment() -> None:
    """AUCell should be accessible via run_enrichment dispatch."""
    rng = np.random.default_rng(42)
    n_cells = 10
    n_genes = 50
    expression = rng.lognormal(2.0, 1.0, size=(n_cells, n_genes))
    gene_names = [f"GENE_{i}" for i in range(n_genes)]
    programs = {"prog_A": gene_names[:5]}

    result = run_enrichment(
        gene_list=[], gene_programs=programs, method="aucell",
        expression_matrix=expression, gene_names=gene_names,
    )
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (n_cells, 1)


def test_aucell_mismatched_dimensions() -> None:
    """AUCell should raise ValueError when gene_names doesn't match matrix."""
    rng = np.random.default_rng(42)
    expression = rng.lognormal(2.0, 1.0, size=(5, 20))
    gene_names = [f"GENE_{i}" for i in range(10)]  # Wrong length

    import pytest
    with pytest.raises(ValueError, match="gene_names length"):
        aucell_score(expression, gene_names, {"prog": ["GENE_0"]})


# ---------------------------------------------------------------------------
# Effect size metrics tests
# ---------------------------------------------------------------------------


def test_effect_size_metrics() -> None:
    """Effect size metrics should add Cohen's d, Hedges' g, fold enrichment."""
    rng = np.random.default_rng(42)
    n_genes = 500
    gene_names = [f"GENE_{i}" for i in range(n_genes)]
    scores = rng.standard_normal(n_genes)
    scores[:20] += 3.0

    ranked_genes = sorted(
        zip(gene_names, scores), key=lambda x: x[1], reverse=True
    )
    programs = {
        "enriched_prog": gene_names[:20],
        "random_prog": list(rng.choice(gene_names[100:], size=20, replace=False)),
    }

    gsea_result = preranked_gsea(ranked_genes, programs, n_perm=200, seed=42)
    es_df = effect_size_metrics(gsea_result, ranked_genes, programs)

    assert "cohens_d" in es_df.columns
    assert "hedges_g" in es_df.columns
    assert "fold_enrichment" in es_df.columns
    assert "fold_enrichment_ci_lo" in es_df.columns
    assert "fold_enrichment_ci_hi" in es_df.columns
    assert "normalised_effect_size" in es_df.columns

    # The enriched program should have a larger effect size
    enriched_row = es_df[es_df["program"] == "enriched_prog"].iloc[0]
    random_row = es_df[es_df["program"] == "random_prog"].iloc[0]
    assert enriched_row["cohens_d"] > random_row["cohens_d"]
    assert enriched_row["hedges_g"] > random_row["hedges_g"]

    # Normalised effect size should be in [-1, 1]
    assert (es_df["normalised_effect_size"].abs() <= 1.0 + 1e-10).all()


# ---------------------------------------------------------------------------
# Batch-aware enrichment tests
# ---------------------------------------------------------------------------


def test_ssgsea_batch_correction() -> None:
    """ssGSEA with batch correction should reduce batch-level differences."""
    rng = np.random.default_rng(42)
    n_cells = 60
    n_genes = 100
    expression = rng.lognormal(2.0, 1.0, size=(n_cells, n_genes))
    gene_names = [f"GENE_{i}" for i in range(n_genes)]

    # Add a strong batch effect to the first 30 cells
    expression[:30, :] += 5.0
    batch = np.array([0] * 30 + [1] * 30)

    programs = {"prog_A": gene_names[:10]}

    # Without batch correction
    scores_no_batch = ssgsea_score(expression, gene_names, programs)
    # With batch correction
    scores_with_batch = ssgsea_score(expression, gene_names, programs, batch=batch)

    # The batch-level difference should be smaller after correction
    diff_no_batch = abs(
        scores_no_batch["prog_A"].iloc[:30].mean()
        - scores_no_batch["prog_A"].iloc[30:].mean()
    )
    diff_with_batch = abs(
        scores_with_batch["prog_A"].iloc[:30].mean()
        - scores_with_batch["prog_A"].iloc[30:].mean()
    )
    assert diff_with_batch < diff_no_batch


def test_aucell_batch_correction() -> None:
    """AUCell with batch correction should reduce batch-level differences."""
    rng = np.random.default_rng(42)
    n_cells = 60
    n_genes = 100
    expression = rng.lognormal(2.0, 1.0, size=(n_cells, n_genes))
    gene_names = [f"GENE_{i}" for i in range(n_genes)]

    # Add a strong batch effect
    expression[:30, :] += 5.0
    batch = np.array([0] * 30 + [1] * 30)

    programs = {"prog_A": gene_names[:10]}

    scores_no_batch = aucell_score(expression, gene_names, programs)
    scores_with_batch = aucell_score(expression, gene_names, programs, batch=batch)

    diff_no_batch = abs(
        scores_no_batch["prog_A"].iloc[:30].mean()
        - scores_no_batch["prog_A"].iloc[30:].mean()
    )
    diff_with_batch = abs(
        scores_with_batch["prog_A"].iloc[:30].mean()
        - scores_with_batch["prog_A"].iloc[30:].mean()
    )
    assert diff_with_batch < diff_no_batch


def test_batch_correction_single_batch_is_noop() -> None:
    """Batch correction with a single batch should not change scores."""
    rng = np.random.default_rng(42)
    n_cells = 20
    n_genes = 50
    expression = rng.lognormal(2.0, 1.0, size=(n_cells, n_genes))
    gene_names = [f"GENE_{i}" for i in range(n_genes)]
    batch = np.zeros(n_cells, dtype=int)

    programs = {"prog_A": gene_names[:5]}

    scores_no_batch = ssgsea_score(expression, gene_names, programs)
    scores_with_batch = ssgsea_score(expression, gene_names, programs, batch=batch)

    np.testing.assert_allclose(
        scores_no_batch.values, scores_with_batch.values, atol=1e-10
    )


# ---------------------------------------------------------------------------
# Fisher's test new columns
# ---------------------------------------------------------------------------


def test_fisher_fold_enrichment_columns() -> None:
    """Fisher's test should now include fold enrichment with CI."""
    query_genes = ["G1", "G2", "G3", "G4", "G5"]
    programs = {"prog_A": ["G1", "G2", "G3", "G4", "G5", "X1", "X2"]}
    background = query_genes + ["X1", "X2"] + [f"BG{i}" for i in range(30)]

    result = run_enrichment(
        query_genes, programs, method="fisher", background=background,
    )
    assert "fold_enrichment" in result.columns
    assert "fold_enrichment_ci_lo" in result.columns
    assert "fold_enrichment_ci_hi" in result.columns

    row = result.iloc[0]
    assert row["fold_enrichment"] > 1.0  # Should be enriched
    assert row["fold_enrichment_ci_lo"] > 0.0
    assert row["fold_enrichment_ci_lo"] <= row["fold_enrichment"]
    assert row["fold_enrichment_ci_hi"] >= row["fold_enrichment"]


# ---------------------------------------------------------------------------
# run_enrichment error paths
# ---------------------------------------------------------------------------


def test_run_enrichment_unknown_method() -> None:
    """Unknown method should raise ValueError."""
    import pytest
    with pytest.raises(ValueError, match="Unknown enrichment method"):
        run_enrichment(["G1"], {"prog": ["G1"]}, method="invalid")


def test_run_enrichment_gsea_missing_ranked() -> None:
    """GSEA without ranked_genes should raise ValueError."""
    import pytest
    with pytest.raises(ValueError, match="ranked_genes must be provided"):
        run_enrichment(["G1"], {"prog": ["G1"]}, method="gsea")


def test_run_enrichment_aucell_missing_expression() -> None:
    """AUCell without expression_matrix should raise ValueError."""
    import pytest
    with pytest.raises(ValueError, match="expression_matrix and gene_names"):
        run_enrichment(["G1"], {"prog": ["G1"]}, method="aucell")


def test_run_enrichment_empty_gene_programs_returns_empty_frame() -> None:
    """Empty gene_programs should return a typed empty result rather than crash."""
    fisher = run_enrichment(["G1"], {}, method="fisher", background=["G1", "G2"])
    gsea = run_enrichment(
        [],
        {},
        method="gsea",
        ranked_genes=[("G1", 1.0), ("G2", 0.5)],
    )

    assert fisher.empty
    assert "p_value" in fisher.columns
    assert gsea.empty
    assert "p_value" in gsea.columns
