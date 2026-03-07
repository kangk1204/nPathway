"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from npathway.evaluation.metrics import (
    benjamini_hochberg_fdr,
    benjamini_hochberg_fdr_grouped,
    bootstrap_metric,
    compare_methods,
    comprehensive_evaluation,
    compute_overlap_matrix,
    coverage,
    effect_size_cohens_d,
    fold_enrichment_with_ci,
    inter_program_distance,
    jaccard_similarity,
    novelty_score,
    paired_rank_biserial,
    paired_rank_biserial_correlation,
    paired_sign_test_pvalue,
    permutation_test_enrichment,
    program_redundancy,
    program_specificity,
    program_stability,
)


def test_jaccard_similarity() -> None:
    """Jaccard similarity should be correct for partially overlapping sets."""
    s1 = {"A", "B", "C", "D"}
    s2 = {"C", "D", "E", "F"}
    # intersection = {C, D}, union = {A, B, C, D, E, F}
    expected = 2.0 / 6.0
    result = jaccard_similarity(s1, s2)
    assert abs(result - expected) < 1e-10


def test_jaccard_identical_sets() -> None:
    """Jaccard of identical sets should be 1.0."""
    s = {"A", "B", "C"}
    assert jaccard_similarity(s, s) == 1.0


def test_jaccard_disjoint_sets() -> None:
    """Jaccard of disjoint sets should be 0.0."""
    s1 = {"A", "B"}
    s2 = {"C", "D"}
    assert jaccard_similarity(s1, s2) == 0.0

    # Both empty
    assert jaccard_similarity(set(), set()) == 0.0


def test_compute_overlap_matrix() -> None:
    """compute_overlap_matrix should return correct pairwise Jaccard values."""
    programs1 = {
        "p1": ["A", "B", "C"],
        "p2": ["D", "E", "F"],
    }
    programs2 = {
        "q1": ["A", "B", "C"],  # identical to p1
        "q2": ["X", "Y", "Z"],  # disjoint from all
    }

    matrix = compute_overlap_matrix(programs1, programs2)
    assert isinstance(matrix, pd.DataFrame)
    assert matrix.shape == (2, 2)

    # p1 vs q1 should be 1.0
    assert abs(matrix.loc["p1", "q1"] - 1.0) < 1e-10
    # p1 vs q2 should be 0.0
    assert abs(matrix.loc["p1", "q2"] - 0.0) < 1e-10
    # p2 vs q1 should be 0.0
    assert abs(matrix.loc["p2", "q1"] - 0.0) < 1e-10
    # p2 vs q2 should be 0.0
    assert abs(matrix.loc["p2", "q2"] - 0.0) < 1e-10


def test_coverage() -> None:
    """Coverage should measure fraction of reference genes in programs."""
    programs = {
        "p1": ["A", "B", "C"],
        "p2": ["C", "D", "E"],
    }
    reference = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    result = coverage(programs, reference)
    # Programs cover A, B, C, D, E = 5 out of 10
    assert abs(result - 0.5) < 1e-10

    # Empty reference -> 0.0
    assert coverage(programs, []) == 0.0


def test_novelty_score() -> None:
    """Novelty score should measure fraction of genes not in curated sets."""
    programs = {
        "p1": ["A", "B", "C"],
        "p2": ["D", "E", "NEW1"],
    }
    curated = {
        "kegg1": ["A", "B", "C", "D", "E"],
    }

    result = novelty_score(programs, curated)
    # Total edges: 6 (A,B,C,D,E,NEW1)
    # Novel edges: 1 (NEW1)
    assert abs(result - 1.0 / 6.0) < 1e-10

    # If all genes are novel
    programs_novel = {"p1": ["X", "Y", "Z"]}
    assert novelty_score(programs_novel, curated) == 1.0

    # If empty
    assert novelty_score({}, curated) == 0.0


def test_program_redundancy() -> None:
    """Redundancy should measure mean pairwise Jaccard across programs."""
    # Identical programs -> high redundancy
    identical_programs = {
        "p1": ["A", "B", "C"],
        "p2": ["A", "B", "C"],
    }
    assert program_redundancy(identical_programs) == 1.0

    # Disjoint programs -> zero redundancy
    disjoint_programs = {
        "p1": ["A", "B", "C"],
        "p2": ["D", "E", "F"],
    }
    assert program_redundancy(disjoint_programs) == 0.0

    # Single program -> 0.0 by definition
    assert program_redundancy({"p1": ["A", "B"]}) == 0.0

    # Partially overlapping
    partial = {
        "p1": ["A", "B", "C", "D"],
        "p2": ["C", "D", "E", "F"],
        "p3": ["E", "F", "G", "H"],
    }
    result = program_redundancy(partial)
    assert 0.0 < result < 1.0


def test_program_specificity() -> None:
    """Specificity measures fraction of genes unique to each program."""
    # Completely disjoint programs -> specificity = 1.0
    disjoint = {
        "p1": ["A", "B", "C"],
        "p2": ["D", "E", "F"],
    }
    assert program_specificity(disjoint) == 1.0

    # Identical programs -> specificity = 0.0
    identical = {
        "p1": ["A", "B", "C"],
        "p2": ["A", "B", "C"],
    }
    assert program_specificity(identical) == 0.0

    # Partial overlap
    partial = {
        "p1": ["A", "B", "C"],
        "p2": ["C", "D", "E"],
    }
    result = program_specificity(partial)
    assert 0.0 < result < 1.0


def test_inter_program_distance() -> None:
    """Inter-program distance in embedding space."""
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((10, 32))
    gene_names = [f"gene_{i}" for i in range(10)]
    programs = {
        "p1": gene_names[:5],
        "p2": gene_names[5:],
    }
    dist = inter_program_distance(programs, embeddings, gene_names)
    assert 0.0 <= dist <= 2.0


def test_permutation_test_enrichment() -> None:
    """Permutation test should produce valid p-value and z-score."""
    programs = {
        "p1": ["A", "B", "C", "D"],
        "p2": ["E", "F", "G", "H"],
    }
    curated = {
        "kegg1": ["A", "B", "C", "D"],  # Perfect match with p1
    }
    universe = [f"gene_{i}" for i in range(100)] + ["A", "B", "C", "D", "E", "F", "G", "H"]

    result = permutation_test_enrichment(
        programs, curated, universe, n_permutations=100, random_state=42
    )
    assert "observed" in result
    assert "p_value" in result
    assert "z_score" in result
    assert 0.0 <= result["p_value"] <= 1.0
    # With a perfect match, observed should be higher than null
    assert result["observed"] > result["null_mean"]


def test_permutation_test_enrichment_empty_universe() -> None:
    """Permutation test should not crash when gene_universe is empty."""
    programs = {"p1": ["A", "B"], "p2": ["C", "D"]}
    curated = {"c1": ["A", "X"]}

    result = permutation_test_enrichment(
        programs,
        curated,
        gene_universe=[],
        n_permutations=20,
        random_state=42,
    )

    assert "p_value" in result
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["n_permutations"] == 20


def test_effect_size_cohens_d() -> None:
    """Cohen's d effect size computation."""
    group1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    group2 = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
    d = effect_size_cohens_d(group1, group2)
    assert d < 0  # group1 < group2 means negative d
    assert abs(d) > 0.5  # Should be a meaningful effect


def test_paired_rank_biserial() -> None:
    """Paired rank-biserial should ignore ties and capture direction."""
    deltas = np.array([1.0, 2.0, -1.0, 0.0, 0.0], dtype=np.float64)
    # wins=2, losses=1 -> (2-1)/3
    r = paired_rank_biserial(deltas)
    assert abs(r - ((2.0 - 1.0) / 3.0)) < 1e-12

    all_ties = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    assert np.isnan(paired_rank_biserial(all_ties))


def test_paired_sign_test_pvalue() -> None:
    """Paired sign test should return valid p-values and honor alternative."""
    deltas = np.array([1.0, 2.0, 1.0, -1.0, 0.0], dtype=np.float64)

    p_greater = paired_sign_test_pvalue(deltas, alternative="greater")
    p_less = paired_sign_test_pvalue(deltas, alternative="less")
    p_two = paired_sign_test_pvalue(deltas, alternative="two-sided")

    assert 0.0 <= p_greater <= 1.0
    assert 0.0 <= p_less <= 1.0
    assert 0.0 <= p_two <= 1.0
    assert p_greater < p_less

    ties_only = np.array([0.0, 0.0], dtype=np.float64)
    assert np.isnan(paired_sign_test_pvalue(ties_only))


def test_benjamini_hochberg_fdr_handles_nan_inputs() -> None:
    """BH helper should preserve shape and keep NaNs for invalid p-values."""
    pvals = np.array([0.001, 0.01, np.nan, 0.2, np.inf], dtype=np.float64)
    qvals = benjamini_hochberg_fdr(pvals)

    assert qvals.shape == pvals.shape
    assert np.isfinite(qvals[0])
    assert np.isfinite(qvals[1])
    assert np.isnan(qvals[2])
    assert np.isnan(qvals[4])
    assert qvals[0] <= qvals[1] <= 1.0


def test_benjamini_hochberg_fdr_grouped_separates_testing_families() -> None:
    """Grouped BH should be computed independently within each family."""
    df = pd.DataFrame(
        {
            "metric": ["m1", "m1", "m2", "m2"],
            "p_value": [0.001, 0.01, 0.001, 0.9],
        }
    )
    q_global = benjamini_hochberg_fdr_grouped(df, p_value_col="p_value")
    q_grouped = benjamini_hochberg_fdr_grouped(
        df,
        p_value_col="p_value",
        group_cols=["metric"],
    )

    # m1 second test is less penalized when corrected only within m1 family.
    assert float(q_grouped.iloc[1]) < float(q_global.iloc[1])
    assert (q_grouped >= 0.0).all()
    assert (q_grouped <= 1.0).all()


def test_paired_rank_biserial_correlation_sign_and_bounds() -> None:
    """Rank-based paired effect size should follow dominant delta direction."""
    mostly_positive = np.array([2.0, 1.5, 1.0, -0.5], dtype=np.float64)
    mostly_negative = np.array([-2.0, -1.5, -1.0, 0.5], dtype=np.float64)
    zeros_only = np.zeros(5, dtype=np.float64)

    rb_pos = paired_rank_biserial_correlation(mostly_positive)
    rb_neg = paired_rank_biserial_correlation(mostly_negative)
    rb_nan = paired_rank_biserial_correlation(zeros_only)

    assert -1.0 <= rb_pos <= 1.0
    assert -1.0 <= rb_neg <= 1.0
    assert rb_pos > 0.0
    assert rb_neg < 0.0
    assert np.isnan(rb_nan)


def test_bootstrap_metric_resamples_array_inputs() -> None:
    """bootstrap_metric should produce non-zero uncertainty for array inputs."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)

    result = bootstrap_metric(
        metric_fn=lambda values: float(np.mean(values)),
        values=values,
        n_bootstrap=300,
        random_state=42,
    )

    assert result["ci_lower"] <= result["point_estimate"] <= result["ci_upper"]
    assert result["std_error"] > 0.0


def test_bootstrap_metric_zero_bootstraps_returns_degenerate_ci() -> None:
    """n_bootstrap=0 should return a zero-width interval instead of crashing."""
    result = bootstrap_metric(
        metric_fn=lambda values: float(np.mean(values)),
        values=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        n_bootstrap=0,
    )

    assert result["ci_lower"] == result["point_estimate"]
    assert result["ci_upper"] == result["point_estimate"]
    assert result["std_error"] == 0.0


def test_fold_enrichment_with_ci() -> None:
    """Fold enrichment with confidence interval."""
    result = fold_enrichment_with_ci(
        overlap=10, program_size=50, query_size=100, background_size=1000
    )
    assert result["fold_enrichment"] > 1.0  # Should be enriched
    assert result["ci_lower"] <= result["fold_enrichment"]
    assert result["ci_upper"] >= result["fold_enrichment"]


def test_comprehensive_evaluation() -> None:
    """Comprehensive evaluation returns expected metrics."""
    programs = {
        "p1": ["A", "B", "C"],
        "p2": ["D", "E", "F"],
    }
    reference = ["A", "B", "C", "D", "E", "F", "G", "H"]
    curated = {"kegg1": ["A", "B", "C", "D"]}

    results = comprehensive_evaluation(
        programs=programs,
        reference_genes=reference,
        curated=curated,
    )
    assert results["n_programs"] == 2.0
    assert results["coverage"] == 0.75
    assert "novelty" in results
    assert "specificity" in results
    assert "redundancy" in results


def test_compare_methods() -> None:
    """Compare methods should produce a DataFrame."""
    method_programs = {
        "method_A": {"p1": ["A", "B", "C"], "p2": ["D", "E"]},
        "method_B": {"p1": ["A", "B"], "p2": ["C", "D", "E", "F"]},
    }
    reference = ["A", "B", "C", "D", "E", "F"]

    df = compare_methods(method_programs, reference)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "method_A" in df.index
    assert "coverage" in df.columns


def test_program_stability_small_gene_universe() -> None:
    """program_stability should handle fewer than 10 genes."""
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((8, 4))
    gene_names = [f"g{i}" for i in range(8)]

    def _dummy_discovery(
        emb: np.ndarray, names: list[str], **kwargs: object
    ) -> dict[str, list[str]]:
        mid = max(1, len(names) // 2)
        return {"p0": names[:mid], "p1": names[mid:]}

    result = program_stability(
        embeddings,
        gene_names,
        _dummy_discovery,
        n_resamples=5,
        random_state=7,
    )

    assert "mean_jaccard" in result
    assert "gene_stability" in result
    assert len(result["gene_stability"]) == len(gene_names)


def test_program_stability_gene_membership_mismatch_not_counted_stable() -> None:
    """Genes should only count as stable when their matched program is consistent."""
    embeddings = np.eye(6, dtype=np.float64)
    gene_names = [f"g{i}" for i in range(6)]
    calls = {"n": 0}

    def _discovery(
        emb: np.ndarray, names: list[str], **kwargs: object
    ) -> dict[str, list[str]]:
        if calls["n"] == 0:
            calls["n"] += 1
            return {"p0": names[:3], "p1": names[3:]}
        calls["n"] += 1
        return {"p0": [names[0], names[3], names[4]], "p1": [names[1], names[2], names[5]]}

    result = program_stability(
        embeddings,
        gene_names,
        _discovery,
        n_resamples=1,
        random_state=0,
        subsample_fraction=1.0,
    )

    assert result["mean_jaccard"] < 1.0
    assert result["mean_gene_stability"] < 1.0
