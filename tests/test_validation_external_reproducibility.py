"""Tests for external reproducibility utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from npathway.validation.external_reproducibility import (
    compare_external_reproducibility,
    summarize_external_reproducibility,
)


def _write_program_table(path: Path, rows: list[dict[str, object]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_compare_external_reproducibility_with_synthetic_csv() -> None:
    """Pairwise external reproducibility metrics should be computed correctly."""
    baseline_df = pd.DataFrame(
        [
            {
                "program": "P1",
                "genes": "A,B,C",
                "nes": 1.5,
                "fdr": 0.001,
                "claim_supported": True,
            },
            {
                "program": "P2",
                "genes": "D,E,F",
                "nes": -1.2,
                "fdr": 0.020,
                "claim_supported": True,
            },
            {
                "program": "P3",
                "genes": "G,H,I",
                "nes": 2.1,
                "fdr": 0.200,
                "claim_supported": False,
            },
        ]
    )
    replicate_df = pd.DataFrame(
        [
            {
                "program": "Q1",
                "genes": "A,B,C",
                "nes": 0.7,
                "fdr": 0.003,
                "claim_supported": True,
            },
            {
                "program": "Q2",
                "genes": "D,E,X",
                "nes": 1.0,
                "fdr": 0.040,
                "claim_supported": True,
            },
            {
                "program": "Q3",
                "genes": "L,M,N",
                "nes": -0.4,
                "fdr": 0.400,
                "claim_supported": False,
            },
        ]
    )

    result = compare_external_reproducibility(
        baseline_df,
        replicate_df,
        top_k=3,
        jaccard_threshold=0.3,
    )

    assert result["n_top_programs"] == 3
    assert np.isclose(result["top_program_replication_rate"], 2.0 / 3.0)
    assert result["n_claim_supported_baseline"] == 2
    assert np.isclose(result["claim_supported_replication_rate"], 1.0)
    assert result["n_replicated_programs"] == 2
    assert np.isclose(result["direction_concordance_rate"], 0.5)

    jaccards = result["best_match_jaccard_distribution"]
    assert len(jaccards) == 3
    assert np.allclose(jaccards, [1.0, 0.5, 0.0])

    best_match_table = result["best_match_table"]
    assert isinstance(best_match_table, pd.DataFrame)
    assert list(best_match_table["baseline_program"]) == ["P1", "P2", "P3"]
    assert list(best_match_table["replicate_program"].iloc[:2]) == ["Q1", "Q2"]
    assert float(best_match_table.iloc[2]["best_jaccard"]) == 0.0
    assert bool(best_match_table.iloc[2]["replicated"]) is False


def test_compare_external_reproducibility_strict_baseline_fdr() -> None:
    """Strict baseline FDR should tighten claim-supported replication denominator."""
    baseline_df = pd.DataFrame(
        [
            {
                "program": "P1",
                "genes": "A,B,C",
                "nes": 1.5,
                "fdr": 0.001,
                "claim_supported": True,
            },
            {
                "program": "P2",
                "genes": "D,E,F",
                "nes": -1.2,
                "fdr": 0.020,
                "claim_supported": True,
            },
        ]
    )
    replicate_df = pd.DataFrame(
        [
            {
                "program": "Q1",
                "genes": "A,B,C",
                "nes": 0.6,
                "fdr": 0.004,
                "claim_supported": True,
            },
            {
                "program": "Q2",
                "genes": "D,E,X",
                "nes": -0.9,
                "fdr": 0.030,
                "claim_supported": True,
            },
        ]
    )

    strict_result = compare_external_reproducibility(
        baseline_df,
        replicate_df,
        top_k=2,
        jaccard_threshold=0.3,
        strict_baseline_fdr=0.01,
    )
    assert strict_result["n_claim_supported_baseline"] == 1
    assert np.isclose(strict_result["claim_supported_replication_rate"], 1.0)

    ultra_strict_result = compare_external_reproducibility(
        baseline_df,
        replicate_df,
        top_k=2,
        jaccard_threshold=0.3,
        strict_baseline_fdr=1e-6,
    )
    assert ultra_strict_result["n_claim_supported_baseline"] == 0
    assert np.isclose(ultra_strict_result["claim_supported_replication_rate"], 0.0)


def test_summarize_external_reproducibility_with_csv_fixtures(tmp_path: Path) -> None:
    """Multi-cohort summary should aggregate directed pairwise comparisons."""
    cohort_a = _write_program_table(
        tmp_path / "cohort_a.csv",
        [
            {"program": "A1", "genes": "A,B,C", "nes": 1.2, "fdr": 0.001, "claim_supported": True},
            {"program": "A2", "genes": "D,E,F", "nes": -1.1, "fdr": 0.020, "claim_supported": True},
        ],
    )
    cohort_b = _write_program_table(
        tmp_path / "cohort_b.csv",
        [
            {"program": "B1", "genes": "A,B,C", "nes": 1.0, "fdr": 0.003, "claim_supported": True},
            {"program": "B2", "genes": "D,E,F", "nes": -0.9, "fdr": 0.040, "claim_supported": True},
        ],
    )
    cohort_c = _write_program_table(
        tmp_path / "cohort_c.csv",
        [
            {"program": "C1", "genes": "X,Y,Z", "nes": 0.8, "fdr": 0.005, "claim_supported": True},
            {"program": "C2", "genes": "L,M,N", "nes": -0.7, "fdr": 0.030, "claim_supported": True},
        ],
    )

    summary_out = summarize_external_reproducibility(
        {"A": cohort_a, "B": cohort_b, "C": cohort_c},
        top_k=2,
        jaccard_threshold=0.4,
    )

    summary = summary_out["summary"]
    pairwise = summary_out["pairwise_metrics"]
    best_tables = summary_out["best_match_tables"]
    all_jaccards = summary_out["overall_best_match_jaccard_distribution"]

    assert summary["n_cohorts"] == 3
    assert summary["n_pairwise_comparisons"] == 6
    assert len(pairwise) == 6
    assert len(best_tables) == 6
    assert len(all_jaccards) == 12  # 6 directed pairs x top_k=2

    row_ab = pairwise[(pairwise["baseline"] == "A") & (pairwise["replicate"] == "B")].iloc[0]
    row_ba = pairwise[(pairwise["baseline"] == "B") & (pairwise["replicate"] == "A")].iloc[0]
    assert np.isclose(float(row_ab["top_program_replication_rate"]), 1.0)
    assert np.isclose(float(row_ba["top_program_replication_rate"]), 1.0)

    # Only A<->B pairs replicate; 2 successful directed pairs out of 6 total.
    assert np.isclose(summary["mean_top_program_replication_rate"], 1.0 / 3.0)


def test_compare_external_reproducibility_empty_baseline_returns_empty_table() -> None:
    """An empty baseline table should not crash or drop expected output columns."""
    baseline_df = pd.DataFrame(
        columns=["program", "genes", "nes", "fdr", "claim_supported"]
    )
    replicate_df = pd.DataFrame(
        [
            {
                "program": "Q1",
                "genes": "A,B,C",
                "nes": 0.7,
                "fdr": 0.003,
                "claim_supported": True,
            }
        ]
    )

    result = compare_external_reproducibility(baseline_df, replicate_df, top_k=5)

    assert result["n_top_programs"] == 0
    assert result["best_match_table"].empty
    assert "baseline_claim_supported" in result["best_match_table"].columns
