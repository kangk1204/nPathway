"""Tests for perturbation robustness validation utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from npathway.validation.robustness import analyze_perturbation_robustness


def _write_synthetic_run(
    outdir: Path,
    programs: dict[str, list[str]],
    claim_supported: dict[str, bool],
    context_metric_values: dict[tuple[str, str], float],
    metric_col: str,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "program": list(programs.keys()),
            "genes": [",".join(genes) for genes in programs.values()],
        }
    ).to_csv(outdir / "program_gene_lists.csv", index=False)

    pd.DataFrame(
        {
            "program": list(programs.keys()),
            "claim_supported": [
                bool(claim_supported.get(program, False))
                for program in programs
            ],
        }
    ).to_csv(outdir / "enrichment_gsea_with_claim_gates.csv", index=False)

    context_rows = []
    for (program, gene), value in context_metric_values.items():
        context_rows.append(
            {"program": program, "gene": gene, metric_col: float(value)}
        )
    pd.DataFrame(context_rows).to_csv(
        outdir / "contextual_membership_scores.csv", index=False
    )


def test_analyze_perturbation_robustness_summary_metrics(tmp_path) -> None:
    """Compute expected Jaccard/claim/context metrics from synthetic runs."""
    baseline_dir = tmp_path / "baseline"
    perturbation_dir = tmp_path / "batch_shift"

    _write_synthetic_run(
        outdir=baseline_dir,
        programs={"p1": ["A", "B", "C"], "p2": ["D", "E"]},
        claim_supported={"p1": True, "p2": True},
        context_metric_values={
            ("p1", "A"): 1.0,
            ("p1", "B"): 2.0,
            ("p1", "C"): 3.0,
            ("p2", "D"): 4.0,
            ("p2", "E"): 5.0,
        },
        metric_col="context_evidence",
    )
    _write_synthetic_run(
        outdir=perturbation_dir,
        programs={"q1": ["A", "B", "C"], "q2": ["D", "F"]},
        claim_supported={"q1": True, "q2": False},
        context_metric_values={
            ("q1", "A"): 2.0,
            ("q1", "B"): 4.0,
            ("q1", "C"): 6.0,
            ("q2", "D"): 8.0,
            ("q2", "F"): 1.0,
        },
        metric_col="context_evidence",
    )

    tables = analyze_perturbation_robustness(
        {
            "baseline": baseline_dir,
            "batch_shift": perturbation_dir,
        },
        baseline_label="baseline",
    )

    summary = tables["summary"]
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["perturbation_label"] == "batch_shift"
    assert np.isclose(row["mean_best_match_program_jaccard"], 2.0 / 3.0)
    assert np.isclose(row["claim_supported_overlap_jaccard"], 0.5)
    assert np.isclose(row["claim_supported_retention"], 0.5)
    assert np.isclose(row["context_metric_stability_correlation"], 1.0)
    assert int(row["n_context_pairs"]) == 4

    program_matches = tables["program_matches"]
    assert set(program_matches["baseline_program"]) == {"p1", "p2"}
    assert set(program_matches["perturbation_program"]) == {"q1", "q2"}
    assert int(program_matches["claim_supported_retained"].sum()) == 1

    context_pairs = tables["context_pairs"]
    assert set(context_pairs["gene"]) == {"A", "B", "C", "D"}
    assert set(context_pairs["baseline_context_metric_name"]) == {"context_evidence"}
    assert set(context_pairs["perturbation_context_metric_name"]) == {
        "context_evidence"
    }


def test_context_metric_falls_back_to_context_shift(tmp_path) -> None:
    """If context_evidence is absent, analysis should use context_shift."""
    baseline_dir = tmp_path / "baseline"
    perturbation_dir = tmp_path / "composition"

    _write_synthetic_run(
        outdir=baseline_dir,
        programs={"p1": ["A", "B"]},
        claim_supported={"p1": True},
        context_metric_values={("p1", "A"): 1.0, ("p1", "B"): 2.0},
        metric_col="context_evidence",
    )
    _write_synthetic_run(
        outdir=perturbation_dir,
        programs={"q1": ["A", "B"]},
        claim_supported={"q1": True},
        context_metric_values={("q1", "A"): 10.0, ("q1", "B"): 20.0},
        metric_col="context_shift",
    )

    tables = analyze_perturbation_robustness(
        {"baseline": baseline_dir, "composition": perturbation_dir},
        baseline_label="baseline",
    )

    summary = tables["summary"]
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["baseline_context_metric_name"] == "context_evidence"
    assert row["perturbation_context_metric_name"] == "context_shift"
    assert np.isclose(row["context_metric_stability_correlation"], 1.0)

    context_pairs = tables["context_pairs"]
    assert set(context_pairs["baseline_context_metric_name"]) == {"context_evidence"}
    assert set(context_pairs["perturbation_context_metric_name"]) == {
        "context_shift"
    }


def test_raises_when_baseline_label_is_missing(tmp_path) -> None:
    """Baseline label should be explicitly present in the input mapping."""
    run_dir = tmp_path / "run_only"
    _write_synthetic_run(
        outdir=run_dir,
        programs={"p1": ["A"]},
        claim_supported={"p1": False},
        context_metric_values={("p1", "A"): 0.1},
        metric_col="context_shift",
    )

    with pytest.raises(ValueError, match="baseline_label"):
        analyze_perturbation_robustness(
            {"run_only": run_dir},
            baseline_label="baseline",
        )
