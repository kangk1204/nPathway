"""Tests for baseline-vs-variant stability/sensitivity validation utilities."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from npathway.validation.stability_sensitivity import (
    summarize_bootstrap_stability,
    summarize_hyperparameter_sensitivity,
)


def _write_program_long(
    results_dir: Path,
    program_to_genes: dict[str, list[str]],
    *,
    filename: str,
) -> None:
    rows: list[dict[str, str]] = []
    for program, genes in program_to_genes.items():
        for gene in genes:
            rows.append({"program": program, "gene": gene})
    pd.DataFrame(rows).to_csv(results_dir / filename, index=False)


def _write_gsea_claims(results_dir: Path, claim_supported: list[bool]) -> None:
    n_rows = len(claim_supported)
    pd.DataFrame(
        {
            "program": [f"program_{i}" for i in range(n_rows)],
            "fdr": [0.01] * n_rows,
            "claim_supported": claim_supported,
        }
    ).to_csv(results_dir / "enrichment_gsea_with_claim_gates.csv", index=False)


def _write_gsea_gates(
    results_dir: Path,
    gate_fdr: list[bool],
    gate_effect: list[bool | float],
) -> None:
    n_rows = len(gate_fdr)
    pd.DataFrame(
        {
            "program": [f"program_{i}" for i in range(n_rows)],
            "fdr": [0.01] * n_rows,
            "gate_fdr": gate_fdr,
            "gate_effect": gate_effect,
        }
    ).to_csv(results_dir / "enrichment_gsea_with_claim_gates.csv", index=False)


def _write_manifest(results_dir: Path, config: dict[str, object]) -> None:
    payload = {"config": config}
    (results_dir / "run_manifest.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_bootstrap_summary_includes_jaccard_claim_deltas_and_membership_fallback(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    bootstrap_a = tmp_path / "bootstrap_001"
    bootstrap_b = tmp_path / "bootstrap_002"
    baseline.mkdir()
    bootstrap_a.mkdir()
    bootstrap_b.mkdir()

    baseline_programs = {
        "base_p1": ["A", "B", "C"],
        "base_p2": ["D", "E"],
    }
    variant_a_programs = {
        "va_p1": ["A", "B"],
        "va_p2": ["D", "E", "F"],
    }
    variant_b_programs = {
        "vb_p1": ["A", "C"],
        "vb_p2": ["D", "G"],
    }

    _write_program_long(
        baseline,
        baseline_programs,
        filename="program_gene_membership_long.csv",
    )
    _write_program_long(
        bootstrap_a,
        variant_a_programs,
        filename="program_gene_membership_long.csv",
    )
    _write_program_long(
        bootstrap_b,
        variant_b_programs,
        filename="dynamic_programs_long.csv",
    )

    _write_gsea_claims(baseline, [True, False, True])
    _write_gsea_claims(bootstrap_a, [True, True])
    _write_gsea_gates(bootstrap_b, [True, True, False], [True, np.nan, True])

    tables = summarize_bootstrap_stability(
        baseline_results_dir=baseline,
        bootstrap_results_dirs=[bootstrap_a, bootstrap_b],
    )
    summary = tables.summary
    pairwise = tables.pairwise_jaccard_long

    assert len(summary) == 2
    assert len(pairwise) == 8
    assert set(pairwise["analysis_type"].tolist()) == {"bootstrap"}

    row_a = summary.loc[summary["variant_label"] == "bootstrap_001"].iloc[0]
    assert row_a["baseline_membership_file"] == "program_gene_membership_long.csv"
    assert row_a["variant_membership_file"] == "program_gene_membership_long.csv"
    assert np.isclose(row_a["mean_best_jaccard_baseline_to_variant"], 2.0 / 3.0)
    assert np.isclose(row_a["mean_best_jaccard_variant_to_baseline"], 2.0 / 3.0)
    assert row_a["baseline_claim_supported_count"] == 2.0
    assert row_a["variant_claim_supported_count"] == 2.0
    assert row_a["claim_supported_count_delta"] == 0.0

    row_b = summary.loc[summary["variant_label"] == "bootstrap_002"].iloc[0]
    assert row_b["variant_membership_file"] == "dynamic_programs_long.csv"
    assert np.isclose(row_b["mean_best_jaccard_baseline_to_variant"], 0.5)
    assert row_b["variant_claim_supported_count"] == 2.0


def test_bootstrap_summary_handles_missing_files_defensively(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    missing_variant = tmp_path / "bootstrap_missing"
    baseline.mkdir()
    missing_variant.mkdir()

    _write_program_long(
        baseline,
        {"base_p1": ["A", "B", "C"]},
        filename="program_gene_membership_long.csv",
    )
    _write_gsea_claims(baseline, [True, False])

    tables = summarize_bootstrap_stability(
        baseline_results_dir=baseline,
        bootstrap_results_dirs=[missing_variant],
    )
    row = tables.summary.iloc[0]

    assert row["n_variant_programs"] == 0
    assert row["n_pairwise_program_pairs"] == 0
    assert np.isnan(row["mean_pairwise_jaccard"])
    assert np.isnan(row["variant_claim_supported_count"])
    assert isinstance(row["notes"], str)
    assert "membership" in row["notes"]
    assert "claims" in row["notes"]



def test_hyperparameter_summary_explodes_manifest_parameter_changes(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    variant_a = tmp_path / "variant_n_components"
    variant_b = tmp_path / "variant_resolution_and_method"
    baseline.mkdir()
    variant_a.mkdir()
    variant_b.mkdir()

    baseline_programs = {
        "base_p1": ["A", "B", "C"],
        "base_p2": ["D", "E"],
    }
    variant_a_programs = {
        "va_p1": ["A", "B"],
        "va_p2": ["D", "E", "F"],
    }
    variant_b_programs = {
        "vb_p1": ["A", "C"],
        "vb_p2": ["D", "G"],
    }

    _write_program_long(
        baseline,
        baseline_programs,
        filename="program_gene_membership_long.csv",
    )
    _write_program_long(
        variant_a,
        variant_a_programs,
        filename="program_gene_membership_long.csv",
    )
    _write_program_long(
        variant_b,
        variant_b_programs,
        filename="program_gene_membership_long.csv",
    )

    _write_gsea_claims(baseline, [True, False, True])
    _write_gsea_claims(variant_a, [True, True])
    _write_gsea_claims(variant_b, [True, False, False])

    _write_manifest(
        baseline,
        {
            "matrix_path": "baseline_matrix.csv",
            "group_a": "case",
            "group_b": "control",
            "n_components": 30,
            "resolution": 1.0,
            "discovery_method": "kmeans",
        },
    )
    _write_manifest(
        variant_a,
        {
            "matrix_path": "variant_a_matrix.csv",
            "group_a": "case",
            "group_b": "control",
            "n_components": 50,
            "resolution": 1.0,
            "discovery_method": "kmeans",
        },
    )
    _write_manifest(
        variant_b,
        {
            "matrix_path": "variant_b_matrix.csv",
            "group_a": "case",
            "group_b": "control",
            "n_components": 30,
            "resolution": 2.0,
            "discovery_method": "leiden",
        },
    )

    tables = summarize_hyperparameter_sensitivity(
        baseline_results_dir=baseline,
        variant_results_dirs=[variant_a, variant_b],
    )

    summary = tables.summary
    assert set(summary["analysis_type"].tolist()) == {"hyperparameter"}

    row_a = summary.loc[summary["variant_label"] == "variant_n_components"]
    assert len(row_a) == 1
    assert row_a.iloc[0]["parameter"] == "n_components"
    assert row_a.iloc[0]["baseline_parameter_value"] == "30"
    assert row_a.iloc[0]["variant_parameter_value"] == "50"
    assert row_a.iloc[0]["n_changed_parameters"] == 1

    row_b = summary.loc[summary["variant_label"] == "variant_resolution_and_method"]
    assert len(row_b) == 2
    assert set(row_b["parameter"].tolist()) == {"resolution", "discovery_method"}
    assert set(row_b["n_changed_parameters"].tolist()) == {2}

    assert len(tables.pairwise_jaccard_long) == 8
    assert set(tables.pairwise_jaccard_long["analysis_type"].tolist()) == {"hyperparameter"}
