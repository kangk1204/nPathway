"""Tests for dynamic dashboard package generation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from npathway.reporting import DashboardConfig, build_dynamic_dashboard_package
from npathway.reporting.dynamic_dashboard import _prepare_overlap_heatmap_long


def test_dynamic_dashboard_builds_assets(tmp_path) -> None:
    """Dashboard builder should create HTML, figures, and tables."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    de = pd.DataFrame(
        {
            "gene": [f"G{i}" for i in range(1, 21)],
            "logfc_a_minus_b": np.linspace(-2, 2, 20),
            "p_value": np.geomspace(1e-6, 0.8, 20),
            "fdr": np.geomspace(1e-5, 0.9, 20),
        }
    )
    de.to_csv(results_dir / "de_results.csv", index=False)

    gsea = pd.DataFrame(
        {
            "program": [f"program_{i}" for i in range(1, 11)],
            "nes": np.linspace(2.5, -1.2, 10),
            "p_value": np.geomspace(1e-4, 0.2, 10),
            "fdr": np.geomspace(1e-3, 0.3, 10),
            "gate_fdr": [True] * 6 + [False] * 4,
            "gate_effect": [True, True, True, False, True, False, True, False, False, False],
            "gate_program_size": [True] * 10,
            "gate_stability": [True] * 10,
            "claim_supported": [True, True, True, False, True, False, True, False, False, False],
        }
    )
    gsea.to_csv(results_dir / "enrichment_gsea_with_claim_gates.csv", index=False)

    sizes = pd.DataFrame(
        {
            "program": [f"program_{i}" for i in range(1, 11)],
            "n_genes": [80, 72, 65, 61, 59, 53, 49, 45, 41, 39],
        }
    )
    sizes.to_csv(results_dir / "dynamic_program_sizes.csv", index=False)

    context_rows = []
    for i in range(1, 11):
        for j in range(1, 4):
            context_rows.append(
                {
                    "program": f"program_{i}",
                    "gene": f"G{i}_{j}",
                    "base_membership": float(0.6 + i * 0.02),
                    "prob_A": float(0.2 + j * 0.2),
                    "prob_B": float(0.8 - j * 0.2),
                    "contextual_score_A": float(0.2 + i * 0.01),
                    "contextual_score_B": float(0.15 + i * 0.01),
                    "context_shift": float((j - 2) * 0.3 + i * 0.01),
                }
            )
    context = pd.DataFrame(context_rows)
    context.to_csv(results_dir / "contextual_membership_scores.csv", index=False)

    fisher = pd.DataFrame(
        {
            "program": [f"program_{i}" for i in range(1, 11)],
            "p_value": np.geomspace(1e-4, 0.4, 10),
            "fdr": np.geomspace(1e-3, 0.5, 10),
        }
    )
    fisher.to_csv(results_dir / "enrichment_fisher.csv", index=False)

    outdir = tmp_path / "dashboard"
    cfg = DashboardConfig(
        results_dir=str(results_dir),
        output_dir=str(outdir),
        top_k=8,
        include_pdf=False,
    )
    artifacts = build_dynamic_dashboard_package(cfg)

    assert Path(artifacts.html_path).exists()
    assert Path(artifacts.figure_dir).exists()
    assert Path(artifacts.table_dir).exists()
    assert Path(artifacts.summary_table_path).exists()

    assert (outdir / "figures" / "figure_1_volcano.png").exists()
    assert (outdir / "figures" / "figure_2_program_sizes.png").exists()
    assert (outdir / "figures" / "figure_3_claim_gates.png").exists()
    assert (outdir / "figures" / "figure_4_context_shift.png").exists()

    assert (outdir / "tables" / "top_enriched_programs.csv").exists()
    assert (outdir / "tables" / "top_context_shift_genes.csv").exists()
    assert (outdir / "tables" / "top_context_evidence_genes.csv").exists()
    assert (outdir / "tables" / "claim_gate_summary.csv").exists()

    html = (outdir / "index.html").read_text(encoding="utf-8")
    assert "Context evidence (Recommended)" in html
    assert "Program Spotlight" in html
    assert "Download Center" in html
    assert 'id="program-search"' in html


def test_dynamic_dashboard_derives_context_shift_from_prob_columns(tmp_path) -> None:
    """Legacy context files without context_shift should still render."""
    results_dir = tmp_path / "results_prob_only"
    results_dir.mkdir(parents=True, exist_ok=True)

    de = pd.DataFrame(
        {
            "gene": ["G1", "G2", "G3", "G4"],
            "logfc_a_minus_b": [0.8, -0.4, 0.1, -0.2],
            "p_value": [0.01, 0.02, 0.2, 0.4],
            "fdr": [0.02, 0.03, 0.3, 0.5],
        }
    )
    de.to_csv(results_dir / "de_results.csv", index=False)

    gsea = pd.DataFrame(
        {
            "program": ["program_1", "program_2"],
            "nes": [1.8, -1.2],
            "p_value": [0.001, 0.02],
            "fdr": ["0.01", "0.20"],
            "gate_fdr": [True, False],
        }
    )
    gsea.to_csv(results_dir / "enrichment_gsea_with_claim_gates.csv", index=False)

    sizes = pd.DataFrame({"program": ["program_1", "program_2"], "n_genes": [30, 24]})
    sizes.to_csv(results_dir / "dynamic_program_sizes.csv", index=False)

    context = pd.DataFrame(
        {
            "program": ["program_1", "program_1", "program_2"],
            "gene": ["G1", "G2", "G3"],
            "base_membership": [0.9, 0.7, 0.5],
            "prob_A": [0.85, 0.3, 0.55],
            "prob_B": [0.15, 0.7, 0.45],
            "p_value": [0.01, 0.05, 0.2],
        }
    )
    context.to_csv(results_dir / "contextual_membership_scores.csv", index=False)

    outdir = tmp_path / "dashboard_prob_only"
    artifacts = build_dynamic_dashboard_package(
        DashboardConfig(
            results_dir=str(results_dir),
            output_dir=str(outdir),
            top_k=10,
            include_pdf=False,
        )
    )
    assert Path(artifacts.html_path).exists()

    top_context = pd.read_csv(outdir / "tables" / "top_context_evidence_genes.csv")
    assert {"context_shift", "signed_significance", "context_evidence"}.issubset(top_context.columns)
    expected_shift = top_context["prob_A"].to_numpy(dtype=float) - top_context["prob_B"].to_numpy(dtype=float)
    assert np.allclose(top_context["context_shift"].to_numpy(dtype=float), expected_shift)
    expected_evidence = expected_shift * (
        -np.log10(np.clip(top_context["p_value"].to_numpy(dtype=float), 1e-300, 1.0))
    )
    assert np.allclose(top_context["context_evidence"].to_numpy(dtype=float), expected_evidence)


def test_dynamic_dashboard_handles_empty_context_rows(tmp_path) -> None:
    """Empty contextual_membership_scores with headers should not crash."""
    results_dir = tmp_path / "results_empty_context"
    results_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "gene": ["G1", "G2"],
            "logfc_a_minus_b": [0.4, -0.2],
            "p_value": [0.02, 0.3],
            "fdr": [0.04, 0.6],
        }
    ).to_csv(results_dir / "de_results.csv", index=False)
    pd.DataFrame(
        {
            "program": ["program_1"],
            "nes": [1.1],
            "p_value": [0.02],
            "fdr": [0.05],
            "gate_fdr": [True],
        }
    ).to_csv(results_dir / "enrichment_gsea_with_claim_gates.csv", index=False)
    pd.DataFrame({"program": ["program_1"], "n_genes": [18]}).to_csv(
        results_dir / "dynamic_program_sizes.csv",
        index=False,
    )
    pd.DataFrame(columns=["program", "gene", "base_membership", "context_shift"]).to_csv(
        results_dir / "contextual_membership_scores.csv",
        index=False,
    )

    outdir = tmp_path / "dashboard_empty_context"
    build_dynamic_dashboard_package(
        DashboardConfig(
            results_dir=str(results_dir),
            output_dir=str(outdir),
            top_k=5,
            include_pdf=False,
        )
    )
    assert (outdir / "figures" / "figure_4_context_shift.png").exists()
    context_table = pd.read_csv(outdir / "tables" / "top_context_evidence_genes.csv")
    assert context_table.empty
    assert {"context_shift", "signed_significance", "context_evidence", "abs_context_evidence"}.issubset(
        context_table.columns
    )


def test_prepare_overlap_heatmap_long_derives_jaccard_when_missing() -> None:
    """Heatmap prep should recover jaccard from overlap counts for older files."""
    overlap = pd.DataFrame(
        {
            "program": ["program_1"],
            "reference_name": ["RefA"],
            "overlap_n": [3],
            "program_n": [6],
            "reference_n": [5],
        }
    )
    gsea = pd.DataFrame({"program": ["program_1"], "fdr": [0.01]})

    out = _prepare_overlap_heatmap_long(overlap_df=overlap, gsea_df=gsea, top_k=5)
    assert len(out) == 1
    assert "jaccard" in out.columns
    assert np.isclose(out.loc[0, "jaccard"], 3 / (6 + 5 - 3))
