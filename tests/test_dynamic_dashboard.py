"""Tests for dynamic dashboard package generation."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from npathway.reporting import DashboardConfig, build_dynamic_dashboard_package
from npathway.reporting.dynamic_dashboard import (
    DashboardPublishingAgent,
    ResultIngestionAgent,
    _prepare_reference_family_rows,
    _prepare_multi_pathway_rows,
    _prepare_overlap_heatmap_long,
    _prepare_reference_source_rows,
)


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

    ranked = pd.DataFrame(
        {
            "gene": [
                "APP", "APOE", "TREM2", "TYROBP", "PLCG2", "INPP5D", "MAPT", "PSEN1",
                "STAT3", "RELA", "PTGS2", "G1", "G2", "G3", "G4", "G5",
            ],
            "score": [5.0, 4.8, 4.6, 4.4, 4.2, 4.0, 3.2, 3.0, 2.5, 2.2, 2.0, -0.5, -0.8, -1.1, -1.4, -1.7],
        }
    )
    ranked.to_csv(results_dir / "ranked_genes_for_gsea.csv", index=False)

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

    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    curated_panel = pd.DataFrame(
        {
            "program": [
                "WP_ALZHEIMERS_DISEASE",
                "KEGG_ALZHEIMERS_DISEASE",
                "MICROGLIAL_ACTIVATION",
                "INFLAMMATORY_RESPONSE",
            ],
            "nes": [1.8, 1.5, 2.1, 1.9],
            "fdr": [0.02, 0.05, 0.01, 0.03],
            "n_hits": [5, 4, 6, 5],
        }
    )
    curated_panel.to_csv(comparison_dir / "curated_panel_gsea.csv", index=False)
    (comparison_dir / "curated_panel_gene_sets.gmt").write_text(
        "WP_ALZHEIMERS_DISEASE\tNA\tAPP\tAPOE\tMAPT\tPSEN1\tTREM2\n"
        "KEGG_ALZHEIMERS_DISEASE\tNA\tAPP\tAPOE\tMAPT\tPSEN1\n"
        "MICROGLIAL_ACTIVATION\tNA\tTREM2\tTYROBP\tPLCG2\tINPP5D\tAPOE\tAPP\n"
        "INFLAMMATORY_RESPONSE\tNA\tSTAT3\tRELA\tPTGS2\tTYROBP\tAPP\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "program": [
                "program_1",
                "program_1",
                "program_2",
                "program_3",
                "program_3",
            ],
            "reference_name": [
                "WP::ALZHEIMERS_DISEASE",
                "REACTOME::MICROGLIAL_SIGNALING",
                "HALLMARK::TNFA_SIGNALING_VIA_NFKB",
                "PATHWAYCOMMONS::COMPLEMENT_CASCADE",
                "KEGG::ALZHEIMERS_DISEASE",
            ],
            "jaccard": [0.24, 0.19, 0.21, 0.16, 0.14],
            "overlap_n": [8, 6, 7, 5, 4],
            "program_n": [20, 20, 18, 16, 16],
            "reference_n": [34, 28, 45, 23, 31],
        }
    ).to_csv(results_dir / "program_reference_overlap_long.csv", index=False)

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
    assert (outdir / "summary.md").exists()

    assert (outdir / "figures" / "figure_1_volcano.png").exists()
    assert (outdir / "figures" / "figure_2_program_sizes.png").exists()
    assert (outdir / "figures" / "figure_3_claim_gates.png").exists()
    assert (outdir / "figures" / "figure_4_context_shift.png").exists()
    assert (outdir / "figures" / "figure_5_multi_pathway.png").exists()
    assert (outdir / "figures" / "figure_6_multi_pathway_enrichment_curves.png").exists()
    assert (outdir / "figures" / "figure_7_reference_ranking_calibration.png").exists()

    assert (outdir / "tables" / "top_enriched_programs.csv").exists()
    assert (outdir / "tables" / "top_context_shift_genes.csv").exists()
    assert (outdir / "tables" / "top_context_evidence_genes.csv").exists()
    assert (outdir / "tables" / "claim_gate_summary.csv").exists()
    assert (outdir / "tables" / "curated_panel_gsea.csv").exists()
    assert (outdir / "tables" / "curated_panel_gene_sets.gmt").exists()
    assert (outdir / "tables" / "reference_source_hits.csv").exists()
    assert (outdir / "tables" / "reference_family_hits.csv").exists()
    assert (outdir / "tables" / "reference_ranking_calibration.csv").exists()

    html = (outdir / "index.html").read_text(encoding="utf-8")
    summary_md = (outdir / "summary.md").read_text(encoding="utf-8")
    assert "How to Read This Page" in html
    assert "Context evidence (Recommended)" in html
    assert "Program Spotlight" in html
    assert "Download Center" in html
    assert "Interactive Tables" in html
    assert "Reference Layers" in html
    assert "Collapsed Reference Families" in html
    assert "Reference Ranking Calibration" in html
    assert "Priority score" in html
    assert "Interpretation score" in html
    assert "Program Summary" in html
    assert "Context Drivers" in html
    assert "Core Genes" in html
    assert "Reactome" in html
    assert "WikiPathways" in html
    assert "Core genes in selected program" in html
    assert "Selected Program: Multi-Pathway Enrichment Curves" in html
    assert "Customize layout" in html
    assert "Custom layout saved" in html
    assert 'id="layout-toggle"' in html
    assert 'id="layout-reset"' in html
    assert 'data-card-id="multi-pathway"' in html
    assert 'data-card-id="table-hub"' in html
    assert "Study Summary" in html
    assert "Program Interpretation" in html
    assert "Reference Evidence" in html
    assert "Downloads / Exports" in html
    assert "Search rows" in html
    assert "Download CSV" in html
    assert "Open this file directly in your browser" in html
    assert 'id="program-search"' in html
    assert 'id="headline-summary-table"' in html
    assert 'id="top-enriched-table"' in html
    assert '"multi_pathway_curves"' in html
    assert '"reference_source_hits"' in html
    assert '"reference_family_hits"' in html
    assert '"reference_ranking_calibration"' in html
    assert '"MICROGLIAL_ACTIVATION"' in html
    assert "scheduleResponsiveRerender" in html
    assert "renderVolcanoPlot();" in html
    assert "renderReferenceLayers();" in html
    assert html.index("Interactive Tables") < html.index("Context evidence by gene")
    assert "Treat disease-prioritized family ranking as an interpretation heuristic" in summary_md
    assert "reference_ranking_calibration.csv" in summary_md

    if shutil.which("node"):
        script = html.split("<script>", 1)[1].rsplit("</script>", 1)[0]
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as handle:
            handle.write(script)
            temp_js = Path(handle.name)
        try:
            syntax = subprocess.run(
                ["node", "--check", str(temp_js)],
                capture_output=True,
                text=True,
                check=False,
            )
            assert syntax.returncode == 0, syntax.stderr
        finally:
            temp_js.unlink(missing_ok=True)


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
    assert "reference_display" in out.columns


def test_prepare_multi_pathway_rows_keeps_top_hits_per_program() -> None:
    """Selected-program multi-pathway view should keep the strongest curated hits."""
    overlap = pd.DataFrame(
        {
            "program": ["program_1", "program_1", "program_1", "program_2"],
            "reference_name": [
                "GO_BP::CELL_CYCLE",
                "GO_BP::MITOTIC_CELL_CYCLE",
                "KEGG::P53_SIGNALING_PATHWAY",
                "GO_BP::IMMUNE_RESPONSE",
            ],
            "overlap_n": [8, 12, 4, 6],
            "program_n": [24, 24, 24, 18],
            "reference_n": [50, 55, 30, 40],
        }
    )
    gsea = pd.DataFrame({"program": ["program_1", "program_2"], "fdr": [0.01, 0.20]})

    out = _prepare_multi_pathway_rows(overlap_df=overlap, gsea_df=gsea, top_k_programs=1, top_n_refs=2)

    assert len(out) == 2
    assert set(out["program"]) == {"program_1"}
    assert "reference_display" in out.columns
    assert "novel_gene_estimate" in out.columns
    assert out.iloc[0]["reference_display"].startswith(("GO BP:", "KEGG:"))


def test_prepare_reference_source_rows_groups_hits_by_source() -> None:
    """Reference-source helper should preserve layer identity for dashboard tabs."""
    overlap = pd.DataFrame(
        {
            "program": ["program_1", "program_1", "program_2", "program_2"],
            "reference_name": [
                "WP::ALZHEIMERS_DISEASE",
                "REACTOME::MICROGLIAL_SIGNALING",
                "PATHWAYCOMMONS::COMPLEMENT_CASCADE",
                "HALLMARK::INTERFERON_ALPHA_RESPONSE",
            ],
            "overlap_n": [8, 6, 5, 4],
            "program_n": [24, 24, 18, 18],
            "reference_n": [50, 55, 30, 40],
            "jaccard": [0.17, 0.15, 0.12, 0.10],
        }
    )
    gsea = pd.DataFrame({"program": ["program_1", "program_2"], "fdr": [0.01, 0.03]})

    out = _prepare_reference_source_rows(
        overlap_df=overlap,
        gsea_df=gsea,
        top_k_programs=5,
        top_n_refs_per_source=5,
    )

    assert not out.empty
    assert {"WikiPathways", "Reactome", "Pathway Commons", "Hallmark"} <= set(
        out["source_display"].astype(str)
    )


def test_prepare_reference_family_rows_collapses_cross_source_duplicates() -> None:
    """Reference families should conservatively merge near-duplicate labels across sources."""
    overlap = pd.DataFrame(
        {
            "program": ["program_1", "program_1", "program_2"],
            "reference_name": [
                "WP::ALZHEIMERS_DISEASE",
                "KEGG::ALZHEIMERS_DISEASE",
                "REACTOME::INNATE_IMMUNE_SYSTEM",
            ],
            "overlap_n": [8, 7, 6],
            "program_n": [24, 24, 18],
            "reference_n": [50, 55, 30],
            "jaccard": [0.17, 0.15, 0.12],
        }
    )
    gsea = pd.DataFrame({"program": ["program_1", "program_2"], "fdr": [0.01, 0.03]})

    out = _prepare_reference_family_rows(
        overlap_df=overlap,
        gsea_df=gsea,
        top_k_programs=5,
        top_n_families=10,
    )

    assert not out.empty
    ad_row = out.loc[out["family_key"] == "ALZHEIMERS DISEASE"].iloc[0]
    assert ad_row["references_merged"] == 2
    assert "KEGG" in ad_row["sources_display"]
    assert "WikiPathways" in ad_row["sources_display"]


def test_signal_interpretation_alert_flags_sparse_deg_with_supported_programs() -> None:
    alert = DashboardPublishingAgent._signal_interpretation_alert(0, 3)
    assert alert is not None
    assert alert["title"] == "Sparse single-gene signal"
    assert "coordinated low-amplitude shifts" in alert["text"]


def test_signal_interpretation_alert_ignores_sparse_deg_without_supported_programs() -> None:
    assert DashboardPublishingAgent._signal_interpretation_alert(0, 0) is None


def test_manifest_relative_paths_resolve_from_cwd(tmp_path, monkeypatch) -> None:
    """Manifest paths recorded relative to the launch directory should resolve."""
    workdir = tmp_path / "work"
    results_dir = workdir / "results_run"
    gmt_dir = results_dir / "demo_inputs"
    gmt_dir.mkdir(parents=True, exist_ok=True)
    gmt_path = gmt_dir / "demo_reference.gmt"
    gmt_path.write_text("Demo_Pathway\tna\tGENE1\tGENE2\n", encoding="utf-8")

    monkeypatch.chdir(workdir)
    resolved = ResultIngestionAgent._resolve_manifest_path(
        results_dir,
        "results_run/demo_inputs/demo_reference.gmt",
    )
    assert resolved == gmt_path.resolve()
