"""Tests for same-ranked-list curated-vs-dynamic GSEA comparison outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from npathway.pipeline.gsea_comparison import compare_curated_vs_dynamic_gsea


def test_compare_curated_vs_dynamic_gsea_writes_anchor_and_focus_outputs(tmp_path) -> None:
    """Comparison helper should materialize side-by-side GSEA outputs from one ranking."""
    ranked = pd.DataFrame(
        {
            "gene": ["APP", "APOE", "TREM2", "TYROBP", "MAPT", "PSEN1", "G1", "G2"],
            "score": [5.0, 4.5, 4.0, 3.8, 2.0, 1.8, -1.0, -2.0],
        }
    )
    ranked_path = tmp_path / "ranked.csv"
    ranked.to_csv(ranked_path, index=False)

    dynamic_gmt = tmp_path / "dynamic.gmt"
    dynamic_gmt.write_text(
        "ProgA\tNA\tAPP\tAPOE\tTREM2\tTYROBP\n"
        "ProgB\tNA\tMAPT\tPSEN1\tG1\n",
        encoding="utf-8",
    )
    curated_gmt = tmp_path / "curated.gmt"
    curated_gmt.write_text(
        "KEGG_ALZHEIMERS_DISEASE\tNA\tAPP\tAPOE\tMAPT\tPSEN1\n"
        "WP_ALZHEIMERS_DISEASE\tNA\tAPP\tAPOE\tPSEN1\n",
        encoding="utf-8",
    )

    result = compare_curated_vs_dynamic_gsea(
        ranked_genes_path=ranked_path,
        dynamic_gmt_path=dynamic_gmt,
        curated_gmt_path=curated_gmt,
        output_dir=tmp_path / "comparison",
        n_perm=50,
        seed=42,
        focus_genes=["TREM2", "TYROBP", "APP"],
    )

    assert Path(result.output_dir).exists()
    assert (tmp_path / "comparison" / "dynamic_gsea.csv").exists()
    assert (tmp_path / "comparison" / "curated_gsea.csv").exists()
    assert (tmp_path / "comparison" / "gsea_comparison_combined.csv").exists()
    assert (tmp_path / "comparison" / "dynamic_curated_overlap.csv").exists()
    assert (tmp_path / "comparison" / "focus_gene_membership.csv").exists()
    assert (tmp_path / "comparison" / "curated_panel_gsea.csv").exists()
    assert (tmp_path / "comparison" / "curated_panel_gene_sets.gmt").exists()
    assert (tmp_path / "comparison" / "curated_panel_manifest.json").exists()
    assert (tmp_path / "comparison" / "summary.md").exists()
    assert result.anchor_program == "ProgA"
    assert result.curated_panel_mode == "auto_result_driven"
    assert result.n_curated_panel_sets >= 2

    focus = pd.read_csv(tmp_path / "comparison" / "focus_gene_membership.csv")
    trem2 = focus.loc[focus["gene"] == "TREM2"].iloc[0]
    assert bool(trem2["in_any_dynamic_program"]) is True
    assert bool(trem2["in_any_curated_set"]) is False

    panel = pd.read_csv(tmp_path / "comparison" / "curated_panel_gsea.csv")
    assert "WP_ALZHEIMERS_DISEASE" in set(panel["program"].astype(str))
    assert "KEGG_ALZHEIMERS_DISEASE" in set(panel["program"].astype(str))
