"""Tests for the comparison-only GSEA CLI."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _module_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return env


def test_compare_gsea_cli_runs_end_to_end(tmp_path) -> None:
    """Installed comparison CLI should produce side-by-side GSEA artifacts."""
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

    outdir = tmp_path / "compare_out"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "npathway.cli.compare_gsea",
            "--ranked-genes",
            str(ranked_path),
            "--dynamic-gmt",
            str(dynamic_gmt),
            "--curated-gmt",
            str(curated_gmt),
            "--focus-genes",
            "TREM2,TYROBP,APP",
            "--gsea-n-perm",
            "50",
            "--output-dir",
            str(outdir),
        ],
        cwd=ROOT,
        env=_module_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (outdir / "dynamic_gsea.csv").exists()
    assert (outdir / "curated_gsea.csv").exists()
    assert (outdir / "gsea_comparison_combined.csv").exists()
    assert (outdir / "focus_gene_membership.csv").exists()
    assert (outdir / "comparison_cli_manifest.json").exists()
    assert "nPathway comparison mode completed." in result.stdout
