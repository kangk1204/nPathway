"""Integration tests for the bulk dynamic pathway CLI."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_bulk_dynamic_pathway.py"
DEMO_DIR = ROOT / "data" / "bulk_demo_case_vs_ctrl"


def test_bulk_dynamic_cli_surfaces_friendly_error_for_invalid_raw_counts(tmp_path) -> None:
    """Bulk CLI should fail fast on malformed raw-count input without a traceback."""
    matrix = pd.DataFrame(
        {
            "gene": ["G1", "G2", "G3"],
            "case_1": [10, 11, 12],
            "case_2": [9, -1, 11],
            "ctrl_1": [4, 5, 6],
            "ctrl_2": [5, 4, 5],
        }
    )
    matrix_path = tmp_path / "bad_counts.csv"
    matrix.to_csv(matrix_path, index=False)

    metadata = pd.DataFrame(
        {
            "sample": ["case_1", "case_2", "ctrl_1", "ctrl_2"],
            "condition": ["case", "case", "control", "control"],
        }
    )
    metadata_path = tmp_path / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--matrix",
            str(matrix_path),
            "--metadata",
            str(metadata_path),
            "--sample-col",
            "sample",
            "--group-col",
            "condition",
            "--group-a",
            "case",
            "--group-b",
            "control",
            "--raw-counts",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "ERROR:" in result.stderr
    assert "non-negative" in result.stderr
    assert "validate_npathway_inputs.py bulk" in result.stderr
    assert "Traceback" not in result.stderr


def test_bulk_dynamic_cli_runs_on_bundled_demo_dataset(tmp_path) -> None:
    """Bundled demo data should support a first end-to-end bulk run."""
    outdir = tmp_path / "demo_bulk_run"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--matrix",
            str(DEMO_DIR / "bulk_matrix_case_ctrl_demo.csv"),
            "--metadata",
            str(DEMO_DIR / "bulk_metadata_case_ctrl_demo.csv"),
            "--sample-col",
            "sample",
            "--group-col",
            "condition",
            "--group-a",
            "case",
            "--group-b",
            "control",
            "--n-programs",
            "8",
            "--n-components",
            "8",
            "--gsea-n-perm",
            "50",
            "--annotate-programs",
            "--annotation-gmt",
            str(DEMO_DIR / "bulk_reference_demo.gmt"),
            "--output-dir",
            str(outdir),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (outdir / "de_results.csv").exists()
    assert (outdir / "dynamic_programs.gmt").exists()
    assert (outdir / "summary.md").exists()
    assert (outdir / "program_annotation_matches.csv").exists()


def test_bulk_dynamic_cli_uses_external_ranking(tmp_path) -> None:
    """Bulk CLI should reuse a user-supplied ranked-gene table for GSEA."""
    matrix = pd.DataFrame(
        {
            "gene": [f"G{i}" for i in range(1, 9)],
            "case_1": [8.2, 8.0, 7.9, 6.1, 5.9, 5.7, 5.5, 5.4],
            "case_2": [8.1, 8.3, 7.8, 6.0, 5.8, 5.9, 5.4, 5.5],
            "ctrl_1": [5.0, 5.1, 5.2, 7.0, 7.1, 7.0, 5.4, 5.5],
            "ctrl_2": [5.2, 5.0, 5.1, 6.9, 7.2, 7.1, 5.3, 5.4],
        }
    )
    matrix_path = tmp_path / "matrix.csv"
    matrix.to_csv(matrix_path, index=False)

    metadata = pd.DataFrame(
        {
            "sample": ["case_1", "case_2", "ctrl_1", "ctrl_2"],
            "condition": ["case", "case", "control", "control"],
        }
    )
    metadata_path = tmp_path / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    ranked = pd.DataFrame(
        {
            "gene": ["G4", "G2", "G1", "G7", "G8", "G3", "G5", "G6"],
            "score": [4.0, 3.5, 3.0, 1.0, 0.5, -0.5, -2.0, -3.0],
        }
    )
    ranked_path = tmp_path / "ranked.csv"
    ranked.to_csv(ranked_path, index=False)

    curated_gmt = tmp_path / "curated.gmt"
    curated_gmt.write_text(
        "Case_Module\tNA\tG1\tG2\tG3\nControl_Module\tNA\tG4\tG5\tG6\n",
        encoding="utf-8",
    )

    outdir = tmp_path / "external_ranking_run"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--matrix",
            str(matrix_path),
            "--metadata",
            str(metadata_path),
            "--sample-col",
            "sample",
            "--group-col",
            "condition",
            "--group-a",
            "case",
            "--group-b",
            "control",
            "--ranked-genes",
            str(ranked_path),
            "--annotation-gmt",
            str(curated_gmt),
            "--annotate-programs",
            "--n-programs",
            "3",
            "--n-components",
            "3",
            "--gsea-n-perm",
            "25",
            "--output-dir",
            str(outdir),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    ranked_out = pd.read_csv(outdir / "ranked_genes_for_gsea.csv")
    assert ranked_out["gene"].tolist()[:4] == ranked["gene"].tolist()[:4]
    assert ranked_out["score"].tolist()[:4] == ranked["score"].tolist()[:4]
