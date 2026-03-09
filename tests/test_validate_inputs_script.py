"""CLI tests for HTML validation report generation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_npathway_inputs.py"
DEMO_DIR = ROOT / "data" / "bulk_demo_case_vs_ctrl"


def test_validate_inputs_cli_writes_html_report_for_valid_bulk(tmp_path) -> None:
    """Validator should export an HTML report for a valid bulk input pair."""
    html_path = tmp_path / "bulk_validation.html"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "bulk",
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
            "--html-out",
            str(html_path),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert html_path.exists()
    html_text = html_path.read_text(encoding="utf-8")
    assert "VALID" in html_text
    assert "nPathway Input Validation - bulk" in html_text
    assert "n_group_a" in html_text


def test_root_cli_validate_bulk_writes_html_report_for_valid_bulk(tmp_path) -> None:
    """Top-level npathway CLI should dispatch to bulk validation."""
    html_path = tmp_path / "bulk_validation_root.html"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "npathway.cli.main",
            "validate",
            "bulk",
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
            "--html-out",
            str(html_path),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert html_path.exists()
    assert "VALID" in html_path.read_text(encoding="utf-8")


def test_validate_inputs_cli_writes_html_report_for_invalid_bulk(tmp_path) -> None:
    """Validator should still export HTML when validation fails."""
    matrix = pd.DataFrame(
        {
            "gene": ["G1", "G2"],
            "case_1": [10, "bad"],
            "case_2": [9, 8],
            "ctrl_1": [4, 5],
        }
    )
    matrix_path = tmp_path / "bad_matrix.csv"
    matrix.to_csv(matrix_path, index=False)

    metadata = pd.DataFrame(
        {
            "sample": ["case_1", "case_2", "ctrl_1"],
            "condition": ["case", "case", "control"],
        }
    )
    metadata_path = tmp_path / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    html_path = tmp_path / "invalid_bulk_validation.html"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "bulk",
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
            "--html-out",
            str(html_path),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert html_path.exists()
    html_text = html_path.read_text(encoding="utf-8")
    assert "INVALID" in html_text
    assert "non-numeric" in html_text
