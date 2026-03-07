"""Tests for installed-style nPathway demo commands."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return env


def test_demo_cli_bulk_runs_end_to_end(tmp_path) -> None:
    """Installed bulk demo module should generate inputs and results."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "npathway.cli.demo",
            "bulk",
            "--output-dir",
            str(tmp_path / "demo_bulk"),
        ],
        cwd=ROOT,
        env=_module_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "demo_bulk" / "de_results.csv").exists()
    assert (tmp_path / "demo_bulk" / "dynamic_programs.gmt").exists()
    assert (tmp_path / "demo_bulk" / "demo_inputs" / "bulk_matrix_case_ctrl_demo.csv").exists()


def test_demo_cli_scrna_runs_end_to_end(tmp_path) -> None:
    """Installed scRNA demo module should generate inputs and results."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "npathway.cli.demo",
            "scrna",
            "--output-dir",
            str(tmp_path / "demo_scrna"),
        ],
        cwd=ROOT,
        env=_module_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "demo_scrna" / "de_results.csv").exists()
    assert (tmp_path / "demo_scrna" / "dynamic_programs.gmt").exists()
    assert (tmp_path / "demo_scrna" / "demo_inputs" / "demo_scrna_case_ctrl.h5ad").exists()
    assert (tmp_path / "demo_scrna" / "inputs" / "pseudobulk_matrix.csv").exists()
