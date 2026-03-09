"""Tests for installed-style nPathway demo commands."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import anndata as ad
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _module_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return env


def test_demo_cli_bulk_runs_end_to_end(tmp_path) -> None:
    """Installed bulk demo module should generate inputs, results, and the default dashboard."""
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
    assert (tmp_path / "demo_bulk" / "differential" / "de_results.csv").exists()
    assert (tmp_path / "demo_bulk" / "discovery" / "dynamic_programs.gmt").exists()
    assert (tmp_path / "demo_bulk" / "demo_inputs" / "bulk_matrix_case_ctrl_demo.csv").exists()
    assert (tmp_path / "demo_bulk" / "index.html").exists()
    assert (tmp_path / "demo_bulk" / "dashboard" / "index.html").exists()
    assert (tmp_path / "demo_bulk" / "figures" / "figure_1_volcano.pdf").exists()
    assert (tmp_path / "demo_bulk" / "figures" / "figure_5_multi_pathway.pdf").exists()
    assert (tmp_path / "demo_bulk" / "figures" / "figure_6_multi_pathway_enrichment_curves.pdf").exists()
    genes = pd.read_csv(tmp_path / "demo_bulk" / "demo_inputs" / "bulk_matrix_case_ctrl_demo.csv")["gene"].tolist()
    assert genes[0] == "HEG1"
    assert not any(g.startswith("G0") for g in genes[:20])
    assert "dashboard_html" in result.stdout


def test_root_cli_demo_bulk_runs_end_to_end(tmp_path) -> None:
    """Top-level npathway CLI should dispatch to the bulk demo and write the dashboard by default."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "npathway.cli.main",
            "demo",
            "bulk",
            "--output-dir",
            str(tmp_path / "root_demo_bulk"),
        ],
        cwd=ROOT,
        env=_module_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "root_demo_bulk" / "differential" / "de_results.csv").exists()
    assert (tmp_path / "root_demo_bulk" / "discovery" / "dynamic_programs.gmt").exists()
    assert (tmp_path / "root_demo_bulk" / "index.html").exists()
    assert (tmp_path / "root_demo_bulk" / "dashboard" / "index.html").exists()


def test_demo_cli_scrna_runs_end_to_end(tmp_path) -> None:
    """Installed scRNA demo module should generate inputs, results, and the default dashboard."""
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
    assert (tmp_path / "demo_scrna" / "differential" / "de_results.csv").exists()
    assert (tmp_path / "demo_scrna" / "discovery" / "dynamic_programs.gmt").exists()
    assert (tmp_path / "demo_scrna" / "demo_inputs" / "demo_scrna_case_ctrl.h5ad").exists()
    assert (tmp_path / "demo_scrna" / "inputs" / "pseudobulk_matrix.csv").exists()
    assert (tmp_path / "demo_scrna" / "index.html").exists()
    assert (tmp_path / "demo_scrna" / "dashboard" / "index.html").exists()
    assert (tmp_path / "demo_scrna" / "figures" / "figure_1_volcano.pdf").exists()
    assert (tmp_path / "demo_scrna" / "figures" / "figure_5_multi_pathway.pdf").exists()
    assert (tmp_path / "demo_scrna" / "figures" / "figure_6_multi_pathway_enrichment_curves.pdf").exists()
    adata = ad.read_h5ad(tmp_path / "demo_scrna" / "demo_inputs" / "demo_scrna_case_ctrl.h5ad")
    genes = adata.var_names.astype(str).tolist()
    assert genes[0] == "IL7R"
    assert not any(g.startswith("G0") for g in genes[:20])
    assert "dashboard_html" in result.stdout
