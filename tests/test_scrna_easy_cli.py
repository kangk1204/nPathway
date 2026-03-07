"""Tests for the beginner-friendly scRNA easy CLI."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "data" / "scrna_demo_case_vs_ctrl"


def _module_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return env


def _has_r_packages(*packages: str) -> bool:
    if shutil.which("Rscript") is None:
        return False
    expr = "pkgs <- c(%s); cat(all(sapply(pkgs, requireNamespace, quietly=TRUE)))" % \
        ",".join(f'\"{pkg}\"' for pkg in packages)
    result = subprocess.run(
        ["Rscript", "-e", expr],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip().upper() == "TRUE"


@pytest.mark.parametrize("mode", ["wizard-only"])
def test_scrna_easy_cli_wizard_autodetects_demo_columns(tmp_path, mode: str) -> None:
    """Wizard mode should auto-detect the basic demo columns and write a preflight report."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "npathway.cli.scrna_easy",
            "--adata",
            str(DEMO_DIR / "demo_scrna_case_ctrl.h5ad"),
            "--case",
            "case",
            "--control",
            "control",
            "--wizard-only",
            "--output-dir",
            str(tmp_path / "wizard"),
        ],
        cwd=ROOT,
        env=_module_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "wizard" / "preflight_report.html").exists()
    assert (tmp_path / "wizard" / "preflight_summary.json").exists()
    assert "detected_sample_col: donor_id" in result.stdout
    assert "detected_group_col: condition" in result.stdout


def test_scrna_easy_cli_runs_multiple_cell_types_with_simple_backend(tmp_path) -> None:
    """Easy CLI should batch over multiple eligible cell types and build a top-level index."""
    rng = np.random.default_rng(42)
    genes = [f"G{i:03d}" for i in range(40)]
    donors = ["case_1", "case_2", "ctrl_1", "ctrl_2"]

    rows = []
    obs_rows = []
    for donor in donors:
        condition = "case" if donor.startswith("case") else "control"
        batch = "b1" if donor.endswith("1") else "b2"
        for cell_type in ["CD4_T", "B_cell"]:
            for _ in range(3):
                counts = rng.poisson(lam=4.0, size=len(genes)).astype(np.float32)
                if cell_type == "CD4_T" and condition == "case":
                    counts[:8] += 6.0
                if cell_type == "B_cell" and condition == "control":
                    counts[8:16] += 5.0
                rows.append(counts)
                obs_rows.append(
                    {
                        "donor_id": donor,
                        "condition": condition,
                        "cell_type": cell_type,
                        "batch": batch,
                        "age": "72" if donor.startswith("case") else "68",
                    }
                )

    adata = ad.AnnData(
        X=np.vstack(rows),
        obs=pd.DataFrame(obs_rows, index=[f"cell_{i}" for i in range(len(rows))]),
        var=pd.DataFrame(index=genes),
    )
    adata.raw = adata.copy()
    adata_path = tmp_path / "synthetic_scrna.h5ad"
    adata.write_h5ad(adata_path)

    curated_gmt = tmp_path / "curated.gmt"
    curated_gmt.write_text(
        "Case_CD4_Module\tNA\tG000\tG001\tG002\tG003\n"
        "Control_B_Module\tNA\tG008\tG009\tG010\tG011\n",
        encoding="utf-8",
    )

    outdir = tmp_path / "easy_out"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "npathway.cli.scrna_easy",
            "--adata",
            str(adata_path),
            "--case",
            "case",
            "--control",
            "control",
            "--all-cell-types",
            "--min-cells-per-sample",
            "1",
            "--force-simple-backend",
            "--curated-gmt",
            str(curated_gmt),
            "--annotation-gmt",
            str(curated_gmt),
            "--n-programs",
            "4",
            "--n-components",
            "4",
            "--gsea-n-perm",
            "25",
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
    assert (outdir / "preflight_report.html").exists()
    assert (outdir / "analysis_index.html").exists()
    assert (outdir / "cell_type_run_summary.csv").exists()
    assert (outdir / "figure_ready" / "figure_ready_manifest.json").exists()
    assert (outdir / "figure_ready" / "figure_inventory.csv").exists()
    assert (outdir / "figure_ready" / "caption_starter.md").exists()
    summary = pd.read_csv(outdir / "cell_type_run_summary.csv")
    assert set(summary["label"]) == {"CD4_T", "B_cell"}
    assert {"simple_pseudobulk_internal_de"} == set(summary["ranking_source"])
    index_text = (outdir / "analysis_index.html").read_text(encoding="utf-8")
    assert "CD4_T" in index_text
    assert "B_cell" in index_text
    assert "Manuscript View" in index_text
    assert "Frontier Signal" in index_text
    assert "Representative Thumbnails" in index_text
    assert "figure_1_volcano.png" in index_text
    assert "Open figure-ready export package" in index_text
    assert (outdir / "analyses" / "CD4_T" / "comparison" / "summary.md").exists()
    assert (outdir / "analyses" / "B_cell" / "comparison" / "summary.md").exists()


@pytest.mark.skipif(
    not _has_r_packages("limma", "edgeR"),
    reason="batch-aware scRNA easy workflow requires R packages limma and edgeR",
)
def test_scrna_easy_cli_uses_batch_aware_backend_on_demo_dataset(tmp_path) -> None:
    """When R dependencies are available, the easy CLI should use the batch-aware backend by default."""
    outdir = tmp_path / "demo_easy"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "npathway.cli.scrna_easy",
            "--adata",
            str(DEMO_DIR / "demo_scrna_case_ctrl.h5ad"),
            "--case",
            "case",
            "--control",
            "control",
            "--annotation-gmt",
            str(DEMO_DIR / "scrna_reference_demo.gmt"),
            "--curated-gmt",
            str(DEMO_DIR / "scrna_reference_demo.gmt"),
            "--n-programs",
            "4",
            "--n-components",
            "3",
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
    summary = pd.read_csv(outdir / "cell_type_run_summary.csv")
    assert list(summary["label"]) == ["all_cells"]
    assert list(summary["ranking_source"]) == ["edgeR_glmQLF"]
    assert summary.loc[0, "batch_qc_dir"]
    assert (outdir / "analyses" / "all_cells" / "workflow_manifest.json").exists()
    assert (outdir / "analysis_index.html").exists()
    qc_dir = outdir / "analyses" / "all_cells" / "prepared_inputs" / "qc"
    assert (qc_dir / "pca_before.png").exists()
    assert (qc_dir / "pca_after.png").exists()
    assert (qc_dir / "correlation_before.png").exists()
    assert (qc_dir / "correlation_after.png").exists()
    assert (outdir / "figure_ready" / "all_cells" / "batch_pca_before.png").exists()
    assert (outdir / "figure_ready" / "all_cells" / "batch_pca_after.png").exists()
