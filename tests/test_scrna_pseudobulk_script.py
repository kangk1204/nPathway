"""Integration test for the scRNA pseudobulk dynamic pathway CLI."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_scrna_pseudobulk_dynamic_pathway.py"
DEMO_DIR = ROOT / "data" / "scrna_demo_case_vs_ctrl"


def test_scrna_pseudobulk_cli_runs_end_to_end(tmp_path) -> None:
    """Synthetic AnnData should run through pseudobulk + bulk dynamic pipeline."""
    rng = np.random.default_rng(42)
    genes = [f"G{i:03d}" for i in range(30)]
    donors = ["case_1", "case_2", "ctrl_1", "ctrl_2"]

    rows = []
    obs_rows = []
    for donor in donors:
        condition = "case" if donor.startswith("case") else "control"
        for cell_idx in range(4):
            cell_type = "CD4_T" if cell_idx < 2 else "B_cell"
            counts = rng.poisson(lam=4.0, size=len(genes)).astype(np.float32)
            if cell_type == "CD4_T" and condition == "case":
                counts[:6] += 6.0
            rows.append(counts)
            obs_rows.append(
                {
                    "donor_id": donor,
                    "condition": condition,
                    "cell_type": cell_type,
                }
            )

    adata = ad.AnnData(
        X=np.vstack(rows),
        obs=pd.DataFrame(obs_rows, index=[f"cell_{i}" for i in range(len(rows))]),
        var=pd.DataFrame(index=genes),
    )
    adata_path = tmp_path / "synthetic_scrna.h5ad"
    adata.write_h5ad(adata_path)

    outdir = tmp_path / "scrna_dynamic"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--adata",
            str(adata_path),
            "--sample-col",
            "donor_id",
            "--group-col",
            "condition",
            "--group-a",
            "case",
            "--group-b",
            "control",
            "--subset-col",
            "cell_type",
            "--subset-value",
            "CD4_T",
            "--min-cells-per-sample",
            "1",
            "--discovery-method",
            "kmeans",
            "--n-programs",
            "4",
            "--n-components",
            "3",
            "--gsea-n-perm",
            "20",
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

    assert (outdir / "inputs" / "pseudobulk.h5ad").exists()
    assert (outdir / "inputs" / "pseudobulk_matrix.csv").exists()
    assert (outdir / "inputs" / "pseudobulk_metadata.csv").exists()
    assert (outdir / "differential" / "de_results.csv").exists()
    assert (outdir / "discovery" / "dynamic_programs.gmt").exists()
    assert (outdir / "summary.md").exists()

    pb_meta = pd.read_csv(outdir / "inputs" / "pseudobulk_metadata.csv")
    assert set(pb_meta["condition"]) == {"case", "control"}
    assert len(pb_meta) == 4
    assert (pb_meta["n_cells"] >= 2).all()


def test_scrna_pseudobulk_cli_surfaces_friendly_error(tmp_path) -> None:
    """Invalid scRNA pseudobulk input should not leak a raw traceback to the user."""
    adata = ad.AnnData(
        X=np.ones((8, 5), dtype=np.float32),
        obs=pd.DataFrame(
            {
                "donor_id": ["d1", "d1", "d2", "d2", "d3", "d3", "d4", "d4"],
                "condition": ["case", "case", "case", "case", "control", "control", "control", "control"],
                "cell_type": ["CD4_T"] * 8,
            },
            index=[f"cell_{i}" for i in range(8)],
        ),
        var=pd.DataFrame(index=[f"G{i}" for i in range(5)]),
    )
    adata_path = tmp_path / "tiny_scrna.h5ad"
    adata.write_h5ad(adata_path)

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--adata",
            str(adata_path),
            "--sample-col",
            "donor_id",
            "--group-col",
            "condition",
            "--group-a",
            "case",
            "--group-b",
            "control",
            "--min-cells-per-sample",
            "10",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "ERROR:" in result.stderr
    assert "validate_npathway_inputs.py scrna" in result.stderr
    assert "Traceback" not in result.stderr


def test_scrna_pseudobulk_cli_runs_on_bundled_demo_dataset(tmp_path) -> None:
    """Bundled scRNA demo data should support a first end-to-end run."""
    outdir = tmp_path / "demo_scrna_run"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--adata",
            str(DEMO_DIR / "demo_scrna_case_ctrl.h5ad"),
            "--sample-col",
            "donor_id",
            "--group-col",
            "condition",
            "--group-a",
            "case",
            "--group-b",
            "control",
            "--subset-col",
            "cell_type",
            "--subset-value",
            "CD4_T",
            "--min-cells-per-sample",
            "10",
            "--n-programs",
            "4",
            "--n-components",
            "3",
            "--gsea-n-perm",
            "50",
            "--annotate-programs",
            "--annotation-gmt",
            str(DEMO_DIR / "scrna_reference_demo.gmt"),
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
    assert (outdir / "inputs" / "pseudobulk.h5ad").exists()
    assert (outdir / "inputs" / "pseudobulk_matrix.csv").exists()
    assert (outdir / "inputs" / "pseudobulk_metadata.csv").exists()
    assert (outdir / "annotation" / "program_annotation_matches.csv").exists()
