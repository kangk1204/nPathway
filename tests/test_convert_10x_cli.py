"""Tests for the 10x-to-h5ad conversion helper CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import anndata as ad
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite

ROOT = Path(__file__).resolve().parents[1]


def _module_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return env


def _write_10x_matrix_dir(base: Path, *, sample_tag: str, counts: list[list[int]], genes: list[tuple[str, str]], barcodes: list[str]) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    matrix = sparse.coo_matrix(counts)
    mmwrite(str(base / "matrix.mtx"), matrix)
    (base / "barcodes.tsv").write_text("\n".join(barcodes) + "\n", encoding="utf-8")
    features_text = "\n".join(f"{gene_id}\t{gene_symbol}\tGene Expression" for gene_id, gene_symbol in genes) + "\n"
    (base / "features.tsv").write_text(features_text, encoding="utf-8")
    return base


def test_convert_10x_cli_check_only_writes_status_report(tmp_path) -> None:
    """Check-only mode should report Python dependency readiness."""
    status_json = tmp_path / "tenx_status.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "npathway.cli.convert_10x",
            "--check-only",
            "--status-json",
            str(status_json),
        ],
        cwd=ROOT,
        env=_module_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(status_json.read_text(encoding="utf-8"))
    assert payload["ready"] is True
    assert set(payload["dependencies"].keys()) == {"scanpy", "anndata", "scipy", "h5py"}
    assert "nPathway 10x conversion readiness" in result.stdout


def test_convert_10x_cli_single_matrix_dir_with_obs_metadata(tmp_path) -> None:
    """A single 10x matrix directory should convert into .h5ad and merge barcode metadata."""
    matrix_dir = _write_10x_matrix_dir(
        tmp_path / "sampleA_10x",
        sample_tag="A",
        counts=[
            [3, 0, 1],
            [0, 5, 2],
        ],
        genes=[("ENSG0001", "APP"), ("ENSG0002", "TREM2")],
        barcodes=["AAAC-1", "AAAG-1", "AAAT-1"],
    )
    obs_csv = tmp_path / "obs.csv"
    pd.DataFrame(
        {
            "barcode": ["AAAC-1", "AAAG-1", "AAAT-1"],
            "donor_id": ["donorA", "donorA", "donorA"],
            "condition": ["case", "case", "case"],
            "cell_type": ["Microglia", "Microglia", "Microglia"],
        }
    ).to_csv(obs_csv, index=False)
    output_h5ad = tmp_path / "sampleA.h5ad"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "npathway.cli.convert_10x",
            "--matrix-dir",
            str(matrix_dir),
            "--sample-id",
            "sampleA",
            "--obs-csv",
            str(obs_csv),
            "--output-h5ad",
            str(output_h5ad),
        ],
        cwd=ROOT,
        env=_module_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    adata = ad.read_h5ad(output_h5ad)
    assert adata.n_obs == 3
    assert adata.n_vars == 2
    assert set(adata.obs["sample_id"].astype(str)) == {"sampleA"}
    assert set(adata.obs["condition"].astype(str)) == {"case"}
    assert set(adata.obs["cell_type"].astype(str)) == {"Microglia"}
    assert all(name.startswith("sampleA:") for name in adata.obs_names.astype(str))



def test_convert_10x_cli_manifest_combines_multiple_samples(tmp_path) -> None:
    """Manifest mode should combine multiple 10x inputs and copy manifest columns into obs."""
    dir_a = _write_10x_matrix_dir(
        tmp_path / "sampleA_10x",
        sample_tag="A",
        counts=[[2, 0], [0, 3]],
        genes=[("ENSG0001", "APP"), ("ENSG0002", "TREM2")],
        barcodes=["A1", "A2"],
    )
    dir_b = _write_10x_matrix_dir(
        tmp_path / "sampleB_10x",
        sample_tag="B",
        counts=[[1, 4], [2, 0]],
        genes=[("ENSG0001", "APP"), ("ENSG0002", "TREM2")],
        barcodes=["B1", "B2"],
    )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "input_path": [str(dir_a), str(dir_b)],
            "sample_id": ["sampleA", "sampleB"],
            "donor_id": ["donorA", "donorB"],
            "condition": ["case", "control"],
            "batch": ["b1", "b2"],
        }
    ).to_csv(manifest, index=False)
    output_h5ad = tmp_path / "combined.h5ad"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "npathway.cli.convert_10x",
            "--manifest",
            str(manifest),
            "--output-h5ad",
            str(output_h5ad),
        ],
        cwd=ROOT,
        env=_module_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    adata = ad.read_h5ad(output_h5ad)
    assert adata.n_obs == 4
    assert set(adata.obs["sample_id"].astype(str)) == {"sampleA", "sampleB"}
    assert set(adata.obs["condition"].astype(str)) == {"case", "control"}
    assert set(adata.obs["batch"].astype(str)) == {"b1", "b2"}
    assert all(":" in name for name in adata.obs_names.astype(str))
