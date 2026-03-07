"""Tests for the batch-aware beginner bulk workflow CLI."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


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


def _write_synthetic_bulk_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    rng = np.random.default_rng(42)
    genes = [
        "APP",
        "APOE",
        "TREM2",
        "TYROBP",
        "MAPT",
        "PSEN1",
    ] + [f"G{i:03d}" for i in range(1, 55)]
    samples = [
        "case_b1_1",
        "case_b1_2",
        "ctrl_b1_1",
        "ctrl_b1_2",
        "case_b2_1",
        "case_b2_2",
        "ctrl_b2_1",
        "ctrl_b2_2",
    ]

    counts = rng.poisson(lam=30, size=(len(genes), len(samples))).astype(int)
    counts[:6, [0, 1, 4, 5]] += 35
    counts[10:20, [2, 3, 6, 7]] += 20
    counts[:, [4, 5, 6, 7]] += 8

    matrix = pd.DataFrame(counts, index=genes, columns=samples).reset_index()
    matrix.columns = ["gene"] + samples
    matrix_path = tmp_path / "counts.csv"
    matrix.to_csv(matrix_path, index=False)

    metadata = pd.DataFrame(
        {
            "sample": samples,
            "condition": ["case", "case", "control", "control", "case", "case", "control", "control"],
            "batch": ["b1", "b1", "b1", "b1", "b2", "b2", "b2", "b2"],
            "age": [71, 74, 69, 70, 73, 76, 68, 72],
        }
    )
    metadata_path = tmp_path / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    curated_gmt = tmp_path / "curated.gmt"
    curated_gmt.write_text(
        "KEGG_ALZHEIMERS_DISEASE\tNA\tAPP\tAPOE\tMAPT\tPSEN1\n"
        "AD_MICROGLIA_EXTENSION\tNA\tTREM2\tTYROBP\tAPP\tAPOE\n",
        encoding="utf-8",
    )
    return matrix_path, metadata_path, curated_gmt


def _write_hidden_factor_bulk_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    rng = np.random.default_rng(0)
    n_genes = 120
    samples: list[str] = []
    groups: list[str] = []
    batches: list[str] = []
    ages: list[int] = []
    latents: list[str] = []
    for group in ["case", "control"]:
        for batch in ["b1", "b2"]:
            for rep in range(4):
                samples.append(f"{group}_{batch}_{rep + 1}")
                groups.append(group)
                batches.append(batch)
                ages.append(70 + rep + int(group == "case"))
                latents.append("l1" if rep % 2 == 0 else "l2")

    counts = rng.poisson(40, size=(n_genes, len(samples))).astype(int)
    case_idx = [idx for idx, value in enumerate(groups) if value == "case"]
    ctrl_idx = [idx for idx, value in enumerate(groups) if value == "control"]
    l1_idx = [idx for idx, value in enumerate(latents) if value == "l1"]
    l2_idx = [idx for idx, value in enumerate(latents) if value == "l2"]
    counts[:15, case_idx] += 25
    counts[15:30, ctrl_idx] += 18
    counts[30:75, l1_idx] += 30
    counts[75:100, l2_idx] += 22

    genes = [f"G{i:03d}" for i in range(n_genes)]
    matrix = pd.DataFrame(counts, index=genes, columns=samples).reset_index()
    matrix.columns = ["gene"] + samples
    matrix_path = tmp_path / "hidden_counts.csv"
    matrix.to_csv(matrix_path, index=False)

    metadata = pd.DataFrame(
        {
            "sample": samples,
            "condition": groups,
            "batch": batches,
            "age": ages,
        }
    )
    metadata_path = tmp_path / "hidden_metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    curated_gmt = tmp_path / "hidden_curated.gmt"
    curated_gmt.write_text("SET1\tNA\t" + "\t".join(genes[:10]) + "\n", encoding="utf-8")
    return matrix_path, metadata_path, curated_gmt


@pytest.mark.skipif(
    not _has_r_packages("limma", "edgeR"),
    reason="batch-aware bulk workflow requires R packages limma and edgeR",
)
def test_bulk_workflow_cli_runs_end_to_end_with_batch_and_curated_comparison(tmp_path) -> None:
    """Workflow CLI should prepare adjusted inputs, run nPathway, and compare against curated GMTs."""
    matrix_path, metadata_path, curated_gmt = _write_synthetic_bulk_inputs(tmp_path)

    outdir = tmp_path / "workflow_out"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "npathway.cli.bulk_workflow",
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
            "--batch-col",
            "batch",
            "--covariate-cols",
            "age",
            "--curated-gmt",
            str(curated_gmt),
            "--focus-genes",
            "TREM2,TYROBP,APP",
            "--n-programs",
            "4",
            "--k-neighbors",
            "3",
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
    assert (outdir / "prepared_inputs" / "discovery_matrix_adjusted.csv").exists()
    assert (outdir / "prepared_inputs" / "aligned_metadata.csv").exists()
    assert (outdir / "prepared_inputs" / "ranked_genes_external.csv").exists()
    assert (outdir / "prepared_inputs" / "de_results_external.csv").exists()
    assert (outdir / "prepared_inputs" / "qc" / "pca_before.png").exists()
    assert (outdir / "prepared_inputs" / "qc" / "pca_after.png").exists()
    assert (outdir / "prepared_inputs" / "qc" / "pca_summary.json").exists()
    assert (outdir / "dynamic_programs.gmt").exists()
    assert (outdir / "comparison" / "dynamic_gsea.csv").exists()
    assert (outdir / "comparison" / "curated_gsea.csv").exists()
    assert (outdir / "comparison" / "focus_gene_membership.csv").exists()
    assert (outdir / "workflow_manifest.json").exists()
    assert "ranking_source: edgeR_glmQLF" in result.stdout


@pytest.mark.skipif(
    not _has_r_packages("limma", "edgeR") or _has_r_packages("sva"),
    reason="auto SVA fallback test requires limma+edgeR and a missing sva package",
)
def test_bulk_workflow_cli_auto_sva_falls_back_cleanly_when_sva_missing(tmp_path) -> None:
    matrix_path, metadata_path, curated_gmt = _write_synthetic_bulk_inputs(tmp_path)
    outdir = tmp_path / "workflow_auto_sva_missing"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "npathway.cli.bulk_workflow",
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
            "--batch-col",
            "batch",
            "--covariate-cols",
            "age",
            "--surrogate-variable-mode",
            "auto",
            "--curated-gmt",
            str(curated_gmt),
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
    manifest = json.loads((outdir / "prepared_inputs" / "preparation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["surrogate_variable_mode"] == "auto"
    assert manifest["surrogate_variable_used"] is False
    assert manifest["n_surrogate_variables"] == 0
    assert "sva package not installed" in manifest["surrogate_variable_message"]
    assert "surrogate_variable_mode: auto" in result.stdout
    assert "surrogate_variable_used: False" in result.stdout


@pytest.mark.skipif(
    not _has_r_packages("limma", "edgeR") or _has_r_packages("sva"),
    reason="required-SVA failure test requires limma+edgeR and a missing sva package",
)
def test_bulk_workflow_cli_requires_sva_when_mode_on_and_package_missing(tmp_path) -> None:
    matrix_path, metadata_path, curated_gmt = _write_synthetic_bulk_inputs(tmp_path)
    outdir = tmp_path / "workflow_sva_required"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "npathway.cli.bulk_workflow",
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
            "--batch-col",
            "batch",
            "--covariate-cols",
            "age",
            "--surrogate-variable-mode",
            "on",
            "--curated-gmt",
            str(curated_gmt),
            "--output-dir",
            str(outdir),
        ],
        cwd=ROOT,
        env=_module_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    combined = f"{result.stdout}\n{result.stderr}"
    assert "requires R packages that are not installed: sva" in combined or "SVA was requested" in combined


@pytest.mark.skipif(
    not _has_r_packages("limma", "edgeR", "sva"),
    reason="guarded SVA residual-df test requires limma+edgeR+sva",
)
def test_bulk_workflow_cli_records_guarded_skip_when_residual_df_too_small(tmp_path) -> None:
    matrix_path, metadata_path, curated_gmt = _write_synthetic_bulk_inputs(tmp_path)
    outdir = tmp_path / "workflow_residual_df_skip"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "npathway.cli.bulk_workflow",
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
            "--batch-col",
            "batch",
            "--covariate-cols",
            "age",
            "--surrogate-variable-mode",
            "auto",
            "--sva-min-residual-df",
            "100",
            "--curated-gmt",
            str(curated_gmt),
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
    manifest = json.loads((outdir / "prepared_inputs" / "preparation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["surrogate_variable_used"] is False
    assert "residual degrees of freedom too small" in manifest["surrogate_variable_message"]


@pytest.mark.skipif(
    not _has_r_packages("limma", "edgeR", "sva"),
    reason="positive SVA usage test requires limma+edgeR+sva",
)
def test_bulk_workflow_cli_uses_surrogate_variables_when_requested_and_supported(tmp_path) -> None:
    matrix_path, metadata_path, curated_gmt = _write_hidden_factor_bulk_inputs(tmp_path)
    outdir = tmp_path / "workflow_sva_positive"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "npathway.cli.bulk_workflow",
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
            "--batch-col",
            "batch",
            "--covariate-cols",
            "age",
            "--surrogate-variable-mode",
            "on",
            "--sva-max-n-sv",
            "2",
            "--gsea-n-perm",
            "20",
            "--n-programs",
            "6",
            "--n-components",
            "6",
            "--curated-gmt",
            str(curated_gmt),
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
    manifest = json.loads((outdir / "prepared_inputs" / "preparation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["surrogate_variable_mode"] == "on"
    assert manifest["surrogate_variable_used"] is True
    assert int(manifest["n_surrogate_variables"]) >= 1
    assert manifest["surrogate_variable_columns"] == ["SV1"]
    assert manifest["discovery_adjustment"].endswith("+sva")
    assert "surrogate_variable_used: True" in result.stdout


@pytest.mark.skipif(
    not _has_r_packages("limma", "edgeR"),
    reason="batch-aware bulk workflow requires R packages limma and edgeR",
)
def test_bulk_workflow_cli_writes_batch_qc_pca_artifacts(tmp_path) -> None:
    matrix_path, metadata_path, curated_gmt = _write_synthetic_bulk_inputs(tmp_path)
    outdir = tmp_path / "workflow_batch_qc"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "npathway.cli.bulk_workflow",
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
            "--batch-col",
            "batch",
            "--covariate-cols",
            "age",
            "--curated-gmt",
            str(curated_gmt),
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
    qc_dir = outdir / "prepared_inputs" / "qc"
    assert (qc_dir / "pca_before.csv").exists()
    assert (qc_dir / "pca_after.csv").exists()
    assert (qc_dir / "pca_before.png").exists()
    assert (qc_dir / "pca_after.png").exists()
    assert (qc_dir / "correlation_before.png").exists()
    assert (qc_dir / "correlation_after.png").exists()
    summary = json.loads((qc_dir / "pca_summary.json").read_text(encoding="utf-8"))
    before = pd.read_csv(qc_dir / "pca_before.csv")
    after = pd.read_csv(qc_dir / "pca_after.csv")
    metadata = pd.read_csv(metadata_path)
    assert len(before) == len(metadata)
    assert len(after) == len(metadata)
    assert "batch" in before.columns
    assert "batch" in after.columns
    assert before[["PC1", "PC2"]].round(6).equals(after[["PC1", "PC2"]].round(6)) is False
    assert summary["n_samples"] == len(metadata)
    assert summary["batch_col"] == "batch"


@pytest.mark.skipif(
    not _has_r_packages("limma", "edgeR"),
    reason="batch-aware bulk workflow requires R packages limma and edgeR",
)
def test_bulk_workflow_cli_manifest_matches_stdout_for_sva_metadata(tmp_path) -> None:
    matrix_path, metadata_path, curated_gmt = _write_synthetic_bulk_inputs(tmp_path)
    outdir = tmp_path / "workflow_manifest_stdout"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "npathway.cli.bulk_workflow",
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
            "--batch-col",
            "batch",
            "--covariate-cols",
            "age",
            "--surrogate-variable-mode",
            "auto",
            "--curated-gmt",
            str(curated_gmt),
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
    manifest = json.loads((outdir / "prepared_inputs" / "preparation_manifest.json").read_text(encoding="utf-8"))
    stdout_map = {}
    for line in result.stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        stdout_map[key.strip("- ").strip()] = value.strip()
    assert stdout_map["surrogate_variable_mode"] == manifest["surrogate_variable_mode"]
    assert stdout_map["surrogate_variable_used"].lower() == str(manifest["surrogate_variable_used"]).lower()
    assert int(stdout_map["n_surrogate_variables"]) == int(manifest["n_surrogate_variables"])
