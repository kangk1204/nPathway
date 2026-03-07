"""Tests for user-facing input validation helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from npathway.pipeline import validate_bulk_input_files, validate_scrna_pseudobulk_input


def test_validate_bulk_input_files_accepts_valid_contrast(tmp_path) -> None:
    """Valid bulk matrix+metadata should return a structured summary."""
    matrix = pd.DataFrame(
        {
            "gene": ["G1", "G2", "G3"],
            "case_1": [10, 11, 12],
            "case_2": [9, 8, 11],
            "ctrl_1": [4, 5, 6],
            "ctrl_2": [5, 4, 5],
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

    report = validate_bulk_input_files(
        matrix_path=matrix_path,
        metadata_path=metadata_path,
        sample_col="sample",
        group_col="condition",
        group_a="case",
        group_b="control",
    )
    assert report.mode == "bulk"
    assert report.summary["n_group_a"] == 2
    assert report.summary["n_group_b"] == 2
    assert report.warnings


def test_validate_bulk_input_files_rejects_non_numeric_matrix(tmp_path) -> None:
    """Validator should fail fast on non-numeric expression cells."""
    matrix = pd.DataFrame(
        {
            "gene": ["G1", "G2"],
            "case_1": [10, "bad"],
            "case_2": [9, 8],
            "ctrl_1": [4, 5],
        }
    )
    matrix_path = tmp_path / "matrix.csv"
    matrix.to_csv(matrix_path, index=False)

    metadata = pd.DataFrame(
        {
            "sample": ["case_1", "case_2", "ctrl_1"],
            "condition": ["case", "case", "control"],
        }
    )
    metadata_path = tmp_path / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    with pytest.raises(ValueError, match="non-numeric"):
        validate_bulk_input_files(
            matrix_path=matrix_path,
            metadata_path=metadata_path,
            sample_col="sample",
            group_col="condition",
            group_a="case",
            group_b="control",
        )


def test_validate_bulk_input_files_rejects_same_group_labels(tmp_path) -> None:
    """Bulk validation should reject identical contrast labels."""
    matrix = pd.DataFrame(
        {
            "gene": ["G1", "G2"],
            "case_1": [10, 8],
            "case_2": [9, 7],
        }
    )
    matrix_path = tmp_path / "matrix_same_group.csv"
    matrix.to_csv(matrix_path, index=False)

    metadata = pd.DataFrame(
        {
            "sample": ["case_1", "case_2"],
            "condition": ["case", "case"],
        }
    )
    metadata_path = tmp_path / "metadata_same_group.csv"
    metadata.to_csv(metadata_path, index=False)

    with pytest.raises(ValueError, match="different labels"):
        validate_bulk_input_files(
            matrix_path=matrix_path,
            metadata_path=metadata_path,
            sample_col="sample",
            group_col="condition",
            group_a="case",
            group_b="case",
        )


def test_validate_bulk_input_files_rejects_non_finite_matrix(tmp_path) -> None:
    """Validator should reject inf/NaN-like numeric cells."""
    matrix = pd.DataFrame(
        {
            "gene": ["G1", "G2", "G3", "G4"],
            "case_1": [10, 11, np.inf, 7],
            "case_2": [9, 8, 7, 6],
            "ctrl_1": [4, 5, 6, 5],
            "ctrl_2": [5, 4, 5, 4],
        }
    )
    matrix_path = tmp_path / "matrix_inf.csv"
    matrix.to_csv(matrix_path, index=False)

    metadata = pd.DataFrame(
        {
            "sample": ["case_1", "case_2", "ctrl_1", "ctrl_2"],
            "condition": ["case", "case", "control", "control"],
        }
    )
    metadata_path = tmp_path / "metadata_inf.csv"
    metadata.to_csv(metadata_path, index=False)

    with pytest.raises(ValueError, match="non-finite"):
        validate_bulk_input_files(
            matrix_path=matrix_path,
            metadata_path=metadata_path,
            sample_col="sample",
            group_col="condition",
            group_a="case",
            group_b="control",
        )


def test_validate_bulk_input_files_rejects_perfectly_confounded_batch(tmp_path) -> None:
    """Batch-aware validation should fail when batch and group are unidentifiable."""
    matrix = pd.DataFrame(
        {
            "gene": ["G1", "G2", "G3"],
            "case_1": [10, 11, 12],
            "case_2": [9, 8, 11],
            "ctrl_1": [4, 5, 6],
            "ctrl_2": [5, 4, 5],
        }
    )
    matrix_path = tmp_path / "matrix_batch_confounded.csv"
    matrix.to_csv(matrix_path, index=False)

    metadata = pd.DataFrame(
        {
            "sample": ["case_1", "case_2", "ctrl_1", "ctrl_2"],
            "condition": ["case", "case", "control", "control"],
            "batch": ["b1", "b1", "b2", "b2"],
        }
    )
    metadata_path = tmp_path / "metadata_batch_confounded.csv"
    metadata.to_csv(metadata_path, index=False)

    with pytest.raises(ValueError, match="perfectly confounded"):
        validate_bulk_input_files(
            matrix_path=matrix_path,
            metadata_path=metadata_path,
            sample_col="sample",
            group_col="condition",
            group_a="case",
            group_b="control",
            batch_col="batch",
        )


@pytest.mark.parametrize(
    ("bad_value", "message"),
    [
        (-1.0, "non-negative"),
        (1.5, "integers"),
    ],
)
def test_validate_bulk_input_files_rejects_invalid_raw_counts(
    tmp_path,
    bad_value: float,
    message: str,
) -> None:
    """Raw-count validation should reject negative and fractional values."""
    matrix = pd.DataFrame(
        {
            "gene": ["G1", "G2", "G3"],
            "case_1": [10, 11, 12],
            "case_2": [9, bad_value, 11],
            "ctrl_1": [4, 5, 6],
            "ctrl_2": [5, 4, 5],
        }
    )
    matrix_path = tmp_path / f"matrix_raw_{message}.csv"
    matrix.to_csv(matrix_path, index=False)

    metadata = pd.DataFrame(
        {
            "sample": ["case_1", "case_2", "ctrl_1", "ctrl_2"],
            "condition": ["case", "case", "control", "control"],
        }
    )
    metadata_path = tmp_path / f"metadata_raw_{message}.csv"
    metadata.to_csv(metadata_path, index=False)

    with pytest.raises(ValueError, match=message):
        validate_bulk_input_files(
            matrix_path=matrix_path,
            metadata_path=metadata_path,
            sample_col="sample",
            group_col="condition",
            group_a="case",
            group_b="control",
            raw_counts=True,
        )


def test_validate_scrna_pseudobulk_input_accepts_valid_h5ad(tmp_path) -> None:
    """Valid scRNA pseudobulk input should return retained sample counts."""
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
    adata_path = tmp_path / "adata.h5ad"
    adata.write_h5ad(adata_path)

    report = validate_scrna_pseudobulk_input(
        adata_path=adata_path,
        sample_col="donor_id",
        group_col="condition",
        group_a="case",
        group_b="control",
        subset_col="cell_type",
        subset_value="CD4_T",
        min_cells_per_sample=2,
    )
    assert report.mode == "scrna"
    assert report.summary["n_group_a"] == 2
    assert report.summary["n_group_b"] == 2


def test_validate_scrna_pseudobulk_input_rejects_mixed_group_sample(tmp_path) -> None:
    """Same donor appearing in multiple groups should be rejected."""
    adata = ad.AnnData(
        X=np.ones((4, 3), dtype=np.float32),
        obs=pd.DataFrame(
            {
                "donor_id": ["d1", "d1", "d2", "d2"],
                "condition": ["case", "control", "case", "control"],
            },
            index=[f"cell_{i}" for i in range(4)],
        ),
        var=pd.DataFrame(index=[f"G{i}" for i in range(3)]),
    )
    adata_path = tmp_path / "adata_bad.h5ad"
    adata.write_h5ad(adata_path)

    with pytest.raises(ValueError, match="exactly one group label"):
        validate_scrna_pseudobulk_input(
            adata_path=adata_path,
            sample_col="donor_id",
            group_col="condition",
            group_a="case",
            group_b="control",
        )


def test_validate_scrna_pseudobulk_input_rejects_missing_sample_id(tmp_path) -> None:
    """Missing donor/sample IDs should fail instead of becoming 'nan' pseudobulk samples."""
    adata = ad.AnnData(
        X=np.ones((8, 3), dtype=np.float32),
        obs=pd.DataFrame(
            {
                "donor_id": ["d1", "d1", np.nan, np.nan, "d3", "d3", "d4", "d4"],
                "condition": ["case", "case", "case", "case", "control", "control", "control", "control"],
            },
            index=[f"cell_{i}" for i in range(8)],
        ),
        var=pd.DataFrame(index=[f"G{i}" for i in range(3)]),
    )
    adata_path = tmp_path / "adata_missing_sample.h5ad"
    adata.write_h5ad(adata_path)

    with pytest.raises(ValueError, match="contains missing values"):
        validate_scrna_pseudobulk_input(
            adata_path=adata_path,
            sample_col="donor_id",
            group_col="condition",
            group_a="case",
            group_b="control",
            min_cells_per_sample=2,
        )


def test_validate_scrna_pseudobulk_input_uses_layer_without_raw_warning(tmp_path) -> None:
    """Supplying a layer should avoid the misleading adata.raw fallback warning."""
    adata = ad.AnnData(
        X=np.ones((8, 3), dtype=np.float32),
        obs=pd.DataFrame(
            {
                "donor_id": ["d1", "d1", "d2", "d2", "d3", "d3", "d4", "d4"],
                "condition": ["case", "case", "case", "case", "control", "control", "control", "control"],
            },
            index=[f"cell_{i}" for i in range(8)],
        ),
        var=pd.DataFrame(index=[f"G{i}" for i in range(3)]),
    )
    adata.layers["counts"] = np.ones((8, 3), dtype=np.float32)
    adata_path = tmp_path / "adata_layer.h5ad"
    adata.write_h5ad(adata_path)

    report = validate_scrna_pseudobulk_input(
        adata_path=adata_path,
        sample_col="donor_id",
        group_col="condition",
        group_a="case",
        group_b="control",
        layer="counts",
        use_raw=True,
        min_cells_per_sample=2,
    )
    assert all("adata.raw is missing" not in warning for warning in report.warnings)
