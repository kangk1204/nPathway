"""Tests for GMT/GMX I/O functions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from npathway.utils.gmt_io import (
    programs_to_df,
    read_gmt,
    read_gmx,
    weighted_programs_to_gmt,
    write_gmt,
    write_gmx,
)


def test_write_read_gmt_roundtrip(
    sample_gene_programs: dict[str, list[str]],
    tmp_output_dir: Path,
) -> None:
    """Write gene programs to GMT, read them back, verify identical content."""
    filepath = str(tmp_output_dir / "test.gmt")
    write_gmt(sample_gene_programs, filepath)
    loaded = read_gmt(filepath)

    assert set(loaded.keys()) == set(sample_gene_programs.keys())
    for name, genes in sample_gene_programs.items():
        assert loaded[name] == genes, f"Mismatch for program '{name}'"


def test_write_read_gmx_roundtrip(
    sample_gene_programs: dict[str, list[str]],
    tmp_output_dir: Path,
) -> None:
    """Write gene programs to GMX, read them back, verify identical content."""
    filepath = str(tmp_output_dir / "test.gmx")
    write_gmx(sample_gene_programs, filepath)
    loaded = read_gmx(filepath)

    assert set(loaded.keys()) == set(sample_gene_programs.keys())
    for name, genes in sample_gene_programs.items():
        assert loaded[name] == genes, f"Mismatch for program '{name}'"


def test_programs_to_df(sample_gene_programs: dict[str, list[str]]) -> None:
    """Convert programs to binary DataFrame and verify structure."""
    df = programs_to_df(sample_gene_programs)

    assert isinstance(df, pd.DataFrame)
    assert df.index.name == "gene"
    assert set(df.columns) == set(sample_gene_programs.keys())

    # Check that each gene-program membership is represented by a 1
    for prog_name, genes in sample_gene_programs.items():
        for gene in genes:
            assert df.loc[gene, prog_name] == 1

    # All values should be 0 or 1
    assert set(df.values.flatten()) <= {0, 1}


def test_weighted_programs_to_gmt(tmp_output_dir: Path) -> None:
    """Write weighted programs to GMT-like file, verify file has correct format."""
    weighted = {
        "prog_A": [("GENE1", 0.95), ("GENE2", 0.87), ("GENE3", 0.60)],
        "prog_B": [("GENE4", 0.78), ("GENE5", 0.55)],
    }
    filepath = str(tmp_output_dir / "weighted.gmt")
    weighted_programs_to_gmt(weighted, filepath)

    # Read back and parse manually
    with open(filepath, "r") as fh:
        lines = fh.readlines()

    assert len(lines) == 2

    # First line should be prog_A
    parts = lines[0].strip().split("\t")
    assert parts[0] == "prog_A"
    assert parts[1] == "na"
    # Gene entries should be in "gene,weight" format
    gene_weight_strs = parts[2:]
    assert len(gene_weight_strs) == 3
    assert gene_weight_strs[0].startswith("GENE1,")


def test_read_gmt_nonexistent_file() -> None:
    """Reading a nonexistent GMT file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        read_gmt("/nonexistent/path/file.gmt")


def test_write_gmt_creates_directory(tmp_output_dir: Path) -> None:
    """write_gmt should create parent directories if they don't exist."""
    nested_path = str(tmp_output_dir / "subdir1" / "subdir2" / "output.gmt")
    programs = {"prog_A": ["GENE1", "GENE2"]}
    write_gmt(programs, nested_path)
    assert Path(nested_path).exists()

    loaded = read_gmt(nested_path)
    assert loaded["prog_A"] == ["GENE1", "GENE2"]


def test_empty_programs(tmp_output_dir: Path) -> None:
    """Writing empty programs should raise ValueError."""
    filepath = str(tmp_output_dir / "empty.gmt")
    with pytest.raises(ValueError, match="empty"):
        write_gmt({}, filepath)

    with pytest.raises(ValueError, match="empty"):
        write_gmx({}, filepath)

    with pytest.raises(ValueError, match="empty"):
        programs_to_df({})

    with pytest.raises(ValueError, match="empty"):
        weighted_programs_to_gmt({}, filepath)
