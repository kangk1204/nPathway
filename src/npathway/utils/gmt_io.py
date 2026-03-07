"""
GMT and GMX file I/O module for gene set / gene program data.

This module provides functions to read and write gene sets in the standard
GMT (Gene Matrix Transposed) and GMX (Gene MatriX) file formats used by
GSEA, fgsea, GSVA, AUCell, and other enrichment analysis tools.

GMT format:
    Tab-separated, one gene set per line.
    Columns: gene_set_name <TAB> description <TAB> gene1 <TAB> gene2 <TAB> ...

GMX format:
    Tab-separated, one gene set per column.
    Row 0: gene set names (header).
    Row 1: descriptions.
    Row 2+: gene symbols, columns may have different lengths (padded with empty strings).

Functions also support conversion to a binary membership DataFrame and
export of weighted gene programs where each gene carries an association weight.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def read_gmt(filepath: str) -> dict[str, list[str]]:
    """Read a GMT file and return gene sets as a dictionary.

    Each line of a GMT file has the format::

        gene_set_name<TAB>description<TAB>gene1<TAB>gene2<TAB>...

    Parameters
    ----------
    filepath : str
        Path to the GMT file.

    Returns
    -------
    dict[str, list[str]]
        Mapping from gene set name to list of gene symbols.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file contains malformed lines (fewer than 3 columns).
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"GMT file not found: {filepath}")

    gene_sets: dict[str, list[str]] = {}

    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.rstrip("\n\r")
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 3:
                raise ValueError(
                    f"Malformed GMT line {line_no}: expected at least 3 tab-separated "
                    f"columns (name, description, genes), got {len(parts)}."
                )

            name = parts[0].strip()
            # parts[1] is the description -- skip it for the gene list
            genes = [g.strip() for g in parts[2:] if g.strip()]

            if name in gene_sets:
                logger.warning(
                    "Duplicate gene set name '%s' at line %d; "
                    "merging genes into existing entry.",
                    name,
                    line_no,
                )
                existing = set(gene_sets[name])
                for gene in genes:
                    if gene not in existing:
                        gene_sets[name].append(gene)
                        existing.add(gene)
            else:
                gene_sets[name] = genes

    logger.info(
        "Read %d gene sets from %s (total %d unique genes).",
        len(gene_sets),
        filepath,
        len({g for gs in gene_sets.values() for g in gs}),
    )
    return gene_sets


def write_gmt(
    gene_programs: dict[str, list[str]],
    filepath: str,
    descriptions: dict[str, str] | None = None,
) -> None:
    """Write gene programs to a GMT file.

    Parameters
    ----------
    gene_programs : dict[str, list[str]]
        Mapping from program name to list of gene symbols.
    filepath : str
        Output file path.
    descriptions : dict[str, str] | None, optional
        Mapping from program name to description string.
        Programs without an entry get an empty description field.

    Raises
    ------
    ValueError
        If *gene_programs* is empty.
    """
    if not gene_programs:
        raise ValueError("gene_programs is empty; nothing to write.")

    if descriptions is None:
        descriptions = {}

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        for name, genes in gene_programs.items():
            desc = descriptions.get(name, "")
            writer.writerow([name, desc] + list(genes))

    logger.info(
        "Wrote %d gene programs (%d total genes) to %s.",
        len(gene_programs),
        sum(len(g) for g in gene_programs.values()),
        filepath,
    )


def read_gmx(filepath: str) -> dict[str, list[str]]:
    """Read a GMX file and return gene sets as a dictionary.

    GMX format stores one gene set per column::

        set1_name   set2_name   ...
        set1_desc   set2_desc   ...
        gene1_1     gene1_2     ...
        gene2_1     gene2_2     ...
        ...         ...

    Columns may have different numbers of genes; empty trailing cells are
    ignored.

    Parameters
    ----------
    filepath : str
        Path to the GMX file.

    Returns
    -------
    dict[str, list[str]]
        Mapping from gene set name to list of gene symbols.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file has fewer than 2 rows (missing header or description).
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"GMX file not found: {filepath}")

    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        rows: list[list[str]] = [row for row in reader if row]

    if len(rows) < 2:
        raise ValueError(
            f"GMX file {filepath} has fewer than 2 rows; "
            "expected at least a header row and a description row."
        )

    names = rows[0]
    # Row 1 = descriptions (not needed for the gene list output)
    gene_rows = rows[2:]

    gene_sets: dict[str, list[str]] = {}
    for col_idx, name in enumerate(names):
        name = name.strip()
        if not name:
            continue
        genes: list[str] = []
        for row in gene_rows:
            if col_idx < len(row):
                gene = row[col_idx].strip()
                if gene:
                    genes.append(gene)
        gene_sets[name] = genes

    logger.info(
        "Read %d gene sets from GMX file %s.",
        len(gene_sets),
        filepath,
    )
    return gene_sets


def write_gmx(
    gene_programs: dict[str, list[str]],
    filepath: str,
) -> None:
    """Write gene programs to a GMX file.

    Parameters
    ----------
    gene_programs : dict[str, list[str]]
        Mapping from program name to list of gene symbols.
    filepath : str
        Output file path.

    Raises
    ------
    ValueError
        If *gene_programs* is empty.
    """
    if not gene_programs:
        raise ValueError("gene_programs is empty; nothing to write.")

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    names = list(gene_programs.keys())
    gene_lists = [gene_programs[n] for n in names]
    max_len = max(len(gl) for gl in gene_lists) if gene_lists else 0

    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")

        # Header row: gene set names
        writer.writerow(names)

        # Description row (empty descriptions)
        writer.writerow([""] * len(names))

        # Gene rows
        for row_idx in range(max_len):
            row: list[str] = []
            for gl in gene_lists:
                if row_idx < len(gl):
                    row.append(gl[row_idx])
                else:
                    row.append("")
            writer.writerow(row)

    logger.info(
        "Wrote %d gene programs to GMX file %s.",
        len(gene_programs),
        filepath,
    )


def programs_to_df(gene_programs: dict[str, list[str]]) -> pd.DataFrame:
    """Convert gene programs to a binary membership DataFrame.

    The resulting DataFrame has genes as rows and programs as columns.
    A cell is 1 if the gene belongs to the program, 0 otherwise.

    Parameters
    ----------
    gene_programs : dict[str, list[str]]
        Mapping from program name to list of gene symbols.

    Returns
    -------
    pd.DataFrame
        Binary membership matrix of shape (n_genes, n_programs) with
        integer dtype. Index is gene symbols, columns are program names.

    Raises
    ------
    ValueError
        If *gene_programs* is empty.
    """
    if not gene_programs:
        raise ValueError("gene_programs is empty; cannot create DataFrame.")

    all_genes: set[str] = set()
    for genes in gene_programs.values():
        all_genes.update(genes)

    sorted_genes = sorted(all_genes)
    sorted_programs = list(gene_programs.keys())

    gene_index: dict[str, int] = {g: i for i, g in enumerate(sorted_genes)}

    data: list[list[int]] = [[0] * len(sorted_programs) for _ in range(len(sorted_genes))]

    for prog_idx, prog_name in enumerate(sorted_programs):
        for gene in gene_programs[prog_name]:
            row_idx = gene_index[gene]
            data[row_idx][prog_idx] = 1

    df = pd.DataFrame(
        data,
        index=sorted_genes,
        columns=sorted_programs,
        dtype=int,
    )
    df.index.name = "gene"

    return df


def weighted_programs_to_gmt(
    gene_programs: dict[str, list[tuple[str, float]]],
    filepath: str,
) -> None:
    """Write weighted gene programs to a GMT-like file.

    Each gene entry is stored as ``gene,weight`` within the tab-separated
    columns. This is a common extension for tools that support weighted
    gene sets (e.g., weighted fgsea, AUCell with weights).

    The output format is::

        program_name <TAB> na <TAB> gene1,0.95 <TAB> gene2,0.87 <TAB> ...

    Parameters
    ----------
    gene_programs : dict[str, list[tuple[str, float]]]
        Mapping from program name to list of (gene_symbol, weight) tuples.
        Weights are typically in [0, 1] but this is not enforced.
    filepath : str
        Output file path.

    Raises
    ------
    ValueError
        If *gene_programs* is empty.
    """
    if not gene_programs:
        raise ValueError("gene_programs is empty; nothing to write.")

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        for name, gene_weight_pairs in gene_programs.items():
            encoded_genes = [f"{gene},{weight:.6f}" for gene, weight in gene_weight_pairs]
            writer.writerow([name, "na"] + encoded_genes)

    logger.info(
        "Wrote %d weighted gene programs to %s.",
        len(gene_programs),
        filepath,
    )
