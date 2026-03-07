"""Convert 10x Genomics input files into .h5ad for nPathway scRNA workflows."""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd
import scanpy as sc
from scipy.io import mmread
from scipy import sparse


def _dep_status() -> dict[str, bool]:
    return {
        name: bool(importlib.util.find_spec(name))
        for name in ("scanpy", "anndata", "scipy", "h5py")
    }


def _write_status_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_table(path: Path, sep: str | None) -> pd.DataFrame:
    if sep is not None:
        return pd.read_csv(path, sep=sep)
    lower = path.name.lower()
    if lower.endswith(".tsv") or lower.endswith(".txt") or lower.endswith(".tsv.gz") or lower.endswith(".txt.gz"):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _infer_source_type(path: Path) -> str:
    if path.is_dir():
        return "matrix_dir"
    lower = path.name.lower()
    if lower.endswith(".h5") or lower.endswith(".h5.gz") or lower.endswith(".hdf5"):
        return "tenx_h5"
    raise ValueError(
        f"Could not infer 10x source type for '{path}'. Use a 10x matrix directory or an .h5 file."
    )


def _load_10x_source(
    *,
    input_path: Path,
    source_type: str,
    var_names: str,
    gex_only: bool,
    make_var_names_unique: bool,
) -> ad.AnnData:
    if source_type == "matrix_dir":
        adata = _read_10x_mtx_manual(input_path, var_names=var_names)
    elif source_type == "tenx_h5":
        adata = sc.read_10x_h5(str(input_path), gex_only=gex_only)
    else:
        raise ValueError(f"Unsupported source_type '{source_type}'.")
    if make_var_names_unique:
        adata.var_names_make_unique()
    return adata


def _resolve_existing_file(base_dir: Path, candidates: tuple[str, ...]) -> Path:
    for name in candidates:
        path = base_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Did not find any of the expected 10x files in {base_dir}: {', '.join(candidates)}")


def _read_tsv_auto(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", header=None, compression="infer")


def _read_10x_mtx_manual(matrix_dir: Path, *, var_names: str) -> ad.AnnData:
    matrix_path = _resolve_existing_file(matrix_dir, ("matrix.mtx.gz", "matrix.mtx"))
    barcode_path = _resolve_existing_file(matrix_dir, ("barcodes.tsv.gz", "barcodes.tsv"))
    feature_path = _resolve_existing_file(matrix_dir, ("features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv"))

    if matrix_path.suffix == ".gz":
        with gzip.open(matrix_path, "rb") as handle:
            matrix = mmread(handle)
    else:
        matrix = mmread(str(matrix_path))
    matrix = sparse.csr_matrix(matrix)

    barcodes = _read_tsv_auto(barcode_path)
    features = _read_tsv_auto(feature_path)
    if barcodes.shape[1] < 1:
        raise ValueError(f"Barcode file '{barcode_path}' is malformed.")
    if features.shape[1] < 2:
        raise ValueError(f"Feature file '{feature_path}' is malformed.")

    barcode_index = barcodes.iloc[:, 0].astype(str)
    gene_ids = features.iloc[:, 0].astype(str)
    gene_symbols = features.iloc[:, 1].astype(str)
    feature_types = (
        features.iloc[:, 2].astype(str)
        if features.shape[1] >= 3
        else pd.Series(["Gene Expression"] * len(features))
    )
    if matrix.shape != (len(gene_ids), len(barcode_index)):
        raise ValueError(
            "10x matrix dimensions do not match features/barcodes. "
            f"Observed matrix={matrix.shape}, features={len(gene_ids)}, barcodes={len(barcode_index)}."
        )

    var_index = gene_symbols if var_names == "gene_symbols" else gene_ids
    var_index = pd.Index(var_index.astype(str), name=None)
    adata = ad.AnnData(
        X=matrix.T.tocsr(),
        obs=pd.DataFrame(index=pd.Index(barcode_index.astype(str), name=None)),
        var=pd.DataFrame(
            {
                "gene_ids": gene_ids.to_numpy(),
                "gene_symbols": gene_symbols.to_numpy(),
                "feature_types": feature_types.to_numpy(),
            },
            index=var_index,
        ),
    )
    adata.var_names.name = None
    adata.obs_names.name = None
    return adata


def _merge_obs_metadata(
    adata: ad.AnnData,
    *,
    obs_csv: Path | None,
    barcode_col: str,
    sep: str | None,
) -> ad.AnnData:
    if obs_csv is None:
        return adata
    obs_df = _load_table(obs_csv, sep=sep)
    if barcode_col not in obs_df.columns:
        raise KeyError(
            f"Metadata file '{obs_csv}' is missing barcode column '{barcode_col}'. "
            f"Available columns: {list(obs_df.columns)}"
        )
    obs_df = obs_df.copy()
    if obs_df[barcode_col].isna().any():
        raise ValueError(f"Metadata file '{obs_csv}' contains missing barcode values.")
    obs_df[barcode_col] = obs_df[barcode_col].astype(str)
    if obs_df[barcode_col].duplicated().any():
        dupes = sorted(set(obs_df.loc[obs_df[barcode_col].duplicated(), barcode_col].astype(str)))
        raise ValueError("Cell metadata barcode values must be unique. Duplicates: " + ", ".join(dupes))
    obs_df = obs_df.set_index(barcode_col)
    overlap = adata.obs_names.intersection(obs_df.index)
    if overlap.empty:
        raise ValueError(
            f"No overlapping barcodes between 10x data and metadata file '{obs_csv}'."
        )
    merged = adata.obs.join(obs_df, how="left")
    adata = adata.copy()
    adata.obs = merged
    return adata


def _prefix_obs_names(adata: ad.AnnData, sample_id: str | None, *, prefix_barcodes: bool) -> ad.AnnData:
    if not prefix_barcodes or not sample_id:
        return adata
    adata = adata.copy()
    adata.obs_names = pd.Index([f"{sample_id}:{barcode}" for barcode in adata.obs_names.astype(str)])
    return adata


def _copy_manifest_row_to_obs(adata: ad.AnnData, row: pd.Series) -> ad.AnnData:
    adata = adata.copy()
    for column, value in row.items():
        if column in {"input_path", "source_type"}:
            continue
        adata.obs[column] = str(value) if pd.notna(value) else ""
    return adata


def _convert_single(
    *,
    input_path: Path,
    output_h5ad: Path,
    sample_id: str | None,
    sample_col: str,
    obs_csv: Path | None,
    barcode_col: str,
    sep: str | None,
    var_names: str,
    gex_only: bool,
    make_var_names_unique: bool,
    prefix_barcodes: bool,
) -> ad.AnnData:
    source_type = _infer_source_type(input_path)
    adata = _load_10x_source(
        input_path=input_path,
        source_type=source_type,
        var_names=var_names,
        gex_only=gex_only,
        make_var_names_unique=make_var_names_unique,
    )
    adata = _merge_obs_metadata(adata, obs_csv=obs_csv, barcode_col=barcode_col, sep=sep)
    if sample_id:
        adata.obs[sample_col] = str(sample_id)
    adata.obs["npathway_source_type"] = source_type
    adata.obs["npathway_input_path"] = str(input_path)
    adata = _prefix_obs_names(adata, sample_id, prefix_barcodes=prefix_barcodes)
    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_h5ad)
    return adata


def _convert_manifest(
    *,
    manifest_path: Path,
    output_h5ad: Path,
    sample_col: str,
    sep: str | None,
    var_names: str,
    gex_only: bool,
    make_var_names_unique: bool,
    prefix_barcodes: bool,
) -> ad.AnnData:
    manifest = _load_table(manifest_path, sep=sep)
    if "input_path" not in manifest.columns:
        raise KeyError("Manifest must contain an 'input_path' column.")
    if sample_col not in manifest.columns:
        raise KeyError(
            f"Manifest must contain a '{sample_col}' column so each input gets a stable sample ID."
        )
    if manifest["input_path"].isna().any() or manifest[sample_col].isna().any():
        raise ValueError("Manifest input_path and sample columns cannot contain missing values.")
    if manifest[sample_col].astype(str).duplicated().any():
        dupes = sorted(set(manifest.loc[manifest[sample_col].astype(str).duplicated(), sample_col].astype(str)))
        raise ValueError("Manifest sample IDs must be unique. Duplicates: " + ", ".join(dupes))

    adatas: list[ad.AnnData] = []
    keys: list[str] = []
    for _, row in manifest.iterrows():
        input_path = Path(str(row["input_path"]))
        source_type = str(row["source_type"]) if "source_type" in manifest.columns and pd.notna(row.get("source_type")) and str(row.get("source_type")).strip() else _infer_source_type(input_path)
        sample_id = str(row[sample_col])
        adata = _load_10x_source(
            input_path=input_path,
            source_type=source_type,
            var_names=var_names,
            gex_only=gex_only,
            make_var_names_unique=make_var_names_unique,
        )
        adata = _copy_manifest_row_to_obs(adata, row)
        adata.obs["npathway_source_type"] = source_type
        adata.obs["npathway_input_path"] = str(input_path)
        adata = _prefix_obs_names(adata, sample_id, prefix_barcodes=prefix_barcodes)
        adatas.append(adata)
        keys.append(sample_id)

    combined = ad.concat(adatas, join="outer", merge="same", fill_value=0)
    if make_var_names_unique:
        combined.var_names_make_unique()
    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    combined.write_h5ad(output_h5ad)
    return combined


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument("--tenx-h5", default=None, help="Path to a single 10x .h5 file.")
    mode.add_argument("--matrix-dir", default=None, help="Path to a single 10x matrix directory containing matrix.mtx, barcodes.tsv, and features.tsv.")
    mode.add_argument("--manifest", default=None, help="CSV/TSV manifest describing multiple 10x inputs. Required columns: input_path and sample_id (or your chosen --sample-col).")
    parser.add_argument("--output-h5ad", default=None, help="Output .h5ad path.")
    parser.add_argument("--sample-id", default=None, help="Optional sample ID to add in single-input mode.")
    parser.add_argument("--sample-col", default="sample_id", help="Column name used to store sample IDs in adata.obs.")
    parser.add_argument("--obs-csv", default=None, help="Optional per-cell metadata CSV/TSV for single-input mode.")
    parser.add_argument("--barcode-col", default="barcode", help="Barcode column in --obs-csv.")
    parser.add_argument("--sep", default=None, help="Delimiter override for manifest or metadata tables.")
    parser.add_argument("--var-names", default="gene_symbols", choices=["gene_symbols", "gene_ids"], help="Variable names to use for matrix-dir inputs.")
    parser.add_argument("--gex-only", action=argparse.BooleanOptionalAction, default=True, help="For 10x .h5 inputs, keep only gene-expression features (default: true).")
    parser.add_argument("--make-var-names-unique", action=argparse.BooleanOptionalAction, default=True, help="Make gene names unique in the converted AnnData (default: true).")
    parser.add_argument("--prefix-barcodes", action=argparse.BooleanOptionalAction, default=True, help="Prefix cell barcodes with the sample ID to keep obs_names unique (default: true).")
    parser.add_argument("--check-only", action="store_true", help="Only report Python dependency readiness for 10x conversion.")
    parser.add_argument("--status-json", default=None, help="Optional JSON path for the readiness/conversion report.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    deps = _dep_status()
    ready = all(deps.values())

    status_json = Path(args.status_json) if args.status_json else None
    base_payload: dict[str, Any] = {
        "dependencies": deps,
        "ready": ready,
        "mode": "check_only" if args.check_only else ("manifest" if args.manifest else "single"),
    }
    if status_json is not None:
        _write_status_json(status_json, base_payload)

    print("nPathway 10x conversion readiness")
    for name, flag in deps.items():
        print(f"- {name}: {'OK' if flag else 'MISSING'}")
    print(f"- ready: {ready}")

    if args.check_only:
        return
    if not ready:
        raise RuntimeError("10x conversion requires scanpy, anndata, scipy, and h5py.")

    if args.output_h5ad is None:
        raise ValueError("--output-h5ad is required unless --check-only is used.")
    output_h5ad = Path(args.output_h5ad)

    if args.manifest is not None:
        adata = _convert_manifest(
            manifest_path=Path(args.manifest),
            output_h5ad=output_h5ad,
            sample_col=str(args.sample_col),
            sep=args.sep,
            var_names=str(args.var_names),
            gex_only=bool(args.gex_only),
            make_var_names_unique=bool(args.make_var_names_unique),
            prefix_barcodes=bool(args.prefix_barcodes),
        )
        mode = "manifest"
        input_desc = str(args.manifest)
    else:
        source = args.tenx_h5 or args.matrix_dir
        if source is None:
            raise ValueError("Provide one of --tenx-h5, --matrix-dir, or --manifest.")
        adata = _convert_single(
            input_path=Path(source),
            output_h5ad=output_h5ad,
            sample_id=args.sample_id,
            sample_col=str(args.sample_col),
            obs_csv=None if args.obs_csv is None else Path(args.obs_csv),
            barcode_col=str(args.barcode_col),
            sep=args.sep,
            var_names=str(args.var_names),
            gex_only=bool(args.gex_only),
            make_var_names_unique=bool(args.make_var_names_unique),
            prefix_barcodes=bool(args.prefix_barcodes),
        )
        mode = "single"
        input_desc = str(source)

    payload = {
        **base_payload,
        "converted": True,
        "input": input_desc,
        "output_h5ad": str(output_h5ad),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "obs_columns": list(map(str, adata.obs.columns)),
    }
    if status_json is not None:
        _write_status_json(status_json, payload)

    print("nPathway 10x conversion completed.")
    print(f"- mode: {mode}")
    print(f"- output_h5ad: {output_h5ad}")
    print(f"- n_cells: {adata.n_obs}")
    print(f"- n_genes: {adata.n_vars}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "Hint: use `npathway-convert-10x --check-only` first, then convert raw 10x data into .h5ad before running npathway-scrna-easy.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
