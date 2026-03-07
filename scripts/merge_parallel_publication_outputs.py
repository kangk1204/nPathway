#!/usr/bin/env python3
"""Merge per-dataset publication outputs into one consolidated outdir."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CSV_TABLES = [
    "expanded_case_studies.csv",
    "expanded_dataset_provenance.csv",
    "expanded_discovery_raw.csv",
    "expanded_fairness_manifest.csv",
    "expanded_method_runtime.csv",
    "expanded_method_summary.csv",
    "expanded_power_raw.csv",
    "expanded_primary_leaderboard.csv",
    "expanded_primary_wilcoxon_vs_cnmf.csv",
    "expanded_recovery_raw.csv",
    "expanded_seed_summary.csv",
    "expanded_skipped_methods.csv",
    "expanded_track_leaderboard.csv",
    "expanded_tuning_log.csv",
    "expanded_wilcoxon_vs_cnmf.csv",
    "expanded_wilcoxon_vs_cnmf_de_novo.csv",
    "expanded_wilcoxon_vs_cnmf_reference_guided.csv",
]


def _find_dataset_dirs(base_dir: Path) -> list[Path]:
    out: list[Path] = []
    for child in sorted(base_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name == "logs":
            continue
        tables = child / "tables"
        if (tables / "expanded_seed_summary.csv").exists():
            out.append(child)
    return out


def _merge_csv_table(dataset_dirs: list[Path], table_name: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for d in dataset_dirs:
        p = d / "tables" / table_name
        if not p.exists():
            continue
        df = pd.read_csv(p)
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    merged = pd.concat(parts, axis=0, ignore_index=True)
    if table_name == "expanded_fairness_manifest.csv":
        merged = merged.drop_duplicates(subset=["method"], keep="first").reset_index(drop=True)
    if table_name == "expanded_dataset_provenance.csv":
        merged = merged.drop_duplicates(subset=["dataset"], keep="first").reset_index(drop=True)
    return merged


def _collect_protocols(
    dataset_dirs: list[Path],
) -> tuple[list[str], list[int], bool, list[dict]]:
    """Read per-dataset manifests and aggregate protocol metadata.

    Returns
    -------
    tuple of (datasets, seeds, strict_datasets, dataset_provenance)
    strict_datasets is True only if ALL per-dataset manifests report True.
    If any manifest is missing or reports False, the merged value is False.
    """
    datasets: list[str] = []
    seeds: set[int] = set()
    strict_flags: list[bool] = []
    dataset_provenance: list[dict] = []
    for d in dataset_dirs:
        p = d / "tables" / "expanded_protocol_manifest.json"
        if not p.exists():
            # Conservative: treat missing manifest as non-strict
            strict_flags.append(False)
            continue
        payload = json.loads(p.read_text(encoding="utf-8"))
        proto = payload.get("protocol", {})
        ds = proto.get("datasets", [])
        sd = proto.get("seeds", [])
        # Read actual strict_datasets value; default False if absent
        strict_flags.append(bool(proto.get("strict_datasets", False)))
        prov = payload.get("dataset_provenance", [])
        for x in ds:
            if x not in datasets:
                datasets.append(x)
        for s in sd:
            try:
                seeds.add(int(s))
            except Exception:
                continue
        for row in prov:
            if isinstance(row, dict):
                dataset_provenance.append(row)
    # strict only if every run was strict (conservative AND)
    strict_datasets: bool = bool(strict_flags) and all(strict_flags)
    seen_ds: set[str] = set()
    deduped: list[dict] = []
    for row in dataset_provenance:
        ds_name = str(row.get("dataset", ""))
        if ds_name in seen_ds:
            continue
        seen_ds.add(ds_name)
        deduped.append(row)
    return datasets, sorted(seeds), strict_datasets, deduped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help="Parent directory containing per-dataset output dirs.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Merged output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    outdir = args.outdir.resolve()
    tables_out = outdir / "tables"
    figures_out = outdir / "figures"
    tables_out.mkdir(parents=True, exist_ok=True)
    figures_out.mkdir(parents=True, exist_ok=True)

    dataset_dirs = _find_dataset_dirs(base_dir)
    if not dataset_dirs:
        raise FileNotFoundError(f"No per-dataset outputs found under {base_dir}")

    for table_name in CSV_TABLES:
        merged = _merge_csv_table(dataset_dirs, table_name)
        if merged.empty:
            continue
        merged.to_csv(tables_out / table_name, index=False)

    datasets, seeds, strict_datasets, dataset_provenance = _collect_protocols(dataset_dirs)
    protocol = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "source_base_dir": str(base_dir),
        "protocol": {
            "datasets": datasets,
            "seeds": seeds,
            "strict_datasets": strict_datasets,
            "note": "Merged from per-dataset parallel runs.",
        },
        "dataset_provenance": dataset_provenance,
    }
    (tables_out / "expanded_protocol_manifest.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )

    print(f"Merged {len(dataset_dirs)} dataset outputs into {outdir}")


if __name__ == "__main__":
    main()
