#!/usr/bin/env python3
"""Run claim-safe validation suite for bulk dynamic pathway outputs.

This script consolidates four evidence tracks:
1) Stability (bootstrap-style variant runs vs baseline)
2) Hyperparameter sensitivity (variant runs vs baseline)
3) Robustness (composition/batch-like perturbation runs vs baseline)
4) External reproducibility (independent cohorts/runs)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from npathway.validation import (
    analyze_perturbation_robustness,
    summarize_bootstrap_stability,
    summarize_external_reproducibility,
    summarize_hyperparameter_sensitivity,
)


def _parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Expected 'label=path' format, got: {value}"
        )
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError(
            f"Expected non-empty 'label=path', got: {value}"
        )
    return label, Path(path)


def _load_program_enrichment_table(results_dir: Path) -> pd.DataFrame:
    """Build table with required columns for external reproducibility module."""
    genes_path = results_dir / "program_gene_lists.csv"
    gsea_path = results_dir / "enrichment_gsea_with_claim_gates.csv"
    if not genes_path.exists():
        raise FileNotFoundError(f"Missing required file: {genes_path}")
    if not gsea_path.exists():
        raise FileNotFoundError(f"Missing required file: {gsea_path}")

    genes_df = pd.read_csv(genes_path)
    gsea_df = pd.read_csv(gsea_path)
    if not {"program", "genes"}.issubset(genes_df.columns):
        raise ValueError(
            f"{genes_path} must include columns ['program', 'genes']"
        )
    req_gsea = {"program", "nes", "fdr"}
    if not req_gsea.issubset(gsea_df.columns):
        raise ValueError(
            f"{gsea_path} must include columns {sorted(req_gsea)}"
        )

    merged = gsea_df.merge(genes_df[["program", "genes"]], on="program", how="left")
    if "claim_supported" not in merged.columns:
        merged["claim_supported"] = merged["fdr"] <= 0.05
    return merged.loc[:, ["program", "genes", "nes", "fdr", "claim_supported"]].copy()


def _write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True, help="Baseline result directory.")
    parser.add_argument(
        "--bootstrap-variant",
        action="append",
        default=[],
        help="Bootstrap-like variant results dir (repeatable).",
    )
    parser.add_argument(
        "--sensitivity-variant",
        action="append",
        default=[],
        help="Hyperparameter variant results dir (repeatable).",
    )
    parser.add_argument(
        "--robustness-run",
        action="append",
        default=[],
        type=_parse_labeled_path,
        help="Robustness run in 'label=path' format (repeatable). Include baseline label too.",
    )
    parser.add_argument(
        "--external-run",
        action="append",
        default=[],
        type=_parse_labeled_path,
        help="External cohort/run in 'label=path' format (repeatable).",
    )
    parser.add_argument(
        "--robustness-baseline-label",
        default="baseline",
        help="Baseline label for robustness track.",
    )
    parser.add_argument(
        "--external-top-k",
        type=int,
        default=20,
        help="Top-K programs for external reproducibility comparison.",
    )
    parser.add_argument(
        "--external-jaccard-threshold",
        type=float,
        default=0.1,
        help="Replication threshold for best-match Jaccard.",
    )
    parser.add_argument(
        "--external-strict-baseline-fdr",
        type=float,
        default=None,
        help="Optional strict baseline FDR gate for external replication claims.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for validation artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_dir = Path(args.baseline_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"baseline_dir": str(baseline_dir)}

    # Track 1: Stability
    bootstrap_variants = [Path(p) for p in args.bootstrap_variant]
    if bootstrap_variants:
        tables = summarize_bootstrap_stability(
            baseline_results_dir=baseline_dir,
            bootstrap_results_dirs=bootstrap_variants,
        )
        _write_df(
            tables.summary,
            output_dir / "stability_bootstrap_summary.csv",
        )
        _write_df(
            tables.pairwise_jaccard_long,
            output_dir / "stability_bootstrap_pairwise_jaccard_long.csv",
        )
        summary["stability"] = {
            "n_variants": int(len(tables.summary)),
            "mean_symmetric_best_jaccard": float(
                pd.to_numeric(
                    tables.summary.get("symmetric_mean_best_jaccard", pd.Series(dtype=float)),
                    errors="coerce",
                ).mean()
            )
            if not tables.summary.empty
            else np.nan,
        }

    # Track 2: Sensitivity
    sensitivity_variants = [Path(p) for p in args.sensitivity_variant]
    if sensitivity_variants:
        tables = summarize_hyperparameter_sensitivity(
            baseline_results_dir=baseline_dir,
            variant_results_dirs=sensitivity_variants,
        )
        _write_df(
            tables.summary,
            output_dir / "sensitivity_hyperparameter_summary.csv",
        )
        _write_df(
            tables.pairwise_jaccard_long,
            output_dir / "sensitivity_hyperparameter_pairwise_jaccard_long.csv",
        )
        summary["sensitivity"] = {
            "n_rows": int(len(tables.summary)),
            "n_unique_parameters": int(
                tables.summary["parameter"].nunique()
                if "parameter" in tables.summary.columns
                else 0
            ),
        }

    # Track 3: Robustness
    if args.robustness_run:
        run_map = {label: path for label, path in args.robustness_run}
        robustness = analyze_perturbation_robustness(
            result_dirs_by_label=run_map,
            baseline_label=args.robustness_baseline_label,
        )
        _write_df(robustness["summary"], output_dir / "robustness_summary.csv")
        _write_df(
            robustness["program_matches"],
            output_dir / "robustness_program_matches.csv",
        )
        _write_df(
            robustness["context_pairs"],
            output_dir / "robustness_context_pairs.csv",
        )
        summary["robustness"] = {
            "n_perturbations": int(len(robustness["summary"])),
            "mean_best_match_program_jaccard": float(
                pd.to_numeric(
                    robustness["summary"].get(
                        "mean_best_match_program_jaccard", pd.Series(dtype=float)
                    ),
                    errors="coerce",
                ).mean()
            )
            if not robustness["summary"].empty
            else np.nan,
        }

    # Track 4: External reproducibility
    if args.external_run:
        external_tables: dict[str, pd.DataFrame] = {}
        for label, run_dir in args.external_run:
            external_tables[label] = _load_program_enrichment_table(run_dir)

        external = summarize_external_reproducibility(
            cohorts=external_tables,
            top_k=max(1, int(args.external_top_k)),
            jaccard_threshold=float(args.external_jaccard_threshold),
            strict_baseline_fdr=args.external_strict_baseline_fdr,
        )
        _write_df(
            external["pairwise_metrics"],
            output_dir / "external_pairwise_metrics.csv",
        )
        for pair_name, pair_df in external["best_match_tables"].items():
            safe = pair_name.replace("/", "_").replace("\\", "_")
            _write_df(
                pair_df,
                output_dir / "external_best_match_tables" / f"{safe}.csv",
            )
        (output_dir / "external_summary.json").write_text(
            json.dumps(external["summary"], indent=2),
            encoding="utf-8",
        )
        summary["external"] = external["summary"]

    # Global summary
    (output_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print("Validation suite completed.")
    print(f"- output_dir: {output_dir}")
    print(f"- summary: {output_dir / 'validation_summary.json'}")


if __name__ == "__main__":
    main()

