"""Utilities for external reproducibility across independent cohorts/runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REQUIRED_COLUMNS = {"program", "genes", "nes", "fdr"}


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "1": True,
                "yes": True,
                "y": True,
                "false": False,
                "0": False,
                "no": False,
                "n": False,
            }
        )
        .fillna(False)
        .astype(bool)
    )


def _parse_gene_set(value: Any) -> set[str]:
    if isinstance(value, (set, frozenset)):
        return {str(g).strip() for g in value if str(g).strip()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {str(g).strip() for g in value if str(g).strip()}
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()

    text = str(value).strip()
    if not text:
        return set()
    return {token.strip() for token in text.split(",") if token.strip()}


def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _load_program_table(
    table: pd.DataFrame | str | Path,
    *,
    label: str,
) -> pd.DataFrame:
    if isinstance(table, pd.DataFrame):
        df = table.copy()
    else:
        df = pd.read_csv(Path(table))

    missing = sorted(_REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {', '.join(missing)}"
        )

    out = df.copy()
    out["program"] = out["program"].astype(str)
    out["fdr"] = pd.to_numeric(out["fdr"], errors="coerce")
    out["nes"] = pd.to_numeric(out["nes"], errors="coerce")
    out["gene_set"] = out["genes"].apply(_parse_gene_set)

    if "claim_supported" not in out.columns:
        out["claim_supported"] = out["fdr"] <= 0.05
    out["claim_supported"] = _coerce_bool_series(out["claim_supported"])
    return out


def _nes_sign(value: float) -> int:
    if not np.isfinite(value):
        return 0
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


_BEST_MATCH_COLUMNS = [
    "baseline_program",
    "baseline_fdr",
    "baseline_nes",
    "baseline_claim_supported",
    "baseline_passes_strict_fdr",
    "replicate_program",
    "replicate_fdr",
    "replicate_nes",
    "replicate_claim_supported",
    "best_jaccard",
    "replicated",
    "direction_concordant",
    "claim_supported_replicated",
]


def compare_external_reproducibility(
    baseline: pd.DataFrame | str | Path,
    replicate: pd.DataFrame | str | Path,
    *,
    top_k: int = 20,
    jaccard_threshold: float = 0.1,
    strict_baseline_fdr: float | None = None,
    baseline_label: str = "baseline",
    replicate_label: str = "replicate",
) -> dict[str, Any]:
    """Compare reproducibility between one baseline and one replicate cohort/run.

    Programs are ranked by baseline FDR, then the top ``top_k`` programs are
    matched to the replicate cohort by maximum Jaccard overlap.
    """
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if not 0.0 <= jaccard_threshold <= 1.0:
        raise ValueError("jaccard_threshold must be in [0, 1]")
    if strict_baseline_fdr is not None and strict_baseline_fdr < 0.0:
        raise ValueError("strict_baseline_fdr must be >= 0 when provided")

    base_df = _load_program_table(baseline, label=baseline_label)
    repl_df = _load_program_table(replicate, label=replicate_label)

    top_df = (
        base_df.sort_values(["fdr", "program"], ascending=[True, True], na_position="last")
        .head(min(top_k, len(base_df)))
        .reset_index(drop=True)
    )

    repl_records = [
        (
            str(row.program),
            set(row.gene_set),
            float(row.nes),
            float(row.fdr),
            bool(row.claim_supported),
        )
        for row in repl_df.itertuples(index=False)
    ]

    rows: list[dict[str, Any]] = []
    for row in top_df.itertuples(index=False):
        base_program = str(row.program)
        base_genes = set(row.gene_set)
        base_nes = float(row.nes)
        base_fdr = float(row.fdr)
        base_claim = bool(row.claim_supported)

        best_program = ""
        best_nes = float("nan")
        best_fdr = float("nan")
        best_claim = False
        best_jaccard = -1.0

        for repl_program, repl_genes, repl_nes, repl_fdr, repl_claim in repl_records:
            jac = _jaccard(base_genes, repl_genes)
            is_better = jac > best_jaccard + 1e-12
            is_tie = np.isclose(jac, best_jaccard, atol=1e-12)
            if is_better or (
                is_tie
                and (
                    (np.isfinite(repl_fdr) and not np.isfinite(best_fdr))
                    or (
                        np.isfinite(repl_fdr)
                        and np.isfinite(best_fdr)
                        and repl_fdr < best_fdr - 1e-12
                    )
                    or (np.isclose(repl_fdr, best_fdr, atol=1e-12) and repl_program < best_program)
                )
            ):
                best_program = repl_program
                best_nes = repl_nes
                best_fdr = repl_fdr
                best_claim = repl_claim
                best_jaccard = jac

        if best_jaccard < 0:
            best_jaccard = 0.0

        replicated = best_jaccard >= jaccard_threshold
        baseline_passes_strict = (
            True
            if strict_baseline_fdr is None
            else bool(np.isfinite(base_fdr) and base_fdr <= strict_baseline_fdr)
        )

        direction_concordant: bool | float = float("nan")
        if replicated:
            base_sign = _nes_sign(base_nes)
            repl_sign = _nes_sign(best_nes)
            if base_sign != 0 and repl_sign != 0:
                direction_concordant = base_sign == repl_sign

        claim_supported_replicated = bool(
            base_claim
            and baseline_passes_strict
            and replicated
            and best_claim
        )

        rows.append(
            {
                "baseline_program": base_program,
                "baseline_fdr": base_fdr,
                "baseline_nes": base_nes,
                "baseline_claim_supported": base_claim,
                "baseline_passes_strict_fdr": baseline_passes_strict,
                "replicate_program": best_program,
                "replicate_fdr": best_fdr,
                "replicate_nes": best_nes,
                "replicate_claim_supported": best_claim,
                "best_jaccard": float(best_jaccard),
                "replicated": bool(replicated),
                "direction_concordant": direction_concordant,
                "claim_supported_replicated": claim_supported_replicated,
            }
        )

    best_match_table = pd.DataFrame(rows, columns=_BEST_MATCH_COLUMNS)

    n_top = int(len(best_match_table))
    top_rate = float(best_match_table["replicated"].mean()) if n_top > 0 else 0.0

    claim_mask = (
        best_match_table["baseline_claim_supported"]
        & best_match_table["baseline_passes_strict_fdr"]
    )
    n_claim = int(claim_mask.sum())
    if n_claim > 0:
        claim_rate = float(best_match_table.loc[claim_mask, "claim_supported_replicated"].mean())
    else:
        claim_rate = 0.0

    direction_mask = best_match_table["direction_concordant"].notna()
    n_direction = int(direction_mask.sum())
    if n_direction > 0:
        direction_rate = float(best_match_table.loc[direction_mask, "direction_concordant"].mean())
    else:
        direction_rate = float("nan")

    jaccard_distribution = best_match_table["best_jaccard"].astype(float).tolist()

    return {
        "baseline_label": baseline_label,
        "replicate_label": replicate_label,
        "n_top_programs": n_top,
        "top_program_replication_rate": top_rate,
        "n_claim_supported_baseline": n_claim,
        "claim_supported_replication_rate": claim_rate,
        "n_replicated_programs": int(best_match_table["replicated"].sum()),
        "best_match_jaccard_distribution": jaccard_distribution,
        "direction_concordance_rate": direction_rate,
        "n_direction_evaluable": n_direction,
        "best_match_table": best_match_table,
    }


def summarize_external_reproducibility(
    cohorts: Mapping[str, pd.DataFrame | str | Path],
    *,
    top_k: int = 20,
    jaccard_threshold: float = 0.1,
    strict_baseline_fdr: float | None = None,
) -> dict[str, Any]:
    """Summarize external reproducibility across two or more cohorts/runs.

    Computes directed pairwise comparisons for all ordered baseline->replicate
    pairs and returns both pairwise and aggregate summaries.
    """
    if len(cohorts) < 2:
        raise ValueError("cohorts must contain at least two entries")

    loaded = {
        name: _load_program_table(table, label=name)
        for name, table in cohorts.items()
    }

    pair_rows: list[dict[str, Any]] = []
    best_match_tables: dict[str, pd.DataFrame] = {}
    all_jaccards: list[float] = []

    names = list(loaded.keys())
    for baseline_name in names:
        for replicate_name in names:
            if baseline_name == replicate_name:
                continue

            result = compare_external_reproducibility(
                loaded[baseline_name],
                loaded[replicate_name],
                top_k=top_k,
                jaccard_threshold=jaccard_threshold,
                strict_baseline_fdr=strict_baseline_fdr,
                baseline_label=baseline_name,
                replicate_label=replicate_name,
            )

            pair_key = f"{baseline_name}__vs__{replicate_name}"
            best_match_tables[pair_key] = result["best_match_table"]
            all_jaccards.extend(result["best_match_jaccard_distribution"])

            pair_rows.append(
                {
                    "baseline": baseline_name,
                    "replicate": replicate_name,
                    "n_top_programs": int(result["n_top_programs"]),
                    "top_program_replication_rate": float(result["top_program_replication_rate"]),
                    "n_claim_supported_baseline": int(result["n_claim_supported_baseline"]),
                    "claim_supported_replication_rate": float(
                        result["claim_supported_replication_rate"]
                    ),
                    "n_replicated_programs": int(result["n_replicated_programs"]),
                    "direction_concordance_rate": float(result["direction_concordance_rate"]),
                    "n_direction_evaluable": int(result["n_direction_evaluable"]),
                }
            )

    pairwise_metrics = pd.DataFrame(pair_rows)

    if pairwise_metrics.empty:
        mean_top = 0.0
        mean_claim = 0.0
        mean_direction = float("nan")
    else:
        mean_top = float(pairwise_metrics["top_program_replication_rate"].mean())
        mean_claim = float(pairwise_metrics["claim_supported_replication_rate"].mean())
        direction_vals = pairwise_metrics["direction_concordance_rate"].to_numpy(dtype=np.float64)
        finite_direction_vals = direction_vals[np.isfinite(direction_vals)]
        mean_direction = (
            float(np.mean(finite_direction_vals))
            if finite_direction_vals.size > 0
            else float("nan")
        )

    jaccard_arr = np.asarray(all_jaccards, dtype=np.float64)
    finite_jaccard = jaccard_arr[np.isfinite(jaccard_arr)]
    if finite_jaccard.size > 0:
        jaccard_mean = float(np.mean(finite_jaccard))
        jaccard_median = float(np.median(finite_jaccard))
    else:
        jaccard_mean = 0.0
        jaccard_median = 0.0

    summary = {
        "n_cohorts": len(loaded),
        "n_pairwise_comparisons": int(len(pairwise_metrics)),
        "mean_top_program_replication_rate": mean_top,
        "mean_claim_supported_replication_rate": mean_claim,
        "mean_direction_concordance_rate": mean_direction,
        "overall_best_match_jaccard_mean": jaccard_mean,
        "overall_best_match_jaccard_median": jaccard_median,
    }

    return {
        "summary": summary,
        "pairwise_metrics": pairwise_metrics,
        "best_match_tables": best_match_tables,
        "overall_best_match_jaccard_distribution": all_jaccards,
    }


__all__ = [
    "compare_external_reproducibility",
    "summarize_external_reproducibility",
]
