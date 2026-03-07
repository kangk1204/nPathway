"""Utilities for comparing bulk dynamic result variants against a baseline.

This module provides two high-level summaries:
- bootstrap stability summary
- hyperparameter sensitivity summary

Both summaries operate directly on result directories and return tidy DataFrames
that combine program-overlap stability (Jaccard) and claim-support deltas.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


@dataclass(frozen=True)
class SensitivitySummaryTables:
    """Container for variant-level and program-pair-level summary tables.

    Attributes:
        summary: Tidy per-variant summary table for plotting/reporting.
        pairwise_jaccard_long: Long-form pairwise program Jaccard table.
    """

    summary: pd.DataFrame
    pairwise_jaccard_long: pd.DataFrame


_MEMBERSHIP_CANDIDATES: tuple[str, ...] = (
    "program_gene_membership_long.csv",
    "tables/program_gene_membership_long.csv",
    "dashboard/tables/program_gene_membership_long.csv",
    "dynamic_programs_long.csv",
)

_GSEA_CANDIDATES: tuple[str, ...] = (
    "enrichment_gsea_with_claim_gates.csv",
    "tables/enrichment_gsea_with_claim_gates.csv",
    "dashboard/tables/enrichment_gsea_with_claim_gates.csv",
)

_IGNORED_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "matrix_path",
        "metadata_path",
        "group_col",
        "group_a",
        "group_b",
        "sample_col",
        "matrix_orientation",
        "sep",
    }
)

_PAIRWISE_COLUMNS: tuple[str, ...] = (
    "analysis_type",
    "baseline_dir",
    "baseline_label",
    "variant_dir",
    "variant_label",
    "baseline_program",
    "variant_program",
    "baseline_program_size",
    "variant_program_size",
    "jaccard",
)

_SUMMARY_COLUMNS: tuple[str, ...] = (
    "analysis_type",
    "baseline_dir",
    "baseline_label",
    "variant_dir",
    "variant_label",
    "baseline_membership_file",
    "variant_membership_file",
    "n_baseline_programs",
    "n_variant_programs",
    "n_pairwise_program_pairs",
    "mean_pairwise_jaccard",
    "mean_best_jaccard_baseline_to_variant",
    "median_best_jaccard_baseline_to_variant",
    "min_best_jaccard_baseline_to_variant",
    "max_best_jaccard_baseline_to_variant",
    "mean_best_jaccard_variant_to_baseline",
    "median_best_jaccard_variant_to_baseline",
    "min_best_jaccard_variant_to_baseline",
    "max_best_jaccard_variant_to_baseline",
    "symmetric_mean_best_jaccard",
    "baseline_enriched_program_count",
    "variant_enriched_program_count",
    "baseline_claim_supported_count",
    "variant_claim_supported_count",
    "claim_supported_count_delta",
    "baseline_claim_supported_rate",
    "variant_claim_supported_rate",
    "claim_supported_rate_delta",
    "notes",
)

_HYPERPARAM_EXTRA_COLUMNS: tuple[str, ...] = (
    "parameter",
    "baseline_parameter_value",
    "variant_parameter_value",
    "n_changed_parameters",
)


@dataclass(frozen=True)
class _ClaimSupportStats:
    """Parsed claim-support summary from enrichment outputs."""

    n_rows: float
    claim_supported_count: float
    claim_supported_rate: float
    source_file: str | None
    note: str | None


def summarize_bootstrap_stability(
    baseline_results_dir: str | Path,
    bootstrap_results_dirs: list[str | Path] | tuple[str | Path, ...],
) -> SensitivitySummaryTables:
    """Summarize bootstrap-run stability against a baseline result directory.

    Args:
        baseline_results_dir: Baseline bulk-dynamic result directory.
        bootstrap_results_dirs: Bootstrap variant result directories.

    Returns:
        ``SensitivitySummaryTables`` with:
        - per-bootstrap variant summary metrics
        - long-form pairwise Jaccard table (baseline program x variant program)

    Notes:
        Program membership is loaded from ``program_gene_membership_long.csv``
        when available, and falls back to ``dynamic_programs_long.csv``.
    """
    return _summarize_variants(
        baseline_results_dir=baseline_results_dir,
        variant_results_dirs=bootstrap_results_dirs,
        analysis_type="bootstrap",
    )


def summarize_hyperparameter_sensitivity(
    baseline_results_dir: str | Path,
    variant_results_dirs: list[str | Path] | tuple[str | Path, ...],
) -> SensitivitySummaryTables:
    """Summarize hyperparameter sensitivity against a baseline result directory.

    The returned summary is exploded into one row per changed config parameter
    (based on ``run_manifest.json``), which is convenient for plotting parameter-
    specific sensitivity trends.

    Args:
        baseline_results_dir: Baseline bulk-dynamic result directory.
        variant_results_dirs: Hyperparameter variant result directories.

    Returns:
        ``SensitivitySummaryTables`` with:
        - per-parameter exploded sensitivity summary rows
        - long-form pairwise Jaccard table (baseline program x variant program)
    """
    tables = _summarize_variants(
        baseline_results_dir=baseline_results_dir,
        variant_results_dirs=variant_results_dirs,
        analysis_type="hyperparameter",
    )

    summary = _explode_hyperparameter_changes(
        summary_df=tables.summary,
        baseline_results_dir=Path(baseline_results_dir),
    )
    return SensitivitySummaryTables(summary=summary, pairwise_jaccard_long=tables.pairwise_jaccard_long)


def _summarize_variants(
    baseline_results_dir: str | Path,
    variant_results_dirs: list[str | Path] | tuple[str | Path, ...],
    analysis_type: str,
) -> SensitivitySummaryTables:
    """Build pairwise-overlap and claim-support summaries for baseline variants."""
    baseline_dir = Path(baseline_results_dir)
    variant_dirs = [Path(p) for p in variant_results_dirs]

    baseline_programs, baseline_membership_file, baseline_membership_note = _load_program_membership(
        baseline_dir
    )
    baseline_claims = _load_claim_support_stats(baseline_dir)

    baseline_label = baseline_dir.name
    summary_rows: list[dict[str, Any]] = []
    pairwise_frames: list[pd.DataFrame] = []

    for variant_dir in variant_dirs:
        variant_label = variant_dir.name
        variant_programs, variant_membership_file, variant_membership_note = _load_program_membership(
            variant_dir
        )
        variant_claims = _load_claim_support_stats(variant_dir)

        pairwise = _pairwise_jaccard_long(
            baseline_programs=baseline_programs,
            variant_programs=variant_programs,
        )
        if not pairwise.empty:
            pairwise = pairwise.assign(
                analysis_type=analysis_type,
                baseline_dir=str(baseline_dir),
                baseline_label=baseline_label,
                variant_dir=str(variant_dir),
                variant_label=variant_label,
            )
            pairwise = pairwise.loc[:, _PAIRWISE_COLUMNS]
            pairwise_frames.append(pairwise)

        jaccard_stats = _summarize_pairwise_jaccard(pairwise)

        baseline_claim_count = baseline_claims.claim_supported_count
        variant_claim_count = variant_claims.claim_supported_count
        baseline_claim_rate = baseline_claims.claim_supported_rate
        variant_claim_rate = variant_claims.claim_supported_rate

        notes = _join_notes(
            baseline_membership_note,
            variant_membership_note,
            baseline_claims.note,
            variant_claims.note,
        )

        summary_rows.append(
            {
                "analysis_type": analysis_type,
                "baseline_dir": str(baseline_dir),
                "baseline_label": baseline_label,
                "variant_dir": str(variant_dir),
                "variant_label": variant_label,
                "baseline_membership_file": baseline_membership_file,
                "variant_membership_file": variant_membership_file,
                "n_baseline_programs": int(len(baseline_programs)),
                "n_variant_programs": int(len(variant_programs)),
                "n_pairwise_program_pairs": int(len(pairwise)),
                "mean_pairwise_jaccard": jaccard_stats["mean_pairwise_jaccard"],
                "mean_best_jaccard_baseline_to_variant": (
                    jaccard_stats["mean_best_jaccard_baseline_to_variant"]
                ),
                "median_best_jaccard_baseline_to_variant": (
                    jaccard_stats["median_best_jaccard_baseline_to_variant"]
                ),
                "min_best_jaccard_baseline_to_variant": (
                    jaccard_stats["min_best_jaccard_baseline_to_variant"]
                ),
                "max_best_jaccard_baseline_to_variant": (
                    jaccard_stats["max_best_jaccard_baseline_to_variant"]
                ),
                "mean_best_jaccard_variant_to_baseline": (
                    jaccard_stats["mean_best_jaccard_variant_to_baseline"]
                ),
                "median_best_jaccard_variant_to_baseline": (
                    jaccard_stats["median_best_jaccard_variant_to_baseline"]
                ),
                "min_best_jaccard_variant_to_baseline": (
                    jaccard_stats["min_best_jaccard_variant_to_baseline"]
                ),
                "max_best_jaccard_variant_to_baseline": (
                    jaccard_stats["max_best_jaccard_variant_to_baseline"]
                ),
                "symmetric_mean_best_jaccard": jaccard_stats["symmetric_mean_best_jaccard"],
                "baseline_enriched_program_count": baseline_claims.n_rows,
                "variant_enriched_program_count": variant_claims.n_rows,
                "baseline_claim_supported_count": baseline_claim_count,
                "variant_claim_supported_count": variant_claim_count,
                "claim_supported_count_delta": _safe_delta(
                    variant_claim_count, baseline_claim_count
                ),
                "baseline_claim_supported_rate": baseline_claim_rate,
                "variant_claim_supported_rate": variant_claim_rate,
                "claim_supported_rate_delta": _safe_delta(variant_claim_rate, baseline_claim_rate),
                "notes": notes,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        summary_df = pd.DataFrame(columns=_SUMMARY_COLUMNS)
    else:
        summary_df = summary_df.loc[:, _SUMMARY_COLUMNS]

    if pairwise_frames:
        pairwise_jaccard_long = pd.concat(pairwise_frames, ignore_index=True)
    else:
        pairwise_jaccard_long = pd.DataFrame(columns=_PAIRWISE_COLUMNS)

    return SensitivitySummaryTables(
        summary=summary_df.reset_index(drop=True),
        pairwise_jaccard_long=pairwise_jaccard_long.reset_index(drop=True),
    )


def _load_program_membership(results_dir: Path) -> tuple[dict[str, set[str]], str | None, str | None]:
    """Load program-to-gene membership map from a result directory."""
    if not results_dir.exists() or not results_dir.is_dir():
        return {}, None, f"membership: directory not found ({results_dir})"

    notes: list[str] = []
    for rel_path in _MEMBERSHIP_CANDIDATES:
        csv_path = results_dir / rel_path
        if not csv_path.exists():
            continue

        try:
            df = pd.read_csv(csv_path)
        except EmptyDataError:
            notes.append(f"membership: empty file ({rel_path})")
            continue
        except Exception as exc:  # pragma: no cover - defensive guard
            notes.append(f"membership: read failed ({rel_path}: {exc})")
            continue

        cleaned = _extract_program_gene_long(df)
        if cleaned is None:
            notes.append(
                "membership: missing required columns "
                f"('program', 'gene') in {rel_path}"
            )
            continue

        programs: dict[str, set[str]] = {}
        for row in cleaned.itertuples(index=False):
            programs.setdefault(str(row.program), set()).add(str(row.gene))

        return programs, rel_path, _join_notes(*notes)

    if notes:
        return {}, None, _join_notes(*notes)
    return {}, None, "membership: no candidate file found"


def _load_claim_support_stats(results_dir: Path) -> _ClaimSupportStats:
    """Load claim-supported counts from enrichment output with robust fallbacks."""
    if not results_dir.exists() or not results_dir.is_dir():
        return _ClaimSupportStats(
            n_rows=np.nan,
            claim_supported_count=np.nan,
            claim_supported_rate=np.nan,
            source_file=None,
            note=f"claims: directory not found ({results_dir})",
        )

    notes: list[str] = []
    for rel_path in _GSEA_CANDIDATES:
        csv_path = results_dir / rel_path
        if not csv_path.exists():
            continue

        try:
            df = pd.read_csv(csv_path)
        except EmptyDataError:
            return _ClaimSupportStats(
                n_rows=0.0,
                claim_supported_count=0.0,
                claim_supported_rate=np.nan,
                source_file=rel_path,
                note="claims: empty enrichment file",
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            notes.append(f"claims: read failed ({rel_path}: {exc})")
            continue

        n_rows = float(len(df))
        claim_series = _resolve_claim_supported(df)
        if claim_series is None:
            return _ClaimSupportStats(
                n_rows=n_rows,
                claim_supported_count=np.nan,
                claim_supported_rate=np.nan,
                source_file=rel_path,
                note=_join_notes(
                    *notes,
                    "claims: missing 'claim_supported' and no derivable gate_* columns",
                ),
            )

        claim_count = float(claim_series.sum())
        claim_rate = claim_count / n_rows if n_rows > 0 else np.nan
        return _ClaimSupportStats(
            n_rows=n_rows,
            claim_supported_count=claim_count,
            claim_supported_rate=claim_rate,
            source_file=rel_path,
            note=_join_notes(*notes),
        )

    if notes:
        note = _join_notes(*notes)
    else:
        note = "claims: enrichment_gsea_with_claim_gates.csv not found"
    return _ClaimSupportStats(
        n_rows=np.nan,
        claim_supported_count=np.nan,
        claim_supported_rate=np.nan,
        source_file=None,
        note=note,
    )


def _extract_program_gene_long(df: pd.DataFrame) -> pd.DataFrame | None:
    """Normalize program-gene long table to columns ['program', 'gene']."""
    lower_to_original: dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key and key not in lower_to_original:
            lower_to_original[key] = str(col)

    if "program" not in lower_to_original or "gene" not in lower_to_original:
        return None

    out = df[[lower_to_original["program"], lower_to_original["gene"]]].copy()
    out.columns = ["program", "gene"]
    out = out.dropna(subset=["program", "gene"])
    if out.empty:
        return out

    out["program"] = out["program"].astype(str).str.strip()
    out["gene"] = out["gene"].astype(str).str.strip()

    out = out[(out["program"] != "") & (out["gene"] != "")]
    out = out[(out["program"].str.lower() != "nan") & (out["gene"].str.lower() != "nan")]
    return out.reset_index(drop=True)


def _resolve_claim_supported(df: pd.DataFrame) -> pd.Series | None:
    """Resolve claim-supported boolean series from explicit or gate columns."""
    lower_to_original: dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key and key not in lower_to_original:
            lower_to_original[key] = str(col)

    claim_col = lower_to_original.get("claim_supported")
    if claim_col is not None:
        claim = _coerce_bool_with_na(df[claim_col]).map(
            lambda x: bool(x) if pd.notna(x) else False
        )
        return claim.astype(bool)

    gate_cols = [c for c in df.columns if str(c).startswith("gate_")]
    if not gate_cols:
        return None

    gate_pass = pd.Series(True, index=df.index, dtype=bool)
    for col in gate_cols:
        gate = _coerce_bool_with_na(df[col])
        gate_pass = gate_pass & (gate.eq(True) | gate.isna())
    return gate_pass.astype(bool)


def _coerce_bool_with_na(series: pd.Series) -> pd.Series:
    """Coerce boolean-like strings/numbers to bool while preserving unknowns."""
    normalized = series.astype(str).str.strip().str.lower()
    mapped = normalized.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
            "y": True,
            "n": False,
            "": np.nan,
            "nan": np.nan,
            "none": np.nan,
            "null": np.nan,
            "na": np.nan,
            "n/a": np.nan,
        }
    )

    numeric = pd.to_numeric(series, errors="coerce")
    mapped = mapped.where(mapped.notna(), numeric.map({1.0: True, 0.0: False}))
    return mapped


def _pairwise_jaccard_long(
    baseline_programs: dict[str, set[str]],
    variant_programs: dict[str, set[str]],
) -> pd.DataFrame:
    """Compute long-form pairwise Jaccard table for two program dictionaries."""
    if not baseline_programs or not variant_programs:
        return pd.DataFrame(
            columns=(
                "baseline_program",
                "variant_program",
                "baseline_program_size",
                "variant_program_size",
                "jaccard",
            )
        )

    rows: list[dict[str, Any]] = []
    for baseline_program in sorted(baseline_programs):
        baseline_genes = baseline_programs[baseline_program]
        for variant_program in sorted(variant_programs):
            variant_genes = variant_programs[variant_program]
            union = baseline_genes | variant_genes
            if union:
                jaccard = float(len(baseline_genes & variant_genes) / len(union))
            else:
                jaccard = 0.0
            rows.append(
                {
                    "baseline_program": baseline_program,
                    "variant_program": variant_program,
                    "baseline_program_size": int(len(baseline_genes)),
                    "variant_program_size": int(len(variant_genes)),
                    "jaccard": jaccard,
                }
            )

    return pd.DataFrame(rows)


def _summarize_pairwise_jaccard(pairwise: pd.DataFrame) -> dict[str, float]:
    """Summarize pairwise Jaccard matrix into robust directional statistics."""
    if pairwise.empty:
        return {
            "mean_pairwise_jaccard": np.nan,
            "mean_best_jaccard_baseline_to_variant": np.nan,
            "median_best_jaccard_baseline_to_variant": np.nan,
            "min_best_jaccard_baseline_to_variant": np.nan,
            "max_best_jaccard_baseline_to_variant": np.nan,
            "mean_best_jaccard_variant_to_baseline": np.nan,
            "median_best_jaccard_variant_to_baseline": np.nan,
            "min_best_jaccard_variant_to_baseline": np.nan,
            "max_best_jaccard_variant_to_baseline": np.nan,
            "symmetric_mean_best_jaccard": np.nan,
        }

    baseline_best = pairwise.groupby("baseline_program", as_index=False)["jaccard"].max()["jaccard"]
    variant_best = pairwise.groupby("variant_program", as_index=False)["jaccard"].max()["jaccard"]

    mean_baseline = _nan_safe_stat(baseline_best, "mean")
    mean_variant = _nan_safe_stat(variant_best, "mean")

    return {
        "mean_pairwise_jaccard": _nan_safe_stat(pairwise["jaccard"], "mean"),
        "mean_best_jaccard_baseline_to_variant": mean_baseline,
        "median_best_jaccard_baseline_to_variant": _nan_safe_stat(baseline_best, "median"),
        "min_best_jaccard_baseline_to_variant": _nan_safe_stat(baseline_best, "min"),
        "max_best_jaccard_baseline_to_variant": _nan_safe_stat(baseline_best, "max"),
        "mean_best_jaccard_variant_to_baseline": mean_variant,
        "median_best_jaccard_variant_to_baseline": _nan_safe_stat(variant_best, "median"),
        "min_best_jaccard_variant_to_baseline": _nan_safe_stat(variant_best, "min"),
        "max_best_jaccard_variant_to_baseline": _nan_safe_stat(variant_best, "max"),
        "symmetric_mean_best_jaccard": _nan_safe_mean([mean_baseline, mean_variant]),
    }


def _explode_hyperparameter_changes(
    summary_df: pd.DataFrame,
    baseline_results_dir: Path,
) -> pd.DataFrame:
    """Expand per-variant summary into one row per changed hyperparameter."""
    if summary_df.empty:
        return pd.DataFrame(columns=(*_SUMMARY_COLUMNS, *_HYPERPARAM_EXTRA_COLUMNS))

    baseline_config, baseline_note = _load_run_manifest_config(baseline_results_dir)

    exploded_rows: list[dict[str, Any]] = []
    for row in summary_df.to_dict(orient="records"):
        variant_dir = Path(str(row["variant_dir"]))
        variant_config, variant_note = _load_run_manifest_config(variant_dir)
        diffs = _diff_configs(baseline_config, variant_config)

        local_note = _join_notes(row.get("notes"), baseline_note, variant_note)

        if not diffs:
            new_row = dict(row)
            new_row["parameter"] = "(none_or_unknown)"
            new_row["baseline_parameter_value"] = np.nan
            new_row["variant_parameter_value"] = np.nan
            new_row["n_changed_parameters"] = 0
            new_row["notes"] = local_note
            exploded_rows.append(new_row)
            continue

        n_changes = len(diffs)
        for param_name, base_val, variant_val in diffs:
            new_row = dict(row)
            new_row["parameter"] = param_name
            new_row["baseline_parameter_value"] = base_val
            new_row["variant_parameter_value"] = variant_val
            new_row["n_changed_parameters"] = n_changes
            new_row["notes"] = local_note
            exploded_rows.append(new_row)

    exploded_df = pd.DataFrame(exploded_rows)
    return exploded_df.loc[:, [*_SUMMARY_COLUMNS, *_HYPERPARAM_EXTRA_COLUMNS]]


def _load_run_manifest_config(results_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Load run configuration from run_manifest.json if available."""
    manifest_path = results_dir / "run_manifest.json"
    if not manifest_path.exists():
        return None, f"manifest: not found ({manifest_path})"

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive guard
        return None, f"manifest: parse failed ({manifest_path}: {exc})"

    if isinstance(payload, dict) and isinstance(payload.get("config"), dict):
        return payload["config"], None
    if isinstance(payload, dict):
        return payload, "manifest: missing top-level 'config'; using root object"
    return None, "manifest: invalid format"


def _diff_configs(
    baseline_config: dict[str, Any] | None,
    variant_config: dict[str, Any] | None,
) -> list[tuple[str, str, str]]:
    """Compute config differences for tunable parameters."""
    if baseline_config is None or variant_config is None:
        return []

    diffs: list[tuple[str, str, str]] = []
    for key in sorted(set(baseline_config) | set(variant_config)):
        if key in _IGNORED_CONFIG_KEYS:
            continue

        base_val = baseline_config.get(key)
        variant_val = variant_config.get(key)
        if _values_equal(base_val, variant_val):
            continue

        diffs.append((key, _format_config_value(base_val), _format_config_value(variant_val)))
    return diffs


def _values_equal(left: Any, right: Any) -> bool:
    """Safe equality check that treats NaN values as equal."""
    if _is_nan_scalar(left) and _is_nan_scalar(right):
        return True

    try:
        equal = left == right
    except Exception:
        return False

    if isinstance(equal, (bool, np.bool_)):
        return bool(equal)

    return False


def _is_nan_scalar(value: Any) -> bool:
    """Return True if value is a scalar NaN."""
    if isinstance(value, (float, np.floating)):
        return bool(math.isnan(float(value)))
    return False


def _format_config_value(value: Any) -> str:
    """Format config values as compact stable strings."""
    if value is None:
        return "null"
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return str(value)


def _nan_safe_stat(series: pd.Series, kind: str) -> float:
    """Compute mean/median/min/max with NaN output for empty or invalid data."""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return np.nan

    if kind == "mean":
        return float(numeric.mean())
    if kind == "median":
        return float(numeric.median())
    if kind == "min":
        return float(numeric.min())
    if kind == "max":
        return float(numeric.max())

    raise ValueError(f"Unsupported stat kind: {kind}")


def _nan_safe_mean(values: list[float]) -> float:
    """Mean across finite values only; return NaN when no finite values exist."""
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return np.nan
    return float(np.mean(finite))


def _safe_delta(current: float, baseline: float) -> float:
    """Return current - baseline when both values are finite, else NaN."""
    if np.isfinite(current) and np.isfinite(baseline):
        return float(current - baseline)
    return np.nan


def _join_notes(*notes: str | None) -> str | None:
    """Combine non-empty note strings into a single semicolon-separated message."""
    clean = [n.strip() for n in notes if isinstance(n, str) and n.strip()]
    if not clean:
        return None
    return "; ".join(clean)


__all__ = [
    "SensitivitySummaryTables",
    "summarize_bootstrap_stability",
    "summarize_hyperparameter_sensitivity",
]
