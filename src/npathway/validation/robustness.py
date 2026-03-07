"""Robustness analysis utilities for perturbation/batch-like run comparisons.

The main entry point compares multiple result directories against a baseline
run and returns tidy tables with:

1. Program robustness (mean best-match Jaccard).
2. Claim-support overlap/retention.
3. Context-metric stability via correlation on shared program-gene pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

PROGRAM_MATCH_COLUMNS = [
    "baseline_label",
    "perturbation_label",
    "baseline_program",
    "perturbation_program",
    "best_match_jaccard",
    "shared_genes_n",
    "baseline_gene_n",
    "perturbation_gene_n",
    "baseline_claim_supported",
    "perturbation_claim_supported",
    "claim_supported_retained",
]

CONTEXT_PAIR_COLUMNS = [
    "baseline_label",
    "perturbation_label",
    "baseline_program",
    "perturbation_program",
    "gene",
    "baseline_context_metric",
    "perturbation_context_metric",
    "baseline_context_metric_name",
    "perturbation_context_metric_name",
]

SUMMARY_COLUMNS = [
    "baseline_label",
    "perturbation_label",
    "n_baseline_programs",
    "n_perturbation_programs",
    "mean_best_match_program_jaccard",
    "n_baseline_claim_supported",
    "n_perturbation_claim_supported",
    "n_claim_supported_retained",
    "claim_supported_overlap_jaccard",
    "claim_supported_retention",
    "n_context_pairs",
    "context_metric_stability_correlation",
    "baseline_context_metric_name",
    "perturbation_context_metric_name",
]


@dataclass(frozen=True)
class _RunArtifacts:
    """Loaded artifacts for one perturbation run directory."""

    label: str
    result_dir: Path
    programs: dict[str, set[str]]
    claim_supported: dict[str, bool]
    context_metric_name: str
    context_table: pd.DataFrame


def analyze_perturbation_robustness(
    result_dirs_by_label: Mapping[str, str | Path],
    baseline_label: str = "baseline",
) -> dict[str, pd.DataFrame]:
    """Compare perturbation-labeled result directories against a baseline.

    Expected files per result directory:
    - ``program_gene_lists.csv`` (preferred) or ``dynamic_programs_long.csv``
    - ``enrichment_gsea_with_claim_gates.csv`` (optional for claim metrics)
    - ``contextual_membership_scores.csv`` (optional for context metrics)

    Args:
        result_dirs_by_label: Mapping from perturbation label to result dir.
        baseline_label: Label in ``result_dirs_by_label`` used as baseline.

    Returns:
        Dict with three tidy DataFrames:
        - ``summary``: one row per perturbation (vs baseline)
        - ``program_matches``: baseline program -> best perturbation program
        - ``context_pairs``: shared program-gene context metric pairs
    """
    if baseline_label not in result_dirs_by_label:
        raise ValueError(
            f"baseline_label='{baseline_label}' not present in result_dirs_by_label."
        )

    run_dirs: dict[str, Path] = {
        str(label): Path(path) for label, path in result_dirs_by_label.items()
    }
    baseline = _load_run_artifacts(run_dirs[baseline_label], baseline_label)

    summary_rows: list[dict[str, Any]] = []
    program_match_tables: list[pd.DataFrame] = []
    context_pair_tables: list[pd.DataFrame] = []

    perturbation_labels = sorted(label for label in run_dirs if label != baseline_label)
    for perturbation_label in perturbation_labels:
        perturbation = _load_run_artifacts(
            run_dirs[perturbation_label], perturbation_label
        )

        baseline_to_pert = _best_matches(baseline.programs, perturbation.programs)
        pert_to_baseline = _best_matches(perturbation.programs, baseline.programs)

        mean_best = _mean_directional_best_match_jaccard(
            baseline_to_pert=baseline_to_pert,
            perturbation_to_baseline=pert_to_baseline,
        )

        program_matches = _build_program_match_table(
            baseline_label=baseline_label,
            perturbation_label=perturbation_label,
            baseline_to_perturbation=baseline_to_pert,
            baseline_claim_supported=baseline.claim_supported,
            perturbation_claim_supported=perturbation.claim_supported,
        )
        context_pairs = _build_context_pair_table(
            baseline=baseline,
            perturbation=perturbation,
            program_matches=program_matches,
        )

        claim_overlap = _claim_overlap_jaccard(program_matches)
        claim_retention = _claim_retention(program_matches)

        summary_rows.append(
            {
                "baseline_label": baseline_label,
                "perturbation_label": perturbation_label,
                "n_baseline_programs": len(baseline.programs),
                "n_perturbation_programs": len(perturbation.programs),
                "mean_best_match_program_jaccard": mean_best,
                "n_baseline_claim_supported": int(
                    sum(bool(v) for v in baseline.claim_supported.values())
                ),
                "n_perturbation_claim_supported": int(
                    sum(bool(v) for v in perturbation.claim_supported.values())
                ),
                "n_claim_supported_retained": int(
                    program_matches["claim_supported_retained"].sum()
                )
                if not program_matches.empty
                else 0,
                "claim_supported_overlap_jaccard": claim_overlap,
                "claim_supported_retention": claim_retention,
                "n_context_pairs": int(len(context_pairs)),
                "context_metric_stability_correlation": _safe_pearson(
                    context_pairs["baseline_context_metric"].to_numpy(
                        dtype=np.float64
                    )
                    if not context_pairs.empty
                    else np.array([], dtype=np.float64),
                    context_pairs["perturbation_context_metric"].to_numpy(
                        dtype=np.float64
                    )
                    if not context_pairs.empty
                    else np.array([], dtype=np.float64),
                ),
                "baseline_context_metric_name": baseline.context_metric_name,
                "perturbation_context_metric_name": perturbation.context_metric_name,
            }
        )

        program_match_tables.append(program_matches)
        context_pair_tables.append(context_pairs)

    summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("perturbation_label").reset_index(drop=True)

    program_matches_df = (
        pd.concat(program_match_tables, ignore_index=True)
        if program_match_tables
        else pd.DataFrame(columns=PROGRAM_MATCH_COLUMNS)
    )
    if not program_matches_df.empty:
        program_matches_df = program_matches_df.sort_values(
            ["perturbation_label", "baseline_program"]
        ).reset_index(drop=True)

    context_pairs_df = (
        pd.concat(context_pair_tables, ignore_index=True)
        if context_pair_tables
        else pd.DataFrame(columns=CONTEXT_PAIR_COLUMNS)
    )
    if not context_pairs_df.empty:
        context_pairs_df = context_pairs_df.sort_values(
            ["perturbation_label", "baseline_program", "gene"]
        ).reset_index(drop=True)

    return {
        "summary": summary_df,
        "program_matches": program_matches_df,
        "context_pairs": context_pairs_df,
    }


def _load_run_artifacts(result_dir: Path, label: str) -> _RunArtifacts:
    """Load required/optional files from a single result directory."""
    if not result_dir.exists():
        raise FileNotFoundError(f"Result directory does not exist: {result_dir}")
    if not result_dir.is_dir():
        raise NotADirectoryError(f"Result path is not a directory: {result_dir}")

    programs = _load_program_gene_sets(result_dir)
    claim_supported = _load_claim_support(result_dir)
    context_table, context_metric_name = _load_context_metrics(result_dir)
    return _RunArtifacts(
        label=label,
        result_dir=result_dir,
        programs=programs,
        claim_supported=claim_supported,
        context_metric_name=context_metric_name,
        context_table=context_table,
    )


def _load_program_gene_sets(result_dir: Path) -> dict[str, set[str]]:
    """Load program gene sets from compact or long-form output tables."""
    compact_path = result_dir / "program_gene_lists.csv"
    long_path = result_dir / "dynamic_programs_long.csv"

    if compact_path.exists():
        compact_df = pd.read_csv(compact_path)
        if not {"program", "genes"}.issubset(compact_df.columns):
            raise ValueError(
                f"{compact_path} must contain columns ['program', 'genes']."
            )

        programs: dict[str, set[str]] = {}
        for _, row in compact_df.iterrows():
            program = str(row["program"])
            genes = _parse_gene_list(row["genes"])
            if program in programs:
                programs[program].update(genes)
            else:
                programs[program] = genes
        return programs

    if long_path.exists():
        long_df = pd.read_csv(long_path)
        if not {"program", "gene"}.issubset(long_df.columns):
            raise ValueError(
                f"{long_path} must contain columns ['program', 'gene']."
            )
        filtered = long_df.dropna(subset=["program", "gene"]).copy()
        filtered["program"] = filtered["program"].astype(str)
        filtered["gene"] = filtered["gene"].astype(str)
        grouped = filtered.groupby("program")["gene"].apply(set)
        return {program: set(genes) for program, genes in grouped.items()}

    raise FileNotFoundError(
        f"{result_dir} is missing both program files: "
        "'program_gene_lists.csv' and 'dynamic_programs_long.csv'."
    )


def _load_claim_support(result_dir: Path) -> dict[str, bool]:
    """Load per-program claim-supported flags if available."""
    path = result_dir / "enrichment_gsea_with_claim_gates.csv"
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    if "program" not in df.columns:
        raise ValueError(f"{path} must contain a 'program' column.")

    if "claim_supported" in df.columns:
        claim_series = df["claim_supported"].map(_coerce_bool)
    else:
        claim_series = pd.Series(False, index=df.index, dtype=bool)

    out: dict[str, bool] = {}
    for program, claim_supported in zip(df["program"], claim_series, strict=False):
        if pd.isna(program):
            continue
        key = str(program)
        out[key] = bool(claim_supported) or out.get(key, False)
    return out


def _load_context_metrics(result_dir: Path) -> tuple[pd.DataFrame, str]:
    """Load context metrics with context_evidence preferred over context_shift."""
    path = result_dir / "contextual_membership_scores.csv"
    if not path.exists():
        return pd.DataFrame(columns=["program", "gene", "context_metric"]), "missing"

    df = pd.read_csv(path)
    if not {"program", "gene"}.issubset(df.columns):
        raise ValueError(f"{path} must contain at least ['program', 'gene'] columns.")

    if "context_evidence" in df.columns:
        metric_name = "context_evidence"
    elif "context_shift" in df.columns:
        metric_name = "context_shift"
    else:
        raise ValueError(
            f"{path} must contain either 'context_evidence' or 'context_shift'."
        )

    context = df[["program", "gene", metric_name]].copy()
    context = context.dropna(subset=["program", "gene"])
    context["program"] = context["program"].astype(str)
    context["gene"] = context["gene"].astype(str)
    context["context_metric"] = pd.to_numeric(
        context[metric_name], errors="coerce"
    )
    context = context.dropna(subset=["context_metric"])
    context = (
        context.groupby(["program", "gene"], as_index=False)["context_metric"]
        .mean()
        .sort_values(["program", "gene"])
        .reset_index(drop=True)
    )
    return context, metric_name


def _best_matches(
    source_programs: Mapping[str, set[str]],
    target_programs: Mapping[str, set[str]],
) -> pd.DataFrame:
    """Return best target program match for each source program by Jaccard."""
    rows: list[dict[str, Any]] = []
    sorted_targets = sorted(target_programs.items(), key=lambda x: x[0])

    for source_program in sorted(source_programs):
        source_genes = source_programs[source_program]
        best_target: str | None = None
        best_jaccard = -1.0
        best_shared_n = 0
        best_target_n = 0

        for target_program, target_genes in sorted_targets:
            score = _jaccard(source_genes, target_genes)
            shared_n = len(source_genes & target_genes)
            if score > best_jaccard:
                best_jaccard = score
                best_target = target_program
                best_shared_n = shared_n
                best_target_n = len(target_genes)

        rows.append(
            {
                "source_program": source_program,
                "target_program": best_target,
                "best_match_jaccard": max(best_jaccard, 0.0),
                "shared_genes_n": int(best_shared_n),
                "source_gene_n": int(len(source_genes)),
                "target_gene_n": int(best_target_n),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "source_program",
            "target_program",
            "best_match_jaccard",
            "shared_genes_n",
            "source_gene_n",
            "target_gene_n",
        ],
    )


def _mean_directional_best_match_jaccard(
    baseline_to_pert: pd.DataFrame,
    perturbation_to_baseline: pd.DataFrame,
) -> float:
    """Mean best-match Jaccard across both comparison directions."""
    values: list[float] = []
    if not baseline_to_pert.empty:
        values.extend(
            baseline_to_pert["best_match_jaccard"].to_numpy(dtype=np.float64).tolist()
        )
    if not perturbation_to_baseline.empty:
        values.extend(
            perturbation_to_baseline["best_match_jaccard"]
            .to_numpy(dtype=np.float64)
            .tolist()
        )
    if not values:
        return float("nan")
    return float(np.mean(values))


def _build_program_match_table(
    baseline_label: str,
    perturbation_label: str,
    baseline_to_perturbation: pd.DataFrame,
    baseline_claim_supported: Mapping[str, bool],
    perturbation_claim_supported: Mapping[str, bool],
) -> pd.DataFrame:
    """Build tidy baseline-program match table enriched with claim flags."""
    if baseline_to_perturbation.empty:
        return pd.DataFrame(columns=PROGRAM_MATCH_COLUMNS)

    table = baseline_to_perturbation.copy()
    table = table.rename(
        columns={
            "source_program": "baseline_program",
            "target_program": "perturbation_program",
            "source_gene_n": "baseline_gene_n",
            "target_gene_n": "perturbation_gene_n",
        }
    )
    table["baseline_claim_supported"] = table["baseline_program"].map(
        lambda p: bool(baseline_claim_supported.get(str(p), False))
    )
    table["perturbation_claim_supported"] = table["perturbation_program"].map(
        lambda p: bool(perturbation_claim_supported.get(str(p), False))
        if pd.notna(p)
        else False
    )
    table["claim_supported_retained"] = (
        table["baseline_claim_supported"] & table["perturbation_claim_supported"]
    )
    table["baseline_label"] = baseline_label
    table["perturbation_label"] = perturbation_label
    return table[PROGRAM_MATCH_COLUMNS].reset_index(drop=True)


def _build_context_pair_table(
    baseline: _RunArtifacts,
    perturbation: _RunArtifacts,
    program_matches: pd.DataFrame,
) -> pd.DataFrame:
    """Build shared program-gene context metric pairs from best matches."""
    if program_matches.empty:
        return pd.DataFrame(columns=CONTEXT_PAIR_COLUMNS)
    if baseline.context_table.empty or perturbation.context_table.empty:
        return pd.DataFrame(columns=CONTEXT_PAIR_COLUMNS)

    baseline_lookup = baseline.context_table.set_index(["program", "gene"])[
        "context_metric"
    ].to_dict()
    perturb_lookup = perturbation.context_table.set_index(["program", "gene"])[
        "context_metric"
    ].to_dict()

    rows: list[dict[str, Any]] = []
    for _, match_row in program_matches.iterrows():
        baseline_program = str(match_row["baseline_program"])
        perturbation_program = match_row["perturbation_program"]
        if pd.isna(perturbation_program):
            continue

        perturbation_program = str(perturbation_program)
        shared_genes = sorted(
            baseline.programs.get(baseline_program, set())
            & perturbation.programs.get(perturbation_program, set())
        )
        for gene in shared_genes:
            baseline_key = (baseline_program, gene)
            perturbation_key = (perturbation_program, gene)
            if baseline_key not in baseline_lookup:
                continue
            if perturbation_key not in perturb_lookup:
                continue
            rows.append(
                {
                    "baseline_label": baseline.label,
                    "perturbation_label": perturbation.label,
                    "baseline_program": baseline_program,
                    "perturbation_program": perturbation_program,
                    "gene": gene,
                    "baseline_context_metric": float(baseline_lookup[baseline_key]),
                    "perturbation_context_metric": float(
                        perturb_lookup[perturbation_key]
                    ),
                    "baseline_context_metric_name": baseline.context_metric_name,
                    "perturbation_context_metric_name": perturbation.context_metric_name,
                }
            )

    return pd.DataFrame(rows, columns=CONTEXT_PAIR_COLUMNS)


def _claim_overlap_jaccard(program_matches: pd.DataFrame) -> float:
    """Jaccard overlap of baseline-supported vs mapped perturb-supported claims."""
    if program_matches.empty:
        return float("nan")
    baseline_supported = set(
        program_matches.loc[
            program_matches["baseline_claim_supported"], "baseline_program"
        ].astype(str)
    )
    perturbation_supported_mapped = set(
        program_matches.loc[
            program_matches["perturbation_claim_supported"], "baseline_program"
        ].astype(str)
    )
    return _jaccard(baseline_supported, perturbation_supported_mapped)


def _claim_retention(program_matches: pd.DataFrame) -> float:
    """Retention of baseline claim-supported programs in the perturbation."""
    if program_matches.empty:
        return float("nan")
    baseline_supported = program_matches["baseline_claim_supported"].to_numpy(dtype=bool)
    retained = program_matches["claim_supported_retained"].to_numpy(dtype=bool)
    denom = int(baseline_supported.sum())
    if denom == 0:
        return float("nan")
    return float(retained.sum() / denom)


def _jaccard(left: set[str], right: set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return float(len(left & right) / len(union))


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation with finite-value and variance safety checks."""
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("x and y must have the same length for correlation.")
    if x_arr.shape[0] < 2:
        return float("nan")

    finite_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if finite_mask.sum() < 2:
        return float("nan")

    x_f = x_arr[finite_mask]
    y_f = y_arr[finite_mask]
    if np.allclose(x_f, x_f[0]) or np.allclose(y_f, y_f[0]):
        return float("nan")
    return float(np.corrcoef(x_f, y_f)[0, 1])


def _parse_gene_list(value: Any) -> set[str]:
    """Parse comma-delimited genes from compact program table."""
    if pd.isna(value):
        return set()
    tokens = [tok.strip() for tok in str(value).split(",")]
    return {tok for tok in tokens if tok}


def _coerce_bool(value: Any) -> bool:
    """Robust bool coercion for CSV values."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, np.integer)):
        return int(value) != 0
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return False
        return float(value) != 0.0
    text = str(value).strip().lower()
    if text in {"true", "t", "1", "yes", "y"}:
        return True
    if text in {"false", "f", "0", "no", "n", ""}:
        return False
    return False


__all__ = ["analyze_perturbation_robustness"]
