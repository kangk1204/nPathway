"""User-facing input validation helpers for nPathway workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from npathway.pipeline.bulk_dynamic import _load_table


@dataclass
class ValidationReport:
    """Structured validation summary returned by input-check helpers."""

    mode: str
    summary: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to a JSON-serializable dictionary."""
        return {
            "mode": self.mode,
            "summary": self.summary,
            "warnings": list(self.warnings),
        }


def _count_non_numeric_entries(df: pd.DataFrame) -> int:
    """Count non-empty cells that fail numeric coercion."""
    numeric = df.apply(pd.to_numeric, errors="coerce")
    original_non_null = df.notna()
    return int((numeric.isna() & original_non_null).sum().sum())


def _count_non_finite_entries(df: pd.DataFrame) -> int:
    """Count finite-check failures after numeric coercion."""
    numeric = df.apply(pd.to_numeric, errors="coerce")
    finite_mask = pd.DataFrame(
        np.isfinite(numeric.to_numpy(dtype=float)),
        index=numeric.index,
        columns=numeric.columns,
    )
    return int((~finite_mask & numeric.notna()).sum().sum())


def _count_non_integer_entries(df: pd.DataFrame) -> int:
    """Count numeric entries that are not integer-like.

    Uses modular arithmetic (``x % 1``) instead of ``round()`` comparison
    to avoid false positives for very large integers (>2^53) where
    float64 loses precision.
    """
    values = df.to_numpy(dtype=np.float64)
    remainder = np.abs(np.mod(values, 1.0))
    return int(((remainder > 1e-9) & (remainder < 1.0 - 1e-9)).sum())


def _warn_if_minimal_group_size(
    warnings: list[str],
    *,
    label: str,
    count: int,
) -> None:
    """Add a warning when a group only meets the hard minimum."""
    if count == 2:
        warnings.append(
            f"Group '{label}' has only 2 samples, which meets the hard minimum but "
            "is fragile for publication-grade inference. Prefer >=3 biological replicates."
        )


def validate_bulk_input_files(
    *,
    matrix_path: str | Path,
    metadata_path: str | Path,
    sample_col: str,
    group_col: str,
    group_a: str,
    group_b: str,
    matrix_orientation: str = "genes_by_samples",
    sep: str | None = None,
    raw_counts: bool = False,
    batch_col: str | None = None,
) -> ValidationReport:
    """Validate a bulk RNA-seq contrast input pair."""
    matrix_raw = _load_table(str(matrix_path), sep=sep)
    metadata = _load_table(str(metadata_path), sep=sep)
    warnings: list[str] = []
    group_a = str(group_a)
    group_b = str(group_b)

    if matrix_raw.shape[1] < 3:
        raise ValueError("matrix must include id column + at least 2 sample/gene columns.")
    if group_a == group_b:
        raise ValueError("group_a and group_b must be different labels.")
    if sample_col not in metadata.columns:
        raise KeyError(
            f"metadata is missing sample column '{sample_col}'. "
            f"Available columns: {list(metadata.columns)}"
        )
    if group_col not in metadata.columns:
        raise KeyError(
            f"metadata is missing group column '{group_col}'. "
            f"Available columns: {list(metadata.columns)}"
        )
    if batch_col is not None and batch_col not in metadata.columns:
        raise KeyError(
            f"metadata is missing batch column '{batch_col}'. "
            f"Available columns: {list(metadata.columns)}"
        )

    orientation = matrix_orientation.lower().strip()
    if orientation not in {"genes_by_samples", "samples_by_genes"}:
        raise ValueError("matrix_orientation must be 'genes_by_samples' or 'samples_by_genes'.")

    if matrix_raw.iloc[:, 0].isna().any():
        raise ValueError("matrix first ID column contains missing values.")
    if metadata[sample_col].isna().any():
        raise ValueError(f"metadata sample column '{sample_col}' contains missing values.")
    if metadata[group_col].isna().any():
        raise ValueError(f"metadata group column '{group_col}' contains missing values.")
    if batch_col is not None and metadata[batch_col].isna().any():
        raise ValueError(f"metadata batch column '{batch_col}' contains missing values.")

    if metadata[sample_col].astype(str).duplicated().any():
        dupes = metadata.loc[metadata[sample_col].astype(str).duplicated(), sample_col].astype(str).tolist()
        raise ValueError(
            "metadata sample IDs must be unique. Duplicates: " + ", ".join(sorted(set(dupes)))
        )

    id_series = matrix_raw.iloc[:, 0].astype(str)
    if id_series.duplicated().any():
        dupes = id_series[id_series.duplicated()].tolist()
        role = "gene IDs" if orientation == "genes_by_samples" else "sample IDs"
        raise ValueError(
            f"matrix first-column {role} must be unique. Duplicates: " + ", ".join(sorted(set(dupes)))
        )

    value_block = matrix_raw.iloc[:, 1:]
    n_missing = int(value_block.isna().sum().sum())
    if n_missing > 0:
        raise ValueError(
            f"matrix contains {n_missing} missing/blank numeric value(s) outside the first ID column."
        )
    n_non_numeric = _count_non_numeric_entries(value_block)
    if n_non_numeric > 0:
        raise ValueError(
            f"matrix contains {n_non_numeric} non-numeric value(s) outside the first ID column."
        )
    n_non_finite = _count_non_finite_entries(value_block)
    if n_non_finite > 0:
        raise ValueError(
            f"matrix contains {n_non_finite} non-finite numeric value(s) outside the first ID column."
        )
    numeric_block = value_block.apply(pd.to_numeric, errors="coerce")
    if raw_counts:
        n_negative = int((numeric_block < 0).sum().sum())
        if n_negative > 0:
            raise ValueError(
                f"raw-count matrix contains {n_negative} negative value(s); counts must be non-negative."
            )
        n_non_integer = _count_non_integer_entries(numeric_block)
        if n_non_integer > 0:
            raise ValueError(
                f"raw-count matrix contains {n_non_integer} non-integer value(s); counts must be integers."
            )

    meta = metadata.copy()
    meta[sample_col] = meta[sample_col].astype(str)
    meta[group_col] = meta[group_col].astype(str)
    if batch_col is not None:
        meta[batch_col] = meta[batch_col].astype(str)

    if orientation == "genes_by_samples":
        sample_ids = pd.Index(matrix_raw.columns[1:].astype(str))
        if sample_ids.duplicated().any():
            dupes = sample_ids[sample_ids.duplicated()].tolist()
            raise ValueError(
                "matrix sample columns must be unique. Duplicates: " + ", ".join(sorted(set(dupes)))
            )
        common = [s for s in meta[sample_col] if s in sample_ids]
        n_genes = int(len(id_series))
        n_samples_in_matrix = int(len(sample_ids))
    else:
        row_ids = id_series
        common = [s for s in meta[sample_col] if s in set(row_ids)]
        n_genes = int(matrix_raw.shape[1] - 1)
        n_samples_in_matrix = int(len(row_ids))

    if not common:
        raise ValueError("No overlapping sample IDs between matrix and metadata.")

    n_metadata = int(len(meta))
    n_common = int(len(common))
    n_matrix_only = int(n_samples_in_matrix - n_common)
    n_metadata_only = int(n_metadata - n_common)
    if n_matrix_only > 0 or n_metadata_only > 0:
        warnings.append(
            f"Sample alignment will drop unmatched entries (matrix_only={n_matrix_only}, metadata_only={n_metadata_only})."
        )

    aligned_meta = meta.loc[meta[sample_col].isin(common)].copy()
    available_groups = set(aligned_meta[group_col].astype(str))
    if group_a not in available_groups:
        raise ValueError(f"group_a='{group_a}' not found among aligned metadata groups: {sorted(available_groups)}")
    if group_b not in available_groups:
        raise ValueError(f"group_b='{group_b}' not found among aligned metadata groups: {sorted(available_groups)}")

    contrast_meta = aligned_meta.loc[aligned_meta[group_col].isin([group_a, group_b])].copy()
    n_group_a = int(contrast_meta[group_col].eq(group_a).sum())
    n_group_b = int(contrast_meta[group_col].eq(group_b).sum())
    if n_group_a < 2 or n_group_b < 2:
        raise ValueError(
            "Each group must have at least 2 samples for two-group analysis. "
            f"Observed: {group_a}={n_group_a}, {group_b}={n_group_b}."
        )
    _warn_if_minimal_group_size(warnings, label=group_a, count=n_group_a)
    _warn_if_minimal_group_size(warnings, label=group_b, count=n_group_b)

    n_other_groups = int(len(aligned_meta) - len(contrast_meta))
    if n_other_groups > 0:
        warnings.append(
            f"{n_other_groups} aligned sample(s) belong to groups outside the requested contrast and will be ignored."
        )

    if batch_col is not None:
        n_batches = int(contrast_meta[batch_col].nunique())
        if n_batches < 2:
            warnings.append(
                f"Batch column '{batch_col}' contains only one unique value in the requested contrast; "
                "batch correction will be a no-op."
            )
        batch_group = pd.crosstab(contrast_meta[batch_col], contrast_meta[group_col])
        if not batch_group.empty:
            batch_support = (batch_group > 0).sum(axis=1)
            group_support = (batch_group > 0).sum(axis=0)
            perfectly_confounded = bool(
                (batch_support <= 1).all()
                and (group_support <= 1).all()
                and n_batches >= 2
            )
            if perfectly_confounded:
                raise ValueError(
                    f"batch_col='{batch_col}' is perfectly confounded with '{group_col}' in the requested contrast. "
                    "A batch-aware model would be unidentifiable."
                )
            if bool((batch_support <= 1).any()):
                warnings.append(
                    f"Some batches in '{batch_col}' contain only one group label. "
                    "Batch-aware estimates may be unstable."
                )

    summary = {
        "matrix_path": str(matrix_path),
        "metadata_path": str(metadata_path),
        "matrix_orientation": orientation,
        "n_genes": n_genes,
        "n_samples_in_matrix": n_samples_in_matrix,
        "n_samples_in_metadata": n_metadata,
        "n_shared_samples": n_common,
        "group_a": group_a,
        "group_b": group_b,
        "n_group_a": n_group_a,
        "n_group_b": n_group_b,
        "batch_col": batch_col,
    }
    return ValidationReport(mode="bulk", summary=summary, warnings=warnings)


def validate_scrna_pseudobulk_input(
    *,
    adata_path: str | Path,
    sample_col: str,
    group_col: str,
    group_a: str,
    group_b: str,
    subset_col: str | None = None,
    subset_value: str | None = None,
    layer: str | None = None,
    use_raw: bool = True,
    min_cells_per_sample: int = 10,
) -> ValidationReport:
    """Validate scRNA `.h5ad` input for the pseudobulk case/control workflow."""
    if min_cells_per_sample < 1:
        raise ValueError("min_cells_per_sample must be >= 1.")
    group_a = str(group_a)
    group_b = str(group_b)
    if group_a == group_b:
        raise ValueError("group_a and group_b must be different labels.")

    adata = ad.read_h5ad(str(adata_path))
    warnings: list[str] = []

    if sample_col not in adata.obs.columns:
        raise KeyError(
            f"adata.obs is missing sample column '{sample_col}'. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    if group_col not in adata.obs.columns:
        raise KeyError(
            f"adata.obs is missing group column '{group_col}'. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    if layer is not None and layer not in adata.layers:
        raise KeyError(
            f"Requested layer '{layer}' not found in adata.layers. "
            f"Available layers: {list(adata.layers.keys())}"
        )
    if use_raw and layer is None and adata.raw is None:
        warnings.append(
            "use_raw=True but adata.raw is missing; pseudobulk would fall back to adata.X."
        )

    obs = adata.obs.copy()
    n_cells_total = int(adata.n_obs)
    if subset_col is not None or subset_value is not None:
        if not subset_col or subset_value is None:
            raise ValueError("subset_col and subset_value must be provided together.")
        if subset_col not in obs.columns:
            raise KeyError(
                f"adata.obs is missing subset column '{subset_col}'. "
                f"Available columns: {list(obs.columns)}"
            )
        keep = obs[subset_col].astype(str) == str(subset_value)
        if int(keep.sum()) == 0:
            raise ValueError(f"No cells matched {subset_col}={subset_value!r}.")
        obs = obs.loc[keep].copy()

    if obs.empty:
        raise ValueError("No cells available after optional subsetting.")

    if obs[sample_col].isna().any():
        raise ValueError(f"adata.obs sample column '{sample_col}' contains missing values.")
    if obs[group_col].isna().any():
        raise ValueError(f"adata.obs group column '{group_col}' contains missing values.")

    obs[sample_col] = obs[sample_col].astype(str)
    obs[group_col] = obs[group_col].astype(str)

    group_counts = obs.groupby(sample_col, observed=False)[group_col].nunique(dropna=False)
    bad = group_counts[group_counts > 1]
    if not bad.empty:
        raise ValueError(
            "Each pseudobulk sample must map to exactly one group label. "
            f"Conflicting sample IDs: {', '.join(bad.index.astype(str).tolist())}"
        )

    sample_cells = (
        obs.groupby(sample_col, observed=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    sample_groups = obs[[sample_col, group_col]].drop_duplicates(subset=[sample_col])
    sample_table = sample_cells.merge(sample_groups, on=sample_col, how="left", validate="1:1")
    retained = sample_table.loc[sample_table["n_cells"].astype(int) >= int(min_cells_per_sample)].copy()
    if retained.empty:
        raise ValueError(
            "No pseudobulk samples remain after applying min_cells_per_sample."
        )

    available_groups = set(retained[group_col].astype(str))
    if group_a not in available_groups:
        raise ValueError(
            f"group_a='{group_a}' not found among retained pseudobulk groups: {sorted(available_groups)}"
        )
    if group_b not in available_groups:
        raise ValueError(
            f"group_b='{group_b}' not found among retained pseudobulk groups: {sorted(available_groups)}"
        )

    n_group_a = int(retained[group_col].eq(group_a).sum())
    n_group_b = int(retained[group_col].eq(group_b).sum())
    if n_group_a < 2 or n_group_b < 2:
        raise ValueError(
            "Pseudobulk analysis requires at least 2 retained samples in each requested group. "
            f"Observed: {group_a}={n_group_a}, {group_b}={n_group_b}."
        )
    _warn_if_minimal_group_size(warnings, label=group_a, count=n_group_a)
    _warn_if_minimal_group_size(warnings, label=group_b, count=n_group_b)

    n_dropped_samples = int(len(sample_table) - len(retained))
    if n_dropped_samples > 0:
        warnings.append(
            f"{n_dropped_samples} sample(s) fall below min_cells_per_sample={min_cells_per_sample} and would be dropped."
        )

    n_other_groups = int(len(retained) - n_group_a - n_group_b)
    if n_other_groups > 0:
        warnings.append(
            f"{n_other_groups} retained pseudobulk sample(s) belong to groups outside the requested contrast and would be ignored."
        )

    summary = {
        "adata_path": str(adata_path),
        "n_cells_total": n_cells_total,
        "n_cells_after_subset": int(len(obs)),
        "n_genes": int(adata.n_vars),
        "n_samples_before_filter": int(sample_table[sample_col].nunique()),
        "n_samples_after_filter": int(retained[sample_col].nunique()),
        "min_cells_per_sample": int(min_cells_per_sample),
        "group_a": group_a,
        "group_b": group_b,
        "n_group_a": n_group_a,
        "n_group_b": n_group_b,
        "subset_col": subset_col,
        "subset_value": subset_value,
    }
    return ValidationReport(mode="scrna", summary=summary, warnings=warnings)
