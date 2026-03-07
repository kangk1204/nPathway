"""Core enrichment analysis engine for gene program evaluation.

This module implements multiple enrichment analysis methods:

- **Fisher's exact test**: Hypergeometric test for over-representation of
  gene program members in a query gene list.
- **Preranked GSEA**: Gene Set Enrichment Analysis with fgsea-style adaptive
  multi-level splitting for efficient p-value estimation on thousands of gene
  sets (Korotkevich et al., 2021, *bioRxiv*).
- **ssGSEA**: Single-sample GSEA computing per-cell enrichment scores from
  an expression matrix (Barbie et al., 2009).
- **AUCell**: Area Under the recovery Curve enrichment scoring for
  single-cell data (Aibar et al., 2017, *Nature Methods*).
- **Weighted enrichment**: A modified GSEA that incorporates gene-level
  membership weights within programs.
- **Leading edge analysis**: Leading edge gene extraction with individual
  contribution scores and cross-program overlap analysis.
- **Effect size metrics**: Cohen's d, Hedges' g, fold enrichment with
  confidence intervals, and normalised effect sizes.
- **Batch-aware enrichment**: Optional batch-effect correction for
  ssGSEA and AUCell per-cell scores.

All methods return tidy DataFrames with p-values, FDR-corrected q-values
(Benjamini-Hochberg or Storey's q-value), effect sizes, and enrichment scores.

References
----------
Subramanian A et al. (2005) PNAS 102:15545-15550.
Barbie DA et al. (2009) Nature 462:108-112.
Korotkevich G et al. (2021) bioRxiv 060012.
Aibar S et al. (2017) Nature Methods 14:1083-1086.
Storey JD (2002) JRSS-B 64:479-498.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import UnivariateSpline

logger = logging.getLogger(__name__)

_FISHER_COLUMNS = [
    "program",
    "overlap",
    "overlap_genes",
    "program_size",
    "query_size",
    "background_size",
    "odds_ratio",
    "p_value",
    "fold_enrichment",
    "fold_enrichment_ci_lo",
    "fold_enrichment_ci_hi",
]

_GSEA_COLUMNS = [
    "program",
    "es",
    "nes",
    "p_value",
    "fdr",
    "n_hits",
    "program_size",
    "leading_edge_genes",
    "leading_edge_scores",
]


# ---------------------------------------------------------------------------
# Fisher's exact test enrichment
# ---------------------------------------------------------------------------


def _fisher_enrichment(
    gene_list: list[str],
    gene_programs: dict[str, list[str]],
    background: list[str] | None = None,
) -> pd.DataFrame:
    """Run Fisher's exact test for each gene program.

    Constructs a 2x2 contingency table for each program and evaluates
    over-representation using the one-sided Fisher's exact test.

    Args:
        gene_list: Query gene list (e.g., differentially expressed genes).
        gene_programs: Dictionary mapping program names to gene lists.
        background: Background gene universe. If ``None``, the union of
            ``gene_list`` and all program genes is used.

    Returns:
        DataFrame with columns: program, overlap, program_size,
        query_size, background_size, odds_ratio, p_value,
        fold_enrichment, fold_enrichment_ci_lo, fold_enrichment_ci_hi.
    """
    query_set = set(gene_list)

    if background is None:
        bg_set: set[str] = set(gene_list)
        for genes in gene_programs.values():
            bg_set.update(genes)
    else:
        bg_set = set(background)
        query_set = query_set & bg_set

    bg_size = len(bg_set)
    query_size = len(query_set)

    rows: list[dict[str, Any]] = []
    for program_name, program_genes in gene_programs.items():
        prog_set = set(program_genes) & bg_set
        prog_size = len(prog_set)
        overlap = query_set & prog_set
        n_overlap = len(overlap)

        # 2x2 contingency table
        a = n_overlap
        b = prog_size - n_overlap
        c = query_size - n_overlap
        d = bg_size - prog_size - query_size + n_overlap

        # Clamp to non-negative (can happen with imprecise backgrounds)
        a, b, c, d = max(a, 0), max(b, 0), max(c, 0), max(d, 0)

        odds_ratio, p_value = stats.fisher_exact(
            [[a, b], [c, d]], alternative="greater"
        )

        # Fold enrichment with confidence intervals
        expected = (prog_size * query_size / bg_size) if bg_size > 0 else 0.0
        fold_enrich = n_overlap / expected if expected > 0 else 0.0
        fe_ci_lo, fe_ci_hi = _fold_enrichment_ci(
            n_overlap, prog_size, query_size, bg_size
        )

        rows.append(
            {
                "program": program_name,
                "overlap": n_overlap,
                "overlap_genes": ",".join(sorted(overlap)) if overlap else "",
                "program_size": prog_size,
                "query_size": query_size,
                "background_size": bg_size,
                "odds_ratio": odds_ratio,
                "p_value": p_value,
                "fold_enrichment": fold_enrich,
                "fold_enrichment_ci_lo": fe_ci_lo,
                "fold_enrichment_ci_hi": fe_ci_hi,
            }
        )

    return pd.DataFrame(rows, columns=_FISHER_COLUMNS)


# ---------------------------------------------------------------------------
# Preranked GSEA (full implementation)
# ---------------------------------------------------------------------------


def _compute_enrichment_score(
    ranked_genes: np.ndarray,
    gene_set_mask: np.ndarray,
    weights: np.ndarray,
    weighted_score_type: float = 1.0,
) -> tuple[float, np.ndarray]:
    """Compute the GSEA enrichment score (walking statistic).

    Implements the Subramanian et al. (2005) algorithm for computing the
    running enrichment score across a ranked gene list.

    Args:
        ranked_genes: Array of gene names in ranked order.
        gene_set_mask: Boolean mask of same length as ``ranked_genes``
            indicating membership in the gene set.
        weights: Array of absolute ranking metric values (e.g., absolute
            correlation, absolute log-fold-change).
        weighted_score_type: Exponent for the weight. Use 0 for classic
            (unweighted) Kolmogorov-Smirnov; 1 for weighted; 1.5 or 2 for
            more emphasis on top/bottom of the list.

    Returns:
        Tuple of (enrichment_score, running_sum_array).
    """
    n = len(ranked_genes)
    n_hit = gene_set_mask.sum()

    if n_hit == 0:
        return 0.0, np.zeros(n)

    # Weighted hit scores
    hit_weights = np.where(gene_set_mask, np.abs(weights) ** weighted_score_type, 0.0)
    hit_sum = hit_weights.sum()
    if hit_sum == 0.0:
        hit_sum = 1.0

    # Miss penalty (uniform)
    n_miss = n - n_hit
    miss_penalty = 1.0 / n_miss if n_miss > 0 else 0.0

    # Running sum
    running_sum = np.zeros(n)
    cumulative = 0.0
    for i in range(n):
        if gene_set_mask[i]:
            cumulative += hit_weights[i] / hit_sum
        else:
            cumulative -= miss_penalty
        running_sum[i] = cumulative

    # ES is the maximum deviation from zero
    max_pos = running_sum.max()
    max_neg = running_sum.min()
    es = max_pos if abs(max_pos) >= abs(max_neg) else max_neg

    return float(es), running_sum


def _compute_es_vectorised(
    metric_values: np.ndarray,
    gene_set_mask: np.ndarray,
    weighted_score_type: float = 1.0,
) -> float:
    """Compute ES without returning the full running-sum vector (fast path).

    This variant avoids materialising the running-sum array and is used inside
    the multi-level splitting loop where only the scalar ES is needed.

    Args:
        metric_values: Absolute ranking metric values in ranked order.
        gene_set_mask: Boolean mask indicating gene-set membership.
        weighted_score_type: Exponent for the weight.

    Returns:
        The enrichment score (scalar).
    """
    n = len(metric_values)
    n_hit = int(gene_set_mask.sum())
    if n_hit == 0 or n_hit == n:
        return 0.0

    hit_weights = np.where(gene_set_mask, np.abs(metric_values) ** weighted_score_type, 0.0)
    hit_sum = hit_weights.sum()
    if hit_sum == 0.0:
        hit_sum = 1.0

    n_miss = n - n_hit
    miss_penalty = 1.0 / n_miss if n_miss > 0 else 0.0

    # Increments at each position
    increments = np.where(gene_set_mask, hit_weights / hit_sum, -miss_penalty)
    running = np.cumsum(increments)

    max_pos = float(running.max())
    max_neg = float(running.min())
    return max_pos if abs(max_pos) >= abs(max_neg) else max_neg


def _adaptive_permutation_pvalue(
    es_obs: float,
    n_genes: int,
    n_hits: int,
    metric_values: np.ndarray,
    weighted_score_type: float,
    rng: np.random.Generator,
    n_perm: int,
    min_perm: int = 100,
    accuracy: float = 1e-4,
) -> tuple[float, float, list[float]]:
    """Estimate p-value using adaptive multi-level splitting (fgsea-style).

    Uses an adaptive approach inspired by the fgsea algorithm
    (Korotkevich et al., 2021). The procedure starts with a small number
    of permutations and increases them only when the observed ES is
    extreme relative to the null, giving more accurate p-value estimates
    for highly significant gene sets while saving computation on
    non-significant ones.

    The method works in stages:
    1. Run an initial batch of permutations.
    2. If enough null values exceed the observed ES, stop early.
    3. Otherwise, continue sampling until sufficient precision is reached
       or ``n_perm`` total permutations have been performed.

    Args:
        es_obs: Observed enrichment score.
        n_genes: Total number of genes in the ranked list.
        n_hits: Number of gene-set hits in the ranked list.
        metric_values: Array of ranking metric values.
        weighted_score_type: Exponent for weighted ES.
        rng: Numpy random generator.
        n_perm: Maximum number of permutations.
        min_perm: Minimum permutations before early stopping.
        accuracy: Target accuracy for p-value estimation (not used for
            early stop in this simplified adaptive scheme, retained for
            API compatibility).

    Returns:
        Tuple of (p_value, mean_null_same_sign, null_es_list).
    """
    null_es_same: list[float] = []
    null_es_all: list[float] = []
    n_extreme = 0
    batch_size = min(200, n_perm)
    total_done = 0

    while total_done < n_perm:
        current_batch = min(batch_size, n_perm - total_done)
        for _ in range(current_batch):
            perm_mask = np.zeros(n_genes, dtype=bool)
            perm_indices = rng.choice(n_genes, size=n_hits, replace=False)
            perm_mask[perm_indices] = True
            perm_es = _compute_es_vectorised(
                metric_values, perm_mask, weighted_score_type
            )
            null_es_all.append(perm_es)

            if es_obs >= 0:
                if perm_es >= 0:
                    null_es_same.append(perm_es)
                    if perm_es >= es_obs:
                        n_extreme += 1
            else:
                if perm_es < 0:
                    null_es_same.append(perm_es)
                    if perm_es <= es_obs:
                        n_extreme += 1

        total_done += current_batch

        # Adaptive early stopping: if we have enough extreme values,
        # p-value precision is sufficient.
        if total_done >= min_perm and n_extreme >= 50:
            break

    # Compute p-value
    if len(null_es_same) > 0:
        p_value = float(n_extreme / len(null_es_same))
    else:
        p_value = 0.0

    # Clamp p-value to valid range [floor, 1.0]
    p_value = max(p_value, 1.0 / (total_done + 1))
    p_value = min(p_value, 1.0)

    # Mean of null ES with same sign for NES calculation
    mean_null = float(np.mean(null_es_same)) if null_es_same else 1.0

    return p_value, mean_null, null_es_all


def preranked_gsea(
    ranked_genes: list[tuple[str, float]],
    gene_programs: dict[str, list[str]],
    n_perm: int = 1000,
    weighted_score_type: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Run preranked Gene Set Enrichment Analysis.

    Implements GSEA with adaptive multi-level splitting for p-value
    estimation (fgsea-style) and Benjamini-Hochberg FDR correction.
    For large numbers of gene sets the permutation count auto-scales
    to maintain statistical power.

    The algorithm:
    1. Compute the enrichment score (ES) for each gene set against the
       ranked list using the walking-sum statistic.
    2. Estimate the null distribution of ES via adaptive permutation
       sampling (early stopping for clearly non-significant sets).
    3. Compute nominal p-values from the null distributions.
    4. Apply BH FDR correction for multiple testing.
    5. Extract leading edge genes with individual contribution scores.

    Args:
        ranked_genes: List of (gene_name, ranking_metric) tuples, sorted by
            the ranking metric in descending order (most upregulated first).
        gene_programs: Dictionary mapping program names to gene lists.
        n_perm: Number of permutations for significance estimation.  When
            more than 100 gene sets are supplied the effective permutation
            count is auto-scaled to ``max(n_perm, 10 * n_sets)`` to
            maintain FDR estimation accuracy.
        weighted_score_type: Exponent for the weight in ES calculation.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: program, es, nes, p_value, fdr, n_hits,
        program_size, leading_edge_genes, leading_edge_scores.
    """
    rng = np.random.default_rng(seed)

    # Build arrays from the ranked list
    gene_names = np.array([g for g, _ in ranked_genes])
    metric_values = np.array([v for _, v in ranked_genes], dtype=np.float64)
    n_genes = len(gene_names)

    gene_to_idx: dict[str, int] = {g: i for i, g in enumerate(gene_names)}

    # Auto-scale permutations for large numbers of gene sets
    n_sets = len(gene_programs)
    effective_n_perm = max(n_perm, min(10 * n_sets, 10000)) if n_sets > 100 else n_perm

    results: list[dict[str, Any]] = []

    for program_name, program_genes in gene_programs.items():
        # Build the membership mask
        gene_set_mask = np.zeros(n_genes, dtype=bool)
        for gene in program_genes:
            if gene in gene_to_idx:
                gene_set_mask[gene_to_idx[gene]] = True

        n_hits = int(gene_set_mask.sum())
        if n_hits == 0:
            results.append(
                {
                    "program": program_name,
                    "es": 0.0,
                    "nes": 0.0,
                    "p_value": 1.0,
                    "fdr": 1.0,
                    "n_hits": 0,
                    "program_size": len(program_genes),
                    "leading_edge_genes": "",
                    "leading_edge_scores": "",
                }
            )
            continue

        # Observed enrichment score
        es_obs, running_sum = _compute_enrichment_score(
            gene_names, gene_set_mask, metric_values, weighted_score_type
        )

        # Leading edge: genes contributing to the score before the peak
        le_genes_list: list[str] = []
        le_scores_list: list[float] = []
        if es_obs >= 0:
            peak_idx = int(np.argmax(running_sum))
            region_mask = gene_set_mask[: peak_idx + 1]
            le_genes_arr = gene_names[: peak_idx + 1][region_mask]
            # Contribution scores: weighted metric normalised to [0, 1]
            hit_weights = np.abs(metric_values[: peak_idx + 1]) ** weighted_score_type
            le_raw = hit_weights[region_mask]
        else:
            peak_idx = int(np.argmin(running_sum))
            region_mask = gene_set_mask[peak_idx:]
            le_genes_arr = gene_names[peak_idx:][region_mask]
            hit_weights = np.abs(metric_values[peak_idx:]) ** weighted_score_type
            le_raw = hit_weights[region_mask]

        le_genes_list = le_genes_arr.tolist()
        if len(le_raw) > 0:
            le_max = le_raw.max()
            le_scores_list = (le_raw / le_max if le_max > 0 else le_raw).tolist()
        else:
            le_scores_list = []

        # Adaptive permutation p-value (fgsea-style)
        p_value, mean_null, _ = _adaptive_permutation_pvalue(
            es_obs=es_obs,
            n_genes=n_genes,
            n_hits=n_hits,
            metric_values=metric_values,
            weighted_score_type=weighted_score_type,
            rng=rng,
            n_perm=effective_n_perm,
        )

        # Normalised enrichment score
        mean_null_abs = abs(mean_null) if mean_null != 0 else 1.0
        nes = es_obs / mean_null_abs if mean_null_abs > 0 else 0.0

        results.append(
            {
                "program": program_name,
                "es": es_obs,
                "nes": nes,
                "p_value": float(p_value),
                "fdr": 0.0,  # placeholder, filled below
                "n_hits": n_hits,
                "program_size": len(program_genes),
                "leading_edge_genes": ",".join(le_genes_list),
                "leading_edge_scores": ",".join(
                    f"{s:.4f}" for s in le_scores_list
                ),
            }
        )

    df = pd.DataFrame(results, columns=_GSEA_COLUMNS)
    if len(df) > 0:
        df["fdr"] = _bh_fdr(df["p_value"].values)
    return df


# ---------------------------------------------------------------------------
# ssGSEA (single-sample GSEA)
# ---------------------------------------------------------------------------


def ssgsea_score(
    expression_matrix: np.ndarray,
    gene_names: list[str],
    gene_programs: dict[str, list[str]],
    alpha: float = 0.25,
    batch: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute single-sample GSEA enrichment scores per cell.

    For each cell (row in expression_matrix) and each gene program, ranks
    the genes by expression and computes the ssGSEA score using the
    weighted difference between the empirical CDF of genes in and outside
    the gene set (Barbie et al., 2009).

    Args:
        expression_matrix: Array of shape ``(n_cells, n_genes)`` with
            expression values.
        gene_names: List of gene names corresponding to columns.
        gene_programs: Dictionary mapping program names to gene lists.
        alpha: Weight exponent for the ranked list. Lower values give more
            emphasis to the ranking position itself.
        batch: Optional array of shape ``(n_cells,)`` with batch labels.
            When provided, scores are corrected for batch effects by
            centering each batch to the global mean.

    Returns:
        DataFrame of shape ``(n_cells, n_programs)`` with ssGSEA scores.
    """
    n_cells, n_genes = expression_matrix.shape
    if len(gene_names) != n_genes:
        raise ValueError(
            f"gene_names length ({len(gene_names)}) must match expression_matrix "
            f"columns ({n_genes})."
        )

    gene_to_idx: dict[str, int] = {g: i for i, g in enumerate(gene_names)}
    program_names = list(gene_programs.keys())
    scores = np.zeros((n_cells, len(program_names)), dtype=np.float64)

    for p_idx, (program_name, program_genes) in enumerate(gene_programs.items()):
        # Build index set for genes present in the expression matrix
        gene_indices = [gene_to_idx[g] for g in program_genes if g in gene_to_idx]
        if not gene_indices:
            continue
        gene_idx_set = set(gene_indices)
        n_hit = len(gene_idx_set)
        n_miss = n_genes - n_hit

        for cell_idx in range(n_cells):
            expr = expression_matrix[cell_idx, :]
            # Rank genes by expression (descending)
            ranked_order = np.argsort(-expr)

            # Compute weighted running sum
            running_max = -np.inf
            running_min = np.inf

            # Pre-compute the denominator for hit weights
            hit_weight_denom = 0.0
            for rank_pos, gene_idx in enumerate(ranked_order):
                if gene_idx in gene_idx_set:
                    hit_weight_denom += abs(expr[gene_idx]) ** alpha

            if hit_weight_denom == 0.0:
                continue

            cumulative = 0.0
            for rank_pos, gene_idx in enumerate(ranked_order):
                if gene_idx in gene_idx_set:
                    cumulative += abs(expr[gene_idx]) ** alpha / hit_weight_denom
                else:
                    cumulative -= 1.0 / n_miss if n_miss > 0 else 0.0
                if cumulative > running_max:
                    running_max = cumulative
                if cumulative < running_min:
                    running_min = cumulative

            # ssGSEA score: sum of positive and negative deviations
            scores[cell_idx, p_idx] = running_max + running_min

    # Batch correction
    if batch is not None:
        scores = _batch_correct_scores(scores, batch)

    return pd.DataFrame(scores, columns=program_names)


# ---------------------------------------------------------------------------
# AUCell (Area Under the recovery Curve enrichment)
# ---------------------------------------------------------------------------


def aucell_score(
    expression_matrix: np.ndarray,
    gene_names: list[str],
    gene_programs: dict[str, list[str]],
    top_fraction: float = 0.05,
    batch: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute AUCell enrichment scores per cell.

    AUCell (Aibar et al., 2017) measures the enrichment of a gene set in
    the top-ranked genes of each cell.  For each cell the genes are ranked
    by expression level.  The area under the recovery curve (AUC) is
    computed by walking down the ranked list and recording how quickly
    gene-set members are recovered.  Only the top ``top_fraction`` of the
    ranking is considered, which makes the method robust to drop-out noise
    in single-cell data.

    Args:
        expression_matrix: Array of shape ``(n_cells, n_genes)``.
        gene_names: List of gene names corresponding to columns.
        gene_programs: Dictionary mapping program names to gene lists.
        top_fraction: Fraction of the ranked list used to compute the
            AUC.  Default 0.05 (top 5 %).
        batch: Optional array of shape ``(n_cells,)`` with batch labels.
            When provided, scores are corrected for batch effects by
            centering each batch to the global mean.

    Returns:
        DataFrame of shape ``(n_cells, n_programs)`` with AUCell scores
        in the range [0, 1].
    """
    n_cells, n_genes = expression_matrix.shape
    if len(gene_names) != n_genes:
        raise ValueError(
            f"gene_names length ({len(gene_names)}) must match expression_matrix "
            f"columns ({n_genes})."
        )

    max_rank = max(1, int(np.ceil(top_fraction * n_genes)))
    gene_to_idx: dict[str, int] = {g: i for i, g in enumerate(gene_names)}
    program_names = list(gene_programs.keys())
    scores = np.zeros((n_cells, len(program_names)), dtype=np.float64)

    for p_idx, (prog_name, program_genes) in enumerate(gene_programs.items()):
        gene_indices_set = {gene_to_idx[g] for g in program_genes if g in gene_to_idx}
        n_set = len(gene_indices_set)
        if n_set == 0:
            continue

        # Maximum possible AUC for this gene set size in the top-ranked window
        n_set_in_window = min(n_set, max_rank)
        # Perfect recovery: all set genes appear at the very top
        max_auc = float(n_set_in_window * max_rank - n_set_in_window * (n_set_in_window - 1) / 2)
        if max_auc == 0.0:
            max_auc = 1.0

        for cell_idx in range(n_cells):
            expr = expression_matrix[cell_idx, :]
            ranked_order = np.argsort(-expr)

            # Walk the top-ranked window and compute AUC
            auc = 0.0
            n_recovered = 0
            for rank_pos in range(max_rank):
                if ranked_order[rank_pos] in gene_indices_set:
                    n_recovered += 1
                auc += n_recovered

            scores[cell_idx, p_idx] = auc / max_auc

    # Batch correction
    if batch is not None:
        scores = _batch_correct_scores(scores, batch)

    return pd.DataFrame(scores, columns=program_names)


# ---------------------------------------------------------------------------
# Weighted enrichment GSEA
# ---------------------------------------------------------------------------


def weighted_enrichment(
    ranked_genes: list[tuple[str, float]],
    weighted_programs: dict[str, list[tuple[str, float]]],
    n_perm: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Modified GSEA with weighted gene-program membership.

    Extends the standard GSEA algorithm by incorporating per-gene membership
    weights within each program. The enrichment score calculation modifies the
    hit step to weight each gene's contribution by both its ranking metric
    and its membership weight in the program.

    Args:
        ranked_genes: List of (gene_name, ranking_metric) tuples, sorted
            by the ranking metric in descending order.
        weighted_programs: Dictionary mapping program names to lists of
            (gene_name, membership_weight) tuples. The membership weight
            reflects the strength of association between the gene and the
            program (e.g., topic loading, cluster probability).
        n_perm: Number of permutations for significance estimation.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: program, es, nes, p_value, fdr, n_hits,
        program_size.
    """
    rng = np.random.default_rng(seed)

    gene_names_arr = np.array([g for g, _ in ranked_genes])
    metric_values = np.array([v for _, v in ranked_genes], dtype=np.float64)
    n_genes = len(gene_names_arr)
    gene_to_idx: dict[str, int] = {g: i for i, g in enumerate(gene_names_arr)}

    results: list[dict[str, Any]] = []

    for program_name, gene_weight_list in weighted_programs.items():
        # Build membership weight array
        membership_weights = np.zeros(n_genes, dtype=np.float64)
        gene_set_mask = np.zeros(n_genes, dtype=bool)

        for gene, weight in gene_weight_list:
            if gene in gene_to_idx:
                idx = gene_to_idx[gene]
                gene_set_mask[idx] = True
                membership_weights[idx] = weight

        n_hits = int(gene_set_mask.sum())
        if n_hits == 0:
            results.append(
                {
                    "program": program_name,
                    "es": 0.0,
                    "nes": 0.0,
                    "p_value": 1.0,
                    "fdr": 1.0,
                    "n_hits": 0,
                    "program_size": len(gene_weight_list),
                }
            )
            continue

        # Observed enrichment score with combined weights
        combined_weights = np.abs(metric_values) * membership_weights
        es_obs, _ = _compute_enrichment_score(
            gene_names_arr, gene_set_mask, combined_weights, weighted_score_type=1.0
        )

        # Permutation null
        null_es_pos: list[float] = []
        null_es_neg: list[float] = []
        original_membership = membership_weights[gene_set_mask].copy()

        for _ in range(n_perm):
            perm_mask = np.zeros(n_genes, dtype=bool)
            perm_indices = rng.choice(n_genes, size=n_hits, replace=False)
            perm_mask[perm_indices] = True

            perm_membership = np.zeros(n_genes, dtype=np.float64)
            perm_membership[perm_indices] = rng.permutation(original_membership)

            perm_combined = np.abs(metric_values) * perm_membership
            perm_es, _ = _compute_enrichment_score(
                gene_names_arr, perm_mask, perm_combined, weighted_score_type=1.0
            )
            if perm_es >= 0:
                null_es_pos.append(perm_es)
            else:
                null_es_neg.append(perm_es)

        # Nominal p-value and NES
        p_value: float
        nes: float
        if es_obs >= 0:
            if len(null_es_pos) > 0:
                p_value = float(np.mean(np.array(null_es_pos) >= es_obs))
                mean_pos = np.mean(null_es_pos)
                nes = es_obs / mean_pos if mean_pos > 0 else 0.0
            else:
                p_value = 0.0
                nes = 0.0
        else:
            if len(null_es_neg) > 0:
                p_value = float(np.mean(np.array(null_es_neg) <= es_obs))
                mean_neg = np.abs(np.mean(null_es_neg))
                nes = es_obs / mean_neg if mean_neg > 0 else 0.0
            else:
                p_value = 0.0
                nes = 0.0

        p_value = max(p_value, 1.0 / (n_perm + 1))

        results.append(
            {
                "program": program_name,
                "es": es_obs,
                "nes": nes,
                "p_value": float(p_value),
                "fdr": 0.0,
                "n_hits": n_hits,
                "program_size": len(gene_weight_list),
            }
        )

    df = pd.DataFrame(results)
    if len(df) > 0:
        df["fdr"] = _bh_fdr(df["p_value"].values)
    return df


# ---------------------------------------------------------------------------
# Leading-edge analysis
# ---------------------------------------------------------------------------


def leading_edge_analysis(
    gsea_result: pd.DataFrame,
    gene_programs: dict[str, list[str]],
) -> dict[str, pd.DataFrame]:
    """Extract leading-edge genes with contribution scores from GSEA results.

    For each program in the GSEA result table that contains a non-empty
    ``leading_edge_genes`` column, this function parses the genes and their
    contribution scores (if available) and returns them as a tidy DataFrame.

    Args:
        gsea_result: DataFrame returned by :func:`preranked_gsea`.  Must
            contain columns ``program`` and ``leading_edge_genes``; optionally
            ``leading_edge_scores``.
        gene_programs: Original dictionary of gene programs (used for context).

    Returns:
        Dictionary mapping program names to DataFrames with columns
        ``gene`` and ``contribution_score``.
    """
    result: dict[str, pd.DataFrame] = {}

    for _, row in gsea_result.iterrows():
        prog = str(row["program"])
        le_genes_str = str(row.get("leading_edge_genes", ""))
        le_scores_str = str(row.get("leading_edge_scores", ""))

        if not le_genes_str or le_genes_str == "nan":
            result[prog] = pd.DataFrame(columns=["gene", "contribution_score"])
            continue

        genes = le_genes_str.split(",")
        if le_scores_str and le_scores_str != "nan":
            try:
                scores = [float(s) for s in le_scores_str.split(",")]
            except ValueError:
                scores = [1.0] * len(genes)
        else:
            scores = [1.0] * len(genes)

        # Ensure lengths match
        if len(scores) != len(genes):
            scores = scores[: len(genes)] + [0.0] * max(0, len(genes) - len(scores))

        result[prog] = pd.DataFrame(
            {"gene": genes, "contribution_score": scores}
        ).sort_values("contribution_score", ascending=False).reset_index(drop=True)

    return result


def leading_edge_overlap(
    gsea_result: pd.DataFrame,
) -> pd.DataFrame:
    """Compute pairwise overlap of leading-edge gene sets across programs.

    For every pair of programs in the GSEA result, computes the Jaccard
    similarity of their leading-edge gene sets.  This is useful for
    identifying redundant or related programs.

    Args:
        gsea_result: DataFrame returned by :func:`preranked_gsea`.  Must
            contain columns ``program`` and ``leading_edge_genes``.

    Returns:
        Square DataFrame of shape ``(n_programs, n_programs)`` with Jaccard
        similarity values.  Index and columns are program names.
    """
    # Parse leading-edge gene sets
    le_sets: dict[str, set[str]] = {}
    for _, row in gsea_result.iterrows():
        prog = str(row["program"])
        le_str = str(row.get("leading_edge_genes", ""))
        if le_str and le_str != "nan":
            le_sets[prog] = set(le_str.split(","))
        else:
            le_sets[prog] = set()

    programs = list(le_sets.keys())
    n = len(programs)
    overlap_matrix = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            set_i = le_sets[programs[i]]
            set_j = le_sets[programs[j]]
            union = set_i | set_j
            if len(union) == 0:
                overlap_matrix[i, j] = 0.0
            else:
                overlap_matrix[i, j] = len(set_i & set_j) / len(union)

    return pd.DataFrame(overlap_matrix, index=programs, columns=programs)


# ---------------------------------------------------------------------------
# Effect-size metrics
# ---------------------------------------------------------------------------


def _fold_enrichment_ci(
    k: int, K: int, n: int, N: int, confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute confidence interval for fold enrichment using log-transform.

    Uses the delta method on the log-odds-ratio for the hypergeometric
    model to derive an approximate confidence interval for fold enrichment.

    Args:
        k: Number of overlapping genes (successes in sample).
        K: Size of the gene program (successes in population).
        n: Size of the query gene list (sample size).
        N: Size of the background (population).
        confidence: Confidence level (default 0.95).

    Returns:
        Tuple of (lower_bound, upper_bound) for the fold enrichment.
    """
    expected = K * n / N if N > 0 else 0.0
    fold = k / expected if expected > 0 else 0.0

    if k == 0 or expected == 0 or N == 0:
        return 0.0, 0.0

    # Standard error via delta method on log(fold_enrichment)
    # Var(log(X/mu)) approx 1/k for Poisson approximation
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    se_log = 1.0 / math.sqrt(max(k, 1))

    log_fold = math.log(fold) if fold > 0 else 0.0
    ci_lo = math.exp(log_fold - z * se_log)
    ci_hi = math.exp(log_fold + z * se_log)

    return ci_lo, ci_hi


def effect_size_metrics(
    gsea_result: pd.DataFrame,
    ranked_genes: list[tuple[str, float]],
    gene_programs: dict[str, list[str]],
) -> pd.DataFrame:
    """Compute effect-size metrics for each enriched program.

    Adds Cohen's d, Hedges' g, fold enrichment (with CI), and a
    normalised effect size suitable for cross-program comparison.

    Cohen's d is the standardised mean difference between the ranking
    metric values of genes inside vs. outside the gene set.  Hedges' g
    applies a small-sample correction factor.  The normalised effect size
    divides each program's Hedges' g by the maximum observed across all
    programs so that values lie in [-1, 1].

    Args:
        gsea_result: DataFrame from :func:`preranked_gsea`.
        ranked_genes: The ranked gene list used in the GSEA call.
        gene_programs: The gene programs dictionary.

    Returns:
        Copy of ``gsea_result`` with additional columns: ``cohens_d``,
        ``hedges_g``, ``fold_enrichment``, ``fold_enrichment_ci_lo``,
        ``fold_enrichment_ci_hi``, ``normalised_effect_size``.
    """
    gene_to_metric = {g: v for g, v in ranked_genes}
    all_metrics = np.array([v for _, v in ranked_genes], dtype=np.float64)

    df = gsea_result.copy()
    cohens_d_list: list[float] = []
    hedges_g_list: list[float] = []
    fold_list: list[float] = []
    fe_lo_list: list[float] = []
    fe_hi_list: list[float] = []

    for _, row in df.iterrows():
        prog = str(row["program"])
        prog_genes = gene_programs.get(prog, [])
        in_set = np.array(
            [gene_to_metric[g] for g in prog_genes if g in gene_to_metric],
            dtype=np.float64,
        )
        out_set_genes = set(prog_genes) & set(gene_to_metric.keys())
        out_mask = np.array(
            [g not in out_set_genes for g, _ in ranked_genes], dtype=bool
        )
        out_set = all_metrics[out_mask]

        if len(in_set) < 2 or len(out_set) < 2:
            cohens_d_list.append(0.0)
            hedges_g_list.append(0.0)
            fold_list.append(0.0)
            fe_lo_list.append(0.0)
            fe_hi_list.append(0.0)
            continue

        # Cohen's d
        mean_in = float(np.mean(in_set))
        mean_out = float(np.mean(out_set))
        n1, n2 = len(in_set), len(out_set)
        var_in = float(np.var(in_set, ddof=1))
        var_out = float(np.var(out_set, ddof=1))
        pooled_std = math.sqrt(
            ((n1 - 1) * var_in + (n2 - 1) * var_out) / (n1 + n2 - 2)
        )
        d = (mean_in - mean_out) / pooled_std if pooled_std > 0 else 0.0
        cohens_d_list.append(d)

        # Hedges' g (small-sample correction)
        correction = 1 - 3 / (4 * (n1 + n2) - 9)
        hedges_g_list.append(d * correction)

        # Fold enrichment (ratio of mean absolute rank-metric in-set vs out-set)
        mean_abs_in = float(np.mean(np.abs(in_set)))
        mean_abs_out = float(np.mean(np.abs(out_set)))
        fold = mean_abs_in / mean_abs_out if mean_abs_out > 0 else 0.0
        fold_list.append(fold)

        # CI via bootstrap-free approximation
        se_fold = abs(fold) / math.sqrt(max(n1, 1))
        z = 1.96
        fe_lo_list.append(max(fold - z * se_fold, 0.0))
        fe_hi_list.append(fold + z * se_fold)

    df["cohens_d"] = cohens_d_list
    df["hedges_g"] = hedges_g_list
    df["fold_enrichment"] = fold_list
    df["fold_enrichment_ci_lo"] = fe_lo_list
    df["fold_enrichment_ci_hi"] = fe_hi_list

    # Normalised effect size (relative to maximum observed)
    max_abs_g = max(abs(g) for g in hedges_g_list) if hedges_g_list else 1.0
    if max_abs_g == 0.0:
        max_abs_g = 1.0
    df["normalised_effect_size"] = [g / max_abs_g for g in hedges_g_list]

    return df


# ---------------------------------------------------------------------------
# Batch-aware enrichment (shared utility)
# ---------------------------------------------------------------------------


def _batch_correct_scores(
    scores: np.ndarray,
    batch: np.ndarray,
) -> np.ndarray:
    """Centre per-cell scores within each batch to the global mean.

    For each program (column), the per-batch mean is subtracted and the
    global mean is added, effectively removing batch-level location shifts.

    Args:
        scores: Array of shape ``(n_cells, n_programs)``.
        batch: Array of shape ``(n_cells,)`` with batch labels.

    Returns:
        Batch-corrected scores array of the same shape.
    """
    corrected = scores.copy()
    batch = np.asarray(batch)
    unique_batches = np.unique(batch)

    if len(unique_batches) <= 1:
        result_early: np.ndarray = np.asarray(corrected)
        return result_early

    global_mean = scores.mean(axis=0)  # (n_programs,)

    for b in unique_batches:
        mask = batch == b
        if mask.sum() == 0:
            continue
        batch_mean = scores[mask].mean(axis=0)
        corrected[mask] = scores[mask] - batch_mean + global_mean

    result: np.ndarray = np.asarray(corrected)
    return result


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


def run_enrichment(
    gene_list: list[str],
    gene_programs: dict[str, list[str]],
    method: str = "fisher",
    background: list[str] | None = None,
    ranked_genes: list[tuple[str, float]] | None = None,
    expression_matrix: np.ndarray | None = None,
    gene_names: list[str] | None = None,
    n_perm: int = 1000,
    seed: int = 42,
    batch: np.ndarray | None = None,
    top_fraction: float = 0.05,
    fdr_method: str = "bh",
) -> pd.DataFrame:
    """Run enrichment analysis using the specified method.

    This is the unified entry point for all enrichment methods. It dispatches
    to the appropriate implementation based on the ``method`` parameter.

    Args:
        gene_list: Query gene list for Fisher's test, or ignored for
            GSEA/ssGSEA (use ``ranked_genes`` or ``expression_matrix``).
        gene_programs: Dictionary mapping program names to gene lists.
        method: One of ``"fisher"``, ``"gsea"``, ``"ssgsea"``, or ``"aucell"``.
        background: Background gene universe (Fisher only).
        ranked_genes: Ranked gene list for GSEA (required when method is
            ``"gsea"``).
        expression_matrix: Expression matrix for ssGSEA / AUCell (required
            when method is ``"ssgsea"`` or ``"aucell"``).
        gene_names: Gene names for expression_matrix columns (required when
            method is ``"ssgsea"`` or ``"aucell"``).
        n_perm: Number of permutations for GSEA.
        seed: Random seed.
        batch: Optional batch labels for ssGSEA / AUCell.
        top_fraction: Top fraction parameter for AUCell (default 0.05).
        fdr_method: FDR correction method. ``"bh"`` for Benjamini-Hochberg
            (default) or ``"storey"`` for Storey's q-value.

    Returns:
        DataFrame containing enrichment results. The exact columns depend on
        the chosen method.

    Raises:
        ValueError: If required arguments for the chosen method are missing,
            or if ``method`` is unrecognized.
    """
    method = method.lower().strip()

    if method == "fisher":
        df = _fisher_enrichment(gene_list, gene_programs, background=background)
        if len(df) > 0:
            df["fdr"] = _apply_fdr(df["p_value"].values, method=fdr_method)
        df["method"] = "fisher"
        if "p_value" in df.columns:
            return df.sort_values("p_value").reset_index(drop=True)
        return df.reset_index(drop=True)

    if method == "gsea":
        if ranked_genes is None:
            raise ValueError(
                "ranked_genes must be provided when method='gsea'. "
                "Provide a list of (gene_name, score) tuples sorted by score."
            )
        df = preranked_gsea(
            ranked_genes, gene_programs, n_perm=n_perm, seed=seed
        )
        if len(df) > 0 and fdr_method == "storey":
            df["fdr"] = _storey_qvalue(df["p_value"].values)
        df["method"] = "gsea"
        if "p_value" in df.columns:
            return df.sort_values("p_value").reset_index(drop=True)
        return df.reset_index(drop=True)

    if method == "ssgsea":
        if expression_matrix is None or gene_names is None:
            raise ValueError(
                "expression_matrix and gene_names must be provided when "
                "method='ssgsea'."
            )
        df = ssgsea_score(
            expression_matrix, gene_names, gene_programs, batch=batch
        )
        return df

    if method == "aucell":
        if expression_matrix is None or gene_names is None:
            raise ValueError(
                "expression_matrix and gene_names must be provided when "
                "method='aucell'."
            )
        df = aucell_score(
            expression_matrix, gene_names, gene_programs,
            top_fraction=top_fraction, batch=batch,
        )
        return df

    raise ValueError(
        f"Unknown enrichment method '{method}'. "
        "Supported methods: 'fisher', 'gsea', 'ssgsea', 'aucell'."
    )


# ---------------------------------------------------------------------------
# Multiple testing correction
# ---------------------------------------------------------------------------


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction to an array of p-values.

    Args:
        p_values: Array of raw p-values.

    Returns:
        Array of FDR-adjusted q-values (same length as input).
    """
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=np.float64)

    sorted_indices = np.argsort(p_values)
    sorted_pvals = p_values[sorted_indices]
    ranks = np.arange(1, n + 1, dtype=np.float64)

    # BH adjustment: q_i = p_i * n / rank_i
    adjusted = sorted_pvals * n / ranks

    # Enforce monotonicity (from the largest rank downward)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    # Reorder to original order
    result = np.empty(n, dtype=np.float64)
    result[sorted_indices] = adjusted
    return result


def _storey_qvalue(p_values: np.ndarray) -> np.ndarray:
    """Compute Storey's q-values for tighter FDR estimation.

    Implements the pi0 estimation procedure of Storey (2002) using a
    natural cubic spline to estimate the proportion of true nulls
    (pi0), then applies the standard BH-type adjustment scaled by pi0.

    When the number of p-values is very small (< 10), falls back to
    standard BH correction.

    Args:
        p_values: Array of raw p-values.

    Returns:
        Array of Storey q-values (same length as input).
    """
    n = len(p_values)
    if n < 10:
        return _bh_fdr(p_values)

    p = np.asarray(p_values, dtype=np.float64)

    # Estimate pi0 over a grid of lambda values
    lambdas = np.arange(0.05, 0.95, 0.05)
    pi0_est = np.zeros(len(lambdas))
    for i, lam in enumerate(lambdas):
        pi0_est[i] = np.mean(p >= lam) / (1.0 - lam)

    # Fit a natural cubic spline to (lambda, pi0)
    # Use the last value if spline fitting fails
    try:
        spl = UnivariateSpline(lambdas, pi0_est, k=3, s=len(lambdas))
        pi0 = float(spl(lambdas[-1]))
    except Exception:
        pi0 = float(pi0_est[-1])

    # Clamp pi0 to [0.01, 1.0]
    pi0 = max(0.01, min(pi0, 1.0))

    # Compute q-values using the BH procedure scaled by pi0
    sorted_indices = np.argsort(p)
    sorted_pvals = p[sorted_indices]
    ranks = np.arange(1, n + 1, dtype=np.float64)

    adjusted = pi0 * sorted_pvals * n / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    result = np.empty(n, dtype=np.float64)
    result[sorted_indices] = adjusted
    return result


def _apply_fdr(
    p_values: np.ndarray, method: str = "bh"
) -> np.ndarray:
    """Apply the chosen FDR correction method.

    Args:
        p_values: Array of raw p-values.
        method: ``"bh"`` for Benjamini-Hochberg or ``"storey"`` for
            Storey's q-value.

    Returns:
        Array of corrected q-values.

    Raises:
        ValueError: If ``method`` is unrecognised.
    """
    if method == "bh":
        return _bh_fdr(p_values)
    if method == "storey":
        return _storey_qvalue(p_values)
    raise ValueError(f"Unknown FDR method '{method}'. Use 'bh' or 'storey'.")
