"""Shared metrics for gene program evaluation.

This module provides reusable metric functions used across all benchmark
classes, including set-overlap measures, biological coherence scores,
redundancy, coverage, novelty metrics, stability analysis, bootstrap
confidence intervals, and permutation-based null hypothesis testing.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from itertools import combinations
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def jaccard_similarity(set1: set[str], set2: set[str]) -> float:
    """Compute the Jaccard similarity index between two sets.

    The Jaccard index is defined as ``|A inter B| / |A union B|``.  Returns
    0.0 when both sets are empty.

    Args:
        set1: First set of gene symbols.
        set2: Second set of gene symbols.

    Returns:
        Jaccard similarity coefficient in [0, 1].
    """
    if not set1 and not set2:
        return 0.0
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union)


def compute_overlap_matrix(
    programs1: dict[str, list[str]],
    programs2: dict[str, list[str]],
) -> pd.DataFrame:
    """Compute pairwise Jaccard similarity between two program collections.

    Args:
        programs1: First collection mapping program names to gene lists.
        programs2: Second collection mapping program names to gene lists.

    Returns:
        A DataFrame of shape ``(len(programs1), len(programs2))`` containing
        pairwise Jaccard similarities.
    """
    names1 = list(programs1.keys())
    names2 = list(programs2.keys())
    matrix = np.zeros((len(names1), len(names2)), dtype=np.float64)
    for i, n1 in enumerate(names1):
        s1 = set(programs1[n1])
        for j, n2 in enumerate(names2):
            s2 = set(programs2[n2])
            matrix[i, j] = jaccard_similarity(s1, s2)
    return pd.DataFrame(matrix, index=names1, columns=names2)


def biological_coherence(
    gene_set: list[str],
    ppi_network: Any | None = None,
) -> float:
    """Measure biological coherence of a gene set using PPI connectivity.

    Coherence is defined as the fraction of possible edges among genes in the
    set that are present in the protein-protein interaction network.  When no
    PPI network is provided, returns ``0.0``.

    Args:
        gene_set: List of gene symbols to evaluate.
        ppi_network: A ``networkx.Graph`` representing the PPI network. Nodes
            should be gene symbols. If ``None``, the function returns 0.0.

    Returns:
        Connectivity ratio in [0, 1]. Higher values indicate that genes in the
        set are more densely connected in the PPI network.
    """
    if ppi_network is None or nx is None:
        return 0.0

    genes_in_network = [g for g in gene_set if ppi_network.has_node(g)]
    n = len(genes_in_network)
    if n < 2:
        return 0.0

    possible_edges = n * (n - 1) / 2
    actual_edges = 0
    for g1, g2 in combinations(genes_in_network, 2):
        if ppi_network.has_edge(g1, g2):
            actual_edges += 1

    return actual_edges / possible_edges


def program_redundancy(programs: dict[str, list[str]]) -> float:
    """Measure redundancy across gene programs via mean pairwise Jaccard.

    A high redundancy score indicates that programs share many genes, which
    may signal that the programs are not well-separated.

    Args:
        programs: Dictionary mapping program names to gene lists.

    Returns:
        Mean pairwise Jaccard similarity across all program pairs. Returns
        0.0 if fewer than two programs are provided.
    """
    names = list(programs.keys())
    if len(names) < 2:
        return 0.0

    similarities: list[float] = []
    for i, j in combinations(range(len(names)), 2):
        s1 = set(programs[names[i]])
        s2 = set(programs[names[j]])
        similarities.append(jaccard_similarity(s1, s2))

    return float(np.mean(similarities))


def coverage(
    programs: dict[str, list[str]],
    reference_genes: list[str],
) -> float:
    """Compute the fraction of reference genes covered by the programs.

    Args:
        programs: Dictionary mapping program names to gene lists.
        reference_genes: List of reference gene symbols (e.g., all genes
            in the dataset).

    Returns:
        Fraction of reference genes that appear in at least one program,
        in [0, 1]. Returns 0.0 if the reference list is empty.
    """
    if not reference_genes:
        return 0.0
    all_program_genes: set[str] = set()
    for gene_list in programs.values():
        all_program_genes.update(gene_list)
    ref_set = set(reference_genes)
    covered = all_program_genes & ref_set
    return len(covered) / len(ref_set)


def novelty_score(
    programs: dict[str, list[str]],
    curated: dict[str, list[str]],
) -> float:
    """Compute the fraction of gene-program memberships not covered by curated sets.

    For each gene-program membership (a gene appearing in a discovered
    program), checks whether that gene appears in **any** curated pathway.
    Novelty measures how much new biology the learned programs capture
    beyond what is already in curated pathway databases.

    Note: this is a gene-level check — if gene X appears in both curated
    and discovered sets, all its program memberships are counted as
    "known" regardless of which specific curated pathway contains it.

    Args:
        programs: Learned gene programs (program name -> gene list).
        curated: Curated pathway gene sets (pathway name -> gene list).

    Returns:
        Fraction of gene-program memberships whose gene does not appear
        in any curated pathway, in [0, 1]. Returns 0.0 if programs
        are empty.
    """
    # Build a set of all genes present in any curated pathway
    curated_edges: set[str] = set()
    for pathway_name, genes in curated.items():
        for gene in genes:
            curated_edges.add(gene)  # gene-level membership

    total_edges = 0
    novel_edges = 0
    for _program_name, genes in programs.items():
        for gene in genes:
            total_edges += 1
            if gene not in curated_edges:
                novel_edges += 1

    if total_edges == 0:
        return 0.0
    return novel_edges / total_edges


def adjusted_rand_index(
    labels1: list[int] | np.ndarray,
    labels2: list[int] | np.ndarray,
) -> float:
    """Compute the Adjusted Rand Index between two label assignments.

    Uses scikit-learn's implementation.

    Args:
        labels1: First label assignment array.
        labels2: Second label assignment array.

    Returns:
        ARI score in [-1, 1]. A value of 1.0 indicates perfect agreement.

    Raises:
        ValueError: If the label arrays have different lengths.
    """
    from sklearn.metrics import adjusted_rand_score

    arr1 = np.asarray(labels1)
    arr2 = np.asarray(labels2)
    if arr1.shape[0] != arr2.shape[0]:
        raise ValueError(
            f"Label arrays must have the same length, got {arr1.shape[0]} "
            f"and {arr2.shape[0]}."
        )
    return float(adjusted_rand_score(arr1, arr2))


def normalized_mutual_info(
    labels1: list[int] | np.ndarray,
    labels2: list[int] | np.ndarray,
) -> float:
    """Compute Normalized Mutual Information between two label assignments.

    Uses scikit-learn's implementation with arithmetic averaging.

    Args:
        labels1: First label assignment array.
        labels2: Second label assignment array.

    Returns:
        NMI score in [0, 1]. A value of 1.0 indicates perfect agreement.

    Raises:
        ValueError: If the label arrays have different lengths.
    """
    from sklearn.metrics import normalized_mutual_info_score

    arr1 = np.asarray(labels1)
    arr2 = np.asarray(labels2)
    if arr1.shape[0] != arr2.shape[0]:
        raise ValueError(
            f"Label arrays must have the same length, got {arr1.shape[0]} "
            f"and {arr2.shape[0]}."
        )
    return float(normalized_mutual_info_score(arr1, arr2, average_method="arithmetic"))


def programs_to_labels(
    programs: dict[str, list[str]],
    gene_universe: list[str],
) -> np.ndarray:
    """Convert program membership to a label vector over a gene universe.

    Each gene receives the label of the first program it belongs to.  Genes
    not assigned to any program receive label ``-1``.

    Args:
        programs: Dictionary mapping program names to gene lists.
        gene_universe: Ordered list of all genes to label.

    Returns:
        Integer label array of length ``len(gene_universe)``.
    """
    gene_to_label: dict[str, int] = {}
    for label_idx, (_, genes) in enumerate(programs.items()):
        for gene in genes:
            if gene not in gene_to_label:
                gene_to_label[gene] = label_idx

    labels = np.full(len(gene_universe), -1, dtype=np.int64)
    for i, gene in enumerate(gene_universe):
        if gene in gene_to_label:
            labels[i] = gene_to_label[gene]
    return labels


# ------------------------------------------------------------------
# Stability and robustness metrics
# ------------------------------------------------------------------


def program_stability(
    embeddings: np.ndarray,
    gene_names: list[str],
    discovery_fn: Callable[..., dict[str, list[str]]],
    n_resamples: int = 20,
    subsample_fraction: float = 0.8,
    random_state: int = 42,
    **discovery_kwargs: Any,
) -> dict[str, Any]:
    """Measure stability of gene programs under gene subsampling.

    Runs the discovery method on random 80% subsamples of genes and measures
    how consistently genes are assigned to the same programs across runs.

    Args:
        embeddings: Gene embedding matrix ``(n_genes, n_dims)``.
        gene_names: Gene identifiers.
        discovery_fn: Callable that takes ``(embeddings, gene_names, **kwargs)``
            and returns a program dict ``{name: [genes]}``.
        n_resamples: Number of subsample iterations.
        subsample_fraction: Fraction of genes to keep per iteration.
        random_state: Random seed.
        **discovery_kwargs: Passed to ``discovery_fn``.

    Returns:
        Dictionary with keys:
        - ``mean_jaccard``: Mean best-match Jaccard across resamples.
        - ``per_resample_jaccard``: List of per-resample Jaccard scores.
        - ``gene_stability``: Per-gene stability (fraction of runs gene appears
          in a consistent program).
    """
    n_genes = len(gene_names)
    if n_genes == 0:
        return {
            "mean_jaccard": 0.0,
            "std_jaccard": 0.0,
            "per_resample_jaccard": [],
            "gene_stability": {},
            "mean_gene_stability": 0.0,
        }

    rng = np.random.default_rng(random_state)
    n_sub = int(n_genes * subsample_fraction)
    n_sub = max(n_sub, 2 if n_genes >= 2 else 1)
    n_sub = min(n_sub, n_genes)

    # Run full discovery as reference
    ref_programs = discovery_fn(embeddings, gene_names, **discovery_kwargs)

    per_resample_jaccard: list[float] = []
    gene_counts: dict[str, int] = {g: 0 for g in gene_names}
    gene_stable: dict[str, int] = {g: 0 for g in gene_names}

    for _ in range(n_resamples):
        idx = rng.choice(n_genes, size=n_sub, replace=False)
        idx.sort()
        sub_emb = embeddings[idx]
        sub_names = [gene_names[i] for i in idx]

        sub_programs = discovery_fn(sub_emb, sub_names, **discovery_kwargs)

        # Best-match Jaccard between ref and sub
        jaccards: list[float] = []
        for ref_name, ref_genes in ref_programs.items():
            ref_set = set(ref_genes) & set(sub_names)
            if not ref_set:
                continue
            best_j = 0.0
            for sub_genes in sub_programs.values():
                j = jaccard_similarity(ref_set, set(sub_genes))
                best_j = max(best_j, j)
            jaccards.append(best_j)

        if jaccards:
            per_resample_jaccard.append(float(np.mean(jaccards)))

        # Per-gene stability
        sub_gene_to_prog: dict[str, int] = {}
        for prog_idx, (_, genes) in enumerate(sub_programs.items()):
            for g in genes:
                if g not in sub_gene_to_prog:
                    sub_gene_to_prog[g] = prog_idx

        ref_gene_to_prog: dict[str, int] = {}
        for prog_idx, (_, genes) in enumerate(ref_programs.items()):
            for g in genes:
                if g not in ref_gene_to_prog:
                    ref_gene_to_prog[g] = prog_idx

        ref_sets = {
            prog_idx: set(genes) & set(sub_names)
            for prog_idx, (_, genes) in enumerate(ref_programs.items())
        }
        sub_prog_to_ref_prog: dict[int, int] = {}
        for sub_prog_idx, (_, genes) in enumerate(sub_programs.items()):
            sub_set = set(genes)
            best_ref_idx = -1
            best_jaccard = -1.0
            for ref_prog_idx, ref_set in ref_sets.items():
                jac = jaccard_similarity(ref_set, sub_set)
                if jac > best_jaccard:
                    best_jaccard = jac
                    best_ref_idx = ref_prog_idx
            sub_prog_to_ref_prog[sub_prog_idx] = best_ref_idx

        for g in sub_names:
            gene_counts[g] += 1
            if g in ref_gene_to_prog and g in sub_gene_to_prog:
                if ref_gene_to_prog[g] == sub_prog_to_ref_prog[sub_gene_to_prog[g]]:
                    gene_stable[g] += 1

    gene_stability_scores: dict[str, float] = {}
    for g in gene_names:
        if gene_counts[g] > 0:
            gene_stability_scores[g] = gene_stable[g] / gene_counts[g]
        else:
            gene_stability_scores[g] = 0.0

    return {
        "mean_jaccard": float(np.mean(per_resample_jaccard)) if per_resample_jaccard else 0.0,
        "std_jaccard": float(np.std(per_resample_jaccard)) if per_resample_jaccard else 0.0,
        "per_resample_jaccard": per_resample_jaccard,
        "gene_stability": gene_stability_scores,
        "mean_gene_stability": float(np.mean(list(gene_stability_scores.values()))),
    }


def bootstrap_metric(
    metric_fn: Callable[..., float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
    **metric_kwargs: Any,
) -> dict[str, float]:
    """Compute bootstrap confidence interval for any metric function.

    Args:
        metric_fn: Function returning a single float metric.
        n_bootstrap: Number of bootstrap iterations.
        confidence: Confidence level for the interval.
        random_state: Random seed.
        **metric_kwargs: Passed to ``metric_fn``.

    Returns:
        Dictionary with ``point_estimate``, ``ci_lower``, ``ci_upper``,
        ``std_error``.
    """
    point = metric_fn(**metric_kwargs)
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be >= 0")
    if n_bootstrap == 0:
        return {
            "point_estimate": point,
            "ci_lower": point,
            "ci_upper": point,
            "std_error": 0.0,
        }
    rng = np.random.default_rng(random_state)

    resampleable_groups: dict[int, list[str]] = {}
    for key, value in metric_kwargs.items():
        if isinstance(value, np.ndarray) and value.ndim >= 1 and len(value) > 1:
            resampleable_groups.setdefault(len(value), []).append(key)
        elif (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and len(value) > 1
        ):
            resampleable_groups.setdefault(len(value), []).append(key)

    boot_values: list[float] = []
    for _ in range(n_bootstrap):
        boot_kwargs = dict(metric_kwargs)
        for n_items, keys in resampleable_groups.items():
            idx = rng.integers(0, n_items, size=n_items)
            for key in keys:
                value = metric_kwargs[key]
                if isinstance(value, np.ndarray):
                    boot_kwargs[key] = value[idx]
                else:
                    seq = list(value)
                    boot_kwargs[key] = [seq[i] for i in idx]
        boot_values.append(metric_fn(**boot_kwargs))

    alpha = (1 - confidence) / 2
    ci_lower = float(np.percentile(boot_values, alpha * 100))
    ci_upper = float(np.percentile(boot_values, (1 - alpha) * 100))

    return {
        "point_estimate": point,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std_error": float(np.std(boot_values)),
    }


# ------------------------------------------------------------------
# Permutation-based null hypothesis testing
# ------------------------------------------------------------------


def permutation_test_enrichment(
    programs: dict[str, list[str]],
    curated: dict[str, list[str]],
    gene_universe: list[str],
    metric_fn: Callable[[dict[str, list[str]], dict[str, list[str]]], float] | None = None,
    n_permutations: int = 1000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Permutation test: are program-curated overlaps better than random?

    Shuffles gene labels and re-computes the metric to build a null
    distribution, then computes an empirical p-value.

    Args:
        programs: Learned gene programs.
        curated: Curated reference pathways.
        gene_universe: All gene symbols (used for shuffling).
        metric_fn: Function ``(programs, curated) -> float``. Defaults to
            mean best-match Jaccard.
        n_permutations: Number of permutations.
        random_state: Random seed.

    Returns:
        Dictionary with ``observed``, ``null_mean``, ``null_std``,
        ``p_value``, ``z_score``.
    """
    rng = np.random.default_rng(random_state)

    if metric_fn is None:
        def metric_fn(progs: dict[str, list[str]], cur: dict[str, list[str]]) -> float:
            jaccards: list[float] = []
            for genes in progs.values():
                s1 = set(genes)
                best = max(
                    (jaccard_similarity(s1, set(c)) for c in cur.values()),
                    default=0.0,
                )
                jaccards.append(best)
            return float(np.mean(jaccards)) if jaccards else 0.0

    observed = metric_fn(programs, curated)

    null_dist: list[float] = []
    universe_arr = list(gene_universe)
    if not universe_arr:
        inferred_universe = {
            g for genes in programs.values() for g in genes
        } | {
            g for genes in curated.values() for g in genes
        }
        universe_arr = sorted(inferred_universe)

    if not universe_arr:
        return {
            "observed": observed,
            "null_mean": observed,
            "null_std": 0.0,
            "p_value": 1.0,
            "z_score": 0.0,
            "n_permutations": n_permutations,
        }

    for _ in range(n_permutations):
        shuffled = universe_arr.copy()
        rng.shuffle(shuffled)
        # Create shuffled programs with same sizes.
        # Each program draws *without replacement* from the shuffled universe
        # so that shuffled programs never share genes (matching real programs).
        shuffled_programs: dict[str, list[str]] = {}
        offset = 0
        for name, genes in programs.items():
            size = len(genes)
            end = offset + size
            if end <= len(shuffled):
                shuffled_programs[name] = shuffled[offset:end]
            else:
                # If total program genes exceed universe size, wrap and
                # re-shuffle to avoid deterministic overlap.
                rng.shuffle(shuffled)
                shuffled_programs[name] = shuffled[:size]
                end = size
            offset = end
        null_dist.append(metric_fn(shuffled_programs, curated))

    null_mean = float(np.mean(null_dist))
    null_std = float(np.std(null_dist))
    p_value = float(np.mean([n >= observed for n in null_dist]))
    z_score = (observed - null_mean) / null_std if null_std > 0 else 0.0

    return {
        "observed": observed,
        "null_mean": null_mean,
        "null_std": null_std,
        "p_value": max(p_value, 1.0 / (n_permutations + 1)),
        "z_score": z_score,
        "n_permutations": n_permutations,
    }


# ------------------------------------------------------------------
# Advanced biological coherence metrics
# ------------------------------------------------------------------


def functional_coherence_go(
    gene_set: list[str],
    go_annotations: dict[str, set[str]] | None = None,
) -> float:
    """Measure functional coherence using GO term overlap.

    For each pair of genes, computes the fraction of shared GO terms.
    Returns the mean pairwise GO overlap.

    Args:
        gene_set: List of gene symbols.
        go_annotations: Mapping from gene symbol to set of GO term IDs.
            If ``None``, returns 0.0.

    Returns:
        Mean pairwise GO term overlap in [0, 1].
    """
    if go_annotations is None:
        return 0.0

    annotated = [g for g in gene_set if g in go_annotations]
    if len(annotated) < 2:
        return 0.0

    overlaps: list[float] = []
    for g1, g2 in combinations(annotated, 2):
        terms1 = go_annotations[g1]
        terms2 = go_annotations[g2]
        if not terms1 or not terms2:
            overlaps.append(0.0)
            continue
        overlap = len(terms1 & terms2) / len(terms1 | terms2)
        overlaps.append(overlap)

    return float(np.mean(overlaps))


def regulatory_coherence(
    gene_set: list[str],
    tf_targets: dict[str, set[str]] | None = None,
) -> float:
    """Measure regulatory coherence: fraction of genes sharing a TF regulator.

    Args:
        gene_set: List of gene symbols.
        tf_targets: Mapping from TF name to set of target gene symbols.
            If ``None``, returns 0.0.

    Returns:
        Fraction of gene pairs sharing at least one TF in [0, 1].
    """
    if tf_targets is None or len(gene_set) < 2:
        return 0.0

    gene_set_s = set(gene_set)
    # Build reverse mapping: gene -> set of TFs targeting it
    gene_to_tfs: dict[str, set[str]] = {}
    for tf, targets in tf_targets.items():
        for g in targets & gene_set_s:
            gene_to_tfs.setdefault(g, set()).add(tf)

    annotated = [g for g in gene_set if g in gene_to_tfs]
    if len(annotated) < 2:
        return 0.0

    shared = 0
    total = 0
    for g1, g2 in combinations(annotated, 2):
        total += 1
        if gene_to_tfs[g1] & gene_to_tfs[g2]:
            shared += 1

    return shared / total if total > 0 else 0.0


def program_specificity(
    programs: dict[str, list[str]],
) -> float:
    """Compute specificity: mean fraction of genes unique to each program.

    Higher specificity means programs capture distinct biology.

    Args:
        programs: Dictionary mapping program names to gene lists.

    Returns:
        Mean specificity in [0, 1].
    """
    if len(programs) < 2:
        return 1.0

    # Count gene appearances across programs
    gene_counts: dict[str, int] = {}
    for genes in programs.values():
        for g in genes:
            gene_counts[g] = gene_counts.get(g, 0) + 1

    specificities: list[float] = []
    for genes in programs.values():
        if not genes:
            continue
        unique = sum(1 for g in genes if gene_counts.get(g, 0) == 1)
        specificities.append(unique / len(genes))

    return float(np.mean(specificities)) if specificities else 0.0


def inter_program_distance(
    programs: dict[str, list[str]],
    embeddings: np.ndarray,
    gene_names: list[str],
) -> float:
    """Compute mean cosine distance between program centroids.

    Higher values indicate more distinct programs in embedding space.

    Args:
        programs: Gene programs.
        embeddings: Gene embedding matrix ``(n_genes, n_dims)``.
        gene_names: Gene names aligned with embeddings.

    Returns:
        Mean pairwise cosine distance between centroids in [0, 2].
    """
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    centroids: list[np.ndarray] = []
    for genes in programs.values():
        idxs = [gene_to_idx[g] for g in genes if g in gene_to_idx]
        if not idxs:
            continue
        centroid = embeddings[idxs].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids.append(centroid)

    if len(centroids) < 2:
        return 0.0

    distances: list[float] = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            cos_sim = float(np.dot(centroids[i], centroids[j]))
            distances.append(1.0 - cos_sim)

    return float(np.mean(distances))


# ------------------------------------------------------------------
# Comprehensive evaluation summary
# ------------------------------------------------------------------


def comprehensive_evaluation(
    programs: dict[str, list[str]],
    reference_genes: list[str],
    curated: dict[str, list[str]] | None = None,
    embeddings: np.ndarray | None = None,
    gene_names: list[str] | None = None,
    ppi_network: Any | None = None,
    go_annotations: dict[str, set[str]] | None = None,
    tf_targets: dict[str, set[str]] | None = None,
) -> dict[str, float]:
    """Compute all available metrics for a set of gene programs.

    Args:
        programs: Discovered gene programs.
        reference_genes: All genes in the dataset.
        curated: Optional curated pathways for novelty/recovery.
        embeddings: Optional gene embeddings for distance metrics.
        gene_names: Gene names aligned with embeddings.
        ppi_network: Optional networkx PPI graph.
        go_annotations: Optional GO term annotations.
        tf_targets: Optional TF-target mappings.

    Returns:
        Dictionary of metric name to value.
    """
    results: dict[str, float] = {}

    # Basic metrics
    results["n_programs"] = float(len(programs))
    sizes = [len(g) for g in programs.values()]
    results["mean_program_size"] = float(np.mean(sizes)) if sizes else 0.0
    results["median_program_size"] = float(np.median(sizes)) if sizes else 0.0
    results["min_program_size"] = float(min(sizes)) if sizes else 0.0
    results["max_program_size"] = float(max(sizes)) if sizes else 0.0
    results["coverage"] = coverage(programs, reference_genes)
    results["redundancy"] = program_redundancy(programs)
    results["specificity"] = program_specificity(programs)

    # Novelty
    if curated is not None:
        results["novelty"] = novelty_score(programs, curated)

    # Embedding-based metrics
    if embeddings is not None and gene_names is not None:
        results["inter_program_distance"] = inter_program_distance(
            programs, embeddings, gene_names
        )

    # Biological coherence
    if ppi_network is not None:
        coherences = [
            biological_coherence(genes, ppi_network)
            for genes in programs.values()
        ]
        results["mean_ppi_coherence"] = float(np.mean(coherences))

    # GO coherence
    if go_annotations is not None:
        go_scores = [
            functional_coherence_go(genes, go_annotations)
            for genes in programs.values()
        ]
        results["mean_go_coherence"] = float(np.mean(go_scores))

    # Regulatory coherence
    if tf_targets is not None:
        reg_scores = [
            regulatory_coherence(genes, tf_targets)
            for genes in programs.values()
        ]
        results["mean_regulatory_coherence"] = float(np.mean(reg_scores))

    return results


def compare_methods(
    method_programs: dict[str, dict[str, list[str]]],
    reference_genes: list[str],
    curated: dict[str, list[str]] | None = None,
    embeddings: np.ndarray | None = None,
    gene_names: list[str] | None = None,
) -> pd.DataFrame:
    """Compare multiple discovery methods across all metrics.

    Args:
        method_programs: Mapping from method name to its discovered programs.
        reference_genes: All genes in the dataset.
        curated: Optional curated pathways.
        embeddings: Optional gene embeddings.
        gene_names: Gene names aligned with embeddings.

    Returns:
        DataFrame with one row per method and one column per metric.
    """
    rows: list[dict[str, Any]] = []
    for method_name, programs in method_programs.items():
        metrics_raw = comprehensive_evaluation(
            programs=programs,
            reference_genes=reference_genes,
            curated=curated,
            embeddings=embeddings,
            gene_names=gene_names,
        )
        metrics: dict[str, Any] = dict(metrics_raw)
        metrics["method"] = method_name
        rows.append(metrics)

    df = pd.DataFrame(rows)
    if "method" in df.columns:
        df = df.set_index("method")
    return df


def benjamini_hochberg_fdr(
    p_values: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Compute Benjamini-Hochberg FDR q-values with NaN-safe handling.

    Args:
        p_values: Raw p-values. Non-finite values are ignored and returned as
            ``np.nan`` in the output.

    Returns:
        Array of BH-adjusted q-values aligned to input order.
    """
    p = np.asarray(p_values, dtype=np.float64)
    q = np.full(p.shape, np.nan, dtype=np.float64)

    valid = np.isfinite(p)
    if not np.any(valid):
        return q

    p_valid = p[valid]
    order = np.argsort(p_valid)
    ranked = p_valid[order]
    n = len(ranked)
    ranks = np.arange(1, n + 1, dtype=np.float64)

    adjusted = ranked * (n / ranks)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    q_valid = np.empty_like(adjusted)
    q_valid[order] = adjusted
    q[valid] = q_valid
    return q


def benjamini_hochberg_fdr_grouped(
    df: pd.DataFrame,
    *,
    p_value_col: str,
    group_cols: Sequence[str] | None = None,
) -> pd.Series:
    """Compute BH-FDR globally or within groups.

    Args:
        df: Input table.
        p_value_col: Column containing raw p-values.
        group_cols: Optional columns defining independent multiple-testing
            families. If omitted, a global BH correction is applied.

    Returns:
        Float Series aligned to ``df.index`` containing BH-adjusted q-values.
    """
    if p_value_col not in df.columns:
        raise KeyError(f"Column not found: {p_value_col}")

    if group_cols:
        missing = [c for c in group_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Group columns not found: {', '.join(missing)}")

    p = pd.to_numeric(df[p_value_col], errors="coerce").to_numpy(dtype=np.float64)
    out = np.full(len(df), np.nan, dtype=np.float64)

    if not group_cols:
        out[:] = benjamini_hochberg_fdr(p)
        return pd.Series(out, index=df.index, dtype=np.float64)

    grouped_indices = df.groupby(list(group_cols), dropna=False, sort=False).indices
    for idx in grouped_indices.values():
        idx_arr = np.asarray(idx, dtype=np.int64)
        out[idx_arr] = benjamini_hochberg_fdr(p[idx_arr])

    return pd.Series(out, index=df.index, dtype=np.float64)


def paired_rank_biserial_correlation(
    delta: Sequence[float] | np.ndarray,
) -> float:
    """Compute paired rank-biserial effect size from paired deltas.

    Uses Wilcoxon signed-rank style absolute ranks after removing zero deltas.

    Args:
        delta: Pairwise method-reference differences.

    Returns:
        Rank-biserial correlation in [-1, 1]. Returns ``np.nan`` when not
        enough non-zero finite deltas are available.
    """
    arr = np.asarray(delta, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr != 0.0]
    if arr.size == 0:
        return np.nan

    ranks = scipy_stats.rankdata(np.abs(arr), method="average")
    w_pos = float(ranks[arr > 0.0].sum())
    w_neg = float(ranks[arr < 0.0].sum())
    denom = w_pos + w_neg
    if denom <= 0.0:
        return np.nan
    return float((w_pos - w_neg) / denom)


def effect_size_cohens_d(
    group1: np.ndarray,
    group2: np.ndarray,
) -> float:
    """Compute Cohen's d effect size between two groups.

    Uses pooled standard deviation with Bessel's correction.

    Args:
        group1: Values for group 1.
        group2: Values for group 2.

    Returns:
        Cohen's d effect size.
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-12:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def paired_rank_biserial(
    deltas: np.ndarray,
) -> float:
    """Compute paired rank-biserial effect size from paired differences.

    For paired comparisons, rank-biserial can be interpreted as:
    ``(wins - losses) / (wins + losses)``, where ties are excluded.
    Positive values indicate group1 > group2 on average.

    Args:
        deltas: Array of paired differences ``group1 - group2``.

    Returns:
        Rank-biserial correlation in [-1, 1]. Returns ``np.nan`` when all
        observations are ties or no finite values are present.
    """
    arr = np.asarray(deltas, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    non_ties = arr[arr != 0.0]
    if len(non_ties) == 0:
        return np.nan
    wins = float(np.sum(non_ties > 0.0))
    losses = float(np.sum(non_ties < 0.0))
    return float((wins - losses) / len(non_ties))


def paired_sign_test_pvalue(
    deltas: np.ndarray,
    *,
    alternative: str = "greater",
) -> float:
    """Compute exact paired sign-test p-value from paired differences.

    Args:
        deltas: Array of paired differences ``group1 - group2``.
        alternative: One of ``"greater"``, ``"less"``, or ``"two-sided"``.

    Returns:
        Exact binomial sign-test p-value. Returns ``np.nan`` when there are no
        non-tied finite observations.

    Raises:
        ValueError: If ``alternative`` is invalid.
    """
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError(
            "alternative must be one of {'greater', 'less', 'two-sided'}."
        )

    arr = np.asarray(deltas, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    non_ties = arr[arr != 0.0]
    if len(non_ties) == 0:
        return np.nan

    wins = int(np.sum(non_ties > 0.0))
    losses = int(np.sum(non_ties < 0.0))
    n_non_ties = wins + losses
    if n_non_ties == 0:
        return np.nan
    return float(
        scipy_stats.binomtest(
            wins,
            n_non_ties,
            p=0.5,
            alternative=alternative,
        ).pvalue
    )


def fold_enrichment_with_ci(
    overlap: int,
    program_size: int,
    query_size: int,
    background_size: int,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Compute fold enrichment with confidence interval.

    Uses log-linear model confidence interval for the odds ratio.

    Args:
        overlap: Number of overlapping genes.
        program_size: Size of the gene program.
        query_size: Size of the query gene list.
        background_size: Total number of genes in the background.
        confidence: Confidence level.

    Returns:
        Dictionary with ``fold_enrichment``, ``ci_lower``, ``ci_upper``.
    """
    expected = (program_size * query_size) / max(background_size, 1)
    fold = overlap / expected if expected > 0 else 0.0

    # Log-space CI using normal approximation
    if overlap > 0 and expected > 0:
        z = scipy_stats.norm.ppf(1 - (1 - confidence) / 2)
        se_log = np.sqrt(
            1.0 / max(overlap, 1)
            + 1.0 / max(program_size - overlap, 1)
            + 1.0 / max(query_size - overlap, 1)
            + 1.0 / max(background_size - program_size - query_size + overlap, 1)
        )
        log_fold = np.log(fold) if fold > 0 else 0.0
        ci_lower = np.exp(log_fold - z * se_log)
        ci_upper = np.exp(log_fold + z * se_log)
    else:
        ci_lower = 0.0
        ci_upper = 0.0

    return {
        "fold_enrichment": fold,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
    }
