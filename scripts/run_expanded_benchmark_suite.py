#!/usr/bin/env python3
"""Expanded benchmark suite: datasets + ablations + tuning + statistics."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import generate_benchmark_report as gbr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats

from npathway.data.datasets import (
    filter_gene_sets_to_adata,
    load_burczynski06,
    load_moignard15,
    load_msigdb_gene_sets,
    load_paul15,
    load_pbmc3k,
    load_pbmc68k_reduced,
    load_tabula_muris,
)
from npathway.data.preprocessing import (
    _safe_toarray,
    build_gene_embeddings_from_expression,
    build_graph_regularized_embeddings,
)
from npathway.discovery.attention_network import AttentionNetworkProgramDiscovery
from npathway.discovery.baselines import (
    CNMFProgramDiscovery,
    ExpressionClusteringBaseline,
    OfficialCNMFProgramDiscovery,
    RandomProgramDiscovery,
    StarCATReferenceProgramDiscovery,
    WGCNAProgramDiscovery,
)
from npathway.discovery.clustering import ClusteringProgramDiscovery
from npathway.discovery.ensemble import EnsembleProgramDiscovery
from npathway.discovery.topic_model import TopicModelProgramDiscovery
from npathway.evaluation.metrics import (
    benjamini_hochberg_fdr_grouped,
    paired_rank_biserial_correlation,
)

logger = logging.getLogger(__name__)

SUPPORTED_DATASETS = (
    "pbmc3k",
    "pbmc68k_reduced",
    "paul15",
    "moignard15",
    "burczynski06",
    "tabula_muris",
)
SUPPORTED_COLLECTIONS = ("hallmark", "kegg", "go_bp")
PRIMARY_METRIC_DIRECTIONS: dict[str, str] = {
    "recovery_mean_n_sig": "greater",
    "recovery_mean_hallmark_jaccard": "greater",
    "discovery_mean_hallmark_alignment": "greater",
    "power_mean_tpr": "greater",
    "power_mean_fpr": "less",
}
PRIMARY_METRIC_DEFAULTS = [
    "discovery_mean_hallmark_alignment",
    "power_mean_fpr",
]


def _ensure_dirs(base_dir: Path) -> tuple[Path, Path]:
    tables_dir = base_dir / "tables"
    figures_dir = base_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir, figures_dir


def _bootstrap_ci_mean(
    values: np.ndarray,
    n_bootstrap: int,
    alpha: float,
    seed: int,
) -> tuple[float, float]:
    if len(values) == 0 or n_bootstrap <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    boot_means = values[idx].mean(axis=1)
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return lo, hi


def _paired_effect_size_dz(delta: np.ndarray) -> float:
    if len(delta) < 2:
        return np.nan
    sd = float(np.std(delta, ddof=1))
    if sd <= 0:
        return np.nan
    return float(np.mean(delta) / sd)


def _prepare_group_labels(
    adata: Any,
    *,
    allow_index_parity_fallback: bool = True,
) -> tuple[Any, str, bool]:
    """Ensure `adata.obs['louvain']` exists for benchmark functions."""
    if "louvain" in adata.obs.columns:
        adata.obs["louvain"] = adata.obs["louvain"].astype(str)
        return adata, "louvain", False
    for candidate in [
        "bulk_labels",
        "cell_type",
        "cell_types",
        "cluster",
        "paul15_clusters",
        "exp_groups",
        "groups",
    ]:
        if candidate in adata.obs.columns:
            adata.obs["louvain"] = adata.obs[candidate].astype(str)
            return adata, candidate, False
    if not allow_index_parity_fallback:
        raise RuntimeError(
            "Dataset is missing a usable group label column for benchmarking. "
            "Strict publication mode disallows index-parity pseudo-label fallback. "
            f"Available obs columns: {list(adata.obs.columns)}"
        )
    # Last-resort fallback: split by cell index parity.
    fallback = np.array([str(i % 2) for i in range(adata.n_obs)])
    adata.obs["louvain"] = fallback
    return adata, "index_parity", True


def _infer_msigdb_species(gene_names: list[str]) -> str:
    """Infer whether gene symbols look human-like (all-caps) or mouse-like."""
    alpha = [g for g in gene_names if any(ch.isalpha() for ch in g)]
    if not alpha:
        return "human"
    sample = alpha[: min(500, len(alpha))]
    human_upper_frac = np.mean([g == g.upper() for g in sample])
    mouse_title_frac = np.mean(
        [
            len(g) >= 2 and g[0].isalpha() and g[0].isupper() and any(ch.islower() for ch in g[1:4])
            for g in sample
        ]
    )
    return "mouse" if mouse_title_frac > human_upper_frac else "human"


def _method_track(method: str) -> str:
    """Classify methods into de novo vs reference-guided tracks."""
    if method.startswith("Curated-") or method == "starCAT-Ref":
        return "reference_guided"
    return "de_novo"


def _split_gene_sets_for_leakage_control(
    gene_sets: dict[str, list[str]],
    tune_fraction: float,
    split_seed: int,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Split one collection into tune/eval partitions to avoid leakage."""
    names = sorted(gene_sets.keys())
    if len(names) < 2:
        return dict(gene_sets), dict(gene_sets)
    frac = min(max(tune_fraction, 0.1), 0.9)
    n_tune = int(round(len(names) * frac))
    n_tune = min(max(n_tune, 1), len(names) - 1)
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(len(names))
    tune_idx = set(int(i) for i in perm[:n_tune])
    tune = {n: gene_sets[n] for i, n in enumerate(names) if i in tune_idx}
    eval_ = {n: gene_sets[n] for i, n in enumerate(names) if i not in tune_idx}
    return tune, eval_


def _collection_with_fallback(
    *,
    collection: str,
    species: str,
    adata: Any,
    msigdb_species_mode: str,
    allow_species_fallback: bool = True,
) -> tuple[dict[str, list[str]], str, bool]:
    """Load and filter one MSigDB collection with species fallback."""
    raw = load_msigdb_gene_sets(collection=collection, species=species)
    filt = filter_gene_sets_to_adata(raw, adata, min_genes=3)
    if filt or msigdb_species_mode != "auto":
        return filt, species, False

    if not allow_species_fallback:
        raise RuntimeError(
            f"No {collection} overlap with inferred species '{species}'. "
            "Strict publication mode disallows automatic fallback to an alternate "
            "MSigDB species. Set --msigdb-species explicitly or fix the gene identifiers."
        )

    fallback_species = "mouse" if species == "human" else "human"
    logger.warning(
        "No %s overlap with inferred species '%s'; retrying with '%s'.",
        collection,
        species,
        fallback_species,
    )
    raw_fb = load_msigdb_gene_sets(collection=collection, species=fallback_species)
    filt_fb = filter_gene_sets_to_adata(raw_fb, adata, min_genes=3)
    if filt_fb:
        return filt_fb, fallback_species, True
    return filt, species, False


def load_dataset_bundle(
    name: str,
    *,
    tabula_muris_path: Path | None = None,
    allow_proxy_datasets: bool = True,
    strict_publication: bool = False,
    msigdb_species: str = "auto",
    tuning_collection: str = "kegg",
    evaluation_collection: str = "hallmark",
    same_collection_tune_fraction: float = 0.5,
    split_seed: int = 42,
) -> dict[str, Any]:
    """Load dataset + embeddings + leakage-safe tune/eval gene-set partitions."""
    if name == "pbmc3k":
        adata = load_pbmc3k(preprocessed=True)
    elif name == "pbmc68k_reduced":
        adata = load_pbmc68k_reduced()
    elif name == "paul15":
        adata = load_paul15()
    elif name == "moignard15":
        adata = load_moignard15()
    elif name == "burczynski06":
        adata = load_burczynski06()
    elif name == "tabula_muris":
        adata = load_tabula_muris(
            method="droplet",
            dataset_path=tabula_muris_path,
            allow_proxy=allow_proxy_datasets,
        )
    else:
        raise ValueError(f"Unknown dataset '{name}'.")

    adata, group_label_source, used_group_label_fallback = _prepare_group_labels(
        adata,
        allow_index_parity_fallback=not strict_publication,
    )
    gene_embeddings, gene_names = build_gene_embeddings_from_expression(
        adata, n_components=50
    )
    graph_embeddings, _ = build_graph_regularized_embeddings(
        adata,
        n_components=50,
        k_neighbors=15,
        n_diffusion_steps=3,
        alpha=0.5,
    )
    species = (
        _infer_msigdb_species(gene_names)
        if msigdb_species == "auto"
        else msigdb_species
    )
    eval_sets, species_eval, used_eval_species_fallback = _collection_with_fallback(
        collection=evaluation_collection,
        species=species,
        adata=adata,
        msigdb_species_mode=msigdb_species,
        allow_species_fallback=not strict_publication,
    )
    species = species_eval

    if tuning_collection == evaluation_collection:
        if strict_publication and len(eval_sets) < 2:
            raise RuntimeError(
                "Strict publication mode requires at least 2 non-empty gene sets when "
                "tuning and evaluation collections are identical, so they can be split "
                "without leakage."
            )
        tuning_sets, evaluation_sets = _split_gene_sets_for_leakage_control(
            eval_sets,
            tune_fraction=same_collection_tune_fraction,
            split_seed=split_seed,
        )
        used_tune_species_fallback = used_eval_species_fallback
    else:
        tuning_sets, species_tune, used_tune_species_fallback = _collection_with_fallback(
            collection=tuning_collection,
            species=species,
            adata=adata,
            msigdb_species_mode=msigdb_species,
            allow_species_fallback=not strict_publication,
        )
        if species_tune != species and msigdb_species == "auto":
            logger.warning(
                "Tuning collection species resolved to %s while evaluation uses %s.",
                species_tune,
                species,
        )
        evaluation_sets = eval_sets

    # Optional analysis collections for biological case studies.
    go_bp_sets, _, used_go_bp_species_fallback = _collection_with_fallback(
        collection="go_bp",
        species=species,
        adata=adata,
        msigdb_species_mode=msigdb_species,
        allow_species_fallback=not strict_publication,
    )
    kegg_sets, _, used_kegg_species_fallback = _collection_with_fallback(
        collection="kegg",
        species=species,
        adata=adata,
        msigdb_species_mode=msigdb_species,
        allow_species_fallback=not strict_publication,
    )

    if not evaluation_sets:
        raise RuntimeError(
            f"No evaluation gene sets overlap for dataset={name}, "
            f"collection={evaluation_collection}, species={species}."
        )
    if not tuning_sets:
        if strict_publication:
            raise RuntimeError(
                f"No tuning gene sets retained for dataset={name}. "
                "Strict publication mode forbids reusing evaluation gene sets for tuning."
            )
        logger.warning(
            "No tuning gene sets retained for dataset=%s; reusing evaluation sets.",
            name,
        )
        tuning_sets = dict(evaluation_sets)

    dataset_meta = dict(adata.uns.get("npathway_dataset", {}))
    if not dataset_meta:
        dataset_meta = {
            "dataset_name": name,
            "source": "unknown",
            "is_proxy": False,
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
        }
    dataset_meta["msigdb_species"] = species
    dataset_meta["tuning_collection"] = tuning_collection
    dataset_meta["evaluation_collection"] = evaluation_collection
    dataset_meta["tuning_sets"] = len(tuning_sets)
    dataset_meta["evaluation_sets"] = len(evaluation_sets)
    dataset_meta["go_bp_sets"] = len(go_bp_sets)
    dataset_meta["kegg_sets"] = len(kegg_sets)
    dataset_meta["group_label_source"] = group_label_source
    dataset_meta["used_group_label_fallback"] = bool(used_group_label_fallback)
    dataset_meta["used_species_fallback_eval"] = bool(used_eval_species_fallback)
    dataset_meta["used_species_fallback_tune"] = bool(used_tune_species_fallback)
    dataset_meta["used_species_fallback_go_bp"] = bool(used_go_bp_species_fallback)
    dataset_meta["used_species_fallback_kegg"] = bool(used_kegg_species_fallback)
    dataset_meta["used_species_fallback_any"] = bool(
        used_eval_species_fallback
        or used_tune_species_fallback
        or used_go_bp_species_fallback
        or used_kegg_species_fallback
    )
    dataset_meta["strict_publication"] = bool(strict_publication)
    return {
        "adata": adata,
        "gene_embeddings": gene_embeddings,
        "graph_embeddings": graph_embeddings,
        "gene_names": gene_names,
        "tuning_sets": tuning_sets,
        "evaluation_sets": evaluation_sets,
        "go_bp_sets": go_bp_sets,
        "kegg_sets": kegg_sets,
        "dataset_meta": dataset_meta,
    }


def _program_score_for_tuning(
    programs: dict[str, list[str]],
    tuning_sets: dict[str, list[str]],
    gene_names: list[str],
) -> float:
    """Single scalar objective for quick hyperparameter tuning."""
    # Suppress verbose benchmark logs during hyperparameter sweeps.
    gbr_logger = logging.getLogger("generate_benchmark_report")
    prev_level = gbr_logger.level
    if prev_level <= logging.INFO:
        gbr_logger.setLevel(logging.WARNING)
    try:
        df = gbr.benchmark_discovery({"candidate": programs}, tuning_sets, gene_names)
    finally:
        if prev_level <= logging.INFO:
            gbr_logger.setLevel(prev_level)
    if df.empty:
        return -np.inf
    row = df.iloc[0]
    # Favor alignment and coverage, lightly penalize redundancy.
    return float(
        row["mean_hallmark_alignment"]
        + 0.08 * row["coverage"]
        + 0.05 * row["specificity"]
        - 0.02 * row["redundancy"]
    )


def _normalize_program_scores(
    scores: dict[str, list[tuple[str, float]]],
) -> dict[str, dict[str, float]]:
    """Normalize per-program gene scores to [0, 1] dictionaries."""
    out: dict[str, dict[str, float]] = {}
    for pname, ranked in scores.items():
        if not ranked:
            continue
        genes = [str(g) for g, _ in ranked]
        vals = np.asarray(
            [max(0.0, float(s)) for _, s in ranked],
            dtype=np.float64,
        )
        vmax = float(vals.max()) if len(vals) > 0 else 0.0
        if vmax > 0:
            vals = vals / vmax
        else:
            vals = np.ones_like(vals)
        out[pname] = {g: float(v) for g, v in zip(genes, vals)}
    return out


def _fuse_program_scores(
    *,
    primary_scores: dict[str, list[tuple[str, float]]],
    secondary_scores: dict[str, list[tuple[str, float]]],
    top_n_genes: int,
    secondary_weight: float,
    min_overlap_jaccard: float = 0.0,
    adapt_weight_by_overlap: bool = False,
) -> dict[str, list[str]]:
    """Backward-compatible single-secondary fusion wrapper."""
    return _fuse_program_scores_multi(
        primary_scores=primary_scores,
        secondary_sources=[("secondary", secondary_scores, secondary_weight)],
        top_n_genes=top_n_genes,
        min_overlap_jaccard=min_overlap_jaccard,
        adapt_weight_by_overlap=adapt_weight_by_overlap,
        fusion_mode="score_sum",
    )


def _best_secondary_overlap_map(
    primary_genes: set[str],
    secondary_maps: dict[str, dict[str, float]],
) -> tuple[dict[str, float] | None, float]:
    """Return best-overlap secondary program map and its Jaccard score."""
    best_name = None
    best_jaccard = -1.0
    for sname, smap in secondary_maps.items():
        sec_genes = set(smap.keys())
        union = primary_genes | sec_genes
        if not union:
            continue
        jaccard = len(primary_genes & sec_genes) / len(union)
        if jaccard > best_jaccard:
            best_jaccard = jaccard
            best_name = sname
    if best_name is None:
        return None, 0.0
    return secondary_maps[best_name], max(0.0, float(best_jaccard))


def _fuse_program_scores_multi(
    *,
    primary_scores: dict[str, list[tuple[str, float]]],
    secondary_sources: list[tuple[str, dict[str, list[tuple[str, float]]], float]],
    top_n_genes: int,
    min_overlap_jaccard: float = 0.0,
    adapt_weight_by_overlap: bool = False,
    fusion_mode: str = "score_sum",
    rrf_k: int = 40,
) -> dict[str, list[str]]:
    """Fuse primary scores with one or more weighted secondary sources."""
    if fusion_mode not in {"score_sum", "rrf"}:
        raise ValueError("fusion_mode must be one of {'score_sum', 'rrf'}.")

    prim_norm = _normalize_program_scores(primary_scores)
    sec_bundle: list[tuple[str, dict[str, dict[str, float]], float]] = []
    for source_name, source_scores, source_weight in secondary_sources:
        if float(source_weight) <= 0.0 or not source_scores:
            continue
        sec_bundle.append(
            (
                source_name,
                _normalize_program_scores(source_scores),
                float(source_weight),
            )
        )

    fused: dict[str, list[str]] = {}
    for idx, (_, pmap) in enumerate(prim_norm.items()):
        merged = dict(pmap)
        primary_genes = set(pmap.keys())
        for _, sec_norm, source_weight in sec_bundle:
            sec_map, best_jaccard = _best_secondary_overlap_map(primary_genes, sec_norm)
            if sec_map is None or best_jaccard < float(min_overlap_jaccard):
                continue
            blend_weight = source_weight
            if adapt_weight_by_overlap:
                blend_weight *= best_jaccard
            if blend_weight <= 0.0:
                continue
            if fusion_mode == "rrf":
                ranked_secondary = sorted(
                    sec_map.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
                for rank, (gene, _) in enumerate(ranked_secondary, start=1):
                    merged[gene] = merged.get(gene, 0.0) + blend_weight / (
                        float(rrf_k) + float(rank)
                    )
            else:
                for gene, sval in sec_map.items():
                    merged[gene] = merged.get(gene, 0.0) + blend_weight * sval

        ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)
        genes = [g for g, _ in ranked[: max(3, top_n_genes)]]
        if len(genes) >= 3:
            fused[f"hybrid_{idx}"] = genes

    if fused:
        return fused

    # Fallback to primary top genes if fusion yields no valid programs.
    fallback: dict[str, list[str]] = {}
    for idx, ranked in enumerate(primary_scores.values()):
        genes = [str(g) for g, _ in ranked[: max(3, top_n_genes)]]
        if len(genes) >= 3:
            fallback[f"hybrid_{idx}"] = genes
    return fallback


def _program_collection_overlap(
    programs: dict[str, list[str]],
    reference_programs: dict[str, list[str]],
) -> float:
    """Mean best Jaccard overlap from programs to reference programs."""
    if not programs or not reference_programs:
        return 0.0
    ref_sets = [set(map(str, genes)) for genes in reference_programs.values() if genes]
    if not ref_sets:
        return 0.0
    best_scores: list[float] = []
    for genes in programs.values():
        pset = set(map(str, genes))
        if not pset:
            continue
        best = 0.0
        for rset in ref_sets:
            union = pset | rset
            if not union:
                continue
            j = len(pset & rset) / len(union)
            if j > best:
                best = j
        best_scores.append(float(best))
    if not best_scores:
        return 0.0
    return float(np.mean(best_scores))


def _program_internal_diversity(programs: dict[str, list[str]]) -> float:
    """Return 1 - mean pairwise Jaccard overlap among programs."""
    sets = [set(map(str, genes)) for genes in programs.values() if genes]
    if len(sets) < 2:
        return 0.0
    overlaps: list[float] = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            if not union:
                continue
            overlaps.append(len(sets[i] & sets[j]) / len(union))
    if not overlaps:
        return 0.0
    return float(1.0 - np.mean(overlaps))


def _robust_program_score_for_tuning(
    programs: dict[str, list[str]],
    tuning_sets: dict[str, list[str]],
    gene_names: list[str],
    auxiliary_tuning_sets: list[tuple[str, dict[str, list[str]]]] | None = None,
    worst_case_weight: float = 0.2,
) -> tuple[float, dict[str, float]]:
    """Robust objective across primary + auxiliary leakage-safe collections."""
    component_scores: dict[str, float] = {}
    primary = _program_score_for_tuning(programs, tuning_sets, gene_names)
    component_scores["primary"] = float(primary)
    if auxiliary_tuning_sets:
        for name, gene_sets in auxiliary_tuning_sets:
            if not gene_sets:
                continue
            component_scores[name] = float(
                _program_score_for_tuning(programs, gene_sets, gene_names)
            )

    finite_vals = np.asarray(
        [v for v in component_scores.values() if np.isfinite(v)],
        dtype=np.float64,
    )
    if len(finite_vals) == 0:
        return -np.inf, component_scores
    mean_val = float(np.mean(finite_vals))
    worst_val = float(np.min(finite_vals))
    wc = min(max(float(worst_case_weight), 0.0), 0.8)
    robust = (1.0 - wc) * mean_val + wc * worst_val
    return float(robust), component_scores


def _programs_fingerprint(programs: dict[str, list[str]]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Deterministic key for caching repeated objective evaluations."""
    key: list[tuple[str, tuple[str, ...]]] = []
    for pname, genes in programs.items():
        key.append((str(pname), tuple(str(g) for g in genes)))
    return tuple(sorted(key))


def tune_kmeans(
    graph_embeddings: np.ndarray,
    gene_names: list[str],
    tuning_sets: dict[str, list[str]],
    seed: int,
    candidates: list[int],
) -> tuple[int, list[dict[str, Any]]]:
    """Grid-search KMeans n_programs."""
    logs: list[dict[str, Any]] = []
    best_k = candidates[0]
    best_score = -np.inf
    for k in candidates:
        model = ClusteringProgramDiscovery(
            method="kmeans",
            n_programs=k,
            random_state=seed,
        )
        model.fit(graph_embeddings, gene_names)
        programs = model.get_programs()
        score = _program_score_for_tuning(programs, tuning_sets, gene_names)
        logs.append(
            {
                "tuning_target": "kmeans_n_programs",
                "candidate": k,
                "score": score,
            }
        )
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, logs


def tune_leiden(
    graph_embeddings: np.ndarray,
    gene_names: list[str],
    tuning_sets: dict[str, list[str]],
    seed: int,
    candidates: list[float],
) -> tuple[float, list[dict[str, Any]]]:
    """Grid-search Leiden resolution."""
    logs: list[dict[str, Any]] = []
    best_res = candidates[0]
    best_score = -np.inf
    for res in candidates:
        model = ClusteringProgramDiscovery(
            method="leiden",
            resolution=res,
            random_state=seed,
        )
        model.fit(graph_embeddings, gene_names)
        programs = model.get_programs()
        score = _program_score_for_tuning(programs, tuning_sets, gene_names)
        logs.append(
            {
                "tuning_target": "leiden_resolution",
                "candidate": res,
                "score": score,
            }
        )
        if score > best_score:
            best_score = score
            best_res = res
    return best_res, logs


def discover_methods_extended(
    *,
    adata: Any,
    graph_embeddings: np.ndarray,
    gene_embeddings: np.ndarray,
    gene_names: list[str],
    tuning_sets: dict[str, list[str]],
    evaluation_sets: dict[str, list[str]],
    seed: int,
    n_programs: int,
    top_n_genes: int,
    etm_epochs: int,
    tune_k_candidates: list[int],
    tune_leiden_candidates: list[float],
    auxiliary_tuning_sets: list[tuple[str, dict[str, list[str]]]] | None = None,
    hybrid_max_source_combo: int = 2,
    hybrid_source_prescreen_topk: int = 5,
    hybrid_worst_case_weight: float = 0.2,
    hybrid_diversity_weight: float = 0.02,
    hybrid_fusion_modes: list[str] | None = None,
    curated_baseline_name: str = "Curated-Eval",
    enable_latest_baselines: bool = False,
    official_cnmf_iters: int = 4,
    official_cnmf_max_iter: int = 400,
    starcat_reference: str = "TCAT.V1",
    starcat_cache_dir: Path = Path("data") / "starcat_cache",
) -> tuple[
    dict[str, dict[str, list[str]]],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Run expanded method set with ablations and tuned variants."""
    rng = np.random.default_rng(seed)
    tuning_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    methods: dict[str, dict[str, list[str]]] = {}
    fusion_modes = hybrid_fusion_modes or ["score_sum", "rrf"]

    def _push_runtime(
        method: str,
        start: float,
        *,
        status: str = "ok",
        reason: str = "",
    ) -> None:
        runtime_rows.append(
            {
                "seed": seed,
                "method": method,
                "fit_sec": float(time.perf_counter() - start),
                "status": status,
                "reason": reason,
            }
        )

    X_expr = _safe_toarray(adata.X).astype(np.float64)
    X_nonneg = np.clip(X_expr, 0, None) + 1e-6

    # Hyperparameter tuning
    best_k, k_logs = tune_kmeans(
        graph_embeddings,
        gene_names,
        tuning_sets,
        seed=seed,
        candidates=tune_k_candidates,
    )
    best_res, l_logs = tune_leiden(
        graph_embeddings,
        gene_names,
        tuning_sets,
        seed=seed,
        candidates=tune_leiden_candidates,
    )
    for row in k_logs + l_logs:
        row.update({"seed": seed})
        tuning_rows.append(row)

    # nPathway core
    t_fit = time.perf_counter()
    km = ClusteringProgramDiscovery(
        method="kmeans", n_programs=n_programs, random_state=seed
    )
    km.fit(graph_embeddings, gene_names)
    methods["nPathway-KMeans"] = km.get_programs()
    km_scores = km.get_program_scores()
    methods["nPathway-Refined"] = {
        prog: [g for g, _ in scored[:top_n_genes]]
        for prog, scored in km_scores.items()
    }
    _push_runtime("nPathway-KMeans", t_fit)

    # Ablation: no graph regularization
    t_fit = time.perf_counter()
    km_nograph = ClusteringProgramDiscovery(
        method="kmeans", n_programs=n_programs, random_state=seed
    )
    km_nograph.fit(gene_embeddings, gene_names)
    methods["nPathway-KMeans-NoGraph"] = km_nograph.get_programs()
    km_nograph_scores = km_nograph.get_program_scores()
    _push_runtime("nPathway-KMeans-NoGraph", t_fit)

    # Tuned variants
    t_fit = time.perf_counter()
    km_tuned = ClusteringProgramDiscovery(
        method="kmeans", n_programs=best_k, random_state=seed
    )
    km_tuned.fit(graph_embeddings, gene_names)
    methods["nPathway-KMeans-Tuned"] = km_tuned.get_programs()
    _push_runtime("nPathway-KMeans-Tuned", t_fit)

    t_fit = time.perf_counter()
    leiden = ClusteringProgramDiscovery(
        method="leiden", random_state=seed
    )
    leiden.fit(graph_embeddings, gene_names)
    methods["nPathway-Leiden"] = leiden.get_programs()
    leiden_scores = leiden.get_program_scores()
    _push_runtime("nPathway-Leiden", t_fit)

    t_fit = time.perf_counter()
    leiden_tuned = ClusteringProgramDiscovery(
        method="leiden", resolution=best_res, random_state=seed
    )
    leiden_tuned.fit(graph_embeddings, gene_names)
    methods["nPathway-Leiden-Tuned"] = leiden_tuned.get_programs()
    leiden_tuned_scores = leiden_tuned.get_program_scores()
    _push_runtime("nPathway-Leiden-Tuned", t_fit)

    # Additional expanded baselines/variants
    t_fit = time.perf_counter()
    spectral = ClusteringProgramDiscovery(
        method="spectral", n_programs=n_programs, random_state=seed
    )
    spectral.fit(graph_embeddings, gene_names)
    methods["nPathway-Spectral"] = spectral.get_programs()
    spectral_scores = spectral.get_program_scores()
    _push_runtime("nPathway-Spectral", t_fit)

    try:
        t_fit = time.perf_counter()
        hdb = ClusteringProgramDiscovery(
            method="hdbscan", min_cluster_size=5, random_state=seed
        )
        hdb.fit(graph_embeddings, gene_names)
        methods["nPathway-HDBSCAN"] = hdb.get_programs()
        _push_runtime("nPathway-HDBSCAN", t_fit)
    except Exception as exc:
        skipped_rows.append(
            {"seed": seed, "method": "nPathway-HDBSCAN", "reason": str(exc)}
        )
        _push_runtime(
            "nPathway-HDBSCAN",
            t_fit,
            status="error",
            reason=str(exc),
        )

    # Topic model
    t_fit = time.perf_counter()
    etm = TopicModelProgramDiscovery(
        n_topics=n_programs,
        n_epochs=etm_epochs,
        top_n_genes=top_n_genes,
        device="cpu",
        random_state=seed,
        early_stopping_patience=15,
        diversity_weight=2.0,
        use_decoder_weights=True,
    )
    etm.fit(graph_embeddings, gene_names, expression_matrix=X_nonneg)
    methods["nPathway-ETM"] = etm.get_programs()
    etm_scores = etm.get_program_scores()
    _push_runtime("nPathway-ETM", t_fit)

    # Attention-network baseline (cosine-attention proxy over embeddings)
    attn_scores: dict[str, list[tuple[str, float]]] = {}
    try:
        t_fit = time.perf_counter()
        attn = AttentionNetworkProgramDiscovery(
            aggregation="mean",
            threshold_quantile=0.9,
            resolution=1.0,
            centrality="pagerank",
            random_state=seed,
        )
        attn.fit(graph_embeddings, gene_names)
        methods["nPathway-AttentionNet"] = attn.get_programs()
        attn_scores = attn.get_program_scores()
        _push_runtime("nPathway-AttentionNet", t_fit)
    except Exception as exc:
        skipped_rows.append(
            {"seed": seed, "method": "nPathway-AttentionNet", "reason": str(exc)}
        )
        _push_runtime(
            "nPathway-AttentionNet",
            t_fit,
            status="error",
            reason=str(exc),
        )

    # Ensemble
    ensemble_scores: dict[str, list[tuple[str, float]]] = {}
    try:
        t_fit = time.perf_counter()
        ensemble = EnsembleProgramDiscovery(
            methods=[
                ClusteringProgramDiscovery(
                    method="kmeans", n_programs=n_programs, random_state=seed
                ),
                ClusteringProgramDiscovery(
                    method="leiden", random_state=seed + 1
                ),
                ClusteringProgramDiscovery(
                    method="spectral", n_programs=n_programs, random_state=seed + 2
                ),
            ],
            consensus_method="leiden",
            resolution=1.0,
            threshold_quantile=0.3,
            min_program_size=5,
            random_state=seed,
        )
        ensemble.fit(graph_embeddings, gene_names)
        methods["nPathway-Ensemble"] = ensemble.get_programs()
        ensemble_scores = ensemble.get_program_scores()
        _push_runtime("nPathway-Ensemble", t_fit)
    except Exception as exc:
        skipped_rows.append(
            {"seed": seed, "method": "nPathway-Ensemble", "reason": str(exc)}
        )
        _push_runtime(
            "nPathway-Ensemble",
            t_fit,
            status="error",
            reason=str(exc),
        )

    # Classical baselines
    t_fit = time.perf_counter()
    wgcna = WGCNAProgramDiscovery(
        n_programs=n_programs,
        soft_power=6,
        min_module_size=5,
        random_state=seed,
    )
    wgcna.fit(X_expr, gene_names)
    methods["WGCNA"] = wgcna.get_programs()
    wgcna_scores = wgcna.get_program_scores()
    _push_runtime("WGCNA", t_fit)

    t_fit = time.perf_counter()
    cnmf = CNMFProgramDiscovery(
        n_programs=n_programs,
        n_iter=5,
        top_n_genes=top_n_genes,
        random_state=seed,
    )
    cnmf.fit(X_nonneg, gene_names)
    methods["cNMF"] = cnmf.get_programs()
    cnmf_scores = cnmf.get_program_scores()
    _push_runtime("cNMF", t_fit)

    # Hybrid tuned fusion: preserve cNMF core while borrowing from multiple
    # complementary discoverers with robust multi-collection objective.
    t_fit = time.perf_counter()
    auxiliary_sets = [
        (name, sets)
        for name, sets in (auxiliary_tuning_sets or [])
        if sets
    ]

    hybrid_best_programs = _fuse_program_scores(
        primary_scores=cnmf_scores,
        secondary_scores=km_nograph_scores,
        top_n_genes=top_n_genes,
        secondary_weight=0.0,
    )
    base_robust_score, _ = _robust_program_score_for_tuning(
        hybrid_best_programs,
        tuning_sets,
        gene_names,
        auxiliary_tuning_sets=auxiliary_sets,
        worst_case_weight=hybrid_worst_case_weight,
    )
    hybrid_best_score = (
        base_robust_score
        + 0.03 * _program_collection_overlap(hybrid_best_programs, methods["cNMF"])
        + float(hybrid_diversity_weight) * _program_internal_diversity(hybrid_best_programs)
    )
    hybrid_best_label = "cnmf_only"
    hybrid_best_mode = "score_sum"
    hybrid_best_gate = 0.0
    hybrid_best_adapt = False
    hybrid_best_weights: list[float] = [0.0]

    source_scores: list[tuple[str, dict[str, list[tuple[str, float]]]]] = [
        ("kmeans_graph", km_scores),
        ("kmeans_nograph", km_nograph_scores),
        ("leiden", leiden_scores),
        ("leiden_tuned", leiden_tuned_scores),
        ("spectral", spectral_scores),
        ("etm", etm_scores),
        ("attention", attn_scores),
        ("wgcna", wgcna_scores),
    ]
    if ensemble_scores:
        source_scores.append(("ensemble", ensemble_scores))
    source_scores = [(name, scores) for name, scores in source_scores if scores]
    source_count_before_prescreen = len(source_scores)
    prescreen_topk = max(1, int(hybrid_source_prescreen_topk))
    if source_count_before_prescreen > prescreen_topk:
        prescreen_ranked: list[tuple[float, str, dict[str, list[tuple[str, float]]]]] = []
        for name, scores in source_scores:
            prelim = _fuse_program_scores(
                primary_scores=cnmf_scores,
                secondary_scores=scores,
                top_n_genes=top_n_genes,
                secondary_weight=0.20,
            )
            robust_score, _ = _robust_program_score_for_tuning(
                prelim,
                tuning_sets,
                gene_names,
                auxiliary_tuning_sets=auxiliary_sets,
                worst_case_weight=hybrid_worst_case_weight,
            )
            objective = (
                robust_score
                + 0.03 * _program_collection_overlap(prelim, methods["cNMF"])
                + float(hybrid_diversity_weight) * _program_internal_diversity(prelim)
            )
            prescreen_ranked.append((objective, name, scores))
        prescreen_ranked.sort(key=lambda x: x[0], reverse=True)
        source_scores = [(name, scores) for _, name, scores in prescreen_ranked[:prescreen_topk]]
        logger.info(
            "Hybrid source prescreen kept %d/%d sources: %s",
            len(source_scores),
            source_count_before_prescreen,
            [name for name, _ in source_scores],
        )

    def _weight_profiles(total_weight: float, n_sources: int) -> list[list[float]]:
        if n_sources == 1:
            return [[float(total_weight)]]
        if total_weight <= 0:
            return [[0.0 for _ in range(n_sources)]]
        profiles: list[list[float]] = [[float(total_weight / n_sources) for _ in range(n_sources)]]
        if n_sources == 2:
            profiles.extend(
                [
                    [0.70 * total_weight, 0.30 * total_weight],
                    [0.30 * total_weight, 0.70 * total_weight],
                ]
            )
        elif n_sources == 3:
            profiles.extend(
                [
                    [0.50 * total_weight, 0.25 * total_weight, 0.25 * total_weight],
                    [0.25 * total_weight, 0.50 * total_weight, 0.25 * total_weight],
                    [0.25 * total_weight, 0.25 * total_weight, 0.50 * total_weight],
                ]
            )
        return profiles

    objective_cache: dict[tuple[tuple[str, tuple[str, ...]], ...], tuple[float, float, float, float]] = {}
    max_combo = min(max(int(hybrid_max_source_combo), 1), min(3, len(source_scores)))
    for combo_size in range(1, max_combo + 1):
        for combo in itertools.combinations(source_scores, combo_size):
            combo_names = [name for name, _ in combo]
            for total_weight in [0.10, 0.20, 0.35]:
                for weights in _weight_profiles(total_weight, combo_size):
                    for gate in [0.00, 0.08]:
                        for adapt in [False, True]:
                            for mode in fusion_modes:
                                secondary_sources = [
                                    (name, scores, float(w))
                                    for (name, scores), w in zip(combo, weights)
                                    if float(w) > 0.0
                                ]
                                candidate = _fuse_program_scores_multi(
                                    primary_scores=cnmf_scores,
                                    secondary_sources=secondary_sources,
                                    top_n_genes=top_n_genes,
                                    min_overlap_jaccard=gate,
                                    adapt_weight_by_overlap=adapt,
                                    fusion_mode=mode,
                                )
                                cache_key = _programs_fingerprint(candidate)
                                cached = objective_cache.get(cache_key)
                                if cached is None:
                                    robust_score, _ = _robust_program_score_for_tuning(
                                        candidate,
                                        tuning_sets,
                                        gene_names,
                                        auxiliary_tuning_sets=auxiliary_sets,
                                        worst_case_weight=hybrid_worst_case_weight,
                                    )
                                    overlap_score = _program_collection_overlap(candidate, methods["cNMF"])
                                    diversity_score = _program_internal_diversity(candidate)
                                    objective = (
                                        robust_score
                                        + 0.03 * overlap_score
                                        + float(hybrid_diversity_weight) * diversity_score
                                    )
                                    objective_cache[cache_key] = (
                                        objective,
                                        robust_score,
                                        overlap_score,
                                        diversity_score,
                                    )
                                else:
                                    objective, robust_score, overlap_score, diversity_score = cached
                                candidate_label = (
                                    f"{'+'.join(combo_names)}:"
                                    f"mode={mode}:"
                                    f"tw={total_weight:.2f}:"
                                    f"w={','.join(f'{w:.2f}' for w in weights)}:"
                                    f"gate={gate:.2f}:adapt={int(adapt)}"
                                )
                                tuning_rows.append(
                                    {
                                        "seed": seed,
                                        "tuning_target": "hybrid_fusion",
                                        "candidate": candidate_label,
                                        "score": objective,
                                    }
                                )
                                if objective > hybrid_best_score:
                                    hybrid_best_score = objective
                                    hybrid_best_label = "+".join(combo_names)
                                    hybrid_best_mode = mode
                                    hybrid_best_gate = gate
                                    hybrid_best_adapt = adapt
                                    hybrid_best_weights = list(weights)
                                    hybrid_best_programs = candidate

    methods["nPathway-Hybrid-Tuned"] = hybrid_best_programs
    logger.info(
        "Hybrid fusion tuned: sources=%s mode=%s weights=%s gate=%.2f adapt=%s score=%.4f",
        hybrid_best_label,
        hybrid_best_mode,
        ",".join(f"{w:.2f}" for w in hybrid_best_weights),
        hybrid_best_gate,
        hybrid_best_adapt,
        hybrid_best_score,
    )
    _push_runtime("nPathway-Hybrid-Tuned", t_fit)

    t_fit = time.perf_counter()
    expr = ExpressionClusteringBaseline(
        n_programs=n_programs,
        random_state=seed,
    )
    expr.fit(X_expr, gene_names)
    methods["Expr-Cluster"] = expr.get_programs()
    _push_runtime("Expr-Cluster", t_fit)

    t_fit = time.perf_counter()
    rand = RandomProgramDiscovery(
        n_programs=n_programs,
        genes_per_program=top_n_genes,
        random_state=int(rng.integers(0, 1_000_000)),
    )
    rand.fit(gene_embeddings, gene_names)
    methods["Random"] = rand.get_programs()
    _push_runtime("Random", t_fit)

    # Optional latest external baselines
    if enable_latest_baselines:
        try:
            t_fit = time.perf_counter()
            official = OfficialCNMFProgramDiscovery(
                n_programs=n_programs,
                n_iter=official_cnmf_iters,
                top_n_genes=top_n_genes,
                max_nmf_iter=official_cnmf_max_iter,
                random_state=seed,
            )
            official.fit(X_nonneg, gene_names)
            methods["cNMF-Official"] = official.get_programs()
            _push_runtime("cNMF-Official", t_fit)
        except Exception as exc:
            skipped_rows.append(
                {"seed": seed, "method": "cNMF-Official", "reason": str(exc)}
            )
            _push_runtime(
                "cNMF-Official",
                t_fit,
                status="error",
                reason=str(exc),
            )

        try:
            t_fit = time.perf_counter()
            starcat = StarCATReferenceProgramDiscovery(
                reference=starcat_reference,
                top_n_genes=top_n_genes,
                cache_dir=starcat_cache_dir,
            )
            starcat.fit(graph_embeddings, gene_names)
            methods["starCAT-Ref"] = starcat.get_programs()
            _push_runtime("starCAT-Ref", t_fit)
        except Exception as exc:
            skipped_rows.append(
                {"seed": seed, "method": "starCAT-Ref", "reason": str(exc)}
            )
            _push_runtime(
                "starCAT-Ref",
                t_fit,
                status="error",
                reason=str(exc),
            )

    # Curated reference baseline
    methods[curated_baseline_name] = dict(evaluation_sets)
    runtime_rows.append(
        {
            "seed": seed,
            "method": curated_baseline_name,
            "fit_sec": 0.0,
            "status": "ok",
            "reason": "",
        }
    )

    tuning_df = pd.DataFrame(tuning_rows)
    skipped_df = pd.DataFrame(skipped_rows)
    runtime_df = pd.DataFrame(runtime_rows)
    return methods, tuning_df, skipped_df, runtime_df


def summarize_results(
    recovery_all: pd.DataFrame,
    discovery_all: pd.DataFrame,
    power_all: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rec_seed = (
        recovery_all
        .groupby(["dataset", "seed", "method"], as_index=False)
        .agg(
            recovery_mean_n_sig=("n_sig", "mean"),
            recovery_mean_hallmark_jaccard=("best_hallmark_jaccard", "mean"),
        )
    )

    disc_seed = discovery_all.rename(
        columns={
            "coverage": "discovery_coverage",
            "redundancy": "discovery_redundancy",
            "novelty": "discovery_novelty",
            "specificity": "discovery_specificity",
            "mean_hallmark_alignment": "discovery_mean_hallmark_alignment",
            "mean_program_size": "discovery_mean_program_size",
            "n_programs": "discovery_n_programs",
        }
    )
    merged = rec_seed.merge(disc_seed, on=["dataset", "seed", "method"], how="outer")

    if not power_all.empty:
        pwr_seed = (
            power_all
            .groupby(["dataset", "seed", "method"], as_index=False)
            .agg(power_mean_tpr=("tpr", "mean"), power_mean_fpr=("fpr", "mean"))
        )
        merged = merged.merge(pwr_seed, on=["dataset", "seed", "method"], how="outer")

    agg = (
        merged
        .groupby(["dataset", "method"], as_index=False)
        .agg({
            col: ["mean", "std"]
            for col in merged.columns
            if col not in {"dataset", "seed", "method"}
        })
    )
    agg.columns = [
        c[0] if c[0] in {"dataset", "method"} else f"{c[0]}_{c[1]}"
        for c in agg.columns.to_flat_index()
    ]
    merged["track"] = merged["method"].map(_method_track)
    agg["track"] = agg["method"].map(_method_track)
    return merged, agg


def pairwise_stats_vs_cnmf(
    seed_summary: pd.DataFrame,
    *,
    metrics: list[tuple[str, str]] | None = None,
    n_bootstrap: int = 2000,
    bootstrap_alpha: float = 0.05,
) -> pd.DataFrame:
    if metrics is None:
        metrics = [
            ("recovery_mean_n_sig", "greater"),
            ("recovery_mean_hallmark_jaccard", "greater"),
            ("discovery_mean_hallmark_alignment", "greater"),
            ("power_mean_tpr", "greater"),
            ("power_mean_fpr", "less"),
        ]

    rows: list[dict[str, Any]] = []
    for dataset in sorted(seed_summary["dataset"].unique()):
        df_d = seed_summary[seed_summary["dataset"] == dataset]
        methods = sorted(df_d["method"].unique())
        if "cNMF" not in methods:
            continue
        ref = df_d[df_d["method"] == "cNMF"]
        for method in methods:
            if method == "cNMF":
                continue
            cur = df_d[df_d["method"] == method]
            pair = cur.merge(ref, on=["dataset", "seed"], suffixes=("_m", "_r"))
            for metric, alt in metrics:
                cm = f"{metric}_m"
                cr = f"{metric}_r"
                if cm not in pair.columns or cr not in pair.columns:
                    continue
                vals = pair[[cm, cr]].dropna()
                n = len(vals)
                delta = vals[cm].values - vals[cr].values if n > 0 else np.array([])
                if n < 2:
                    p = np.nan
                else:
                    try:
                        p = float(
                            scipy_stats.wilcoxon(
                                vals[cm].values,
                                vals[cr].values,
                                alternative=alt,
                                zero_method="wilcox",
                            ).pvalue
                        )
                    except ValueError:
                        p = np.nan
                mean_m = float(vals[cm].mean()) if n > 0 else np.nan
                mean_r = float(vals[cr].mean()) if n > 0 else np.nan
                delta_mean = (
                    mean_m - mean_r
                    if np.isfinite(mean_m) and np.isfinite(mean_r)
                    else np.nan
                )
                delta_median = float(np.median(delta)) if n > 0 else np.nan
                delta_std = float(np.std(delta, ddof=1)) if n >= 2 else np.nan
                dz = _paired_effect_size_dz(delta)
                rank_biserial = paired_rank_biserial_correlation(delta)
                ci_low, ci_high = _bootstrap_ci_mean(
                    delta.astype(np.float64),
                    n_bootstrap=n_bootstrap,
                    alpha=bootstrap_alpha,
                    seed=13 + n,
                )

                if n > 0:
                    if alt == "greater":
                        wins = int(np.sum(delta > 0))
                        losses = int(np.sum(delta < 0))
                    else:
                        wins = int(np.sum(delta < 0))
                        losses = int(np.sum(delta > 0))
                    n_non_ties = wins + losses
                    win_rate = float(wins / n_non_ties) if n_non_ties > 0 else np.nan
                    if n_non_ties >= 2:
                        sign_p = float(
                            scipy_stats.binomtest(
                                wins,
                                n_non_ties,
                                p=0.5,
                                alternative="greater",
                            ).pvalue
                        )
                    else:
                        sign_p = np.nan
                else:
                    n_non_ties = 0
                    win_rate = np.nan
                    sign_p = np.nan

                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "reference": "cNMF",
                        "metric": metric,
                        "alternative": alt,
                        "n_pairs": n,
                        "n_non_ties": n_non_ties,
                        "method_mean": mean_m,
                        "reference_mean": mean_r,
                        "delta_mean": delta_mean,
                        "delta_median": delta_median,
                        "delta_std": delta_std,
                        "effect_size_dz": dz,
                        "effect_size_rank_biserial": rank_biserial,
                        "effect_size_rank_biserial_directional": (
                            rank_biserial if alt == "greater" else -rank_biserial
                        ),
                        "delta_ci95_low": ci_low,
                        "delta_ci95_high": ci_high,
                        "win_rate": win_rate,
                        "delta_directional": (
                            delta_mean if alt == "greater" else -delta_mean
                        ),
                        "p_value": p,
                        "sign_p_value": sign_p,
                        "track": _method_track(method),
                    }
                )

    stats_df = pd.DataFrame(rows)
    if stats_df.empty:
        stats_df["fdr_bh"] = []
        stats_df["fdr_bh_by_metric"] = []
        stats_df["fdr_bh_by_track_metric"] = []
        stats_df["sign_fdr_bh"] = []
        stats_df["sign_fdr_bh_by_metric"] = []
        stats_df["sign_fdr_bh_by_track_metric"] = []
        return stats_df

    stats_df["fdr_bh"] = benjamini_hochberg_fdr_grouped(
        stats_df, p_value_col="p_value"
    ).to_numpy(dtype=np.float64)
    stats_df["fdr_bh_by_metric"] = benjamini_hochberg_fdr_grouped(
        stats_df, p_value_col="p_value", group_cols=["metric"]
    ).to_numpy(dtype=np.float64)
    stats_df["fdr_bh_by_track_metric"] = benjamini_hochberg_fdr_grouped(
        stats_df,
        p_value_col="p_value",
        group_cols=["track", "metric", "alternative"],
    ).to_numpy(dtype=np.float64)

    stats_df["sign_fdr_bh"] = benjamini_hochberg_fdr_grouped(
        stats_df, p_value_col="sign_p_value"
    ).to_numpy(dtype=np.float64)
    stats_df["sign_fdr_bh_by_metric"] = benjamini_hochberg_fdr_grouped(
        stats_df, p_value_col="sign_p_value", group_cols=["metric"]
    ).to_numpy(dtype=np.float64)
    stats_df["sign_fdr_bh_by_track_metric"] = benjamini_hochberg_fdr_grouped(
        stats_df,
        p_value_col="sign_p_value",
        group_cols=["track", "metric", "alternative"],
    ).to_numpy(dtype=np.float64)
    return stats_df


def _summary_metric_column(metric: str) -> str:
    return f"{metric}_mean"


def build_primary_leaderboard(
    method_summary: pd.DataFrame,
    primary_metrics: list[str],
) -> pd.DataFrame:
    """Build track-wise leaderboard from pre-registered primary endpoints."""
    if not primary_metrics:
        return pd.DataFrame(
            columns=[
                "dataset",
                "track",
                "method",
                "mean_rank",
                "n_primary_metrics",
            ]
        )
    rows: list[dict[str, Any]] = []
    for dataset in sorted(method_summary["dataset"].unique()):
        df_d = method_summary[method_summary["dataset"] == dataset]
        for track in sorted(df_d["track"].unique()):
            df_t = df_d[df_d["track"] == track].copy()
            if df_t.empty:
                continue
            ranks_store: dict[str, list[float]] = {
                m: [] for m in df_t["method"].tolist()
            }
            for metric in primary_metrics:
                col = _summary_metric_column(metric)
                if col not in df_t.columns:
                    continue
                alt = PRIMARY_METRIC_DIRECTIONS.get(metric, "greater")
                asc = alt == "less"
                ranks = df_t[col].rank(ascending=asc, method="average")
                for method, rank in zip(df_t["method"], ranks):
                    ranks_store[method].append(float(rank))
            for method, vals in ranks_store.items():
                if not vals:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "track": track,
                        "method": method,
                        "mean_rank": float(np.mean(vals)),
                        "n_primary_metrics": len(vals),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["dataset", "track", "mean_rank", "method"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def build_track_leaderboard(method_summary: pd.DataFrame) -> pd.DataFrame:
    """Build per-track leaderboard ranks from method summary metrics."""
    metrics = [
        ("recovery_mean_n_sig_mean", False),
        ("discovery_mean_hallmark_alignment_mean", False),
        ("power_mean_tpr_mean", False),
        ("power_mean_fpr_mean", True),
    ]
    rows: list[dict[str, Any]] = []
    for dataset in sorted(method_summary["dataset"].unique()):
        df_d = method_summary[method_summary["dataset"] == dataset]
        for track in sorted(df_d["track"].unique()):
            df_t = df_d[df_d["track"] == track].copy()
            if df_t.empty:
                continue
            rank_store: dict[str, list[float]] = {
                m: [] for m in df_t["method"].tolist()
            }
            for metric, asc in metrics:
                if metric not in df_t.columns:
                    continue
                ranks = df_t[metric].rank(ascending=asc, method="average")
                for method, rank in zip(df_t["method"], ranks):
                    rank_store[method].append(float(rank))
            for method, vals in rank_store.items():
                if not vals:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "track": track,
                        "method": method,
                        "mean_rank": float(np.mean(vals)),
                        "n_rank_metrics": len(vals),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["dataset", "track", "mean_rank", "method"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def make_figures(seed_summary: pd.DataFrame, agg: pd.DataFrame, figures_dir: Path) -> None:
    sns.set_theme(style="whitegrid")

    # Figure 1: Ablation-focused boxplot on recovery
    focus_methods = [
        "nPathway-KMeans",
        "nPathway-KMeans-NoGraph",
        "nPathway-KMeans-Tuned",
        "nPathway-Refined",
        "cNMF",
    ]
    sub = seed_summary[seed_summary["method"].isin(focus_methods)].copy()
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.boxplot(
            data=sub,
            x="method",
            y="recovery_mean_n_sig",
            hue="dataset",
            ax=ax,
        )
        ax.set_title("Ablation/Tuning: Recovery Mean n_sig")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(figures_dir / "expanded_ablation_recovery.png", dpi=180)
        plt.close(fig)

    # Figure 2: Per-dataset method rank heatmap
    rank_rows: list[dict[str, Any]] = []
    for dataset in sorted(agg["dataset"].unique()):
        df_d = agg[agg["dataset"] == dataset].copy()
        for metric, asc in [
            ("recovery_mean_n_sig_mean", False),
            ("discovery_mean_hallmark_alignment_mean", False),
            ("power_mean_tpr_mean", False),
            ("power_mean_fpr_mean", True),
        ]:
            if metric not in df_d.columns:
                continue
            rank = df_d[metric].rank(ascending=asc, method="average")
            for method, r in zip(df_d["method"], rank):
                rank_rows.append(
                    {"dataset": dataset, "method": method, "metric": metric, "rank": r}
                )
    if rank_rows:
        rank_df = pd.DataFrame(rank_rows)
        pivot = (
            rank_df
            .groupby(["dataset", "method"], as_index=False)["rank"]
            .mean()
            .pivot(index="method", columns="dataset", values="rank")
            .sort_values(by=sorted(rank_df["dataset"].unique().tolist()), ascending=True)
        )
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGn_r", ax=ax, cbar_kws={"label": "Mean rank (lower better)"})
        ax.set_title("Expanded Suite: Mean Rank by Dataset")
        fig.tight_layout()
        fig.savefig(figures_dir / "expanded_dataset_rank_heatmap.png", dpi=180)
        plt.close(fig)


def _max_jaccard_to_collection(
    genes: set[str],
    collection: dict[str, list[str]],
) -> float:
    best = 0.0
    for gset in collection.values():
        ref = set(gset)
        union = genes | ref
        if not union:
            continue
        best = max(best, len(genes & ref) / len(union))
    return float(best)


def _best_set_match(
    genes: set[str],
    collection: dict[str, list[str]],
) -> tuple[str, float, int]:
    best_name = ""
    best_j = 0.0
    best_overlap = 0
    for name, gset in collection.items():
        ref = set(gset)
        union = genes | ref
        if not union:
            continue
        overlap = len(genes & ref)
        j = overlap / len(union)
        if j > best_j:
            best_name = name
            best_j = float(j)
            best_overlap = int(overlap)
    return best_name, best_j, best_overlap


def _best_fisher_enrichment(
    genes: set[str],
    collection: dict[str, list[str]],
    universe: set[str],
    fdr_alpha: float = 0.05,
) -> tuple[str, float, float, int, float]:
    """Fisher's exact test with Benjamini-Hochberg FDR correction.

    Returns the best-matching term by FDR-adjusted p-value.

    Returns
    -------
    tuple of (term_name, raw_p, odds_ratio, overlap_count, fdr_adjusted_p)
    Returns empty strings / NaN when no significant enrichment is found.
    """
    if not genes or not collection:
        return "", np.nan, np.nan, 0, np.nan
    g = genes & universe
    if len(g) < 3:
        return "", np.nan, np.nan, 0, np.nan
    n = len(universe)

    # Collect all results first for FDR correction
    results: list[tuple[str, float, float, int]] = []
    for name, members in collection.items():
        m = set(members) & universe
        if len(m) < 3:
            continue
        a = len(g & m)
        if a == 0:
            continue
        b = len(g) - a
        c = len(m) - a
        d = n - a - b - c
        if d < 0:
            continue
        odds_ratio, p = scipy_stats.fisher_exact(
            [[a, b], [c, d]],
            alternative="greater",
        )
        if not np.isfinite(p):
            continue
        results.append((name, float(p), float(odds_ratio) if np.isfinite(odds_ratio) else np.nan, int(a)))

    if not results:
        return "", np.nan, np.nan, 0, np.nan

    # Benjamini-Hochberg FDR correction (Benjamini & Hochberg 1995)
    # Algorithm: sort p-values ascending → apply p[i]*m/i → step-up monotonicity
    m_tests = len(results)
    p_values = np.array([r[1] for r in results])
    sorted_indices = np.argsort(p_values)          # indices that sort p ascending
    sorted_p = p_values[sorted_indices]             # p_(1) ≤ p_(2) ≤ ... ≤ p_(m)
    ranks = np.arange(1, m_tests + 1, dtype=float)  # ranks 1..m in sorted order
    bh_raw = sorted_p * m_tests / ranks             # BH formula: p_(i) * m / i
    # Step-up: enforce monotonicity from largest rank downward
    bh_monotone = np.minimum.accumulate(bh_raw[::-1])[::-1]
    bh_final = np.empty(m_tests)
    bh_final[sorted_indices] = np.clip(bh_monotone, 0.0, 1.0)

    # Pick best term by adjusted p-value
    best_idx = int(np.argmin(bh_final))
    best_name, best_p, best_or, best_overlap = results[best_idx]
    best_fdr = float(bh_final[best_idx])

    return best_name, best_p, best_or, best_overlap, best_fdr


def build_case_studies_from_methods(
    *,
    dataset: str,
    methods: dict[str, dict[str, list[str]]],
    evaluation_sets: dict[str, list[str]],
    go_bp_sets: dict[str, list[str]],
    kegg_sets: dict[str, list[str]],
    gene_names: list[str],
    top_k: int = 2,
) -> pd.DataFrame:
    """Extract concise biological case studies from discovered programs.

    Improvements over the original implementation:
    - GO/KEGG enrichment uses BH-FDR correction across all tested sets.
    - case_score is based purely on eval-set Jaccard (not inflated by novelty).
    - cNMF counterpart Jaccard and top genes are reported for direct comparison.
    - Both nPathway-Hybrid-Tuned and fallback methods are tried in priority order.
    """
    method_priority = [
        "nPathway-Hybrid-Tuned",
        "nPathway-DiscRefined",
        "nPathway-Refined",
        "nPathway-KMeans",
        "nPathway-KMeans-NoGraph",
    ]
    target_method = next(
        (m for m in method_priority if m in methods),
        None,
    )
    if target_method is None:
        return pd.DataFrame()

    programs = methods[target_method]
    cnmf_programs = methods.get("cNMF", {})

    universe = set(gene_names)
    cnmf_sets = {k: set(v) for k, v in cnmf_programs.items()}
    rows: list[dict[str, Any]] = []
    for pname, genes_list in programs.items():
        gset = set(genes_list)
        if len(gset) < 3:
            continue
        best_eval_name, best_eval_j, best_eval_overlap = _best_set_match(
            gset,
            evaluation_sets,
        )

        # cNMF counterpart: most similar cNMF program (for direct comparison)
        cnmf_best_prog = ""
        cnmf_max_j = 0.0
        cnmf_counterpart_genes = ""
        for cname, cset in cnmf_sets.items():
            union = gset | cset
            if not union:
                continue
            j = len(gset & cset) / len(union)
            if j > cnmf_max_j:
                cnmf_max_j = j
                cnmf_best_prog = cname
                cnmf_counterpart_genes = ",".join(list(cnmf_programs[cname])[:10])
        novelty = 1.0 - cnmf_max_j

        # GO and KEGG enrichment with BH-FDR correction
        go_name, go_p, go_or, go_overlap, go_fdr = _best_fisher_enrichment(
            gset, go_bp_sets, universe
        )
        kegg_name, kegg_p, kegg_or, kegg_overlap, kegg_fdr = _best_fisher_enrichment(
            gset, kegg_sets, universe
        )

        # case_score: eval-set Jaccard only (not inflated by novelty)
        # Programs with strong biological annotation rank highest
        case_score = best_eval_j
        # Boost programs with significant GO/KEGG hits (FDR < 0.05)
        if np.isfinite(go_fdr) and go_fdr < 0.05:
            case_score += 0.1
        if np.isfinite(kegg_fdr) and kegg_fdr < 0.05:
            case_score += 0.1

        rows.append(
            {
                "dataset": dataset,
                "method": target_method,
                "program": pname,
                "n_genes": len(gset),
                "best_eval_set": best_eval_name,
                "best_eval_jaccard": best_eval_j,
                "best_eval_overlap": best_eval_overlap,
                "novelty_vs_cnmf": novelty,
                "cnmf_counterpart_program": cnmf_best_prog,
                "cnmf_counterpart_jaccard": round(cnmf_max_j, 4),
                "cnmf_counterpart_top_genes": cnmf_counterpart_genes,
                "best_go_bp": go_name,
                "best_go_bp_p": go_p,
                "best_go_bp_fdr": go_fdr,
                "best_go_bp_odds_ratio": go_or,
                "best_go_bp_overlap": go_overlap,
                "best_kegg": kegg_name,
                "best_kegg_p": kegg_p,
                "best_kegg_fdr": kegg_fdr,
                "best_kegg_odds_ratio": kegg_or,
                "best_kegg_overlap": kegg_overlap,
                "top_genes_preview": ",".join(list(genes_list)[:10]),
                "case_score": case_score,
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(
        "case_score",
        ascending=False,
    )
    return df.head(max(1, top_k)).reset_index(drop=True)


def build_fairness_manifest(
    seed_summary: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Construct method-level fairness metadata for publication reporting."""
    methods = sorted(seed_summary["method"].dropna().unique().tolist())
    rows: list[dict[str, Any]] = []
    for method in methods:
        rows.append(
            {
                "method": method,
                "track": _method_track(method),
                "shared_input_pipeline": not method.startswith("Curated-"),
                "seed_controlled": not method.startswith("Curated-"),
                "n_programs_budget": args.n_programs,
                "top_n_genes_budget": args.top_n_genes,
                "etm_epochs_budget": args.etm_epochs,
                "official_cnmf_max_iter_budget": args.official_cnmf_max_iter,
                "official_cnmf_iters_budget": args.official_cnmf_iters,
                "hybrid_max_source_combo_budget": args.hybrid_max_source_combo,
                "hybrid_source_prescreen_topk_budget": args.hybrid_source_prescreen_topk,
                "hybrid_worst_case_weight_budget": args.hybrid_worst_case_weight,
                "hybrid_diversity_weight_budget": args.hybrid_diversity_weight,
                "hybrid_fusion_modes_budget": ",".join(args.hybrid_fusion_modes),
            }
        )
    return pd.DataFrame(rows)


def _module_version(module_name: str) -> str:
    try:
        mod = __import__(module_name)
        return str(getattr(mod, "__version__", "unknown"))
    except Exception:
        return "unavailable"


def write_protocol_manifest(
    *,
    out_path: Path,
    args: argparse.Namespace,
    primary_metrics: list[str],
    elapsed_sec: float,
    provenance_df: pd.DataFrame,
) -> None:
    """Write JSON manifest containing protocol, environment, and fairness knobs."""
    manifest = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "elapsed_sec": float(elapsed_sec),
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": {
            "numpy": _module_version("numpy"),
            "pandas": _module_version("pandas"),
            "scipy": _module_version("scipy"),
            "scanpy": _module_version("scanpy"),
            "anndata": _module_version("anndata"),
            "torch": _module_version("torch"),
        },
        "protocol": {
            "datasets": args.datasets,
            "seeds": args.seeds,
            "strict_datasets": bool(args.strict_datasets),
            "tuning_collection": args.tuning_collection,
            "evaluation_collection": args.evaluation_collection,
            "same_collection_tune_fraction": args.same_collection_tune_fraction,
            "hybrid_max_source_combo": args.hybrid_max_source_combo,
            "hybrid_source_prescreen_topk": args.hybrid_source_prescreen_topk,
            "hybrid_worst_case_weight": args.hybrid_worst_case_weight,
            "hybrid_diversity_weight": args.hybrid_diversity_weight,
            "hybrid_fusion_modes": args.hybrid_fusion_modes,
            "primary_metrics": primary_metrics,
            "stats_bootstrap": int(args.stats_bootstrap),
            "power_trials": int(args.power_trials),
        },
        "dataset_provenance": provenance_df.to_dict(orient="records"),
    }
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["pbmc3k", "tabula_muris"],
        help="Datasets to benchmark.",
    )
    parser.add_argument(
        "--publication-datasets",
        action="store_true",
        help=(
            "Use publication validation panel: pbmc3k pbmc68k_reduced "
            "paul15 moignard15 burczynski06 tabula_muris."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44],
        help="Random seeds.",
    )
    parser.add_argument(
        "--n-programs",
        type=int,
        default=20,
        help="Default number of programs for fixed-k methods.",
    )
    parser.add_argument(
        "--top-n-genes",
        type=int,
        default=50,
        help="Top genes for refined/cNMF/ETM outputs.",
    )
    parser.add_argument(
        "--etm-epochs",
        type=int,
        default=80,
        help="ETM epochs for expanded suite.",
    )
    parser.add_argument(
        "--power-trials",
        type=int,
        default=10,
        help="Trials per fold-change in power benchmark.",
    )
    parser.add_argument(
        "--tabula-muris-path",
        type=Path,
        default=None,
        help="Optional local path to a real Tabula Muris .h5ad file.",
    )
    parser.add_argument(
        "--strict-datasets",
        action="store_true",
        help="Disallow proxy datasets and publication-unsafe fallbacks (pseudo-labels, MSigDB species auto-switch, tune/eval reuse).",
    )
    parser.add_argument(
        "--stats-bootstrap",
        type=int,
        default=2000,
        help="Bootstrap resamples for paired delta confidence intervals.",
    )
    parser.add_argument(
        "--msigdb-species",
        choices=["auto", "human", "mouse"],
        default="auto",
        help="MSigDB species for Hallmark gene sets.",
    )
    parser.add_argument(
        "--tuning-collection",
        choices=SUPPORTED_COLLECTIONS,
        default="kegg",
        help="MSigDB collection used strictly for hyperparameter tuning.",
    )
    parser.add_argument(
        "--evaluation-collection",
        choices=SUPPORTED_COLLECTIONS,
        default="hallmark",
        help="MSigDB collection used strictly for final evaluation.",
    )
    parser.add_argument(
        "--same-collection-tune-fraction",
        type=float,
        default=0.5,
        help=(
            "If tuning and evaluation collections are identical, fraction of "
            "sets assigned to tuning split."
        ),
    )
    parser.add_argument(
        "--hybrid-max-source-combo",
        type=int,
        default=2,
        help="Max number of secondary sources jointly fused for hybrid tuning.",
    )
    parser.add_argument(
        "--hybrid-source-prescreen-topk",
        type=int,
        default=5,
        help="Pre-screen sources and keep top-k before multi-source hybrid search.",
    )
    parser.add_argument(
        "--hybrid-worst-case-weight",
        type=float,
        default=0.2,
        help="Weight on worst-collection score in robust hybrid objective.",
    )
    parser.add_argument(
        "--hybrid-diversity-weight",
        type=float,
        default=0.02,
        help="Weight on internal program diversity in hybrid objective.",
    )
    parser.add_argument(
        "--hybrid-fusion-modes",
        nargs="+",
        choices=["score_sum", "rrf"],
        default=["score_sum", "rrf"],
        help="Fusion operators for hybrid search (score_sum and/or rrf).",
    )
    parser.add_argument(
        "--primary-metrics",
        nargs="+",
        choices=sorted(PRIMARY_METRIC_DIRECTIONS.keys()),
        default=PRIMARY_METRIC_DEFAULTS,
        help="Pre-registered primary endpoint metrics used for primary leaderboard.",
    )
    parser.add_argument(
        "--case-study-topk",
        type=int,
        default=2,
        help="Number of biological case-study programs to extract per dataset.",
    )
    parser.add_argument(
        "--enable-latest-baselines",
        action="store_true",
        help="Enable latest external baselines (official cNMF, starCAT reference).",
    )
    parser.add_argument(
        "--official-cnmf-iters",
        type=int,
        default=4,
        help="Number of NMF repeats for official cNMF baseline.",
    )
    parser.add_argument(
        "--official-cnmf-max-iter",
        type=int,
        default=400,
        help="Per-run max NMF optimization iterations for official cNMF.",
    )
    parser.add_argument(
        "--starcat-reference",
        type=str,
        default="TCAT.V1",
        help="starCAT reference atlas key (e.g., TCAT.V1).",
    )
    parser.add_argument(
        "--starcat-cache-dir",
        type=Path,
        default=Path("data") / "starcat_cache",
        help="Cache directory for downloaded starCAT references.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results") / "expanded_suite",
        help="Output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.publication_datasets:
        args.datasets = list(SUPPORTED_DATASETS)
    unknown_datasets = sorted(set(args.datasets) - set(SUPPORTED_DATASETS))
    if unknown_datasets:
        raise ValueError(
            f"Unsupported dataset(s): {unknown_datasets}. "
            f"Supported: {list(SUPPORTED_DATASETS)}"
        )
    if not args.primary_metrics:
        raise ValueError("At least one --primary-metrics value is required.")
    primary_metric_specs = [
        (m, PRIMARY_METRIC_DIRECTIONS[m]) for m in args.primary_metrics
    ]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    t0 = time.time()

    tables_dir, figures_dir = _ensure_dirs(args.outdir)
    logger.info("Expanded suite started")
    logger.info("Datasets=%s", args.datasets)
    logger.info("Seeds=%s", args.seeds)
    logger.info(
        "Leakage-safe protocol: tuning=%s evaluation=%s primary=%s",
        args.tuning_collection,
        args.evaluation_collection,
        args.primary_metrics,
    )
    logger.info(
        "Latest baselines=%s (official_cNMF iters=%d, max_iter=%d; starCAT ref=%s)",
        args.enable_latest_baselines,
        args.official_cnmf_iters,
        args.official_cnmf_max_iter,
        args.starcat_reference,
    )
    logger.info(
        "Hybrid tuning: max_combo=%d prescreen_topk=%d worst_case=%.2f diversity_w=%.3f modes=%s",
        args.hybrid_max_source_combo,
        args.hybrid_source_prescreen_topk,
        args.hybrid_worst_case_weight,
        args.hybrid_diversity_weight,
        args.hybrid_fusion_modes,
    )
    if args.strict_datasets:
        logger.info(
            "Strict publication safeguards enabled: proxy datasets, index-parity pseudo-labels, "
            "MSigDB species auto-switch, and tune/eval fallback reuse are disabled."
        )
    if len(args.seeds) < 5:
        logger.warning(
            "Only %d seeds configured; statistical power may be limited.",
            len(args.seeds),
        )

    bundles: dict[str, dict[str, Any]] = {}
    provenance_rows: list[dict[str, Any]] = []
    for ds in args.datasets:
        logger.info("Loading dataset bundle: %s", ds)
        bundles[ds] = load_dataset_bundle(
            ds,
            tabula_muris_path=args.tabula_muris_path,
            allow_proxy_datasets=not args.strict_datasets,
            strict_publication=args.strict_datasets,
            msigdb_species=args.msigdb_species,
            tuning_collection=args.tuning_collection,
            evaluation_collection=args.evaluation_collection,
            same_collection_tune_fraction=args.same_collection_tune_fraction,
            split_seed=args.seeds[0],
        )
        meta = bundles[ds]["dataset_meta"]
        logger.info(
            "  cells=%d genes=%d tune_sets=%d eval_sets=%d source=%s proxy=%s msigdb=%s",
            bundles[ds]["adata"].n_obs,
            len(bundles[ds]["gene_names"]),
            len(bundles[ds]["tuning_sets"]),
            len(bundles[ds]["evaluation_sets"]),
            meta.get("source", "unknown"),
            meta.get("is_proxy", False),
            meta.get("msigdb_species", "human"),
        )
        provenance_rows.append(
            {
                "dataset": ds,
                "source": meta.get("source", "unknown"),
                "is_proxy": bool(meta.get("is_proxy", False)),
                "msigdb_species": meta.get("msigdb_species", "human"),
                "tuning_collection": meta.get("tuning_collection", args.tuning_collection),
                "evaluation_collection": meta.get("evaluation_collection", args.evaluation_collection),
                "tuning_sets": int(meta.get("tuning_sets", len(bundles[ds]["tuning_sets"]))),
                "evaluation_sets": int(meta.get("evaluation_sets", len(bundles[ds]["evaluation_sets"]))),
                "go_bp_sets": int(meta.get("go_bp_sets", len(bundles[ds]["go_bp_sets"]))),
                "kegg_sets": int(meta.get("kegg_sets", len(bundles[ds]["kegg_sets"]))),
                "group_label_source": meta.get("group_label_source", "unknown"),
                "used_group_label_fallback": bool(meta.get("used_group_label_fallback", False)),
                "used_species_fallback_any": bool(meta.get("used_species_fallback_any", False)),
                "strict_publication": bool(meta.get("strict_publication", False)),
                "n_obs": int(meta.get("n_obs", bundles[ds]["adata"].n_obs)),
                "n_vars": int(meta.get("n_vars", len(bundles[ds]["gene_names"]))),
            }
        )

    recovery_frames: list[pd.DataFrame] = []
    discovery_frames: list[pd.DataFrame] = []
    power_frames: list[pd.DataFrame] = []
    tuning_frames: list[pd.DataFrame] = []
    skipped_frames: list[pd.DataFrame] = []
    runtime_frames: list[pd.DataFrame] = []
    case_study_frames: list[pd.DataFrame] = []

    curated_baseline_name = f"Curated-{args.evaluation_collection.upper()}"
    for ds in args.datasets:
        b = bundles[ds]
        auxiliary_tuning_sets: list[tuple[str, dict[str, list[str]]]] = []
        if (
            args.tuning_collection != "go_bp"
            and args.evaluation_collection != "go_bp"
            and b["go_bp_sets"]
        ):
            auxiliary_tuning_sets.append(("go_bp", b["go_bp_sets"]))
        if (
            args.tuning_collection != "kegg"
            and args.evaluation_collection != "kegg"
            and b["kegg_sets"]
        ):
            auxiliary_tuning_sets.append(("kegg", b["kegg_sets"]))
        for seed in args.seeds:
            logger.info("Run dataset=%s seed=%d", ds, seed)
            gbr.SEED = seed
            methods, tuning_df, skipped_df, runtime_df = discover_methods_extended(
                adata=b["adata"],
                graph_embeddings=b["graph_embeddings"],
                gene_embeddings=b["gene_embeddings"],
                gene_names=b["gene_names"],
                tuning_sets=b["tuning_sets"],
                evaluation_sets=b["evaluation_sets"],
                seed=seed,
                n_programs=args.n_programs,
                top_n_genes=args.top_n_genes,
                etm_epochs=args.etm_epochs,
                tune_k_candidates=[15, 20, 25],
                tune_leiden_candidates=[0.5, 1.0, 1.5],
                auxiliary_tuning_sets=auxiliary_tuning_sets,
                hybrid_max_source_combo=args.hybrid_max_source_combo,
                hybrid_source_prescreen_topk=args.hybrid_source_prescreen_topk,
                hybrid_worst_case_weight=args.hybrid_worst_case_weight,
                hybrid_diversity_weight=args.hybrid_diversity_weight,
                hybrid_fusion_modes=args.hybrid_fusion_modes,
                curated_baseline_name=curated_baseline_name,
                enable_latest_baselines=args.enable_latest_baselines,
                official_cnmf_iters=args.official_cnmf_iters,
                official_cnmf_max_iter=args.official_cnmf_max_iter,
                starcat_reference=args.starcat_reference,
                starcat_cache_dir=args.starcat_cache_dir,
            )

            recovery = gbr.benchmark_pathway_recovery(
                methods, b["evaluation_sets"], b["gene_names"], b["adata"]
            )
            discovery = gbr.benchmark_discovery(
                methods, b["evaluation_sets"], b["gene_names"]
            )
            power = gbr.benchmark_power(
                methods, b["gene_names"], b["adata"], n_trials=args.power_trials
            )

            recovery["dataset"] = ds
            discovery["dataset"] = ds
            recovery["seed"] = seed
            discovery["seed"] = seed
            if not power.empty:
                power["dataset"] = ds
                power["seed"] = seed

            if tuning_df.empty:
                tuning_df = pd.DataFrame(
                    columns=["seed", "tuning_target", "candidate", "score"]
                )
            tuning_df["dataset"] = ds

            if skipped_df.empty:
                skipped_df = pd.DataFrame(columns=["seed", "method", "reason"])
            skipped_df["dataset"] = ds

            if runtime_df.empty:
                runtime_df = pd.DataFrame(
                    columns=["seed", "method", "fit_sec", "status", "reason"]
                )
            runtime_df["dataset"] = ds

            recovery_frames.append(recovery)
            discovery_frames.append(discovery)
            if not power.empty:
                power_frames.append(power)
            tuning_frames.append(tuning_df)
            skipped_frames.append(skipped_df)
            runtime_frames.append(runtime_df)

            # Extract publication-ready biological case studies from ALL seeds
            # (not just the first) to enable cross-seed reproducibility analysis.
            case_df = build_case_studies_from_methods(
                dataset=ds,
                methods=methods,
                evaluation_sets=b["evaluation_sets"],
                go_bp_sets=b["go_bp_sets"],
                kegg_sets=b["kegg_sets"],
                gene_names=b["gene_names"],
                top_k=args.case_study_topk,
            )
            if not case_df.empty:
                case_df["seed"] = seed
                case_study_frames.append(case_df)

    recovery_all = pd.concat(recovery_frames, ignore_index=True)
    discovery_all = pd.concat(discovery_frames, ignore_index=True)
    power_all = (
        pd.concat(power_frames, ignore_index=True)
        if power_frames else pd.DataFrame(
            columns=["dataset", "seed", "method", "fold_change", "tpr", "fpr"]
        )
    )
    tuning_all = (
        pd.concat(tuning_frames, ignore_index=True)
        if tuning_frames else pd.DataFrame(columns=["dataset", "seed", "tuning_target", "candidate", "score"])
    )
    skipped_all = (
        pd.concat(skipped_frames, ignore_index=True)
        if skipped_frames else pd.DataFrame(columns=["dataset", "seed", "method", "reason"])
    )
    runtime_all = (
        pd.concat(runtime_frames, ignore_index=True)
        if runtime_frames
        else pd.DataFrame(
            columns=["dataset", "seed", "method", "fit_sec", "status", "reason"]
        )
    )
    case_studies = (
        pd.concat(case_study_frames, ignore_index=True)
        if case_study_frames
        else pd.DataFrame()
    )

    # Cross-seed reproducibility: for each (dataset, best_eval_set) pair,
    # count how many seeds produced a matching top program.
    if not case_studies.empty and "seed" in case_studies.columns:
        repro_rows: list[dict[str, Any]] = []
        n_seeds = len(args.seeds)
        for (ds_name, eval_set), grp in case_studies.groupby(["dataset", "best_eval_set"]):
            seeds_seen = grp["seed"].nunique()
            mean_jaccard = grp["best_eval_jaccard"].mean()
            std_jaccard = grp["best_eval_jaccard"].std()
            mean_fdr_go = grp["best_go_bp_fdr"].mean() if "best_go_bp_fdr" in grp.columns else np.nan
            repro_rows.append({
                "dataset": ds_name,
                "best_eval_set": eval_set,
                "n_seeds_reproduced": seeds_seen,
                "n_seeds_total": n_seeds,
                "reproducibility_fraction": seeds_seen / max(n_seeds, 1),
                "mean_eval_jaccard": mean_jaccard,
                "std_eval_jaccard": std_jaccard,
                "mean_go_fdr": mean_fdr_go,
            })
        reproducibility_df = pd.DataFrame(repro_rows).sort_values(
            ["dataset", "reproducibility_fraction"], ascending=[True, False]
        )
    else:
        reproducibility_df = pd.DataFrame()

    provenance_df = pd.DataFrame(provenance_rows).drop_duplicates(ignore_index=True)

    seed_summary, method_summary = summarize_results(
        recovery_all, discovery_all, power_all
    )
    stats_df = pairwise_stats_vs_cnmf(
        seed_summary,
        n_bootstrap=args.stats_bootstrap,
        bootstrap_alpha=0.05,
    )
    primary_stats_df = pairwise_stats_vs_cnmf(
        seed_summary,
        metrics=primary_metric_specs,
        n_bootstrap=args.stats_bootstrap,
        bootstrap_alpha=0.05,
    )
    track_leaderboard = build_track_leaderboard(method_summary)
    primary_leaderboard = build_primary_leaderboard(
        method_summary,
        primary_metrics=args.primary_metrics,
    )
    fairness_manifest = build_fairness_manifest(seed_summary, args)
    if "method" in stats_df.columns:
        stats_de_novo = stats_df[
            stats_df["method"].map(_method_track) == "de_novo"
        ].reset_index(drop=True)
        stats_reference = stats_df[
            stats_df["method"].map(_method_track) == "reference_guided"
        ].reset_index(drop=True)
    else:
        stats_de_novo = stats_df.copy()
        stats_reference = stats_df.copy()

    recovery_all.to_csv(tables_dir / "expanded_recovery_raw.csv", index=False)
    discovery_all.to_csv(tables_dir / "expanded_discovery_raw.csv", index=False)
    power_all.to_csv(tables_dir / "expanded_power_raw.csv", index=False)
    seed_summary.to_csv(tables_dir / "expanded_seed_summary.csv", index=False)
    method_summary.to_csv(tables_dir / "expanded_method_summary.csv", index=False)
    stats_df.to_csv(tables_dir / "expanded_wilcoxon_vs_cnmf.csv", index=False)
    primary_stats_df.to_csv(
        tables_dir / "expanded_primary_wilcoxon_vs_cnmf.csv",
        index=False,
    )
    stats_de_novo.to_csv(
        tables_dir / "expanded_wilcoxon_vs_cnmf_de_novo.csv", index=False
    )
    stats_reference.to_csv(
        tables_dir / "expanded_wilcoxon_vs_cnmf_reference_guided.csv", index=False
    )
    track_leaderboard.to_csv(
        tables_dir / "expanded_track_leaderboard.csv", index=False
    )
    primary_leaderboard.to_csv(
        tables_dir / "expanded_primary_leaderboard.csv",
        index=False,
    )
    tuning_all.to_csv(tables_dir / "expanded_tuning_log.csv", index=False)
    skipped_all.to_csv(tables_dir / "expanded_skipped_methods.csv", index=False)
    runtime_all.to_csv(tables_dir / "expanded_method_runtime.csv", index=False)
    provenance_df.to_csv(tables_dir / "expanded_dataset_provenance.csv", index=False)
    fairness_manifest.to_csv(tables_dir / "expanded_fairness_manifest.csv", index=False)
    case_studies.to_csv(tables_dir / "expanded_case_studies.csv", index=False)
    if not reproducibility_df.empty:
        reproducibility_df.to_csv(
            tables_dir / "expanded_case_study_reproducibility.csv", index=False
        )

    make_figures(seed_summary, method_summary, figures_dir)

    elapsed = time.time() - t0
    write_protocol_manifest(
        out_path=tables_dir / "expanded_protocol_manifest.json",
        args=args,
        primary_metrics=args.primary_metrics,
        elapsed_sec=elapsed,
        provenance_df=provenance_df,
    )
    logger.info("Expanded suite completed in %.1f sec", elapsed)
    logger.info("Outputs: %s", args.outdir)
    logger.info("Key summary: %s", tables_dir / "expanded_method_summary.csv")
    logger.info("Key stats: %s", tables_dir / "expanded_wilcoxon_vs_cnmf.csv")


if __name__ == "__main__":
    main()
