"""End-to-end dynamic pathway discovery for bulk RNA-seq two-group contrasts."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

from npathway.data.datasets import load_msigdb_gene_sets
from npathway.data.preprocessing import build_graph_regularized_embeddings
from npathway.discovery.clustering import ClusteringProgramDiscovery
from npathway.evaluation.enrichment import run_enrichment
from npathway.evaluation.metrics import benjamini_hochberg_fdr, compute_overlap_matrix
from npathway.utils.gmt_io import read_gmt, write_gmt

logger = logging.getLogger(__name__)


@dataclass
class BulkDynamicConfig:
    """Configuration for bulk dynamic pathway discovery."""

    matrix_path: str
    metadata_path: str
    group_col: str
    group_a: str
    group_b: str
    sample_col: str = "sample"
    matrix_orientation: str = "genes_by_samples"
    sep: str | None = None
    raw_counts: bool = True
    discovery_method: str = "kmeans"
    n_programs: int | None = 20
    k_neighbors: int = 15
    resolution: float = 1.0
    n_components: int = 30
    n_diffusion_steps: int = 3
    diffusion_alpha: float = 0.5
    de_test: str = "welch"
    de_alpha: float = 0.05
    min_abs_logfc_for_claim: float = 0.2
    gsea_n_perm: int = 1000
    min_genes_per_program_claim: int = 10
    n_bootstrap: int = 0
    min_stability_for_claim: float = 0.25
    random_seed: int = 42
    annotate_programs: bool = False
    annotation_collections: tuple[str, ...] = ("hallmark", "go_bp", "kegg")
    annotation_species: str = "human"
    annotation_gmt_path: str | None = None
    annotation_topk_per_program: int = 15
    annotation_min_jaccard_for_label: float = 0.03
    ranked_genes_path: str | None = None
    ranked_genes_sep: str | None = None
    ranked_gene_col: str = "gene"
    ranked_score_col: str = "score"


@dataclass
class BulkDynamicResult:
    """Result payload for dynamic pathway discovery."""

    output_dir: str
    n_samples: int
    n_genes: int
    n_programs: int
    n_sig_de_genes: int
    mean_program_size: float
    stability_mean_best_match_jaccard: float | None
    stability_ci_low: float | None
    stability_ci_high: float | None
    n_annotated_programs: int | None = None


def run_bulk_dynamic_pipeline(
    config: BulkDynamicConfig,
    output_dir: str | Path,
) -> BulkDynamicResult:
    """Run dynamic pathway discovery from matrix+metadata inputs."""
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    expression, metadata = _load_and_align_inputs(config)
    expression = _prepare_expression(expression, raw_counts=config.raw_counts)

    labels = metadata[config.group_col].astype(str).values
    group_mask = np.isin(labels, [config.group_a, config.group_b])
    expression = expression.loc[group_mask].copy()
    metadata = metadata.loc[group_mask].copy()
    labels = metadata[config.group_col].astype(str).values

    if np.sum(labels == config.group_a) < 2 or np.sum(labels == config.group_b) < 2:
        raise ValueError("Each group must have at least 2 samples for two-group analysis.")

    adata = _expression_to_adata(expression, metadata)
    embeddings, gene_names = build_graph_regularized_embeddings(
        adata=adata,
        n_components=config.n_components,
        k_neighbors=config.k_neighbors,
        n_diffusion_steps=config.n_diffusion_steps,
        alpha=config.diffusion_alpha,
        use_raw=False,
    )

    discovery = ClusteringProgramDiscovery(
        method=config.discovery_method,  # type: ignore[arg-type]
        n_programs=config.n_programs,
        k_neighbors=config.k_neighbors,
        resolution=config.resolution,
        random_state=config.random_seed,
    )
    discovery.fit(embeddings, gene_names)
    programs_raw = discovery.get_programs()
    program_scores_raw = discovery.get_program_scores()

    annotation_df = pd.DataFrame()
    overlap_long_df = pd.DataFrame()
    renaming_map = {p: p for p in programs_raw}
    if config.annotate_programs:
        annotation_df, overlap_long_df, renaming_map = _annotate_programs_with_references(
            programs=programs_raw,
            gene_universe=set(expression.columns.astype(str)),
            config=config,
        )
        annotation_df.to_csv(outdir / "program_annotation_matches.csv", index=False)
        overlap_long_df.to_csv(outdir / "program_reference_overlap_long.csv", index=False)
        pd.DataFrame(
            {
                "program_raw": list(renaming_map.keys()),
                "program": [renaming_map[k] for k in renaming_map],
            }
        ).to_csv(outdir / "program_renaming_map.csv", index=False)

    programs = {renaming_map[p]: genes for p, genes in programs_raw.items()}
    program_scores = {renaming_map[p]: scores for p, scores in program_scores_raw.items()}

    de_df = _compute_de_two_group(
        expression=expression,
        labels=labels,
        group_a=config.group_a,
        group_b=config.group_b,
        test=config.de_test,
    )
    de_df["fdr"] = benjamini_hochberg_fdr(de_df["p_value"].to_numpy(dtype=np.float64))
    de_df.to_csv(outdir / "de_results.csv", index=False)

    if config.ranked_genes_path is not None:
        ranked_genes = _load_ranked_genes(
            path=config.ranked_genes_path,
            sep=config.ranked_genes_sep,
            gene_col=config.ranked_gene_col,
            score_col=config.ranked_score_col,
            gene_universe=set(expression.columns.astype(str)),
        )
    else:
        ranked_genes = _build_ranked_genes(de_df)
    ranked_df = pd.DataFrame(ranked_genes, columns=["gene", "score"])
    ranked_df.to_csv(outdir / "ranked_genes_for_gsea.csv", index=False)

    sig_de = de_df[de_df["fdr"] <= config.de_alpha]["gene"].tolist()
    if not sig_de:
        fallback_n = min(200, len(de_df))
        sig_de = de_df.sort_values("p_value").head(fallback_n)["gene"].tolist()

    fisher_df = run_enrichment(
        gene_list=sig_de,
        gene_programs=programs,
        method="fisher",
        background=list(expression.columns),
    )
    fisher_df.to_csv(outdir / "enrichment_fisher.csv", index=False)

    gsea_df = run_enrichment(
        gene_list=[],
        gene_programs=programs,
        method="gsea",
        ranked_genes=ranked_genes,
        n_perm=config.gsea_n_perm,
        seed=config.random_seed,
    )
    gsea_df = _attach_claim_gates(
        gsea_df=gsea_df,
        fisher_df=fisher_df,
        programs=programs,
        de_df=de_df,
        min_abs_logfc_for_claim=config.min_abs_logfc_for_claim,
        min_genes_per_program_claim=config.min_genes_per_program_claim,
    )
    if gsea_df.empty:
        gsea_df = gsea_df.copy()
        gsea_df["gate_fdr"] = pd.Series(dtype=bool)
        gsea_df["gate_effect"] = pd.Series(dtype=bool)
        gsea_df["gate_program_size"] = pd.Series(dtype=bool)

    stability_df: pd.DataFrame | None = None
    stability_mean: float | None = None
    stability_lo: float | None = None
    stability_hi: float | None = None
    if config.n_bootstrap > 0:
        stability_df = _bootstrap_program_stability(
            expression=expression,
            labels=labels,
            reference_programs=programs,
            config=config,
        )
        stability_df.to_csv(outdir / "bootstrap_stability.csv", index=False)
        stability_mean = float(stability_df["best_match_jaccard"].mean())
        stability_lo = float(stability_df["best_match_jaccard"].quantile(0.025))
        stability_hi = float(stability_df["best_match_jaccard"].quantile(0.975))
        gsea_df["gate_stability"] = stability_mean >= config.min_stability_for_claim
    else:
        gsea_df["gate_stability"] = np.nan

    gate_cols = [
        "gate_fdr",
        "gate_effect",
        "gate_program_size",
        "gate_stability",
    ]
    claim_supported = np.ones(len(gsea_df), dtype=bool)
    for col in gate_cols:
        col_ok = gsea_df[col].eq(True) | gsea_df[col].isna()
        claim_supported = claim_supported & col_ok.to_numpy(dtype=bool)
    gsea_df["claim_supported"] = claim_supported
    gsea_df.to_csv(outdir / "enrichment_gsea_with_claim_gates.csv", index=False)

    context_scores = _build_context_membership_scores(
        program_scores=program_scores,
        de_df=de_df,
        group_a=config.group_a,
        group_b=config.group_b,
    )
    context_scores.to_csv(outdir / "contextual_membership_scores.csv", index=False)
    _write_gene_membership_tables(context_scores, outdir)

    _write_program_outputs(programs, outdir)
    _write_run_manifest(
        config=config,
        outdir=outdir,
        n_samples=expression.shape[0],
        n_genes=expression.shape[1],
        n_programs=len(programs),
    )
    _write_summary_markdown(
        outdir=outdir,
        config=config,
        n_samples=expression.shape[0],
        de_df=de_df,
        programs=programs,
        gsea_df=gsea_df,
        stability_mean=stability_mean,
        stability_ci=(stability_lo, stability_hi) if stability_lo is not None else None,
    )

    return BulkDynamicResult(
        output_dir=str(outdir),
        n_samples=int(expression.shape[0]),
        n_genes=int(expression.shape[1]),
        n_programs=int(len(programs)),
        n_sig_de_genes=int((de_df["fdr"] <= config.de_alpha).sum()),
        mean_program_size=(
            float(np.mean([len(v) for v in programs.values()])) if programs else 0.0
        ),
        stability_mean_best_match_jaccard=stability_mean,
        stability_ci_low=stability_lo,
        stability_ci_high=stability_hi,
        n_annotated_programs=(
            int((annotation_df["best_jaccard"] > 0).sum())
            if not annotation_df.empty and "best_jaccard" in annotation_df.columns
            else None
        ),
    )


def _load_table(path: str, sep: str | None = None) -> pd.DataFrame:
    """Load table with automatic delimiter inference when sep is None."""
    if sep is not None:
        return pd.read_csv(path, sep=sep)
    # sep=None + python engine infers common delimiters.
    return pd.read_csv(path, sep=None, engine="python")


def _load_and_align_inputs(config: BulkDynamicConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load matrix and metadata and align samples."""
    matrix_raw = _load_table(config.matrix_path, sep=config.sep)
    metadata = _load_table(config.metadata_path, sep=config.sep)

    if config.sample_col not in metadata.columns:
        raise KeyError(
            f"metadata is missing sample column '{config.sample_col}'. "
            f"Available columns: {list(metadata.columns)}"
        )
    if config.group_col not in metadata.columns:
        raise KeyError(
            f"metadata is missing group column '{config.group_col}'. "
            f"Available columns: {list(metadata.columns)}"
        )

    if matrix_raw.shape[1] < 3:
        raise ValueError("matrix must include id column + at least 2 sample/gene columns.")

    matrix = matrix_raw.set_index(matrix_raw.columns[0]).copy()
    matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    orientation = config.matrix_orientation.lower().strip()
    if orientation not in {"genes_by_samples", "samples_by_genes"}:
        raise ValueError("matrix_orientation must be 'genes_by_samples' or 'samples_by_genes'.")

    meta = metadata.copy()
    meta[config.sample_col] = meta[config.sample_col].astype(str)
    meta = meta.drop_duplicates(subset=[config.sample_col]).set_index(config.sample_col)

    if orientation == "genes_by_samples":
        samples_in_matrix = matrix.columns.astype(str)
        common = [s for s in meta.index if s in samples_in_matrix]
        if not common:
            raise ValueError("No overlapping sample IDs between matrix columns and metadata.")
        expr = matrix.loc[:, common].T
        expr.index = expr.index.astype(str)
        aligned_meta = meta.loc[common].copy()
    else:
        matrix.index = matrix.index.astype(str)
        common = [s for s in meta.index if s in matrix.index]
        if not common:
            raise ValueError("No overlapping sample IDs between matrix rows and metadata.")
        expr = matrix.loc[common].copy()
        aligned_meta = meta.loc[common].copy()

    return expr, aligned_meta


def _prepare_expression(expression: pd.DataFrame, raw_counts: bool) -> pd.DataFrame:
    """Normalize expression matrix for downstream dynamic pathway discovery."""
    expr = expression.astype(np.float64)
    if not raw_counts:
        return expr
    lib_size = expr.sum(axis=1).replace(0.0, np.nan)
    cpm = expr.div(lib_size, axis=0) * 1e6
    cpm = cpm.fillna(0.0)
    return np.log1p(cpm)


def _expression_to_adata(expression: pd.DataFrame, metadata: pd.DataFrame) -> ad.AnnData:
    """Build AnnData object from sample x gene expression."""
    obs = metadata.copy()
    obs.index = expression.index
    var = pd.DataFrame(index=expression.columns)
    return ad.AnnData(X=expression.to_numpy(dtype=np.float32), obs=obs, var=var)


def _compute_de_two_group(
    expression: pd.DataFrame,
    labels: np.ndarray,
    group_a: str,
    group_b: str,
    test: str = "welch",
) -> pd.DataFrame:
    """Compute per-gene two-group differential expression."""
    mask_a = labels == group_a
    mask_b = labels == group_b
    x_a = expression.loc[mask_a].to_numpy(dtype=np.float64)
    x_b = expression.loc[mask_b].to_numpy(dtype=np.float64)

    mean_a = np.mean(x_a, axis=0)
    mean_b = np.mean(x_b, axis=0)
    logfc = mean_a - mean_b

    test = test.lower().strip()
    if test == "welch":
        _, pvals = stats.ttest_ind(x_a, x_b, axis=0, equal_var=False, nan_policy="omit")
    elif test in {"mwu", "mannwhitney"}:
        pvals = np.array(
            [
                stats.mannwhitneyu(x_a[:, i], x_b[:, i], alternative="two-sided").pvalue
                for i in range(x_a.shape[1])
            ],
            dtype=np.float64,
        )
    else:
        raise ValueError("de_test must be one of: 'welch', 'mwu'.")

    pvals = np.nan_to_num(pvals, nan=1.0, posinf=1.0, neginf=1.0)
    pvals = np.clip(pvals, 1e-300, 1.0)
    de_df = pd.DataFrame(
        {
            "gene": expression.columns.astype(str),
            "mean_group_a": mean_a,
            "mean_group_b": mean_b,
            "logfc_a_minus_b": logfc,
            "p_value": pvals,
        }
    ).sort_values("p_value", ascending=True)
    return de_df.reset_index(drop=True)


def _build_ranked_genes(de_df: pd.DataFrame) -> list[tuple[str, float]]:
    """Build ranked gene list for preranked GSEA."""
    score = -np.log10(np.clip(de_df["p_value"].to_numpy(dtype=np.float64), 1e-300, 1.0))
    score = score * np.sign(de_df["logfc_a_minus_b"].to_numpy(dtype=np.float64))
    ranked = list(zip(de_df["gene"].astype(str).tolist(), score.tolist()))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def _load_ranked_genes(
    *,
    path: str,
    sep: str | None,
    gene_col: str,
    score_col: str,
    gene_universe: set[str],
) -> list[tuple[str, float]]:
    """Load an external ranked-gene table for preranked GSEA."""
    ranked_df = _load_table(path, sep=sep)
    if gene_col not in ranked_df.columns:
        raise KeyError(
            f"ranked gene table is missing gene column '{gene_col}'. "
            f"Available columns: {list(ranked_df.columns)}"
        )
    if score_col not in ranked_df.columns:
        raise KeyError(
            f"ranked gene table is missing score column '{score_col}'. "
            f"Available columns: {list(ranked_df.columns)}"
        )

    ranked_df = ranked_df[[gene_col, score_col]].copy()
    if ranked_df[gene_col].isna().any():
        raise ValueError("ranked gene table contains missing gene identifiers.")
    if ranked_df[score_col].isna().any():
        raise ValueError("ranked gene table contains missing score values.")

    ranked_df[gene_col] = ranked_df[gene_col].astype(str)
    ranked_df[score_col] = pd.to_numeric(ranked_df[score_col], errors="coerce")
    if ranked_df[score_col].isna().any():
        raise ValueError("ranked gene table contains non-numeric score values.")
    finite_mask = np.isfinite(ranked_df[score_col].to_numpy(dtype=np.float64))
    if not bool(np.all(finite_mask)):
        raise ValueError("ranked gene table contains non-finite score values.")

    duplicated = ranked_df[gene_col].duplicated(keep=False)
    if duplicated.any():
        n_dupes = int(duplicated.sum())
        logger.warning(
            "External ranked gene table contained %d duplicated gene rows; keeping the largest "
            "absolute score per gene.",
            n_dupes,
        )
        ranked_df = ranked_df.assign(_abs_score=ranked_df[score_col].abs())
        ranked_df = (
            ranked_df.sort_values("_abs_score", ascending=False)
            .drop_duplicates(subset=[gene_col], keep="first")
            .drop(columns="_abs_score")
        )

    ranked_df = ranked_df.loc[ranked_df[gene_col].isin(gene_universe)].copy()
    if ranked_df.empty:
        raise ValueError(
            "No genes from the external ranked-gene table overlapped the expression matrix."
        )

    ranked_df = ranked_df.sort_values(score_col, ascending=False).reset_index(drop=True)
    return list(
        zip(
            ranked_df[gene_col].astype(str).tolist(),
            ranked_df[score_col].to_numpy(dtype=np.float64).tolist(),
        )
    )


def _attach_claim_gates(
    gsea_df: pd.DataFrame,
    fisher_df: pd.DataFrame,
    programs: dict[str, list[str]],
    de_df: pd.DataFrame,
    min_abs_logfc_for_claim: float,
    min_genes_per_program_claim: int,
) -> pd.DataFrame:
    """Attach claim-control gate columns to enrichment results."""
    if gsea_df.empty:
        return gsea_df

    fisher_lookup = fisher_df.set_index("program")["fdr"].to_dict() if not fisher_df.empty else {}
    logfc_lookup = de_df.set_index("gene")["logfc_a_minus_b"].to_dict()

    gsea_df = gsea_df.copy()
    gsea_df["program_size"] = gsea_df["program"].map(lambda p: len(programs.get(str(p), [])))
    gsea_df["mean_abs_logfc_program"] = gsea_df["program"].map(
        lambda p: float(
            np.mean([abs(logfc_lookup.get(g, 0.0)) for g in programs.get(str(p), [])])
        )
        if programs.get(str(p), [])
        else 0.0
    )
    gsea_df["fisher_fdr"] = gsea_df["program"].map(lambda p: fisher_lookup.get(str(p), np.nan))

    gsea_df["gate_fdr"] = gsea_df["fdr"] <= 0.05
    gsea_df["gate_effect"] = gsea_df["mean_abs_logfc_program"] >= min_abs_logfc_for_claim
    gsea_df["gate_program_size"] = gsea_df["program_size"] >= min_genes_per_program_claim
    return gsea_df


def _build_context_membership_scores(
    program_scores: dict[str, list[tuple[str, float]]],
    de_df: pd.DataFrame,
    group_a: str,
    group_b: str,
) -> pd.DataFrame:
    """Convert static membership to probabilistic/context-aware membership."""
    output_columns = [
        "program",
        "gene",
        "base_membership",
        f"prob_{group_a}",
        f"prob_{group_b}",
        f"contextual_score_{group_a}",
        f"contextual_score_{group_b}",
        "p_value",
        "neglog10_p_value",
        "context_shift",
        "signed_significance",
        "context_evidence",
    ]
    logfc_lookup = de_df.set_index("gene")["logfc_a_minus_b"].to_dict()
    pval_lookup = de_df.set_index("gene")["p_value"].to_dict()
    rows: list[dict[str, float | str]] = []

    for program, scored_genes in program_scores.items():
        for gene, base_score in scored_genes:
            lf = float(logfc_lookup.get(gene, 0.0))
            pval = float(pval_lookup.get(gene, 1.0))
            neglog10p = float(-np.log10(np.clip(pval, 1e-300, 1.0)))
            # Equivalent to sigmoid(logFC) but numerically stable for large effects.
            context_shift = float(np.tanh(lf / 2.0))
            p_a = 0.5 * (1.0 + context_shift)
            p_b = 1.0 - p_a
            base = float(base_score)
            if not np.isfinite(base):
                base = 0.0
            rows.append(
                {
                    "program": program,
                    "gene": gene,
                    "base_membership": base,
                    f"prob_{group_a}": p_a,
                    f"prob_{group_b}": p_b,
                    f"contextual_score_{group_a}": base * p_a,
                    f"contextual_score_{group_b}": base * p_b,
                    "p_value": pval,
                    "neglog10_p_value": neglog10p,
                    "context_shift": context_shift,
                    "signed_significance": float(np.sign(lf)) * neglog10p,
                    "context_evidence": context_shift * neglog10p,
                }
            )

    out = pd.DataFrame(rows, columns=output_columns)
    return out.sort_values(["program", "base_membership"], ascending=[True, False]).reset_index(
        drop=True
    )


def _write_program_outputs(programs: dict[str, list[str]], outdir: Path) -> None:
    """Persist gene program outputs."""
    write_gmt(programs, str(outdir / "dynamic_programs.gmt"))
    rows = []
    for program, genes in programs.items():
        for gene in genes:
            rows.append({"program": program, "gene": gene})
    pd.DataFrame(rows).to_csv(outdir / "dynamic_programs_long.csv", index=False)
    pd.DataFrame(
        {
            "program": list(programs.keys()),
            "n_genes": [len(v) for v in programs.values()],
        }
    ).to_csv(outdir / "dynamic_program_sizes.csv", index=False)


def _write_gene_membership_tables(context_scores: pd.DataFrame, outdir: Path) -> None:
    """Write researcher-facing program gene membership tables."""
    ranked = context_scores.sort_values(
        ["program", "base_membership"], ascending=[True, False]
    ).reset_index(drop=True)
    ranked.to_csv(outdir / "program_gene_membership_ranked.csv", index=False)

    top_by_program = (
        ranked.groupby("program", as_index=False, sort=False)
        .head(20)
        .reset_index(drop=True)
    )
    top_by_program.to_csv(outdir / "program_gene_membership_top20.csv", index=False)

    compact = (
        ranked.groupby("program", sort=False)["gene"]
        .apply(lambda s: ",".join(s.astype(str).tolist()))
        .reset_index()
        .rename(columns={"gene": "genes"})
    )
    compact.to_csv(outdir / "program_gene_lists.csv", index=False)


def _annotate_programs_with_references(
    programs: dict[str, list[str]],
    gene_universe: set[str],
    config: BulkDynamicConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Match discovered programs to reference pathway collections."""
    ref_sets = _load_reference_gene_sets(config=config, gene_universe=gene_universe)
    empty_best = pd.DataFrame(
        columns=[
            "program_raw",
            "program",
            "best_reference_name",
            "best_jaccard",
            "best_overlap_n",
            "program_n",
            "reference_n",
            "best_overlap_genes",
        ]
    )
    empty_overlap = pd.DataFrame(
        columns=[
            "program_raw",
            "program",
            "reference_name",
            "jaccard",
            "overlap_n",
            "program_n",
            "reference_n",
        ]
    )
    if not ref_sets:
        logger.warning("Program annotation requested but no reference sets were available.")
        return empty_best, empty_overlap, {p: p for p in programs}

    best_rows: list[dict[str, object]] = []
    overlap_rows: list[dict[str, object]] = []
    renaming_map: dict[str, str] = {}
    used_labels: set[str] = set()

    for program, genes in programs.items():
        pset = set(genes)
        scored: list[dict[str, object]] = []
        for ref_name, ref_genes in ref_sets.items():
            rset = set(ref_genes)
            union = pset | rset
            inter = pset & rset
            jacc = float(len(inter) / len(union)) if union else 0.0
            scored.append(
                {
                    "program_raw": program,
                    "reference_name": ref_name,
                    "jaccard": jacc,
                    "overlap_n": len(inter),
                    "program_n": len(pset),
                    "reference_n": len(rset),
                    "overlap_genes": ",".join(sorted(inter)) if inter else "",
                }
            )

        scored.sort(
            key=lambda x: (
                float(x["jaccard"]),
                int(x["overlap_n"]),
                -int(x["reference_n"]),
            ),
            reverse=True,
        )
        top_hits = scored[: max(1, config.annotation_topk_per_program)]
        best = top_hits[0]

        best_name = str(best["reference_name"])
        best_j = float(best["jaccard"])
        if best_j >= config.annotation_min_jaccard_for_label:
            stem = _sanitize_program_label(best_name)
        else:
            stem = "Unmatched"
        candidate = f"{program}__{stem}"
        mapped = _dedupe_name(candidate, used_labels)
        used_labels.add(mapped)
        renaming_map[program] = mapped

        best_rows.append(
            {
                "program_raw": program,
                "program": mapped,
                "best_reference_name": best_name,
                "best_jaccard": best_j,
                "best_overlap_n": int(best["overlap_n"]),
                "program_n": int(best["program_n"]),
                "reference_n": int(best["reference_n"]),
                "best_overlap_genes": str(best["overlap_genes"]),
            }
        )
        for row in top_hits:
            overlap_rows.append(
                {
                    "program_raw": program,
                    "program": mapped,
                    "reference_name": str(row["reference_name"]),
                    "jaccard": float(row["jaccard"]),
                    "overlap_n": int(row["overlap_n"]),
                    "program_n": int(row["program_n"]),
                    "reference_n": int(row["reference_n"]),
                }
            )

    return pd.DataFrame(best_rows), pd.DataFrame(overlap_rows), renaming_map


def _load_reference_gene_sets(
    config: BulkDynamicConfig,
    gene_universe: set[str],
) -> dict[str, list[str]]:
    """Load and filter reference gene sets for annotation."""
    refs: dict[str, list[str]] = {}

    if config.annotation_gmt_path:
        try:
            custom = read_gmt(config.annotation_gmt_path)
            for name, genes in custom.items():
                kept = [g for g in genes if g in gene_universe]
                if len(kept) >= 3:
                    refs[f"CUSTOM::{name}"] = kept
        except Exception as exc:
            logger.warning("Failed to read custom GMT '%s': %s", config.annotation_gmt_path, exc)

    for collection in config.annotation_collections:
        try:
            gs = load_msigdb_gene_sets(
                collection=collection,
                species=config.annotation_species,
            )
            for name, genes in gs.items():
                kept = [g for g in genes if g in gene_universe]
                if len(kept) >= 3:
                    refs[f"{collection.upper()}::{name}"] = kept
        except Exception as exc:
            logger.warning("MSigDB annotation load failed for %s: %s", collection, exc)

    return refs


def _sanitize_program_label(reference_name: str, max_len: int = 60) -> str:
    """Convert reference names into compact safe labels."""
    raw = reference_name.split("::", 1)[-1]
    safe_chars = []
    for ch in raw:
        if ch.isalnum():
            safe_chars.append(ch)
        elif ch in {" ", "-", ".", "/", ":", "_"}:
            safe_chars.append("_")
    safe = "".join(safe_chars)
    while "__" in safe:
        safe = safe.replace("__", "_")
    safe = safe.strip("_")
    # Keep canonical collection prefixes readable for downstream dashboards.
    if safe.startswith("GOBP") and not safe.startswith("GOBP_"):
        safe = safe.replace("GOBP", "GOBP_", 1)
    if safe.startswith("KEGG") and not safe.startswith("KEGG_"):
        safe = safe.replace("KEGG", "KEGG_", 1)
    if safe.startswith("HALLMARK") and not safe.startswith("HALLMARK_"):
        safe = safe.replace("HALLMARK", "HALLMARK_", 1)
    if not safe:
        safe = "Annotated"
    return safe[:max_len]


def _dedupe_name(candidate: str, used: set[str]) -> str:
    """Ensure unique names by appending index when needed."""
    if candidate not in used:
        return candidate
    idx = 2
    while f"{candidate}_{idx}" in used:
        idx += 1
    return f"{candidate}_{idx}"


def _bootstrap_program_stability(
    expression: pd.DataFrame,
    labels: np.ndarray,
    reference_programs: dict[str, list[str]],
    config: BulkDynamicConfig,
) -> pd.DataFrame:
    """Estimate discovery stability by group-wise bootstrap resampling."""
    rng = np.random.default_rng(config.random_seed)
    idx_a = np.where(labels == config.group_a)[0]
    idx_b = np.where(labels == config.group_b)[0]
    rows: list[dict[str, float | int]] = []

    for b in range(config.n_bootstrap):
        sample_a = rng.choice(idx_a, size=len(idx_a), replace=True)
        sample_b = rng.choice(idx_b, size=len(idx_b), replace=True)
        sample_idx = np.concatenate([sample_a, sample_b])

        expr_boot = expression.iloc[sample_idx].copy()
        meta_boot = pd.DataFrame(
            {
                config.group_col: [config.group_a] * len(sample_a) + [config.group_b] * len(sample_b)
            },
            index=expr_boot.index,
        )

        adata_boot = _expression_to_adata(expr_boot, meta_boot)
        emb_boot, gn_boot = build_graph_regularized_embeddings(
            adata=adata_boot,
            n_components=config.n_components,
            k_neighbors=config.k_neighbors,
            n_diffusion_steps=config.n_diffusion_steps,
            alpha=config.diffusion_alpha,
            use_raw=False,
        )
        disc_boot = ClusteringProgramDiscovery(
            method=config.discovery_method,  # type: ignore[arg-type]
            n_programs=config.n_programs,
            k_neighbors=config.k_neighbors,
            resolution=config.resolution,
            random_state=config.random_seed + b + 1,
        )
        disc_boot.fit(emb_boot, gn_boot)
        boot_programs = disc_boot.get_programs()
        overlap = compute_overlap_matrix(reference_programs, boot_programs)
        score = float(overlap.max(axis=1).mean()) if not overlap.empty else 0.0
        rows.append({"bootstrap_id": b, "best_match_jaccard": score})

    return pd.DataFrame(rows)


def _write_run_manifest(
    config: BulkDynamicConfig,
    outdir: Path,
    n_samples: int,
    n_genes: int,
    n_programs: int,
) -> None:
    """Write JSON manifest for reproducibility."""
    payload = {
        "config": asdict(config),
        "n_samples_used": n_samples,
        "n_genes_used": n_genes,
        "n_programs": n_programs,
    }
    (outdir / "run_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_summary_markdown(
    outdir: Path,
    config: BulkDynamicConfig,
    n_samples: int,
    de_df: pd.DataFrame,
    programs: dict[str, list[str]],
    gsea_df: pd.DataFrame,
    stability_mean: float | None,
    stability_ci: tuple[float | None, float | None] | None,
) -> None:
    """Write concise markdown summary with claim-control outcomes."""
    n_supported = int(gsea_df["claim_supported"].sum()) if "claim_supported" in gsea_df.columns else 0
    n_total = int(len(gsea_df))
    n_sig_de = int((de_df["fdr"] <= config.de_alpha).sum())
    top_programs = (
        gsea_df.sort_values("fdr").head(10)[["program", "nes", "p_value", "fdr", "claim_supported"]]
        if not gsea_df.empty
        else pd.DataFrame()
    )

    lines = [
        "# Bulk Dynamic Pathway Summary",
        "",
        "## Run Info",
        f"- Contrast: {config.group_a} vs {config.group_b}",
        f"- Samples used: {n_samples}",
        f"- Genes used: {len(de_df)}",
        f"- Programs discovered: {len(programs)}",
        f"- Significant DE genes (FDR <= {config.de_alpha}): {n_sig_de}",
        "",
        "## Claim-Control Summary",
        f"- Claim-supported enriched programs: {n_supported}/{n_total}",
    ]

    if stability_mean is not None and stability_ci is not None:
        lines.append(
            f"- Bootstrap stability (mean best-match Jaccard): {stability_mean:.3f} "
            f"(95% CI {stability_ci[0]:.3f}-{stability_ci[1]:.3f})"
        )

    lines.extend(["", "## Top Enriched Programs (by GSEA FDR)"])
    if top_programs.empty:
        lines.append("- No enriched programs found.")
    else:
        for _, row in top_programs.iterrows():
            lines.append(
                f"- {row['program']}: NES={row['nes']:.3f}, p={row['p_value']:.3g}, "
                f"FDR={row['fdr']:.3g}, claim_supported={bool(row['claim_supported'])}"
            )

    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
