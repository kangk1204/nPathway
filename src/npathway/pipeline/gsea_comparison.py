"""Compare curated pathway GSEA against dynamic program GSEA on the same ranking."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from npathway.evaluation.enrichment import run_enrichment
from npathway.evaluation.metrics import compute_overlap_matrix
from npathway.utils.gmt_io import read_gmt


@dataclass
class GSEAComparisonResult:
    """Summary of a same-ranking curated-vs-dynamic comparison."""

    output_dir: str
    n_ranked_genes: int
    n_dynamic_programs: int
    n_curated_sets: int
    anchor_program: str | None
    anchor_reference: str | None
    anchor_jaccard: float | None
    n_focus_genes: int
    n_focus_genes_in_anchor_program: int
    n_focus_genes_in_curated_sets: int


def load_ranked_gene_table(
    *,
    path: str | Path,
    gene_col: str = "gene",
    score_col: str = "score",
    sep: str | None = None,
) -> list[tuple[str, float]]:
    """Load a ranked-gene table from CSV/TSV."""
    ranked_df = pd.read_csv(path, sep=sep if sep is not None else None, engine="python")
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
    if not np.isfinite(ranked_df[score_col].to_numpy(dtype=np.float64)).all():
        raise ValueError("ranked gene table contains non-finite score values.")

    ranked_df = ranked_df.assign(_abs_score=ranked_df[score_col].abs())
    ranked_df = (
        ranked_df.sort_values(["_abs_score", score_col], ascending=[False, False])
        .drop_duplicates(subset=[gene_col], keep="first")
        .drop(columns="_abs_score")
        .sort_values(score_col, ascending=False)
        .reset_index(drop=True)
    )
    return list(
        zip(
            ranked_df[gene_col].astype(str).tolist(),
            ranked_df[score_col].to_numpy(dtype=np.float64).tolist(),
        )
    )


def compare_curated_vs_dynamic_gsea(
    *,
    ranked_genes_path: str | Path,
    dynamic_gmt_path: str | Path,
    curated_gmt_path: str | Path,
    output_dir: str | Path,
    gene_col: str = "gene",
    score_col: str = "score",
    sep: str | None = None,
    n_perm: int = 1000,
    seed: int = 42,
    focus_genes: list[str] | None = None,
) -> GSEAComparisonResult:
    """Run same-ranked-list GSEA against curated and dynamic gene sets."""
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    ranked_genes = load_ranked_gene_table(
        path=ranked_genes_path,
        gene_col=gene_col,
        score_col=score_col,
        sep=sep,
    )
    gene_universe = {gene for gene, _ in ranked_genes}
    dynamic_programs = _filter_gene_sets_to_universe(
        read_gmt(str(dynamic_gmt_path)),
        gene_universe=gene_universe,
    )
    curated_sets = _filter_gene_sets_to_universe(
        read_gmt(str(curated_gmt_path)),
        gene_universe=gene_universe,
    )
    if not dynamic_programs:
        raise ValueError(f"No dynamic programs were found in {dynamic_gmt_path}.")
    if not curated_sets:
        raise ValueError(f"No curated gene sets were found in {curated_gmt_path}.")

    dynamic_df = run_enrichment(
        gene_list=[],
        gene_programs=dynamic_programs,
        method="gsea",
        ranked_genes=ranked_genes,
        n_perm=n_perm,
        seed=seed,
    )
    curated_df = run_enrichment(
        gene_list=[],
        gene_programs=curated_sets,
        method="gsea",
        ranked_genes=ranked_genes,
        n_perm=n_perm,
        seed=seed,
    )
    dynamic_df.to_csv(outdir / "dynamic_gsea.csv", index=False)
    curated_df.to_csv(outdir / "curated_gsea.csv", index=False)
    combined_df = pd.concat(
        [
            dynamic_df.assign(source="dynamic"),
            curated_df.assign(source="curated"),
        ],
        ignore_index=True,
        sort=False,
    )
    combined_df.to_csv(outdir / "gsea_comparison_combined.csv", index=False)

    overlap_df = _build_overlap_long(dynamic_programs, curated_sets)
    overlap_df.to_csv(outdir / "dynamic_curated_overlap.csv", index=False)

    anchor_program, anchor_reference, anchor_jaccard = _select_anchor_program(overlap_df)

    focus_gene_df = _build_focus_gene_table(
        focus_genes=focus_genes or [],
        dynamic_programs=dynamic_programs,
        curated_sets=curated_sets,
        anchor_program=anchor_program,
    )
    focus_gene_df.to_csv(outdir / "focus_gene_membership.csv", index=False)

    side_by_side_df = _build_side_by_side(dynamic_df, curated_df, anchor_program, anchor_reference)
    side_by_side_df.to_csv(outdir / "gsea_side_by_side.csv", index=False)

    result = GSEAComparisonResult(
        output_dir=str(outdir),
        n_ranked_genes=len(ranked_genes),
        n_dynamic_programs=len(dynamic_programs),
        n_curated_sets=len(curated_sets),
        anchor_program=anchor_program,
        anchor_reference=anchor_reference,
        anchor_jaccard=anchor_jaccard,
        n_focus_genes=len(focus_genes or []),
        n_focus_genes_in_anchor_program=int(focus_gene_df["in_anchor_program"].sum())
        if not focus_gene_df.empty
        else 0,
        n_focus_genes_in_curated_sets=int(focus_gene_df["in_any_curated_set"].sum())
        if not focus_gene_df.empty
        else 0,
    )
    (outdir / "comparison_summary.json").write_text(
        json.dumps(asdict(result), indent=2), encoding="utf-8"
    )
    _write_summary_markdown(
        result=result,
        dynamic_df=dynamic_df,
        curated_df=curated_df,
        overlap_df=overlap_df,
        focus_gene_df=focus_gene_df,
        outdir=outdir,
    )
    return result


def _filter_gene_sets_to_universe(
    gene_sets: dict[str, list[str]],
    *,
    gene_universe: set[str],
    min_genes: int = 3,
) -> dict[str, list[str]]:
    """Restrict gene sets to the ranked-gene universe before comparison."""
    filtered: dict[str, list[str]] = {}
    for name, genes in gene_sets.items():
        kept = [gene for gene in genes if gene in gene_universe]
        if len(kept) >= min_genes:
            filtered[str(name)] = kept
    return filtered


def _build_overlap_long(
    dynamic_programs: dict[str, list[str]],
    curated_sets: dict[str, list[str]],
) -> pd.DataFrame:
    overlap = compute_overlap_matrix(dynamic_programs, curated_sets)
    if overlap.empty:
        return pd.DataFrame(
            columns=[
                "dynamic_program",
                "curated_set",
                "jaccard",
                "overlap_n",
                "dynamic_only_genes",
                "curated_only_genes",
            ]
        )

    rows: list[dict[str, object]] = []
    for dynamic_program in overlap.index:
        dynamic_genes = set(dynamic_programs[dynamic_program])
        for curated_set in overlap.columns:
            curated_genes = set(curated_sets[curated_set])
            shared = sorted(dynamic_genes & curated_genes)
            dynamic_only = sorted(dynamic_genes - curated_genes)
            curated_only = sorted(curated_genes - dynamic_genes)
            rows.append(
                {
                    "dynamic_program": dynamic_program,
                    "curated_set": curated_set,
                    "jaccard": float(overlap.loc[dynamic_program, curated_set]),
                    "overlap_n": len(shared),
                    "shared_genes": ",".join(shared),
                    "dynamic_only_genes": ",".join(dynamic_only),
                    "curated_only_genes": ",".join(curated_only),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["jaccard", "overlap_n", "dynamic_program", "curated_set"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)


def _select_anchor_program(overlap_df: pd.DataFrame) -> tuple[str | None, str | None, float | None]:
    if overlap_df.empty:
        return None, None, None
    top = overlap_df.iloc[0]
    return str(top["dynamic_program"]), str(top["curated_set"]), float(top["jaccard"])


def _build_focus_gene_table(
    *,
    focus_genes: list[str],
    dynamic_programs: dict[str, list[str]],
    curated_sets: dict[str, list[str]],
    anchor_program: str | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    anchor_genes = set(dynamic_programs.get(anchor_program, [])) if anchor_program else set()
    for gene in [str(g).strip() for g in focus_genes if str(g).strip()]:
        dynamic_hits = sorted([name for name, genes in dynamic_programs.items() if gene in set(genes)])
        curated_hits = sorted([name for name, genes in curated_sets.items() if gene in set(genes)])
        rows.append(
            {
                "gene": gene,
                "in_any_dynamic_program": bool(dynamic_hits),
                "dynamic_programs": ",".join(dynamic_hits),
                "in_anchor_program": gene in anchor_genes,
                "in_any_curated_set": bool(curated_hits),
                "curated_sets": ",".join(curated_hits),
            }
        )
    return pd.DataFrame(rows)


def _build_side_by_side(
    dynamic_df: pd.DataFrame,
    curated_df: pd.DataFrame,
    anchor_program: str | None,
    anchor_reference: str | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if anchor_program is not None and not dynamic_df.empty:
        match = dynamic_df.loc[dynamic_df["program"].astype(str) == anchor_program]
        if not match.empty:
            row = match.iloc[0]
            rows.append(
                {
                    "collection": "dynamic",
                    "name": anchor_program,
                    "nes": float(row["nes"]),
                    "fdr": float(row["fdr"]),
                    "leading_edge_genes": str(row.get("leading_edge_genes", "")),
                }
            )
    if anchor_reference is not None and not curated_df.empty:
        match = curated_df.loc[curated_df["program"].astype(str) == anchor_reference]
        if not match.empty:
            row = match.iloc[0]
            rows.append(
                {
                    "collection": "curated",
                    "name": anchor_reference,
                    "nes": float(row["nes"]),
                    "fdr": float(row["fdr"]),
                    "leading_edge_genes": str(row.get("leading_edge_genes", "")),
                }
            )
    return pd.DataFrame(rows)


def _write_summary_markdown(
    *,
    result: GSEAComparisonResult,
    dynamic_df: pd.DataFrame,
    curated_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    focus_gene_df: pd.DataFrame,
    outdir: Path,
) -> None:
    dynamic_top = dynamic_df.head(5)
    curated_top = curated_df.head(5)
    overlap_top = overlap_df.head(5)

    lines = [
        "# Curated vs Dynamic GSEA Comparison",
        "",
        f"- ranked_genes: {result.n_ranked_genes}",
        f"- dynamic_programs: {result.n_dynamic_programs}",
        f"- curated_sets: {result.n_curated_sets}",
        f"- anchor_program: {result.anchor_program or 'NA'}",
        f"- anchor_reference: {result.anchor_reference or 'NA'}",
        f"- anchor_jaccard: {result.anchor_jaccard if result.anchor_jaccard is not None else 'NA'}",
        "",
        "## Top Dynamic Hits",
        "",
    ]
    if dynamic_top.empty:
        lines.append("No dynamic GSEA hits were produced.")
    else:
        for _, row in dynamic_top.iterrows():
            lines.append(
                f"- {row['program']}: NES={float(row['nes']):.3f}, FDR={float(row['fdr']):.4f}"
            )

    lines.extend(["", "## Top Curated Hits", ""])
    if curated_top.empty:
        lines.append("No curated GSEA hits were produced.")
    else:
        for _, row in curated_top.iterrows():
            lines.append(
                f"- {row['program']}: NES={float(row['nes']):.3f}, FDR={float(row['fdr']):.4f}"
            )

    lines.extend(["", "## Top Dynamic-Curated Overlaps", ""])
    if overlap_top.empty:
        lines.append("No dynamic-curated overlaps were available.")
    else:
        for _, row in overlap_top.iterrows():
            lines.append(
                f"- {row['dynamic_program']} vs {row['curated_set']}: "
                f"Jaccard={float(row['jaccard']):.3f}, overlap_n={int(row['overlap_n'])}"
            )

    if not focus_gene_df.empty:
        lines.extend(["", "## Focus Genes", ""])
        for _, row in focus_gene_df.iterrows():
            lines.append(
                f"- {row['gene']}: anchor={bool(row['in_anchor_program'])}, "
                f"curated={bool(row['in_any_curated_set'])}, "
                f"dynamic_programs={row['dynamic_programs'] or 'NA'}, "
                f"curated_sets={row['curated_sets'] or 'NA'}"
            )

    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
