"""Multi-pathway annotation for data-driven gene programs.

This is nPathway's core differentiator: discovered gene programs are
annotated against *multiple* curated pathways simultaneously, revealing
cross-pathway biology that static GSEA cannot capture.

Traditional GSEA tests each curated pathway independently.  A disease
module that spans TP53 signalling, metabolism, and immune regulation
would appear as three marginally significant pathways.  nPathway
discovers the unified module first, then annotates it:

    Program_1 → TP53_SIGNALING (12 genes, p=1e-5)
              + METABOLISM (8 genes, p=1e-3)
              + IMMUNE_RESPONSE (6 genes, p=1e-4)
              + novel: TREM2, BIN1, PICALM (no curated pathway)

Key functions
-------------
- :func:`annotate_program` -- Annotate a single program against reference DB.
- :func:`annotate_all_programs` -- Annotate all discovered programs.
- :func:`multi_pathway_score` -- Composite score capturing multi-pathway span.
- :func:`pathway_annotation_report` -- Human-readable annotation report.
- :func:`compare_with_gsea` -- Head-to-head comparison vs standard GSEA.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

_DISEASE_ANCHOR_KEYWORDS: tuple[str, ...] = (
    "ALZHEIMER",
    "AMYLOID",
    "TAU",
    "DEPRESS",
    "GLUCOCORTICOID",
    "NEUROINFLAMMATORY",
)

_HIGH_PRIORITY_KEYWORDS: tuple[str, ...] = (
    "MICROGL",
    "COMPLEMENT",
    "INFLAM",
    "INTERFERON",
    "CYTOKINE",
    "IMMUNE",
    "DEFENSE RESPONSE",
    "PHAGOCYT",
    "ENDOLYSOSOM",
    "LYSOSOM",
    "SYNAPT",
    "NEURON",
    "NEUROGEN",
    "NEUROTRANSMITTER",
    "ASTROCY",
    "OLIGOD",
    "GLIAL",
    "AXON",
    "DENDRITE",
    "MYELIN",
    "STRESS",
    "TNFA",
    "NFKB",
    "JAK",
    "CHEMOKINE",
)

_MEDIUM_PRIORITY_KEYWORDS: tuple[str, ...] = (
    "OXIDATIVE PHOSPHORYLATION",
    "MITOCHONDRIAL ELECTRON TRANSPORT",
    "CELL CELL SIGNALING",
    "TRANSPORT",
    "LIPID",
    "CHOLESTEROL",
)

_LOW_SIGNAL_KEYWORDS: tuple[str, ...] = (
    "FASTK FAMILY",
    "TRNA PROCESSING",
    "RRNA PROCESSING",
    "MRNA MODIFICATION",
    "RNA DEGRADATION",
    "TRANSLATION",
    "PEPTIDE CHAIN",
    "RIBOSOME",
    "SPLICING",
    "HTTP BIOREGISTRY",
    "PATHBANK",
    "HUMANCYC",
    "NON STOP MRNA",
    "VIRAL MRNA",
)


def _normalize_reference_label(reference_name: str) -> str:
    """Normalize source-qualified pathway labels for keyword matching."""
    raw = str(reference_name).split("::", 1)[-1]
    raw = re.sub(
        r"^(GOBP|GOCC|GOMF|WP|KEGG|REACTOME|HALLMARK|CUSTOM)_+",
        "",
        raw,
        flags=re.IGNORECASE,
    )
    return raw.replace("_", " ").upper()


def reference_relevance_score(reference_name: str) -> float:
    """Heuristic score prioritizing disease-relevant over housekeeping labels.

    The score is intentionally conservative. It boosts pathway labels that are
    directly interpretable in neuro/immune disease settings and demotes labels
    that are primarily housekeeping or opaque database identifiers.
    """
    label = _normalize_reference_label(reference_name)
    disease_anchor_hits = sum(keyword in label for keyword in _DISEASE_ANCHOR_KEYWORDS)
    high_hits = sum(keyword in label for keyword in _HIGH_PRIORITY_KEYWORDS)
    medium_hits = sum(keyword in label for keyword in _MEDIUM_PRIORITY_KEYWORDS)
    low_hits = sum(keyword in label for keyword in _LOW_SIGNAL_KEYWORDS)

    score = 0.0
    score += min(10.0, disease_anchor_hits * 6.0)
    score += min(12.0, high_hits * 3.0)
    score += min(6.0, medium_hits * 2.0)
    if high_hits == 0:
        score -= min(8.0, low_hits * 2.0)
    else:
        score -= min(2.0, low_hits * 0.5)

    if not any(ch.isalpha() for ch in label):
        score -= 4.0
    return float(score)


def reference_relevance_band(reference_name: str) -> str:
    """Map relevance score into a compact priority label."""
    score = reference_relevance_score(reference_name)
    if score >= 6.0:
        return "Disease-prioritized"
    if score >= 2.0:
        return "Supportive"
    return "Background"


def family_interpretation_score(
    reference_name: str,
    best_jaccard: float,
    programs_covered: int = 1,
    source_count: int = 1,
) -> float:
    """Composite interpretation score for collapsed reference families.

    This is an interpretation-layer ranking heuristic, not a new significance
    model. It combines disease relevance, overlap strength, how many programs
    the family explains, and how many independent sources support it.
    """
    priority = reference_relevance_score(reference_name)
    jaccard = max(0.0, float(best_jaccard))
    programs = max(1.0, min(float(programs_covered), 6.0))
    sources = max(1.0, min(float(source_count), 4.0))
    score = (2.0 * priority) + (15.0 * jaccard) + (0.8 * programs) + (0.6 * sources)
    return float(score)


def source_interpretation_score(
    best_priority_score: float,
    best_jaccard: float,
    programs_covered: int = 1,
    references_covered: int = 1,
) -> float:
    """Composite interpretation score for source-level summaries."""
    priority = float(best_priority_score)
    jaccard = max(0.0, float(best_jaccard))
    programs = max(1.0, min(float(programs_covered), 8.0))
    references = max(1.0, min(float(references_covered), 12.0))
    score = (2.0 * priority) + (14.0 * jaccard) + (0.8 * programs) + (0.15 * references)
    return float(score)


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------


@dataclass
class PathwayHit:
    """A single curated pathway that overlaps with a discovered program.

    Attributes
    ----------
    pathway_name : str
        Name of the curated pathway (e.g., ``"HALLMARK_P53_PATHWAY"``).
    overlap_genes : list[str]
        Genes shared between the program and the pathway.
    overlap_size : int
        Number of overlapping genes.
    pathway_size : int
        Total genes in the curated pathway.
    p_value : float
        Fisher's exact test p-value (one-sided, greater).
    fdr : float
        BH-adjusted q-value (set after correction across all hits).
    fold_enrichment : float
        Observed / expected overlap ratio.
    fraction_of_program : float
        Fraction of program genes that belong to this pathway.
    """

    pathway_name: str
    overlap_genes: list[str]
    overlap_size: int
    pathway_size: int
    p_value: float
    fdr: float = 1.0
    fold_enrichment: float = 0.0
    fraction_of_program: float = 0.0


@dataclass
class ProgramAnnotation:
    """Multi-pathway annotation for a single discovered program.

    Attributes
    ----------
    program_name : str
        Name of the discovered program.
    program_genes : list[str]
        All genes in the program.
    significant_pathways : list[PathwayHit]
        Curated pathways with FDR-significant overlap (sorted by p-value).
    novel_genes : list[str]
        Genes in the program that do not appear in *any* significant pathway.
    novelty_fraction : float
        Fraction of program genes not covered by any significant pathway.
    pathway_span : int
        Number of distinct significant pathways this program spans.
    multi_pathway_score : float
        Composite score: pathway_span * (1 - novelty_fraction) + novelty_bonus.
    """

    program_name: str
    program_genes: list[str]
    significant_pathways: list[PathwayHit] = field(default_factory=list)
    novel_genes: list[str] = field(default_factory=list)
    novelty_fraction: float = 0.0
    pathway_span: int = 0
    multi_pathway_score: float = 0.0


# ------------------------------------------------------------------
# Core annotation engine
# ------------------------------------------------------------------


def annotate_program(
    program_genes: list[str],
    curated_pathways: dict[str, list[str]],
    background: list[str] | None = None,
    fdr_threshold: float = 0.25,
) -> ProgramAnnotation:
    """Annotate a single program against a curated pathway database.

    For each curated pathway, computes Fisher's exact test for overlap.
    FDR correction is applied across all pathways.  Returns significant
    hits sorted by p-value, plus a list of novel genes not in any hit.

    Parameters
    ----------
    program_genes : list[str]
        Genes in the discovered program.
    curated_pathways : dict[str, list[str]]
        Reference pathway database (e.g., MSigDB Hallmark).
    background : list[str] | None
        Background gene universe.  If ``None``, uses the union of
        program genes and all pathway genes.
    fdr_threshold : float
        FDR threshold for significance.

    Returns
    -------
    ProgramAnnotation
        Full annotation with significant pathways and novel genes.
    """
    prog_set = set(program_genes)

    # Background
    if background is None:
        bg_set: set[str] = set(program_genes)
        for genes in curated_pathways.values():
            bg_set.update(genes)
    else:
        bg_set = set(background)
        prog_set = prog_set & bg_set

    bg_size = len(bg_set)
    prog_size = len(prog_set)

    # Fisher's test for each pathway
    hits: list[PathwayHit] = []
    for pw_name, pw_genes in curated_pathways.items():
        pw_set = set(pw_genes) & bg_set
        pw_size = len(pw_set)
        if pw_size == 0:
            continue

        overlap = prog_set & pw_set
        n_overlap = len(overlap)

        # 2×2 contingency table
        a = n_overlap
        b = pw_size - n_overlap
        c = prog_size - n_overlap
        d = bg_size - pw_size - prog_size + n_overlap
        a, b, c, d = max(a, 0), max(b, 0), max(c, 0), max(d, 0)

        _, p_value = stats.fisher_exact([[a, b], [c, d]], alternative="greater")

        expected = (pw_size * prog_size / bg_size) if bg_size > 0 else 0.0
        fold = n_overlap / expected if expected > 0 else 0.0
        frac = n_overlap / prog_size if prog_size > 0 else 0.0

        hits.append(PathwayHit(
            pathway_name=pw_name,
            overlap_genes=sorted(overlap),
            overlap_size=n_overlap,
            pathway_size=pw_size,
            p_value=p_value,
            fold_enrichment=fold,
            fraction_of_program=frac,
        ))

    # BH FDR correction
    if hits:
        p_vals = np.array([h.p_value for h in hits])
        fdr_vals = _benjamini_hochberg(p_vals)
        for h, q in zip(hits, fdr_vals):
            h.fdr = float(q)

    # Filter significant
    sig_hits = sorted(
        [h for h in hits if h.fdr <= fdr_threshold],
        key=lambda h: h.p_value,
    )

    # Novel genes: not in ANY significant pathway
    covered_by_sig: set[str] = set()
    for h in sig_hits:
        covered_by_sig.update(h.overlap_genes)
    novel = sorted(prog_set - covered_by_sig)
    novelty_frac = len(novel) / prog_size if prog_size > 0 else 0.0

    # Multi-pathway score
    span = len(sig_hits)
    novelty_bonus = 0.5 * novelty_frac if novelty_frac > 0.1 else 0.0
    mps = span * (1 - novelty_frac) + novelty_bonus

    return ProgramAnnotation(
        program_name="",  # Caller sets this
        program_genes=list(prog_set),
        significant_pathways=sig_hits,
        novel_genes=novel,
        novelty_fraction=novelty_frac,
        pathway_span=span,
        multi_pathway_score=mps,
    )


def annotate_all_programs(
    programs: dict[str, list[str]],
    curated_pathways: dict[str, list[str]],
    background: list[str] | None = None,
    fdr_threshold: float = 0.25,
) -> dict[str, ProgramAnnotation]:
    """Annotate all discovered programs against a curated pathway database.

    Parameters
    ----------
    programs : dict[str, list[str]]
        Discovered gene programs.
    curated_pathways : dict[str, list[str]]
        Reference pathway database.
    background : list[str] | None
        Background gene universe.
    fdr_threshold : float
        FDR threshold for pathway significance.

    Returns
    -------
    dict[str, ProgramAnnotation]
        Annotation for each program.
    """
    result: dict[str, ProgramAnnotation] = {}
    for prog_name, prog_genes in programs.items():
        ann = annotate_program(
            prog_genes, curated_pathways,
            background=background, fdr_threshold=fdr_threshold,
        )
        ann.program_name = prog_name
        result[prog_name] = ann
    return result


# ------------------------------------------------------------------
# Multi-pathway score and comparison
# ------------------------------------------------------------------


def multi_pathway_score(
    annotations: dict[str, ProgramAnnotation],
) -> dict[str, float]:
    """Return the multi-pathway score for each program.

    The score captures how many distinct curated pathways a single
    program spans, rewarding biological breadth while acknowledging
    novel (uncurated) content.

    Parameters
    ----------
    annotations : dict[str, ProgramAnnotation]
        Output of :func:`annotate_all_programs`.

    Returns
    -------
    dict[str, float]
        ``{program_name: score}``.
    """
    return {name: ann.multi_pathway_score for name, ann in annotations.items()}


def compare_with_gsea(
    programs: dict[str, list[str]],
    curated_pathways: dict[str, list[str]],
    ranked_genes: list[tuple[str, float]] | None = None,
    background: list[str] | None = None,
    fdr_threshold: float = 0.25,
) -> pd.DataFrame:
    """Head-to-head comparison: nPathway annotation vs standard GSEA.

    For each curated pathway, reports:
    - GSEA result (NES, FDR) if ``ranked_genes`` provided
    - Whether any nPathway program captures it (with overlap stats)
    - Whether nPathway programs capture *additional* biology (novel genes)

    Parameters
    ----------
    programs : dict[str, list[str]]
        Discovered gene programs.
    curated_pathways : dict[str, list[str]]
        Reference pathways (same used for GSEA).
    ranked_genes : list[tuple[str, float]] | None
        Ranked gene list for GSEA comparison.
    background : list[str] | None
        Background gene universe.
    fdr_threshold : float
        FDR threshold.

    Returns
    -------
    pd.DataFrame
        Comparison table with one row per curated pathway.
    """
    from npathway.evaluation.enrichment import preranked_gsea

    annotations = annotate_all_programs(
        programs, curated_pathways,
        background=background, fdr_threshold=fdr_threshold,
    )

    # Build pathway → best program mapping
    pw_to_best: dict[str, tuple[str, PathwayHit]] = {}
    for prog_name, ann in annotations.items():
        for hit in ann.significant_pathways:
            pw = hit.pathway_name
            if pw not in pw_to_best or hit.p_value < pw_to_best[pw][1].p_value:
                pw_to_best[pw] = (prog_name, hit)

    # GSEA results (if ranked genes provided)
    gsea_results: dict[str, dict[str, float]] = {}
    if ranked_genes is not None:
        gsea_df = preranked_gsea(ranked_genes, curated_pathways, n_perm=1000, seed=42)
        for _, row in gsea_df.iterrows():
            gsea_results[row["program"]] = {
                "gsea_nes": row.get("nes", 0.0),
                "gsea_pvalue": row.get("p_value", 1.0),
                "gsea_fdr": row.get("fdr", 1.0),
            }

    rows: list[dict[str, Any]] = []
    for pw_name in curated_pathways:
        row: dict[str, Any] = {"pathway": pw_name, "pathway_size": len(curated_pathways[pw_name])}

        # GSEA columns
        if pw_name in gsea_results:
            row.update(gsea_results[pw_name])
            row["gsea_significant"] = gsea_results[pw_name]["gsea_fdr"] <= fdr_threshold
        else:
            row["gsea_nes"] = np.nan
            row["gsea_pvalue"] = np.nan
            row["gsea_fdr"] = np.nan
            row["gsea_significant"] = False

        # nPathway annotation columns
        if pw_name in pw_to_best:
            prog_name, hit = pw_to_best[pw_name]
            row["npathway_program"] = prog_name
            row["npathway_overlap"] = hit.overlap_size
            row["npathway_pvalue"] = hit.p_value
            row["npathway_fdr"] = hit.fdr
            row["npathway_fold"] = hit.fold_enrichment
            row["npathway_fraction"] = hit.fraction_of_program
            row["npathway_significant"] = True

            # Novel genes in that program
            ann = annotations[prog_name]
            row["npathway_novel_genes"] = len(ann.novel_genes)
            row["npathway_pathway_span"] = ann.pathway_span
        else:
            row["npathway_program"] = ""
            row["npathway_overlap"] = 0
            row["npathway_pvalue"] = 1.0
            row["npathway_fdr"] = 1.0
            row["npathway_fold"] = 0.0
            row["npathway_fraction"] = 0.0
            row["npathway_significant"] = False
            row["npathway_novel_genes"] = 0
            row["npathway_pathway_span"] = 0

        rows.append(row)

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------


def pathway_annotation_report(
    annotations: dict[str, ProgramAnnotation],
) -> str:
    """Generate a human-readable multi-pathway annotation report.

    Parameters
    ----------
    annotations : dict[str, ProgramAnnotation]
        Output of :func:`annotate_all_programs`.

    Returns
    -------
    str
        Formatted report string.
    """
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("nPathway Multi-Pathway Annotation Report")
    lines.append("=" * 70)
    lines.append("")

    # Summary statistics
    total_programs = len(annotations)
    multi_pw = sum(1 for a in annotations.values() if a.pathway_span >= 2)
    single_pw = sum(1 for a in annotations.values() if a.pathway_span == 1)
    novel_only = sum(1 for a in annotations.values() if a.pathway_span == 0)
    lines.append(f"Total programs: {total_programs}")
    lines.append(f"  Multi-pathway programs (span >= 2): {multi_pw}")
    lines.append(f"  Single-pathway programs: {single_pw}")
    lines.append(f"  Novel-only programs (no significant overlap): {novel_only}")
    lines.append("")

    # Per-program details
    sorted_anns = sorted(
        annotations.values(),
        key=lambda a: a.multi_pathway_score,
        reverse=True,
    )

    for ann in sorted_anns:
        lines.append("-" * 70)
        lines.append(f"Program: {ann.program_name}")
        lines.append(f"  Size: {len(ann.program_genes)} genes")
        lines.append(f"  Pathway span: {ann.pathway_span}")
        lines.append(f"  Multi-pathway score: {ann.multi_pathway_score:.2f}")
        lines.append(f"  Novelty fraction: {ann.novelty_fraction:.1%}")

        if ann.significant_pathways:
            lines.append("  Significant pathway overlaps:")
            for hit in ann.significant_pathways:
                lines.append(
                    f"    → {hit.pathway_name}: "
                    f"{hit.overlap_size} genes "
                    f"(p={hit.p_value:.2e}, FDR={hit.fdr:.3f}, "
                    f"fold={hit.fold_enrichment:.1f}, "
                    f"{hit.fraction_of_program:.0%} of program)"
                )

        if ann.novel_genes:
            top_novel = ann.novel_genes[:10]
            more = f" (+{len(ann.novel_genes) - 10} more)" if len(ann.novel_genes) > 10 else ""
            lines.append(f"  Novel genes: {', '.join(top_novel)}{more}")

        lines.append("")

    return "\n".join(lines)


def annotation_to_dataframe(
    annotations: dict[str, ProgramAnnotation],
) -> pd.DataFrame:
    """Convert annotations to a summary DataFrame.

    Parameters
    ----------
    annotations : dict[str, ProgramAnnotation]
        Output of :func:`annotate_all_programs`.

    Returns
    -------
    pd.DataFrame
        One row per program with summary columns.
    """
    rows: list[dict[str, Any]] = []
    for name, ann in annotations.items():
        pw_names = [h.pathway_name for h in ann.significant_pathways]
        rows.append({
            "program": name,
            "n_genes": len(ann.program_genes),
            "pathway_span": ann.pathway_span,
            "multi_pathway_score": ann.multi_pathway_score,
            "significant_pathways": "; ".join(pw_names),
            "n_novel_genes": len(ann.novel_genes),
            "novelty_fraction": ann.novelty_fraction,
            "top_novel_genes": ", ".join(ann.novel_genes[:5]),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Compute BH FDR q-values.

    Parameters
    ----------
    p_values : np.ndarray
        Raw p-values.

    Returns
    -------
    np.ndarray
        Adjusted q-values.
    """
    p = np.asarray(p_values, dtype=np.float64)
    n = len(p)
    if n == 0:
        return np.array([], dtype=np.float64)

    order = np.argsort(p)
    ranked = p[order]
    ranks = np.arange(1, n + 1, dtype=np.float64)

    adjusted = ranked * (n / ranks)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    result = np.empty(n, dtype=np.float64)
    result[order] = adjusted
    return result
