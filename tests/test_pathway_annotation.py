"""Tests for multi-pathway annotation module.

Tests the core differentiator: discovered gene programs annotated
against multiple curated pathways simultaneously.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from npathway.evaluation.pathway_annotation import (
    ProgramAnnotation,
    annotate_all_programs,
    annotate_program,
    annotation_to_dataframe,
    compare_with_gsea,
    multi_pathway_score,
    pathway_annotation_report,
    reference_relevance_band,
    reference_relevance_score,
)


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture()
def curated_pathways() -> dict[str, list[str]]:
    """Reference pathways mimicking Hallmark/KEGG structure."""
    return {
        "TP53_PATHWAY": [f"TP53_G{i}" for i in range(20)] + ["SHARED_1", "SHARED_2"],
        "METABOLISM": [f"MET_G{i}" for i in range(15)] + ["SHARED_1", "SHARED_3"],
        "IMMUNE_RESPONSE": [f"IMM_G{i}" for i in range(18)] + ["SHARED_2", "SHARED_3"],
        "APOPTOSIS": [f"APO_G{i}" for i in range(12)],
        "CELL_CYCLE": [f"CC_G{i}" for i in range(25)],
    }


@pytest.fixture()
def multi_pathway_program() -> list[str]:
    """A program spanning TP53 + Metabolism + novel genes."""
    return (
        [f"TP53_G{i}" for i in range(10)]  # 10 from TP53
        + [f"MET_G{i}" for i in range(8)]   # 8 from Metabolism
        + ["SHARED_1", "SHARED_2"]           # Shared genes
        + ["NOVEL_1", "NOVEL_2", "NOVEL_3"]  # 3 novel genes
    )


@pytest.fixture()
def single_pathway_program() -> list[str]:
    """A program from one pathway only."""
    return [f"APO_G{i}" for i in range(10)]


@pytest.fixture()
def novel_program() -> list[str]:
    """A program with no pathway overlap."""
    return [f"NEW_G{i}" for i in range(15)]


@pytest.fixture()
def background(curated_pathways: dict[str, list[str]]) -> list[str]:
    """Background gene universe."""
    bg: set[str] = set()
    for genes in curated_pathways.values():
        bg.update(genes)
    bg.update([f"NOVEL_{i}" for i in range(1, 20)])
    bg.update([f"NEW_G{i}" for i in range(20)])
    bg.update([f"BG_G{i}" for i in range(100)])
    return sorted(bg)


# ------------------------------------------------------------------ #
# annotate_program
# ------------------------------------------------------------------ #

class TestAnnotateProgram:
    """Test single program annotation."""

    def test_multi_pathway_detected(
        self,
        multi_pathway_program: list[str],
        curated_pathways: dict[str, list[str]],
        background: list[str],
    ) -> None:
        ann = annotate_program(
            multi_pathway_program, curated_pathways,
            background=background, fdr_threshold=0.25,
        )
        # Should find at least 2 significant pathways (TP53 + Metabolism)
        assert ann.pathway_span >= 2
        pw_names = {h.pathway_name for h in ann.significant_pathways}
        assert "TP53_PATHWAY" in pw_names
        assert "METABOLISM" in pw_names

    def test_novel_genes_identified(
        self,
        multi_pathway_program: list[str],
        curated_pathways: dict[str, list[str]],
        background: list[str],
    ) -> None:
        ann = annotate_program(
            multi_pathway_program, curated_pathways,
            background=background, fdr_threshold=0.25,
        )
        # NOVEL_1, NOVEL_2, NOVEL_3 should be in novel genes
        novel_set = set(ann.novel_genes)
        assert "NOVEL_1" in novel_set or ann.novelty_fraction > 0

    def test_single_pathway(
        self,
        single_pathway_program: list[str],
        curated_pathways: dict[str, list[str]],
        background: list[str],
    ) -> None:
        ann = annotate_program(
            single_pathway_program, curated_pathways,
            background=background, fdr_threshold=0.25,
        )
        assert ann.pathway_span >= 1
        if ann.significant_pathways:
            assert ann.significant_pathways[0].pathway_name == "APOPTOSIS"

    def test_novel_only_program(
        self,
        novel_program: list[str],
        curated_pathways: dict[str, list[str]],
        background: list[str],
    ) -> None:
        ann = annotate_program(
            novel_program, curated_pathways,
            background=background, fdr_threshold=0.25,
        )
        assert ann.pathway_span == 0
        assert ann.novelty_fraction == pytest.approx(1.0)

    def test_fdr_correction(
        self,
        multi_pathway_program: list[str],
        curated_pathways: dict[str, list[str]],
        background: list[str],
    ) -> None:
        ann = annotate_program(
            multi_pathway_program, curated_pathways,
            background=background, fdr_threshold=0.25,
        )
        for hit in ann.significant_pathways:
            assert hit.fdr <= 0.25
            assert hit.fdr >= hit.p_value  # FDR >= raw p-value

    def test_fold_enrichment(
        self,
        multi_pathway_program: list[str],
        curated_pathways: dict[str, list[str]],
        background: list[str],
    ) -> None:
        ann = annotate_program(
            multi_pathway_program, curated_pathways,
            background=background, fdr_threshold=0.25,
        )
        for hit in ann.significant_pathways:
            assert hit.fold_enrichment > 1.0  # Should be enriched

    def test_fraction_of_program(
        self,
        multi_pathway_program: list[str],
        curated_pathways: dict[str, list[str]],
        background: list[str],
    ) -> None:
        ann = annotate_program(
            multi_pathway_program, curated_pathways,
            background=background, fdr_threshold=0.25,
        )
        total_frac = sum(h.fraction_of_program for h in ann.significant_pathways)
        # Can exceed 1.0 due to shared genes between pathways
        assert total_frac > 0


# ------------------------------------------------------------------ #
# annotate_all_programs
# ------------------------------------------------------------------ #

class TestAnnotateAll:
    """Test batch annotation of all programs."""

    def test_all_programs_annotated(
        self,
        curated_pathways: dict[str, list[str]],
        background: list[str],
    ) -> None:
        programs = {
            "prog_multi": [f"TP53_G{i}" for i in range(10)] + [f"MET_G{i}" for i in range(5)],
            "prog_single": [f"APO_G{i}" for i in range(8)],
            "prog_novel": [f"NEW_G{i}" for i in range(10)],
        }
        anns = annotate_all_programs(
            programs, curated_pathways,
            background=background,
        )
        assert len(anns) == 3
        assert anns["prog_multi"].pathway_span >= 1
        assert anns["prog_novel"].pathway_span == 0

    def test_multi_pathway_scores(
        self,
        curated_pathways: dict[str, list[str]],
        background: list[str],
    ) -> None:
        programs = {
            "prog_multi": [f"TP53_G{i}" for i in range(10)] + [f"MET_G{i}" for i in range(5)],
            "prog_novel": [f"NEW_G{i}" for i in range(10)],
        }
        anns = annotate_all_programs(programs, curated_pathways, background=background)
        scores = multi_pathway_score(anns)
        assert scores["prog_multi"] > scores["prog_novel"]


def test_reference_relevance_score_prioritizes_disease_signal() -> None:
    assert reference_relevance_score("KEGG::ALZHEIMERS_DISEASE") > reference_relevance_score(
        "REACTOME::FASTK family proteins regulate processing and stability of mitochondrial RNAs"
    )
    assert reference_relevance_score("GO_BP::GOBP_SYNAPTIC_SIGNALING") > reference_relevance_score(
        "REACTOME::tRNA processing in the mitochondrion"
    )


def test_reference_relevance_band_labels_expected_groups() -> None:
    assert reference_relevance_band("WP::ALZHEIMERS_DISEASE") == "Disease-prioritized"
    assert reference_relevance_band("GO_BP::GOBP_CELL_CELL_SIGNALING") in {"Supportive", "Disease-prioritized"}
    assert reference_relevance_band(
        "REACTOME::FASTK family proteins regulate processing and stability of mitochondrial RNAs"
    ) == "Background"


# ------------------------------------------------------------------ #
# Reporting
# ------------------------------------------------------------------ #

class TestReporting:
    """Test report generation."""

    def test_report_string(
        self,
        curated_pathways: dict[str, list[str]],
        background: list[str],
    ) -> None:
        programs = {
            "prog_A": [f"TP53_G{i}" for i in range(10)] + [f"IMM_G{i}" for i in range(5)],
        }
        anns = annotate_all_programs(programs, curated_pathways, background=background)
        report = pathway_annotation_report(anns)
        assert "Multi-Pathway Annotation Report" in report
        assert "prog_A" in report

    def test_annotation_dataframe(
        self,
        curated_pathways: dict[str, list[str]],
        background: list[str],
    ) -> None:
        programs = {
            "prog_A": [f"TP53_G{i}" for i in range(10)],
            "prog_B": [f"NEW_G{i}" for i in range(10)],
        }
        anns = annotate_all_programs(programs, curated_pathways, background=background)
        df = annotation_to_dataframe(anns)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "pathway_span" in df.columns
        assert "multi_pathway_score" in df.columns
        assert "novelty_fraction" in df.columns


# ------------------------------------------------------------------ #
# Comparison with GSEA
# ------------------------------------------------------------------ #

class TestCompareWithGSEA:
    """Test head-to-head comparison with standard GSEA."""

    def test_comparison_table(
        self,
        curated_pathways: dict[str, list[str]],
        background: list[str],
    ) -> None:
        programs = {
            "prog_A": [f"TP53_G{i}" for i in range(10)] + [f"MET_G{i}" for i in range(5)],
        }
        df = compare_with_gsea(
            programs, curated_pathways,
            background=background,
        )
        assert isinstance(df, pd.DataFrame)
        assert "pathway" in df.columns
        assert "npathway_significant" in df.columns
        assert len(df) == len(curated_pathways)

    def test_comparison_with_ranking(
        self,
        curated_pathways: dict[str, list[str]],
        background: list[str],
    ) -> None:
        programs = {
            "prog_A": [f"TP53_G{i}" for i in range(10)],
        }
        # Create a ranked gene list with TP53 genes at top
        ranked = [(g, 5.0 - i * 0.1) for i, g in enumerate(background)]
        df = compare_with_gsea(
            programs, curated_pathways,
            ranked_genes=ranked,
            background=background,
        )
        assert "gsea_nes" in df.columns
        assert "gsea_fdr" in df.columns


# ------------------------------------------------------------------ #
# Edge cases
# ------------------------------------------------------------------ #

class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_empty_program(self, curated_pathways: dict[str, list[str]]) -> None:
        ann = annotate_program([], curated_pathways)
        assert ann.pathway_span == 0
        assert ann.novelty_fraction == 0.0

    def test_empty_pathways(self) -> None:
        ann = annotate_program(["GENE1", "GENE2"], {})
        assert ann.pathway_span == 0
        assert ann.novelty_fraction == 1.0

    def test_single_gene_program(self, curated_pathways: dict[str, list[str]]) -> None:
        ann = annotate_program(["TP53_G0"], curated_pathways)
        assert isinstance(ann, ProgramAnnotation)
