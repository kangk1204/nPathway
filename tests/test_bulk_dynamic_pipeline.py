"""Tests for bulk dynamic pathway pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from npathway.pipeline.bulk_dynamic import (
    BulkDynamicConfig,
    _build_context_membership_scores,
    _summarize_reference_annotation_layers,
    run_bulk_dynamic_pipeline,
)


def test_bulk_dynamic_pipeline_3v3(tmp_path) -> None:
    """Pipeline should run on 3v3 bulk contrast and emit core outputs."""
    rng = np.random.default_rng(42)
    genes = [f"G{i:03d}" for i in range(40)]
    samples = [f"S{i}" for i in range(1, 7)]

    base = rng.normal(loc=5.0, scale=0.6, size=(40, 6))
    # Inject a group-A signal for the first 8 genes.
    base[:8, :3] += 2.5
    base = np.clip(base, 0.0, None)

    matrix = pd.DataFrame(base, index=genes, columns=samples).reset_index()
    matrix.columns = ["gene"] + samples
    matrix_path = tmp_path / "matrix.csv"
    matrix.to_csv(matrix_path, index=False)

    meta = pd.DataFrame(
        {
            "sample": samples,
            "group": ["A", "A", "A", "B", "B", "B"],
        }
    )
    metadata_path = tmp_path / "metadata.csv"
    meta.to_csv(metadata_path, index=False)

    outdir = tmp_path / "out"
    cfg = BulkDynamicConfig(
        matrix_path=str(matrix_path),
        metadata_path=str(metadata_path),
        group_col="group",
        group_a="A",
        group_b="B",
        sample_col="sample",
        matrix_orientation="genes_by_samples",
        raw_counts=False,
        discovery_method="kmeans",
        n_programs=6,
        n_components=10,
        gsea_n_perm=50,
        n_bootstrap=0,
    )
    result = run_bulk_dynamic_pipeline(cfg, outdir)

    assert result.n_samples == 6
    assert result.n_genes == 40
    assert result.n_programs > 0

    de_path = outdir / "differential" / "de_results.csv"
    gsea_path = outdir / "enrichment" / "enrichment_gsea_with_claim_gates.csv"
    context_path = outdir / "enrichment" / "contextual_membership_scores.csv"
    summary_path = outdir / "summary.md"
    gmt_path = outdir / "discovery" / "dynamic_programs.gmt"

    assert de_path.exists()
    assert gsea_path.exists()
    assert context_path.exists()
    assert summary_path.exists()
    assert gmt_path.exists()

    gsea = pd.read_csv(gsea_path)
    assert "claim_supported" in gsea.columns
    assert "gate_effect" in gsea.columns

    context = pd.read_csv(context_path)
    assert "prob_A" in context.columns
    assert "prob_B" in context.columns
    assert "context_shift" in context.columns
    assert "signed_significance" in context.columns
    assert "context_evidence" in context.columns
    assert "neglog10_p_value" in context.columns
    assert np.allclose(
        context["context_evidence"].to_numpy(dtype=float),
        context["context_shift"].to_numpy(dtype=float) * context["neglog10_p_value"].to_numpy(dtype=float),
    )

    de_lookup = pd.read_csv(de_path).set_index("gene")
    merged = context.merge(
        de_lookup[["logfc_a_minus_b"]],
        left_on="gene",
        right_index=True,
        how="left",
    )
    expected_signed = np.sign(merged["logfc_a_minus_b"].to_numpy(dtype=float)) * merged[
        "neglog10_p_value"
    ].to_numpy(dtype=float)
    assert np.allclose(
        merged["signed_significance"].to_numpy(dtype=float),
        expected_signed,
        rtol=0.0,
        atol=1e-9,
    )


def test_bulk_dynamic_pipeline_program_annotation_with_custom_gmt(tmp_path) -> None:
    """Custom GMT should annotate programs and emit overlap tables."""
    rng = np.random.default_rng(123)
    genes = [f"G{i:03d}" for i in range(60)]
    samples = [f"S{i}" for i in range(1, 9)]  # 4 vs 4

    base = rng.normal(loc=5.0, scale=0.7, size=(60, 8))
    base[:10, :4] += 2.3  # signal for group A
    base[10:20, 4:] += 2.0  # signal for group B
    base = np.clip(base, 0.0, None)

    matrix = pd.DataFrame(base, index=genes, columns=samples).reset_index()
    matrix.columns = ["gene"] + samples
    matrix_path = tmp_path / "matrix.csv"
    matrix.to_csv(matrix_path, index=False)

    meta = pd.DataFrame(
        {
            "sample": samples,
            "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
        }
    )
    metadata_path = tmp_path / "metadata.csv"
    meta.to_csv(metadata_path, index=False)

    gmt_path = tmp_path / "custom_ref.gmt"
    with open(gmt_path, "w", encoding="utf-8") as fh:
        fh.write("Custom_A\tNA\t" + "\t".join(genes[:12]) + "\n")
        fh.write("Custom_B\tNA\t" + "\t".join(genes[10:24]) + "\n")

    outdir = tmp_path / "out_annot"
    cfg = BulkDynamicConfig(
        matrix_path=str(matrix_path),
        metadata_path=str(metadata_path),
        group_col="group",
        group_a="A",
        group_b="B",
        sample_col="sample",
        matrix_orientation="genes_by_samples",
        raw_counts=False,
        discovery_method="kmeans",
        n_programs=8,
        n_components=12,
        gsea_n_perm=50,
        annotate_programs=True,
        annotation_collections=tuple(),
        annotation_gmt_path=str(gmt_path),
        annotation_topk_per_program=5,
    )
    run_bulk_dynamic_pipeline(cfg, outdir)

    ann_path = outdir / "annotation" / "program_annotation_matches.csv"
    overlap_path = outdir / "annotation" / "program_reference_overlap_long.csv"
    source_summary_path = outdir / "annotation" / "program_reference_source_summary.csv"
    family_summary_path = outdir / "annotation" / "program_reference_family_summary.csv"
    rename_path = outdir / "annotation" / "program_renaming_map.csv"
    assert ann_path.exists()
    assert overlap_path.exists()
    assert source_summary_path.exists()
    assert family_summary_path.exists()
    assert rename_path.exists()

    ann = pd.read_csv(ann_path)
    assert "best_reference_name" in ann.columns
    assert "best_jaccard" in ann.columns
    assert len(ann) > 0


def test_summarize_reference_annotation_layers_collapses_duplicate_families() -> None:
    """Annotation summary should provide source- and family-level exports."""
    overlap = pd.DataFrame(
        {
            "program": ["program_1", "program_1", "program_2"],
            "reference_name": [
                "WP::ALZHEIMERS_DISEASE",
                "KEGG::ALZHEIMERS_DISEASE",
                "REACTOME::INNATE_IMMUNE_SYSTEM",
            ],
            "jaccard": [0.17, 0.15, 0.12],
            "overlap_n": [8, 7, 6],
            "program_n": [24, 24, 18],
            "reference_n": [50, 55, 30],
        }
    )

    source_summary, family_summary = _summarize_reference_annotation_layers(overlap)

    assert {"WikiPathways", "KEGG", "Reactome"} <= set(source_summary["source_display"].astype(str))
    ad_row = family_summary.loc[family_summary["family_key"] == "ALZHEIMERS DISEASE"].iloc[0]
    assert ad_row["references_merged"] == 2
    assert "WikiPathways" in ad_row["sources_display"]
    assert "KEGG" in ad_row["sources_display"]
    assert float(ad_row["disease_priority_score"]) > 0
    assert ad_row["priority_band"] == "Disease-prioritized"
    assert float(ad_row["interpretation_score"]) > float(ad_row["best_jaccard"])


def test_summarize_reference_annotation_layers_prioritizes_disease_families() -> None:
    overlap = pd.DataFrame(
        {
            "program": ["program_1", "program_2"],
            "reference_name": [
                "KEGG::ALZHEIMERS_DISEASE",
                "REACTOME::FASTK family proteins regulate processing and stability of mitochondrial RNAs",
            ],
            "jaccard": [0.11, 0.71],
            "overlap_n": [6, 14],
            "program_n": [24, 24],
            "reference_n": [55, 28],
        }
    )

    _, family_summary = _summarize_reference_annotation_layers(overlap)
    assert family_summary.iloc[0]["family_key"] == "ALZHEIMERS DISEASE"
    assert "interpretation_score" in family_summary.columns


def test_build_context_membership_scores_empty_input_schema() -> None:
    """Empty program_scores should still return a stable output schema."""
    de_df = pd.DataFrame(
        {
            "gene": ["G1", "G2"],
            "logfc_a_minus_b": [0.5, -0.3],
            "p_value": [0.01, 0.2],
        }
    )
    out = _build_context_membership_scores(
        program_scores={},
        de_df=de_df,
        group_a="A",
        group_b="B",
    )
    assert out.empty
    expected_cols = {
        "program",
        "gene",
        "base_membership",
        "prob_A",
        "prob_B",
        "contextual_score_A",
        "contextual_score_B",
        "p_value",
        "neglog10_p_value",
        "context_shift",
        "signed_significance",
        "context_evidence",
    }
    assert expected_cols.issubset(set(out.columns))
