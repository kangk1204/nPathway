"""Tests for benchmark classes."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from npathway.evaluation.benchmark_context import ContextSpecificityBenchmark
from npathway.evaluation.benchmark_perturbation import PerturbationBenchmark
from npathway.evaluation.benchmark_power import PowerBenchmark
from npathway.evaluation.benchmark_robustness import CrossModelBenchmark

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
EXPANDED_SUITE_PATH = SCRIPTS_DIR / "run_expanded_benchmark_suite.py"
MULTISEED_PATH = SCRIPTS_DIR / "run_multiseed_benchmark.py"


def _load_expanded_suite_module():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(
        "expanded_suite_for_benchmark_tests",
        EXPANDED_SUITE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_multiseed_module():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(
        "multiseed_for_benchmark_tests",
        MULTISEED_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_gene_programs(prefix: str, n_programs: int = 5, genes_per: int = 20) -> dict[str, list[str]]:
    """Helper to create synthetic gene programs."""
    programs = {}
    for p in range(n_programs):
        start = p * genes_per
        programs[f"prog_{p}"] = [f"{prefix}_GENE_{start + i}" for i in range(genes_per)]
    return programs


def test_power_benchmark_synthetic() -> None:
    """PowerBenchmark should run with synthetic data and return valid metrics."""
    rng = np.random.default_rng(42)

    # Create gene programs
    programs = _make_gene_programs("P", n_programs=5, genes_per=15)

    # Create synthetic expression data covering the genes
    all_genes = sorted({g for gs in programs.values() for g in gs})
    n_cells = 100
    n_genes = len(all_genes)
    expression = rng.lognormal(2.0, 1.0, size=(n_cells, n_genes)).astype(np.float64)

    benchmark = PowerBenchmark(
        fold_changes=[1.5, 2.0],
        n_replicates=2,
        fdr_threshold=0.05,
        treatment_fraction=0.3,
        seed=42,
    )

    results = benchmark.run(
        programs,
        expression_matrix=expression,
        gene_names=all_genes,
        target_programs=["prog_0"],
        enrichment_method="fisher",
        n_top_de=50,
    )

    assert isinstance(results, dict)
    assert "per_collection" in results
    assert "learned" in results["per_collection"]
    assert "tpr_by_fc" in results["per_collection"]["learned"]
    assert "fpr_by_fc" in results["per_collection"]["learned"]
    assert "tpr_ci_by_fc" in results["per_collection"]["learned"]
    assert "fpr_ci_by_fc" in results["per_collection"]["learned"]
    assert "auc_power_curve" in results["per_collection"]["learned"]
    assert results["n_simulations"] > 0

    tpr_ci = results["per_collection"]["learned"]["tpr_ci_by_fc"][1.5]
    fpr_ci = results["per_collection"]["learned"]["fpr_ci_by_fc"][1.5]
    assert len(tpr_ci) == 2 and len(fpr_ci) == 2
    assert 0.0 <= tpr_ci[0] <= tpr_ci[1] <= 1.0
    assert 0.0 <= fpr_ci[0] <= fpr_ci[1] <= 1.0

    # Should be able to get results DataFrame
    df = benchmark.get_results_df()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_context_specificity_benchmark() -> None:
    """ContextSpecificityBenchmark should compute metrics across contexts."""
    # Create different programs for different contexts
    # Context A and B share some genes, Context C is more different
    context_a = {
        "prog_0": ["GENE_0", "GENE_1", "GENE_2", "GENE_3", "GENE_4"],
        "prog_1": ["GENE_5", "GENE_6", "GENE_7", "GENE_8", "GENE_9"],
    }
    context_b = {
        "prog_0": ["GENE_0", "GENE_1", "GENE_2", "GENE_10", "GENE_11"],
        "prog_1": ["GENE_5", "GENE_6", "GENE_12", "GENE_13", "GENE_14"],
    }
    context_c = {
        "prog_0": ["GENE_20", "GENE_21", "GENE_22", "GENE_23", "GENE_24"],
        "prog_1": ["GENE_25", "GENE_26", "GENE_27", "GENE_28", "GENE_29"],
    }

    context_programs = {
        "celltype_A": context_a,
        "celltype_B": context_b,
        "celltype_C": context_c,
    }

    benchmark = ContextSpecificityBenchmark()
    results = benchmark.run(
        context_a,
        context_programs=context_programs,
    )

    assert isinstance(results, dict)
    assert "mean_cross_context_jaccard" in results
    assert "mean_reassignment_freq" in results
    assert "mean_specificity_score" in results
    assert results["n_contexts"] == 3

    # Jaccard should be between 0 and 1
    assert 0.0 <= results["mean_cross_context_jaccard"] <= 1.0

    # Results DataFrame should be available
    df = benchmark.get_results_df()
    assert isinstance(df, pd.DataFrame)


def test_cross_model_benchmark() -> None:
    """CrossModelBenchmark should compare gene programs from different models."""
    # Two models with some overlapping programs
    model_a_programs = {
        "prog_0": ["G1", "G2", "G3", "G4", "G5"],
        "prog_1": ["G6", "G7", "G8", "G9", "G10"],
        "prog_2": ["G11", "G12", "G13", "G14", "G15"],
    }
    model_b_programs = {
        "prog_0": ["G1", "G2", "G3", "G16", "G17"],  # partial overlap with model_a prog_0
        "prog_1": ["G6", "G7", "G18", "G19", "G20"],  # partial overlap with model_a prog_1
        "prog_2": ["G21", "G22", "G23", "G24", "G25"],  # no overlap
    }

    model_programs = {
        "scgpt": model_a_programs,
        "geneformer": model_b_programs,
    }

    benchmark = CrossModelBenchmark(jaccard_threshold=0.2)
    results = benchmark.run(
        model_a_programs,
        model_programs=model_programs,
    )

    assert isinstance(results, dict)
    assert "mean_best_match_jaccard" in results
    assert "mean_ari" in results
    assert "mean_nmi" in results
    assert "n_consensus_programs" in results
    assert results["n_models"] == 2

    # Check the pairwise results
    df = benchmark.get_results_df()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1  # One pair: scgpt vs geneformer

    # Check overlap matrix
    overlap = benchmark.get_overlap_matrix("scgpt_vs_geneformer")
    assert isinstance(overlap, pd.DataFrame)
    assert overlap.shape == (3, 3)


def test_perturbation_benchmark_structure() -> None:
    """PerturbationBenchmark should run with pre-computed DE results."""
    # Create gene programs
    programs = {
        "pathway_A": ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10"],
        "pathway_B": ["G11", "G12", "G13", "G14", "G15", "G16", "G17", "G18", "G19", "G20"],
        "pathway_C": ["G21", "G22", "G23", "G24", "G25", "G26", "G27", "G28", "G29", "G30"],
    }

    # Create synthetic DE results where perturbation of pathway_A genes
    # should recover pathway_A
    rng = np.random.default_rng(42)
    all_genes = sorted({g for gs in programs.values() for g in gs})

    # For pert_A: genes in pathway_A have high scores
    de_pert_a = []
    for gene in all_genes:
        if gene in programs["pathway_A"]:
            score = rng.uniform(5.0, 10.0)
        else:
            score = rng.uniform(-1.0, 1.0)
        de_pert_a.append((gene, float(score)))
    de_pert_a.sort(key=lambda x: x[1], reverse=True)

    # For pert_B: genes in pathway_B have high scores
    de_pert_b = []
    for gene in all_genes:
        if gene in programs["pathway_B"]:
            score = rng.uniform(5.0, 10.0)
        else:
            score = rng.uniform(-1.0, 1.0)
        de_pert_b.append((gene, float(score)))
    de_pert_b.sort(key=lambda x: x[1], reverse=True)

    de_results = {
        "knockout_A": de_pert_a,
        "knockout_B": de_pert_b,
    }
    perturbation_to_pathway = {
        "knockout_A": "pathway_A",
        "knockout_B": "pathway_B",
    }

    benchmark = PerturbationBenchmark(fdr_threshold=0.05)
    results = benchmark.run(
        programs,
        de_results=de_results,
        perturbation_to_pathway=perturbation_to_pathway,
        enrichment_method="fisher",
    )

    assert isinstance(results, dict)
    assert "recovery_rate" in results
    assert "mean_rank" in results
    assert "mrr" in results
    assert "aurc" in results
    assert results["n_perturbations"] == 2

    # Check per-perturbation DataFrame
    df = benchmark.get_results_df()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "perturbation" in df.columns
    assert "rank" in df.columns
    assert "recovered" in df.columns


def test_perturbation_benchmark_plot_handles_empty_results(tmp_path: Path) -> None:
    """plot_results should tolerate a run with zero valid perturbations."""
    benchmark = PerturbationBenchmark()
    benchmark._store_results(
        {
            "recovery_rate": 0.0,
            "mean_rank": float("nan"),
            "mrr": 0.0,
            "aurc": 0.0,
            "n_perturbations": 0,
        }
    )
    benchmark._per_perturbation_results = []

    fig = benchmark.plot_results(save_path=str(tmp_path / "empty_perturbation.png"))
    import matplotlib.pyplot as plt

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_pairwise_stats_exports_stratified_fdr_columns() -> None:
    """Expanded-suite pairwise stats should include grouped BH columns."""
    mod = _load_expanded_suite_module()
    seed_summary = pd.DataFrame(
        {
            "dataset": ["d1"] * 12,
            "seed": [0, 1, 2, 3] * 3,
            "method": (
                ["cNMF"] * 4
                + ["nPathway-Ensemble"] * 4
                + ["Curated-Hallmark"] * 4
            ),
            "discovery_mean_hallmark_alignment": (
                [0.40, 0.41, 0.39, 0.40]
                + [0.55, 0.56, 0.54, 0.57]
                + [0.46, 0.47, 0.45, 0.46]
            ),
            "power_mean_fpr": (
                [0.10, 0.11, 0.10, 0.09]
                + [0.06, 0.07, 0.05, 0.06]
                + [0.08, 0.09, 0.08, 0.09]
            ),
        }
    )
    stats = mod.pairwise_stats_vs_cnmf(
        seed_summary,
        metrics=[
            ("discovery_mean_hallmark_alignment", "greater"),
            ("power_mean_fpr", "less"),
        ],
        n_bootstrap=300,
    )

    expected_cols = {
        "effect_size_rank_biserial",
        "effect_size_rank_biserial_directional",
        "fdr_bh_by_metric",
        "fdr_bh_by_track_metric",
        "sign_fdr_bh_by_metric",
        "sign_fdr_bh_by_track_metric",
    }
    assert expected_cols.issubset(set(stats.columns))
    assert stats["effect_size_rank_biserial"].between(-1.0, 1.0).all()
    assert stats["effect_size_rank_biserial_directional"].between(-1.0, 1.0).all()

    finite_rows = stats[stats["p_value"].notna()]
    assert finite_rows["fdr_bh_by_metric"].notna().all()
    assert finite_rows["fdr_bh_by_track_metric"].notna().all()


def test_multiseed_pairwise_stats_exports_stratified_fdr_columns() -> None:
    """Multi-seed stats should expose effect size and grouped FDR columns."""
    mod = _load_multiseed_module()
    seed_summary = pd.DataFrame(
        {
            "seed": [0, 1, 2, 3] * 3,
            "method": (
                ["cNMF"] * 4
                + ["nPathway-Ensemble"] * 4
                + ["Curated-Hallmark"] * 4
            ),
            "discovery_mean_hallmark_alignment": (
                [0.40, 0.41, 0.39, 0.40]
                + [0.55, 0.56, 0.54, 0.57]
                + [0.46, 0.47, 0.45, 0.46]
            ),
            "power_mean_fpr": (
                [0.10, 0.11, 0.10, 0.09]
                + [0.06, 0.07, 0.05, 0.06]
                + [0.08, 0.09, 0.08, 0.09]
            ),
            "recovery_mean_n_sig": (
                [4.0, 4.0, 4.0, 4.0]
                + [6.0, 6.0, 5.0, 6.0]
                + [5.0, 5.0, 5.0, 5.0]
            ),
            "recovery_mean_hallmark_jaccard": (
                [0.2, 0.21, 0.2, 0.2]
                + [0.35, 0.34, 0.33, 0.36]
                + [0.25, 0.26, 0.24, 0.25]
            ),
            "power_mean_tpr": (
                [0.3, 0.31, 0.29, 0.3]
                + [0.55, 0.56, 0.54, 0.57]
                + [0.45, 0.44, 0.46, 0.45]
            ),
        }
    )
    stats = mod.pairwise_stats(seed_summary, n_bootstrap=300)

    expected_cols = {
        "effect_size_rank_biserial",
        "effect_size_rank_biserial_directional",
        "fdr_bh_by_metric",
        "fdr_bh_by_track_metric",
        "sign_fdr_bh_by_metric",
        "sign_fdr_bh_by_track_metric",
    }
    assert expected_cols.issubset(set(stats.columns))
    assert stats["effect_size_rank_biserial"].between(-1.0, 1.0).all()
    assert stats["effect_size_rank_biserial_directional"].between(-1.0, 1.0).all()
