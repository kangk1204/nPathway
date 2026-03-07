"""Perturbation recovery benchmark for gene program evaluation.

This module evaluates gene programs by testing whether enrichment analysis
can recover the expected biological pathway when a known perturbation
(e.g., gene knockout) is applied. It uses Perturb-seq or CRISPR-screen
data stored in AnnData format, performs differential expression analysis
for each perturbation, and measures how well enrichment analysis with the
learned programs identifies the correct pathway.
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from npathway.evaluation.base import BaseBenchmark
from npathway.evaluation.enrichment import preranked_gsea, run_enrichment

logger = logging.getLogger(__name__)


class PerturbationBenchmark(BaseBenchmark):
    """Benchmark that measures recovery of known biology from perturbation data.

    Given Perturb-seq data where each cell has been subjected to a known
    genetic perturbation (e.g., CRISPR knockout), this benchmark:

    1. Identifies differentially expressed (DE) genes for each perturbation
       vs. control cells using the Wilcoxon rank-sum test.
    2. Runs enrichment analysis (Fisher or GSEA) with the provided gene
       programs.
    3. Evaluates whether the expected pathway/program is recovered among
       the top-ranked enrichment results.

    Metrics reported:
    - **Recovery rate**: Fraction of perturbations where the expected
      pathway is significant (FDR < threshold).
    - **Mean rank**: Average rank of the expected pathway across all
      perturbations.
    - **Mean reciprocal rank (MRR)**: Average of 1/rank for the expected
      pathway.
    - **Area under recall curve (AURC)**: Integration of cumulative
      recall as a function of the rank cutoff.

    Attributes:
        perturbation_key: Column in ``adata.obs`` containing perturbation
            labels.
        control_label: Label identifying control (non-perturbed) cells.
        fdr_threshold: FDR cutoff for declaring significant enrichment.
    """

    def __init__(
        self,
        perturbation_key: str = "perturbation",
        control_label: str = "control",
        fdr_threshold: float = 0.05,
    ) -> None:
        """Initialize the perturbation recovery benchmark.

        Args:
            perturbation_key: Column in ``adata.obs`` that stores the
                perturbation label for each cell.
            control_label: The label value in ``perturbation_key`` that
                identifies unperturbed (control) cells.
            fdr_threshold: FDR threshold below which a pathway is considered
                recovered.
        """
        super().__init__(name="PerturbationRecovery")
        self.perturbation_key: str = perturbation_key
        self.control_label: str = control_label
        self.fdr_threshold: float = fdr_threshold
        self._per_perturbation_results: list[dict[str, Any]] = []

    def run(
        self,
        gene_programs: dict[str, list[str]],
        *,
        adata: Any | None = None,
        perturbation_to_pathway: dict[str, str] | None = None,
        enrichment_method: str = "fisher",
        n_top_de: int = 200,
        de_results: dict[str, list[tuple[str, float]]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute the perturbation recovery benchmark.

        There are two modes of operation:

        1. **Full mode**: Provide ``adata`` and ``perturbation_to_pathway``.
           DE analysis is performed internally.
        2. **Pre-computed mode**: Provide ``de_results`` and
           ``perturbation_to_pathway``. Skip the DE step.

        Args:
            gene_programs: Gene programs to evaluate (program name -> gene list).
            adata: AnnData object in Perturb-seq format. Required for full mode.
            perturbation_to_pathway: Mapping from perturbation name to the
                expected pathway/program name in ``gene_programs``.
            enrichment_method: Enrichment method (``"fisher"`` or ``"gsea"``).
            n_top_de: Number of top DE genes to use for Fisher enrichment.
            de_results: Pre-computed DE results as a dict mapping perturbation
                name to a ranked list of (gene, score) tuples.
            **kwargs: Additional keyword arguments passed to the enrichment
                function.

        Returns:
            Dictionary containing aggregated metrics: recovery_rate,
            mean_rank, mrr, aurc.

        Raises:
            ValueError: If required arguments are missing.
        """
        if perturbation_to_pathway is None:
            raise ValueError(
                "perturbation_to_pathway must be provided: a dict mapping "
                "perturbation names to expected pathway/program names."
            )

        if de_results is None:
            if adata is None:
                raise ValueError(
                    "Either adata (for full DE analysis) or de_results "
                    "(pre-computed) must be provided."
                )
            de_results = self._run_de_analysis(adata)

        self._per_perturbation_results = []
        n_programs = len(gene_programs)

        for pert_name, expected_pathway in perturbation_to_pathway.items():
            if pert_name not in de_results:
                logger.warning(
                    "Perturbation '%s' not found in DE results; skipping.",
                    pert_name,
                )
                continue

            if expected_pathway not in gene_programs:
                logger.warning(
                    "Expected pathway '%s' for perturbation '%s' not found "
                    "in gene programs; skipping.",
                    expected_pathway,
                    pert_name,
                )
                continue

            ranked_genes = de_results[pert_name]

            # Run enrichment
            if enrichment_method == "gsea":
                enrichment_df = preranked_gsea(
                    ranked_genes, gene_programs, **kwargs
                )
                enrichment_df = enrichment_df.sort_values(
                    "p_value"
                ).reset_index(drop=True)
            else:
                # Fisher: use top N DE genes
                top_genes = [g for g, _ in ranked_genes[:n_top_de]]
                enrichment_df = run_enrichment(
                    top_genes, gene_programs, method="fisher", **kwargs
                )

            # Find the rank of the expected pathway
            enrichment_df = enrichment_df.reset_index(drop=True)
            expected_rows = enrichment_df[
                enrichment_df["program"] == expected_pathway
            ]

            if expected_rows.empty:
                rank = n_programs
                p_val = 1.0
                fdr_val = 1.0
                recovered = False
            else:
                row_idx = expected_rows.index[0]
                rank = row_idx + 1  # 1-based rank
                p_val = float(expected_rows.iloc[0]["p_value"])
                fdr_val = float(expected_rows.iloc[0]["fdr"])
                recovered = fdr_val < self.fdr_threshold

            self._per_perturbation_results.append(
                {
                    "perturbation": pert_name,
                    "expected_pathway": expected_pathway,
                    "rank": rank,
                    "p_value": p_val,
                    "fdr": fdr_val,
                    "recovered": recovered,
                    "n_programs": n_programs,
                }
            )

        # Compute aggregate metrics
        if not self._per_perturbation_results:
            aggregated = {
                "recovery_rate": 0.0,
                "mean_rank": float("nan"),
                "mrr": 0.0,
                "aurc": 0.0,
                "n_perturbations": 0,
            }
        else:
            ranks = [r["rank"] for r in self._per_perturbation_results]
            recovered_flags = [
                r["recovered"] for r in self._per_perturbation_results
            ]

            recovery_rate = sum(recovered_flags) / len(recovered_flags)
            mean_rank = float(np.mean(ranks))
            mrr = float(np.mean([1.0 / r for r in ranks]))
            aurc = self._compute_aurc(ranks, n_programs)

            aggregated = {
                "recovery_rate": recovery_rate,
                "mean_rank": mean_rank,
                "mrr": mrr,
                "aurc": aurc,
                "n_perturbations": len(self._per_perturbation_results),
            }

        self._store_results(aggregated)
        return aggregated

    def get_results_df(self) -> pd.DataFrame:
        """Return per-perturbation results as a DataFrame.

        Returns:
            DataFrame with one row per perturbation, including columns for
            perturbation name, expected pathway, rank, p-value, FDR, and
            whether the pathway was recovered.

        Raises:
            RuntimeError: If the benchmark has not been run yet.
        """
        self._check_has_results()
        return pd.DataFrame(self._per_perturbation_results)

    def plot_results(self, save_path: str | None = None) -> plt.Figure:
        """Generate visualizations of perturbation recovery results.

        Creates a two-panel figure:
        - Left: Histogram of expected pathway ranks across perturbations.
        - Right: Bar chart of aggregate metrics (recovery rate, MRR, AURC).

        Args:
            save_path: Optional file path to save the figure.

        Returns:
            matplotlib Figure object.

        Raises:
            RuntimeError: If the benchmark has not been run yet.
        """
        self._check_has_results()
        df = self.get_results_df()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left panel: rank distribution
        ax1 = axes[0]
        if df.empty or "rank" not in df.columns:
            ax1.text(
                0.5,
                0.5,
                "No valid perturbation recoveries",
                transform=ax1.transAxes,
                ha="center",
                va="center",
            )
            ax1.set_axis_off()
        else:
            max_rank = max(int(df["rank"].max()), 1)
            bins = min(max_rank, 20)
            ax1.hist(df["rank"].values, bins=bins, color="#2196F3", edgecolor="white", alpha=0.85)
            ax1.set_xlabel("Rank of Expected Pathway")
            ax1.set_ylabel("Count")
            ax1.set_title("Distribution of Expected Pathway Ranks")
            ax1.axvline(x=1, color="red", linestyle="--", linewidth=1.5, label="Rank 1")
            ax1.legend()

        # Right panel: aggregate metrics
        ax2 = axes[1]
        metrics = {
            "Recovery\nRate": self.results["recovery_rate"],
            "MRR": self.results["mrr"],
            "AURC": self.results["aurc"],
        }
        bars = ax2.bar(
            metrics.keys(),
            metrics.values(),
            color=["#4CAF50", "#FF9800", "#9C27B0"],
            edgecolor="white",
            width=0.5,
        )
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("Score")
        ax2.set_title("Aggregate Recovery Metrics")
        for bar, val in zip(bars, metrics.values()):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=11,
            )

        fig.suptitle("Perturbation Recovery Benchmark", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Saved perturbation benchmark plot to %s", save_path)

        return fig

    def compare_collections(
        self,
        program_collections: dict[str, dict[str, list[str]]],
        adata: Any | None = None,
        perturbation_to_pathway: dict[str, str] | None = None,
        de_results: dict[str, list[tuple[str, float]]] | None = None,
        enrichment_method: str = "fisher",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compare multiple gene program collections on the same perturbation data.

        Args:
            program_collections: Dict mapping collection names to gene program
                dictionaries.
            adata: AnnData with perturbation data.
            perturbation_to_pathway: Mapping from perturbation to expected
                pathway. The pathway names should exist in all collections.
            de_results: Pre-computed DE results.
            enrichment_method: Enrichment method to use.
            **kwargs: Extra arguments for enrichment.

        Returns:
            DataFrame combining results from all collections.
        """
        all_results: list[pd.DataFrame] = []
        for collection_name, programs in program_collections.items():
            self.run(
                programs,
                adata=adata,
                perturbation_to_pathway=perturbation_to_pathway,
                de_results=de_results,
                enrichment_method=enrichment_method,
                **kwargs,
            )
            df = self.get_results_df()
            df["collection"] = collection_name
            df["recovery_rate"] = self.results["recovery_rate"]
            df["mrr"] = self.results["mrr"]
            df["aurc"] = self.results["aurc"]
            all_results.append(df)
        return pd.concat(all_results, ignore_index=True)

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _run_de_analysis(
        self,
        adata: Any,
    ) -> dict[str, list[tuple[str, float]]]:
        """Run Wilcoxon rank-sum DE analysis for each perturbation vs. control.

        Args:
            adata: AnnData object with perturbation annotations.

        Returns:
            Dictionary mapping perturbation names to ranked gene lists
            (gene_name, -log10_pvalue * sign_of_logfc) sorted descending.
        """
        import warnings

        obs = adata.obs
        if self.perturbation_key not in obs.columns:
            raise ValueError(
                f"Perturbation key '{self.perturbation_key}' not found in "
                f"adata.obs columns: {list(obs.columns)}"
            )

        perturbations = [
            p for p in obs[self.perturbation_key].unique()
            if p != self.control_label
        ]
        control_mask = obs[self.perturbation_key] == self.control_label

        # Get expression matrix (dense)
        try:
            X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
        except Exception:
            X = np.array(adata.X)

        gene_names = list(adata.var_names)
        control_expr = X[control_mask.values, :]

        de_results: dict[str, list[tuple[str, float]]] = {}

        for pert in perturbations:
            pert_mask = obs[self.perturbation_key] == pert
            pert_expr = X[pert_mask.values, :]

            if pert_expr.shape[0] < 3:
                logger.warning(
                    "Perturbation '%s' has fewer than 3 cells; skipping.", pert
                )
                continue

            ranked: list[tuple[str, float]] = []
            for g_idx, gene in enumerate(gene_names):
                ctrl_vals = control_expr[:, g_idx]
                pert_vals = pert_expr[:, g_idx]

                # Skip genes with zero variance
                if np.std(ctrl_vals) == 0 and np.std(pert_vals) == 0:
                    ranked.append((gene, 0.0))
                    continue

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stat, pval = scipy_stats.ranksums(pert_vals, ctrl_vals)

                # Compute log fold-change direction
                mean_pert = np.mean(pert_vals)
                mean_ctrl = np.mean(ctrl_vals)
                logfc_sign = 1.0 if mean_pert >= mean_ctrl else -1.0

                # Score: -log10(p) * sign(logFC)
                score = -np.log10(max(pval, 1e-300)) * logfc_sign
                ranked.append((gene, float(score)))

            # Sort by score descending
            ranked.sort(key=lambda x: x[1], reverse=True)
            de_results[pert] = ranked

        logger.info(
            "Completed DE analysis for %d perturbations.", len(de_results)
        )
        return de_results

    @staticmethod
    def _compute_aurc(ranks: list[int], n_programs: int) -> float:
        """Compute the Area Under the Recall Curve.

        The recall at cutoff k is the fraction of perturbations where the
        expected pathway is ranked at or above k.

        Args:
            ranks: List of ranks (1-based) for the expected pathway.
            n_programs: Total number of programs.

        Returns:
            AURC value normalized to [0, 1].
        """
        if not ranks or n_programs == 0:
            return 0.0

        n_total = len(ranks)
        rank_arr = np.array(ranks)
        recall_values: list[float] = []

        for k in range(1, n_programs + 1):
            recall = np.sum(rank_arr <= k) / n_total
            recall_values.append(recall)

        # Normalize: perfect AURC = 1.0 (everything at rank 1)
        aurc = float(np.mean(recall_values))
        return aurc
