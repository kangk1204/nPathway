"""Statistical power benchmark for gene program evaluation.

This module evaluates the statistical power of enrichment analysis using
learned gene programs compared to curated pathway databases. It simulates
differential expression by spiking in signal at varying effect sizes into
real background expression data, then measures sensitivity (true positive
rate) and specificity (false positive rate) of enrichment detection.

The benchmark produces ROC-like curves and power curves that compare
different gene set collections under controlled conditions.
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from npathway.evaluation.base import BaseBenchmark
from npathway.evaluation.enrichment import run_enrichment

logger = logging.getLogger(__name__)


class PowerBenchmark(BaseBenchmark):
    """Benchmark evaluating statistical power of enrichment analysis.

    This benchmark takes real expression data as background and spikes in
    differential expression signal at controlled effect sizes (fold changes).
    It then runs enrichment analysis with the provided gene programs and
    measures sensitivity (TPR) and specificity (FPR) at each effect size.

    The simulation approach:
    1. Select a target gene program (the "true positive" pathway).
    2. For each effect size (fold change):
       a. Randomly select cells for a "treatment" group.
       b. Upregulate genes in the target program by the fold change.
       c. Perform DE analysis between treatment and control.
       d. Run enrichment with all gene programs.
       e. Record whether the target program is detected (TP) and how
          many non-target programs are falsely detected (FP).
    3. Repeat across multiple target programs and replicates.

    Attributes:
        fold_changes: List of fold changes to test.
        n_replicates: Number of simulation replicates per condition.
        fdr_threshold: FDR cutoff for significance.
        treatment_fraction: Fraction of cells used as "treatment".
    """

    def __init__(
        self,
        fold_changes: list[float] | None = None,
        n_replicates: int = 20,
        fdr_threshold: float = 0.05,
        treatment_fraction: float = 0.3,
        seed: int = 42,
    ) -> None:
        """Initialize the statistical power benchmark.

        Args:
            fold_changes: List of fold changes to simulate. Defaults to
                biologically realistic range [1.1, 1.3, 1.5, 2.0, 3.0].
            n_replicates: Number of replicates per fold change.
            fdr_threshold: FDR threshold for detecting enrichment.
            treatment_fraction: Fraction of cells used as treatment group.
            seed: Random seed for reproducibility.
        """
        super().__init__(name="StatisticalPower")
        self.fold_changes: list[float] = fold_changes or [1.1, 1.3, 1.5, 2.0, 3.0]
        self.n_replicates: int = n_replicates
        self.fdr_threshold: float = fdr_threshold
        self.treatment_fraction: float = treatment_fraction
        self.seed: int = seed
        self._simulation_results: list[dict[str, Any]] = []

    def run(
        self,
        gene_programs: dict[str, list[str]],
        *,
        expression_matrix: np.ndarray | None = None,
        gene_names: list[str] | None = None,
        target_programs: list[str] | None = None,
        comparison_collections: dict[str, dict[str, list[str]]] | None = None,
        enrichment_method: str = "fisher",
        n_top_de: int = 200,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute the statistical power benchmark.

        Args:
            gene_programs: Primary gene programs to evaluate.
            expression_matrix: Array of shape ``(n_cells, n_genes)`` with
                background expression data. If ``None``, synthetic data
                is generated.
            gene_names: Gene names corresponding to columns of the
                expression matrix.
            target_programs: List of program names to use as targets for
                spike-in. If ``None``, up to 5 programs are selected.
            comparison_collections: Optional dictionary mapping collection
                names to gene program dicts for side-by-side comparison.
            enrichment_method: Enrichment method (``"fisher"`` or ``"gsea"``).
            n_top_de: Number of top DE genes for Fisher enrichment.
            **kwargs: Additional arguments.

        Returns:
            Dictionary with aggregate metrics: mean_tpr_by_fc,
            mean_fpr_by_fc, auc_power_curve.

        Raises:
            ValueError: If required arguments are missing.
        """
        rng = np.random.default_rng(self.seed)

        # Build or validate expression data
        if expression_matrix is None or gene_names is None:
            expression_matrix, gene_names = self._generate_synthetic_data(
                gene_programs, rng
            )

        n_cells, n_genes_total = expression_matrix.shape
        gene_to_idx: dict[str, int] = {g: i for i, g in enumerate(gene_names)}

        # Select target programs
        if target_programs is None:
            eligible = [
                name for name, genes in gene_programs.items()
                if len([g for g in genes if g in gene_to_idx]) >= 5
            ]
            target_programs = eligible[:5] if len(eligible) > 5 else eligible

        if not target_programs:
            raise ValueError(
                "No eligible target programs found. Programs must have at "
                "least 5 genes present in the expression matrix."
            )

        # Build all collections to evaluate
        collections: dict[str, dict[str, list[str]]] = {
            "learned": gene_programs
        }
        if comparison_collections is not None:
            collections.update(comparison_collections)

        self._simulation_results = []

        for target_name in target_programs:
            for fc in self.fold_changes:
                for rep in range(self.n_replicates):
                    for coll_name, coll_programs in collections.items():
                        result = self._simulate_single(
                            expression_matrix=expression_matrix,
                            gene_names=gene_names,
                            gene_to_idx=gene_to_idx,
                            gene_programs=coll_programs,
                            target_program_name=target_name,
                            target_genes=gene_programs[target_name],
                            fold_change=fc,
                            enrichment_method=enrichment_method,
                            n_top_de=n_top_de,
                            rng=rng,
                            collection_name=coll_name,
                            replicate=rep,
                        )
                        self._simulation_results.append(result)

        df = pd.DataFrame(self._simulation_results)

        # Compute aggregate metrics per collection and fold change
        aggregated: dict[str, Any] = {"per_collection": {}}
        for coll_name in collections:
            coll_df = df[df["collection"] == coll_name]
            tpr_by_fc: dict[float, float] = {}
            fpr_by_fc: dict[float, float] = {}
            tpr_ci_by_fc: dict[float, tuple[float, float]] = {}
            fpr_ci_by_fc: dict[float, tuple[float, float]] = {}

            for fc in self.fold_changes:
                fc_df = coll_df[coll_df["fold_change"] == fc]
                if len(fc_df) > 0:
                    tpr_by_fc[fc] = float(fc_df["tp"].mean())
                    fpr_by_fc[fc] = float(fc_df["fpr"].mean())
                    tpr_ci_by_fc[fc] = self._wilson_interval(
                        successes=int(fc_df["tp"].sum()),
                        n=len(fc_df),
                    )
                    coll_seed = self.seed + int(fc * 1000) + sum(ord(ch) for ch in coll_name)
                    fpr_ci_by_fc[fc] = self._bootstrap_mean_ci(
                        values=fc_df["fpr"].to_numpy(dtype=np.float64),
                        seed=coll_seed,
                    )
                else:
                    tpr_by_fc[fc] = 0.0
                    fpr_by_fc[fc] = 0.0
                    tpr_ci_by_fc[fc] = (0.0, 0.0)
                    fpr_ci_by_fc[fc] = (0.0, 0.0)

            # Power curve AUC (trapezoidal rule over fold changes)
            fcs_sorted = sorted(tpr_by_fc.keys())
            tpr_vals = [tpr_by_fc[fc] for fc in fcs_sorted]
            if len(fcs_sorted) > 1:
                trapz_fn = getattr(np, "trapezoid", np.trapz)
                auc = float(trapz_fn(tpr_vals, fcs_sorted))
                # Normalize by range
                fc_range = fcs_sorted[-1] - fcs_sorted[0]
                auc_norm = auc / fc_range if fc_range > 0 else 0.0
            else:
                auc_norm = tpr_vals[0] if tpr_vals else 0.0

            aggregated["per_collection"][coll_name] = {
                "tpr_by_fc": tpr_by_fc,
                "fpr_by_fc": fpr_by_fc,
                "tpr_ci_by_fc": tpr_ci_by_fc,
                "fpr_ci_by_fc": fpr_ci_by_fc,
                "auc_power_curve": auc_norm,
            }

        aggregated["n_simulations"] = len(self._simulation_results)
        aggregated["fold_changes"] = self.fold_changes
        self._store_results(aggregated)
        return aggregated

    def get_results_df(self) -> pd.DataFrame:
        """Return per-simulation results as a DataFrame.

        Returns:
            DataFrame with one row per simulation trial, containing columns:
            target_program, fold_change, replicate, collection, tp, fp,
            n_significant, fpr.

        Raises:
            RuntimeError: If the benchmark has not been run.
        """
        self._check_has_results()
        return pd.DataFrame(self._simulation_results)

    def plot_results(self, save_path: str | None = None) -> plt.Figure:
        """Generate power curve and ROC-like plots.

        Creates a two-panel figure:
        - Left: Power curve (TPR vs. fold change) for each collection.
        - Right: FPR vs. fold change for each collection.

        Args:
            save_path: Optional file path to save the figure.

        Returns:
            matplotlib Figure object.

        Raises:
            RuntimeError: If the benchmark has not been run.
        """
        self._check_has_results()

        per_collection = self.results.get("per_collection", {})
        fold_changes = sorted(self.results.get("fold_changes", []))

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

        # Left panel: TPR (power) curve
        ax1 = axes[0]
        for idx, (coll_name, metrics) in enumerate(per_collection.items()):
            tpr_vals = [metrics["tpr_by_fc"].get(fc, 0.0) for fc in fold_changes]
            color = colors[idx % len(colors)]
            ax1.plot(
                fold_changes, tpr_vals, "o-", label=coll_name,
                color=color, linewidth=2, markersize=6,
            )
        ax1.set_xlabel("Fold Change (Effect Size)")
        ax1.set_ylabel("True Positive Rate (Sensitivity)")
        ax1.set_title("Power Curve")
        ax1.legend()
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True, alpha=0.3)

        # Right panel: FPR curve
        ax2 = axes[1]
        for idx, (coll_name, metrics) in enumerate(per_collection.items()):
            fpr_vals = [metrics["fpr_by_fc"].get(fc, 0.0) for fc in fold_changes]
            color = colors[idx % len(colors)]
            ax2.plot(
                fold_changes, fpr_vals, "s--", label=coll_name,
                color=color, linewidth=2, markersize=6,
            )
        ax2.axhline(
            y=self.fdr_threshold, color="red", linestyle=":",
            label=f"FDR threshold ({self.fdr_threshold})",
        )
        ax2.set_xlabel("Fold Change (Effect Size)")
        ax2.set_ylabel("False Positive Rate")
        ax2.set_title("False Positive Rate vs. Effect Size")
        ax2.legend()
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, alpha=0.3)

        fig.suptitle(
            "Statistical Power Benchmark", fontsize=14, fontweight="bold"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Saved power benchmark plot to %s", save_path)

        return fig

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _simulate_single(
        self,
        expression_matrix: np.ndarray,
        gene_names: list[str],
        gene_to_idx: dict[str, int],
        gene_programs: dict[str, list[str]],
        target_program_name: str,
        target_genes: list[str],
        fold_change: float,
        enrichment_method: str,
        n_top_de: int,
        rng: np.random.Generator,
        collection_name: str,
        replicate: int,
    ) -> dict[str, Any]:
        """Run a single simulation trial.

        Args:
            expression_matrix: Background expression matrix.
            gene_names: Gene name list.
            gene_to_idx: Gene name to column index mapping.
            gene_programs: Programs to test enrichment against.
            target_program_name: Name of the target program (true positive).
            target_genes: Gene list of the target program.
            fold_change: Fold change to apply.
            enrichment_method: Enrichment method.
            n_top_de: Number of top DE genes.
            rng: Random number generator.
            collection_name: Name of the program collection.
            replicate: Replicate index.

        Returns:
            Dictionary with simulation trial results.
        """
        from scipy import stats as scipy_stats

        n_cells = expression_matrix.shape[0]
        n_treatment = max(int(n_cells * self.treatment_fraction), 2)

        # Split cells into treatment and control
        all_indices = np.arange(n_cells)
        rng.shuffle(all_indices)
        treatment_idx = all_indices[:n_treatment]
        control_idx = all_indices[n_treatment:]

        # Create modified expression matrix (spike in signal)
        modified = expression_matrix.copy()
        target_col_indices = [
            gene_to_idx[g] for g in target_genes if g in gene_to_idx
        ]
        if target_col_indices:
            cols = np.array(target_col_indices)
            modified[np.ix_(treatment_idx, cols)] *= fold_change

        # Run DE analysis (Wilcoxon rank-sum for each gene)
        ranked: list[tuple[str, float]] = []
        for g_idx, gene in enumerate(gene_names):
            treatment_vals = modified[treatment_idx, g_idx]
            control_vals = modified[control_idx, g_idx]

            if np.std(treatment_vals) == 0 and np.std(control_vals) == 0:
                ranked.append((gene, 0.0))
                continue

            stat, pval = scipy_stats.ranksums(treatment_vals, control_vals)
            logfc_sign = 1.0 if np.mean(treatment_vals) >= np.mean(control_vals) else -1.0
            score = -np.log10(max(pval, 1e-300)) * logfc_sign
            ranked.append((gene, float(score)))

        ranked.sort(key=lambda x: x[1], reverse=True)

        # Run enrichment
        if enrichment_method == "gsea":
            from npathway.evaluation.enrichment import preranked_gsea
            enrichment_df = preranked_gsea(
                ranked, gene_programs, n_perm=500, seed=int(rng.integers(0, 2**31))
            )
        else:
            top_genes = [g for g, _ in ranked[:n_top_de]]
            enrichment_df = run_enrichment(
                top_genes, gene_programs, method="fisher"
            )

        # Evaluate results
        significant = enrichment_df[enrichment_df["fdr"] < self.fdr_threshold]
        n_significant = len(significant)

        # True positive: target program detected
        tp = target_program_name in significant["program"].values if len(significant) > 0 else False

        # False positives: non-target programs falsely detected
        n_fp = n_significant - (1 if tp else 0)
        n_non_target = len(gene_programs) - 1
        fpr = n_fp / n_non_target if n_non_target > 0 else 0.0

        return {
            "target_program": target_program_name,
            "fold_change": fold_change,
            "replicate": replicate,
            "collection": collection_name,
            "tp": int(tp),
            "fp": n_fp,
            "n_significant": n_significant,
            "fpr": fpr,
        }

    @staticmethod
    def _generate_synthetic_data(
        gene_programs: dict[str, list[str]],
        rng: np.random.Generator,
        n_cells: int = 1000,
        base_mean: float = 5.0,
        noise_scale: float = 1.0,
    ) -> tuple[np.ndarray, list[str]]:
        """Generate realistic synthetic expression data with correlated genes.

        Creates a matrix where genes within the same program share a latent
        factor (simulating co-regulation), producing more biologically
        realistic expression patterns than independent lognormal noise.

        Args:
            gene_programs: Gene programs (for extracting gene names).
            rng: Random number generator.
            n_cells: Number of cells to simulate (default 1000 for power).
            base_mean: Mean of the log-normal baseline.
            noise_scale: Standard deviation of the noise.

        Returns:
            Tuple of (expression_matrix, gene_names).
        """
        all_genes: set[str] = set()
        for genes in gene_programs.values():
            all_genes.update(genes)
        gene_names = sorted(all_genes)
        n_genes = len(gene_names)
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}

        # Base log-means per gene
        log_means = rng.uniform(1.0, base_mean, size=n_genes)

        # Generate correlated expression within programs using latent factors
        expression = np.zeros((n_cells, n_genes), dtype=np.float64)

        # Independent noise for all genes
        for g_idx in range(n_genes):
            expression[:, g_idx] = log_means[g_idx] + noise_scale * rng.standard_normal(n_cells)

        # Add program-level latent factors (intra-program correlation)
        for prog_genes in gene_programs.values():
            prog_factor = 0.5 * rng.standard_normal(n_cells)
            for g in prog_genes:
                if g in gene_to_idx:
                    expression[:, gene_to_idx[g]] += prog_factor

        # Add cell-type structure (3-4 groups with different mean profiles)
        n_types = min(4, max(2, n_cells // 200))
        type_labels = rng.integers(0, n_types, size=n_cells)
        for t in range(n_types):
            mask = type_labels == t
            n_affected = max(1, n_genes // n_types)
            affected = rng.choice(n_genes, size=n_affected, replace=False)
            expression[mask][:, affected] += rng.uniform(0.3, 1.0, size=n_affected)

        # Exponentiate to get lognormal-like counts
        expression = np.exp(expression)

        # Add dropout (zero-inflation typical of scRNA-seq)
        dropout_prob = 0.1
        dropout_mask = rng.random((n_cells, n_genes)) < dropout_prob
        expression[dropout_mask] = 0.0

        logger.info(
            "Generated synthetic expression data: %d cells x %d genes "
            "(with %d programs, %d cell types, %.0f%% dropout).",
            n_cells, n_genes, len(gene_programs), n_types,
            dropout_prob * 100,
        )
        return expression, gene_names

    @staticmethod
    def _wilson_interval(
        successes: int,
        n: int,
        alpha: float = 0.05,
    ) -> tuple[float, float]:
        """Wilson score interval for a Bernoulli mean (used for TPR CI)."""
        if n <= 0:
            return 0.0, 0.0
        from scipy import stats as scipy_stats

        z = float(scipy_stats.norm.ppf(1.0 - alpha / 2.0))
        p = float(successes) / float(n)
        denom = 1.0 + (z * z) / float(n)
        center = (p + (z * z) / (2.0 * float(n))) / denom
        margin = (
            z
            * np.sqrt((p * (1.0 - p) + (z * z) / (4.0 * float(n))) / float(n))
            / denom
        )
        lo = max(0.0, center - margin)
        hi = min(1.0, center + margin)
        return float(lo), float(hi)

    @staticmethod
    def _bootstrap_mean_ci(
        values: np.ndarray,
        seed: int,
        *,
        n_bootstrap: int = 500,
        alpha: float = 0.05,
    ) -> tuple[float, float]:
        """Percentile bootstrap CI for mean values (used for FPR CI)."""
        if len(values) == 0:
            return 0.0, 0.0
        if len(values) == 1:
            val = float(values[0])
            return val, val
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
        boot = values[idx].mean(axis=1)
        lo = float(np.quantile(boot, alpha / 2.0))
        hi = float(np.quantile(boot, 1.0 - alpha / 2.0))
        return lo, hi
