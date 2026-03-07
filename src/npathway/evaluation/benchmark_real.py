"""Real-data benchmark for gene program evaluation.

This module implements benchmarks that use publicly available scRNA-seq
datasets (e.g., PBMC 3k) and curated pathway databases (MSigDB) to
evaluate gene program discovery methods under realistic conditions.

Two benchmark classes are provided:

* :class:`RealDataBenchmark` -- end-to-end evaluation that downloads
  data, preprocesses it, runs multiple discovery methods, downloads
  MSigDB reference gene sets, and evaluates pathway recovery, program
  coherence, gene coverage, redundancy, and novelty.

* :class:`CellTypeMarkerBenchmark` -- uses cell-type annotations as
  ground-truth gene programs and measures how well discovered programs
  recover cell-type-specific marker genes via ARI, NMI, and Jaccard.
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from npathway.evaluation.base import BaseBenchmark
from npathway.evaluation.metrics import (
    adjusted_rand_index,
    coverage,
    jaccard_similarity,
    normalized_mutual_info,
    novelty_score,
    program_redundancy,
    programs_to_labels,
)

logger = logging.getLogger(__name__)


# ======================================================================
# Helper: best-match pathway recovery
# ======================================================================


def _pathway_recovery(
    discovered: dict[str, list[str]],
    reference: dict[str, list[str]],
    jaccard_threshold: float = 0.1,
) -> dict[str, Any]:
    """Compute how well discovered programs recover reference pathways.

    For each reference pathway, find the discovered program with the
    highest Jaccard similarity.  A pathway is considered "recovered" if
    the best-match Jaccard exceeds *jaccard_threshold*.

    Args:
        discovered: Discovered gene programs.
        reference: Reference pathways (e.g., MSigDB Hallmark).
        jaccard_threshold: Minimum Jaccard for a pathway to be
            considered recovered.

    Returns:
        Dictionary with recovery metrics:
        - ``recovery_rate``: Fraction of reference pathways recovered.
        - ``mean_best_jaccard``: Mean best-match Jaccard across pathways.
        - ``per_pathway``: List of per-pathway detail dicts.
    """
    per_pathway: list[dict[str, Any]] = []

    for ref_name, ref_genes in reference.items():
        ref_set = set(ref_genes)
        best_jaccard = 0.0
        best_match = ""

        for disc_name, disc_genes in discovered.items():
            jac = jaccard_similarity(ref_set, set(disc_genes))
            if jac > best_jaccard:
                best_jaccard = jac
                best_match = disc_name

        per_pathway.append({
            "reference_pathway": ref_name,
            "best_match_program": best_match,
            "best_jaccard": best_jaccard,
            "recovered": best_jaccard >= jaccard_threshold,
        })

    n_recovered = sum(1 for p in per_pathway if p["recovered"])
    n_total = len(per_pathway)
    recovery_rate = n_recovered / n_total if n_total > 0 else 0.0
    mean_best_jaccard = float(
        np.mean([p["best_jaccard"] for p in per_pathway])
    ) if per_pathway else 0.0

    return {
        "recovery_rate": recovery_rate,
        "mean_best_jaccard": mean_best_jaccard,
        "n_recovered": n_recovered,
        "n_total": n_total,
        "per_pathway": per_pathway,
    }


# ======================================================================
# RealDataBenchmark
# ======================================================================


class RealDataBenchmark(BaseBenchmark):
    """End-to-end benchmark using real scRNA-seq data and MSigDB references.

    This benchmark:

    1. Downloads the PBMC 3k dataset.
    2. Preprocesses it (filter, normalize, HVGs, PCA).
    3. Builds expression-based gene embeddings (PCA on gene profiles).
    4. Runs all available discovery methods on the embeddings.
    5. Downloads MSigDB Hallmark gene sets as ground-truth reference.
    6. Evaluates discovered programs on multiple metrics:
       - **Pathway recovery**: How many Hallmark pathways are recovered.
       - **Program coherence**: Gene-gene co-expression within programs.
       - **Gene coverage**: Fraction of genes assigned to programs.
       - **Redundancy**: Mean pairwise Jaccard between programs.
       - **Novelty**: Fraction of gene assignments not in reference.

    Attributes:
        n_programs: Number of programs for methods that require it.
        top_n_genes: Number of top genes per program.
        recovery_threshold: Jaccard threshold for pathway recovery.
    """

    def __init__(
        self,
        n_programs: int = 20,
        top_n_genes: int = 50,
        recovery_threshold: float = 0.1,
        seed: int = 42,
    ) -> None:
        """Initialize the real-data benchmark.

        Args:
            n_programs: Number of programs for parametric methods.
            top_n_genes: Number of top genes per program.
            recovery_threshold: Jaccard threshold for pathway recovery.
            seed: Random seed for reproducibility.
        """
        super().__init__(name="RealDataBenchmark")
        self.n_programs = n_programs
        self.top_n_genes = top_n_genes
        self.recovery_threshold = recovery_threshold
        self.seed = seed
        self._method_results: list[dict[str, Any]] = []

    def run(
        self,
        gene_programs: dict[str, list[str]],
        *,
        adata: Any | None = None,
        gene_embeddings: np.ndarray | None = None,
        gene_names: list[str] | None = None,
        reference_gene_sets: dict[str, list[str]] | None = None,
        run_discovery: bool = True,
        additional_methods: dict[str, dict[str, list[str]]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute the real-data benchmark.

        When called with *gene_programs* only, the benchmark evaluates
        those programs against the downloaded reference.  When
        *run_discovery* is True, the benchmark additionally runs all
        built-in discovery methods and compares them.

        Args:
            gene_programs: Pre-computed gene programs to evaluate.
                Also used if *run_discovery* is ``False``.
            adata: Optional pre-loaded AnnData.  If ``None``, PBMC 3k
                is downloaded and preprocessed.
            gene_embeddings: Optional pre-computed gene embeddings.
                If ``None`` and *run_discovery* is True, PCA-based
                embeddings are built from expression data.
            gene_names: Gene names aligned with *gene_embeddings*.
            reference_gene_sets: Optional pre-loaded reference pathways.
                If ``None``, MSigDB Hallmark sets are downloaded.
            run_discovery: If ``True``, run all discovery methods.
            additional_methods: Extra method results to include, mapping
                method names to their discovered programs.
            **kwargs: Additional keyword arguments.

        Returns:
            Dictionary containing per-method results with all metrics.
        """

        # --- Step 1: Load data ---
        if adata is None:
            adata = self._load_and_preprocess_data()

        # --- Step 2: Build embeddings ---
        if gene_embeddings is None or gene_names is None:
            from npathway.data.preprocessing import (
                build_gene_embeddings_from_expression,
            )
            gene_embeddings, gene_names = build_gene_embeddings_from_expression(
                adata, n_components=min(50, adata.n_vars - 1)
            )

        # --- Step 3: Load reference gene sets ---
        if reference_gene_sets is None:
            reference_gene_sets = self._load_reference_gene_sets(adata)

        # --- Step 4: Collect all method results ---
        all_methods: dict[str, dict[str, list[str]]] = {
            "provided": gene_programs,
        }

        if run_discovery:
            discovery_results = self._run_discovery_methods(
                gene_embeddings, gene_names, adata
            )
            all_methods.update(discovery_results)

        if additional_methods is not None:
            all_methods.update(additional_methods)

        # --- Step 5: Evaluate all methods ---
        self._method_results = []
        all_adata_genes = list(adata.var_names)

        for method_name, programs in all_methods.items():
            metrics = self._evaluate_programs(
                programs=programs,
                reference=reference_gene_sets,
                gene_universe=all_adata_genes,
            )
            metrics["method"] = method_name
            metrics["n_programs"] = len(programs)
            metrics["mean_program_size"] = float(
                np.mean([len(g) for g in programs.values()])
            ) if programs else 0.0
            self._method_results.append(metrics)

            logger.info(
                "Method '%s': recovery=%.3f, redundancy=%.3f, "
                "coverage=%.3f, novelty=%.3f",
                method_name,
                metrics["recovery_rate"],
                metrics["redundancy"],
                metrics["coverage"],
                metrics["novelty"],
            )

        # Aggregate
        results_df = pd.DataFrame(self._method_results)
        aggregated: dict[str, Any] = {
            "per_method": self._method_results,
            "results_df": results_df.to_dict(orient="records"),
            "n_methods": len(all_methods),
            "n_reference_pathways": len(reference_gene_sets),
            "n_genes_in_dataset": len(all_adata_genes),
        }
        self._store_results(aggregated)
        return aggregated

    def get_results_df(self) -> pd.DataFrame:
        """Return benchmark results as a tidy DataFrame.

        Returns:
            DataFrame with one row per method and columns for all metrics.

        Raises:
            RuntimeError: If the benchmark has not been run.
        """
        self._check_has_results()
        return pd.DataFrame(self._method_results)

    def plot_results(self, save_path: str | None = None) -> plt.Figure:
        """Generate a multi-panel comparison of all methods.

        Creates a figure with grouped bar charts comparing methods on
        pathway recovery, redundancy, coverage, and novelty.

        Args:
            save_path: Optional path to save the figure.

        Returns:
            matplotlib Figure.

        Raises:
            RuntimeError: If the benchmark has not been run.
        """
        self._check_has_results()
        df = self.get_results_df()

        metrics_to_plot = [
            ("recovery_rate", "Pathway Recovery Rate"),
            ("mean_best_jaccard", "Mean Best Jaccard"),
            ("redundancy", "Program Redundancy"),
            ("coverage", "Gene Coverage"),
            ("novelty", "Novelty Score"),
        ]

        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]

        colors = plt.cm.Set2.colors  # type: ignore[attr-defined]
        method_names = df["method"].tolist()

        for ax, (metric_key, metric_label) in zip(axes, metrics_to_plot):
            if metric_key not in df.columns:
                ax.set_visible(False)
                continue

            values = df[metric_key].values
            x = np.arange(len(method_names))
            bars = ax.bar(
                x,
                values,
                color=[colors[i % len(colors)] for i in range(len(method_names))],
                edgecolor="white",
                linewidth=0.5,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(method_names, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel(metric_label)
            ax.set_title(metric_label, fontsize=10)
            ax.set_ylim(0, max(1.0, values.max() * 1.1) if len(values) > 0 else 1.0)
            ax.grid(True, alpha=0.3, axis="y")

            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

        fig.suptitle(
            "Real-Data Benchmark: Method Comparison",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Saved real-data benchmark plot to %s", save_path)

        return fig

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    @staticmethod
    def _load_and_preprocess_data() -> Any:
        """Download and preprocess PBMC 3k data.

        Returns:
            Preprocessed AnnData.
        """
        from npathway.data.datasets import load_pbmc3k

        logger.info("Loading and preprocessing PBMC 3k dataset ...")
        adata = load_pbmc3k(preprocessed=True)

        # The processed version already has PCA, neighbors, etc.
        # Ensure var_names are gene symbols
        if hasattr(adata, "raw") and adata.raw is not None:
            logger.info("PBMC 3k loaded (pre-processed version).")
        return adata

    @staticmethod
    def _load_reference_gene_sets(adata: Any) -> dict[str, list[str]]:
        """Download Hallmark gene sets and filter to genes in the dataset.

        Args:
            adata: AnnData for gene filtering.

        Returns:
            Filtered Hallmark gene-set dictionary.
        """
        from npathway.data.datasets import (
            filter_gene_sets_to_adata,
            load_msigdb_gene_sets,
        )

        logger.info("Loading MSigDB Hallmark gene sets ...")
        hallmark = load_msigdb_gene_sets(collection="hallmark")
        filtered = filter_gene_sets_to_adata(hallmark, adata, min_genes=3)
        logger.info(
            "Hallmark reference: %d gene sets after filtering (%d before).",
            len(filtered),
            len(hallmark),
        )
        return filtered

    def _run_discovery_methods(
        self,
        gene_embeddings: np.ndarray,
        gene_names: list[str],
        adata: Any,
    ) -> dict[str, dict[str, list[str]]]:
        """Run all discovery methods and return their programs.

        Args:
            gene_embeddings: Gene embedding matrix.
            gene_names: Gene names.
            adata: AnnData for expression-based methods.

        Returns:
            Dictionary mapping method names to their discovered programs.
        """
        from npathway.data.preprocessing import _safe_toarray

        results: dict[str, dict[str, list[str]]] = {}

        # 1. Clustering (k-means on embeddings)
        try:
            from npathway.discovery.clustering import ClusteringProgramDiscovery

            km = ClusteringProgramDiscovery(
                method="kmeans",
                n_programs=self.n_programs,
                random_state=self.seed,
            )
            km.fit(gene_embeddings, gene_names)
            results["clustering_kmeans"] = km.get_programs()
            logger.info("Clustering (k-means): %d programs.", len(results["clustering_kmeans"]))
        except Exception as exc:
            logger.warning("Clustering (k-means) failed: %s", exc)

        # 2. ETM (topic model)
        try:
            from npathway.discovery.topic_model import TopicModelProgramDiscovery

            X_expr = _safe_toarray(adata.X)
            etm = TopicModelProgramDiscovery(
                n_topics=self.n_programs,
                n_epochs=50,
                top_n_genes=self.top_n_genes,
                device="cpu",
                random_state=self.seed,
            )
            etm.fit(
                gene_embeddings,
                gene_names,
                expression_matrix=X_expr,
            )
            results["etm"] = etm.get_programs()
            logger.info("ETM: %d programs.", len(results["etm"]))
        except Exception as exc:
            logger.warning("ETM failed: %s", exc)

        # 3. WGCNA-like
        try:
            from npathway.discovery.baselines import WGCNAProgramDiscovery

            X_expr = _safe_toarray(adata.X)
            wgcna = WGCNAProgramDiscovery(
                n_programs=self.n_programs,
                min_module_size=5,
                random_state=self.seed,
            )
            wgcna.fit(X_expr, gene_names)
            results["wgcna"] = wgcna.get_programs()
            logger.info("WGCNA: %d programs.", len(results["wgcna"]))
        except Exception as exc:
            logger.warning("WGCNA failed: %s", exc)

        # 4. cNMF-like
        try:
            from npathway.discovery.baselines import CNMFProgramDiscovery

            X_expr = _safe_toarray(adata.X)
            # Shift to non-negative for NMF
            X_nn = X_expr - X_expr.min() + 1e-6
            cnmf = CNMFProgramDiscovery(
                n_programs=self.n_programs,
                n_iter=3,
                top_n_genes=self.top_n_genes,
                random_state=self.seed,
            )
            cnmf.fit(X_nn, gene_names)
            results["cnmf"] = cnmf.get_programs()
            logger.info("cNMF: %d programs.", len(results["cnmf"]))
        except Exception as exc:
            logger.warning("cNMF failed: %s", exc)

        # 5. Expression clustering baseline
        try:
            from npathway.discovery.baselines import ExpressionClusteringBaseline

            X_expr = _safe_toarray(adata.X)
            expr_clust = ExpressionClusteringBaseline(
                n_programs=self.n_programs,
                random_state=self.seed,
            )
            expr_clust.fit(X_expr, gene_names)
            results["expression_clustering"] = expr_clust.get_programs()
            logger.info(
                "Expression clustering: %d programs.",
                len(results["expression_clustering"]),
            )
        except Exception as exc:
            logger.warning("Expression clustering failed: %s", exc)

        # 6. Random baseline
        try:
            from npathway.discovery.baselines import RandomProgramDiscovery

            random_disc = RandomProgramDiscovery(
                n_programs=self.n_programs,
                genes_per_program=self.top_n_genes,
                random_state=self.seed,
            )
            random_disc.fit(gene_embeddings, gene_names)
            results["random"] = random_disc.get_programs()
            logger.info("Random baseline: %d programs.", len(results["random"]))
        except Exception as exc:
            logger.warning("Random baseline failed: %s", exc)

        return results

    def _evaluate_programs(
        self,
        programs: dict[str, list[str]],
        reference: dict[str, list[str]],
        gene_universe: list[str],
    ) -> dict[str, Any]:
        """Evaluate a set of programs against the reference.

        Args:
            programs: Discovered gene programs.
            reference: Reference pathways.
            gene_universe: All genes in the dataset.

        Returns:
            Dictionary of evaluation metrics.
        """
        # Pathway recovery
        recovery = _pathway_recovery(
            programs, reference, jaccard_threshold=self.recovery_threshold
        )

        # Coherence (co-expression-based): mean intra-program pairwise Jaccard
        # as a proxy when PPI is not available
        intra_jaccards: list[float] = []
        program_list = list(programs.values())
        for pg in program_list:
            if len(pg) < 2:
                continue
            # Compare to each reference pathway and take the max
            best_jac = 0.0
            for ref_genes in reference.values():
                jac = jaccard_similarity(set(pg), set(ref_genes))
                if jac > best_jac:
                    best_jac = jac
            intra_jaccards.append(best_jac)
        mean_coherence = float(np.mean(intra_jaccards)) if intra_jaccards else 0.0

        # Coverage
        cov = coverage(programs, gene_universe)

        # Redundancy
        redundancy = program_redundancy(programs)

        # Novelty
        nov = novelty_score(programs, reference)

        return {
            "recovery_rate": recovery["recovery_rate"],
            "mean_best_jaccard": recovery["mean_best_jaccard"],
            "n_recovered": recovery["n_recovered"],
            "n_reference": recovery["n_total"],
            "coherence": mean_coherence,
            "coverage": cov,
            "redundancy": redundancy,
            "novelty": nov,
        }


# ======================================================================
# CellTypeMarkerBenchmark
# ======================================================================


class CellTypeMarkerBenchmark(BaseBenchmark):
    """Benchmark using cell-type markers as ground-truth gene programs.

    This benchmark extracts marker genes per cell type from cell-type
    annotations in the dataset, treats them as ground-truth programs,
    and evaluates how well the discovered programs recover these
    markers.

    Metrics:
    - **ARI** (Adjusted Rand Index) between discovered and marker labels.
    - **NMI** (Normalized Mutual Information) between label assignments.
    - **Mean Jaccard**: Average best-match Jaccard between discovered
      programs and marker gene sets.
    - **Per-cell-type recovery**: Fraction of marker sets recovered
      above a Jaccard threshold.

    Attributes:
        groupby: Column in ``adata.obs`` for cell-type labels.
        n_marker_genes: Number of marker genes per cell type.
        recovery_threshold: Jaccard threshold for marker recovery.
    """

    def __init__(
        self,
        groupby: str = "louvain",
        n_marker_genes: int = 50,
        recovery_threshold: float = 0.1,
    ) -> None:
        """Initialize the cell-type marker benchmark.

        Args:
            groupby: Column in ``adata.obs`` for cell-type labels.
            n_marker_genes: Number of marker genes to extract per type.
            recovery_threshold: Jaccard threshold for considering a
                marker gene set recovered.
        """
        super().__init__(name="CellTypeMarkerBenchmark")
        self.groupby = groupby
        self.n_marker_genes = n_marker_genes
        self.recovery_threshold = recovery_threshold
        self._per_celltype_results: list[dict[str, Any]] = []

    def run(
        self,
        gene_programs: dict[str, list[str]],
        *,
        adata: Any | None = None,
        marker_programs: dict[str, list[str]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute the cell-type marker benchmark.

        Args:
            gene_programs: Discovered gene programs to evaluate.
            adata: AnnData with cell-type annotations.  Required if
                *marker_programs* is not provided.
            marker_programs: Optional pre-computed marker gene programs.
                If ``None``, markers are extracted from *adata* using
                differential expression.
            **kwargs: Additional keyword arguments.

        Returns:
            Dictionary with ARI, NMI, mean Jaccard, recovery rate,
            and per-cell-type details.

        Raises:
            ValueError: If neither *adata* nor *marker_programs* is provided.
        """
        if marker_programs is None:
            if adata is None:
                raise ValueError(
                    "Either 'adata' or 'marker_programs' must be provided."
                )
            from npathway.data.preprocessing import extract_cell_type_markers

            marker_programs = extract_cell_type_markers(
                adata,
                groupby=self.groupby,
                n_genes=self.n_marker_genes,
            )

        logger.info(
            "Evaluating %d programs against %d cell-type marker sets.",
            len(gene_programs),
            len(marker_programs),
        )

        # Build gene universe
        all_genes: set[str] = set()
        for genes in gene_programs.values():
            all_genes.update(genes)
        for genes in marker_programs.values():
            all_genes.update(genes)
        gene_universe = sorted(all_genes)

        # Convert to label vectors for ARI/NMI
        labels_discovered = programs_to_labels(gene_programs, gene_universe)
        labels_markers = programs_to_labels(marker_programs, gene_universe)

        # Only consider genes assigned in both
        assigned_mask = (labels_discovered >= 0) & (labels_markers >= 0)
        n_assigned = int(assigned_mask.sum())

        if n_assigned > 0:
            ari = adjusted_rand_index(
                labels_discovered[assigned_mask],
                labels_markers[assigned_mask],
            )
            nmi = normalized_mutual_info(
                labels_discovered[assigned_mask],
                labels_markers[assigned_mask],
            )
        else:
            ari = 0.0
            nmi = 0.0

        # Per-cell-type Jaccard recovery
        self._per_celltype_results = []
        for ct_name, ct_genes in marker_programs.items():
            ct_set = set(ct_genes)
            best_jaccard = 0.0
            best_match = ""

            for prog_name, prog_genes in gene_programs.items():
                jac = jaccard_similarity(ct_set, set(prog_genes))
                if jac > best_jaccard:
                    best_jaccard = jac
                    best_match = prog_name

            self._per_celltype_results.append({
                "cell_type": ct_name,
                "best_match_program": best_match,
                "best_jaccard": best_jaccard,
                "recovered": best_jaccard >= self.recovery_threshold,
                "n_markers": len(ct_genes),
            })

        n_recovered = sum(
            1 for r in self._per_celltype_results if r["recovered"]
        )
        n_total = len(self._per_celltype_results)
        recovery_rate = n_recovered / n_total if n_total > 0 else 0.0
        mean_jaccard = float(
            np.mean([r["best_jaccard"] for r in self._per_celltype_results])
        ) if self._per_celltype_results else 0.0

        aggregated: dict[str, Any] = {
            "ari": ari,
            "nmi": nmi,
            "mean_best_jaccard": mean_jaccard,
            "recovery_rate": recovery_rate,
            "n_recovered": n_recovered,
            "n_cell_types": n_total,
            "n_genes_assigned_both": n_assigned,
            "per_celltype": self._per_celltype_results,
        }
        self._store_results(aggregated)
        return aggregated

    def get_results_df(self) -> pd.DataFrame:
        """Return per-cell-type results as a DataFrame.

        Returns:
            DataFrame with one row per cell type.

        Raises:
            RuntimeError: If the benchmark has not been run.
        """
        self._check_has_results()
        return pd.DataFrame(self._per_celltype_results)

    def plot_results(self, save_path: str | None = None) -> plt.Figure:
        """Generate a summary visualization.

        Creates a two-panel figure:
        - Left: Bar chart of best-match Jaccard per cell type.
        - Right: Summary metrics (ARI, NMI, recovery rate).

        Args:
            save_path: Optional path to save the figure.

        Returns:
            matplotlib Figure.

        Raises:
            RuntimeError: If the benchmark has not been run.
        """
        self._check_has_results()
        df = self.get_results_df()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: per-cell-type Jaccard
        ax1 = axes[0]
        ct_names = df["cell_type"].tolist()
        jaccards = df["best_jaccard"].values
        x = np.arange(len(ct_names))
        colors_list = [
            "#4CAF50" if r else "#F44336"
            for r in df["recovered"].tolist()
        ]
        ax1.bar(x, jaccards, color=colors_list, edgecolor="white")
        ax1.set_xticks(x)
        ax1.set_xticklabels(ct_names, rotation=45, ha="right", fontsize=8)
        ax1.set_ylabel("Best-Match Jaccard")
        ax1.set_title("Marker Recovery per Cell Type")
        ax1.axhline(
            y=self.recovery_threshold,
            color="gray",
            linestyle="--",
            label=f"Threshold ({self.recovery_threshold})",
        )
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3, axis="y")

        # Right: summary metrics
        ax2 = axes[1]
        summary_metrics = {
            "ARI": self.results.get("ari", 0.0),
            "NMI": self.results.get("nmi", 0.0),
            "Recovery Rate": self.results.get("recovery_rate", 0.0),
            "Mean Jaccard": self.results.get("mean_best_jaccard", 0.0),
        }
        metric_names = list(summary_metrics.keys())
        metric_values = list(summary_metrics.values())
        x2 = np.arange(len(metric_names))
        bars = ax2.bar(
            x2,
            metric_values,
            color=["#2196F3", "#FF9800", "#9C27B0", "#00BCD4"],
            edgecolor="white",
        )
        ax2.set_xticks(x2)
        ax2.set_xticklabels(metric_names, fontsize=10)
        ax2.set_ylabel("Score")
        ax2.set_title("Summary Metrics")
        ax2.set_ylim(-0.1, 1.1)
        ax2.grid(True, alpha=0.3, axis="y")

        for bar, val in zip(bars, metric_values):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        fig.suptitle(
            "Cell-Type Marker Recovery Benchmark",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Saved marker benchmark plot to %s", save_path)

        return fig
