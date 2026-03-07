"""Cross-model robustness benchmark for gene program evaluation.

This module evaluates how consistent gene programs are when derived from
different foundation models (e.g., scGPT, Geneformer, scBERT). Robust
programs should show high agreement across models, indicating that the
discovered biology is not an artifact of a particular model architecture.

Metrics include Jaccard similarity, Adjusted Rand Index (ARI),
Normalized Mutual Information (NMI), and identification of consensus
programs that appear consistently across models.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from npathway.evaluation.base import BaseBenchmark
from npathway.evaluation.metrics import (
    adjusted_rand_index,
    compute_overlap_matrix,
    jaccard_similarity,
    normalized_mutual_info,
    programs_to_labels,
)

logger = logging.getLogger(__name__)


class CrossModelBenchmark(BaseBenchmark):
    """Benchmark comparing gene programs derived from different foundation models.

    This benchmark measures the consistency (robustness) of gene programs
    across models using multiple complementary metrics:

    1. **Pairwise Jaccard similarity**: Best-match Jaccard between programs
       from different models, averaged over all program pairs.
    2. **Adjusted Rand Index (ARI)**: Global agreement of gene-to-program
       assignments across models.
    3. **Normalized Mutual Information (NMI)**: Information-theoretic
       measure of agreement.
    4. **Consensus programs**: Programs that have a best-match Jaccard above
       a threshold in all pairwise model comparisons.

    Attributes:
        model_programs: Dictionary mapping model names to their gene program
            dictionaries.
        jaccard_threshold: Minimum Jaccard to consider two programs a match.
    """

    def __init__(
        self,
        jaccard_threshold: float = 0.3,
    ) -> None:
        """Initialize the cross-model robustness benchmark.

        Args:
            jaccard_threshold: Jaccard threshold above which two programs
                from different models are considered matching.
        """
        super().__init__(name="CrossModelRobustness")
        self.jaccard_threshold: float = jaccard_threshold
        self.model_programs: dict[str, dict[str, list[str]]] = {}
        self._pairwise_results: list[dict[str, Any]] = []
        self._overlap_matrices: dict[str, pd.DataFrame] = {}
        self._consensus_programs: dict[str, list[str]] = {}

    def run(
        self,
        gene_programs: dict[str, list[str]],
        *,
        model_programs: dict[str, dict[str, list[str]]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute the cross-model robustness benchmark.

        Args:
            gene_programs: Gene programs from one model (used if
                ``model_programs`` is not provided, as the sole entry).
            model_programs: Dictionary mapping model names to their
                respective gene program dictionaries. At least 2 required.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Dictionary containing:
            - mean_best_match_jaccard: Mean of best-match Jaccard across all
              model pairs.
            - mean_ari: Mean Adjusted Rand Index across model pairs.
            - mean_nmi: Mean Normalized Mutual Information across model pairs.
            - n_consensus_programs: Number of consensus programs.
            - n_models: Number of models compared.

        Raises:
            ValueError: If fewer than 2 models are provided.
        """
        if model_programs is None:
            model_programs = {"model_0": gene_programs}

        if len(model_programs) < 2:
            raise ValueError(
                "At least 2 models are required for cross-model comparison. "
                f"Got {len(model_programs)}."
            )

        self.model_programs = model_programs
        model_names = list(model_programs.keys())

        # Build gene universe (union of all genes across all models)
        gene_universe: set[str] = set()
        for programs in model_programs.values():
            for genes in programs.values():
                gene_universe.update(genes)
        gene_universe_list = sorted(gene_universe)

        # Pairwise comparisons
        self._pairwise_results = []
        self._overlap_matrices = {}
        best_match_jaccards: list[float] = []
        ari_scores: list[float] = []
        nmi_scores: list[float] = []

        for model_a, model_b in combinations(model_names, 2):
            programs_a = model_programs[model_a]
            programs_b = model_programs[model_b]

            # Compute overlap matrix (Jaccard)
            overlap = compute_overlap_matrix(programs_a, programs_b)
            pair_key = f"{model_a}_vs_{model_b}"
            self._overlap_matrices[pair_key] = overlap

            # Best-match Jaccard for each program in model A
            if overlap.shape[0] > 0 and overlap.shape[1] > 0:
                best_a_to_b = overlap.max(axis=1).values
                best_b_to_a = overlap.max(axis=0).values
                mean_best = float(
                    np.mean(np.concatenate([best_a_to_b, best_b_to_a]))
                )
            else:
                mean_best = 0.0
            best_match_jaccards.append(mean_best)

            # ARI and NMI (gene-level labels)
            labels_a = programs_to_labels(programs_a, gene_universe_list)
            labels_b = programs_to_labels(programs_b, gene_universe_list)

            # Only consider genes assigned in both
            assigned_mask = (labels_a >= 0) & (labels_b >= 0)
            if assigned_mask.sum() > 0:
                ari_val = adjusted_rand_index(
                    labels_a[assigned_mask], labels_b[assigned_mask]
                )
                nmi_val = normalized_mutual_info(
                    labels_a[assigned_mask], labels_b[assigned_mask]
                )
            else:
                ari_val = 0.0
                nmi_val = 0.0

            ari_scores.append(ari_val)
            nmi_scores.append(nmi_val)

            self._pairwise_results.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "mean_best_match_jaccard": mean_best,
                    "ari": ari_val,
                    "nmi": nmi_val,
                    "n_programs_a": len(programs_a),
                    "n_programs_b": len(programs_b),
                }
            )

        # Identify consensus programs
        self._consensus_programs = self._find_consensus_programs(
            model_programs, self.jaccard_threshold
        )

        aggregated: dict[str, Any] = {
            "mean_best_match_jaccard": float(np.mean(best_match_jaccards))
            if best_match_jaccards
            else 0.0,
            "mean_ari": float(np.mean(ari_scores)) if ari_scores else 0.0,
            "mean_nmi": float(np.mean(nmi_scores)) if nmi_scores else 0.0,
            "n_consensus_programs": len(self._consensus_programs),
            "n_models": len(model_names),
        }

        self._store_results(aggregated)
        return aggregated

    def get_results_df(self) -> pd.DataFrame:
        """Return pairwise model comparison results.

        Returns:
            DataFrame with one row per model pair, including Jaccard, ARI,
            and NMI scores.

        Raises:
            RuntimeError: If the benchmark has not been run.
        """
        self._check_has_results()
        return pd.DataFrame(self._pairwise_results)

    def get_overlap_matrix(self, pair_key: str) -> pd.DataFrame:
        """Return the Jaccard overlap matrix for a specific model pair.

        Args:
            pair_key: Key in the format ``"modelA_vs_modelB"``.

        Returns:
            DataFrame with Jaccard similarities between programs.

        Raises:
            KeyError: If the pair key is not found.
            RuntimeError: If the benchmark has not been run.
        """
        self._check_has_results()
        if pair_key not in self._overlap_matrices:
            available = list(self._overlap_matrices.keys())
            raise KeyError(
                f"Pair key '{pair_key}' not found. Available: {available}"
            )
        return self._overlap_matrices[pair_key].copy()

    def get_consensus_programs(self) -> dict[str, list[str]]:
        """Return the identified consensus programs.

        Consensus programs are defined as programs from the first model
        that have a best-match Jaccard above the threshold with at least
        one program from every other model. The returned gene lists are
        the intersection of matched programs across all models.

        Returns:
            Dictionary mapping consensus program names to gene lists.

        Raises:
            RuntimeError: If the benchmark has not been run.
        """
        self._check_has_results()
        return dict(self._consensus_programs)

    def plot_results(self, save_path: str | None = None) -> plt.Figure:
        """Generate comparison heatmaps and summary bar charts.

        Creates a multi-panel figure:
        - Top row: Overlap heatmaps for each model pair (up to 3 shown).
        - Bottom: Bar chart of pairwise metrics (Jaccard, ARI, NMI).

        Args:
            save_path: Optional file path to save the figure.

        Returns:
            matplotlib Figure object.

        Raises:
            RuntimeError: If the benchmark has not been run.
        """
        self._check_has_results()
        pair_keys = list(self._overlap_matrices.keys())
        n_pairs = len(pair_keys)

        # Layout: top row for heatmaps (up to 3), bottom row for bar chart
        n_heatmaps = min(n_pairs, 3)
        fig, axes = plt.subplots(
            2, max(n_heatmaps, 1),
            figsize=(6 * max(n_heatmaps, 1), 10),
            gridspec_kw={"height_ratios": [1.2, 0.8]},
        )

        if n_heatmaps == 1:
            top_axes = [axes[0]]
            bottom_axes = [axes[1]]
        else:
            top_axes = axes[0] if n_heatmaps > 1 else [axes[0]]
            bottom_axes = axes[1] if n_heatmaps > 1 else [axes[1]]

        # Top row: heatmaps
        for i in range(n_heatmaps):
            ax = top_axes[i]
            pair_key = pair_keys[i]
            matrix = self._overlap_matrices[pair_key]

            # Limit display size
            max_display = 20
            if matrix.shape[0] > max_display:
                matrix = matrix.iloc[:max_display, :]
            if matrix.shape[1] > max_display:
                matrix = matrix.iloc[:, :max_display]

            sns.heatmap(
                matrix,
                ax=ax,
                cmap="YlOrRd",
                vmin=0,
                vmax=1,
                cbar_kws={"label": "Jaccard"},
                xticklabels=matrix.shape[1] <= 20,
                yticklabels=matrix.shape[0] <= 20,
            )
            ax.set_title(pair_key.replace("_vs_", " vs "), fontsize=10)

        # Hide unused top axes
        for i in range(n_heatmaps, len(top_axes) if isinstance(top_axes, list) else 0):
            top_axes[i].set_visible(False)

        # Bottom row: bar chart of pairwise metrics
        df = self.get_results_df()
        if n_heatmaps > 1:
            # Merge bottom axes for the bar chart
            for ax in bottom_axes[1:]:
                ax.set_visible(False)
            ax_bar = bottom_axes[0]
        else:
            ax_bar = bottom_axes[0] if isinstance(bottom_axes, list) else bottom_axes

        pair_labels = [
            f"{r['model_a']}\nvs\n{r['model_b']}"
            for _, r in df.iterrows()
        ]
        x = np.arange(len(pair_labels))
        width = 0.25

        ax_bar.bar(
            x - width, df["mean_best_match_jaccard"], width,
            label="Jaccard", color="#2196F3", alpha=0.85,
        )
        ax_bar.bar(
            x, df["ari"], width,
            label="ARI", color="#4CAF50", alpha=0.85,
        )
        ax_bar.bar(
            x + width, df["nmi"], width,
            label="NMI", color="#FF9800", alpha=0.85,
        )
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(pair_labels, fontsize=8)
        ax_bar.set_ylabel("Score")
        ax_bar.set_title("Pairwise Model Agreement")
        ax_bar.legend()
        ax_bar.set_ylim(-0.1, 1.1)
        ax_bar.grid(True, alpha=0.3, axis="y")

        fig.suptitle(
            "Cross-Model Robustness Benchmark",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Saved cross-model benchmark plot to %s", save_path)

        return fig

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    @staticmethod
    def _find_consensus_programs(
        model_programs: dict[str, dict[str, list[str]]],
        jaccard_threshold: float,
    ) -> dict[str, list[str]]:
        """Identify consensus programs present across all models.

        For each program in the first model, find the best-matching program
        in every other model. If the best match exceeds the Jaccard threshold
        for all models, the program is considered a consensus program. The
        consensus gene list is the intersection of matched programs.

        Args:
            model_programs: Dict mapping model names to program dicts.
            jaccard_threshold: Minimum Jaccard for a match.

        Returns:
            Dictionary mapping consensus program names to their consensus
            (intersection) gene lists.
        """
        model_names = list(model_programs.keys())
        if len(model_names) < 2:
            return {}

        reference_model = model_names[0]
        reference_programs = model_programs[reference_model]
        other_models = model_names[1:]

        consensus: dict[str, list[str]] = {}

        for prog_name, prog_genes in reference_programs.items():
            ref_set = set(prog_genes)
            consensus_genes = set(prog_genes)
            is_consensus = True

            for other_model in other_models:
                other_programs = model_programs[other_model]
                best_jaccard = 0.0
                best_match_genes: set[str] = set()

                for other_genes in other_programs.values():
                    other_set = set(other_genes)
                    jac = jaccard_similarity(ref_set, other_set)
                    if jac > best_jaccard:
                        best_jaccard = jac
                        best_match_genes = other_set

                if best_jaccard < jaccard_threshold:
                    is_consensus = False
                    break

                consensus_genes = consensus_genes & best_match_genes

            if is_consensus and consensus_genes:
                consensus[prog_name] = sorted(consensus_genes)

        logger.info(
            "Found %d consensus programs (threshold=%.2f) across %d models.",
            len(consensus),
            jaccard_threshold,
            len(model_names),
        )
        return consensus
