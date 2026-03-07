"""Context-specificity benchmark for gene program evaluation.

This module measures how gene programs vary across cellular contexts
(cell types, tissues, conditions). Context-specific programs should change
meaningfully between contexts, whereas static curated pathways remain the
same. Programs that are too uniform across contexts may lack biological
resolution, while programs that change too drastically may be noisy.

Metrics include Jaccard similarity of same-indexed programs across contexts,
gene reassignment frequency, and entropy-based program specificity scores.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from npathway.evaluation.base import BaseBenchmark
from npathway.evaluation.metrics import jaccard_similarity

logger = logging.getLogger(__name__)


class ContextSpecificityBenchmark(BaseBenchmark):
    """Benchmark evaluating context-specificity of gene programs.

    Given gene programs derived from different cellular contexts (e.g.,
    different cell types), this benchmark measures:

    1. **Cross-context Jaccard similarity**: How much overlap exists between
       the same program index in different contexts. Lower values indicate
       more context-specific programs.
    2. **Gene reassignment frequency**: How often each gene switches from
       one program to another across contexts.
    3. **Program specificity score**: Entropy-based measure of how
       concentrated a program is across contexts.

    Attributes:
        cell_type_key: Column in ``adata.obs`` for cell type labels.
        context_programs: Dictionary mapping context names to their program
            dictionaries.
    """

    def __init__(self, cell_type_key: str = "cell_type") -> None:
        """Initialize the context-specificity benchmark.

        Args:
            cell_type_key: Column name in ``adata.obs`` that contains cell
                type annotations.
        """
        super().__init__(name="ContextSpecificity")
        self.cell_type_key: str = cell_type_key
        self.context_programs: dict[str, dict[str, list[str]]] = {}
        self._specificity_matrix: pd.DataFrame | None = None
        self._jaccard_results: pd.DataFrame | None = None
        self._reassignment_df: pd.DataFrame | None = None

    def run(
        self,
        gene_programs: dict[str, list[str]],
        *,
        context_programs: dict[str, dict[str, list[str]]] | None = None,
        curated_programs: dict[str, list[str]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute the context-specificity benchmark.

        Args:
            gene_programs: A representative set of gene programs (e.g., from
                one context). Used as baseline for comparison.
            context_programs: Dictionary mapping context names (e.g., cell
                types) to their context-specific gene program dictionaries.
                If provided, these are used for the cross-context analysis.
            curated_programs: Optional curated pathways (e.g., KEGG) for
                comparison. Curated programs are static across contexts, so
                they serve as a baseline.
            **kwargs: Additional parameters (unused, for API consistency).

        Returns:
            Dictionary containing:
            - mean_cross_context_jaccard: Mean Jaccard across contexts.
            - mean_reassignment_freq: Mean gene reassignment frequency.
            - mean_specificity_score: Mean program specificity score.
            - n_contexts: Number of contexts analyzed.

        Raises:
            ValueError: If fewer than 2 contexts are provided.
        """
        if context_programs is None:
            context_programs = {"default": gene_programs}

        if len(context_programs) < 2:
            raise ValueError(
                "At least 2 contexts are required for context-specificity "
                f"analysis. Got {len(context_programs)}."
            )

        self.context_programs = context_programs

        # 1. Cross-context Jaccard similarity
        self._jaccard_results = self._compute_cross_context_jaccard(
            context_programs
        )
        mean_jaccard = float(self._jaccard_results["jaccard"].mean())

        # 2. Gene reassignment frequency
        self._reassignment_df = self._compute_gene_reassignment(
            context_programs
        )
        mean_reassignment = float(
            self._reassignment_df["reassignment_freq"].mean()
        )

        # 3. Program specificity score (entropy-based)
        self._specificity_matrix = self._compute_specificity_scores(
            context_programs
        )
        mean_specificity = float(self._specificity_matrix.values.mean())

        # 4. If curated programs are provided, compute their (static) metrics
        curated_metrics: dict[str, float] = {}
        if curated_programs is not None:
            # Curated programs are the same for all contexts
            static_context = {
                ctx: curated_programs for ctx in context_programs
            }
            curated_jaccard_df = self._compute_cross_context_jaccard(
                static_context
            )
            curated_metrics["curated_mean_jaccard"] = float(
                curated_jaccard_df["jaccard"].mean()
            )

        aggregated: dict[str, Any] = {
            "mean_cross_context_jaccard": mean_jaccard,
            "mean_reassignment_freq": mean_reassignment,
            "mean_specificity_score": mean_specificity,
            "n_contexts": len(context_programs),
            **curated_metrics,
        }
        self._store_results(aggregated)
        return aggregated

    def get_results_df(self) -> pd.DataFrame:
        """Return the cross-context Jaccard similarity results.

        Returns:
            DataFrame with columns: context1, context2, program,
            jaccard, indicating the Jaccard similarity of each program
            between each pair of contexts.

        Raises:
            RuntimeError: If the benchmark has not been run.
        """
        self._check_has_results()
        if self._jaccard_results is None:
            return pd.DataFrame()
        return self._jaccard_results.copy()

    def get_reassignment_df(self) -> pd.DataFrame:
        """Return gene reassignment frequency results.

        Returns:
            DataFrame with columns: gene, n_programs, reassignment_freq.

        Raises:
            RuntimeError: If the benchmark has not been run.
        """
        self._check_has_results()
        if self._reassignment_df is None:
            return pd.DataFrame()
        return self._reassignment_df.copy()

    def get_specificity_matrix(self) -> pd.DataFrame:
        """Return the program-context specificity matrix.

        Returns:
            DataFrame of shape ``(n_programs, n_contexts)`` with specificity
            scores. Lower values (lower entropy) indicate higher specificity.

        Raises:
            RuntimeError: If the benchmark has not been run.
        """
        self._check_has_results()
        if self._specificity_matrix is None:
            return pd.DataFrame()
        return self._specificity_matrix.copy()

    def plot_results(self, save_path: str | None = None) -> plt.Figure:
        """Generate a heatmap of program-context specificity.

        Creates a multi-panel figure:
        - Left: Heatmap of program specificity scores across contexts.
        - Right: Histogram of gene reassignment frequencies.

        Args:
            save_path: Optional file path to save the figure.

        Returns:
            matplotlib Figure object.

        Raises:
            RuntimeError: If the benchmark has not been run.
        """
        self._check_has_results()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: specificity heatmap
        ax1 = axes[0]
        if self._specificity_matrix is not None and not self._specificity_matrix.empty:
            matrix = self._specificity_matrix
            # Limit display for readability
            max_display = 30
            if matrix.shape[0] > max_display:
                matrix = matrix.iloc[:max_display, :]

            sns.heatmap(
                matrix,
                ax=ax1,
                cmap="YlOrRd_r",
                cbar_kws={"label": "Specificity Score\n(lower = more specific)"},
                xticklabels=True,
                yticklabels=matrix.shape[0] <= 30,
            )
            ax1.set_title("Program-Context Specificity")
            ax1.set_xlabel("Context")
            ax1.set_ylabel("Program")
        else:
            ax1.text(
                0.5, 0.5, "No specificity data", ha="center", va="center",
                transform=ax1.transAxes,
            )

        # Right: reassignment histogram
        ax2 = axes[1]
        if self._reassignment_df is not None and not self._reassignment_df.empty:
            freqs = self._reassignment_df["reassignment_freq"].values
            ax2.hist(
                freqs, bins=20, color="#FF5722", edgecolor="white", alpha=0.85
            )
            ax2.set_xlabel("Reassignment Frequency")
            ax2.set_ylabel("Number of Genes")
            ax2.set_title("Gene Reassignment Across Contexts")
            ax2.axvline(
                x=np.mean(freqs), color="black", linestyle="--",
                label=f"Mean = {np.mean(freqs):.3f}",
            )
            ax2.legend()
        else:
            ax2.text(
                0.5, 0.5, "No reassignment data", ha="center", va="center",
                transform=ax2.transAxes,
            )

        fig.suptitle(
            "Context-Specificity Benchmark", fontsize=14, fontweight="bold"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Saved context-specificity plot to %s", save_path)

        return fig

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_cross_context_jaccard(
        context_programs: dict[str, dict[str, list[str]]],
    ) -> pd.DataFrame:
        """Compute Jaccard similarity between same-indexed programs across contexts.

        Args:
            context_programs: Dict mapping context name to program dict.

        Returns:
            DataFrame with columns: context1, context2, program, jaccard.
        """
        context_names = list(context_programs.keys())
        rows: list[dict[str, Any]] = []

        for ctx1, ctx2 in combinations(context_names, 2):
            programs1 = context_programs[ctx1]
            programs2 = context_programs[ctx2]
            shared_program_names = set(programs1.keys()) & set(programs2.keys())

            for prog_name in shared_program_names:
                s1 = set(programs1[prog_name])
                s2 = set(programs2[prog_name])
                jac = jaccard_similarity(s1, s2)
                rows.append(
                    {
                        "context1": ctx1,
                        "context2": ctx2,
                        "program": prog_name,
                        "jaccard": jac,
                    }
                )

        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["context1", "context2", "program", "jaccard"]
        )

    @staticmethod
    def _compute_gene_reassignment(
        context_programs: dict[str, dict[str, list[str]]],
    ) -> pd.DataFrame:
        """Compute how often each gene changes program assignment across contexts.

        Reassignment frequency is defined as the number of distinct programs
        a gene belongs to (across all contexts) divided by the number of
        contexts it appears in.

        Args:
            context_programs: Dict mapping context name to program dict.

        Returns:
            DataFrame with columns: gene, n_programs, n_contexts,
            reassignment_freq.
        """
        # For each gene, track which programs it belongs to in each context
        gene_assignments: dict[str, list[str]] = defaultdict(list)
        gene_contexts: dict[str, set[str]] = defaultdict(set)

        for ctx_name, programs in context_programs.items():
            for prog_name, genes in programs.items():
                for gene in genes:
                    gene_assignments[gene].append(prog_name)
                    gene_contexts[gene].add(ctx_name)

        rows: list[dict[str, Any]] = []
        for gene, assignments in gene_assignments.items():
            n_contexts = len(gene_contexts[gene])
            n_unique_programs = len(set(assignments))

            # Reassignment frequency: fraction of unique programs out of
            # total assignments (how often the gene "switches")
            if len(assignments) <= 1:
                freq = 0.0
            else:
                # Normalized entropy of the assignment distribution
                counts = Counter(assignments)
                total = sum(counts.values())
                probs = np.array([c / total for c in counts.values()])
                max_entropy = np.log2(len(counts)) if len(counts) > 1 else 1.0
                entropy = -np.sum(probs * np.log2(probs + 1e-12))
                freq = entropy / max_entropy if max_entropy > 0 else 0.0

            rows.append(
                {
                    "gene": gene,
                    "n_programs": n_unique_programs,
                    "n_contexts": n_contexts,
                    "reassignment_freq": freq,
                }
            )

        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["gene", "n_programs", "n_contexts", "reassignment_freq"]
        )

    @staticmethod
    def _compute_specificity_scores(
        context_programs: dict[str, dict[str, list[str]]],
    ) -> pd.DataFrame:
        """Compute entropy-based program specificity scores.

        For each program, compute the distribution of its genes across
        contexts and measure entropy. Lower entropy indicates that the
        program's genes are concentrated in fewer contexts (more specific).

        Args:
            context_programs: Dict mapping context name to program dict.

        Returns:
            DataFrame of shape ``(n_programs, n_contexts)`` containing
            normalized specificity scores.
        """
        context_names = list(context_programs.keys())
        n_contexts = len(context_names)

        # Collect all program names across contexts
        all_program_names: set[str] = set()
        for programs in context_programs.values():
            all_program_names.update(programs.keys())

        sorted_programs = sorted(all_program_names)

        # Build the specificity matrix
        matrix = np.zeros((len(sorted_programs), n_contexts), dtype=np.float64)

        for p_idx, prog_name in enumerate(sorted_programs):
            sizes: list[int] = []
            for c_idx, ctx_name in enumerate(context_names):
                programs = context_programs[ctx_name]
                if prog_name in programs:
                    size = len(programs[prog_name])
                else:
                    size = 0
                sizes.append(size)
                matrix[p_idx, c_idx] = size

            # Convert sizes to proportions (context-specific specificity)
            total = sum(sizes)
            if total > 0:
                probs = np.array(sizes, dtype=np.float64) / total
                matrix[p_idx, :] = probs

        return pd.DataFrame(
            matrix, index=sorted_programs, columns=context_names
        )
