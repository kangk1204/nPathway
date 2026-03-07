"""Abstract base class for gene program benchmarking."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


class BaseBenchmark(ABC):
    """Base class for benchmarking gene programs.

    All benchmark implementations inherit from this class and must implement
    the three core methods: ``run``, ``get_results_df``, and ``plot_results``.

    Attributes:
        name: Human-readable name of the benchmark.
        results: Dictionary storing the most recent benchmark results.
        _history: List of all past result dictionaries for longitudinal tracking.
    """

    def __init__(self, name: str = "BaseBenchmark") -> None:
        """Initialize the benchmark.

        Args:
            name: Human-readable name for this benchmark instance.
        """
        self.name: str = name
        self.results: dict[str, Any] = {}
        self._history: list[dict[str, Any]] = []

    @abstractmethod
    def run(
        self,
        gene_programs: dict[str, list[str]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute the benchmark on a set of gene programs.

        Args:
            gene_programs: Dictionary mapping program names to lists of gene
                symbols belonging to that program.
            **kwargs: Additional benchmark-specific parameters.

        Returns:
            Dictionary containing benchmark results with metric names as keys.
        """
        ...

    @abstractmethod
    def get_results_df(self) -> pd.DataFrame:
        """Return the benchmark results as a tidy pandas DataFrame.

        Returns:
            A DataFrame containing one row per evaluated condition/metric
            combination, suitable for downstream analysis and plotting.

        Raises:
            RuntimeError: If the benchmark has not been run yet.
        """
        ...

    @abstractmethod
    def plot_results(self, save_path: str | None = None) -> plt.Figure:
        """Generate a summary visualization of the benchmark results.

        Args:
            save_path: Optional file path to save the figure. If ``None``,
                the figure is returned without saving.

        Returns:
            A matplotlib Figure object containing the benchmark visualization.

        Raises:
            RuntimeError: If the benchmark has not been run yet.
        """
        ...

    def _check_has_results(self) -> None:
        """Verify that results are available.

        Raises:
            RuntimeError: If no results have been computed.
        """
        if not self.results:
            raise RuntimeError(
                f"Benchmark '{self.name}' has not been run yet. "
                "Call run() before accessing results."
            )

    def _store_results(self, results: dict[str, Any]) -> None:
        """Store results and append to history.

        Args:
            results: The result dictionary to store.
        """
        self.results = results
        self._history.append(results)
        logger.info("Benchmark '%s' completed. Stored results.", self.name)

    def compare(
        self,
        program_collections: dict[str, dict[str, list[str]]],
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Run the benchmark on multiple gene program collections and compare.

        Args:
            program_collections: Dictionary mapping collection names (e.g.,
                ``"learned"``, ``"KEGG"``, ``"Reactome"``) to their respective
                gene program dictionaries.
            **kwargs: Additional benchmark-specific parameters passed to ``run``.

        Returns:
            A DataFrame with results from all collections, including a
            ``collection`` column for identification.
        """
        frames: list[pd.DataFrame] = []
        for collection_name, programs in program_collections.items():
            logger.info(
                "Running benchmark '%s' on collection '%s' (%d programs)",
                self.name,
                collection_name,
                len(programs),
            )
            self.run(programs, **kwargs)
            df = self.get_results_df()
            df["collection"] = collection_name
            frames.append(df)
        return pd.concat(frames, ignore_index=True)
