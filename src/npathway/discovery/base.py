"""Abstract base class for gene program discovery methods.

This module defines the interface that all gene program discovery methods
must implement, ensuring a consistent API across clustering, topic model,
attention-based, and ensemble approaches.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class BaseProgramDiscovery(ABC):
    """Base class for gene program discovery methods.

    All discovery methods share a common fit/transform interface.  After
    calling :meth:`fit`, the discovered programs can be retrieved via
    :meth:`get_programs` (gene lists) or :meth:`get_program_scores`
    (gene lists with membership scores).

    Attributes
    ----------
    programs_ : dict[str, list[str]] | None
        Discovered gene programs after fitting.  ``None`` before fit.
    program_scores_ : dict[str, list[tuple[str, float]]] | None
        Gene-program membership scores after fitting.  ``None`` before fit.
    """

    def __init__(self) -> None:
        self.programs_: dict[str, list[str]] | None = None
        self.program_scores_: dict[str, list[tuple[str, float]]] | None = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(
        self,
        embeddings: np.ndarray,
        gene_names: list[str],
        **kwargs: object,
    ) -> "BaseProgramDiscovery":
        """Discover gene programs from *embeddings*.

        Parameters
        ----------
        embeddings : np.ndarray
            Gene embedding matrix of shape ``(n_genes, n_dims)``.
        gene_names : list[str]
            Gene identifiers aligned with *embeddings* rows.
        **kwargs : object
            Method-specific keyword arguments.

        Returns
        -------
        BaseProgramDiscovery
            ``self`` (fitted instance).
        """
        ...

    @abstractmethod
    def get_programs(self) -> dict[str, list[str]]:
        """Return discovered gene programs.

        Returns
        -------
        dict[str, list[str]]
            Mapping from program name to list of gene names.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        ...

    @abstractmethod
    def get_program_scores(self) -> dict[str, list[tuple[str, float]]]:
        """Return gene-program membership scores.

        Returns
        -------
        dict[str, list[tuple[str, float]]]
            Mapping from program name to list of ``(gene_name, score)``
            tuples sorted by descending score.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        ...

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def _check_is_fitted(self) -> None:
        """Raise ``RuntimeError`` if the model has not been fitted."""
        if self.programs_ is None:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted yet. "
                "Call .fit() before accessing results."
            )

    def get_n_programs(self) -> int:
        """Return the number of discovered programs.

        Returns
        -------
        int
            Number of gene programs.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        self._check_is_fitted()
        assert self.programs_ is not None  # for type checker
        return len(self.programs_)

    def to_gmt(self, filepath: str) -> None:
        """Export gene programs in GMT (Gene Matrix Transposed) format.

        Each line in the GMT file has the format::

            program_name<TAB>description<TAB>gene1<TAB>gene2<TAB>...

        Parameters
        ----------
        filepath : str
            Destination path for the GMT file.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        self._check_is_fitted()
        assert self.programs_ is not None  # for type checker
        out = Path(filepath)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            for name, genes in self.programs_.items():
                if not genes:
                    logger.warning("Skipping empty program '%s' in GMT export.", name)
                    continue
                line = "\t".join([name, name] + genes)
                fh.write(line + "\n")
        logger.info("Wrote %d programs to %s", len(self.programs_), filepath)

    def summary(self) -> str:
        """Return a human-readable summary of discovered programs.

        Returns
        -------
        str
            Multi-line summary string.
        """
        self._check_is_fitted()
        assert self.programs_ is not None
        lines: list[str] = [
            f"Discovery method: {self.__class__.__name__}",
            f"Number of programs: {len(self.programs_)}",
        ]
        sizes = [len(g) for g in self.programs_.values()]
        if sizes:
            lines.append(
                f"Program sizes: min={min(sizes)}, "
                f"median={int(np.median(sizes))}, "
                f"max={max(sizes)}, "
                f"total_genes={sum(sizes)}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        fitted = self.programs_ is not None
        return (
            f"{self.__class__.__name__}("
            f"fitted={fitted}"
            f"{', n_programs=' + str(len(self.programs_)) if fitted and self.programs_ is not None else ''}"
            f")"
        )
