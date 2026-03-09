"""Abstract base class for gene program discovery methods.

This module defines the interface that all gene program discovery methods
must implement, ensuring a consistent API across clustering, topic model,
attention-based, and ensemble approaches.

Supports both **hard** (each gene in exactly one program) and **soft**
(each gene has weighted membership across multiple programs) partitioning.
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
    :meth:`get_programs` (gene lists), :meth:`get_program_scores`
    (gene lists with membership scores), or :meth:`get_soft_programs`
    (weighted multi-program gene membership).

    Attributes
    ----------
    programs_ : dict[str, list[str]] | None
        Discovered gene programs after fitting.  ``None`` before fit.
    program_scores_ : dict[str, list[tuple[str, float]]] | None
        Gene-program membership scores after fitting.  ``None`` before fit.
    soft_programs_ : dict[str, dict[str, float]] | None
        Soft (weighted) gene-program membership.  Each key is a program
        name and each value maps gene names to their membership weight.
        ``None`` before fit; subclasses may populate this natively.
    """

    def __init__(self) -> None:
        self.programs_: dict[str, list[str]] | None = None
        self.program_scores_: dict[str, list[tuple[str, float]]] | None = None
        self.soft_programs_: dict[str, dict[str, float]] | None = None

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
    # Soft membership interface (concrete, with fallback)
    # ------------------------------------------------------------------

    def get_soft_programs(
        self, threshold: float = 0.01
    ) -> dict[str, dict[str, float]]:
        """Return soft (weighted) gene-program membership.

        If the subclass populates :attr:`soft_programs_` natively (e.g.
        ETM topic distributions, distance-to-centroid weights), that is
        returned after filtering by *threshold*.  Otherwise a fallback is
        constructed from :meth:`get_program_scores` with weight 1.0 for
        genes that only appear in the hard partition.

        Parameters
        ----------
        threshold : float
            Minimum membership weight to include a gene in a program.

        Returns
        -------
        dict[str, dict[str, float]]
            ``{program_name: {gene: weight, ...}, ...}``
        """
        self._check_is_fitted()

        if self.soft_programs_ is not None:
            return {
                prog: {g: w for g, w in gw.items() if w >= threshold}
                for prog, gw in self.soft_programs_.items()
            }

        # Fallback: construct from program_scores_
        assert self.program_scores_ is not None
        soft: dict[str, dict[str, float]] = {}
        for prog, scored in self.program_scores_.items():
            soft[prog] = {g: w for g, w in scored if w >= threshold}
        return soft

    def get_gene_memberships(self) -> dict[str, dict[str, float]]:
        """Return per-gene membership across all programs.

        This is the transpose view of :meth:`get_soft_programs`: for
        each gene, a mapping of ``{program_name: weight}``.

        Returns
        -------
        dict[str, dict[str, float]]
            ``{gene: {program_name: weight, ...}, ...}``
        """
        soft = self.get_soft_programs(threshold=0.0)
        gene_map: dict[str, dict[str, float]] = {}
        for prog, gw in soft.items():
            for gene, weight in gw.items():
                gene_map.setdefault(gene, {})[prog] = weight
        return gene_map

    def to_weighted_gmt(self, filepath: str, threshold: float = 0.01) -> None:
        """Export soft programs in weighted GMT format.

        Each gene entry is written as ``gene,weight`` using the utility
        :func:`~npathway.utils.gmt_io.weighted_programs_to_gmt`.

        Parameters
        ----------
        filepath : str
            Destination file path.
        threshold : float
            Minimum weight to include a gene.
        """
        from npathway.utils.gmt_io import weighted_programs_to_gmt

        soft = self.get_soft_programs(threshold=threshold)
        # Convert dict[str, dict[str, float]] -> dict[str, list[tuple[str, float]]]
        weighted: dict[str, list[tuple[str, float]]] = {}
        for prog, gw in soft.items():
            pairs = sorted(gw.items(), key=lambda t: t[1], reverse=True)
            if pairs:
                weighted[prog] = pairs
        if not weighted:
            logger.warning("No programs passed threshold %.4f for GMT export.", threshold)
            return
        weighted_programs_to_gmt(weighted, filepath)

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
