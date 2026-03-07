"""Attention-based gene program discovery.

This module discovers gene programs by analysing attention matrices
produced by transformer-based foundation models (e.g. scGPT, Geneformer).

Workflow
--------
1. Accept pre-computed attention matrices ``(n_heads, n_genes, n_genes)``.
2. Aggregate across attention heads (mean or max).
3. Threshold the aggregated matrix to construct a weighted gene--gene
   network.
4. Apply community detection (Leiden algorithm) to identify gene programs.
5. Score genes within each program by their attention-derived centrality
   (PageRank or eigenvector centrality).
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from sklearn.preprocessing import normalize

from npathway.discovery.base import BaseProgramDiscovery

logger = logging.getLogger(__name__)

AggregationMethod = Literal["mean", "max"]
CentralityMethod = Literal["pagerank", "eigenvector"]


class AttentionNetworkProgramDiscovery(BaseProgramDiscovery):
    """Discover gene programs from transformer attention matrices.

    Parameters
    ----------
    aggregation : {"mean", "max"}
        How to aggregate attention across heads.
    threshold_quantile : float
        Quantile of edge weights below which edges are pruned.  E.g.
        ``0.90`` keeps only the top 10 % of edges.
    resolution : float
        Resolution parameter for Leiden community detection.
    centrality : {"pagerank", "eigenvector"}
        Method used to score genes within each program.
    random_state : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        aggregation: AggregationMethod = "mean",
        threshold_quantile: float = 0.90,
        resolution: float = 1.0,
        centrality: CentralityMethod = "pagerank",
        random_state: int | None = 42,
    ) -> None:
        super().__init__()
        self.aggregation: AggregationMethod = aggregation
        self.threshold_quantile = threshold_quantile
        self.resolution = resolution
        self.centrality: CentralityMethod = centrality
        self.random_state = random_state

        # Fitted state
        self._gene_names: list[str] | None = None
        self._agg_attention: np.ndarray | None = None
        self._adj: np.ndarray | None = None
        self.labels_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(  # type: ignore[override]
        self,
        embeddings: np.ndarray,
        gene_names: list[str],
        *,
        attention_matrices: np.ndarray | None = None,
        **kwargs: object,
    ) -> "AttentionNetworkProgramDiscovery":
        """Discover gene programs from attention matrices.

        Parameters
        ----------
        embeddings : np.ndarray
            Gene embedding matrix ``(n_genes, n_dims)``.  Used as a
            fallback to construct a similarity-based pseudo-attention
            matrix when *attention_matrices* is not provided.
        gene_names : list[str]
            Gene identifiers.
        attention_matrices : np.ndarray | None
            Pre-computed attention tensors.  Accepted shapes:

            * ``(n_heads, n_genes, n_genes)``
            * ``(n_layers, n_heads, n_genes, n_genes)`` -- layers are
              averaged first, then heads are aggregated.
            * ``(n_genes, n_genes)`` -- treated as a single-head matrix.

            If ``None``, a cosine-similarity matrix is constructed from
            *embeddings* as a proxy.
        **kwargs : object
            Ignored.

        Returns
        -------
        AttentionNetworkProgramDiscovery
        """
        embeddings = np.asarray(embeddings, dtype=np.float64)
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2-D embeddings, got {embeddings.shape}")
        n_genes = embeddings.shape[0]
        if n_genes != len(gene_names):
            raise ValueError("Embeddings rows must match gene_names length.")

        self._gene_names = list(gene_names)

        # Resolve attention
        if attention_matrices is not None:
            attn = np.asarray(attention_matrices, dtype=np.float64)
            attn = self._normalise_attention_shape(attn, n_genes)
        else:
            logger.info(
                "No attention_matrices provided; constructing cosine "
                "similarity proxy from embeddings."
            )
            attn = self._cosine_attention_proxy(embeddings)

        # Aggregate across heads
        agg = self._aggregate_heads(attn)
        self._agg_attention = agg

        # Symmetrise: A_sym = (A + A^T) / 2
        agg_sym = (agg + agg.T) / 2.0
        np.fill_diagonal(agg_sym, 0.0)

        # Threshold
        adj = self._threshold(agg_sym)
        self._adj = adj

        # Community detection (Leiden)
        self._run_leiden(adj)

        # Build programs with centrality scores
        assert self.labels_ is not None
        self._build_programs(adj)

        return self

    def get_programs(self) -> dict[str, list[str]]:
        """Return discovered gene programs.

        Returns
        -------
        dict[str, list[str]]
        """
        self._check_is_fitted()
        assert self.programs_ is not None
        return dict(self.programs_)

    def get_program_scores(self) -> dict[str, list[tuple[str, float]]]:
        """Return scored gene programs.

        Returns
        -------
        dict[str, list[tuple[str, float]]]
        """
        self._check_is_fitted()
        assert self.program_scores_ is not None
        return dict(self.program_scores_)

    def get_aggregated_attention(self) -> np.ndarray:
        """Return the aggregated attention matrix.

        Returns
        -------
        np.ndarray
            ``(n_genes, n_genes)`` aggregated attention.
        """
        self._check_is_fitted()
        assert self._agg_attention is not None
        result: np.ndarray = np.asarray(self._agg_attention.copy())
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_attention_shape(
        attn: np.ndarray, expected_n_genes: int
    ) -> np.ndarray:
        """Reshape attention to ``(n_heads, n_genes, n_genes)``.

        Parameters
        ----------
        attn : np.ndarray
            Raw attention array.
        expected_n_genes : int
            Expected number of genes for validation.

        Returns
        -------
        np.ndarray
            ``(n_heads, n_genes, n_genes)``
        """
        if attn.ndim == 2:
            if attn.shape[0] != expected_n_genes or attn.shape[1] != expected_n_genes:
                raise ValueError(
                    f"2-D attention shape {attn.shape} does not match "
                    f"expected ({expected_n_genes}, {expected_n_genes})."
                )
            return attn[np.newaxis, :, :]

        if attn.ndim == 3:
            if attn.shape[1] != expected_n_genes or attn.shape[2] != expected_n_genes:
                raise ValueError(
                    f"3-D attention shape {attn.shape} does not match "
                    f"expected (n_heads, {expected_n_genes}, {expected_n_genes})."
                )
            return attn

        if attn.ndim == 4:
            # (n_layers, n_heads, n_genes, n_genes) -> average layers
            if attn.shape[2] != expected_n_genes or attn.shape[3] != expected_n_genes:
                raise ValueError(
                    f"4-D attention shape {attn.shape} does not match "
                    f"expected (n_layers, n_heads, {expected_n_genes}, "
                    f"{expected_n_genes})."
                )
            result: np.ndarray = np.asarray(attn.mean(axis=0))  # (n_heads, G, G)
            return result

        raise ValueError(
            f"Unsupported attention tensor rank {attn.ndim}. "
            "Expected 2, 3, or 4 dimensions."
        )

    @staticmethod
    def _cosine_attention_proxy(embeddings: np.ndarray) -> np.ndarray:
        """Build a single-head pseudo-attention from cosine similarity.

        Parameters
        ----------
        embeddings : np.ndarray
            ``(n_genes, n_dims)``

        Returns
        -------
        np.ndarray
            ``(1, n_genes, n_genes)``
        """
        emb_norm = normalize(embeddings, norm="l2")
        sim = emb_norm @ emb_norm.T  # cosine sim in [-1, 1]
        sim = np.clip(sim, 0.0, None)  # keep non-negative
        np.fill_diagonal(sim, 0.0)
        sim_result: np.ndarray = np.asarray(sim[np.newaxis, :, :])
        return sim_result

    def _aggregate_heads(self, attn: np.ndarray) -> np.ndarray:
        """Aggregate across attention heads.

        Parameters
        ----------
        attn : np.ndarray
            ``(n_heads, n_genes, n_genes)``

        Returns
        -------
        np.ndarray
            ``(n_genes, n_genes)``
        """
        if self.aggregation == "mean":
            agg_result: np.ndarray = np.asarray(attn.mean(axis=0))
            return agg_result
        if self.aggregation == "max":
            max_result: np.ndarray = np.asarray(attn.max(axis=0))
            return max_result
        raise ValueError(f"Unknown aggregation '{self.aggregation}'.")

    def _threshold(self, adj: np.ndarray) -> np.ndarray:
        """Zero out edges below the quantile threshold.

        Parameters
        ----------
        adj : np.ndarray
            Symmetric adjacency matrix ``(n, n)``.

        Returns
        -------
        np.ndarray
            Thresholded adjacency.
        """
        # Use only positive upper-triangle edges for quantile.
        # Including zeros makes high quantiles collapse to 0 on sparse graphs.
        upper = adj[np.triu_indices_from(adj, k=1)]
        positive = upper[upper > 0]
        if positive.size == 0:
            adj_out = np.zeros_like(adj)
            np.fill_diagonal(adj_out, 0.0)
            logger.info(
                "Thresholding skipped: no positive edges found in attention graph."
            )
            return adj_out

        cutoff = float(np.quantile(positive, self.threshold_quantile))
        adj_out = adj.copy()
        adj_out[adj_out < cutoff] = 0.0
        np.fill_diagonal(adj_out, 0.0)
        n_edges = int((adj_out > 0).sum()) // 2
        logger.info(
            "Thresholded at quantile=%.2f (cutoff=%.6f): %d edges retained.",
            self.threshold_quantile,
            cutoff,
            n_edges,
        )
        thresh_result: np.ndarray = np.asarray(adj_out)
        return thresh_result

    def _run_leiden(self, adj: np.ndarray) -> None:
        """Run Leiden community detection on *adj*.

        Parameters
        ----------
        adj : np.ndarray
            Symmetric weighted adjacency ``(n, n)``.
        """
        try:
            import igraph as ig
            import leidenalg
        except ImportError as exc:
            raise ImportError(
                "Leiden community detection requires 'igraph' and 'leidenalg'. "
                "Install them with: pip install igraph leidenalg"
            ) from exc

        n = adj.shape[0]
        edges: list[tuple[int, int]] = []
        weights: list[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                w = adj[i, j]
                if w > 0:
                    edges.append((i, j))
                    weights.append(float(w))

        if not edges:
            logger.warning(
                "No edges after thresholding; every gene is its own cluster."
            )
            self.labels_ = np.arange(n, dtype=np.intp)
            return

        g = ig.Graph(n=n, edges=edges, directed=False)
        g.es["weight"] = weights

        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights=g.es["weight"],
            resolution_parameter=self.resolution,
            seed=self.random_state if self.random_state is not None else 0,
        )
        self.labels_ = np.array(partition.membership, dtype=np.intp)
        logger.info("Leiden found %d communities.", len(set(self.labels_)))

    def _build_programs(self, adj: np.ndarray) -> None:
        """Build programs and score genes by centrality.

        Parameters
        ----------
        adj : np.ndarray
            Weighted adjacency ``(n, n)``.
        """
        assert self.labels_ is not None
        assert self._gene_names is not None

        labels = self.labels_
        gene_names = self._gene_names
        unique_labels = sorted(set(labels))

        programs: dict[str, list[str]] = {}
        scores: dict[str, list[tuple[str, float]]] = {}

        for label in unique_labels:
            prog_name = f"attention_program_{label}"
            indices = [i for i, lab in enumerate(labels) if lab == label]

            if not indices:
                continue

            gene_list = [gene_names[i] for i in indices]

            # Compute centrality within the sub-graph
            centrality_values = self._compute_centrality(adj, indices)

            scored = list(zip(gene_list, centrality_values))
            scored.sort(key=lambda t: t[1], reverse=True)

            programs[prog_name] = [t[0] for t in scored]
            scores[prog_name] = scored

        self.programs_ = programs
        self.program_scores_ = scores

    def _compute_centrality(
        self, adj: np.ndarray, indices: list[int]
    ) -> list[float]:
        """Compute per-gene centrality within a subgraph.

        Parameters
        ----------
        adj : np.ndarray
            Full adjacency matrix.
        indices : list[int]
            Gene indices belonging to the community.

        Returns
        -------
        list[float]
            Centrality scores for each gene in *indices*.
        """
        n = len(indices)
        if n == 1:
            return [1.0]

        # Extract sub-adjacency
        sub_adj = adj[np.ix_(indices, indices)]

        if self.centrality == "pagerank":
            return self._pagerank(sub_adj)
        if self.centrality == "eigenvector":
            return self._eigenvector_centrality(sub_adj)
        raise ValueError(f"Unknown centrality method '{self.centrality}'.")

    @staticmethod
    def _pagerank(
        adj: np.ndarray, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-8
    ) -> list[float]:
        """Compute PageRank on a weighted adjacency matrix.

        Parameters
        ----------
        adj : np.ndarray
            ``(n, n)`` weighted adjacency.
        damping : float
            Damping factor.
        max_iter : int
            Maximum iterations.
        tol : float
            Convergence tolerance.

        Returns
        -------
        list[float]
            PageRank scores.
        """
        n = adj.shape[0]
        if n == 0:
            return []

        # Build column-stochastic transition matrix
        col_sums = adj.sum(axis=0)
        col_sums = np.where(col_sums == 0, 1.0, col_sums)
        M = adj / col_sums  # column-stochastic

        pr = np.full(n, 1.0 / n)
        for _ in range(max_iter):
            pr_new = (1 - damping) / n + damping * (M @ pr)
            if np.abs(pr_new - pr).sum() < tol:
                pr = pr_new
                break
            pr = pr_new

        # Normalise to [0, 1]
        pr_min, pr_max = pr.min(), pr.max()
        if pr_max - pr_min > 1e-12:
            pr = (pr - pr_min) / (pr_max - pr_min)
        else:
            pr = np.ones(n)
        return [float(x) for x in pr.tolist()]

    @staticmethod
    def _eigenvector_centrality(
        adj: np.ndarray, max_iter: int = 200, tol: float = 1e-8
    ) -> list[float]:
        """Compute eigenvector centrality via power iteration.

        Parameters
        ----------
        adj : np.ndarray
            ``(n, n)`` weighted symmetric adjacency.
        max_iter : int
            Maximum iterations.
        tol : float
            Convergence tolerance.

        Returns
        -------
        list[float]
            Centrality scores normalised to ``[0, 1]``.
        """
        n = adj.shape[0]
        if n == 0:
            return []

        x = np.ones(n, dtype=np.float64) / np.sqrt(n)
        for _ in range(max_iter):
            x_new = adj @ x
            norm = np.linalg.norm(x_new)
            if norm < 1e-12:
                break
            x_new /= norm
            if np.abs(x_new - x).sum() < tol:
                x = x_new
                break
            x = x_new

        x = np.abs(x)
        x_min, x_max = x.min(), x.max()
        if x_max - x_min > 1e-12:
            x = (x - x_min) / (x_max - x_min)
        else:
            x = np.ones(n)
        return [float(x_i) for x_i in x.tolist()]
