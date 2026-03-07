"""Ensemble / consensus gene program discovery.

This module combines the outputs of multiple :class:`BaseProgramDiscovery`
instances into a single set of consensus gene programs.

Strategy
--------
1. Run each constituent discovery method independently.
2. Build a **co-occurrence matrix**: for every pair of genes that appear
   in the same program in *any* method, increment their co-occurrence
   count.
3. Normalise the co-occurrence matrix to ``[0, 1]``.
4. Cluster the co-occurrence matrix (Leiden on the thresholded graph, or
   hierarchical clustering) to produce consensus programs.
5. Assign a **confidence score** to each gene--program assignment based
   on the fraction of constituent methods that placed the gene in the
   same consensus cluster.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from npathway.discovery.base import BaseProgramDiscovery

logger = logging.getLogger(__name__)

ConsensusMethod = Literal["leiden", "hierarchical"]
MethodWeighting = Literal["uniform", "quality", "coherence"]


class EnsembleProgramDiscovery(BaseProgramDiscovery):
    """Consensus gene program discovery from multiple methods.

    Parameters
    ----------
    methods : list[BaseProgramDiscovery]
        Discovery method instances.  Each must be *unfitted*; the
        ensemble will call ``.fit()`` on every method.
    consensus_method : {"leiden", "hierarchical"}
        Clustering algorithm applied to the co-occurrence matrix.
    n_programs : int | None
        Number of consensus programs.  Required for ``hierarchical``.
        For ``leiden`` it is determined by the resolution parameter.
    resolution : float
        Resolution for Leiden community detection on the co-occurrence
        graph.
    threshold_quantile : float
        Quantile below which co-occurrence edges are pruned before
        Leiden clustering.
    min_program_size : int
        Minimum number of genes per consensus program; smaller programs
        are merged into the nearest neighbour program.
    method_weighting : {"uniform", "quality", "coherence"}
        How to combine constituent methods when building the co-occurrence
        matrix. ``"uniform"`` gives each method equal weight.
        ``"quality"`` uses each method's internal quality metrics
        (when available) to emphasize more stable/coherent methods.
        ``"coherence"`` computes a method score directly from discovered
        programs and embeddings (mean cosine-to-centroid coherence), so it
        works even when methods do not expose explicit quality metrics.
    minimum_method_weight : float
        Lower-bound weight floor before re-normalization. Helps prevent one
        method from fully dominating consensus aggregation.
    adaptive_threshold : bool
        If ``True`` and ``consensus_method="leiden"``, sweep multiple
        threshold quantiles and pick the one maximizing an internal graph
        quality objective (modularity + node coverage - singleton penalty).
    threshold_grid : tuple[float, ...]
        Candidate quantiles used when ``adaptive_threshold=True``.
    random_state : int | None
        Random seed.

    Attributes
    ----------
    methods_ : list[BaseProgramDiscovery]
        Fitted constituent methods (populated after ``.fit()``).
    cooccurrence_ : np.ndarray | None
        Normalised co-occurrence matrix after fitting.
    confidence_ : dict[str, list[tuple[str, float]]] | None
        Per-gene confidence scores for each consensus program.
    """

    def __init__(
        self,
        methods: list[BaseProgramDiscovery],
        consensus_method: ConsensusMethod = "leiden",
        n_programs: int | None = None,
        resolution: float = 1.0,
        threshold_quantile: float = 0.5,
        min_program_size: int = 3,
        method_weighting: MethodWeighting = "coherence",
        minimum_method_weight: float = 0.05,
        adaptive_threshold: bool = False,
        threshold_grid: tuple[float, ...] = (0.2, 0.35, 0.5, 0.65, 0.8),
        random_state: int | None = 42,
    ) -> None:
        super().__init__()
        if not methods:
            raise ValueError("At least one discovery method is required.")
        self.methods: list[BaseProgramDiscovery] = methods
        self.consensus_method: ConsensusMethod = consensus_method
        self.n_programs = n_programs
        self.resolution = resolution
        self.threshold_quantile = threshold_quantile
        self.min_program_size = min_program_size
        self.method_weighting: MethodWeighting = method_weighting
        self.minimum_method_weight = float(max(minimum_method_weight, 0.0))
        self.adaptive_threshold = adaptive_threshold
        self.threshold_grid = tuple(float(q) for q in threshold_grid)
        self.random_state = random_state
        if not (0.0 <= self.threshold_quantile <= 1.0):
            raise ValueError(
                f"threshold_quantile must be in [0, 1], got {self.threshold_quantile}"
            )
        for q in self.threshold_grid:
            if not (0.0 <= q <= 1.0):
                raise ValueError(f"All threshold_grid values must be in [0, 1], got {q}")
        if self.method_weighting not in {"uniform", "quality", "coherence"}:
            raise ValueError(
                f"Unknown method_weighting='{self.method_weighting}'. "
                "Choose from {'uniform', 'quality', 'coherence'}."
            )

        # Fitted state
        self.methods_: list[BaseProgramDiscovery] = []
        self.cooccurrence_: np.ndarray | None = None
        self.confidence_: dict[str, list[tuple[str, float]]] | None = None
        self.method_weights_: np.ndarray | None = None
        self.method_coherence_: np.ndarray | None = None
        self.chosen_threshold_quantile_: float | None = None
        self._gene_names: list[str] | None = None
        self._labels: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        embeddings: np.ndarray,
        gene_names: list[str],
        **kwargs: object,
    ) -> "EnsembleProgramDiscovery":
        """Run all methods, build co-occurrence, and extract consensus.

        Parameters
        ----------
        embeddings : np.ndarray
            Gene embedding matrix ``(n_genes, n_dims)``.
        gene_names : list[str]
            Gene identifiers.
        **kwargs : object
            Additional keyword arguments forwarded to each constituent
            method's ``.fit()`` call.

        Returns
        -------
        EnsembleProgramDiscovery
        """
        embeddings = np.asarray(embeddings, dtype=np.float64)
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2-D embeddings, got {embeddings.shape}")
        if embeddings.shape[0] != len(gene_names):
            raise ValueError("Embeddings rows must match gene_names length.")

        self._gene_names = list(gene_names)
        n_genes = len(gene_names)
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}

        # 1. Fit each method
        self.methods_ = []
        all_programs: list[dict[str, list[str]]] = []
        for i, method in enumerate(self.methods):
            logger.info(
                "Fitting method %d/%d: %s",
                i + 1,
                len(self.methods),
                method.__class__.__name__,
            )
            method.fit(embeddings, gene_names, **kwargs)
            self.methods_.append(method)
            all_programs.append(method.get_programs())

        method_weights, method_coherence = self._compute_method_weights(
            methods=self.methods_,
            all_programs=all_programs,
            embeddings=embeddings,
            gene_to_idx=gene_to_idx,
        )
        self.method_weights_ = method_weights
        self.method_coherence_ = method_coherence

        # 2. Build co-occurrence matrix
        cooccurrence = np.zeros((n_genes, n_genes), dtype=np.float64)
        for method_weight, prog_dict in zip(method_weights, all_programs):
            for _prog_name, genes in prog_dict.items():
                idxs = [gene_to_idx[g] for g in genes if g in gene_to_idx]
                if len(idxs) < 2:
                    continue
                arr = np.asarray(idxs, dtype=np.intp)
                cooccurrence[np.ix_(arr, arr)] += float(method_weight)

        np.fill_diagonal(cooccurrence, 0.0)

        # Normalise by the total method weight.
        total_weight = float(method_weights.sum())
        if total_weight > 0:
            cooccurrence /= total_weight
        self.cooccurrence_ = cooccurrence

        # 3. Consensus clustering
        if self.consensus_method == "leiden":
            labels = self._consensus_leiden(cooccurrence)
        elif self.consensus_method == "hierarchical":
            labels = self._consensus_hierarchical(cooccurrence)
        else:
            raise ValueError(
                f"Unknown consensus_method '{self.consensus_method}'."
            )
        self._labels = labels

        # 4. Merge small programs
        labels = self._merge_small_programs(labels, cooccurrence)
        self._labels = labels

        # 5. Build programs and confidence scores
        self._build_programs(
            labels=labels,
            cooccurrence=cooccurrence,
            all_programs=all_programs,
            gene_to_idx=gene_to_idx,
            method_weights=method_weights,
        )
        return self

    def get_programs(self) -> dict[str, list[str]]:
        """Return consensus gene programs.

        Returns
        -------
        dict[str, list[str]]
        """
        self._check_is_fitted()
        assert self.programs_ is not None
        return dict(self.programs_)

    def get_program_scores(self) -> dict[str, list[tuple[str, float]]]:
        """Return scored consensus programs (confidence-based).

        Returns
        -------
        dict[str, list[tuple[str, float]]]
        """
        self._check_is_fitted()
        assert self.program_scores_ is not None
        return dict(self.program_scores_)

    def get_cooccurrence_matrix(self) -> np.ndarray:
        """Return the normalised co-occurrence matrix.

        Returns
        -------
        np.ndarray
            ``(n_genes, n_genes)`` co-occurrence matrix in ``[0, 1]``.
        """
        self._check_is_fitted()
        assert self.cooccurrence_ is not None
        result: np.ndarray = np.asarray(self.cooccurrence_.copy())
        return result

    def get_method_results(self) -> list[dict[str, list[str]]]:
        """Return the programs discovered by each constituent method.

        Returns
        -------
        list[dict[str, list[str]]]
            One program dictionary per fitted method.
        """
        self._check_is_fitted()
        return [m.get_programs() for m in self.methods_]

    def get_method_weights(self) -> np.ndarray:
        """Return normalized method weights used for consensus aggregation."""
        self._check_is_fitted()
        assert self.method_weights_ is not None
        out: np.ndarray = np.asarray(self.method_weights_.copy())
        return out

    def get_method_coherence(self) -> np.ndarray:
        """Return per-method coherence scores used in weighting."""
        self._check_is_fitted()
        assert self.method_coherence_ is not None
        out: np.ndarray = np.asarray(self.method_coherence_.copy())
        return out

    # ------------------------------------------------------------------
    # Consensus back-ends
    # ------------------------------------------------------------------

    def _consensus_leiden(self, cooccurrence: np.ndarray) -> np.ndarray:
        """Leiden community detection on the co-occurrence graph.

        Parameters
        ----------
        cooccurrence : np.ndarray
            ``(n, n)``

        Returns
        -------
        np.ndarray
            Cluster labels.
        """
        try:
            import igraph as ig
            import leidenalg
        except ImportError as exc:
            raise ImportError(
                "Leiden consensus requires 'igraph' and 'leidenalg'. "
                "Install with: pip install igraph leidenalg"
            ) from exc

        n = cooccurrence.shape[0]

        candidate_quantiles = [self.threshold_quantile]
        if self.adaptive_threshold:
            candidate_quantiles.extend(self.threshold_grid)
        # Preserve order while removing duplicates.
        candidate_quantiles = list(dict.fromkeys(float(q) for q in candidate_quantiles))

        best_membership: np.ndarray | None = None
        best_score = -np.inf
        best_quantile = self.threshold_quantile

        for quantile in candidate_quantiles:
            edges, weights = self._build_weighted_edges(
                cooccurrence=cooccurrence,
                quantile=quantile,
            )
            if not edges:
                continue

            g = ig.Graph(n=n, edges=edges, directed=False)
            g.es["weight"] = weights

            partition = leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                weights=g.es["weight"],
                resolution_parameter=self.resolution,
                seed=self.random_state if self.random_state is not None else 0,
            )
            membership = np.array(partition.membership, dtype=np.intp)

            if not self.adaptive_threshold:
                self.chosen_threshold_quantile_ = quantile
                return membership

            unique, counts = np.unique(membership, return_counts=True)
            n_singletons = int(np.sum(counts == 1))
            singleton_frac = n_singletons / max(len(membership), 1)
            active_nodes = np.unique(np.array(edges, dtype=np.intp).ravel())
            coverage = len(active_nodes) / max(n, 1)
            # Objective: high modularity and high coverage, penalize
            # degenerate singleton-heavy partitions.
            score = float(partition.modularity) + 0.25 * coverage - 0.2 * singleton_frac

            if score > best_score:
                best_score = score
                best_membership = membership
                best_quantile = quantile

        if best_membership is not None:
            self.chosen_threshold_quantile_ = best_quantile
            logger.info(
                "Adaptive Leiden threshold selected quantile=%.2f (score=%.4f).",
                best_quantile,
                best_score,
            )
            return best_membership

        # Final fallback: use all positive edges.
        edges, weights = self._build_weighted_edges(cooccurrence=cooccurrence, quantile=0.0)
        if not edges:
            logger.warning(
                "No positive co-occurrence edges at all; "
                "assigning each gene to its own program."
            )
            self.chosen_threshold_quantile_ = 0.0
            return np.arange(n, dtype=np.intp)

        g = ig.Graph(n=n, edges=edges, directed=False)
        g.es["weight"] = weights

        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights=g.es["weight"],
            resolution_parameter=self.resolution,
            seed=self.random_state if self.random_state is not None else 0,
        )
        self.chosen_threshold_quantile_ = 0.0
        return np.array(partition.membership, dtype=np.intp)

    def _consensus_hierarchical(self, cooccurrence: np.ndarray) -> np.ndarray:
        """Agglomerative clustering on the co-occurrence matrix.

        Parameters
        ----------
        cooccurrence : np.ndarray
            ``(n, n)``

        Returns
        -------
        np.ndarray
            Cluster labels.
        """
        n = cooccurrence.shape[0]
        if n == 0:
            return np.array([], dtype=np.intp)
        if n == 1:
            return np.array([0], dtype=np.intp)
        k = self.n_programs if self.n_programs is not None else max(2, n // 10)
        k = min(k, n)

        # Convert similarity to distance
        max_val = cooccurrence.max()
        if max_val > 0:
            dist = max_val - cooccurrence
        else:
            dist = np.ones_like(cooccurrence)
        np.fill_diagonal(dist, 0.0)

        agg = AgglomerativeClustering(
            n_clusters=k,
            metric="precomputed",
            linkage="average",
        )
        result: np.ndarray = np.asarray(agg.fit_predict(dist).astype(np.intp))
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_weighted_edges(
        self,
        cooccurrence: np.ndarray,
        quantile: float,
    ) -> tuple[list[tuple[int, int]], list[float]]:
        """Build weighted edges by thresholding co-occurrence at a quantile."""
        n = cooccurrence.shape[0]
        upper = cooccurrence[np.triu_indices(n, k=1)]
        positive = upper[upper > 0]
        cutoff = float(np.quantile(positive, quantile)) if positive.size > 0 else 0.0

        edges: list[tuple[int, int]] = []
        weights: list[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                w = float(cooccurrence[i, j])
                if w > 0 and w >= cutoff:
                    edges.append((i, j))
                    weights.append(w)
        return edges, weights

    def _compute_method_weights(
        self,
        methods: list[BaseProgramDiscovery],
        all_programs: list[dict[str, list[str]]],
        embeddings: np.ndarray,
        gene_to_idx: dict[str, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute non-negative method weights for co-occurrence aggregation."""
        n_methods = len(methods)
        if n_methods == 0:
            empty = np.array([], dtype=np.float64)
            return empty, empty

        if self.method_weighting == "uniform":
            uniform_weights = np.ones(n_methods, dtype=np.float64)
            uniform_coherence = np.ones(n_methods, dtype=np.float64)
            return uniform_weights, uniform_coherence

        weight_values: list[float] = []
        coherence_scores = np.zeros(n_methods, dtype=np.float64)
        if self.method_weighting == "quality":
            for method in methods:
                quality_score = 1.0
                getter = getattr(method, "get_quality_metrics", None)
                if callable(getter):
                    try:
                        metrics = getter()
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.warning(
                            "Failed to read quality metrics from %s: %s",
                            method.__class__.__name__,
                            exc,
                        )
                        metrics = {}
                    candidates: list[float] = []
                    if "stability_score" in metrics:
                        stability = float(metrics["stability_score"])
                        if np.isfinite(stability):
                            candidates.append(max(stability, 0.0))
                    if "silhouette_score" in metrics:
                        # Map silhouette from [-1, 1] to [0, 1].
                        silhouette = float(metrics["silhouette_score"])
                        if np.isfinite(silhouette):
                            candidates.append(np.clip((silhouette + 1.0) / 2.0, 0.0, 1.0))
                    if candidates:
                        quality_score = float(np.mean(candidates))
                weight_values.append(max(quality_score, 1e-3))
        elif self.method_weighting == "coherence":
            embeddings_norm = self._normalize_rows(np.asarray(embeddings, dtype=np.float64))
            for i, prog_dict in enumerate(all_programs):
                coherence = self._program_coherence(
                    programs=prog_dict,
                    embeddings_norm=embeddings_norm,
                    gene_to_idx=gene_to_idx,
                )
                coherence_scores[i] = coherence
                weight_values.append(max(coherence, 1e-3))
        else:
            raise ValueError(
                f"Unknown method_weighting='{self.method_weighting}'. "
                "Choose from {'uniform', 'quality', 'coherence'}."
            )

        arr = np.asarray(weight_values, dtype=np.float64)
        if float(arr.sum()) <= 0.0:
            arr = np.ones(n_methods, dtype=np.float64)

        floor = float(self.minimum_method_weight)
        if floor > 0.0:
            max_floor = 1.0 / n_methods
            floor = min(floor, max_floor)
            if floor * n_methods >= 1.0 - 1e-12:
                arr = np.ones(n_methods, dtype=np.float64)
            else:
                arr = floor + (1.0 - floor * n_methods) * (arr / float(arr.sum()))

        arr /= float(arr.sum())
        if self.method_weighting != "coherence":
            coherence_scores = arr.copy()
        return arr, coherence_scores

    @staticmethod
    def _normalize_rows(arr: np.ndarray) -> np.ndarray:
        """L2-normalize rows for cosine-based coherence computations."""
        x = np.asarray(arr, dtype=np.float64)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        out = np.asarray(x / norms, dtype=np.float64)
        return out

    @staticmethod
    def _program_coherence(
        programs: dict[str, list[str]],
        embeddings_norm: np.ndarray,
        gene_to_idx: dict[str, int],
    ) -> float:
        """Return mean cosine-to-centroid coherence over all non-trivial programs."""
        coherence_scores: list[float] = []
        for genes in programs.values():
            idx = [gene_to_idx[g] for g in genes if g in gene_to_idx]
            if len(idx) < 2:
                continue
            vecs = embeddings_norm[np.asarray(idx, dtype=np.intp)]
            centroid = vecs.mean(axis=0)
            cent_norm = float(np.linalg.norm(centroid))
            if cent_norm < 1e-12:
                continue
            centroid /= cent_norm
            coherence_scores.append(float(np.mean(vecs @ centroid)))

        if not coherence_scores:
            return 0.0
        return float(np.mean(coherence_scores))

    def _merge_small_programs(
        self, labels: np.ndarray, cooccurrence: np.ndarray
    ) -> np.ndarray:
        """Merge programs smaller than *min_program_size* into neighbours.

        Parameters
        ----------
        labels : np.ndarray
            Current cluster labels.
        cooccurrence : np.ndarray
            ``(n, n)`` co-occurrence matrix.

        Returns
        -------
        np.ndarray
            Updated labels.
        """
        labels = labels.copy()
        unique, counts = np.unique(labels, return_counts=True)
        small = unique[counts < self.min_program_size]
        large = unique[counts >= self.min_program_size]

        if len(large) == 0:
            # All programs are too small; keep the largest ones as seeds
            # and merge the rest into them.  Pick top-K by size, where K
            # is a reasonable fraction of unique labels.
            n_keep = max(1, len(unique) // 5)
            order = np.argsort(counts)[::-1]
            large = unique[order[:n_keep]]
            small = unique[order[n_keep:]]
            if len(small) == 0:
                # Re-label contiguously and return
                unique_new = sorted(set(labels))
                mapping = {old: new for new, old in enumerate(unique_new)}
                return np.array([mapping[val] for val in labels], dtype=np.intp)

        for s_label in small:
            s_mask = labels == s_label
            s_indices = np.where(s_mask)[0]

            best_target = large[0]
            best_score = -1.0
            for l_label in large:
                l_indices = np.where(labels == l_label)[0]
                score = cooccurrence[np.ix_(s_indices, l_indices)].mean()
                if score > best_score:
                    best_score = score
                    best_target = l_label

            labels[s_mask] = best_target

        # Re-label to be contiguous
        unique_new = sorted(set(labels))
        mapping = {old: new for new, old in enumerate(unique_new)}
        labels = np.array([mapping[val] for val in labels], dtype=np.intp)
        return labels

    def _build_programs(
        self,
        labels: np.ndarray,
        cooccurrence: np.ndarray,
        all_programs: list[dict[str, list[str]]],
        gene_to_idx: dict[str, int],
        method_weights: np.ndarray,
    ) -> None:
        """Build consensus programs with confidence scores.

        Parameters
        ----------
        labels : np.ndarray
            Consensus cluster labels.
        cooccurrence : np.ndarray
            ``(n, n)`` normalised co-occurrence matrix.
        all_programs : list[dict[str, list[str]]]
            Programs from each constituent method.
        gene_to_idx : dict[str, int]
            Gene name to index mapping.
        """
        assert self._gene_names is not None
        gene_names = self._gene_names
        unique_labels = sorted(set(labels))
        total_weight = float(method_weights.sum())

        programs: dict[str, list[str]] = {}
        scores: dict[str, list[tuple[str, float]]] = {}

        # For each method, build a reverse mapping: gene_idx -> set of
        # program names that contain it
        method_gene_programs: list[dict[int, set[str]]] = []
        for prog_dict in all_programs:
            gp: dict[int, set[str]] = {}
            for prog_name, genes in prog_dict.items():
                for g in genes:
                    if g in gene_to_idx:
                        idx = gene_to_idx[g]
                        gp.setdefault(idx, set()).add(prog_name)
            method_gene_programs.append(gp)

        for label in unique_labels:
            prog_name = f"consensus_{label}"
            indices = [i for i, lab in enumerate(labels) if lab == label]
            if not indices:
                continue

            gene_list = [gene_names[i] for i in indices]

            # Confidence: for each gene, count in how many methods it
            # co-occurs with at least one other member of this consensus
            # program in the same original program.
            confidences: list[float] = []
            for gene_idx in indices:
                agreement_weight = 0.0
                other_indices = set(indices) - {gene_idx}
                for m, gp in enumerate(method_gene_programs):
                    gene_progs = gp.get(gene_idx, set())
                    if not gene_progs:
                        continue
                    # Check if any other gene in this consensus cluster
                    # shares a program with this gene in method m
                    for other_idx in other_indices:
                        other_progs = gp.get(other_idx, set())
                        if gene_progs & other_progs:
                            agreement_weight += float(method_weights[m])
                            break
                confidences.append(agreement_weight / max(total_weight, 1e-12))

            scored = list(zip(gene_list, confidences))
            scored.sort(key=lambda t: t[1], reverse=True)

            programs[prog_name] = [t[0] for t in scored]
            scores[prog_name] = scored

        self.programs_ = programs
        self.program_scores_ = scores
        self.confidence_ = scores
