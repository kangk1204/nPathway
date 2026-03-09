"""Clustering-based gene program discovery.

This module discovers gene programs by clustering genes in embedding space.
Supported algorithms:

* **leiden** -- Community detection via the Leiden algorithm on a kNN graph.
* **spectral** -- Spectral clustering on the kNN affinity matrix.
* **kmeans** -- Standard k-means clustering.
* **hdbscan** -- Density-based hierarchical clustering (no *K* required).

Additional capabilities for publication-quality analysis:

* **Multi-resolution Leiden** -- Sweep resolutions, select optimal via
  modularity + biological coherence (``multi_resolution_leiden``).
* **Consensus K selection** -- Combined criterion using silhouette, gap
  statistic, and Calinski-Harabasz index with majority vote.
* **UMAP pre-processing** -- Optional dimensionality reduction before
  clustering for improved neighborhood structure.
* **Stability analysis** -- Subsampling-based Jaccard agreement across
  repeated clusterings (``compute_stability``).
* **Enhanced gene scoring** -- Per-gene silhouette confidence, connectivity
  score, and cosine-to-centroid score.
* **Comprehensive quality metrics** -- Silhouette, Calinski-Harabasz,
  Davies-Bouldin, and stability scores.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Literal

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from npathway.discovery.base import BaseProgramDiscovery

logger = logging.getLogger(__name__)

# Type alias for supported methods
ClusteringMethod = Literal["leiden", "spectral", "kmeans", "hdbscan"]

_METHODS_REQUIRING_K: set[str] = {"spectral", "kmeans"}


class ClusteringProgramDiscovery(BaseProgramDiscovery):
    """Discover gene programs by clustering gene embeddings.

    Parameters
    ----------
    method : {"leiden", "spectral", "kmeans", "hdbscan"}
        Clustering algorithm to use.
    n_programs : int | None
        Number of clusters / programs.  Required for ``spectral`` and
        ``kmeans``.  If ``None``, auto-selected via consensus criterion
        over *k_range*.  Ignored by ``hdbscan``.
    k_neighbors : int
        Number of nearest neighbours for the kNN graph (used by
        ``leiden`` and ``spectral``).
    resolution : float
        Resolution parameter for the Leiden algorithm.  Higher values
        produce more (smaller) communities.
    k_range : tuple[int, int]
        ``(min_k, max_k)`` range for automatic cluster count selection.
    min_cluster_size : int
        ``min_cluster_size`` parameter for HDBSCAN.
    random_state : int | None
        Random seed for reproducibility.
    use_umap : bool
        If ``True``, apply UMAP dimensionality reduction to embeddings
        before clustering.  Preserves local neighborhood structure and
        can improve clustering quality for high-dimensional embeddings.
    umap_n_components : int
        Number of UMAP dimensions when ``use_umap=True``.
    umap_n_neighbors : int
        Number of neighbors for UMAP graph construction.
    umap_min_dist : float
        Minimum distance parameter for UMAP embedding.

    Attributes
    ----------
    labels_ : np.ndarray | None
        Cluster labels per gene after fitting.
    quality_ : dict[str, float] | None
        Clustering quality metrics after fitting.
    gene_confidence_ : dict[str, dict[str, float]] | None
        Per-gene confidence scores keyed by ``{program: {gene: score}}``.
    umap_embeddings_ : np.ndarray | None
        UMAP-reduced embeddings if ``use_umap=True``; otherwise ``None``.
    """

    def __init__(
        self,
        method: ClusteringMethod = "leiden",
        n_programs: int | None = None,
        k_neighbors: int = 15,
        resolution: float = 1.0,
        k_range: tuple[int, int] = (5, 30),
        min_cluster_size: int = 5,
        random_state: int | None = 42,
        use_umap: bool = False,
        umap_n_components: int = 10,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.0,
    ) -> None:
        super().__init__()
        self.method: ClusteringMethod = method
        self.n_programs = n_programs
        self.k_neighbors = k_neighbors
        self.resolution = resolution
        self.k_range = k_range
        self.min_cluster_size = min_cluster_size
        self.random_state = random_state
        self.use_umap = use_umap
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist

        # Fitted state
        self.labels_: np.ndarray | None = None
        self.quality_: dict[str, float] | None = None
        self.gene_confidence_: dict[str, dict[str, float]] | None = None
        self.umap_embeddings_: np.ndarray | None = None
        self._embeddings: np.ndarray | None = None
        self._clustering_embeddings: np.ndarray | None = None
        self._gene_names: list[str] | None = None
        self._centroids: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        embeddings: np.ndarray,
        gene_names: list[str],
        **kwargs: object,
    ) -> "ClusteringProgramDiscovery":
        """Cluster genes in embedding space to discover gene programs.

        Parameters
        ----------
        embeddings : np.ndarray
            Gene embedding matrix of shape ``(n_genes, n_dims)``.
        gene_names : list[str]
            Gene identifiers aligned with *embeddings* rows.
        **kwargs : object
            Ignored.  Present for API compatibility.

        Returns
        -------
        ClusteringProgramDiscovery
            Fitted instance (``self``).
        """
        embeddings = np.asarray(embeddings, dtype=np.float64)
        if embeddings.ndim != 2:
            raise ValueError(
                f"Expected 2-D embeddings, got shape {embeddings.shape}"
            )
        if embeddings.shape[0] != len(gene_names):
            raise ValueError(
                f"Mismatch: {embeddings.shape[0]} embeddings vs "
                f"{len(gene_names)} gene names."
            )

        self._embeddings = embeddings
        self._gene_names = list(gene_names)

        # Optional UMAP pre-processing
        clustering_emb = self._apply_umap(embeddings) if self.use_umap else embeddings
        self._clustering_embeddings = clustering_emb

        # Dispatch
        from collections.abc import Callable as _Callable
        dispatch: dict[str, _Callable[[np.ndarray], None]] = {
            "leiden": self._fit_leiden,
            "spectral": self._fit_spectral,
            "kmeans": self._fit_kmeans,
            "hdbscan": self._fit_hdbscan,
        }
        if self.method not in dispatch:
            raise ValueError(
                f"Unknown method '{self.method}'. "
                f"Choose from {sorted(dispatch.keys())}."
            )
        dispatch[self.method](clustering_emb)

        # Build programs & scores from labels
        self._build_programs()
        self._compute_quality(clustering_emb)
        self._compute_gene_confidence(clustering_emb)
        return self

    def get_programs(self) -> dict[str, list[str]]:
        """Return discovered gene programs as ``{name: [genes]}``.

        Returns
        -------
        dict[str, list[str]]
            Gene program mapping.
        """
        self._check_is_fitted()
        assert self.programs_ is not None
        return dict(self.programs_)

    def get_program_scores(self) -> dict[str, list[tuple[str, float]]]:
        """Return gene membership scores per program.

        Scores are cosine similarities between each gene embedding and
        the cluster centroid, rescaled to ``[0, 1]``.

        Returns
        -------
        dict[str, list[tuple[str, float]]]
            ``{program_name: [(gene, score), ...]}`` sorted descending.
        """
        self._check_is_fitted()
        assert self.program_scores_ is not None
        return dict(self.program_scores_)

    def get_quality_metrics(self) -> dict[str, float]:
        """Return comprehensive clustering quality metrics.

        Returns
        -------
        dict[str, float]
            Dictionary containing ``silhouette_score``,
            ``calinski_harabasz_score``, ``davies_bouldin_score``,
            ``n_programs``, and ``n_noise``.
        """
        self._check_is_fitted()
        assert self.quality_ is not None
        return dict(self.quality_)

    def get_discriminative_programs(
        self,
        top_n: int = 50,
    ) -> dict[str, list[str]]:
        """Return refined programs using discriminative gene scoring.

        Each gene is scored by the margin between its cosine similarity to
        its own cluster centroid and the maximum cosine similarity to any
        other centroid.  This concentrates each program on genes that are
        both close to the program center AND far from competing programs,
        yielding tighter, more biologically specific gene lists.

        Parameters
        ----------
        top_n : int
            Maximum number of genes to retain per program.

        Returns
        -------
        dict[str, list[str]]
            ``{program_name: [gene, ...]}`` with at most *top_n* genes per
            program, sorted by discriminative score descending.
        """
        self._check_is_fitted()
        assert self._embeddings is not None
        assert self._gene_names is not None
        assert self.labels_ is not None
        assert self._centroids is not None

        embeddings = self._embeddings
        gene_names = self._gene_names
        labels = self.labels_
        centroids = self._centroids  # shape (n_programs, n_dims)

        if centroids.shape[0] == 0:
            return {}

        # Normalise embeddings and centroids for cosine similarity
        emb_norm = normalize(embeddings, norm="l2")  # (n_genes, d)
        cent_norm = normalize(centroids, norm="l2")  # (n_programs, d)

        # Cosine similarity of every gene to every centroid: (n_genes, n_programs)
        sim_matrix = emb_norm @ cent_norm.T

        unique_labels = sorted(set(labels))
        # Build index mapping: centroid index → program name (same insertion order)
        prog_names: list[str] = []
        for lbl in unique_labels:
            if lbl == -1:
                prog_names.append("noise")
            else:
                prog_names.append(f"program_{lbl}")

        discriminative: dict[str, list[str]] = {}
        for centroid_idx, (lbl, prog_name) in enumerate(
            zip(unique_labels, prog_names)
        ):
            gene_mask = labels == lbl
            gene_indices = np.where(gene_mask)[0]

            if len(gene_indices) == 0:
                continue

            # Cosine to own centroid
            own_sims = sim_matrix[gene_indices, centroid_idx]  # (n_cluster,)

            # Max cosine to any *other* centroid
            n_centroids = sim_matrix.shape[1]
            if n_centroids > 1:
                other_cols = np.array([c for c in range(n_centroids) if c != centroid_idx], dtype=np.intp)
                other_sims = sim_matrix[np.ix_(gene_indices, other_cols)]
                max_other = other_sims.max(axis=1)
            else:
                max_other = np.zeros(len(gene_indices))

            # Discriminative margin: own − max_other
            disc_scores = own_sims - max_other

            # Sort by discriminative score, keep top_n
            order = np.argsort(disc_scores)[::-1][:top_n]
            top_gene_names = [gene_names[gene_indices[i]] for i in order]
            discriminative[prog_name] = top_gene_names

        return discriminative

    def get_gene_confidence(self) -> dict[str, dict[str, float]]:
        """Return per-gene confidence scores for each program.

        Confidence scores combine silhouette-based confidence and
        connectivity-based scores.

        Returns
        -------
        dict[str, dict[str, float]]
            ``{program_name: {gene_name: confidence}}`` where confidence
            is in ``[0, 1]``.
        """
        self._check_is_fitted()
        if self.gene_confidence_ is None:
            return {}
        return {k: dict(v) for k, v in self.gene_confidence_.items()}

    # ------------------------------------------------------------------
    # Multi-resolution Leiden
    # ------------------------------------------------------------------

    def multi_resolution_leiden(
        self,
        embeddings: np.ndarray,
        gene_names: list[str],
        resolutions: list[float] | None = None,
        n_resolutions: int = 20,
        resolution_range: tuple[float, float] = (0.1, 3.0),
    ) -> dict[str, Any]:
        """Sweep Leiden resolutions and select optimal via modularity + coherence.

        This method runs the Leiden algorithm at multiple resolution values
        and selects the one maximizing a combined score of graph modularity
        and biological coherence (measured by mean intra-cluster cosine
        similarity).

        Parameters
        ----------
        embeddings : np.ndarray
            Gene embedding matrix of shape ``(n_genes, n_dims)``.
        gene_names : list[str]
            Gene identifiers aligned with *embeddings* rows.
        resolutions : list[float] | None
            Explicit list of resolution values to try.  If ``None``,
            ``n_resolutions`` values are sampled log-uniformly from
            *resolution_range*.
        n_resolutions : int
            Number of resolution values to sample when *resolutions* is
            ``None``.
        resolution_range : tuple[float, float]
            ``(min_res, max_res)`` range for sampling resolutions.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:

            - ``"best_resolution"`` : float -- optimal resolution value.
            - ``"best_modularity"`` : float -- modularity at best resolution.
            - ``"best_coherence"`` : float -- biological coherence at best.
            - ``"best_combined_score"`` : float -- combined selection score.
            - ``"n_communities"`` : int -- number of communities at best.
            - ``"resolution_sweep"`` : list[dict] -- per-resolution details.
            - ``"labels"`` : np.ndarray -- cluster labels at best resolution.
        """
        try:
            import igraph as ig
            import leidenalg
        except ImportError as exc:
            raise ImportError(
                "Multi-resolution Leiden requires 'igraph' and 'leidenalg'. "
                "Install them with: pip install igraph leidenalg"
            ) from exc

        embeddings = np.asarray(embeddings, dtype=np.float64)
        if embeddings.ndim != 2:
            raise ValueError(
                f"Expected 2-D embeddings, got shape {embeddings.shape}"
            )
        if embeddings.shape[0] != len(gene_names):
            raise ValueError(
                f"Mismatch: {embeddings.shape[0]} embeddings vs "
                f"{len(gene_names)} gene names."
            )

        # Store for _build_programs
        self._embeddings = embeddings
        self._gene_names = list(gene_names)

        # Optionally apply UMAP
        clustering_emb = self._apply_umap(embeddings) if self.use_umap else embeddings
        self._clustering_embeddings = clustering_emb

        # Build kNN graph once
        adj = self._build_knn_adjacency(clustering_emb)
        n = adj.shape[0]

        edges: list[tuple[int, int]] = []
        weights: list[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                w = adj[i, j]
                if w > 0:
                    edges.append((i, j))
                    weights.append(float(w))

        g = ig.Graph(n=n, edges=edges, directed=False)
        g.es["weight"] = weights

        # Build resolution schedule
        if resolutions is None:
            lo_r, hi_r = resolution_range
            resolutions = list(
                np.exp(np.linspace(np.log(lo_r), np.log(hi_r), n_resolutions))
            )

        # Cosine similarity matrix for coherence
        emb_norm = normalize(clustering_emb, norm="l2")
        cos_sim = emb_norm @ emb_norm.T

        sweep_results: list[dict[str, Any]] = []
        best_combined = -np.inf
        best_result: dict[str, Any] | None = None

        for res in resolutions:
            partition = leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                weights=g.es["weight"],
                resolution_parameter=res,
                seed=self.random_state if self.random_state is not None else 0,
            )

            labels = np.array(partition.membership, dtype=np.intp)
            n_communities = len(set(labels))
            modularity = float(partition.modularity)

            # Biological coherence: mean intra-cluster cosine similarity
            coherence_values: list[float] = []
            for c in set(labels):
                mask = labels == c
                n_c = int(mask.sum())
                if n_c < 2:
                    continue
                cluster_sim = cos_sim[np.ix_(mask, mask)]
                # Mean off-diagonal similarity
                off_diag_sum = float(cluster_sim.sum() - np.trace(cluster_sim))
                off_diag_mean = off_diag_sum / (n_c * (n_c - 1))
                coherence_values.append(off_diag_mean)

            coherence = float(np.mean(coherence_values)) if coherence_values else 0.0

            # Combined score: weighted sum of modularity and coherence
            # Modularity typically in [-0.5, 1], coherence in [0, 1]
            combined = 0.5 * modularity + 0.5 * coherence

            result = {
                "resolution": float(res),
                "n_communities": n_communities,
                "modularity": modularity,
                "coherence": coherence,
                "combined_score": combined,
                "labels": labels,
            }
            sweep_results.append(result)

            if combined > best_combined:
                best_combined = combined
                best_result = result

        assert best_result is not None

        # Fit the model with the best resolution
        self.labels_ = best_result["labels"]
        self.resolution = best_result["resolution"]
        self._build_programs()
        self._compute_quality(clustering_emb)
        self._compute_gene_confidence(clustering_emb)

        logger.info(
            "Multi-resolution Leiden: best resolution=%.4f, "
            "modularity=%.4f, coherence=%.4f, n_communities=%d.",
            best_result["resolution"],
            best_result["modularity"],
            best_result["coherence"],
            best_result["n_communities"],
        )

        return {
            "best_resolution": best_result["resolution"],
            "best_modularity": best_result["modularity"],
            "best_coherence": best_result["coherence"],
            "best_combined_score": best_result["combined_score"],
            "n_communities": best_result["n_communities"],
            "resolution_sweep": [
                {k: v for k, v in r.items() if k != "labels"} for r in sweep_results
            ],
            "labels": best_result["labels"],
        }

    # ------------------------------------------------------------------
    # Stability Analysis
    # ------------------------------------------------------------------

    def compute_stability(
        self,
        embeddings: np.ndarray,
        gene_names: list[str],
        n_iterations: int = 20,
        subsample_fraction: float = 0.8,
    ) -> dict[str, Any]:
        """Measure clustering stability via subsampled Jaccard agreement.

        Repeatedly subsamples a fraction of genes, re-runs clustering,
        and measures Jaccard similarity between the resulting programs
        and those from the full dataset.  High stability indicates robust
        program definitions.

        Parameters
        ----------
        embeddings : np.ndarray
            Gene embedding matrix of shape ``(n_genes, n_dims)``.
        gene_names : list[str]
            Gene identifiers aligned with *embeddings* rows.
        n_iterations : int
            Number of subsampling iterations.
        subsample_fraction : float
            Fraction of genes to retain in each subsample (0 < frac < 1).

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:

            - ``"mean_jaccard"`` : float -- Mean pairwise Jaccard across all
              iterations and programs.
            - ``"std_jaccard"`` : float -- Std of pairwise Jaccard.
            - ``"per_iteration_jaccard"`` : list[float] -- Mean Jaccard
              per iteration.
            - ``"stability_score"`` : float -- Summary stability in [0, 1].
        """
        embeddings = np.asarray(embeddings, dtype=np.float64)
        if embeddings.ndim != 2:
            raise ValueError(
                f"Expected 2-D embeddings, got shape {embeddings.shape}"
            )

        # First, fit on full dataset to get reference programs
        if self.programs_ is None:
            self.fit(embeddings, gene_names)

        assert self.programs_ is not None
        ref_programs = self.programs_

        rng = np.random.default_rng(
            self.random_state if self.random_state is not None else 42
        )
        n_genes = embeddings.shape[0]
        if n_genes <= 1:
            return {
                "mean_jaccard": 1.0,
                "std_jaccard": 0.0,
                "per_iteration_jaccard": [1.0] * max(n_iterations, 0),
                "stability_score": 1.0,
            }
        n_subsample = min(n_genes, max(int(n_genes * subsample_fraction), 2))

        per_iteration_jaccard: list[float] = []

        for iteration in range(n_iterations):
            # Subsample
            indices = rng.choice(n_genes, size=n_subsample, replace=False)
            indices.sort()
            sub_emb = embeddings[indices]
            sub_names = [gene_names[i] for i in indices]

            # Create a fresh instance with same parameters
            sub_model = ClusteringProgramDiscovery(
                method=self.method,
                n_programs=self.n_programs,
                k_neighbors=min(self.k_neighbors, n_subsample - 1),
                resolution=self.resolution,
                k_range=self.k_range,
                min_cluster_size=self.min_cluster_size,
                random_state=(
                    (self.random_state + iteration + 1)
                    if self.random_state is not None
                    else None
                ),
                use_umap=self.use_umap,
                umap_n_components=min(
                    self.umap_n_components, n_subsample - 2
                ),
                umap_n_neighbors=min(self.umap_n_neighbors, n_subsample - 1),
                umap_min_dist=self.umap_min_dist,
            )

            try:
                sub_model.fit(sub_emb, sub_names)
            except Exception as exc:
                logger.warning(
                    "Stability iteration %d failed: %s", iteration, exc
                )
                continue

            if sub_model.programs_ is None:
                continue

            sub_programs = sub_model.programs_

            # Compute best-match Jaccard between ref and sub programs
            jaccard_scores = self._compute_pairwise_jaccard(
                ref_programs, sub_programs
            )
            per_iteration_jaccard.append(jaccard_scores)

        if not per_iteration_jaccard:
            return {
                "mean_jaccard": 0.0,
                "std_jaccard": 0.0,
                "per_iteration_jaccard": [],
                "stability_score": 0.0,
            }

        arr = np.array(per_iteration_jaccard)
        mean_j = float(np.mean(arr))
        std_j = float(np.std(arr))

        # Store stability in quality metrics
        if self.quality_ is not None:
            self.quality_["stability_score"] = mean_j

        logger.info(
            "Stability analysis: mean_jaccard=%.4f +/- %.4f over %d iterations.",
            mean_j,
            std_j,
            len(per_iteration_jaccard),
        )

        return {
            "mean_jaccard": mean_j,
            "std_jaccard": std_j,
            "per_iteration_jaccard": per_iteration_jaccard,
            "stability_score": mean_j,
        }

    # ------------------------------------------------------------------
    # Clustering back-ends
    # ------------------------------------------------------------------

    def _fit_leiden(self, embeddings: np.ndarray) -> None:
        """Leiden community detection on a kNN graph."""
        try:
            import igraph as ig
            import leidenalg
        except ImportError as exc:
            raise ImportError(
                "Leiden clustering requires 'igraph' and 'leidenalg'. "
                "Install them with: pip install igraph leidenalg"
            ) from exc

        adj = self._build_knn_adjacency(embeddings)
        n = adj.shape[0]

        # Build igraph from the symmetric adjacency
        edges: list[tuple[int, int]] = []
        weights: list[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                w = adj[i, j]
                if w > 0:
                    edges.append((i, j))
                    weights.append(float(w))

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

    def _fit_spectral(self, embeddings: np.ndarray) -> None:
        """Spectral clustering on the kNN affinity matrix."""
        from sklearn.cluster import SpectralClustering

        k = self._resolve_k(embeddings)
        adj = self._build_knn_adjacency(embeddings)

        sc = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            random_state=self.random_state,
            assign_labels="kmeans",
        )
        self.labels_ = sc.fit_predict(adj)

    def _fit_kmeans(self, embeddings: np.ndarray) -> None:
        """Standard k-means clustering."""
        from sklearn.cluster import KMeans

        k = self._resolve_k(embeddings)
        km = KMeans(
            n_clusters=k,
            n_init=10,
            max_iter=300,
            random_state=self.random_state,
        )
        self.labels_ = km.fit_predict(embeddings)

    def _fit_hdbscan(self, embeddings: np.ndarray) -> None:
        """Density-based clustering with HDBSCAN."""
        try:
            import hdbscan as _hdbscan
        except ImportError as exc:
            raise ImportError(
                "HDBSCAN clustering requires the 'hdbscan' package. "
                "Install it with: pip install hdbscan"
            ) from exc

        clusterer = _hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric="euclidean",
            core_dist_n_jobs=1,
        )
        self.labels_ = clusterer.fit_predict(embeddings)

        # HDBSCAN assigns -1 to noise points.  We place them in a
        # dedicated "noise" cluster so that no genes are lost.
        if (self.labels_ == -1).any():
            n_noise = int((self.labels_ == -1).sum())
            logger.info(
                "HDBSCAN: %d / %d genes classified as noise.",
                n_noise,
                len(self.labels_),
            )

    # ------------------------------------------------------------------
    # UMAP pre-processing
    # ------------------------------------------------------------------

    def _apply_umap(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce dimensionality with UMAP using cosine metric.

        UMAP preserves local neighborhood structure in the embedding
        space, yielding tighter clusters and improved separation.

        Parameters
        ----------
        embeddings : np.ndarray
            ``(n_genes, n_dims)`` embedding matrix.

        Returns
        -------
        np.ndarray
            ``(n_genes, umap_n_components)`` reduced embedding matrix.
        """
        try:
            import umap
        except ImportError as exc:
            raise ImportError(
                "UMAP pre-processing requires the 'umap-learn' package. "
                "Install it with: pip install umap-learn"
            ) from exc

        n_genes = embeddings.shape[0]
        n_components = min(self.umap_n_components, n_genes - 2)
        n_neighbors = min(self.umap_n_neighbors, n_genes - 1)

        if n_components < 2:
            logger.warning(
                "Too few genes (%d) for UMAP; skipping reduction.", n_genes
            )
            self.umap_embeddings_ = None
            return embeddings

        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=self.umap_min_dist,
            metric="cosine",
            random_state=self.random_state,
        )

        reduced = reducer.fit_transform(embeddings)
        self.umap_embeddings_ = reduced
        logger.info(
            "UMAP reduced embeddings from %d to %d dimensions.",
            embeddings.shape[1],
            n_components,
        )
        umap_result: np.ndarray = np.asarray(reduced)
        return umap_result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_knn_adjacency(self, embeddings: np.ndarray) -> np.ndarray:
        """Build a symmetric kNN adjacency matrix with cosine weights.

        Parameters
        ----------
        embeddings : np.ndarray
            ``(n_genes, n_dims)`` embedding matrix.

        Returns
        -------
        np.ndarray
            Symmetric ``(n_genes, n_genes)`` adjacency matrix.
        """
        n = embeddings.shape[0]
        k = min(self.k_neighbors, n - 1)
        if k < 1:
            raise ValueError(
                f"Cannot build a kNN graph with only {n} gene(s)."
            )

        emb_norm = normalize(embeddings, norm="l2")
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
        nn.fit(emb_norm)
        distances, indices = nn.kneighbors(emb_norm)

        # Convert cosine distance to similarity
        adj = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for rank in range(1, k + 1):  # skip self (rank 0)
                j = indices[i, rank]
                sim = 1.0 - distances[i, rank]
                adj[i, j] = max(sim, 0.0)
                adj[j, i] = max(sim, 0.0)

        return adj

    def _resolve_k(self, embeddings: np.ndarray) -> int:
        """Return *n_programs* or auto-select via consensus criterion.

        Uses silhouette score, gap statistic, and Calinski-Harabasz
        index, then takes a majority vote to determine optimal K.

        Parameters
        ----------
        embeddings : np.ndarray
            Gene embedding matrix.

        Returns
        -------
        int
            Chosen number of clusters.
        """
        if self.n_programs is not None:
            return self.n_programs

        lo, hi = self.k_range
        n = embeddings.shape[0]
        if n <= 1:
            return 1
        hi = min(max(hi, 1), n)
        lo = min(max(lo, 2), n)
        if lo > hi:
            logger.warning(
                "k_range (%d, %d) invalid for %d genes; defaulting to k=%d.",
                self.k_range[0],
                self.k_range[1],
                n,
                hi,
            )
            return hi

        logger.info(
            "Auto-selecting K via consensus criterion [%d, %d].", lo, hi
        )

        from sklearn.cluster import KMeans

        k_values = list(range(lo, hi + 1))
        sil_scores: list[float] = []
        ch_scores: list[float] = []
        inertias: list[float] = []
        labels_per_k: dict[int, np.ndarray] = {}

        for k in k_values:
            km = KMeans(
                n_clusters=k,
                n_init=5,
                max_iter=200,
                random_state=self.random_state,
            )
            labels = km.fit_predict(embeddings)
            labels_per_k[k] = labels
            n_unique = len(set(labels))
            if n_unique < 2 or n_unique >= len(labels):
                sil_scores.append(-1.0)
                ch_scores.append(0.0)
                inertias.append(float(km.inertia_))
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sil_scores.append(float(silhouette_score(embeddings, labels)))
                ch_scores.append(
                    float(calinski_harabasz_score(embeddings, labels))
                )
            inertias.append(float(km.inertia_))

        # 1. Silhouette: pick K with highest silhouette
        best_sil_idx = int(np.argmax(sil_scores))
        k_sil = k_values[best_sil_idx]

        # 2. Calinski-Harabasz: pick K with highest CH
        best_ch_idx = int(np.argmax(ch_scores))
        k_ch = k_values[best_ch_idx]

        # 3. Gap statistic (simplified: compare to uniform reference)
        k_gap = self._compute_gap_statistic_best_k(
            embeddings, k_values, inertias
        )

        # Consensus vote
        votes: dict[int, int] = {}
        for k_vote in [k_sil, k_ch, k_gap]:
            votes[k_vote] = votes.get(k_vote, 0) + 1

        # Sort by votes (descending), break ties by silhouette score
        candidates = sorted(
            votes.keys(),
            key=lambda kk: (
                votes[kk],
                sil_scores[k_values.index(kk)] if kk in k_values else -1.0,
            ),
            reverse=True,
        )
        best_k = candidates[0]

        logger.info(
            "Consensus K selection: silhouette->%d, CH->%d, gap->%d. "
            "Selected K=%d (votes: %s).",
            k_sil,
            k_ch,
            k_gap,
            best_k,
            votes,
        )
        return best_k

    def _compute_gap_statistic_best_k(
        self,
        embeddings: np.ndarray,
        k_values: list[int],
        inertias: list[float],
    ) -> int:
        """Compute gap statistic and return best K.

        The gap statistic compares the observed within-cluster dispersion
        to that expected under a uniform reference distribution.  The
        optimal K is the smallest value where gap(K) >= gap(K+1) - s(K+1),
        using the standard gap criterion from Tibshirani et al. (2001).

        Parameters
        ----------
        embeddings : np.ndarray
            Gene embedding matrix.
        k_values : list[int]
            Candidate K values.
        inertias : list[float]
            Within-cluster sum of squares for each K (from real data).

        Returns
        -------
        int
            Best K according to gap statistic.
        """
        from sklearn.cluster import KMeans

        n_ref = 10
        rng = np.random.default_rng(
            self.random_state if self.random_state is not None else 42
        )

        # Generate reference data: uniform in bounding box of embeddings
        mins = embeddings.min(axis=0)
        maxs = embeddings.max(axis=0)

        ref_inertias_per_k: list[list[float]] = [[] for _ in k_values]

        for _ in range(n_ref):
            ref_data = rng.uniform(
                low=mins, high=maxs, size=embeddings.shape
            )
            for ki, k in enumerate(k_values):
                km = KMeans(
                    n_clusters=k,
                    n_init=3,
                    max_iter=100,
                    random_state=self.random_state,
                )
                km.fit(ref_data)
                ref_inertias_per_k[ki].append(float(km.inertia_))

        # Compute gap = E[log(W_ref)] - log(W_obs)
        gaps: list[float] = []
        gap_stds: list[float] = []
        for ki in range(len(k_values)):
            log_ref = np.log(np.array(ref_inertias_per_k[ki]) + 1e-12)
            gap = float(np.mean(log_ref)) - np.log(inertias[ki] + 1e-12)
            sd = float(np.std(log_ref)) * np.sqrt(1.0 + 1.0 / n_ref)
            gaps.append(gap)
            gap_stds.append(sd)

        # Tibshirani criterion: smallest K where gap(K) >= gap(K+1) - s(K+1)
        for ki in range(len(k_values) - 1):
            if gaps[ki] >= gaps[ki + 1] - gap_stds[ki + 1]:
                return k_values[ki]

        # Fallback: K with largest gap
        return k_values[int(np.argmax(gaps))]

    def _build_programs(self) -> None:
        """Populate ``programs_`` and ``program_scores_`` from ``labels_``."""
        assert self.labels_ is not None
        assert self._gene_names is not None
        assert self._embeddings is not None

        labels = self.labels_
        gene_names = self._gene_names
        embeddings = self._embeddings

        unique_labels = sorted(set(labels))
        programs: dict[str, list[str]] = {}
        scores: dict[str, list[tuple[str, float]]] = {}
        centroids: list[np.ndarray] = []

        for label in unique_labels:
            mask = labels == label
            if label == -1:
                prog_name = "noise"
            else:
                prog_name = f"program_{label}"

            cluster_genes = [gene_names[i] for i in range(len(gene_names)) if mask[i]]
            cluster_embeddings = embeddings[mask]

            if len(cluster_genes) == 0:
                continue

            centroid = cluster_embeddings.mean(axis=0)
            centroids.append(centroid)

            # Cosine similarity to centroid
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-12)
            emb_norms = normalize(cluster_embeddings, norm="l2")
            sims = emb_norms @ centroid_norm

            # Rescale to [0, 1]
            sim_min = float(sims.min())
            sim_max = float(sims.max())
            if sim_max - sim_min > 1e-12:
                sims_scaled = (sims - sim_min) / (sim_max - sim_min)
            else:
                sims_scaled = np.ones_like(sims)

            scored = [
                (g, float(s))
                for g, s in zip(cluster_genes, sims_scaled)
            ]
            scored.sort(key=lambda t: t[1], reverse=True)

            programs[prog_name] = [t[0] for t in scored]
            scores[prog_name] = scored

        self.programs_ = programs
        self.program_scores_ = scores
        self._centroids = (
            np.vstack(centroids) if centroids else np.empty((0, embeddings.shape[1]))
        )

        # Build soft membership: cosine similarity to every centroid
        self._build_soft_programs(embeddings, gene_names)

    def _build_soft_programs(
        self, embeddings: np.ndarray, gene_names: list[str]
    ) -> None:
        """Build soft membership from cosine similarity to centroids.

        For each gene, computes the cosine similarity to every program
        centroid, then applies softmax to obtain a probability distribution
        over programs.  This allows genes to have non-zero membership in
        multiple programs.

        Parameters
        ----------
        embeddings : np.ndarray
            Gene embedding matrix ``(n_genes, n_dims)``.
        gene_names : list[str]
            Gene identifiers aligned with *embeddings* rows.
        """
        if self._centroids is None or self._centroids.shape[0] == 0:
            return
        if self.programs_ is None:
            return

        prog_names = list(self.programs_.keys())
        centroids = self._centroids  # (n_programs, n_dims)

        # Cosine similarity: (n_genes, n_programs)
        emb_norm = normalize(embeddings, norm="l2")
        cent_norm = normalize(centroids, norm="l2")
        sim_matrix = emb_norm @ cent_norm.T  # (n_genes, n_programs)

        # Softmax with temperature=1.0 to get probabilities
        # Shift for numerical stability
        sim_shifted = sim_matrix - sim_matrix.max(axis=1, keepdims=True)
        exp_sim = np.exp(sim_shifted)
        soft_weights = exp_sim / exp_sim.sum(axis=1, keepdims=True)  # (n_genes, n_programs)

        soft: dict[str, dict[str, float]] = {}
        for p_idx, prog_name in enumerate(prog_names):
            soft[prog_name] = {
                gene_names[g_idx]: float(soft_weights[g_idx, p_idx])
                for g_idx in range(len(gene_names))
            }

        self.soft_programs_ = soft

    def _compute_quality(self, embeddings: np.ndarray) -> None:
        """Compute and store comprehensive clustering quality metrics.

        Computes silhouette score, Calinski-Harabasz index, and
        Davies-Bouldin index for the current clustering.

        Parameters
        ----------
        embeddings : np.ndarray
            Embedding matrix used for clustering (may be UMAP-reduced).
        """
        assert self.labels_ is not None
        quality: dict[str, float] = {}
        # Exclude noise points for quality metrics
        mask = self.labels_ != -1
        valid_labels = self.labels_[mask]
        valid_emb = embeddings[mask]
        n_unique = len(set(valid_labels))

        if n_unique >= 2 and len(valid_labels) > n_unique:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                quality["silhouette_score"] = float(
                    silhouette_score(valid_emb, valid_labels)
                )
                quality["calinski_harabasz_score"] = float(
                    calinski_harabasz_score(valid_emb, valid_labels)
                )
                quality["davies_bouldin_score"] = float(
                    davies_bouldin_score(valid_emb, valid_labels)
                )
        else:
            quality["silhouette_score"] = float("nan")
            quality["calinski_harabasz_score"] = float("nan")
            quality["davies_bouldin_score"] = float("nan")

        quality["n_programs"] = float(len(self.programs_)) if self.programs_ else 0.0
        quality["n_noise"] = float(int((self.labels_ == -1).sum()))

        self.quality_ = quality
        logger.info("Clustering quality: %s", quality)

    def _compute_gene_confidence(self, embeddings: np.ndarray) -> None:
        """Compute per-gene confidence scores combining multiple criteria.

        For each gene, computes:

        1. **Silhouette confidence** -- Per-sample silhouette rescaled
           to ``[0, 1]``.  Measures how well a gene fits its assigned
           cluster versus the nearest alternative.
        2. **Connectivity score** -- Mean cosine similarity to all other
           genes in the same program.  Measures intra-cluster cohesion.

        The final confidence is the average of both scores.

        Parameters
        ----------
        embeddings : np.ndarray
            Embedding matrix used for clustering (may be UMAP-reduced).
        """
        assert self.labels_ is not None
        assert self._gene_names is not None
        assert self._embeddings is not None

        labels = self.labels_
        gene_names = self._gene_names
        # Use original embeddings for scoring (better semantic space)
        orig_emb = self._embeddings
        n = len(gene_names)

        # Skip if too few clusters
        mask = labels != -1
        valid_labels = labels[mask]
        n_unique = len(set(valid_labels))

        if n_unique < 2:
            self.gene_confidence_ = {}
            return

        # 1. Per-gene silhouette scores
        if n_unique < len(valid_labels):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sil_per_gene = silhouette_samples(embeddings, labels)
            sil_rescaled = (sil_per_gene + 1.0) / 2.0
        else:
            sil_rescaled = np.ones(n, dtype=np.float64)

        # 2. Connectivity score: mean cosine similarity within cluster
        emb_norm = normalize(orig_emb, norm="l2")
        cos_sim = emb_norm @ emb_norm.T

        connectivity = np.zeros(n, dtype=np.float64)
        for label in set(labels):
            cluster_mask = labels == label
            n_c = int(cluster_mask.sum())
            if n_c < 2:
                connectivity[cluster_mask] = 1.0
                continue
            cluster_indices = np.where(cluster_mask)[0]
            for idx in cluster_indices:
                other_indices = cluster_indices[cluster_indices != idx]
                connectivity[idx] = float(np.mean(cos_sim[idx, other_indices]))

        # Rescale connectivity to [0, 1]
        conn_min = float(connectivity.min())
        conn_max = float(connectivity.max())
        if conn_max - conn_min > 1e-12:
            conn_rescaled = (connectivity - conn_min) / (conn_max - conn_min)
        else:
            conn_rescaled = np.ones_like(connectivity)

        # Combined confidence: average of silhouette and connectivity
        combined = (sil_rescaled + conn_rescaled) / 2.0

        # Organize by program
        gene_confidence: dict[str, dict[str, float]] = {}
        for label in sorted(set(labels)):
            if label == -1:
                prog_name = "noise"
            else:
                prog_name = f"program_{label}"

            cluster_mask = labels == label
            conf: dict[str, float] = {}
            for i in range(n):
                if cluster_mask[i]:
                    conf[gene_names[i]] = float(np.clip(combined[i], 0.0, 1.0))
            if conf:
                gene_confidence[prog_name] = conf

        self.gene_confidence_ = gene_confidence

    @staticmethod
    def _compute_pairwise_jaccard(
        ref_programs: dict[str, list[str]],
        sub_programs: dict[str, list[str]],
    ) -> float:
        """Compute best-match mean Jaccard between two program sets.

        For each reference program, finds the sub-program with highest
        Jaccard overlap, then averages across all reference programs.

        Parameters
        ----------
        ref_programs : dict[str, list[str]]
            Reference (full-dataset) programs.
        sub_programs : dict[str, list[str]]
            Subsampled programs.

        Returns
        -------
        float
            Mean best-match Jaccard similarity.
        """
        if not ref_programs or not sub_programs:
            return 0.0

        ref_sets = {k: set(v) for k, v in ref_programs.items()}
        sub_sets = {k: set(v) for k, v in sub_programs.items()}

        best_jaccards: list[float] = []
        for ref_name, ref_set in ref_sets.items():
            best_j = 0.0
            for sub_name, sub_set in sub_sets.items():
                intersection = len(ref_set & sub_set)
                union = len(ref_set | sub_set)
                if union > 0:
                    j = intersection / union
                    best_j = max(best_j, j)
            best_jaccards.append(best_j)

        return float(np.mean(best_jaccards)) if best_jaccards else 0.0
