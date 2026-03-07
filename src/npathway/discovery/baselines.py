"""Baseline gene program discovery methods for benchmarking.

Implements simplified versions of existing SOTA approaches for fair comparison:
- WGCNA-like: Weighted correlation network analysis on expression data
- cNMF-like: Consensus NMF on expression data
- Random: Null baseline with random gene assignments
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

from npathway.discovery.base import BaseProgramDiscovery

logger = logging.getLogger(__name__)


class WGCNAProgramDiscovery(BaseProgramDiscovery):
    """WGCNA-like gene program discovery using weighted correlation networks.

    Constructs a gene-gene correlation network from expression data, raises it to
    a soft-thresholding power, and applies hierarchical clustering to identify
    gene modules (programs).

    Parameters
    ----------
    n_programs : int | None
        Target number of programs. If None, auto-determined by dynamic tree cut.
    soft_power : int
        Soft-thresholding power for scale-free topology fitting.
    min_module_size : int
        Minimum number of genes per module.
    merge_threshold : float
        Merge modules with correlation above this threshold.
    """

    def __init__(
        self,
        n_programs: int | None = None,
        soft_power: int = 6,
        min_module_size: int = 10,
        merge_threshold: float = 0.8,
        random_state: int | None = 42,
    ) -> None:
        super().__init__()
        self.n_programs = n_programs
        self.soft_power = soft_power
        self.min_module_size = min_module_size
        self.merge_threshold = merge_threshold
        self.random_state = random_state

        self._gene_names: list[str] = []
        self._labels: np.ndarray = np.array([])

    def fit(
        self,
        embeddings: np.ndarray,
        gene_names: list[str],
        **kwargs: object,
    ) -> "WGCNAProgramDiscovery":
        """Fit WGCNA-like modules from expression matrix.

        Parameters
        ----------
        embeddings : np.ndarray
            Expression matrix of shape (n_cells, n_genes) or gene correlation matrix
            of shape (n_genes, n_genes).
        gene_names : list[str]
            Gene names corresponding to columns.

        Returns
        -------
        WGCNAProgramDiscovery
            Fitted instance.
        """
        self._gene_names = list(gene_names)
        n_genes = len(gene_names)

        # Compute gene-gene correlation
        if embeddings.shape[0] == embeddings.shape[1] == n_genes:
            cor_matrix = embeddings
        else:
            cor_matrix = np.corrcoef(embeddings.T)

        # Handle NaN correlations
        cor_matrix = np.nan_to_num(cor_matrix, nan=0.0)

        # Compute adjacency (soft thresholding)
        adjacency = np.abs(cor_matrix) ** self.soft_power

        # Topological overlap matrix (TOM) - simplified
        # TOM_{ij} = (sum_u(a_iu * a_uj) + a_ij) / (min(k_i, k_j) + 1 - a_ij)
        connectivity = adjacency.sum(axis=1) - np.diag(adjacency)
        numerator = adjacency @ adjacency + adjacency
        min_k = np.minimum(connectivity[:, None], connectivity[None, :])
        denominator = min_k + 1.0 - adjacency
        denominator[denominator < 1e-10] = 1e-10
        tom = numerator / denominator
        np.fill_diagonal(tom, 1.0)

        # Distance from TOM
        dist_tom = 1.0 - tom
        dist_tom = np.clip(dist_tom, 0, None)

        # Hierarchical clustering
        if self.n_programs is not None:
            n_clusters = self.n_programs
        else:
            n_clusters = max(2, n_genes // self.min_module_size)

        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage="average",
        )
        labels = clustering.fit_predict(dist_tom)
        self._labels = labels

        # Build programs, filtering small modules
        programs: dict[str, list[str]] = {}
        scores: dict[str, list[tuple[str, float]]] = {}
        for label in sorted(set(labels)):
            indices = [i for i, lab in enumerate(labels) if lab == label]
            if len(indices) < self.min_module_size:
                continue
            prog_name = f"wgcna_module_{len(programs)}"
            gene_list = [gene_names[i] for i in indices]

            # Score: mean connectivity within module
            sub_adj = adjacency[np.ix_(indices, indices)]
            mean_conn = sub_adj.mean(axis=1)
            if mean_conn.max() > 0:
                norm_conn = mean_conn / mean_conn.max()
            else:
                norm_conn = np.ones(len(indices))

            programs[prog_name] = gene_list
            scores[prog_name] = [
                (gene_list[j], float(norm_conn[j])) for j in range(len(gene_list))
            ]

        # Handle unassigned genes (from small modules) - assign to nearest
        assigned_genes = set()
        for genes in programs.values():
            assigned_genes.update(genes)

        if not programs:
            # Fallback: create a single program with all genes
            programs["wgcna_module_0"] = list(gene_names)
            scores["wgcna_module_0"] = [(g, 1.0) for g in gene_names]

        self.programs_ = programs
        self.program_scores_ = scores
        logger.info(
            f"WGCNA-like discovery: {len(programs)} modules, "
            f"covering {len(assigned_genes)}/{n_genes} genes"
        )
        return self

    def get_programs(self) -> dict[str, list[str]]:
        """Return discovered gene programs."""
        self._check_is_fitted()
        assert self.programs_ is not None
        return dict(self.programs_)

    def get_program_scores(self) -> dict[str, list[tuple[str, float]]]:
        """Return weighted gene-program membership scores."""
        self._check_is_fitted()
        assert self.program_scores_ is not None
        return dict(self.program_scores_)


class CNMFProgramDiscovery(BaseProgramDiscovery):
    """cNMF-like gene program discovery using non-negative matrix factorization.

    Applies NMF to expression data to identify gene expression programs.
    Runs multiple NMF iterations with different seeds and takes consensus.

    Parameters
    ----------
    n_programs : int
        Number of programs (NMF components) to discover.
    n_iter : int
        Number of NMF runs for consensus.
    top_n_genes : int
        Number of top genes per program to include.
    nmf_max_iter : int
        Maximum number of optimization iterations for each NMF run.
    nmf_tol : float
        Convergence tolerance for each NMF run.
    """

    def __init__(
        self,
        n_programs: int = 20,
        n_iter: int = 5,
        top_n_genes: int = 50,
        nmf_max_iter: int = 1000,
        nmf_tol: float = 1e-4,
        random_state: int | None = 42,
    ) -> None:
        super().__init__()
        self.n_programs = n_programs
        self.n_iter = n_iter
        self.top_n_genes = top_n_genes
        self.nmf_max_iter = nmf_max_iter
        self.nmf_tol = nmf_tol
        self.random_state = random_state

        self._gene_names: list[str] = []
        self._W: np.ndarray | None = None
        self._H: np.ndarray | None = None

    def fit(
        self,
        embeddings: np.ndarray,
        gene_names: list[str],
        **kwargs: object,
    ) -> "CNMFProgramDiscovery":
        """Fit cNMF programs from expression matrix.

        Parameters
        ----------
        embeddings : np.ndarray
            Non-negative expression matrix of shape (n_cells, n_genes).
            If negative values exist, they will be shifted to non-negative.
        gene_names : list[str]
            Gene names corresponding to columns.

        Returns
        -------
        CNMFProgramDiscovery
            Fitted instance.
        """
        self._gene_names = list(gene_names)
        n_genes = len(gene_names)

        # Ensure non-negative
        X = embeddings.copy()
        if X.min() < 0:
            X = X - X.min()

        # If input is square (embedding/correlation), create synthetic expression
        if X.shape[0] == X.shape[1] == n_genes:
            # Treat as similarity matrix, create synthetic cells
            rng = np.random.default_rng(self.random_state)
            n_synthetic = max(200, n_genes)
            X = np.abs(rng.standard_normal((n_synthetic, n_genes))) * np.abs(X.mean(axis=0))
            X = X - X.min() + 1e-6

        # Consensus NMF: run multiple times and average
        all_H = []
        for run_idx in range(self.n_iter):
            seed = (self.random_state or 0) + run_idx
            max_iter = int(self.nmf_max_iter)
            W = None
            H = None
            for retry_idx in range(3):
                model = NMF(
                    n_components=min(self.n_programs, n_genes - 1),
                    init="nndsvda",
                    random_state=seed,
                    max_iter=max_iter,
                    tol=self.nmf_tol,
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    W = model.fit_transform(X)
                    H = model.components_  # (n_programs, n_genes)

                converged = not any(
                    issubclass(w.category, ConvergenceWarning) for w in caught
                )
                if converged:
                    break
                if retry_idx < 2:
                    next_max_iter = max_iter * 2
                    logger.info(
                        "cNMF run %d/%d reached max_iter=%d; retrying with max_iter=%d.",
                        run_idx + 1,
                        self.n_iter,
                        max_iter,
                        next_max_iter,
                    )
                    max_iter = next_max_iter
                else:
                    logger.warning(
                        "cNMF run %d/%d did not fully converge by max_iter=%d; proceeding with current factors.",
                        run_idx + 1,
                        self.n_iter,
                        max_iter,
                    )

            assert H is not None
            all_H.append(H)

        # Average gene loadings across runs
        H_consensus = np.mean(all_H, axis=0)
        self._H = H_consensus
        self._W = W  # From last run

        # Build programs from top genes per component
        programs: dict[str, list[str]] = {}
        scores: dict[str, list[tuple[str, float]]] = {}
        for k in range(H_consensus.shape[0]):
            loadings = H_consensus[k]
            top_indices = np.argsort(loadings)[::-1][:self.top_n_genes]
            # Filter to genes with meaningful loading
            threshold = loadings.mean() + loadings.std()
            sig_indices = [i for i in top_indices if loadings[i] > threshold]
            if len(sig_indices) < 3:
                sig_indices = top_indices[:max(3, self.top_n_genes // 5)].tolist()

            prog_name = f"cnmf_program_{k}"
            gene_list = [gene_names[i] for i in sig_indices]
            max_loading = loadings[sig_indices].max()
            if max_loading > 0:
                norm_loadings = loadings[sig_indices] / max_loading
            else:
                norm_loadings = np.ones(len(sig_indices))

            programs[prog_name] = gene_list
            scores[prog_name] = [
                (gene_list[j], float(norm_loadings[j]))
                for j in range(len(gene_list))
            ]

        self.programs_ = programs
        self.program_scores_ = scores
        logger.info(
            f"cNMF-like discovery: {len(programs)} programs, "
            f"top {self.top_n_genes} genes per program"
        )
        return self

    def get_programs(self) -> dict[str, list[str]]:
        """Return discovered gene programs."""
        self._check_is_fitted()
        assert self.programs_ is not None
        return dict(self.programs_)

    def get_program_scores(self) -> dict[str, list[tuple[str, float]]]:
        """Return weighted gene-program membership scores."""
        self._check_is_fitted()
        assert self.program_scores_ is not None
        return dict(self.program_scores_)

    def get_gene_loadings(self) -> np.ndarray:
        """Return the full gene loading matrix (n_programs x n_genes).

        Returns
        -------
        np.ndarray
            Gene loading matrix from consensus NMF.
        """
        self._check_is_fitted()
        if self._H is None:
            raise RuntimeError("No loadings available.")
        result: np.ndarray = np.array(self._H)
        return result


class OfficialCNMFProgramDiscovery(BaseProgramDiscovery):
    """Official cNMF baseline using the `cnmf` package pipeline.

    This wraps the maintained cNMF implementation and converts its
    consensus output into the common `{program: [genes]}` interface.

    Notes
    -----
    This method writes temporary files because the upstream cNMF API is
    file-based (`prepare` -> `factorize` -> `combine` -> `consensus`).
    """

    def __init__(
        self,
        n_programs: int = 20,
        n_iter: int = 6,
        top_n_genes: int = 50,
        density_threshold: float = 0.5,
        max_nmf_iter: int = 600,
        num_highvar_genes: int = 2000,
        random_state: int | None = 42,
        work_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.n_programs = n_programs
        self.n_iter = n_iter
        self.top_n_genes = top_n_genes
        self.density_threshold = density_threshold
        self.max_nmf_iter = max_nmf_iter
        self.num_highvar_genes = num_highvar_genes
        self.random_state = random_state
        self.work_dir = Path(work_dir) if work_dir is not None else None

        self._gene_names: list[str] = []
        self._spectra: pd.DataFrame | None = None

    def fit(
        self,
        embeddings: np.ndarray,
        gene_names: list[str],
        **kwargs: object,
    ) -> "OfficialCNMFProgramDiscovery":
        """Run official cNMF and extract top genes per consensus program."""
        try:
            import anndata as ad
            from cnmf import cNMF
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise ImportError(
                "Official cNMF baseline requires optional package `cnmf`."
            ) from exc

        work_gene_names = list(gene_names)
        n_genes = len(work_gene_names)
        if n_genes < 4:
            raise ValueError("Official cNMF requires at least 4 genes.")

        X = np.asarray(embeddings, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected 2-D expression matrix, got {X.shape}.")

        if X.shape[1] == n_genes:
            expr = X
        elif X.shape[0] == n_genes:
            expr = X.T
        else:
            raise ValueError(
                f"Expression shape {X.shape} is incompatible with {n_genes} genes."
            )
        finite_mask = np.isfinite(expr)
        if not np.any(finite_mask):
            raise ValueError("Official cNMF input contains no finite values.")
        if not np.all(finite_mask):
            n_bad = int(expr.size - int(finite_mask.sum()))
            logger.warning(
                "Official cNMF input had %d non-finite entries; replacing with zeros.",
                n_bad,
            )
            expr = np.nan_to_num(expr, nan=0.0, posinf=0.0, neginf=0.0)

        # Keep relative dynamic range for datasets with negative values
        # (e.g., centered/log-transformed matrices), then enforce non-negativity.
        col_min = np.min(expr, axis=0, keepdims=True)
        if np.any(col_min < 0.0):
            expr = expr - np.minimum(col_min, 0.0)
        expr = np.clip(expr, 0.0, None)
        if expr.size > 0:
            max_expr = float(np.max(expr))
            if np.isfinite(max_expr) and max_expr > 1e6:
                cap = float(np.quantile(expr, 0.999))
                if not np.isfinite(cap) or cap <= 0.0:
                    cap = 1e6
                if cap < max_expr:
                    logger.warning(
                        "Official cNMF input max %.3e is extreme; winsorizing at %.3e.",
                        max_expr,
                        cap,
                    )
                    expr = np.minimum(expr, cap)

        gene_var = np.var(expr, axis=0, dtype=np.float64)
        keep = np.isfinite(gene_var) & (gene_var > 1e-12)
        if not np.any(keep):
            raise ValueError("Official cNMF input has no non-zero variance genes.")
        if int(np.sum(keep)) < len(work_gene_names):
            dropped = int(len(work_gene_names) - int(np.sum(keep)))
            logger.warning(
                "Official cNMF dropping %d zero-variance genes before factorization.",
                dropped,
            )
            expr = expr[:, keep]
            work_gene_names = [g for g, flag in zip(work_gene_names, keep) if flag]

        n_genes = len(work_gene_names)
        if n_genes < 4:
            raise ValueError("Official cNMF requires at least 4 non-zero variance genes.")

        expr = expr + 1e-8

        k = max(2, min(self.n_programs, n_genes - 1))
        hvg = max(10, min(self.num_highvar_genes, n_genes))
        self._gene_names = list(work_gene_names)

        temp_dir = tempfile.mkdtemp(
            prefix="npathway_official_cnmf_",
            dir=str(self.work_dir) if self.work_dir is not None else None,
        )
        try:
            cell_names = [f"cell_{i}" for i in range(expr.shape[0])]
            adata = ad.AnnData(
                X=expr.astype(np.float32),
                obs=pd.DataFrame(index=cell_names),
                var=pd.DataFrame(index=work_gene_names),
            )
            counts_path = Path(temp_dir) / "counts.h5ad"
            adata.write_h5ad(counts_path)

            model = cNMF(output_dir=temp_dir, name="npathway")
            model.prepare(
                counts_fn=str(counts_path),
                components=[k],
                n_iter=self.n_iter,
                seed=self.random_state,
                num_highvar_genes=hvg,
                max_NMF_iter=self.max_nmf_iter,
            )
            model.factorize(worker_i=0, total_workers=1)
            model.combine(components=[k], skip_missing_files=False)
            density_used = self.density_threshold
            try:
                model.consensus(
                    k=k,
                    density_threshold=density_used,
                    show_clustering=False,
                    build_ref=False,
                    refit_usage=True,
                )
            except RuntimeError as exc:
                if "Zero components remain after density filtering" not in str(exc):
                    raise
                density_used = max(2.0, float(self.density_threshold) * 2.0)
                logger.warning(
                    "Official cNMF zero-component density filter at %.3f; retrying at %.3f.",
                    self.density_threshold,
                    density_used,
                )
                try:
                    model.consensus(
                        k=k,
                        density_threshold=density_used,
                        show_clustering=False,
                        build_ref=False,
                        refit_usage=True,
                    )
                except RuntimeError as second_exc:
                    if "Zero components remain after density filtering" not in str(second_exc):
                        raise
                    logger.warning(
                        "Official cNMF density filtering failed twice; "
                        "falling back to in-repo cNMF implementation."
                    )
                    fallback = CNMFProgramDiscovery(
                        n_programs=k,
                        n_iter=max(1, min(self.n_iter, 2)),
                        top_n_genes=self.top_n_genes,
                        nmf_max_iter=min(self.max_nmf_iter, 200),
                        nmf_tol=5e-4,
                        random_state=self.random_state,
                    )
                    fallback.fit(expr, work_gene_names)
                    self.programs_ = fallback.get_programs()
                    self.program_scores_ = fallback.get_program_scores()
                    self._spectra = None
                    return self
            _usage, spectra, _spectra_tpm, top_genes = model.load_results(
                K=k,
                density_threshold=density_used,
                n_top_genes=self.top_n_genes,
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self._spectra = spectra.copy()
        genes_set = set(work_gene_names)

        programs: dict[str, list[str]] = {}
        scores: dict[str, list[tuple[str, float]]] = {}
        for comp in top_genes.columns:
            raw_genes = [
                g for g in top_genes[comp].astype(str).tolist()
                if g and g != "nan" and g in genes_set
            ]
            seen: set[str] = set()
            picked: list[str] = []
            for g in raw_genes:
                if g not in seen:
                    seen.add(g)
                    picked.append(g)

            if len(picked) < 3:
                ranked = spectra[comp].sort_values(ascending=False)
                picked = [g for g in ranked.index.tolist() if g in genes_set][:self.top_n_genes]
            if len(picked) < 3:
                continue

            s = spectra[comp].reindex(picked).fillna(0.0).to_numpy(dtype=np.float64)
            smax = float(np.max(s)) if len(s) > 0 else 0.0
            if smax > 0:
                s = s / smax
            else:
                s = np.ones_like(s)

            pname = f"official_cnmf_program_{comp}"
            programs[pname] = picked
            scores[pname] = [(g, float(w)) for g, w in zip(picked, s)]

        if not programs:
            raise RuntimeError("Official cNMF produced no valid programs.")

        self.programs_ = programs
        self.program_scores_ = scores
        logger.info(
            "Official cNMF discovery: %d programs, top %d genes per program",
            len(programs),
            self.top_n_genes,
        )
        return self

    def get_programs(self) -> dict[str, list[str]]:
        """Return discovered programs from official cNMF."""
        self._check_is_fitted()
        assert self.programs_ is not None
        return dict(self.programs_)

    def get_program_scores(self) -> dict[str, list[tuple[str, float]]]:
        """Return weighted genes per official cNMF program."""
        self._check_is_fitted()
        assert self.program_scores_ is not None
        return dict(self.program_scores_)

    def get_gene_loadings(self) -> pd.DataFrame:
        """Return consensus spectra matrix (genes x programs)."""
        self._check_is_fitted()
        if self._spectra is None:
            raise RuntimeError("No spectra available.")
        return self._spectra.copy()


class StarCATReferenceProgramDiscovery(BaseProgramDiscovery):
    """Reference-guided programs from starCAT reference spectra.

    This is not de novo discovery: programs are taken from a starCAT
    reference atlas and mapped onto the dataset gene vocabulary.
    """

    def __init__(
        self,
        reference: str = "TCAT.V1",
        top_n_genes: int = 50,
        min_weight: float = 0.0,
        case_insensitive: bool = True,
        cache_dir: str | Path = "data/starcat_cache",
        reference_table: pd.DataFrame | None = None,
    ) -> None:
        super().__init__()
        self.reference = reference
        self.top_n_genes = top_n_genes
        self.min_weight = min_weight
        self.case_insensitive = case_insensitive
        self.cache_dir = Path(cache_dir)
        self.reference_table = reference_table

        self._reference_loaded: pd.DataFrame | None = None

    def fit(
        self,
        embeddings: np.ndarray,
        gene_names: list[str],
        **kwargs: object,
    ) -> "StarCATReferenceProgramDiscovery":
        """Load starCAT reference spectra and extract top mapped genes."""
        if len(gene_names) == 0:
            raise ValueError("gene_names must be non-empty.")

        if self.reference_table is not None:
            ref = self.reference_table.copy()
        else:
            try:
                from starcat import starCAT
            except Exception as exc:  # pragma: no cover - optional dependency path
                raise ImportError(
                    "starCAT baseline requires optional package `starcatpy`."
                ) from exc
            model = starCAT(reference=self.reference, cachedir=str(self.cache_dir))
            ref = model.ref.copy()

        if ref.empty:
            raise RuntimeError("starCAT reference is empty.")

        if not isinstance(ref.index, pd.Index):
            ref.index = pd.Index(ref.index)
        if not isinstance(ref.columns, pd.Index):
            ref.columns = pd.Index(ref.columns)

        gene_lookup: dict[str, str] = {}
        for g in gene_names:
            key = g.upper() if self.case_insensitive else g
            gene_lookup.setdefault(key, g)

        programs: dict[str, list[str]] = {}
        program_scores: dict[str, list[tuple[str, float]]] = {}
        matched_total: set[str] = set()

        for prog_name in ref.index:
            row = ref.loc[prog_name].astype(np.float64).sort_values(ascending=False)
            picked_genes: list[str] = []
            picked_scores: list[float] = []
            used: set[str] = set()

            for rgene, rweight in row.items():
                if not np.isfinite(rweight) or rweight <= self.min_weight:
                    continue
                key = str(rgene).upper() if self.case_insensitive else str(rgene)
                if key not in gene_lookup:
                    continue
                mapped = gene_lookup[key]
                if mapped in used:
                    continue
                picked_genes.append(mapped)
                picked_scores.append(float(rweight))
                used.add(mapped)
                matched_total.add(mapped)
                if len(picked_genes) >= self.top_n_genes:
                    break

            if len(picked_genes) < 3:
                continue

            arr = np.asarray(picked_scores, dtype=np.float64)
            amax = float(arr.max()) if len(arr) > 0 else 0.0
            if amax > 0:
                arr = arr / amax
            else:
                arr = np.ones_like(arr)

            pname = f"starcat_{prog_name}"
            programs[pname] = picked_genes
            program_scores[pname] = [
                (g, float(s)) for g, s in zip(picked_genes, arr)
            ]

        if not programs:
            raise RuntimeError(
                "starCAT mapping produced no programs with >=3 genes. "
                "Check reference/dataset gene symbol overlap."
            )

        self._reference_loaded = ref
        self.programs_ = programs
        self.program_scores_ = program_scores
        logger.info(
            "starCAT reference discovery: %d programs mapped, %d matched genes",
            len(programs),
            len(matched_total),
        )
        return self

    def get_programs(self) -> dict[str, list[str]]:
        """Return mapped starCAT reference programs."""
        self._check_is_fitted()
        assert self.programs_ is not None
        return dict(self.programs_)

    def get_program_scores(self) -> dict[str, list[tuple[str, float]]]:
        """Return mapped starCAT reference program scores."""
        self._check_is_fitted()
        assert self.program_scores_ is not None
        return dict(self.program_scores_)


class RandomProgramDiscovery(BaseProgramDiscovery):
    """Random gene program assignment as null baseline.

    Assigns genes to programs uniformly at random, preserving the same number
    and sizes as a reference set of programs.

    Parameters
    ----------
    n_programs : int
        Number of random programs to create.
    genes_per_program : int
        Number of genes per program.
    """

    def __init__(
        self,
        n_programs: int = 20,
        genes_per_program: int = 25,
        random_state: int | None = 42,
    ) -> None:
        super().__init__()
        self.n_programs = n_programs
        self.genes_per_program = genes_per_program
        self.random_state = random_state

        self._gene_names: list[str] = []

    def fit(
        self,
        embeddings: np.ndarray,
        gene_names: list[str],
        **kwargs: object,
    ) -> "RandomProgramDiscovery":
        """Create random gene programs.

        Parameters
        ----------
        embeddings : np.ndarray
            Ignored. Present for API compatibility.
        gene_names : list[str]
            Gene names to randomly assign.

        Returns
        -------
        RandomProgramDiscovery
            Fitted instance.
        """
        self._gene_names = list(gene_names)
        rng = np.random.default_rng(self.random_state)

        programs: dict[str, list[str]] = {}
        scores: dict[str, list[tuple[str, float]]] = {}
        all_genes = list(gene_names)
        n_genes = len(all_genes)

        for k in range(self.n_programs):
            size = min(self.genes_per_program, n_genes)
            selected = rng.choice(all_genes, size=size, replace=False).tolist()
            prog_name = f"random_program_{k}"
            programs[prog_name] = selected
            scores[prog_name] = [(g, 1.0) for g in selected]

        self.programs_ = programs
        self.program_scores_ = scores
        logger.info(
            f"Random baseline: {len(programs)} programs, "
            f"{self.genes_per_program} genes each"
        )
        return self

    def get_programs(self) -> dict[str, list[str]]:
        """Return random gene programs."""
        self._check_is_fitted()
        assert self.programs_ is not None
        return dict(self.programs_)

    def get_program_scores(self) -> dict[str, list[tuple[str, float]]]:
        """Return uniform scores (all 1.0)."""
        self._check_is_fitted()
        assert self.program_scores_ is not None
        return dict(self.program_scores_)


class ExpressionClusteringBaseline(BaseProgramDiscovery):
    """Simple k-means on raw expression profiles as a non-trivial baseline.

    Clusters genes by their expression patterns across cells, without any
    foundation model embeddings.

    Parameters
    ----------
    n_programs : int
        Number of gene clusters.
    """

    def __init__(
        self,
        n_programs: int = 20,
        random_state: int | None = 42,
    ) -> None:
        super().__init__()
        self.n_programs = n_programs
        self.random_state = random_state

        self._gene_names: list[str] = []

    def fit(
        self,
        embeddings: np.ndarray,
        gene_names: list[str],
        **kwargs: object,
    ) -> "ExpressionClusteringBaseline":
        """Cluster genes by expression profiles.

        Parameters
        ----------
        embeddings : np.ndarray
            Expression matrix of shape (n_cells, n_genes).
        gene_names : list[str]
            Gene names.

        Returns
        -------
        ExpressionClusteringBaseline
            Fitted instance.
        """
        self._gene_names = list(gene_names)

        # Transpose: gene x cell for clustering genes
        if embeddings.shape[1] == len(gene_names):
            gene_profiles = embeddings.T  # (n_genes, n_cells)
        else:
            gene_profiles = embeddings  # Already (n_genes, features)

        # Normalize gene profiles
        norms = np.linalg.norm(gene_profiles, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        gene_profiles_norm = gene_profiles / norms

        n_genes = len(gene_names)
        if n_genes == 0:
            raise ValueError("gene_names must contain at least one gene.")
        if n_genes == 1:
            self.programs_ = {"expr_cluster_0": [gene_names[0]]}
            self.program_scores_ = {"expr_cluster_0": [(gene_names[0], 1.0)]}
            logger.info("Expression clustering baseline: 1 program")
            return self

        # K-means
        kmeans = KMeans(
            n_clusters=max(1, min(self.n_programs, n_genes)),
            random_state=self.random_state,
            n_init=10,
        )
        labels = kmeans.fit_predict(gene_profiles_norm)

        # Build programs
        programs: dict[str, list[str]] = {}
        prog_scores: dict[str, list[tuple[str, float]]] = {}
        for label in sorted(set(labels)):
            indices = [i for i, lab in enumerate(labels) if lab == label]
            prog_name = f"expr_cluster_{label}"
            gene_list = [gene_names[i] for i in indices]

            # Score based on distance to centroid
            centroid = kmeans.cluster_centers_[label]
            distances = np.linalg.norm(
                gene_profiles_norm[indices] - centroid, axis=1
            )
            max_dist = distances.max() if distances.max() > 0 else 1.0
            scores = 1.0 - (distances / max_dist)

            programs[prog_name] = gene_list
            prog_scores[prog_name] = [
                (gene_list[j], float(scores[j])) for j in range(len(gene_list))
            ]

        self.programs_ = programs
        self.program_scores_ = prog_scores
        logger.info(
            f"Expression clustering baseline: {len(programs)} programs"
        )
        return self

    def get_programs(self) -> dict[str, list[str]]:
        """Return discovered gene programs."""
        self._check_is_fitted()
        assert self.programs_ is not None
        return dict(self.programs_)

    def get_program_scores(self) -> dict[str, list[tuple[str, float]]]:
        """Return weighted gene-program membership scores."""
        self._check_is_fitted()
        assert self.program_scores_ is not None
        return dict(self.program_scores_)
