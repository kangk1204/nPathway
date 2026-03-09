"""Competitor baseline wrappers for external gene program discovery tools.

Wraps third-party gene program discovery packages (scSpectra, scETM) so they
can be benchmarked against nPathway methods using the common
:class:`~npathway.discovery.base.BaseProgramDiscovery` interface.

Each wrapper:
- Gracefully falls back if the external package is not installed.
- Exposes an ``is_available()`` classmethod for runtime availability checks.
- Populates both hard programs (``programs_``) and native soft membership
  (``soft_programs_``) when the underlying model provides continuous loadings.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from npathway.discovery.base import BaseProgramDiscovery

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports with graceful fallback
# ---------------------------------------------------------------------------

_spectra_available: bool
try:
    import Spectra as _spectra_mod  # type: ignore[import-untyped]

    _spectra_available = True
except ImportError:
    _spectra_mod = None  # type: ignore[assignment]
    _spectra_available = False

_scetm_available: bool
try:
    import scETM as _scetm_mod  # type: ignore[import-untyped]

    _scetm_available = True
except ImportError:
    _scetm_mod = None  # type: ignore[assignment]
    _scetm_available = False


# ===================================================================
# Spectra wrapper
# ===================================================================


class SpectraProgramDiscovery(BaseProgramDiscovery):
    """Gene program discovery using the scSpectra package.

    scSpectra discovers gene programs (factors) from single-cell RNA-seq
    data by combining gene-level embeddings with expression data and,
    optionally, cell-type annotations.  The learned factors provide
    native soft (weighted) gene-program membership.

    Parameters
    ----------
    n_programs : int
        Number of gene programs (factors) to learn.
    lam : float
        Sparsity regularization strength for gene-factor loadings.
        Higher values yield sparser programs.
    top_n_genes : int
        Number of top-loaded genes to include in each hard program.
    n_epochs : int
        Number of training epochs.
    lr : float
        Learning rate for the optimizer.
    use_cell_types : bool
        If ``True`` and ``cell_type_labels`` is supplied via ``fit(...)``,
        cell-type information is incorporated into factor learning.
    random_state : int | None
        Random seed for reproducibility.

    Raises
    ------
    ImportError
        At ``fit`` time if the ``Spectra`` package is not installed.

    Examples
    --------
    >>> if SpectraProgramDiscovery.is_available():
    ...     model = SpectraProgramDiscovery(n_programs=10)
    ...     model.fit(gene_embeddings, gene_names,
    ...               expression=expr_matrix,
    ...               cell_type_labels=labels)
    ...     programs = model.get_programs()
    """

    def __init__(
        self,
        n_programs: int = 20,
        lam: float = 0.01,
        top_n_genes: int = 50,
        n_epochs: int = 10000,
        lr: float = 1e-4,
        use_cell_types: bool = True,
        random_state: int | None = 42,
    ) -> None:
        super().__init__()
        self.n_programs = n_programs
        self.lam = lam
        self.top_n_genes = top_n_genes
        self.n_epochs = n_epochs
        self.lr = lr
        self.use_cell_types = use_cell_types
        self.random_state = random_state

        self._gene_names: list[str] = []
        self._factors: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Availability check
    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """Return ``True`` if the ``Spectra`` package is importable.

        Returns
        -------
        bool
            Whether scSpectra is installed and importable.
        """
        return _spectra_available

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        embeddings: np.ndarray,
        gene_names: list[str],
        **kwargs: Any,
    ) -> "SpectraProgramDiscovery":
        """Discover gene programs via scSpectra.

        Parameters
        ----------
        embeddings : np.ndarray
            Gene embedding matrix of shape ``(n_genes, n_dims)``.
        gene_names : list[str]
            Gene identifiers aligned with rows of *embeddings*.
        **kwargs : Any
            Additional keyword arguments:

            expression : np.ndarray
                Expression matrix of shape ``(n_cells, n_genes)``.
                **Required**.
            cell_type_labels : np.ndarray | list[str] | None
                Per-cell cell-type annotations of length ``n_cells``.
                Used only when ``use_cell_types=True``.

        Returns
        -------
        SpectraProgramDiscovery
            Fitted instance (``self``).

        Raises
        ------
        ImportError
            If the ``Spectra`` package is not installed.
        ValueError
            If *expression* is not provided or shapes are incompatible.
        """
        if not _spectra_available:
            raise ImportError(
                "scSpectra is not installed. Install it with: "
                "pip install scSpectra  (or see https://github.com/dpeerlab/spectra)"
            )

        import anndata as ad  # type: ignore[import-untyped]

        # ---- Validate inputs -----------------------------------------
        expression: np.ndarray | None = kwargs.get("expression")  # type: ignore[assignment]
        if expression is None:
            raise ValueError(
                "SpectraProgramDiscovery requires an expression matrix. "
                "Pass it as: fit(embeddings, gene_names, expression=X)"
            )

        self._gene_names = list(gene_names)
        n_genes = len(gene_names)

        expression = np.asarray(expression, dtype=np.float32)
        if expression.shape[1] != n_genes:
            raise ValueError(
                f"Expression matrix has {expression.shape[1]} columns but "
                f"{n_genes} gene names were provided."
            )

        cell_type_labels: np.ndarray | list[str] | None = kwargs.get(  # type: ignore[assignment]
            "cell_type_labels"
        )

        # ---- Build AnnData -------------------------------------------
        adata = ad.AnnData(X=expression)
        adata.var_names = gene_names
        adata.obs_names = [f"cell_{i}" for i in range(expression.shape[0])]

        # Determine whether to use cell types in Spectra
        use_ct = self.use_cell_types and cell_type_labels is not None
        if use_ct:
            adata.obs["cell_type"] = list(cell_type_labels)
            cell_type_key: str = "cell_type"
            # Nested dict: {cell_type: {set_name: [genes]}, "global": {}}
            unique_cts = sorted(set(cell_type_labels))  # type: ignore[arg-type]
            gs_dict: dict[str, Any] = {ct: {} for ct in unique_cts}
            gs_dict["global"] = {}
        else:
            cell_type_key = "cell_type"  # not used when use_cell_types=False
            # Flat dict: {set_name: [genes]}
            gs_dict = {}

        # ---- Run Spectra ---------------------------------------------
        spectra_model = _spectra_mod.est_spectra(
            adata=adata,
            gene_set_dictionary=gs_dict,
            L=self.n_programs,
            use_highly_variable=False,
            cell_type_key=cell_type_key,
            use_cell_types=use_ct,
            lam=self.lam,
            n_top_vals=self.top_n_genes,
            clean_gs=False,
            filter_sets=False,
            label_factors=False,
            num_epochs=self.n_epochs,
            lr_schedule=[self.lr, self.lr * 0.1],
        )

        # ---- Extract factors -----------------------------------------
        # est_spectra returns a SPECTRA_Model and also stores results in adata.uns.
        # The factor matrix shape is (n_factors, n_vocab_genes). n_vocab_genes
        # equals n_genes when use_highly_variable=False.
        if hasattr(spectra_model, "factors") and spectra_model.factors is not None:
            factors = np.asarray(spectra_model.factors)  # (n_factors, n_vocab)
        elif "SPECTRA_factors" in adata.uns:
            factors = np.asarray(adata.uns["SPECTRA_factors"])
        else:
            raise RuntimeError(
                "Could not locate Spectra factors in the output. "
                "Check scSpectra version compatibility."
            )

        # Build gene-name-to-index mapping for the vocabulary Spectra used.
        # When use_highly_variable=False, vocab == gene_names, but we
        # handle the general case where Spectra may subset.
        if "spectra_vocab" in adata.var.columns:
            vocab_mask = adata.var["spectra_vocab"].values.astype(bool)
            vocab_genes: list[str] = list(adata.var_names[vocab_mask])
        else:
            vocab_genes = list(adata.var_names)
        vocab_gene_set = set(vocab_genes)
        vocab_idx = {g: i for i, g in enumerate(vocab_genes)}

        self._factors = factors  # (n_factors, n_vocab)

        # ---- Extract markers -----------------------------------------
        # SPECTRA_markers is a 2-D numpy array of shape (n_factors, n_top_vals)
        # containing gene name strings.
        markers_arr: np.ndarray | None = adata.uns.get("SPECTRA_markers")

        # ---- Build programs and soft memberships ---------------------
        programs: dict[str, list[str]] = {}
        scores: dict[str, list[tuple[str, float]]] = {}
        soft: dict[str, dict[str, float]] = {}

        n_factors_total = factors.shape[0]
        for k in range(n_factors_total):
            prog_name = f"spectra_factor_{k}"
            loadings = factors[k]  # (n_vocab,)

            # Hard program: use markers array if available, else top-loaded genes
            if markers_arr is not None and k < markers_arr.shape[0]:
                marker_genes = markers_arr[k]
                hard_genes = [
                    str(g) for g in marker_genes
                    if str(g) and str(g) != "" and str(g) in vocab_gene_set
                ]
            else:
                top_idx = np.argsort(np.abs(loadings))[::-1][: self.top_n_genes]
                hard_genes = [vocab_genes[i] for i in top_idx]

            if len(hard_genes) < 1:
                continue

            # Scores for hard program members
            hard_loadings = np.array(
                [np.abs(loadings[vocab_idx[g]]) for g in hard_genes]
            )
            max_load = float(hard_loadings.max()) if hard_loadings.max() > 0 else 1.0
            norm_loadings = hard_loadings / max_load

            programs[prog_name] = hard_genes
            scores[prog_name] = [
                (g, float(s)) for g, s in zip(hard_genes, norm_loadings)
            ]

            # Soft membership: all genes with non-negligible loading
            abs_loadings = np.abs(loadings)
            factor_max = float(abs_loadings.max()) if abs_loadings.max() > 0 else 1.0
            n_vocab = len(vocab_genes)
            soft[prog_name] = {
                vocab_genes[i]: float(abs_loadings[i] / factor_max)
                for i in range(n_vocab)
                if abs_loadings[i] > 0
            }

        if not programs:
            raise RuntimeError(
                "scSpectra produced no valid programs. "
                "Try adjusting n_programs or lam."
            )

        self.programs_ = programs
        self.program_scores_ = scores
        self.soft_programs_ = soft

        logger.info(
            "scSpectra discovery: %d factors, %d programs retained",
            n_factors_total,
            len(programs),
        )
        return self

    # ------------------------------------------------------------------
    # Required interface methods
    # ------------------------------------------------------------------

    def get_programs(self) -> dict[str, list[str]]:
        """Return discovered gene programs from Spectra factors.

        Returns
        -------
        dict[str, list[str]]
            Mapping from program name to list of gene names.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        self._check_is_fitted()
        assert self.programs_ is not None
        return dict(self.programs_)

    def get_program_scores(self) -> dict[str, list[tuple[str, float]]]:
        """Return gene-program membership scores from Spectra factor loadings.

        Returns
        -------
        dict[str, list[tuple[str, float]]]
            Mapping from program name to list of ``(gene, score)`` tuples.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        self._check_is_fitted()
        assert self.program_scores_ is not None
        return dict(self.program_scores_)

    def get_factors(self) -> np.ndarray:
        """Return the raw factor loading matrix from Spectra.

        Returns
        -------
        np.ndarray
            Factor matrix of shape ``(n_programs, n_genes)``.

        Raises
        ------
        RuntimeError
            If the model has not been fitted or factors are unavailable.
        """
        self._check_is_fitted()
        if self._factors is None:
            raise RuntimeError("No factors available.")
        return np.array(self._factors)


# ===================================================================
# scETM wrapper
# ===================================================================


class ScETMProgramDiscovery(BaseProgramDiscovery):
    """Gene program discovery using the scETM (single-cell Embedded Topic Model) package.

    scETM discovers gene programs as latent topics from single-cell
    expression data using a variational autoencoder with an embedded
    topic model architecture.  The topic-gene loading matrix provides
    native soft (weighted) gene-program membership.

    Parameters
    ----------
    n_programs : int
        Number of topics (programs) to learn.
    top_n_genes : int
        Number of top-loaded genes per topic for the hard program.
    n_epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size for training.
    lr : float
        Learning rate.
    emb_dim : int
        Dimensionality of the gene and topic embedding space.
    enc_hidden_dim : int
        Hidden dimension of the encoder network.
    random_state : int | None
        Random seed for reproducibility.

    Raises
    ------
    ImportError
        At ``fit`` time if the ``scETM`` package is not installed.

    Examples
    --------
    >>> if ScETMProgramDiscovery.is_available():
    ...     model = ScETMProgramDiscovery(n_programs=15)
    ...     model.fit(gene_embeddings, gene_names, expression=expr_matrix)
    ...     soft = model.get_soft_programs()
    """

    def __init__(
        self,
        n_programs: int = 20,
        top_n_genes: int = 50,
        n_epochs: int = 200,
        batch_size: int = 128,
        lr: float = 5e-3,
        emb_dim: int = 400,
        enc_hidden_dim: int = 128,
        random_state: int | None = 42,
    ) -> None:
        super().__init__()
        self.n_programs = n_programs
        self.top_n_genes = top_n_genes
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.emb_dim = emb_dim
        self.enc_hidden_dim = enc_hidden_dim
        self.random_state = random_state

        self._gene_names: list[str] = []
        self._topic_gene_matrix: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Availability check
    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """Return ``True`` if the ``scETM`` package is importable.

        Returns
        -------
        bool
            Whether scETM is installed and importable.
        """
        return _scetm_available

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        embeddings: np.ndarray,
        gene_names: list[str],
        **kwargs: Any,
    ) -> "ScETMProgramDiscovery":
        """Discover gene programs via scETM topic modelling.

        Parameters
        ----------
        embeddings : np.ndarray
            Gene embedding matrix of shape ``(n_genes, n_dims)``.
            Used to initialise the gene embedding layer in scETM.
        gene_names : list[str]
            Gene identifiers aligned with rows of *embeddings*.
        **kwargs : Any
            Additional keyword arguments:

            expression : np.ndarray
                Expression count matrix of shape ``(n_cells, n_genes)``.
                **Required**.
            batch_labels : np.ndarray | list[str] | None
                Per-cell batch labels for batch correction.

        Returns
        -------
        ScETMProgramDiscovery
            Fitted instance (``self``).

        Raises
        ------
        ImportError
            If the ``scETM`` package is not installed.
        ValueError
            If *expression* is not provided or shapes are incompatible.
        """
        if not _scetm_available:
            raise ImportError(
                "scETM is not installed. Install it with: "
                "pip install scETM  (or see https://github.com/CompCy-lab/scETM)"
            )

        import anndata as ad  # type: ignore[import-untyped]
        import torch  # type: ignore[import-untyped]

        # ---- Validate inputs -----------------------------------------
        expression: np.ndarray | None = kwargs.get("expression")  # type: ignore[assignment]
        if expression is None:
            raise ValueError(
                "ScETMProgramDiscovery requires an expression matrix. "
                "Pass it as: fit(embeddings, gene_names, expression=X)"
            )

        self._gene_names = list(gene_names)
        n_genes = len(gene_names)

        expression = np.asarray(expression, dtype=np.float32)
        if expression.shape[1] != n_genes:
            raise ValueError(
                f"Expression matrix has {expression.shape[1]} columns but "
                f"{n_genes} gene names were provided."
            )

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.shape[0] != n_genes:
            raise ValueError(
                f"Embeddings have {embeddings.shape[0]} rows but "
                f"{n_genes} gene names were provided."
            )

        batch_labels: np.ndarray | list[str] | None = kwargs.get(  # type: ignore[assignment]
            "batch_labels"
        )

        # ---- Seed for reproducibility --------------------------------
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        # ---- Build AnnData -------------------------------------------
        adata = ad.AnnData(X=expression)
        adata.var_names = gene_names
        adata.obs_names = [f"cell_{i}" for i in range(expression.shape[0])]

        if batch_labels is not None:
            adata.obs["batch"] = list(batch_labels)
            batch_col: str | None = "batch"
        else:
            batch_col = None

        # ---- Initialise and train scETM ------------------------------
        emb_dim = min(self.emb_dim, embeddings.shape[1])

        model = _scetm_mod.scETM(
            adata,
            n_topics=self.n_programs,
            emb_dim=emb_dim,
            enc_hidden_dim=self.enc_hidden_dim,
            trainable_gene_emb_dim=emb_dim,
            batch_col=batch_col,
        )

        # Inject pre-trained gene embeddings as initialisation
        if hasattr(model, "rho") and embeddings.shape[1] >= emb_dim:
            init_emb = embeddings[:, :emb_dim]
            with torch.no_grad():
                if hasattr(model.rho, "weight"):
                    model.rho.weight.copy_(torch.from_numpy(init_emb))
                elif isinstance(model.rho, torch.Tensor):
                    model.rho.copy_(torch.from_numpy(init_emb))

        model.train(
            adata,
            max_epochs=self.n_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
        )

        # ---- Extract topic-gene distribution -------------------------
        # scETM stores the topic-gene weight matrix in model.get_beta()
        # or model.beta, shape (n_topics, n_genes)
        if hasattr(model, "get_beta"):
            beta = model.get_beta()
            if isinstance(beta, torch.Tensor):
                beta = beta.detach().cpu().numpy()
            else:
                beta = np.asarray(beta)
        elif hasattr(model, "beta"):
            beta = model.beta
            if isinstance(beta, torch.Tensor):
                beta = beta.detach().cpu().numpy()
            else:
                beta = np.asarray(beta)
        else:
            # Reconstruct from rho and alpha embeddings:
            # beta = softmax(alpha @ rho^T)
            alpha = model.alpha.detach().cpu().numpy()  # (n_topics, emb_dim)
            rho = model.rho.detach().cpu().numpy() if isinstance(
                model.rho, torch.Tensor
            ) else model.rho.weight.detach().cpu().numpy()  # (n_genes, emb_dim)
            logits = alpha @ rho.T  # (n_topics, n_genes)
            # Softmax along gene axis
            exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
            beta = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        self._topic_gene_matrix = beta.astype(np.float64)

        # ---- Build programs and soft memberships ---------------------
        programs: dict[str, list[str]] = {}
        program_scores: dict[str, list[tuple[str, float]]] = {}
        soft: dict[str, dict[str, float]] = {}

        n_topics = beta.shape[0]
        for k in range(n_topics):
            prog_name = f"scetm_topic_{k}"
            topic_weights = beta[k]  # (n_genes,)

            # Hard program: top N genes by loading
            top_idx = np.argsort(topic_weights)[::-1][: self.top_n_genes]
            hard_genes = [gene_names[i] for i in top_idx]

            # Normalized scores for hard program
            max_w = float(topic_weights[top_idx[0]]) if len(top_idx) > 0 else 1.0
            if max_w <= 0:
                max_w = 1.0
            norm_scores = topic_weights[top_idx] / max_w

            programs[prog_name] = hard_genes
            program_scores[prog_name] = [
                (g, float(s)) for g, s in zip(hard_genes, norm_scores)
            ]

            # Soft membership: full topic distribution over all genes
            topic_max = float(topic_weights.max()) if topic_weights.max() > 0 else 1.0
            soft[prog_name] = {
                gene_names[i]: float(topic_weights[i] / topic_max)
                for i in range(n_genes)
                if topic_weights[i] > 0
            }

        if not programs:
            raise RuntimeError(
                "scETM produced no valid topics. "
                "Try adjusting n_programs or n_epochs."
            )

        self.programs_ = programs
        self.program_scores_ = program_scores
        self.soft_programs_ = soft

        logger.info(
            "scETM discovery: %d topics, %d programs retained",
            n_topics,
            len(programs),
        )
        return self

    # ------------------------------------------------------------------
    # Required interface methods
    # ------------------------------------------------------------------

    def get_programs(self) -> dict[str, list[str]]:
        """Return discovered gene programs from scETM topics.

        Returns
        -------
        dict[str, list[str]]
            Mapping from program name to list of gene names.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        self._check_is_fitted()
        assert self.programs_ is not None
        return dict(self.programs_)

    def get_program_scores(self) -> dict[str, list[tuple[str, float]]]:
        """Return gene-program membership scores from scETM topic loadings.

        Returns
        -------
        dict[str, list[tuple[str, float]]]
            Mapping from program name to list of ``(gene, score)`` tuples.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        self._check_is_fitted()
        assert self.program_scores_ is not None
        return dict(self.program_scores_)

    def get_topic_gene_matrix(self) -> np.ndarray:
        """Return the full topic-gene distribution matrix.

        Returns
        -------
        np.ndarray
            Matrix of shape ``(n_topics, n_genes)`` with topic-gene weights.

        Raises
        ------
        RuntimeError
            If the model has not been fitted or the matrix is unavailable.
        """
        self._check_is_fitted()
        if self._topic_gene_matrix is None:
            raise RuntimeError("No topic-gene matrix available.")
        return np.array(self._topic_gene_matrix)
