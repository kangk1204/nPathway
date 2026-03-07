"""Neural topic model-based gene program discovery.

This module implements an Embedded Topic Model (ETM) approach for
discovering gene programs from single-cell RNA-seq data.  Cells are
treated as documents and genes as words.  Pre-trained gene embeddings
are used to initialise the topic--gene distribution, encouraging
semantically related genes to co-occur in the same topic.

Architecture
------------
* **Encoder** -- MLP that maps a bag-of-words (BOW) cell profile to a
  mean and log-variance in topic space (VAE-style).  Supports
  configurable depth, dropout, and optional batch normalisation.
* **Decoder** -- Computes topic--gene log-probabilities via the inner
  product between learned topic embeddings and (optionally frozen) gene
  embeddings, then reconstructs the BOW through topic proportions.
  A direct decoder weight matrix is also available for more principled
  topic-gene loading extraction.
* **Training** -- Minimises the ELBO (negative reconstruction +
  KL divergence) with the reparameterisation trick.  Supports early
  stopping, cosine-annealing LR schedule, validation-based perplexity
  monitoring, and full per-epoch diagnostic tracking.

References
----------
Dieng, A. B., Ruiz, F. J. R., & Blei, D. M. (2020).
    Topic Modeling in Embedding Spaces. *TACL*, 8, 439--453.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from npathway.discovery.base import BaseProgramDiscovery

logger = logging.getLogger(__name__)


# ======================================================================
# Training history data class
# ======================================================================

@dataclass
class TrainingHistory:
    """Container for per-epoch training diagnostics.

    Attributes
    ----------
    elbo : list[float]
        Total ELBO loss per epoch.
    reconstruction_loss : list[float]
        Reconstruction (negative log-likelihood) component per epoch.
    kl_divergence : list[float]
        KL divergence component per epoch.
    val_perplexity : list[float]
        Held-out perplexity per epoch (empty if no validation split).
    stopped_epoch : int | None
        Epoch at which early stopping triggered, or ``None``.
    """

    elbo: list[float] = field(default_factory=list)
    reconstruction_loss: list[float] = field(default_factory=list)
    kl_divergence: list[float] = field(default_factory=list)
    val_perplexity: list[float] = field(default_factory=list)
    stopped_epoch: int | None = None


# ======================================================================
# PyTorch ETM Architecture
# ======================================================================

class _Encoder(nn.Module):
    """BOW -> (mu, log_sigma) in topic space.

    Supports configurable depth, dropout, and optional batch normalisation.

    Parameters
    ----------
    vocab_size : int
        Input dimensionality (number of genes).
    hidden_dim : int
        Width of hidden layers.
    n_topics : int
        Output dimensionality (latent topic space).
    dropout : float
        Dropout probability applied to input and between hidden layers.
    n_hidden_layers : int
        Number of hidden layers (minimum 1).
    use_batch_norm : bool
        Whether to apply batch normalisation after each hidden layer.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        n_topics: int,
        dropout: float,
        n_hidden_layers: int = 2,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        n_hidden_layers = max(1, n_hidden_layers)

        self.input_drop = nn.Dropout(dropout)

        layers: list[nn.Module] = []
        in_dim = vocab_size
        for i in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Softplus())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.hidden = nn.Sequential(*layers)
        self.mu = nn.Linear(hidden_dim, n_topics)
        self.log_sigma = nn.Linear(hidden_dim, n_topics)

    def forward(self, bow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode BOW to topic-space mean and log-variance.

        Parameters
        ----------
        bow : torch.Tensor
            Bag-of-words input of shape ``(batch, vocab_size)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(mu, log_sigma)`` each of shape ``(batch, n_topics)``.
        """
        h = self.input_drop(bow)
        h = self.hidden(h)
        return self.mu(h), self.log_sigma(h)


class _ETMCore(nn.Module):
    """Embedded Topic Model core.

    Parameters
    ----------
    vocab_size : int
        Number of genes (vocabulary size).
    n_topics : int
        Number of latent topics (gene programs).
    embed_dim : int
        Dimensionality of gene embeddings.
    hidden_dim : int
        Hidden layer width in the encoder.
    dropout : float
        Dropout rate in the encoder.
    n_hidden_layers : int
        Number of hidden layers in the encoder.
    use_batch_norm : bool
        Whether to use batch normalisation in the encoder.
    gene_embeddings : torch.Tensor | None
        Pre-trained gene embeddings ``(vocab_size, embed_dim)``.
        If provided, they are used to initialise the gene embedding
        matrix and can optionally be frozen.
    freeze_gene_embeddings : bool
        If ``True``, gene embeddings are not updated during training.
    embedding_init : str
        ``"pretrained"`` uses *gene_embeddings* (or raises if None);
        ``"random"`` always uses random initialisation (for ablation).
    """

    def __init__(
        self,
        vocab_size: int,
        n_topics: int,
        embed_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        n_hidden_layers: int = 2,
        use_batch_norm: bool = True,
        gene_embeddings: torch.Tensor | None = None,
        freeze_gene_embeddings: bool = False,
        embedding_init: str = "pretrained",
    ) -> None:
        super().__init__()
        self.n_topics = n_topics
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Encoder
        self.encoder = _Encoder(
            vocab_size,
            hidden_dim,
            n_topics,
            dropout,
            n_hidden_layers=n_hidden_layers,
            use_batch_norm=use_batch_norm,
        )

        # Topic embeddings (learned)
        self.topic_embeddings = nn.Parameter(
            torch.randn(n_topics, embed_dim) * 0.02
        )

        # Direct decoder weight matrix for principled topic-gene loadings
        self.decoder_weights = nn.Linear(n_topics, vocab_size, bias=False)
        nn.init.xavier_uniform_(self.decoder_weights.weight)

        # Gene embeddings (optionally pre-initialised / frozen)
        if embedding_init == "pretrained" and gene_embeddings is not None:
            if gene_embeddings.shape != (vocab_size, embed_dim):
                raise ValueError(
                    f"gene_embeddings shape {gene_embeddings.shape} does not match "
                    f"expected ({vocab_size}, {embed_dim})."
                )
            self.gene_embeddings = nn.Parameter(
                gene_embeddings.clone(), requires_grad=not freeze_gene_embeddings
            )
        else:
            # Random init for ablation or when no pretrained embeddings given
            self.gene_embeddings = nn.Parameter(
                torch.randn(vocab_size, embed_dim) * 0.02
            )

    # ----- helpers -----

    def _reparameterise(
        self, mu: torch.Tensor, log_sigma: torch.Tensor
    ) -> torch.Tensor:
        """Sample z from N(mu, sigma^2) using the reparameterisation trick.

        Parameters
        ----------
        mu : torch.Tensor
            Mean, shape ``(batch, n_topics)``.
        log_sigma : torch.Tensor
            Log standard-deviation, shape ``(batch, n_topics)``.

        Returns
        -------
        torch.Tensor
            Sampled latent ``z`` of shape ``(batch, n_topics)``.
        """
        if self.training:
            std = torch.exp(log_sigma)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def _get_topic_gene_logits(self) -> torch.Tensor:
        """Compute topic-gene logits via embedding similarity.

        Returns
        -------
        torch.Tensor
            Shape ``(n_topics, vocab_size)``.
        """
        # L2-normalise both for cosine-like similarity
        t_norm = F.normalize(self.topic_embeddings, dim=-1)
        g_norm = F.normalize(self.gene_embeddings, dim=-1)
        return t_norm @ g_norm.T  # (K, V)

    def get_decoder_topic_gene_loadings(self) -> torch.Tensor:
        """Return topic-gene loadings from the direct decoder weight matrix.

        The decoder weight matrix ``W`` has shape ``(vocab_size, n_topics)``.
        The loadings ``W^T`` have shape ``(n_topics, vocab_size)`` and give
        a principled, non-embedding-based view of topic-gene associations.

        Returns
        -------
        torch.Tensor
            Shape ``(n_topics, vocab_size)``.
        """
        return self.decoder_weights.weight.T  # (K, V)

    # ----- forward -----

    def forward(
        self, bow: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Parameters
        ----------
        bow : torch.Tensor
            Normalised bag-of-words ``(batch, vocab_size)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``(recon, mu, log_sigma)`` where *recon* has shape
            ``(batch, vocab_size)`` (log-probabilities).
        """
        mu, log_sigma = self.encoder(bow)
        z = self._reparameterise(mu, log_sigma)
        theta = F.softmax(z, dim=-1)  # (batch, K)

        topic_gene = self._get_topic_gene_logits()  # (K, V)
        # Log-probability of each gene given the topic mixture
        log_prob = torch.log_softmax(topic_gene, dim=-1)  # (K, V)
        recon = theta @ log_prob  # (batch, V)

        return recon, mu, log_sigma


# ======================================================================
# Loss
# ======================================================================

def _elbo_loss(
    recon: torch.Tensor,
    bow: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the negative ELBO loss, returning components separately.

    Parameters
    ----------
    recon : torch.Tensor
        Reconstructed log-probabilities ``(batch, V)``.
    bow : torch.Tensor
        Original bag-of-words ``(batch, V)``.
    mu : torch.Tensor
        Encoder mean ``(batch, K)``.
    log_sigma : torch.Tensor
        Encoder log-std ``(batch, K)``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(total_loss, nll, kl)`` each a scalar tensor (mean over batch).
    """
    # Reconstruction: negative log-likelihood
    nll = -(bow * recon).sum(dim=-1).mean()

    # KL divergence: KL(q || N(0,I))
    kl = -0.5 * (1.0 + 2.0 * log_sigma - mu.pow(2) - (2.0 * log_sigma).exp())
    kl = kl.sum(dim=-1).mean()

    return nll + kl, nll, kl


def _topic_diversity_loss(topic_embeddings: torch.Tensor) -> torch.Tensor:
    """Penalize similarity between topic embeddings to prevent mode collapse.

    Computes the mean off-diagonal cosine similarity between all pairs
    of topic embeddings. Minimizing this pushes topics apart in embedding
    space, encouraging diverse gene programs.

    Parameters
    ----------
    topic_embeddings : torch.Tensor
        Topic embedding matrix of shape ``(n_topics, embed_dim)``.

    Returns
    -------
    torch.Tensor
        Scalar diversity loss (higher = more topic overlap = worse).
    """
    t_norm = F.normalize(topic_embeddings, dim=-1)
    sim = t_norm @ t_norm.T  # (K, K)
    n_topics = sim.shape[0]
    # Mean off-diagonal similarity
    mask = ~torch.eye(n_topics, dtype=torch.bool, device=sim.device)
    return sim[mask].mean()


# ======================================================================
# Discovery class
# ======================================================================

class TopicModelProgramDiscovery(BaseProgramDiscovery):
    """Discover gene programs using an Embedded Topic Model (ETM).

    This is a SOTA implementation following Dieng et al. (2020) with
    extensions for single-cell genomics data: proper expression matrix
    preprocessing, early stopping, training diagnostics, NPMI-based
    topic coherence, and ablation support.

    Parameters
    ----------
    n_topics : int
        Number of topics (gene programs) to discover.
    hidden_dim : int
        Width of the encoder hidden layers.
    n_hidden_layers : int
        Number of hidden layers in the encoder MLP.
    dropout : float
        Dropout rate in the encoder.
    use_batch_norm : bool
        Whether to use batch normalisation in the encoder.
    freeze_gene_embeddings : bool
        If ``True``, the pre-trained gene embeddings are not updated
        during training.
    embedding_init : str
        ``"pretrained"`` uses provided gene embeddings; ``"random"``
        forces random initialisation (for ablation studies).
    lr : float
        Initial learning rate for the AdamW optimiser.
    use_lr_scheduler : bool
        Whether to apply cosine annealing learning rate scheduling.
    n_epochs : int
        Maximum number of training epochs.
    batch_size : int
        Mini-batch size.
    top_n_genes : int
        Number of top genes to include per topic when building programs.
    early_stopping_patience : int
        Number of epochs with no improvement in validation loss before
        stopping.  Set to 0 to disable early stopping.
    val_fraction : float
        Fraction of cells held out for validation (0.0 to disable).
    n_top_hvg : int | None
        If set, select the top *n_top_hvg* highly variable genes from
        the expression matrix before training.  ``None`` uses all genes.
    coherence_threshold : float | None
        If set, topics with NPMI coherence below this threshold are
        filtered from the final programs.  ``None`` keeps all topics.
    diversity_weight : float
        Weight for topic diversity regularization loss. Penalizes cosine
        similarity between topic embeddings to prevent mode collapse.
        Default 1.0; set to 0 to disable.
    use_decoder_weights : bool
        If ``True``, extract gene programs from the decoder weight matrix
        rather than embedding cosine similarity. The decoder weights are
        an independent linear layer that avoids the mode collapse inherent
        in softmax(cosine similarity) over many genes.
    device : {"cpu", "cuda", "mps", "auto"}
        PyTorch device string. ``"auto"`` picks CUDA/MPS if available.
    random_state : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_topics: int = 20,
        hidden_dim: int = 256,
        n_hidden_layers: int = 2,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
        freeze_gene_embeddings: bool = False,
        embedding_init: Literal["pretrained", "random"] = "pretrained",
        lr: float = 1e-3,
        use_lr_scheduler: bool = True,
        n_epochs: int = 100,
        batch_size: int = 128,
        top_n_genes: int = 50,
        early_stopping_patience: int = 10,
        val_fraction: float = 0.1,
        n_top_hvg: int | None = None,
        coherence_threshold: float | None = None,
        diversity_weight: float = 1.0,
        use_decoder_weights: bool = True,
        device: Literal["cpu", "cuda", "mps", "auto"] = "auto",
        random_state: int | None = 42,
    ) -> None:
        super().__init__()
        if not 0.0 <= val_fraction < 1.0:
            raise ValueError("val_fraction must be in [0.0, 1.0).")
        self.n_topics = n_topics
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.freeze_gene_embeddings = freeze_gene_embeddings
        self.embedding_init = embedding_init
        self.lr = lr
        self.use_lr_scheduler = use_lr_scheduler
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.top_n_genes = top_n_genes
        self.early_stopping_patience = early_stopping_patience
        self.val_fraction = val_fraction
        self.n_top_hvg = n_top_hvg
        self.coherence_threshold = coherence_threshold
        self.diversity_weight = diversity_weight
        self.use_decoder_weights = use_decoder_weights
        self.device_str = device
        self.random_state = random_state

        # Fitted state
        self._model: _ETMCore | None = None
        self._gene_names: list[str] | None = None
        self._topic_gene_weights: np.ndarray | None = None
        self._training_history: TrainingHistory | None = None
        self._topic_coherence: dict[str, float] | None = None
        self._bow_np: np.ndarray | None = None  # stored for coherence computation

    # ----- device resolution -----

    def _resolve_device(self) -> torch.device:
        """Return the appropriate ``torch.device``.

        Returns
        -------
        torch.device
        """
        if self.device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device_str)

    # ----- expression matrix preprocessing -----

    @staticmethod
    def _preprocess_expression(
        expression_matrix: np.ndarray,
        n_top_hvg: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Preprocess expression matrix: library-size normalisation + log1p.

        Parameters
        ----------
        expression_matrix : np.ndarray
            Raw cell-by-gene expression matrix ``(n_cells, n_genes)``.
        n_top_hvg : int | None
            If set, select the top *n_top_hvg* highly variable genes.

        Returns
        -------
        tuple[np.ndarray, np.ndarray | None]
            Preprocessed matrix and optional boolean mask of selected genes
            (``None`` if all genes kept).
        """
        mat = np.asarray(expression_matrix, dtype=np.float64)

        # Handle already-processed data (may contain negative values from scaling)
        if mat.min() < 0:
            # Data has been mean-centered/scaled; shift to non-negative
            mat = mat - mat.min(axis=0, keepdims=True)

        # Library-size normalisation: scale each cell to median total count
        lib_sizes = mat.sum(axis=1, keepdims=True)
        lib_sizes = np.where(lib_sizes == 0, 1.0, lib_sizes)
        median_lib = np.median(lib_sizes)
        mat = mat / lib_sizes * median_lib

        # Log1p transform (safe: mat is now non-negative)
        mat = np.log1p(mat).astype(np.float32)

        # Highly variable gene selection (variance/mean-based)
        hvg_mask: np.ndarray | None = None
        if n_top_hvg is not None and n_top_hvg < mat.shape[1]:
            gene_means = mat.mean(axis=0)
            gene_vars = mat.var(axis=0)
            # Coefficient of variation squared (Fano factor-like)
            safe_means = np.where(gene_means == 0, 1.0, gene_means)
            dispersion = gene_vars / safe_means
            top_idx = np.argsort(dispersion)[::-1][:n_top_hvg]
            hvg_mask = np.zeros(mat.shape[1], dtype=bool)
            hvg_mask[top_idx] = True
            mat = mat[:, hvg_mask]

        return mat, hvg_mask

    # ----- public API -----

    def fit(  # type: ignore[override]
        self,
        embeddings: np.ndarray,
        gene_names: list[str],
        *,
        expression_matrix: np.ndarray | None = None,
        **kwargs: object,
    ) -> "TopicModelProgramDiscovery":
        """Train the ETM and extract gene programs.

        Parameters
        ----------
        embeddings : np.ndarray
            Gene embedding matrix of shape ``(n_genes, embed_dim)``.
        gene_names : list[str]
            Gene names aligned with rows of *embeddings*.
        expression_matrix : np.ndarray | None
            Cell-by-gene expression matrix ``(n_cells, n_genes)``.  If
            ``None``, a synthetic BOW is generated from the embeddings
            (useful for embedding-only workflows).
        **kwargs : object
            Ignored.

        Returns
        -------
        TopicModelProgramDiscovery
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2-D embeddings, got {embeddings.shape}")
        if embeddings.shape[0] != len(gene_names):
            raise ValueError("Embeddings rows must match gene_names length.")

        self._gene_names = list(gene_names)
        n_genes = len(gene_names)
        embed_dim = embeddings.shape[1]

        # Prepare BOW from expression data with proper preprocessing
        hvg_mask: np.ndarray | None = None
        if expression_matrix is not None:
            expr = np.asarray(expression_matrix, dtype=np.float32)
            if expr.shape[1] != n_genes:
                raise ValueError(
                    f"expression_matrix has {expr.shape[1]} genes but "
                    f"expected {n_genes}."
                )
            bow_np, hvg_mask = self._preprocess_expression(expr, self.n_top_hvg)

            if hvg_mask is not None:
                # Subset embeddings and gene names to HVGs
                embeddings = embeddings[hvg_mask]
                self._gene_names = [g for g, m in zip(gene_names, hvg_mask) if m]
                n_genes = len(self._gene_names)
                embed_dim = embeddings.shape[1]
        else:
            # Synthetic BOW from pairwise similarities
            logger.info(
                "No expression_matrix provided. Generating synthetic "
                "BOW from gene embedding similarities."
            )
            bow_np = self._synthetic_bow(embeddings)

        # Store raw (pre-normalisation) BOW for coherence computation
        self._bow_np = bow_np.copy()

        # Normalise BOW per cell (row)
        row_sums = bow_np.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        bow_np = bow_np / row_sums

        device = self._resolve_device()
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        gene_emb_tensor = torch.tensor(embeddings, dtype=torch.float32)

        model = _ETMCore(
            vocab_size=n_genes,
            n_topics=self.n_topics,
            embed_dim=embed_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            n_hidden_layers=self.n_hidden_layers,
            use_batch_norm=self.use_batch_norm,
            gene_embeddings=gene_emb_tensor,
            freeze_gene_embeddings=self.freeze_gene_embeddings,
            embedding_init=self.embedding_init,
        ).to(device)

        optimiser = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-5)

        # Optional cosine annealing scheduler
        scheduler: torch.optim.lr_scheduler.CosineAnnealingLR | None = None
        if self.use_lr_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser, T_max=self.n_epochs, eta_min=self.lr * 0.01
            )

        # Train/validation split
        n_cells = bow_np.shape[0]
        bow_tensor = torch.tensor(bow_np, dtype=torch.float32)

        val_loader: DataLoader | None = None
        if self.val_fraction > 0.0 and n_cells >= 10:
            n_val = min(max(1, int(n_cells * self.val_fraction)), n_cells - 1)
            perm = np.random.permutation(n_cells)
            val_idx, train_idx = perm[:n_val], perm[n_val:]

            train_dataset = TensorDataset(bow_tensor[train_idx])
            val_dataset = TensorDataset(bow_tensor[val_idx])
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )
        else:
            train_dataset = TensorDataset(bow_tensor)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        # Training history
        history = TrainingHistory()

        # Early stopping state
        best_val_loss = float("inf")
        patience_counter = 0
        best_state_dict: dict | None = None

        # Training loop
        for epoch in range(1, self.n_epochs + 1):
            model.train()
            total_loss = 0.0
            total_nll = 0.0
            total_kl = 0.0
            n_batches = 0

            for (batch_bow,) in train_loader:
                batch_bow = batch_bow.to(device)
                recon, mu, log_sigma = model(batch_bow)
                loss, nll, kl = _elbo_loss(recon, batch_bow, mu, log_sigma)

                # Topic diversity regularization to prevent mode collapse
                if self.diversity_weight > 0:
                    div_loss = _topic_diversity_loss(model.topic_embeddings)
                    loss = loss + self.diversity_weight * div_loss

                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimiser.step()
                total_loss += loss.item()
                total_nll += nll.item()
                total_kl += kl.item()
                n_batches += 1

            if scheduler is not None:
                scheduler.step()

            avg_loss = total_loss / max(n_batches, 1)
            avg_nll = total_nll / max(n_batches, 1)
            avg_kl = total_kl / max(n_batches, 1)

            history.elbo.append(avg_loss)
            history.reconstruction_loss.append(avg_nll)
            history.kl_divergence.append(avg_kl)

            # Validation perplexity
            val_ppl = float("nan")
            val_loss_for_stopping = avg_loss  # fallback if no val set
            if val_loader is not None:
                model.eval()
                val_total_nll = 0.0
                val_total_tokens = 0.0
                with torch.no_grad():
                    for (vbatch,) in val_loader:
                        vbatch = vbatch.to(device)
                        vrecon, vmu, vlog = model(vbatch)
                        vnll = -(vbatch * vrecon).sum().item()
                        val_total_nll += vnll
                        val_total_tokens += vbatch.sum().item()
                if val_total_tokens > 0:
                    val_ppl = float(np.exp(val_total_nll / val_total_tokens))
                else:
                    val_ppl = float("inf")
                history.val_perplexity.append(val_ppl)
                val_loss_for_stopping = val_total_nll / max(val_total_tokens, 1.0)

            # Early stopping check
            if self.early_stopping_patience > 0:
                if val_loss_for_stopping < best_val_loss - 1e-6:
                    best_val_loss = val_loss_for_stopping
                    patience_counter = 0
                    best_state_dict = {
                        k: v.cpu().clone() for k, v in model.state_dict().items()
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        history.stopped_epoch = epoch
                        logger.info(
                            "Early stopping at epoch %d (patience=%d).",
                            epoch, self.early_stopping_patience,
                        )
                        break

            if epoch % max(1, self.n_epochs // 10) == 0 or epoch == 1:
                lr_now = (
                    scheduler.get_last_lr()[0] if scheduler is not None
                    else self.lr
                )
                val_info = f"  val_ppl={val_ppl:.2f}" if val_loader is not None else ""
                logger.info(
                    "Epoch %d/%d  loss=%.4f  nll=%.4f  kl=%.4f  lr=%.2e%s",
                    epoch, self.n_epochs, avg_loss, avg_nll, avg_kl, lr_now, val_info,
                )

        # Restore best model if early stopping was used and found a best
        if best_state_dict is not None:
            model.load_state_dict(
                {k: v.to(device) for k, v in best_state_dict.items()}
            )

        self._model = model
        self._training_history = history

        # Extract topic-gene weights
        model.eval()
        with torch.no_grad():
            if self.use_decoder_weights:
                # Use decoder weight matrix: more diverse, avoids softmax
                # concentration from cosine similarity
                loadings = model.get_decoder_topic_gene_loadings()  # (K, V)
                weights = torch.softmax(loadings, dim=-1).cpu().numpy()
            else:
                logits = model._get_topic_gene_logits()  # (K, V)
                weights = torch.softmax(logits, dim=-1).cpu().numpy()  # (K, V)

        self._topic_gene_weights = weights

        # Compute topic coherence
        self._compute_topic_coherence(weights)

        # Build programs (with optional coherence filtering)
        self._build_programs(weights)
        return self

    def get_programs(self) -> dict[str, list[str]]:
        """Return gene programs.

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

    def get_topic_gene_weights(self) -> np.ndarray:
        """Return the full topic--gene weight matrix.

        Returns
        -------
        np.ndarray
            Shape ``(n_topics, n_genes)`` with non-negative weights
            summing to 1 per topic.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        self._check_is_fitted()
        assert self._topic_gene_weights is not None
        result: np.ndarray = np.asarray(self._topic_gene_weights.copy())
        return result

    def get_training_history(self) -> TrainingHistory:
        """Return the full training diagnostic history.

        Returns
        -------
        TrainingHistory
            Dataclass with per-epoch ELBO, reconstruction loss,
            KL divergence, validation perplexity, and stopped epoch.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        self._check_is_fitted()
        assert self._training_history is not None
        return self._training_history

    def get_topic_coherence(self) -> dict[str, float]:
        """Return NPMI-based topic coherence scores.

        Returns
        -------
        dict[str, float]
            Mapping from topic name to its coherence score (higher is
            better, range roughly [-1, 1]).

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        self._check_is_fitted()
        assert self._topic_coherence is not None
        return dict(self._topic_coherence)

    def get_decoder_topic_gene_weights(self) -> np.ndarray:
        """Return topic-gene loadings from the direct decoder weight matrix.

        This provides a more principled view of topic-gene associations
        that does not rely on embedding similarity.

        Returns
        -------
        np.ndarray
            Shape ``(n_topics, n_genes)``, softmax-normalised per topic.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        self._check_is_fitted()
        assert self._model is not None
        self._model.eval()
        with torch.no_grad():
            loadings = self._model.get_decoder_topic_gene_loadings()
            weights = torch.softmax(loadings, dim=-1).cpu().numpy()
        return weights

    # ----- internal helpers -----

    @staticmethod
    def _synthetic_bow(embeddings: np.ndarray, n_samples: int = 500) -> np.ndarray:
        """Generate synthetic bag-of-words from embedding similarities.

        Each synthetic "cell" is created by sampling a random direction
        in embedding space and weighting genes by their cosine similarity
        to that direction, then applying softmax to create a probability
        distribution that is treated as a bag-of-words.

        Parameters
        ----------
        embeddings : np.ndarray
            ``(n_genes, embed_dim)``
        n_samples : int
            Number of synthetic cells to generate.

        Returns
        -------
        np.ndarray
            ``(n_samples, n_genes)`` non-negative BOW matrix.
        """
        n_genes, embed_dim = embeddings.shape
        emb_norm = embeddings / (
            np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        )

        rng = np.random.default_rng(seed=42)
        directions = rng.standard_normal((n_samples, embed_dim))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12

        # Cosine similarity between each direction and each gene
        sims = directions @ emb_norm.T  # (n_samples, n_genes)

        # Shift and exponentiate to create non-negative pseudo-counts
        sims_shifted = sims - sims.min(axis=1, keepdims=True)
        bow = np.exp(sims_shifted) - 1.0
        bow = np.maximum(bow, 0.0)
        bow_result: np.ndarray = np.asarray(bow.astype(np.float32))
        return bow_result

    def _compute_topic_coherence(self, weights: np.ndarray) -> None:
        """Compute NPMI-based coherence for each topic.

        Uses the top genes per topic and the BOW matrix to compute
        normalised pointwise mutual information (NPMI).

        Parameters
        ----------
        weights : np.ndarray
            ``(n_topics, n_genes)`` topic-gene weight matrix.
        """
        assert self._bow_np is not None
        assert self._gene_names is not None

        bow = self._bow_np  # (n_cells, n_genes), raw counts / pseudo-counts
        n_cells = bow.shape[0]

        # Binary occurrence: gene is "present" if above median for that gene
        gene_medians = np.median(bow, axis=0, keepdims=True)
        occurrence = (bow > gene_medians).astype(np.float64)

        # Document frequency: fraction of cells where gene is present
        doc_freq = occurrence.mean(axis=0)  # (n_genes,)
        # Clamp to avoid log(0)
        doc_freq = np.clip(doc_freq, 1.0 / (n_cells + 1), 1.0)

        # Co-occurrence matrix (precompute for efficiency with top genes)
        n_top = min(self.top_n_genes, weights.shape[1])
        coherence: dict[str, float] = {}

        for k in range(weights.shape[0]):
            topic_name = f"topic_{k}"
            top_genes = np.argsort(weights[k])[::-1][:n_top]

            if len(top_genes) < 2:
                coherence[topic_name] = 0.0
                continue

            npmi_sum = 0.0
            n_pairs = 0

            for i in range(len(top_genes)):
                for j in range(i + 1, len(top_genes)):
                    gi, gj = top_genes[i], top_genes[j]
                    p_i = doc_freq[gi]
                    p_j = doc_freq[gj]
                    # Joint probability
                    p_ij = (occurrence[:, gi] * occurrence[:, gj]).mean()
                    p_ij = max(p_ij, 1.0 / (n_cells + 1))

                    # PMI
                    pmi = math.log(p_ij) - math.log(p_i) - math.log(p_j)
                    # Normalise by -log(p_ij)
                    denom = -math.log(p_ij)
                    if denom > 0:
                        npmi_sum += pmi / denom
                    n_pairs += 1

            coherence[topic_name] = npmi_sum / max(n_pairs, 1)

        self._topic_coherence = coherence

    def _build_programs(self, weights: np.ndarray) -> None:
        """Build program dictionaries from topic-gene weight matrix.

        If ``coherence_threshold`` is set, topics with coherence below
        the threshold are excluded from the final programs.

        Parameters
        ----------
        weights : np.ndarray
            ``(n_topics, n_genes)``
        """
        assert self._gene_names is not None
        gene_names = self._gene_names
        programs: dict[str, list[str]] = {}
        scores: dict[str, list[tuple[str, float]]] = {}

        for k in range(weights.shape[0]):
            prog_name = f"topic_{k}"

            # Coherence filtering
            if (
                self.coherence_threshold is not None
                and self._topic_coherence is not None
                and self._topic_coherence.get(prog_name, 0.0) < self.coherence_threshold
            ):
                logger.info(
                    "Filtering topic '%s' with coherence %.4f < threshold %.4f.",
                    prog_name,
                    self._topic_coherence.get(prog_name, 0.0),
                    self.coherence_threshold,
                )
                continue

            gene_weights = weights[k]  # (V,)
            top_indices = np.argsort(gene_weights)[::-1][: self.top_n_genes]

            scored_genes: list[tuple[str, float]] = []
            gene_list: list[str] = []
            for idx in top_indices:
                g = gene_names[idx]
                w = float(gene_weights[idx])
                scored_genes.append((g, w))
                gene_list.append(g)

            programs[prog_name] = gene_list
            scores[prog_name] = scored_genes

        self.programs_ = programs
        self.program_scores_ = scores
