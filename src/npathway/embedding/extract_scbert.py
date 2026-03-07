"""scBERT embedding extractor for gene representations.

This module provides a concrete implementation of
:class:`~npathway.embedding.base.BaseEmbeddingExtractor` that loads a
pre-trained scBERT model and extracts gene embeddings using the masked
language model (MLM) approach.

Reference:
    Yang, F. et al. "scBERT as a large-scale pretrained deep language model
    for cell type annotation of single-cell RNA-seq data." Nature Machine
    Intelligence (2022).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

try:
    from anndata import AnnData
except ImportError:
    AnnData = Any  # type: ignore[assignment, misc]

from npathway.embedding.base import BaseEmbeddingExtractor

logger = logging.getLogger(__name__)

_DEFAULT_BATCH_SIZE: int = 64
_SCBERT_N_GENES: int = 16906  # Number of genes in the default scBERT vocab.


class ScBERTEmbeddingExtractor(BaseEmbeddingExtractor):
    """Extract gene embeddings from a pre-trained scBERT model.

    scBERT employs a BERT-style transformer with a performer backbone trained
    on large-scale scRNA-seq data via masked language modelling.  Gene
    embeddings are derived from the hidden states of the gene tokens.

    This extractor supports:
    * **Universal embeddings** -- averaged across all cells.
    * **Context-specific embeddings** -- averaged within each cell-type group.

    Attributes:
        gene_vocab: Ordered list of gene names in the scBERT vocabulary.
        batch_size: Number of cells per forward-pass batch.
    """

    def __init__(
        self,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        device: str | None = None,
    ) -> None:
        """Initialise the scBERT embedding extractor.

        Args:
            batch_size: Number of cells per batch.
            device: Computation device or ``None`` for auto-detection.
        """
        super().__init__(model_name="scBERT")
        self.gene_vocab: list[str] = []
        self.batch_size: int = batch_size
        self._n_layers: int = 0
        self._set_device(device)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self, model_path: str) -> None:
        """Load a pre-trained scBERT model.

        The model directory should contain:
        * A checkpoint file (``*.pt`` or ``*.pth``).
        * Optionally a gene list file (``gene_list.txt`` or ``gene_vocab.txt``).

        If the gene vocabulary file is not found, the extractor will attempt to
        derive the vocabulary from the model's embedding layer dimensions.

        Args:
            model_path: Path to the directory containing model artefacts.

        Raises:
            ImportError: If ``torch`` or required modelling libraries are not
                installed.
            FileNotFoundError: If the model directory does not exist or no
                checkpoint is found.
            RuntimeError: If model loading fails.
        """
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for scBERT embedding extraction. "
                "Install it with: pip install torch"
            ) from exc

        model_dir = Path(model_path)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"scBERT model directory not found: {model_dir}"
            )

        # -- Load gene vocabulary -------------------------------------------
        self._load_gene_vocab(model_dir)

        # -- Locate checkpoint ----------------------------------------------
        checkpoint_path: Path | None = None
        for pattern in ["best_model.pt", "scbert.pt", "*.pt", "*.pth"]:
            candidates = list(model_dir.glob(pattern))
            if candidates:
                checkpoint_path = candidates[0]
                break

        if checkpoint_path is None:
            raise FileNotFoundError(
                f"No model checkpoint (.pt/.pth) found in {model_dir}"
            )
        logger.info("Loading scBERT checkpoint from %s", checkpoint_path)

        # -- Build model architecture ---------------------------------------
        # scBERT uses the Performer (FAVOR+) attention mechanism.  We try to
        # import the original scBERT module first; if unavailable we fall back
        # to a standard transformer encoder.
        model: nn.Module
        try:
            from performer_pytorch import PerformerLM
            n_genes = len(self.gene_vocab) if self.gene_vocab else _SCBERT_N_GENES
            model = PerformerLM(
                num_tokens=n_genes + 1,  # +1 for CLS or PAD
                dim=200,
                depth=6,
                heads=10,
                max_seq_len=n_genes + 1,
                causal=False,
            )
            self._n_layers = 6
            self.embedding_dim = 200
            logger.info("Built scBERT with PerformerLM backbone")
        except ImportError:
            logger.info(
                "performer_pytorch not found; building scBERT with standard "
                "TransformerEncoder backbone."
            )
            # Infer dimensions from checkpoint state_dict.
            state_dict = torch.load(
                str(checkpoint_path),
                map_location="cpu",
                weights_only=True,
            )
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]

            # Try to infer embedding dim from keys.
            embed_dim = 200
            n_layers = 6
            n_heads = 10
            for key, val in state_dict.items():
                if "token_emb" in key and "weight" in key:
                    embed_dim = val.shape[1]
                    break
                if "embedding" in key.lower() and "weight" in key:
                    embed_dim = val.shape[-1]
                    break

            n_genes = len(self.gene_vocab) if self.gene_vocab else _SCBERT_N_GENES
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                dim_feedforward=embed_dim * 4,
                batch_first=True,
            )
            encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            embedding = nn.Embedding(n_genes + 1, embed_dim)

            class _ScBERTFallback(nn.Module):
                """Minimal scBERT-compatible wrapper."""

                def __init__(
                    self,
                    embedding: nn.Embedding,
                    encoder: nn.TransformerEncoder,
                    n_layers: int,
                ) -> None:
                    super().__init__()
                    self.embedding = embedding
                    self.encoder = encoder
                    self._n_layers = n_layers

                def forward(
                    self,
                    x: torch.Tensor,
                    values: torch.Tensor | None = None,
                ) -> torch.Tensor:
                    emb = self.embedding(x)
                    if values is not None:
                        emb = emb * values.unsqueeze(-1)
                    return self.encoder(emb)  # type: ignore[no-any-return, return-value]

            model = _ScBERTFallback(embedding, encoder, n_layers)
            self._n_layers = n_layers
            self.embedding_dim = embed_dim
            # Try loading state dict (may be partial match).
            try:
                model.load_state_dict(state_dict, strict=False)
            except Exception as load_exc:
                logger.warning(
                    "Partial state_dict load for fallback model: %s", load_exc
                )

            # Re-load state dict for Performer path is handled below.
            state_dict = None  # Indicate already loaded.

        # Load weights for Performer path.
        if not isinstance(model, nn.Module):
            raise RuntimeError("Failed to construct scBERT model.")

        # Load state dict if not already loaded in fallback.
        if not hasattr(model, "_n_layers") or getattr(model, "_n_layers", 0) == 0:
            state_dict = torch.load(
                str(checkpoint_path),
                map_location=self._device,
                weights_only=True,
            )
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            try:
                model.load_state_dict(state_dict, strict=False)
            except Exception as exc:
                logger.warning("Partial state_dict load: %s", exc)

        model.to(self._device)
        model.eval()
        self.model = model
        self.is_loaded = True
        logger.info(
            "scBERT model loaded (%d layers, dim=%d)",
            self._n_layers,
            self.embedding_dim,
        )

    def _load_gene_vocab(self, model_dir: Path) -> None:
        """Load the ordered gene vocabulary for scBERT.

        Args:
            model_dir: Path to the model directory.
        """
        for name in [
            "gene_list.txt",
            "gene_vocab.txt",
            "panglao_gene_list.txt",
            "gene_names.txt",
        ]:
            p = model_dir / name
            if p.exists():
                with open(p, "r", encoding="utf-8") as fh:
                    self.gene_vocab = [
                        line.strip() for line in fh if line.strip()
                    ]
                logger.info(
                    "Loaded scBERT gene vocabulary (%d genes) from %s",
                    len(self.gene_vocab),
                    p,
                )
                return

        # Try pickle format.
        import pickle

        for name in ["gene_list.pkl", "gene_vocab.pkl"]:
            p = model_dir / name
            if p.exists():
                with open(p, "rb") as fh:
                    self.gene_vocab = pickle.load(fh)
                logger.info(
                    "Loaded scBERT gene vocabulary (%d genes) from %s",
                    len(self.gene_vocab),
                    p,
                )
                return

        logger.warning(
            "Gene vocabulary not found in %s. Gene mapping will rely on "
            "positional matching with the dataset.",
            model_dir,
        )

    # ------------------------------------------------------------------
    # Gene mapping
    # ------------------------------------------------------------------

    def _map_genes(
        self,
        gene_names: list[str],
    ) -> tuple[list[str], list[int], list[int]]:
        """Map dataset gene names to scBERT vocabulary positions.

        Args:
            gene_names: Gene names from the AnnData object.

        Returns:
            Tuple of (matched_gene_names, vocab_positions, dataset_col_indices).

        Raises:
            ValueError: If no overlapping genes are found.
        """
        if not self.gene_vocab:
            # Without a vocabulary, assume positional correspondence.
            n = min(len(gene_names), _SCBERT_N_GENES)
            logger.warning(
                "No scBERT gene vocabulary loaded; using first %d genes "
                "in positional order.",
                n,
            )
            return (
                gene_names[:n],
                list(range(n)),
                list(range(n)),
            )

        vocab_set: dict[str, int] = {
            g: i for i, g in enumerate(self.gene_vocab)
        }
        # Also try uppercased versions.
        vocab_upper: dict[str, int] = {
            g.upper(): i for i, g in enumerate(self.gene_vocab)
        }

        matched_names: list[str] = []
        matched_vocab_pos: list[int] = []
        matched_cols: list[int] = []

        for col_idx, gene in enumerate(gene_names):
            if gene in vocab_set:
                matched_names.append(gene)
                matched_vocab_pos.append(vocab_set[gene])
                matched_cols.append(col_idx)
            elif gene.upper() in vocab_upper:
                matched_names.append(gene)
                matched_vocab_pos.append(vocab_upper[gene.upper()])
                matched_cols.append(col_idx)

        if not matched_names:
            raise ValueError(
                "No overlapping genes found between the dataset and the "
                "scBERT vocabulary. Check that gene names use the expected "
                "nomenclature."
            )

        logger.info(
            "Mapped %d / %d dataset genes to scBERT vocabulary",
            len(matched_names),
            len(gene_names),
        )
        return matched_names, matched_vocab_pos, matched_cols

    # ------------------------------------------------------------------
    # Embedding extraction helpers
    # ------------------------------------------------------------------

    def _extract_with_hooks(
        self,
        gene_ids: Any,
        values: Any,
        layer: int | str,
    ) -> np.ndarray:
        """Run forward pass and collect hidden states from transformer layers.

        Args:
            gene_ids: Tensor of shape ``(batch, seq_len)``.
            values: Tensor of shape ``(batch, seq_len)``.
            layer: Resolved layer index or ``"all"``.

        Returns:
            Numpy array of shape ``(batch, seq_len, dim)``.
        """
        import torch

        hidden_states: list[torch.Tensor] = []
        hooks: list[Any] = []

        # Identify the transformer layers to hook.
        encoder = None
        if hasattr(self.model, "encoder"):
            encoder = self.model.encoder
        elif hasattr(self.model, "net") and hasattr(self.model.net, "layers"):
            encoder = self.model.net

        if encoder is not None:
            layers_iterable = None
            if hasattr(encoder, "layers"):
                layers_iterable = encoder.layers
            elif hasattr(encoder, "layer"):
                layers_iterable = encoder.layer

            if layers_iterable is not None:
                for enc_layer in layers_iterable:
                    hook = enc_layer.register_forward_hook(
                        lambda _m, _i, o, _hs=hidden_states: _hs.append(
                            o if isinstance(o, torch.Tensor) else o[0]
                        )
                    )
                    hooks.append(hook)

        with torch.no_grad():
            try:
                _ = self.model(gene_ids, values)
            except TypeError:
                try:
                    _ = self.model(gene_ids)
                except Exception:
                    pass

        for h in hooks:
            h.remove()

        if not hidden_states:
            # Fallback: use the static embedding table.
            logger.warning(
                "Could not capture hidden states; using static embeddings."
            )
            emb_layer = None
            if hasattr(self.model, "embedding"):
                emb_layer = self.model.embedding
            elif hasattr(self.model, "token_emb"):
                emb_layer = self.model.token_emb

            if emb_layer is not None:
                with torch.no_grad():
                    emb = emb_layer(gene_ids)
                    if values is not None:
                        emb = emb * values.unsqueeze(-1)
                scbert_result: np.ndarray = np.asarray(emb.cpu().numpy())
                return scbert_result
            raise RuntimeError(
                "Unable to extract embeddings from scBERT model."
            )

        if layer == "all":
            stacked = torch.cat(hidden_states, dim=-1)
            return stacked.cpu().numpy()

        idx = int(layer)
        if idx >= len(hidden_states):
            idx = len(hidden_states) - 1
        return hidden_states[idx].cpu().numpy()

    def _process_cells(
        self,
        X: np.ndarray,
        matched_vocab_pos: list[int],
        matched_cols: list[int],
        n_genes: int,
        resolved_layer: int | str,
    ) -> np.ndarray:
        """Process cells through the model and accumulate per-gene embeddings.

        Args:
            X: Dense expression matrix ``(n_cells, n_total_genes)``.
            matched_vocab_pos: Vocabulary positions for matched genes.
            matched_cols: Column indices into X for matched genes.
            n_genes: Number of matched genes.
            resolved_layer: Layer to extract from.

        Returns:
            Averaged gene embedding array ``(n_genes, dim)``.
        """
        import torch

        dim = (
            self.embedding_dim
            if resolved_layer != "all"
            else self.embedding_dim * self._n_layers
        )
        embedding_sum = np.zeros((n_genes, dim), dtype=np.float64)
        n_cells = X.shape[0]

        gene_ids_tensor = torch.tensor(
            matched_vocab_pos, dtype=torch.long
        ).to(self._device)

        for start in range(0, n_cells, self.batch_size):
            end = min(start + self.batch_size, n_cells)
            batch_expr = X[start:end][:, matched_cols]  # (bs, n_genes)
            bs = batch_expr.shape[0]

            gene_ids_batch = gene_ids_tensor.unsqueeze(0).expand(bs, -1)
            values_batch = torch.tensor(
                batch_expr, dtype=torch.float32
            ).to(self._device)

            hidden = self._extract_with_hooks(
                gene_ids_batch, values_batch, resolved_layer
            )  # (bs, n_genes, dim)

            # Handle dimension mismatch if model output has different seq_len.
            if hidden.shape[1] != n_genes:
                # Take the first n_genes positions.
                hidden = hidden[:, :n_genes, :]

            embedding_sum += hidden.sum(axis=0)

        embedding_sum /= max(n_cells, 1)
        return embedding_sum.astype(np.float32)

    # ------------------------------------------------------------------
    # Public extraction methods
    # ------------------------------------------------------------------

    def extract_gene_embeddings(
        self,
        adata: AnnData,
        layer: int | str = -1,
    ) -> np.ndarray:
        """Extract universal gene embeddings averaged over all cells.

        Args:
            adata: AnnData object with gene expression data.
            layer: Transformer layer to extract from (default: last layer).

        Returns:
            Array of shape ``(n_matched_genes, embedding_dim)``.
        """
        self._check_loaded()
        resolved_layer = self._resolve_layer(layer, self._n_layers)

        gene_names = list(adata.var_names)
        matched_names, matched_vocab_pos, matched_cols = self._map_genes(
            gene_names
        )
        self.gene_names = matched_names

        if hasattr(adata.X, "toarray"):
            X = adata.X.toarray()
        else:
            X = np.asarray(adata.X)

        embeddings = self._process_cells(
            X, matched_vocab_pos, matched_cols, len(matched_names), resolved_layer
        )
        self.embedding_dim = embeddings.shape[1]
        logger.info(
            "Extracted scBERT gene embeddings: shape %s", embeddings.shape
        )
        return embeddings

    def extract_context_embeddings(
        self,
        adata: AnnData,
        cell_type_key: str,
        layer: int | str = -1,
    ) -> dict[str, np.ndarray]:
        """Extract gene embeddings specific to each cell type.

        Args:
            adata: AnnData object with expression data and cell-type annotations.
            cell_type_key: Column in ``adata.obs`` containing cell-type labels.
            layer: Transformer layer to extract from (default: last layer).

        Returns:
            Dictionary mapping cell-type labels to embedding arrays.
        """
        self._check_loaded()

        if cell_type_key not in adata.obs.columns:
            raise KeyError(
                f"Cell type key '{cell_type_key}' not found in adata.obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )

        resolved_layer = self._resolve_layer(layer, self._n_layers)
        gene_names = list(adata.var_names)
        matched_names, matched_vocab_pos, matched_cols = self._map_genes(
            gene_names
        )
        self.gene_names = matched_names

        if hasattr(adata.X, "toarray"):
            X = adata.X.toarray()
        else:
            X = np.asarray(adata.X)

        cell_types = adata.obs[cell_type_key]
        unique_types = sorted(cell_types.unique())

        results: dict[str, np.ndarray] = {}
        for ct in unique_types:
            mask = (cell_types == ct).values
            ct_X = X[mask]
            if ct_X.shape[0] == 0:
                continue

            emb = self._process_cells(
                ct_X,
                matched_vocab_pos,
                matched_cols,
                len(matched_names),
                resolved_layer,
            )
            results[str(ct)] = emb
            logger.info(
                "Extracted scBERT context embeddings for '%s' (%d cells)",
                ct,
                ct_X.shape[0],
            )

        return results
