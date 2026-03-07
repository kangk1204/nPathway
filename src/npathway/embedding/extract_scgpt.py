"""scGPT embedding extractor for gene representations.

This module provides a concrete implementation of
:class:`~npathway.embedding.base.BaseEmbeddingExtractor` that loads a
pre-trained scGPT model and extracts gene token embeddings from specified
transformer layers.

Reference:
    Cui, H. et al. "scGPT: toward building a foundation model for
    single-cell multi-omics using generative AI." Nature Methods (2024).
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

# Batch size for processing cells through the model.
_DEFAULT_BATCH_SIZE: int = 64


class ScGPTEmbeddingExtractor(BaseEmbeddingExtractor):
    """Extract gene embeddings from a pre-trained scGPT model.

    scGPT uses a transformer architecture trained on large-scale single-cell
    RNA-seq data.  Gene tokens are embedded via a learned vocabulary, and
    contextual representations are obtained by running expression-value-aware
    forward passes through the transformer.

    This extractor supports:
    * **Universal embeddings** -- averaged across all cells in the dataset.
    * **Context-specific embeddings** -- averaged within each cell-type group.

    Attributes:
        vocab: Mapping from gene name to vocabulary index.
        pad_token_id: Token id used for padding.
        batch_size: Number of cells processed per forward-pass batch.
    """

    def __init__(
        self,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        device: str | None = None,
    ) -> None:
        """Initialise the scGPT embedding extractor.

        Args:
            batch_size: Number of cells to process in each batch during
                embedding extraction.
            device: Computation device (``"cpu"``, ``"cuda"``, ``"mps"``).
                Automatically selected when ``None``.
        """
        super().__init__(model_name="scGPT")
        self.vocab: dict[str, int] = {}
        self.pad_token_id: int = 0
        self.batch_size: int = batch_size
        self._n_layers: int = 0
        self._set_device(device)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self, model_path: str) -> None:
        """Load a pre-trained scGPT model from a local directory.

        The directory is expected to contain:
        * ``best_model.pt`` -- model checkpoint.
        * ``vocab.json`` (or ``vocab.pkl``) -- gene-to-id mapping.
        * ``args.json`` -- model hyper-parameters.

        Args:
            model_path: Path to the directory containing model artefacts.

        Raises:
            ImportError: If ``scgpt`` is not installed.
            FileNotFoundError: If the model directory or required files are
                missing.
            RuntimeError: If model loading fails for any other reason.
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for scGPT embedding extraction. "
                "Install it with: pip install torch"
            ) from exc

        try:
            from scgpt.model import TransformerModel
            from scgpt.tokenizer import GeneVocab
        except ImportError as exc:
            raise ImportError(
                "scGPT is required but not installed. "
                "Install it with: pip install scgpt"
            ) from exc

        model_dir = Path(model_path)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"scGPT model directory not found: {model_dir}"
            )

        # -- Load vocabulary ------------------------------------------------
        vocab_path = model_dir / "vocab.json"
        if not vocab_path.exists():
            # Older releases may use a pickle file.
            vocab_path_pkl = model_dir / "vocab.pkl"
            if vocab_path_pkl.exists():
                vocab_path = vocab_path_pkl
            else:
                raise FileNotFoundError(
                    "Gene vocabulary file (vocab.json or vocab.pkl) not "
                    f"found in {model_dir}"
                )

        gene_vocab = GeneVocab.from_file(str(vocab_path))
        self.vocab = dict(gene_vocab)
        self.pad_token_id = self.vocab.get("<pad>", 0)
        logger.info(
            "Loaded scGPT vocabulary with %d genes", len(self.vocab)
        )

        # -- Load model configuration --------------------------------------
        import json

        args_path = model_dir / "args.json"
        if args_path.exists():
            with open(args_path, "r", encoding="utf-8") as fh:
                model_args = json.load(fh)
        else:
            # Fall back to sensible defaults matching the published model.
            logger.warning(
                "args.json not found in %s; using default hyper-parameters.",
                model_dir,
            )
            model_args = {
                "embsize": 512,
                "d_hid": 512,
                "nlayers": 12,
                "nhead": 8,
            }

        embsize: int = int(model_args.get("embsize", 512))
        d_hid: int = int(model_args.get("d_hid", 512))
        nlayers: int = int(model_args.get("nlayers", 12))
        nhead: int = int(model_args.get("nhead", 8))
        self._n_layers = nlayers
        self.embedding_dim = embsize

        # -- Instantiate model and load weights ----------------------------
        n_tokens = len(self.vocab)
        model = TransformerModel(
            ntoken=n_tokens,
            d_model=embsize,
            nhead=nhead,
            d_hid=d_hid,
            nlayers=nlayers,
            vocab=gene_vocab,
            pad_token=self.vocab.get("<pad>", "<pad>"),
        )

        checkpoint_path = model_dir / "best_model.pt"
        if not checkpoint_path.exists():
            # Try alternative naming conventions.
            candidates = list(model_dir.glob("*.pt")) + list(
                model_dir.glob("*.pth")
            )
            if not candidates:
                raise FileNotFoundError(
                    f"No model checkpoint (.pt/.pth) found in {model_dir}"
                )
            checkpoint_path = candidates[0]
            logger.info("Using checkpoint: %s", checkpoint_path)

        state_dict = torch.load(
            str(checkpoint_path),
            map_location=self._device,
            weights_only=True,
        )
        # The checkpoint may store the state_dict under a key.
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        model.load_state_dict(state_dict, strict=False)
        model.to(self._device)
        model.eval()

        self.model = model
        self.is_loaded = True
        logger.info(
            "scGPT model loaded successfully (%d layers, dim=%d)",
            nlayers,
            embsize,
        )

    # ------------------------------------------------------------------
    # Embedding extraction helpers
    # ------------------------------------------------------------------

    def _map_genes(self, gene_names: list[str]) -> tuple[list[str], list[int]]:
        """Map dataset gene names to model vocabulary indices.

        Args:
            gene_names: Gene names from the AnnData object.

        Returns:
            Tuple of (matched_gene_names, vocab_indices).

        Raises:
            ValueError: If no genes overlap with the model vocabulary.
        """
        matched_names: list[str] = []
        matched_ids: list[int] = []
        for gene in gene_names:
            if gene in self.vocab:
                matched_names.append(gene)
                matched_ids.append(self.vocab[gene])
            else:
                # Try upper-cased name (common for mouse/human mapping).
                upper = gene.upper()
                if upper in self.vocab:
                    matched_names.append(gene)
                    matched_ids.append(self.vocab[upper])

        if not matched_names:
            raise ValueError(
                "No overlapping genes found between the dataset and the "
                "scGPT vocabulary. Ensure gene names follow the expected "
                "nomenclature (e.g. HGNC symbols)."
            )

        logger.info(
            "Mapped %d / %d dataset genes to scGPT vocabulary",
            len(matched_names),
            len(gene_names),
        )
        return matched_names, matched_ids

    def _tokenize_cell(
        self,
        expression: np.ndarray,
        gene_ids: list[int],
    ) -> tuple[Any, Any]:
        """Prepare token and value tensors for a single cell.

        Args:
            expression: 1-D expression vector for the cell (only matched genes).
            gene_ids: Vocabulary indices corresponding to the matched genes.

        Returns:
            Tuple of (gene_id_tensor, expression_value_tensor) both of shape
            ``(n_matched_genes,)``.
        """
        import torch

        gene_id_tensor = torch.tensor(gene_ids, dtype=torch.long)
        # Normalise expression values to [0, max_value_bin] as scGPT expects.
        expr = expression.astype(np.float32)
        # Bin expression values (rank-based binning simplified to direct values).
        value_tensor = torch.tensor(expr, dtype=torch.float32)
        return gene_id_tensor, value_tensor

    def _extract_layer_output(
        self,
        gene_ids_batch: Any,
        values_batch: Any,
        layer: int | str,
    ) -> np.ndarray:
        """Run forward pass and collect hidden states from the requested layer.

        Args:
            gene_ids_batch: Tensor of shape ``(batch, seq_len)``.
            values_batch: Tensor of shape ``(batch, seq_len)``.
            layer: Resolved layer index or ``"all"``.

        Returns:
            Numpy array of shape ``(batch, seq_len, embed_dim)`` (or wider
            when ``layer="all"``).
        """
        import torch

        with torch.no_grad():
            # scGPT's forward returns (output, hidden_states) when
            # output_hidden_states=True.  We use a hook-based approach to be
            # compatible with various scGPT releases.
            hidden_states: list[torch.Tensor] = []

            hooks = []
            if hasattr(self.model, "transformer_encoder"):
                encoder = self.model.transformer_encoder
                if hasattr(encoder, "layers"):
                    for enc_layer in encoder.layers:
                        hook = enc_layer.register_forward_hook(
                            lambda _mod, _inp, out, _hs=hidden_states: _hs.append(
                                out if isinstance(out, torch.Tensor) else out[0]
                            )
                        )
                        hooks.append(hook)

            # Forward pass through the model.
            try:
                src_key_padding_mask = gene_ids_batch.eq(self.pad_token_id)
                _ = self.model(
                    gene_ids_batch,
                    values_batch,
                    src_key_padding_mask=src_key_padding_mask,
                )
            except TypeError:
                # Fallback: some scGPT versions have a simpler forward.
                _ = self.model(gene_ids_batch, values_batch)

            for h in hooks:
                h.remove()

            if not hidden_states:
                # If hooks didn't capture anything, fall back to the static
                # gene embedding table (non-contextual).
                logger.warning(
                    "Could not capture hidden states from transformer layers. "
                    "Falling back to static gene embeddings."
                )
                if hasattr(self.model, "encoder"):
                    emb = self.model.encoder(gene_ids_batch)
                elif hasattr(self.model, "gene_encoder"):
                    emb = self.model.gene_encoder(gene_ids_batch)
                else:
                    raise RuntimeError(
                        "Unable to extract embeddings from scGPT model: "
                        "no encoder or hidden states available."
                    )
                emb_result: np.ndarray = np.asarray(emb.cpu().numpy())
                return emb_result

            if layer == "all":
                stacked = torch.cat(hidden_states, dim=-1)
                return stacked.cpu().numpy()

            resolved = int(layer)
            if resolved >= len(hidden_states):
                resolved = len(hidden_states) - 1
            return hidden_states[resolved].cpu().numpy()

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
        import torch

        self._check_loaded()

        resolved_layer = self._resolve_layer(layer, self._n_layers)
        gene_names = list(adata.var_names)
        matched_names, matched_ids = self._map_genes(gene_names)
        self.gene_names = matched_names

        # Build column index map for fast expression slicing.
        name_to_col: dict[str, int] = {
            g: i for i, g in enumerate(gene_names)
        }
        col_indices = [name_to_col[g] for g in matched_names]

        # Expression matrix (dense).
        if hasattr(adata.X, "toarray"):
            X = adata.X.toarray()
        else:
            X = np.asarray(adata.X)

        X_matched = X[:, col_indices]  # (n_cells, n_matched_genes)

        n_cells = X_matched.shape[0]
        if n_cells == 0:
            raise ValueError("Cannot extract scGPT gene embeddings from an AnnData object with 0 cells.")
        n_genes = len(matched_ids)
        embedding_accum = np.zeros(
            (n_genes, self.embedding_dim if resolved_layer != "all" else self.embedding_dim * self._n_layers),
            dtype=np.float64,
        )

        gene_ids_tensor = torch.tensor(matched_ids, dtype=torch.long).to(
            self._device
        )

        for start in range(0, n_cells, self.batch_size):
            end = min(start + self.batch_size, n_cells)
            batch_expr = X_matched[start:end]  # (bs, n_genes)
            bs = batch_expr.shape[0]

            gene_ids_batch = gene_ids_tensor.unsqueeze(0).expand(bs, -1)
            values_batch = (
                torch.tensor(batch_expr, dtype=torch.float32)
                .to(self._device)
            )

            hidden = self._extract_layer_output(
                gene_ids_batch, values_batch, resolved_layer
            )  # (bs, n_genes, dim)
            embedding_accum += hidden.sum(axis=0)

        embedding_accum /= n_cells
        self.embedding_dim = embedding_accum.shape[1]
        logger.info(
            "Extracted scGPT gene embeddings: shape %s", embedding_accum.shape
        )
        return embedding_accum.astype(np.float32)

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
            Dictionary mapping cell-type labels to embedding arrays of shape
            ``(n_matched_genes, embedding_dim)``.
        """
        import torch

        self._check_loaded()

        if cell_type_key not in adata.obs.columns:
            raise KeyError(
                f"Cell type key '{cell_type_key}' not found in adata.obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )

        resolved_layer = self._resolve_layer(layer, self._n_layers)
        gene_names = list(adata.var_names)
        matched_names, matched_ids = self._map_genes(gene_names)
        self.gene_names = matched_names

        name_to_col: dict[str, int] = {
            g: i for i, g in enumerate(gene_names)
        }
        col_indices = [name_to_col[g] for g in matched_names]

        if hasattr(adata.X, "toarray"):
            X = adata.X.toarray()
        else:
            X = np.asarray(adata.X)
        X_matched = X[:, col_indices]

        gene_ids_tensor = torch.tensor(matched_ids, dtype=torch.long).to(
            self._device
        )
        n_genes = len(matched_ids)

        cell_types = adata.obs[cell_type_key]
        unique_types = sorted(cell_types.unique())

        results: dict[str, np.ndarray] = {}
        for ct in unique_types:
            mask = (cell_types == ct).values
            ct_expr = X_matched[mask]
            n_ct = ct_expr.shape[0]

            if n_ct == 0:
                continue

            dim = (
                self.embedding_dim
                if resolved_layer != "all"
                else self.embedding_dim * self._n_layers
            )
            accum = np.zeros((n_genes, dim), dtype=np.float64)

            for start in range(0, n_ct, self.batch_size):
                end = min(start + self.batch_size, n_ct)
                batch_expr = ct_expr[start:end]
                bs = batch_expr.shape[0]

                gene_ids_batch = gene_ids_tensor.unsqueeze(0).expand(bs, -1)
                values_batch = (
                    torch.tensor(batch_expr, dtype=torch.float32)
                    .to(self._device)
                )

                hidden = self._extract_layer_output(
                    gene_ids_batch, values_batch, resolved_layer
                )
                accum += hidden.sum(axis=0)

            accum /= n_ct
            results[str(ct)] = accum.astype(np.float32)
            logger.info(
                "Extracted context embeddings for '%s' (%d cells)", ct, n_ct
            )

        return results
