"""Geneformer embedding extractor for gene representations.

This module provides a concrete implementation of
:class:`~npathway.embedding.base.BaseEmbeddingExtractor` that loads a
pre-trained Geneformer model from HuggingFace and extracts gene token
embeddings from specified transformer layers.

Reference:
    Theodoris, C.V. et al. "Transfer learning enables predictions in
    network biology." Nature (2023).
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

_DEFAULT_BATCH_SIZE: int = 32
_GENEFORMER_HF_DEFAULT: str = "ctheodoris/Geneformer"


class GeneformerEmbeddingExtractor(BaseEmbeddingExtractor):
    """Extract gene embeddings from a pre-trained Geneformer model.

    Geneformer represents cells as rank-value encoded sequences of gene tokens
    ordered by expression level.  This extractor processes single-cell data
    through the model and collects per-gene hidden-state representations from
    the specified transformer layer.

    Attributes:
        token_dict: Mapping from Ensembl gene id to Geneformer token id.
        gene_name_to_ensembl: Mapping from gene symbol to Ensembl id.
        batch_size: Number of cells processed per forward-pass batch.
        max_input_size: Maximum number of gene tokens per cell.
    """

    def __init__(
        self,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        max_input_size: int = 2048,
        device: str | None = None,
    ) -> None:
        """Initialise the Geneformer embedding extractor.

        Args:
            batch_size: Number of cells per batch.
            max_input_size: Maximum number of gene tokens per cell sequence.
            device: Computation device string or ``None`` for auto-detection.
        """
        super().__init__(model_name="Geneformer")
        self.token_dict: dict[str, int] = {}
        self.gene_name_to_ensembl: dict[str, str] = {}
        self.batch_size: int = batch_size
        self.max_input_size: int = max_input_size
        self._n_layers: int = 0
        self._tokenizer: Any = None
        self._set_device(device)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self, model_path: str) -> None:
        """Load a pre-trained Geneformer model.

        Args:
            model_path: Local directory path or HuggingFace model identifier
                (e.g. ``"ctheodoris/Geneformer"``).

        Raises:
            ImportError: If ``transformers`` is not installed.
            FileNotFoundError: If a local model path does not exist.
            RuntimeError: If model loading fails.
        """
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for Geneformer embedding extraction. "
                "Install it with: pip install torch"
            ) from exc

        try:
            from transformers import BertConfig, BertForMaskedLM, PreTrainedConfig  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "The `transformers` library is required for Geneformer. "
                "Install it with: pip install transformers"
            ) from exc

        local_path = Path(model_path)
        is_local = local_path.exists() and local_path.is_dir()

        if not is_local and not model_path.startswith(("ctheodoris/", "geneformer/")):
            raise FileNotFoundError(
                f"Model path '{model_path}' is not a valid local directory "
                "and does not look like a HuggingFace model id."
            )

        # -- Load the token dictionary --------------------------------------
        self._load_token_dictionary(model_path if is_local else None)

        # -- Load model -----------------------------------------------------
        try:
            model = BertForMaskedLM.from_pretrained(
                model_path,
                output_hidden_states=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Geneformer model from '{model_path}': {exc}"
            ) from exc

        config: PreTrainedConfig = model.config
        self._n_layers = config.num_hidden_layers
        self.embedding_dim = config.hidden_size

        model.to(self._device)  # type: ignore[arg-type]
        model.eval()

        self.model = model
        self.is_loaded = True
        logger.info(
            "Geneformer model loaded (%d layers, dim=%d, vocab=%d)",
            self._n_layers,
            self.embedding_dim,
            config.vocab_size,
        )

    def _load_token_dictionary(self, local_dir: str | None) -> None:
        """Load Geneformer's Ensembl-to-token mapping.

        The token dictionary maps Ensembl gene ids to integer token ids used
        by the model vocabulary.  This method also attempts to build a reverse
        mapping from common gene symbols to Ensembl ids.

        Args:
            local_dir: Optional local directory that may contain
                ``token_dictionary.pkl``.
        """
        import pickle

        token_dict_loaded = False

        # Try loading from the local model directory.
        if local_dir is not None:
            for candidate in ["token_dictionary.pkl", "geneformer/token_dictionary.pkl"]:
                pkl_path = Path(local_dir) / candidate
                if pkl_path.exists():
                    with open(pkl_path, "rb") as fh:
                        self.token_dict = pickle.load(fh)
                    token_dict_loaded = True
                    logger.info(
                        "Loaded token dictionary from %s (%d entries)",
                        pkl_path,
                        len(self.token_dict),
                    )
                    break

        # Try the geneformer package.
        if not token_dict_loaded:
            try:
                from geneformer import TranscriptomeTokenizer

                tokenizer = TranscriptomeTokenizer()
                if hasattr(tokenizer, "gene_token_dict"):
                    self.token_dict = tokenizer.gene_token_dict
                    token_dict_loaded = True
                elif hasattr(tokenizer, "token_dictionary"):
                    self.token_dict = tokenizer.token_dictionary
                    token_dict_loaded = True
            except ImportError:
                pass
            except Exception:
                pass

        if not token_dict_loaded:
            logger.warning(
                "Could not load Geneformer token dictionary. "
                "Gene-to-token mapping will rely on adata.var metadata."
            )

        # Build a gene-symbol to Ensembl mapping if possible. This is a best
        # effort -- users may need to provide their own mapping.
        try:
            import json

            if local_dir is not None:
                for name in [
                    "gene_name_id_dict.pkl",
                    "geneformer/gene_name_id_dict.pkl",
                    "ensembl_mapping.json",
                ]:
                    p = Path(local_dir) / name
                    if p.exists():
                        if p.suffix == ".pkl":
                            with open(p, "rb") as fh:
                                self.gene_name_to_ensembl = pickle.load(fh)
                        else:
                            with open(p, "r", encoding="utf-8") as fh:
                                self.gene_name_to_ensembl = json.load(fh)
                        logger.info(
                            "Loaded gene-symbol-to-Ensembl mapping (%d entries)",
                            len(self.gene_name_to_ensembl),
                        )
                        break
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Rank-value encoding
    # ------------------------------------------------------------------

    def _rank_value_encode(
        self,
        expression: np.ndarray,
        gene_token_ids: list[int],
    ) -> list[int]:
        """Rank-value encode a single cell's expression into a token sequence.

        Genes are sorted by expression level in descending order and the
        corresponding token ids form the input sequence, truncated to
        ``max_input_size``.

        Args:
            expression: 1-D expression vector (only matched genes).
            gene_token_ids: Token ids corresponding to the matched genes.

        Returns:
            List of token ids ordered by descending expression.
        """
        # Argsort descending.
        order = np.argsort(-expression)
        # Filter out zero-expression genes.
        nonzero = [i for i in order if expression[i] > 0]
        tokens = [gene_token_ids[i] for i in nonzero[: self.max_input_size]]
        return tokens

    # ------------------------------------------------------------------
    # Embedding extraction helpers
    # ------------------------------------------------------------------

    def _map_genes(
        self,
        gene_names: list[str],
        adata: AnnData | None = None,
    ) -> tuple[list[str], list[int], list[int]]:
        """Map dataset gene names to Geneformer token ids.

        Args:
            gene_names: Gene names from the AnnData object.
            adata: Optional AnnData for extracting Ensembl ids from ``adata.var``.

        Returns:
            Tuple of (matched_gene_names, token_ids, column_indices) where
            column_indices point into the original gene_names list.

        Raises:
            ValueError: If no genes can be mapped.
        """
        # Try to get Ensembl ids from adata.var if available.
        ensembl_ids: dict[str, str] = {}
        if adata is not None:
            for col in ["ensembl_id", "gene_ids", "ensembl", "gene_id"]:
                if col in adata.var.columns:
                    ensembl_ids = dict(
                        zip(adata.var_names, adata.var[col].astype(str))
                    )
                    break

        matched_names: list[str] = []
        matched_tokens: list[int] = []
        matched_cols: list[int] = []

        for idx, gene in enumerate(gene_names):
            token_id: int | None = None

            # Strategy 1: gene name is directly an Ensembl id.
            if gene in self.token_dict:
                token_id = self.token_dict[gene]
            # Strategy 2: lookup via the symbol-to-Ensembl mapping.
            elif gene in self.gene_name_to_ensembl:
                ens = self.gene_name_to_ensembl[gene]
                if ens in self.token_dict:
                    token_id = self.token_dict[ens]
            # Strategy 3: lookup via adata.var Ensembl column.
            elif gene in ensembl_ids:
                ens = ensembl_ids[gene]
                if ens in self.token_dict:
                    token_id = self.token_dict[ens]
            # Strategy 4: uppercase fallback.
            elif gene.upper() in self.token_dict:
                token_id = self.token_dict[gene.upper()]

            if token_id is not None:
                matched_names.append(gene)
                matched_tokens.append(token_id)
                matched_cols.append(idx)

        if not matched_names:
            raise ValueError(
                "No overlapping genes found between the dataset and the "
                "Geneformer vocabulary. Ensure gene identifiers are Ensembl "
                "ids or provide a gene_name_to_ensembl mapping."
            )

        logger.info(
            "Mapped %d / %d dataset genes to Geneformer vocabulary",
            len(matched_names),
            len(gene_names),
        )
        return matched_names, matched_tokens, matched_cols

    def _extract_hidden_states(
        self,
        input_ids_batch: Any,
        attention_mask_batch: Any,
        layer: int | str,
    ) -> Any:
        """Forward pass and return hidden states from the specified layer.

        Args:
            input_ids_batch: Tensor of shape ``(batch, seq_len)``.
            attention_mask_batch: Tensor of shape ``(batch, seq_len)``.
            layer: Resolved layer index or ``"all"``.

        Returns:
            Tensor of hidden states.
        """
        import torch

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
            )

        # outputs.hidden_states is a tuple of (n_layers + 1) tensors
        # (first is the embedding layer output).
        hidden_states = outputs.hidden_states  # tuple of (batch, seq, dim)

        if layer == "all":
            # Skip the initial embedding layer (index 0).
            stacked = torch.cat(list(hidden_states[1:]), dim=-1)
            return stacked
        else:
            # +1 to skip embedding layer at index 0.
            idx = int(layer) + 1
            if idx >= len(hidden_states):
                idx = len(hidden_states) - 1
            return hidden_states[idx]

    def _collect_gene_embeddings_from_cells(
        self,
        X: np.ndarray,
        matched_tokens: list[int],
        matched_cols: list[int],
        n_genes: int,
        resolved_layer: int | str,
    ) -> np.ndarray:
        """Process cells and accumulate per-gene embeddings.

        For each cell, the expression of matched genes is rank-value encoded.
        After the forward pass, the hidden-state vector for each gene token is
        accumulated into a per-gene embedding buffer.

        Args:
            X: Dense expression matrix of shape ``(n_cells, n_total_genes)``.
            matched_tokens: Token ids for matched genes.
            matched_cols: Column indices into X for matched genes.
            n_genes: Number of matched genes.
            resolved_layer: Layer to extract from.

        Returns:
            Averaged gene embedding array of shape ``(n_genes, dim)``.
        """
        import torch

        dim = (
            self.embedding_dim
            if resolved_layer != "all"
            else self.embedding_dim * self._n_layers
        )
        gene_embedding_sum = np.zeros((n_genes, dim), dtype=np.float64)
        gene_counts = np.zeros(n_genes, dtype=np.int64)

        # Build a token-id to matched-gene-index map.
        token_to_gene_idx: dict[int, int] = {
            tid: gi for gi, tid in enumerate(matched_tokens)
        }

        n_cells = X.shape[0]
        for start in range(0, n_cells, self.batch_size):
            end = min(start + self.batch_size, n_cells)
            batch_expr = X[start:end][:, matched_cols]  # (bs, n_genes)
            bs = batch_expr.shape[0]

            # Rank-value encode each cell.
            encoded_sequences: list[list[int]] = []
            for i in range(bs):
                tokens = self._rank_value_encode(
                    batch_expr[i], matched_tokens
                )
                if not tokens:
                    # If all genes are zero, use a dummy token.
                    tokens = [matched_tokens[0]]
                encoded_sequences.append(tokens)

            # Pad to the same length.
            max_len = max(len(s) for s in encoded_sequences)
            padded = np.zeros((bs, max_len), dtype=np.int64)
            attention = np.zeros((bs, max_len), dtype=np.int64)
            for i, seq in enumerate(encoded_sequences):
                padded[i, : len(seq)] = seq
                attention[i, : len(seq)] = 1

            input_ids = torch.tensor(padded, dtype=torch.long).to(
                self._device
            )
            attn_mask = torch.tensor(attention, dtype=torch.long).to(
                self._device
            )

            hidden = self._extract_hidden_states(
                input_ids, attn_mask, resolved_layer
            )  # (bs, max_len, dim)
            hidden_np = hidden.cpu().numpy()

            # Scatter hidden states back to gene indices.
            for i in range(bs):
                seq = encoded_sequences[i]
                for pos, tid in enumerate(seq):
                    gene_idx = token_to_gene_idx.get(tid)
                    if gene_idx is not None:
                        gene_embedding_sum[gene_idx] += hidden_np[i, pos]
                        gene_counts[gene_idx] += 1

        # Average, avoiding division by zero.
        nonzero_mask = gene_counts > 0
        gene_embedding_sum[nonzero_mask] /= gene_counts[nonzero_mask, None]

        return gene_embedding_sum.astype(np.float32)

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
        matched_names, matched_tokens, matched_cols = self._map_genes(
            gene_names, adata
        )
        self.gene_names = matched_names

        if hasattr(adata.X, "toarray"):
            X = adata.X.toarray()
        else:
            X = np.asarray(adata.X)

        embeddings = self._collect_gene_embeddings_from_cells(
            X, matched_tokens, matched_cols, len(matched_names), resolved_layer
        )
        self.embedding_dim = embeddings.shape[1]
        logger.info(
            "Extracted Geneformer gene embeddings: shape %s", embeddings.shape
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
        matched_names, matched_tokens, matched_cols = self._map_genes(
            gene_names, adata
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

            emb = self._collect_gene_embeddings_from_cells(
                ct_X,
                matched_tokens,
                matched_cols,
                len(matched_names),
                resolved_layer,
            )
            results[str(ct)] = emb
            logger.info(
                "Extracted Geneformer context embeddings for '%s' (%d cells)",
                ct,
                ct_X.shape[0],
            )

        return results
