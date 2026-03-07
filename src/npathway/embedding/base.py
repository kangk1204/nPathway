"""Abstract base class for gene embedding extraction from foundation models."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

try:
    from anndata import AnnData
except ImportError:
    AnnData = Any  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)


class BaseEmbeddingExtractor(ABC):
    """Base class for gene embedding extraction from single-cell foundation models.

    This abstract class defines the interface that all embedding extractors must
    implement. It provides shared utilities for saving/loading embeddings and
    managing gene name mappings.

    Attributes:
        model: The loaded foundation model instance.
        gene_names: List of gene names corresponding to embedding rows.
        embedding_dim: Dimensionality of the extracted embeddings.
        model_name: Human-readable name of the foundation model.
        is_loaded: Whether the model has been successfully loaded.
    """

    def __init__(self, model_name: str = "base") -> None:
        """Initialize the base embedding extractor.

        Args:
            model_name: Human-readable name identifying the foundation model.
        """
        self.model: Any = None
        self.gene_names: list[str] = []
        self.embedding_dim: int = 0
        self.model_name: str = model_name
        self.is_loaded: bool = False
        self._device: str = "cpu"

    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """Load a pre-trained foundation model from disk or a remote source.

        Args:
            model_path: Path to the pre-trained model directory or checkpoint,
                or a HuggingFace model identifier.

        Raises:
            FileNotFoundError: If the model path does not exist.
            ImportError: If required dependencies are not installed.
            RuntimeError: If the model fails to load.
        """
        ...

    @abstractmethod
    def extract_gene_embeddings(
        self,
        adata: AnnData,
        layer: int | str = -1,
    ) -> np.ndarray:
        """Extract gene embeddings from the foundation model.

        Extracts universal gene embeddings by processing all cells in the dataset
        and averaging token representations across cells.

        Args:
            adata: An AnnData object containing gene expression data. Gene names
                should be stored in ``adata.var_names`` or ``adata.var['gene_name']``.
            layer: Transformer layer from which to extract embeddings. Use an
                integer for a specific layer index (negative indexing supported)
                or ``"all"`` to concatenate all layers.

        Returns:
            A numpy array of shape ``(n_genes, embedding_dim)`` where ``n_genes``
            is the number of genes found in both the dataset and the model
            vocabulary.

        Raises:
            RuntimeError: If the model has not been loaded.
            ValueError: If no overlapping genes are found between the dataset
                and model vocabulary.
        """
        ...

    @abstractmethod
    def extract_context_embeddings(
        self,
        adata: AnnData,
        cell_type_key: str,
        layer: int | str = -1,
    ) -> dict[str, np.ndarray]:
        """Extract context-specific gene embeddings grouped by cell type.

        Processes cells in the dataset grouped by their cell type annotation
        and extracts separate gene embeddings for each cell type by averaging
        token representations within each group.

        Args:
            adata: An AnnData object containing gene expression data with cell
                type annotations.
            cell_type_key: Column name in ``adata.obs`` containing cell type
                labels.
            layer: Transformer layer from which to extract embeddings. Use an
                integer for a specific layer index (negative indexing supported)
                or ``"all"`` to concatenate all layers.

        Returns:
            A dictionary mapping cell type names to numpy arrays of shape
            ``(n_genes, embedding_dim)``.

        Raises:
            RuntimeError: If the model has not been loaded.
            KeyError: If ``cell_type_key`` is not found in ``adata.obs``.
            ValueError: If no overlapping genes are found.
        """
        ...

    def get_gene_names(self) -> list[str]:
        """Return the list of gene names corresponding to the extracted embeddings.

        Returns:
            A list of gene name strings in the same order as embedding rows.

        Raises:
            RuntimeError: If no embeddings have been extracted yet.
        """
        if not self.gene_names:
            raise RuntimeError(
                "No gene names available. Extract embeddings first using "
                "extract_gene_embeddings() or extract_context_embeddings()."
            )
        return list(self.gene_names)

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        gene_names: list[str],
        output_path: str,
    ) -> None:
        """Save extracted embeddings and gene names to disk.

        Creates a compressed ``.npz`` file containing the embedding matrix and
        a companion ``.json`` file with metadata (gene names, dimensions, model).

        Args:
            embeddings: Numpy array of shape ``(n_genes, embedding_dim)``.
            gene_names: List of gene names matching embedding rows.
            output_path: Base path for the output files. The ``.npz`` and
                ``.json`` extensions are appended automatically.

        Raises:
            ValueError: If the number of gene names does not match the number
                of embedding rows.
        """
        if embeddings.shape[0] != len(gene_names):
            raise ValueError(
                f"Number of gene names ({len(gene_names)}) does not match "
                f"number of embedding rows ({embeddings.shape[0]})."
            )

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        npz_path = output.with_suffix(".npz")
        np.savez_compressed(str(npz_path), embeddings=embeddings)
        logger.info("Saved embeddings to %s", npz_path)

        metadata = {
            "gene_names": gene_names,
            "embedding_dim": int(embeddings.shape[1]),
            "n_genes": int(embeddings.shape[0]),
            "model_name": self.model_name,
            "dtype": str(embeddings.dtype),
        }
        json_path = output.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)
        logger.info("Saved metadata to %s", json_path)

    def load_embeddings(self, input_path: str) -> tuple[np.ndarray, list[str]]:
        """Load previously saved embeddings and gene names from disk.

        Args:
            input_path: Base path to the saved files (without extension), or
                a path ending in ``.npz``.

        Returns:
            A tuple of ``(embeddings, gene_names)`` where embeddings is a numpy
            array and gene_names is the corresponding list of gene name strings.

        Raises:
            FileNotFoundError: If the embedding or metadata files are missing.
        """
        base = Path(input_path)
        if base.suffix == ".npz":
            base = base.with_suffix("")

        npz_path = base.with_suffix(".npz")
        json_path = base.with_suffix(".json")

        if not npz_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {npz_path}")
        if not json_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {json_path}")

        data = np.load(str(npz_path))
        embeddings = data["embeddings"]

        with open(json_path, "r", encoding="utf-8") as fh:
            metadata = json.load(fh)

        gene_names: list[str] = metadata["gene_names"]

        if embeddings.shape[0] != len(gene_names):
            raise ValueError(
                f"Mismatch between embedding rows ({embeddings.shape[0]}) and "
                f"gene names ({len(gene_names)}) in loaded files."
            )

        logger.info(
            "Loaded %d gene embeddings of dimension %d from %s",
            len(gene_names),
            embeddings.shape[1],
            npz_path,
        )
        return embeddings, gene_names

    def _check_loaded(self) -> None:
        """Verify that the model has been loaded before extraction.

        Raises:
            RuntimeError: If the model is not loaded.
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError(
                f"{self.model_name} model is not loaded. "
                "Call load_model() before extracting embeddings."
            )

    def _resolve_layer(self, layer: int | str, n_layers: int) -> int | str:
        """Resolve a layer specification to a concrete index.

        Args:
            layer: Layer index (int, supports negative indexing) or ``"all"``.
            n_layers: Total number of transformer layers in the model.

        Returns:
            Resolved non-negative layer index, or the string ``"all"``.

        Raises:
            ValueError: If the layer index is out of range or unrecognised.
        """
        if isinstance(layer, str):
            if layer.lower() == "all":
                return "all"
            raise ValueError(
                f"Unrecognised layer specification '{layer}'. "
                "Use an integer or 'all'."
            )
        if layer < 0:
            layer = n_layers + layer
        if layer < 0 or layer >= n_layers:
            raise ValueError(
                f"Layer index {layer} out of range for model with "
                f"{n_layers} layers."
            )
        return layer

    def _set_device(self, device: str | None = None) -> str:
        """Determine and set the computation device.

        Args:
            device: Explicit device string (e.g., ``"cpu"``, ``"cuda:0"``).
                If ``None``, automatically selects CUDA when available.

        Returns:
            The device string that was set.
        """
        if device is not None:
            self._device = device
            return device

        try:
            import torch

            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        except ImportError:
            self._device = "cpu"

        logger.info("Using device: %s", self._device)
        return self._device
