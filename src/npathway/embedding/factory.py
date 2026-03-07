"""Factory for creating embedding extractor instances by model name.

This module provides a single entry point for obtaining the correct
:class:`~npathway.embedding.base.BaseEmbeddingExtractor` subclass given a
model name string.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from npathway.embedding.base import BaseEmbeddingExtractor


# Registry of supported model names mapped to their class import paths.
# We use lazy imports so that heavyweight dependencies (torch, transformers,
# scgpt, etc.) are only loaded when the corresponding extractor is requested.
_REGISTRY: dict[str, tuple[str, str]] = {
    "scgpt": (
        "npathway.embedding.extract_scgpt",
        "ScGPTEmbeddingExtractor",
    ),
    "geneformer": (
        "npathway.embedding.extract_geneformer",
        "GeneformerEmbeddingExtractor",
    ),
    "scbert": (
        "npathway.embedding.extract_scbert",
        "ScBERTEmbeddingExtractor",
    ),
}


def get_extractor(model_name: str, **kwargs: Any) -> BaseEmbeddingExtractor:
    """Create an embedding extractor instance for the given model name.

    This factory supports lazy imports so that only the dependencies required
    by the selected model are loaded.

    Args:
        model_name: Name of the foundation model. Supported values are
            ``"scgpt"``, ``"geneformer"``, and ``"scbert"`` (case-insensitive).
        **kwargs: Additional keyword arguments forwarded to the extractor
            constructor (e.g. ``batch_size``, ``device``).

    Returns:
        An instance of the corresponding
        :class:`~npathway.embedding.base.BaseEmbeddingExtractor` subclass.

    Raises:
        ValueError: If ``model_name`` is not recognised.
        ImportError: If the required dependencies for the chosen model are
            not installed.

    Examples:
        >>> extractor = get_extractor("scgpt", batch_size=32, device="cuda")
        >>> extractor.load_model("/path/to/scgpt_checkpoint")
    """
    key = model_name.strip().lower()
    if key not in _REGISTRY:
        supported = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model name '{model_name}'. "
            f"Supported models: {supported}"
        )

    module_path, class_name = _REGISTRY[key]

    import importlib

    module = importlib.import_module(module_path)
    extractor_cls = getattr(module, class_name)
    extractor: BaseEmbeddingExtractor = extractor_cls(**kwargs)
    return extractor


def list_available_models() -> list[str]:
    """Return the list of supported model names.

    Returns:
        Sorted list of model name strings that can be passed to
        :func:`get_extractor`.
    """
    return sorted(_REGISTRY.keys())
