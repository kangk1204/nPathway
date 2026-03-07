"""Gene embedding extraction from single-cell foundation models.

This sub-package provides a unified interface for extracting gene
representations from pre-trained foundation models such as scGPT,
Geneformer, and scBERT.

Quick start::

    from npathway.embedding import get_extractor

    extractor = get_extractor("scgpt", device="cuda")
    extractor.load_model("/path/to/scgpt_model")
    embeddings = extractor.extract_gene_embeddings(adata, layer=-1)
"""

from npathway.embedding.base import BaseEmbeddingExtractor
from npathway.embedding.extract_geneformer import GeneformerEmbeddingExtractor
from npathway.embedding.extract_scbert import ScBERTEmbeddingExtractor
from npathway.embedding.extract_scgpt import ScGPTEmbeddingExtractor
from npathway.embedding.factory import get_extractor, list_available_models

__all__ = [
    "BaseEmbeddingExtractor",
    "GeneformerEmbeddingExtractor",
    "ScBERTEmbeddingExtractor",
    "ScGPTEmbeddingExtractor",
    "get_extractor",
    "list_available_models",
]
