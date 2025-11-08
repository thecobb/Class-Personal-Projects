"""Embedding module for text and multimodal content."""

from .embedding_service import EmbeddingService
from .image_describer import ImageDescriber
from .multimodal_embeddings import CLIPEmbeddings, MultimodalEmbeddings

__all__ = [
    "EmbeddingService",
    "ImageDescriber",
    "MultimodalEmbeddings",
    "CLIPEmbeddings",
]
