"""Multimodal RAG System.

A production-grade RAG system following Jason Liu's methodology.
"""

__version__ = "0.1.0"

from . import config, document_processing, embeddings, evaluation, generation, retrieval

__all__ = [
    "config",
    "document_processing",
    "embeddings",
    "retrieval",
    "generation",
    "evaluation",
]
