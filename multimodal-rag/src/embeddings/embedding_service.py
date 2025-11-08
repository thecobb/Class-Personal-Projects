"""Embedding service for generating text embeddings."""

from typing import List, Optional

from langchain_openai import OpenAIEmbeddings
from loguru import logger

from ..config import settings


class EmbeddingService:
    """Service for generating embeddings from text."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize embedding service.

        Args:
            model: Embedding model name
            api_key: OpenAI API key
        """
        self.model = model or settings.text_embedding_model
        self.api_key = api_key or settings.openai_api_key

        if not self.api_key:
            logger.warning("No OpenAI API key provided. Embeddings will fail.")

        self.embeddings = OpenAIEmbeddings(
            model=self.model,
            openai_api_key=self.api_key,
        )

        logger.info(f"Initialized EmbeddingService with model: {self.model}")

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.embeddings.embed_documents(texts)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    @property
    def dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        # Common OpenAI embedding dimensions
        dimension_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dimension_map.get(self.model, 1536)
