"""Reranking module for improving retrieval quality."""

from typing import Any, List, Optional, Tuple

import cohere
from loguru import logger

from ..config import settings
from ..document_processing.chunking import Chunk


class Reranker:
    """
    Reranker for improving retrieval quality.

    Following Jason Liu's methodology: reranking provides 20%+ accuracy improvements
    and is used in every RAG product he builds. Cohere's rerank models are recommended.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        top_n: Optional[int] = None,
    ):
        """
        Initialize reranker.

        Args:
            model: Rerank model name
            api_key: Cohere API key
            top_n: Number of top results to return after reranking
        """
        self.model = model or settings.rerank_model
        self.api_key = api_key or settings.cohere_api_key
        self.top_n = top_n or settings.rerank_top_k

        if not self.api_key:
            logger.warning("No Cohere API key provided. Reranking will be disabled.")
            self.client = None
        else:
            self.client = cohere.Client(api_key=self.api_key)

        logger.info(f"Initialized Reranker with model: {self.model}")

    def rerank(
        self,
        query: str,
        results: List[Tuple[Any, float]],
        top_n: Optional[int] = None,
    ) -> List[Tuple[Any, float]]:
        """
        Rerank search results.

        Args:
            query: Original query
            results: List of (content, score) tuples from initial retrieval
            top_n: Number of top results to return (overrides default)

        Returns:
            Reranked list of (content, score) tuples
        """
        if not self.client:
            logger.warning("Reranking disabled, returning original results")
            return results

        if not results:
            return results

        top_n = top_n or self.top_n

        # Extract text from results
        documents = []
        original_content = []

        for content, score in results:
            # Handle different content types
            if isinstance(content, Chunk):
                documents.append(content.text)
                original_content.append(content)
            elif isinstance(content, str):
                documents.append(content)
                original_content.append(content)
            elif hasattr(content, "text"):
                documents.append(content.text)
                original_content.append(content)
            else:
                # For images or other non-text content, skip reranking
                logger.warning(f"Cannot rerank non-text content: {type(content)}")
                documents.append("")
                original_content.append(content)

        try:
            # Call Cohere rerank API
            logger.debug(f"Reranking {len(documents)} documents with query: {query[:100]}...")

            rerank_response = self.client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=min(top_n, len(documents)),
                return_documents=False,  # We already have the documents
            )

            # Build reranked results
            reranked_results = []
            for result in rerank_response.results:
                idx = result.index
                relevance_score = result.relevance_score

                reranked_results.append((original_content[idx], relevance_score))

            logger.info(f"Reranked {len(documents)} results to top {len(reranked_results)}")

            return reranked_results

        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # Fall back to original results
            return results[:top_n]

    def rerank_chunks(
        self,
        query: str,
        chunks_with_scores: List[Tuple[Chunk, float]],
        top_n: Optional[int] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Convenience method for reranking chunks specifically.

        Args:
            query: Original query
            chunks_with_scores: List of (Chunk, score) tuples
            top_n: Number of top results to return

        Returns:
            Reranked list of (Chunk, score) tuples
        """
        return self.rerank(query, chunks_with_scores, top_n=top_n)


class SimpleReranker:
    """
    Simple reranker that doesn't require API calls.
    Uses basic text matching heuristics as fallback.
    """

    def __init__(self, top_n: int = 5):
        """
        Initialize simple reranker.

        Args:
            top_n: Number of top results to return
        """
        self.top_n = top_n

    def rerank(
        self,
        query: str,
        results: List[Tuple[Any, float]],
        top_n: Optional[int] = None,
    ) -> List[Tuple[Any, float]]:
        """
        Rerank using simple heuristics.

        Args:
            query: Original query
            results: List of (content, score) tuples
            top_n: Number of top results to return

        Returns:
            Reranked list of (content, score) tuples
        """
        top_n = top_n or self.top_n

        # Extract query terms
        query_terms = set(query.lower().split())

        # Score results based on term overlap
        scored_results = []
        for content, original_score in results:
            # Get text content
            if isinstance(content, Chunk):
                text = content.text.lower()
            elif isinstance(content, str):
                text = content.lower()
            elif hasattr(content, "text"):
                text = content.text.lower()
            else:
                scored_results.append((content, original_score))
                continue

            # Calculate term overlap
            text_terms = set(text.split())
            overlap = len(query_terms.intersection(text_terms))

            # Combine with original score
            # Weight: 70% original score, 30% term overlap
            combined_score = 0.7 * original_score + 0.3 * (overlap / len(query_terms))

            scored_results.append((content, combined_score))

        # Sort by combined score
        scored_results.sort(key=lambda x: x[1], reverse=True)

        return scored_results[:top_n]
