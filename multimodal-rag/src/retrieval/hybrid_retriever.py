"""Hybrid retriever combining BM25 and vector search."""

from typing import Any, Dict, List, Optional, Tuple

from langchain.retrievers import BM25Retriever, EnsembleRetriever
from loguru import logger

from ..config import settings
from ..document_processing.chunking import Chunk
from .vector_store import VectorStore


class HybridRetriever:
    """
    Hybrid retriever combining dense (vector) and sparse (BM25) search.

    Following Jason Liu's methodology: hybrid search is non-negotiable for production,
    achieving 15-30% accuracy improvements over pure vector search.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        chunks: Optional[List[Chunk]] = None,
        alpha: Optional[float] = None,
    ):
        """
        Initialize hybrid retriever.

        Args:
            vector_store: Vector store for dense retrieval
            chunks: List of chunks for BM25 indexing
            alpha: Weight for semantic search (0-1). Default from settings.
                  alpha=1.0 means pure vector search
                  alpha=0.0 means pure BM25
                  alpha=0.6 means 60% vector, 40% BM25
        """
        self.vector_store = vector_store
        self.alpha = alpha if alpha is not None else settings.hybrid_search_alpha

        # Initialize BM25 retriever
        self.bm25_retriever: Optional[BM25Retriever] = None
        if chunks:
            self._initialize_bm25(chunks)

        logger.info(f"Initialized HybridRetriever with alpha={self.alpha}")

    def _initialize_bm25(self, chunks: List[Chunk]) -> None:
        """
        Initialize BM25 retriever with chunks.

        Args:
            chunks: List of chunks to index
        """
        logger.info(f"Initializing BM25 retriever with {len(chunks)} chunks")

        # Convert chunks to LangChain Document format
        from langchain.schema import Document

        documents = [
            Document(
                page_content=chunk.text,
                metadata={**chunk.metadata, "chunk_id": chunk.chunk_id},
            )
            for chunk in chunks
        ]

        self.bm25_retriever = BM25Retriever.from_documents(documents)
        logger.info("BM25 retriever initialized")

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Add chunks to both vector store and BM25 index.

        Args:
            chunks: List of chunks to add
        """
        # Add to vector store
        self.vector_store.add_chunks(chunks)

        # Re-initialize BM25 with all chunks
        # Note: This is inefficient for incremental updates
        # In production, consider using a more sophisticated BM25 implementation
        if self.bm25_retriever is None:
            self._initialize_bm25(chunks)
        else:
            # For now, we'll need to get all chunks and re-index
            # This is a limitation of the current BM25Retriever implementation
            logger.warning("BM25 re-indexing not implemented for incremental updates")

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Retrieve chunks using hybrid search.

        Args:
            query: Query text
            k: Number of results to return
            filter: Metadata filter for vector search

        Returns:
            List of (Chunk, score) tuples sorted by relevance
        """
        # If BM25 not initialized, fall back to pure vector search
        if self.bm25_retriever is None:
            logger.warning("BM25 not initialized, using pure vector search")
            return self.vector_store.similarity_search(query, k=k, filter=filter)

        # Get results from vector search
        vector_results = self.vector_store.similarity_search(query, k=k * 2, filter=filter)

        # Get results from BM25
        self.bm25_retriever.k = k * 2
        bm25_docs = self.bm25_retriever.get_relevant_documents(query)

        # Convert BM25 results to Chunk objects
        bm25_results = []
        for doc in bm25_docs:
            chunk = Chunk(
                text=doc.page_content,
                chunk_id=doc.metadata.get("chunk_id", ""),
                metadata=doc.metadata,
            )
            # BM25 doesn't provide scores in this implementation
            # Use a default score of 0.5
            bm25_results.append((chunk, 0.5))

        # Combine results using Reciprocal Rank Fusion (RRF)
        combined_results = self._reciprocal_rank_fusion(
            vector_results, bm25_results, k=k
        )

        return combined_results

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Tuple[Chunk, float]],
        bm25_results: List[Tuple[Chunk, float]],
        k: int = 5,
        rrf_k: int = 60,
    ) -> List[Tuple[Chunk, float]]:
        """
        Combine results using Reciprocal Rank Fusion.

        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            k: Number of final results to return
            rrf_k: RRF constant (typically 60)

        Returns:
            Combined and re-ranked results
        """
        # Build score dictionary
        scores: Dict[str, float] = {}
        chunk_map: Dict[str, Chunk] = {}

        # Add vector search scores
        for rank, (chunk, score) in enumerate(vector_results):
            chunk_id = chunk.chunk_id
            rrf_score = 1.0 / (rrf_k + rank + 1)
            scores[chunk_id] = scores.get(chunk_id, 0) + self.alpha * rrf_score
            chunk_map[chunk_id] = chunk

        # Add BM25 scores
        for rank, (chunk, score) in enumerate(bm25_results):
            chunk_id = chunk.chunk_id
            rrf_score = 1.0 / (rrf_k + rank + 1)
            scores[chunk_id] = scores.get(chunk_id, 0) + (1 - self.alpha) * rrf_score
            chunk_map[chunk_id] = chunk

        # Sort by combined score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Return top k results
        results = [(chunk_map[chunk_id], scores[chunk_id]) for chunk_id in sorted_ids[:k]]

        return results

    def _weighted_fusion(
        self,
        vector_results: List[Tuple[Chunk, float]],
        bm25_results: List[Tuple[Chunk, float]],
        k: int = 5,
    ) -> List[Tuple[Chunk, float]]:
        """
        Alternative fusion strategy using weighted scores.

        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            k: Number of final results to return

        Returns:
            Combined and re-ranked results
        """
        scores: Dict[str, float] = {}
        chunk_map: Dict[str, Chunk] = {}

        # Normalize and weight vector scores
        if vector_results:
            max_vector_score = max(score for _, score in vector_results)
            for chunk, score in vector_results:
                normalized_score = score / max_vector_score if max_vector_score > 0 else 0
                chunk_id = chunk.chunk_id
                scores[chunk_id] = scores.get(chunk_id, 0) + self.alpha * normalized_score
                chunk_map[chunk_id] = chunk

        # Normalize and weight BM25 scores
        if bm25_results:
            max_bm25_score = max(score for _, score in bm25_results)
            for chunk, score in bm25_results:
                normalized_score = score / max_bm25_score if max_bm25_score > 0 else 0
                chunk_id = chunk.chunk_id
                scores[chunk_id] = scores.get(chunk_id, 0) + (1 - self.alpha) * normalized_score
                chunk_map[chunk_id] = chunk

        # Sort and return top k
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        results = [(chunk_map[chunk_id], scores[chunk_id]) for chunk_id in sorted_ids[:k]]

        return results
