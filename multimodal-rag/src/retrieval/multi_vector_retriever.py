"""Multi-vector retriever for multimodal RAG."""

import uuid
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from ..document_processing.chunking import Chunk
from ..document_processing.parsers import ImageElement, TableElement
from .vector_store import VectorStore


class DocumentStore:
    """Abstract document store for raw content."""

    def __init__(self) -> None:
        """Initialize document store."""
        self.store: Dict[str, Any] = {}

    def add(self, doc_id: str, content: Any) -> None:
        """Add document to store."""
        self.store[doc_id] = content

    def get(self, doc_id: str) -> Optional[Any]:
        """Get document from store."""
        return self.store.get(doc_id)

    def get_many(self, doc_ids: List[str]) -> List[Any]:
        """Get multiple documents from store."""
        return [self.store.get(doc_id) for doc_id in doc_ids if doc_id in self.store]

    def delete(self, doc_id: str) -> None:
        """Delete document from store."""
        if doc_id in self.store:
            del self.store[doc_id]


class MultiVectorRetriever:
    """
    Multi-vector retriever implementing the pattern of storing summaries
    in vector DB while keeping original content (images, full text) separate.

    This is the recommended architecture from Jason Liu's methodology:
    - Store summaries in vector database for fast retrieval
    - Store original content in separate document store
    - Link through unique IDs
    - Query retrieves summaries, then looks up original content
    """

    def __init__(
        self,
        vector_store: VectorStore,
        document_store: Optional[DocumentStore] = None,
    ):
        """
        Initialize multi-vector retriever.

        Args:
            vector_store: Vector store for summary embeddings
            document_store: Store for original content (images, full text, tables)
        """
        self.vector_store = vector_store
        self.document_store = document_store or DocumentStore()

        # Track mapping between summary IDs and document IDs
        self.summary_to_doc: Dict[str, str] = {}

        logger.info("Initialized MultiVectorRetriever")

    def add_text_chunks(
        self,
        chunks: List[Chunk],
        summaries: Optional[List[str]] = None,
    ) -> None:
        """
        Add text chunks with optional summaries.

        Args:
            chunks: Original text chunks
            summaries: Optional summaries (if None, uses chunks as-is)
        """
        if not chunks:
            return

        logger.info(f"Adding {len(chunks)} text chunks to MultiVectorRetriever")

        # If no summaries provided, use chunks directly
        if summaries is None:
            self.vector_store.add_chunks(chunks)
            return

        # Create summary chunks
        summary_chunks = []
        for chunk, summary in zip(chunks, summaries):
            # Generate unique ID for summary
            summary_id = f"{chunk.chunk_id}_summary"

            # Create summary chunk
            summary_chunk = Chunk(
                text=summary,
                chunk_id=summary_id,
                metadata={
                    **chunk.metadata,
                    "content_type": "text",
                    "is_summary": True,
                    "original_id": chunk.chunk_id,
                },
            )
            summary_chunks.append(summary_chunk)

            # Store original chunk
            self.document_store.add(chunk.chunk_id, chunk)

            # Track mapping
            self.summary_to_doc[summary_id] = chunk.chunk_id

        # Add summaries to vector store
        self.vector_store.add_chunks(summary_chunks)

        logger.info(f"Added {len(summary_chunks)} text summaries")

    def add_images(
        self,
        images: List[ImageElement],
        descriptions: List[str],
    ) -> None:
        """
        Add images with their descriptions.

        Args:
            images: List of image elements
            descriptions: Descriptions/summaries of images for search
        """
        if not images or not descriptions:
            return

        if len(images) != len(descriptions):
            raise ValueError("Number of images must match number of descriptions")

        logger.info(f"Adding {len(images)} images to MultiVectorRetriever")

        # Create description chunks
        description_chunks = []
        for image, description in zip(images, descriptions):
            # Generate unique ID
            image_id = image.element_id or str(uuid.uuid4())
            summary_id = f"{image_id}_description"

            # Create description chunk
            desc_chunk = Chunk(
                text=description,
                chunk_id=summary_id,
                metadata={
                    "content_type": "image",
                    "is_summary": True,
                    "original_id": image_id,
                    "page_number": image.page_number,
                    "caption": image.caption,
                },
            )
            description_chunks.append(desc_chunk)

            # Store original image
            self.document_store.add(image_id, image)

            # Track mapping
            self.summary_to_doc[summary_id] = image_id

        # Add descriptions to vector store
        self.vector_store.add_chunks(description_chunks)

        logger.info(f"Added {len(description_chunks)} image descriptions")

    def add_tables(
        self,
        tables: List[TableElement],
        summaries: Optional[List[str]] = None,
    ) -> None:
        """
        Add tables with optional summaries.

        Args:
            tables: List of table elements
            summaries: Optional summaries (if None, uses markdown representation)
        """
        if not tables:
            return

        logger.info(f"Adding {len(tables)} tables to MultiVectorRetriever")

        # Create table summary chunks
        summary_chunks = []
        for idx, table in enumerate(tables):
            # Generate unique ID
            table_id = table.element_id or str(uuid.uuid4())
            summary_id = f"{table_id}_summary"

            # Use provided summary or create from markdown
            if summaries and idx < len(summaries):
                summary = summaries[idx]
            else:
                # Create summary from caption + markdown preview
                summary_parts = []
                if table.caption:
                    summary_parts.append(f"Table: {table.caption}")
                # Include markdown (truncate if too long)
                markdown_preview = table.markdown[:500] + "..." if len(table.markdown) > 500 else table.markdown
                summary_parts.append(markdown_preview)
                summary = "\n".join(summary_parts)

            # Create summary chunk
            summary_chunk = Chunk(
                text=summary,
                chunk_id=summary_id,
                metadata={
                    "content_type": "table",
                    "is_summary": True,
                    "original_id": table_id,
                    "page_number": table.page_number,
                    "caption": table.caption,
                },
            )
            summary_chunks.append(summary_chunk)

            # Store original table
            self.document_store.add(table_id, table)

            # Track mapping
            self.summary_to_doc[summary_id] = table_id

        # Add summaries to vector store
        self.vector_store.add_chunks(summary_chunks)

        logger.info(f"Added {len(summary_chunks)} table summaries")

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        return_originals: bool = True,
    ) -> List[Tuple[Any, float]]:
        """
        Retrieve documents using summary search.

        Args:
            query: Query text
            k: Number of results to return
            filter: Metadata filter
            return_originals: If True, return original content; if False, return summaries

        Returns:
            List of (content, score) tuples
        """
        # Search using summaries
        summary_results = self.vector_store.similarity_search(query, k=k, filter=filter)

        if not return_originals:
            return summary_results

        # Look up original content
        original_results = []
        for summary_chunk, score in summary_results:
            summary_id = summary_chunk.chunk_id

            # Get original document ID
            if summary_id in self.summary_to_doc:
                doc_id = self.summary_to_doc[summary_id]
                original_doc = self.document_store.get(doc_id)

                if original_doc:
                    original_results.append((original_doc, score))
                else:
                    # Fall back to summary if original not found
                    logger.warning(f"Original document not found for {doc_id}, using summary")
                    original_results.append((summary_chunk, score))
            else:
                # No mapping found, return summary
                original_results.append((summary_chunk, score))

        return original_results

    def retrieve_with_context(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents with both original content and summaries.

        Args:
            query: Query text
            k: Number of results to return
            filter: Metadata filter

        Returns:
            List of dictionaries with 'content', 'summary', 'score', and 'metadata'
        """
        # Get summary results
        summary_results = self.vector_store.similarity_search(query, k=k, filter=filter)

        # Build results with context
        results = []
        for summary_chunk, score in summary_results:
            summary_id = summary_chunk.chunk_id

            result = {
                "summary": summary_chunk.text,
                "score": score,
                "metadata": summary_chunk.metadata,
                "content": None,
            }

            # Look up original content
            if summary_id in self.summary_to_doc:
                doc_id = self.summary_to_doc[summary_id]
                original_doc = self.document_store.get(doc_id)
                result["content"] = original_doc

            results.append(result)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the retriever."""
        return {
            "vector_store_stats": self.vector_store.get_collection_stats(),
            "document_store_count": len(self.document_store.store),
            "summary_mappings": len(self.summary_to_doc),
        }
