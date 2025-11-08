"""Vector store abstraction for multiple database backends."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from langchain_community.vectorstores import Chroma
from loguru import logger

from ..config import settings
from ..document_processing.chunking import Chunk
from ..embeddings.embedding_service import EmbeddingService


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add_chunks(self, chunks: List[Chunk], embeddings: Optional[List[List[float]]] = None) -> None:
        """Add chunks to the vector store."""
        pass

    @abstractmethod
    def similarity_search(
        self, query: str, k: int = 5, filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Chunk, float]]:
        """Perform similarity search."""
        pass

    @abstractmethod
    def delete_collection(self) -> None:
        """Delete the collection."""
        pass

    @abstractmethod
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        pass


class ChromaVectorStore(VectorStore):
    """ChromaDB-based vector store implementation."""

    def __init__(
        self,
        collection_name: str,
        persist_directory: Optional[Path] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist the database
            embedding_service: Embedding service to use
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory or settings.vector_db_path
        self.embedding_service = embedding_service or EmbeddingService()

        # Ensure directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing ChromaDB at {self.persist_directory}")

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))

        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Created new collection: {collection_name}")

        # Also create LangChain wrapper for advanced features
        self.langchain_store = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embedding_service.embeddings,
        )

    def add_chunks(self, chunks: List[Chunk], embeddings: Optional[List[List[float]]] = None) -> None:
        """
        Add chunks to the vector store.

        Args:
            chunks: List of chunks to add
            embeddings: Pre-computed embeddings (optional)
        """
        if not chunks:
            return

        logger.info(f"Adding {len(chunks)} chunks to ChromaDB")

        # Prepare data
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # Generate embeddings if not provided
        if embeddings is None:
            logger.debug("Generating embeddings for chunks")
            embeddings = self.embedding_service.embed_texts(documents)

        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(f"Successfully added {len(chunks)} chunks")

    def similarity_search(
        self, query: str, k: int = 5, filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Chunk, float]]:
        """
        Perform similarity search.

        Args:
            query: Query text
            k: Number of results to return
            filter: Metadata filter

        Returns:
            List of (Chunk, score) tuples
        """
        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query)

        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filter,
            include=["documents", "metadatas", "distances"],
        )

        # Convert to Chunk objects
        chunks_with_scores = []
        for i in range(len(results["ids"][0])):
            chunk = Chunk(
                text=results["documents"][0][i],
                chunk_id=results["ids"][0][i],
                metadata=results["metadatas"][0][i],
            )
            # ChromaDB returns distances, convert to similarity scores
            distance = results["distances"][0][i]
            similarity = 1 - distance  # Cosine distance to similarity

            chunks_with_scores.append((chunk, similarity))

        return chunks_with_scores

    def delete_collection(self) -> None:
        """Delete the collection."""
        logger.info(f"Deleting collection: {self.collection_name}")
        self.client.delete_collection(name=self.collection_name)

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        count = self.collection.count()
        return {
            "name": self.collection_name,
            "count": count,
            "persist_directory": str(self.persist_directory),
        }


class LanceDBVectorStore(VectorStore):
    """LanceDB-based vector store implementation."""

    def __init__(
        self,
        collection_name: str,
        persist_directory: Optional[Path] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        """
        Initialize LanceDB vector store.

        Args:
            collection_name: Name of the table
            persist_directory: Directory to persist the database
            embedding_service: Embedding service to use
        """
        try:
            import lancedb
        except ImportError:
            raise ImportError("lancedb not installed. Install with: pip install lancedb")

        self.collection_name = collection_name
        self.persist_directory = persist_directory or settings.vector_db_path
        self.embedding_service = embedding_service or EmbeddingService()

        # Ensure directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing LanceDB at {self.persist_directory}")

        # Initialize LanceDB
        self.db = lancedb.connect(str(self.persist_directory))
        self.table_name = collection_name

        logger.info(f"Connected to LanceDB")

    def add_chunks(self, chunks: List[Chunk], embeddings: Optional[List[List[float]]] = None) -> None:
        """Add chunks to LanceDB."""
        if not chunks:
            return

        logger.info(f"Adding {len(chunks)} chunks to LanceDB")

        # Generate embeddings if not provided
        if embeddings is None:
            documents = [chunk.text for chunk in chunks]
            embeddings = self.embedding_service.embed_texts(documents)

        # Prepare data
        data = []
        for chunk, embedding in zip(chunks, embeddings):
            data.append(
                {
                    "id": chunk.chunk_id,
                    "text": chunk.text,
                    "vector": embedding,
                    "metadata": str(chunk.metadata),  # LanceDB stores as string
                }
            )

        # Create or append to table
        if self.table_name in self.db.table_names():
            table = self.db.open_table(self.table_name)
            table.add(data)
        else:
            self.db.create_table(self.table_name, data)

        logger.info(f"Successfully added {len(chunks)} chunks")

    def similarity_search(
        self, query: str, k: int = 5, filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Chunk, float]]:
        """Perform similarity search in LanceDB."""
        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query)

        # Open table
        table = self.db.open_table(self.table_name)

        # Search
        results = table.search(query_embedding).limit(k).to_pandas()

        # Convert to Chunk objects
        chunks_with_scores = []
        for _, row in results.iterrows():
            import ast

            metadata = ast.literal_eval(row["metadata"]) if isinstance(row["metadata"], str) else {}

            chunk = Chunk(
                text=row["text"],
                chunk_id=row["id"],
                metadata=metadata,
            )
            similarity = float(row["_distance"])  # LanceDB returns distance

            chunks_with_scores.append((chunk, similarity))

        return chunks_with_scores

    def delete_collection(self) -> None:
        """Delete the table."""
        logger.info(f"Deleting table: {self.table_name}")
        self.db.drop_table(self.table_name)

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the table."""
        if self.table_name in self.db.table_names():
            table = self.db.open_table(self.table_name)
            count = table.count_rows()
        else:
            count = 0

        return {
            "name": self.table_name,
            "count": count,
            "persist_directory": str(self.persist_directory),
        }


class VectorStoreFactory:
    """Factory for creating vector store instances."""

    @staticmethod
    def create(
        store_type: Optional[str] = None,
        collection_name: Optional[str] = None,
        persist_directory: Optional[Path] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ) -> VectorStore:
        """
        Create a vector store instance.

        Args:
            store_type: Type of vector store ('chromadb', 'lancedb', etc.)
            collection_name: Name of the collection/table
            persist_directory: Directory to persist the database
            embedding_service: Embedding service to use

        Returns:
            VectorStore instance
        """
        store_type = store_type or settings.vector_db_type
        collection_name = collection_name or settings.collection_name
        persist_directory = persist_directory or settings.vector_db_path
        embedding_service = embedding_service or EmbeddingService()

        logger.info(f"Creating {store_type} vector store")

        if store_type == "chromadb":
            return ChromaVectorStore(collection_name, persist_directory, embedding_service)
        elif store_type == "lancedb":
            return LanceDBVectorStore(collection_name, persist_directory, embedding_service)
        else:
            raise ValueError(
                f"Unsupported vector store type: {store_type}. "
                f"Supported types: chromadb, lancedb"
            )
