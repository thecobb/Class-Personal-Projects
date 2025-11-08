"""Chunking strategies for document processing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

from langchain.text_splitter import RecursiveCharacterTextSplitter

from .parsers import MultimodalDocument, TextElement


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""

    text: str
    chunk_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: List[float] = field(default_factory=list)


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""

    @abstractmethod
    def chunk(self, document: MultimodalDocument) -> List[Chunk]:
        """
        Chunk a document into smaller pieces.

        Args:
            document: MultimodalDocument to chunk

        Returns:
            List of Chunk objects
        """
        pass


class RecursiveChunker(ChunkingStrategy):
    """Recursive character-based chunking strategy."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize recursive chunker.

        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, document: MultimodalDocument) -> List[Chunk]:
        """
        Chunk document using recursive character splitting.

        Args:
            document: MultimodalDocument to chunk

        Returns:
            List of Chunk objects
        """
        chunks = []
        full_text = document.full_text

        text_chunks = self.splitter.split_text(full_text)

        for idx, text in enumerate(text_chunks):
            chunk = Chunk(
                text=text,
                chunk_id=f"{document.metadata.get('source', 'doc')}_{idx}",
                metadata={
                    **document.metadata,
                    "chunk_index": idx,
                    "chunking_strategy": "recursive",
                },
            )
            chunks.append(chunk)

        return chunks


class SectionChunker(ChunkingStrategy):
    """Section-based chunking that preserves document structure."""

    def __init__(self, max_section_size: int = 2000):
        """
        Initialize section chunker.

        Args:
            max_section_size: Maximum size for a section before splitting
        """
        self.max_section_size = max_section_size

    def chunk(self, document: MultimodalDocument) -> List[Chunk]:
        """
        Chunk document by sections, preserving structure.

        Args:
            document: MultimodalDocument to chunk

        Returns:
            List of Chunk objects
        """
        chunks = []
        current_section = []
        current_size = 0
        section_idx = 0

        for elem in document.text_elements:
            # Start new section on titles or if size limit reached
            if elem.element_type == "Title" or current_size >= self.max_section_size:
                if current_section:
                    chunk = self._create_chunk(current_section, document, section_idx)
                    chunks.append(chunk)
                    section_idx += 1
                current_section = [elem]
                current_size = len(elem.text)
            else:
                current_section.append(elem)
                current_size += len(elem.text)

        # Add final section
        if current_section:
            chunk = self._create_chunk(current_section, document, section_idx)
            chunks.append(chunk)

        return chunks

    def _create_chunk(
        self, elements: List[TextElement], document: MultimodalDocument, section_idx: int
    ) -> Chunk:
        """Create a chunk from a list of text elements."""
        text = "\n\n".join(elem.text for elem in elements)
        return Chunk(
            text=text,
            chunk_id=f"{document.metadata.get('source', 'doc')}_section_{section_idx}",
            metadata={
                **document.metadata,
                "chunk_index": section_idx,
                "chunking_strategy": "section",
                "num_elements": len(elements),
            },
        )


class ElementBasedChunker(ChunkingStrategy):
    """
    Element-based chunking that treats each element separately.
    Best for documents with mixed text, tables, and images.
    """

    def __init__(self, group_small_elements: bool = True, min_element_size: int = 100):
        """
        Initialize element-based chunker.

        Args:
            group_small_elements: Whether to group small consecutive elements
            min_element_size: Minimum size for standalone elements
        """
        self.group_small_elements = group_small_elements
        self.min_element_size = min_element_size

    def chunk(self, document: MultimodalDocument) -> List[Chunk]:
        """
        Chunk document by elements, optionally grouping small ones.

        Args:
            document: MultimodalDocument to chunk

        Returns:
            List of Chunk objects
        """
        chunks = []

        if self.group_small_elements:
            chunks = self._chunk_with_grouping(document)
        else:
            chunks = self._chunk_by_element(document)

        return chunks

    def _chunk_by_element(self, document: MultimodalDocument) -> List[Chunk]:
        """Create one chunk per element."""
        chunks = []

        for idx, elem in enumerate(document.text_elements):
            chunk = Chunk(
                text=elem.text,
                chunk_id=f"{document.metadata.get('source', 'doc')}_elem_{idx}",
                metadata={
                    **document.metadata,
                    "chunk_index": idx,
                    "chunking_strategy": "element",
                    "element_type": elem.element_type,
                    "element_id": elem.element_id,
                },
            )
            chunks.append(chunk)

        return chunks

    def _chunk_with_grouping(self, document: MultimodalDocument) -> List[Chunk]:
        """Group small consecutive elements together."""
        chunks = []
        current_group = []
        current_size = 0
        chunk_idx = 0

        for elem in document.text_elements:
            elem_size = len(elem.text)

            # If element is large enough or current group would be too large
            if elem_size >= self.min_element_size or (
                current_size + elem_size > self.min_element_size * 2
            ):
                # Save current group if it exists
                if current_group:
                    chunk = self._create_grouped_chunk(current_group, document, chunk_idx)
                    chunks.append(chunk)
                    chunk_idx += 1
                    current_group = []
                    current_size = 0

                # Add large element as its own chunk
                if elem_size >= self.min_element_size:
                    chunk = Chunk(
                        text=elem.text,
                        chunk_id=f"{document.metadata.get('source', 'doc')}_elem_{chunk_idx}",
                        metadata={
                            **document.metadata,
                            "chunk_index": chunk_idx,
                            "chunking_strategy": "element_grouped",
                            "element_type": elem.element_type,
                            "element_id": elem.element_id,
                        },
                    )
                    chunks.append(chunk)
                    chunk_idx += 1
                else:
                    current_group.append(elem)
                    current_size += elem_size
            else:
                current_group.append(elem)
                current_size += elem_size

        # Add final group
        if current_group:
            chunk = self._create_grouped_chunk(current_group, document, chunk_idx)
            chunks.append(chunk)

        return chunks

    def _create_grouped_chunk(
        self, elements: List[TextElement], document: MultimodalDocument, chunk_idx: int
    ) -> Chunk:
        """Create a chunk from grouped elements."""
        text = "\n\n".join(elem.text for elem in elements)
        return Chunk(
            text=text,
            chunk_id=f"{document.metadata.get('source', 'doc')}_group_{chunk_idx}",
            metadata={
                **document.metadata,
                "chunk_index": chunk_idx,
                "chunking_strategy": "element_grouped",
                "num_elements": len(elements),
            },
        )
