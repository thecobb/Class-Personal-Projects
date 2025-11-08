"""Document parsers for multimodal content extraction."""

import base64
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from PIL import Image
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf


@dataclass
class ImageElement:
    """Represents an extracted image with metadata."""

    image_data: bytes
    image_base64: str
    page_number: Optional[int] = None
    element_id: Optional[str] = None
    caption: Optional[str] = None
    surrounding_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_pil_image(
        cls,
        image: Image.Image,
        page_number: Optional[int] = None,
        element_id: Optional[str] = None,
        caption: Optional[str] = None,
    ) -> "ImageElement":
        """Create ImageElement from PIL Image."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_data = buffer.getvalue()
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        return cls(
            image_data=image_data,
            image_base64=image_base64,
            page_number=page_number,
            element_id=element_id,
            caption=caption,
        )


@dataclass
class TableElement:
    """Represents an extracted table."""

    html: str
    markdown: str
    page_number: Optional[int] = None
    element_id: Optional[str] = None
    caption: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TextElement:
    """Represents a text chunk."""

    text: str
    page_number: Optional[int] = None
    element_id: Optional[str] = None
    element_type: str = "text"  # NarrativeText, Title, ListItem, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultimodalDocument:
    """Container for a parsed multimodal document."""

    text_elements: List[TextElement] = field(default_factory=list)
    image_elements: List[ImageElement] = field(default_factory=list)
    table_elements: List[TableElement] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Get all text content concatenated."""
        return "\n\n".join(elem.text for elem in self.text_elements)

    @property
    def num_images(self) -> int:
        """Number of images in document."""
        return len(self.image_elements)

    @property
    def num_tables(self) -> int:
        """Number of tables in document."""
        return len(self.table_elements)


class DocumentParser:
    """Parser for extracting multimodal content from documents."""

    def __init__(self, extract_images: bool = True, extract_tables: bool = True):
        """
        Initialize document parser.

        Args:
            extract_images: Whether to extract images from documents
            extract_tables: Whether to extract tables from documents
        """
        self.extract_images = extract_images
        self.extract_tables = extract_tables

    def parse(self, file_path: Union[str, Path]) -> MultimodalDocument:
        """
        Parse a document and extract multimodal elements.

        Args:
            file_path: Path to document file

        Returns:
            MultimodalDocument containing extracted elements
        """
        file_path = Path(file_path)
        logger.info(f"Parsing document: {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Use appropriate parser based on file type
        if file_path.suffix.lower() == ".pdf":
            return self._parse_pdf(file_path)
        else:
            return self._parse_generic(file_path)

    def _parse_pdf(self, file_path: Path) -> MultimodalDocument:
        """
        Parse PDF with advanced extraction.

        Args:
            file_path: Path to PDF file

        Returns:
            MultimodalDocument with extracted content
        """
        doc = MultimodalDocument(metadata={"source": str(file_path), "file_type": "pdf"})

        try:
            # Use hi_res strategy for better image and table extraction
            elements = partition_pdf(
                filename=str(file_path),
                strategy="hi_res",
                extract_images_in_pdf=self.extract_images,
                infer_table_structure=self.extract_tables,
            )

            for idx, element in enumerate(elements):
                element_type = type(element).__name__
                element_id = f"{file_path.stem}_{idx}"

                # Extract text elements
                if hasattr(element, "text") and element.text:
                    text_elem = TextElement(
                        text=element.text,
                        element_id=element_id,
                        element_type=element_type,
                        metadata=element.metadata.to_dict() if hasattr(element, "metadata") else {},
                    )
                    doc.text_elements.append(text_elem)

                # Extract tables
                if element_type == "Table" and self.extract_tables:
                    html_table = element.metadata.text_as_html if hasattr(element, "metadata") else ""
                    markdown_table = self._html_to_markdown(html_table)

                    table_elem = TableElement(
                        html=html_table,
                        markdown=markdown_table,
                        element_id=element_id,
                        metadata=element.metadata.to_dict() if hasattr(element, "metadata") else {},
                    )
                    doc.table_elements.append(table_elem)

            logger.info(
                f"Extracted {len(doc.text_elements)} text elements, "
                f"{len(doc.table_elements)} tables from {file_path}"
            )

        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {e}")
            raise

        return doc

    def _parse_generic(self, file_path: Path) -> MultimodalDocument:
        """
        Parse generic document types (DOCX, PPTX, HTML, etc.).

        Args:
            file_path: Path to document file

        Returns:
            MultimodalDocument with extracted content
        """
        doc = MultimodalDocument(
            metadata={"source": str(file_path), "file_type": file_path.suffix}
        )

        try:
            elements = partition(filename=str(file_path))

            for idx, element in enumerate(elements):
                element_type = type(element).__name__
                element_id = f"{file_path.stem}_{idx}"

                if hasattr(element, "text") and element.text:
                    text_elem = TextElement(
                        text=element.text,
                        element_id=element_id,
                        element_type=element_type,
                        metadata=element.metadata.to_dict() if hasattr(element, "metadata") else {},
                    )
                    doc.text_elements.append(text_elem)

            logger.info(f"Extracted {len(doc.text_elements)} text elements from {file_path}")

        except Exception as e:
            logger.error(f"Error parsing document {file_path}: {e}")
            raise

        return doc

    @staticmethod
    def _html_to_markdown(html: str) -> str:
        """
        Convert HTML table to Markdown format.

        Args:
            html: HTML table string

        Returns:
            Markdown formatted table
        """
        if not html:
            return ""

        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            table = soup.find("table")

            if not table:
                return ""

            markdown_lines = []

            # Process headers
            headers = []
            header_row = table.find("thead")
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all("th")]
            elif table.find("tr"):
                # First row might be headers
                first_row = table.find("tr")
                headers = [td.get_text(strip=True) for td in first_row.find_all(["th", "td"])]

            if headers:
                markdown_lines.append("| " + " | ".join(headers) + " |")
                markdown_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

            # Process body rows
            tbody = table.find("tbody") or table
            for row in tbody.find_all("tr"):
                cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
                if cells and cells != headers:  # Skip header row if it was in tbody
                    markdown_lines.append("| " + " | ".join(cells) + " |")

            return "\n".join(markdown_lines)

        except Exception as e:
            logger.warning(f"Error converting HTML to Markdown: {e}")
            return html
