"""Table processing utilities for multimodal RAG."""

from typing import Dict, List, Optional

from bs4 import BeautifulSoup
from loguru import logger


class TableProcessor:
    """Processor for extracting and formatting tables."""

    @staticmethod
    def html_to_markdown(html: str) -> str:
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
            soup = BeautifulSoup(html, "html.parser")
            table = soup.find("table")

            if not table:
                return html

            markdown_lines = []

            # Extract headers
            headers = []
            thead = table.find("thead")
            if thead:
                header_row = thead.find("tr")
                if header_row:
                    headers = [th.get_text(strip=True) for th in header_row.find_all(["th", "td"])]

            # If no thead, check first row
            if not headers:
                first_row = table.find("tr")
                if first_row:
                    cells = first_row.find_all(["th", "td"])
                    # Check if first row looks like headers
                    if cells and cells[0].name == "th":
                        headers = [cell.get_text(strip=True) for cell in cells]

            # Build markdown header
            if headers:
                markdown_lines.append("| " + " | ".join(headers) + " |")
                markdown_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

            # Extract body rows
            tbody = table.find("tbody") if table.find("tbody") else table
            for row in tbody.find_all("tr"):
                cells = [cell.get_text(strip=True) for cell in row.find_all(["td", "th"])]
                if cells:
                    # Skip if this was the header row
                    if not (headers and cells == headers):
                        markdown_lines.append("| " + " | ".join(cells) + " |")

            return "\n".join(markdown_lines)

        except Exception as e:
            logger.warning(f"Error converting HTML table to Markdown: {e}")
            return html

    @staticmethod
    def extract_table_data(markdown_table: str) -> List[Dict[str, str]]:
        """
        Extract structured data from markdown table.

        Args:
            markdown_table: Markdown formatted table

        Returns:
            List of dictionaries, one per row
        """
        if not markdown_table:
            return []

        try:
            lines = [line.strip() for line in markdown_table.split("\n") if line.strip()]

            if len(lines) < 2:
                return []

            # Parse header
            header_line = lines[0].strip("|").strip()
            headers = [h.strip() for h in header_line.split("|")]

            # Skip separator line
            data_lines = lines[2:] if len(lines) > 2 else []

            # Parse data rows
            data = []
            for line in data_lines:
                if not line.startswith("|"):
                    continue

                values = [v.strip() for v in line.strip("|").strip().split("|")]

                # Create dictionary for this row
                if len(values) == len(headers):
                    row_dict = dict(zip(headers, values))
                    data.append(row_dict)

            return data

        except Exception as e:
            logger.warning(f"Error extracting table data: {e}")
            return []

    @staticmethod
    def create_table_summary(markdown_table: str, caption: Optional[str] = None) -> str:
        """
        Create a text summary of a table for better searchability.

        Args:
            markdown_table: Markdown formatted table
            caption: Optional table caption

        Returns:
            Text summary of the table
        """
        summary_parts = []

        if caption:
            summary_parts.append(f"Table: {caption}")

        # Extract headers
        try:
            lines = [line.strip() for line in markdown_table.split("\n") if line.strip()]
            if lines:
                header_line = lines[0].strip("|").strip()
                headers = [h.strip() for h in header_line.split("|")]
                summary_parts.append(f"Columns: {', '.join(headers)}")

                # Count rows
                data_lines = [l for l in lines[2:] if l.startswith("|")]
                summary_parts.append(f"Number of rows: {len(data_lines)}")

        except Exception as e:
            logger.warning(f"Error creating table summary: {e}")

        return "\n".join(summary_parts)
