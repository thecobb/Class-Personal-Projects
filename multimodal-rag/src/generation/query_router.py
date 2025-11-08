"""Query routing for multimodal RAG."""

from enum import Enum
from typing import Dict, List, Optional

import instructor
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel, Field

from ..config import settings


class QueryType(str, Enum):
    """Types of queries for routing."""

    TEXT_ONLY = "text_only"
    IMAGE_ONLY = "image_only"
    MULTIMODAL = "multimodal"
    TABLE_SEARCH = "table_search"
    CHART_ANALYSIS = "chart_analysis"


class QueryClassification(BaseModel):
    """Structured output for query classification."""

    query_type: QueryType = Field(description="The type of query")
    confidence: float = Field(
        description="Confidence score 0-1",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(description="Brief explanation of the classification")
    metadata_filters: Dict[str, str] = Field(
        default_factory=dict,
        description="Suggested metadata filters to apply"
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Key terms from the query"
    )


class QueryRouter:
    """
    Intelligent query router for multimodal RAG.

    Uses structured outputs (via Instructor library) to classify queries
    and route them to appropriate retrieval strategies.
    """

    CLASSIFICATION_PROMPT = """Analyze the following user query and classify it:

Query: {query}

Determine:
1. What type of content would best answer this query?
   - text_only: Can be answered with text documents alone
   - image_only: Looking specifically for images/visual content
   - multimodal: Requires both text and images
   - table_search: Looking for tabular data
   - chart_analysis: Looking for charts/graphs to analyze

2. How confident are you in this classification? (0-1)

3. What metadata filters might help narrow the search?
   - date ranges
   - document types
   - categories
   - etc.

4. What are the key terms/concepts?
"""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
    ):
        """
        Initialize query router.

        Args:
            model: Model to use for classification
            api_key: OpenAI API key
        """
        self.model = model
        self.api_key = api_key or settings.openai_api_key

        if not self.api_key:
            logger.warning("No OpenAI API key provided. Query routing will fail.")

        # Initialize Instructor client for structured outputs
        self.client = instructor.from_openai(OpenAI(api_key=self.api_key))

        logger.info(f"Initialized QueryRouter with model: {self.model}")

    def classify_query(self, query: str) -> QueryClassification:
        """
        Classify a query into appropriate routing category.

        Args:
            query: User query text

        Returns:
            QueryClassification with routing information
        """
        try:
            # Format prompt
            prompt = self.CLASSIFICATION_PROMPT.format(query=query)

            # Get structured classification
            classification = self.client.chat.completions.create(
                model=self.model,
                response_model=QueryClassification,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing search queries."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
            )

            logger.debug(
                f"Classified query as {classification.query_type} "
                f"(confidence: {classification.confidence:.2f})"
            )

            return classification

        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            # Return default classification
            return QueryClassification(
                query_type=QueryType.TEXT_ONLY,
                confidence=0.5,
                reasoning="Error during classification, defaulting to text_only",
                keywords=query.split()[:5]
            )

    def route(self, query: str) -> Dict[str, any]:
        """
        Route a query and return retrieval strategy.

        Args:
            query: User query

        Returns:
            Dictionary with routing decisions
        """
        classification = self.classify_query(query)

        routing = {
            "query_type": classification.query_type,
            "confidence": classification.confidence,
            "search_text": True,
            "search_images": False,
            "search_tables": False,
            "metadata_filters": classification.metadata_filters,
            "keywords": classification.keywords,
            "reasoning": classification.reasoning,
        }

        # Set search flags based on query type
        if classification.query_type == QueryType.IMAGE_ONLY:
            routing["search_images"] = True
            routing["search_text"] = False
        elif classification.query_type == QueryType.MULTIMODAL:
            routing["search_text"] = True
            routing["search_images"] = True
        elif classification.query_type == QueryType.TABLE_SEARCH:
            routing["search_tables"] = True
            routing["search_text"] = True
        elif classification.query_type == QueryType.CHART_ANALYSIS:
            routing["search_images"] = True
            routing["search_text"] = True
            # Add filter for chart content
            routing["metadata_filters"]["content_type"] = "chart"

        return routing


class SimpleQueryRouter:
    """
    Simple rule-based query router (no API calls needed).

    Use as fallback when API-based routing is not available.
    """

    # Keywords indicating different query types
    IMAGE_KEYWORDS = ["image", "photo", "picture", "visual", "show me", "diagram"]
    TABLE_KEYWORDS = ["table", "data", "statistics", "numbers", "rows", "columns"]
    CHART_KEYWORDS = ["chart", "graph", "plot", "trend", "visualization"]

    def classify_query(self, query: str) -> QueryType:
        """
        Simple keyword-based classification.

        Args:
            query: User query

        Returns:
            QueryType classification
        """
        query_lower = query.lower()

        # Check for chart keywords
        if any(keyword in query_lower for keyword in self.CHART_KEYWORDS):
            return QueryType.CHART_ANALYSIS

        # Check for table keywords
        if any(keyword in query_lower for keyword in self.TABLE_KEYWORDS):
            return QueryType.TABLE_SEARCH

        # Check for image keywords
        if any(keyword in query_lower for keyword in self.IMAGE_KEYWORDS):
            return QueryType.IMAGE_ONLY

        # Default to text
        return QueryType.TEXT_ONLY

    def route(self, query: str) -> Dict[str, any]:
        """
        Route query using simple rules.

        Args:
            query: User query

        Returns:
            Routing dictionary
        """
        query_type = self.classify_query(query)

        routing = {
            "query_type": query_type,
            "confidence": 0.7,  # Fixed confidence for rule-based
            "search_text": True,
            "search_images": False,
            "search_tables": False,
            "metadata_filters": {},
            "keywords": query.split()[:5],
            "reasoning": f"Rule-based classification: {query_type}",
        }

        # Set search flags
        if query_type == QueryType.IMAGE_ONLY:
            routing["search_images"] = True
            routing["search_text"] = False
        elif query_type == QueryType.MULTIMODAL:
            routing["search_text"] = True
            routing["search_images"] = True
        elif query_type == QueryType.TABLE_SEARCH:
            routing["search_tables"] = True
        elif query_type == QueryType.CHART_ANALYSIS:
            routing["search_images"] = True
            routing["metadata_filters"]["content_type"] = "chart"

        return routing
