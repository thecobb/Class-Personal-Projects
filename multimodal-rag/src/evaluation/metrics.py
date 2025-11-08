"""RAG evaluation metrics following Jason Liu's 6-tier framework."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger
from openai import OpenAI

from ..config import settings


@dataclass
class RAGMetrics:
    """
    Container for RAG evaluation metrics.

    Follows Jason Liu's 6-tier framework:
    - Tier 1: Retrieval Precision & Recall (no LLM needed)
    - Tier 2: Context Relevance (C|Q), Faithfulness (A|C), Answer Relevance (A|Q)
    - Tier 3: Advanced relationships
    """

    # Tier 1: Foundation metrics
    retrieval_precision: float = 0.0
    retrieval_recall: float = 0.0

    # Tier 2: Primary RAG relationships
    context_relevance: float = 0.0  # C|Q - Does context address the question?
    faithfulness: float = 0.0  # A|C - Is answer grounded in context?
    answer_relevance: float = 0.0  # A|Q - Does answer address the question?

    # Tier 3: Advanced relationships
    context_coverage: float = 0.0  # C|A - Does context support all answer claims?
    answerability: float = 0.0  # Q|C - Can context answer the question?

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "retrieval_precision": self.retrieval_precision,
            "retrieval_recall": self.retrieval_recall,
            "context_relevance": self.context_relevance,
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "context_coverage": self.context_coverage,
            "answerability": self.answerability,
        }


class MetricsCalculator:
    """Calculate RAG evaluation metrics."""

    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        """
        Initialize metrics calculator.

        Args:
            model: Model to use for LLM-based evaluation
            api_key: OpenAI API key
        """
        self.model = model
        self.api_key = api_key or settings.openai_api_key

        if not self.api_key:
            logger.warning("No OpenAI API key. LLM-based metrics will be disabled.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)

    def calculate_retrieval_precision(
        self,
        retrieved_chunks: List[str],
        relevant_chunks: List[str],
    ) -> float:
        """
        Calculate retrieval precision.

        Precision = (# relevant retrieved) / (# total retrieved)

        Args:
            retrieved_chunks: List of retrieved chunk IDs
            relevant_chunks: List of relevant chunk IDs (ground truth)

        Returns:
            Precision score (0-1)
        """
        if not retrieved_chunks:
            return 0.0

        retrieved_set = set(retrieved_chunks)
        relevant_set = set(relevant_chunks)

        num_relevant_retrieved = len(retrieved_set.intersection(relevant_set))
        precision = num_relevant_retrieved / len(retrieved_set)

        return precision

    def calculate_retrieval_recall(
        self,
        retrieved_chunks: List[str],
        relevant_chunks: List[str],
    ) -> float:
        """
        Calculate retrieval recall.

        Recall = (# relevant retrieved) / (# total relevant)

        Args:
            retrieved_chunks: List of retrieved chunk IDs
            relevant_chunks: List of relevant chunk IDs (ground truth)

        Returns:
            Recall score (0-1)
        """
        if not relevant_chunks:
            return 0.0

        retrieved_set = set(retrieved_chunks)
        relevant_set = set(relevant_chunks)

        num_relevant_retrieved = len(retrieved_set.intersection(relevant_set))
        recall = num_relevant_retrieved / len(relevant_set)

        return recall

    def calculate_context_relevance(
        self,
        query: str,
        context: str,
    ) -> float:
        """
        Calculate context relevance (C|Q).

        Does the retrieved context actually address the question?

        Args:
            query: User question
            context: Retrieved context

        Returns:
            Relevance score (0-1)
        """
        if not self.client:
            logger.warning("LLM client not available, skipping context relevance")
            return 0.0

        prompt = f"""Rate how relevant the following context is to answering the question.

Question: {query}

Context: {context}

Rate the relevance on a scale of 1-5:
1 - Not relevant at all
2 - Slightly relevant
3 - Moderately relevant
4 - Very relevant
5 - Perfectly relevant

Provide only the number (1-5).
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )

            score_text = response.choices[0].message.content.strip()
            score = int(score_text)

            # Normalize to 0-1
            normalized_score = (score - 1) / 4.0

            return normalized_score

        except Exception as e:
            logger.error(f"Error calculating context relevance: {e}")
            return 0.0

    def calculate_faithfulness(
        self,
        answer: str,
        context: str,
    ) -> float:
        """
        Calculate faithfulness/groundedness (A|C).

        Is the answer derived from the context without hallucination?

        Args:
            answer: Generated answer
            context: Retrieved context

        Returns:
            Faithfulness score (0-1)
        """
        if not self.client:
            logger.warning("LLM client not available, skipping faithfulness")
            return 0.0

        prompt = f"""Evaluate if the answer is faithful to the context (no hallucination).

Context: {context}

Answer: {answer}

Rate the faithfulness on a scale of 1-5:
1 - Answer contains mostly hallucinated information
2 - Answer contains some hallucination
3 - Answer is partially grounded
4 - Answer is mostly grounded
5 - Answer is completely grounded in context

Provide only the number (1-5).
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )

            score_text = response.choices[0].message.content.strip()
            score = int(score_text)

            # Normalize to 0-1
            normalized_score = (score - 1) / 4.0

            return normalized_score

        except Exception as e:
            logger.error(f"Error calculating faithfulness: {e}")
            return 0.0

    def calculate_answer_relevance(
        self,
        query: str,
        answer: str,
    ) -> float:
        """
        Calculate answer relevance (A|Q).

        Does the answer directly address the question?

        Args:
            query: User question
            answer: Generated answer

        Returns:
            Relevance score (0-1)
        """
        if not self.client:
            logger.warning("LLM client not available, skipping answer relevance")
            return 0.0

        prompt = f"""Rate how well the answer addresses the question.

Question: {query}

Answer: {answer}

Rate on a scale of 1-5:
1 - Answer is off-topic
2 - Answer is tangentially related
3 - Answer partially addresses the question
4 - Answer mostly addresses the question
5 - Answer fully addresses the question

Provide only the number (1-5).
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )

            score_text = response.choices[0].message.content.strip()
            score = int(score_text)

            # Normalize to 0-1
            normalized_score = (score - 1) / 4.0

            return normalized_score

        except Exception as e:
            logger.error(f"Error calculating answer relevance: {e}")
            return 0.0

    def calculate_all_metrics(
        self,
        query: str,
        answer: str,
        context: str,
        retrieved_chunk_ids: List[str],
        relevant_chunk_ids: List[str],
    ) -> RAGMetrics:
        """
        Calculate all RAG metrics.

        Args:
            query: User question
            answer: Generated answer
            context: Retrieved context (combined)
            retrieved_chunk_ids: IDs of retrieved chunks
            relevant_chunk_ids: IDs of relevant chunks (ground truth)

        Returns:
            RAGMetrics object with all scores
        """
        metrics = RAGMetrics()

        # Tier 1: Retrieval metrics
        metrics.retrieval_precision = self.calculate_retrieval_precision(
            retrieved_chunk_ids, relevant_chunk_ids
        )
        metrics.retrieval_recall = self.calculate_retrieval_recall(
            retrieved_chunk_ids, relevant_chunk_ids
        )

        # Tier 2: LLM-based metrics
        if self.client:
            metrics.context_relevance = self.calculate_context_relevance(query, context)
            metrics.faithfulness = self.calculate_faithfulness(answer, context)
            metrics.answer_relevance = self.calculate_answer_relevance(query, answer)

        logger.info(f"Calculated metrics: {metrics.to_dict()}")

        return metrics
