"""RAG system evaluator."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from .metrics import MetricsCalculator, RAGMetrics
from .synthetic_data import SyntheticDataGenerator


class RAGEvaluator:
    """
    Comprehensive RAG system evaluator.

    Following Jason Liu's methodology:
    1. Generate synthetic data (5 questions/chunk, target 97% recall)
    2. Run evaluation suite in <5 minutes
    3. Measure retrieval before generation
    4. Segment by query type
    5. Iterate based on data
    """

    def __init__(
        self,
        synthetic_generator: Optional[SyntheticDataGenerator] = None,
        metrics_calculator: Optional[MetricsCalculator] = None,
    ):
        """
        Initialize RAG evaluator.

        Args:
            synthetic_generator: Generator for synthetic data
            metrics_calculator: Calculator for metrics
        """
        self.synthetic_generator = synthetic_generator or SyntheticDataGenerator()
        self.metrics_calculator = metrics_calculator or MetricsCalculator()

        logger.info("Initialized RAGEvaluator")

    def evaluate_retrieval(
        self,
        test_dataset: List[Dict[str, Any]],
        retrieve_fn: callable,
        k: int = 5,
    ) -> Dict[str, float]:
        """
        Evaluate retrieval quality on a test dataset.

        Args:
            test_dataset: List of dicts with 'question' and 'chunk_id'
            retrieve_fn: Function that takes query and returns list of chunk IDs
            k: Number of results to retrieve

        Returns:
            Dictionary with retrieval metrics
        """
        logger.info(f"Evaluating retrieval on {len(test_dataset)} questions")

        precision_scores = []
        recall_scores = []
        recall_at_k = []

        for item in test_dataset:
            question = item["question"]
            expected_chunk_id = item["chunk_id"]

            # Retrieve chunks
            try:
                retrieved_chunk_ids = retrieve_fn(question, k=k)

                # Calculate precision and recall
                precision = self.metrics_calculator.calculate_retrieval_precision(
                    retrieved_chunk_ids,
                    [expected_chunk_id]
                )
                recall = self.metrics_calculator.calculate_retrieval_recall(
                    retrieved_chunk_ids,
                    [expected_chunk_id]
                )

                precision_scores.append(precision)
                recall_scores.append(recall)

                # Recall@k: was the expected chunk in top k?
                recall_at_k.append(1.0 if expected_chunk_id in retrieved_chunk_ids else 0.0)

            except Exception as e:
                logger.error(f"Error retrieving for question '{question}': {e}")
                continue

        # Calculate averages
        avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
        avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
        avg_recall_at_k = sum(recall_at_k) / len(recall_at_k) if recall_at_k else 0.0

        results = {
            "precision": avg_precision,
            "recall": avg_recall,
            f"recall@{k}": avg_recall_at_k,
            "num_evaluated": len(precision_scores)
        }

        logger.info(f"Retrieval Results: {results}")

        return results

    def evaluate_end_to_end(
        self,
        test_dataset: List[Dict[str, Any]],
        rag_fn: callable,
    ) -> Dict[str, Any]:
        """
        Evaluate end-to-end RAG pipeline.

        Args:
            test_dataset: List of dicts with 'question' and expected data
            rag_fn: Function that takes query and returns dict with:
                   - answer: generated answer
                   - context: retrieved context
                   - chunk_ids: retrieved chunk IDs

        Returns:
            Aggregated evaluation metrics
        """
        logger.info(f"Evaluating end-to-end RAG on {len(test_dataset)} questions")

        all_metrics = []

        for item in test_dataset:
            question = item["question"]
            expected_chunk_id = item.get("chunk_id")

            try:
                # Run RAG
                result = rag_fn(question)

                answer = result.get("answer", "")
                context = result.get("context", "")
                retrieved_chunk_ids = result.get("chunk_ids", [])

                # Calculate metrics
                metrics = RAGMetrics()

                # Retrieval metrics (if ground truth available)
                if expected_chunk_id:
                    metrics.retrieval_precision = (
                        self.metrics_calculator.calculate_retrieval_precision(
                            retrieved_chunk_ids,
                            [expected_chunk_id]
                        )
                    )
                    metrics.retrieval_recall = (
                        self.metrics_calculator.calculate_retrieval_recall(
                            retrieved_chunk_ids,
                            [expected_chunk_id]
                        )
                    )

                # Generation metrics
                if context and answer:
                    metrics.context_relevance = (
                        self.metrics_calculator.calculate_context_relevance(
                            question, context
                        )
                    )
                    metrics.faithfulness = (
                        self.metrics_calculator.calculate_faithfulness(
                            answer, context
                        )
                    )
                    metrics.answer_relevance = (
                        self.metrics_calculator.calculate_answer_relevance(
                            question, answer
                        )
                    )

                all_metrics.append(metrics)

            except Exception as e:
                logger.error(f"Error evaluating question '{question}': {e}")
                continue

        # Aggregate metrics
        aggregated = self._aggregate_metrics(all_metrics)

        logger.info(f"End-to-End Results: {aggregated}")

        return aggregated

    def _aggregate_metrics(self, metrics_list: List[RAGMetrics]) -> Dict[str, float]:
        """Aggregate metrics across multiple evaluations."""
        if not metrics_list:
            return {}

        aggregated = {
            "retrieval_precision": 0.0,
            "retrieval_recall": 0.0,
            "context_relevance": 0.0,
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
        }

        for metrics in metrics_list:
            for key in aggregated.keys():
                aggregated[key] += getattr(metrics, key, 0.0)

        # Average
        n = len(metrics_list)
        for key in aggregated.keys():
            aggregated[key] /= n

        aggregated["num_evaluated"] = n

        return aggregated

    def save_results(
        self,
        results: Dict[str, Any],
        output_path: Path,
    ) -> None:
        """
        Save evaluation results to file.

        Args:
            results: Evaluation results dictionary
            output_path: Path to save results
        """
        import json

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved evaluation results to {output_path}")

    def create_leaderboard(
        self,
        results_list: List[Dict[str, Any]],
        names: List[str],
    ) -> pd.DataFrame:
        """
        Create leaderboard comparing different configurations.

        Args:
            results_list: List of evaluation results
            names: Names for each configuration

        Returns:
            DataFrame with comparison
        """
        data = []

        for name, results in zip(names, results_list):
            row = {"name": name}
            row.update(results)
            data.append(row)

        df = pd.DataFrame(data)

        # Sort by overall score (weighted average)
        if "faithfulness" in df.columns and "answer_relevance" in df.columns:
            df["overall_score"] = (
                0.3 * df.get("retrieval_recall", 0) +
                0.3 * df.get("faithfulness", 0) +
                0.4 * df.get("answer_relevance", 0)
            )
            df = df.sort_values("overall_score", ascending=False)

        return df
