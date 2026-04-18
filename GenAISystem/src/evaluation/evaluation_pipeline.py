"""
Evaluation Pipeline — End-to-end RAG evaluation runner with test dataset management.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """End-to-end evaluation pipeline for RAG systems.

    Features:
    - Test dataset management (Q&A pairs with ground truth)
    - Multi-evaluator execution (RAGAS, custom metrics)
    - Report generation with metrics summary
    - Results persistence
    """

    def __init__(
        self,
        evaluators: Optional[List] = None,
        rag_pipeline: Optional[Callable] = None,
    ):
        """
        Args:
            evaluators: List of evaluator instances (e.g., RagasEvaluator, CustomMetrics).
            rag_pipeline: A callable that takes a query string and returns
                          {'answer': str, 'context': list, 'sources': list}.
        """
        self.evaluators = evaluators or []
        self.rag_pipeline = rag_pipeline

    # ------------------------------------------------------------------
    # Test Dataset Management
    # ------------------------------------------------------------------

    @staticmethod
    def load_test_dataset(path: str) -> List[Dict[str, Any]]:
        """Load a test dataset from a JSON file.

        Expected format:
        [
            {
                "query": "What is...",
                "ground_truth": "The answer is...",
                "expected_sources": ["doc1.pdf"]  // optional
            }
        ]
        """
        with open(path, "r") as f:
            dataset = json.load(f)
        logger.info(f"Loaded {len(dataset)} test cases from {path}")
        return dataset

    @staticmethod
    def create_test_dataset(
        queries: List[str],
        ground_truths: List[str],
        expected_sources: Optional[List[List[str]]] = None,
        output_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Create a test dataset from parallel lists.

        Args:
            queries: List of test questions.
            ground_truths: List of expected answers.
            expected_sources: Optional list of expected source documents per query.
            output_path: If provided, save the dataset to this path.

        Returns:
            The test dataset as a list of dicts.
        """
        dataset = []
        for i, (q, gt) in enumerate(zip(queries, ground_truths)):
            entry = {"query": q, "ground_truth": gt}
            if expected_sources and i < len(expected_sources):
                entry["expected_sources"] = expected_sources[i]
            dataset.append(entry)

        if output_path:
            with open(output_path, "w") as f:
                json.dump(dataset, f, indent=2)
            logger.info(f"Test dataset saved to {output_path}")

        return dataset

    # ------------------------------------------------------------------
    # Evaluation Execution
    # ------------------------------------------------------------------

    def run_evaluation(
        self,
        dataset: List[Dict[str, Any]],
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run the full evaluation pipeline over a test dataset.

        Args:
            dataset: List of test cases with 'query' and 'ground_truth'.
            verbose: If True, log progress for each test case.

        Returns:
            Comprehensive evaluation report.
        """
        if not self.rag_pipeline:
            raise ValueError("No RAG pipeline configured. Set self.rag_pipeline before running.")

        results = []
        total_start = time.perf_counter()

        for i, test_case in enumerate(dataset):
            query = test_case["query"]
            ground_truth = test_case.get("ground_truth", "")

            if verbose:
                logger.info(f"Evaluating [{i+1}/{len(dataset)}]: {query[:60]}...")

            # Run the RAG pipeline
            case_start = time.perf_counter()
            try:
                rag_result = self.rag_pipeline(query)
            except Exception as e:
                logger.error(f"RAG pipeline failed for query '{query[:40]}': {e}")
                results.append({
                    "query": query,
                    "error": str(e),
                    "metrics": {},
                })
                continue
            case_time = (time.perf_counter() - case_start) * 1000  # ms

            # Collect metrics from each evaluator
            case_metrics = {"latency_ms": round(case_time, 2)}
            answer = rag_result.get("answer", "")
            context = rag_result.get("context", [])

            for evaluator in self.evaluators:
                try:
                    if hasattr(evaluator, "evaluate"):
                        metrics = evaluator.evaluate(
                            query=query,
                            answer=answer,
                            context=context,
                            ground_truth=ground_truth,
                        )
                        case_metrics.update(metrics)
                    elif hasattr(evaluator, "check_hallucination"):
                        context_text = " ".join(
                            doc.get("content", "") for doc in context
                        ) if isinstance(context, list) else str(context)
                        hallucination = evaluator.check_hallucination(answer, context_text, self.evaluators[0])
                        case_metrics["hallucination"] = hallucination
                except Exception as e:
                    logger.warning(f"Evaluator {type(evaluator).__name__} failed: {e}")

            results.append({
                "query": query,
                "ground_truth": ground_truth,
                "answer": answer,
                "metrics": case_metrics,
            })

        total_time = (time.perf_counter() - total_start) * 1000

        # Generate summary report
        report = self._generate_report(results, total_time)

        if verbose:
            self._print_report(report)

        return report

    # ------------------------------------------------------------------
    # Report Generation
    # ------------------------------------------------------------------

    def _generate_report(
        self,
        results: List[Dict[str, Any]],
        total_time_ms: float,
    ) -> Dict[str, Any]:
        """Generate an aggregate evaluation report."""
        num_cases = len(results)
        num_errors = sum(1 for r in results if "error" in r)

        # Aggregate numeric metrics
        all_metrics = {}
        for r in results:
            for key, value in r.get("metrics", {}).items():
                if isinstance(value, (int, float)):
                    all_metrics.setdefault(key, []).append(value)

        averages = {
            f"avg_{key}": round(sum(values) / len(values), 4)
            for key, values in all_metrics.items()
        }

        return {
            "summary": {
                "total_cases": num_cases,
                "successful": num_cases - num_errors,
                "errors": num_errors,
                "total_time_ms": round(total_time_ms, 2),
                "avg_time_per_case_ms": round(total_time_ms / max(num_cases, 1), 2),
            },
            "aggregate_metrics": averages,
            "detailed_results": results,
        }

    @staticmethod
    def _print_report(report: Dict[str, Any]):
        """Pretty-print the evaluation report."""
        summary = report["summary"]
        metrics = report["aggregate_metrics"]

        logger.info("=" * 60)
        logger.info("RAG EVALUATION REPORT")
        logger.info("=" * 60)
        logger.info(f"Total cases: {summary['total_cases']}")
        logger.info(f"Successful:  {summary['successful']}")
        logger.info(f"Errors:      {summary['errors']}")
        logger.info(f"Total time:  {summary['total_time_ms']:.0f}ms")
        logger.info(f"Avg/case:    {summary['avg_time_per_case_ms']:.0f}ms")
        logger.info("-" * 60)
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save_report(report: Dict[str, Any], path: str):
        """Save the evaluation report to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Evaluation report saved to {path}")
