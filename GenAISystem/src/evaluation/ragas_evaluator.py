"""
RAGAS Evaluator for RAG Pipeline Quality.
 Evaluates Faithfulness, Answer Relevance, and Context Precision.
"""

from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RagasEvaluator:
    """Evaluates RAG results using the RAGAS framework."""

    def __init__(self, llm: Any = None, embeddings: Any = None):
        self.llm = llm
        self.embeddings = embeddings

    def evaluate_response(
        self, 
        question: str, 
        answer: str, 
        contexts: list[str], 
        ground_truth: str | None = None
    ) -> dict[str, float]:
        """
        Evaluate a single RAG interaction.
        """
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy, context_precision

            logger.info("Running RAGAS evaluation...")

            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            }
            if ground_truth:
                data["ground_truths"] = [[ground_truth]]

            dataset = Dataset.from_dict(data)

            # Default metrics
            metrics = [faithfulness, answer_relevancy, context_precision]

            # RAGAS evaluate expects LangChain models to be wrapped
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=self.llm,
                embeddings=self.embeddings,
                raise_exceptions=False,
            )

            return dict(result)

        except ImportError:
            logger.warning("ragas or datasets library not installed.")
            return {"error": -1.0}
        except Exception as e:
            logger.error(f"RAGAS Evaluation failed: {e}")
            return {"error": -1.0}
