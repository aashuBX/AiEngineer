"""
Custom Metrics — Latency tracking, hallucination rate (model-as-judge), and citation accuracy.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class CustomMetrics:
    """Custom RAG evaluation metrics beyond standard RAGAS."""

    # ------------------------------------------------------------------
    # Latency Tracking
    # ------------------------------------------------------------------

    @staticmethod
    def measure_latency(func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Measure execution time of a function.

        Returns:
            Dict with 'result', 'latency_seconds', 'latency_ms'.
        """
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        return {
            "result": result,
            "latency_seconds": round(elapsed, 4),
            "latency_ms": round(elapsed * 1000, 2),
        }

    @staticmethod
    async def measure_async_latency(coro) -> Dict[str, Any]:
        """Measure execution time of an async coroutine."""
        import asyncio
        start = time.perf_counter()
        result = await coro
        elapsed = time.perf_counter() - start

        return {
            "result": result,
            "latency_seconds": round(elapsed, 4),
            "latency_ms": round(elapsed * 1000, 2),
        }

    # ------------------------------------------------------------------
    # Hallucination Detection (LLM-as-Judge)
    # ------------------------------------------------------------------

    HALLUCINATION_PROMPT = """You are a factual accuracy judge. Determine whether the given
answer is fully supported by the provided context.

Context:
\"\"\"
{context}
\"\"\"

Answer:
\"\"\"
{answer}
\"\"\"

Evaluate:
1. Is every claim in the answer supported by the context? (yes/no)
2. Are there any fabricated facts not found in the context? (yes/no)
3. Rate the hallucination risk from 0.0 (no hallucination) to 1.0 (completely hallucinated).

Respond in JSON format:
{{
    "all_claims_supported": true/false,
    "has_fabricated_facts": true/false,
    "hallucination_score": 0.0-1.0,
    "unsupported_claims": ["list of unsupported claims if any"]
}}
"""

    @staticmethod
    def check_hallucination(
        answer: str,
        context: str,
        llm,
    ) -> Dict[str, Any]:
        """Use an LLM as a judge to detect hallucination.

        Args:
            answer: The generated answer to evaluate.
            context: The retrieved context used for generation.
            llm: LangChain ChatModel to act as judge.

        Returns:
            Dict with hallucination assessment.
        """
        prompt = CustomMetrics.HALLUCINATION_PROMPT.format(
            context=context,
            answer=answer,
        )

        try:
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)

            # Try to parse JSON from the response
            import json
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "hallucination_score": result.get("hallucination_score", 0.5),
                    "all_claims_supported": result.get("all_claims_supported", False),
                    "has_fabricated_facts": result.get("has_fabricated_facts", True),
                    "unsupported_claims": result.get("unsupported_claims", []),
                }
        except Exception as e:
            logger.warning(f"Hallucination check failed: {e}")

        return {
            "hallucination_score": 0.5,
            "all_claims_supported": False,
            "has_fabricated_facts": False,
            "unsupported_claims": [],
            "error": "Failed to parse LLM judge response",
        }

    # ------------------------------------------------------------------
    # Citation Accuracy
    # ------------------------------------------------------------------

    @staticmethod
    def check_citation_accuracy(
        answer: str,
        sources: List[Dict[str, Any]],
        context: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate whether inline citations [1], [2] in the answer correctly
        reference the source documents.

        Args:
            answer: The generated answer with inline citations.
            sources: List of source metadata dicts.
            context: List of retrieved context documents.

        Returns:
            Citation accuracy metrics.
        """
        import re

        cited_indices = set(int(m) for m in re.findall(r'\[(\d+)\]', answer))
        total_sources = len(context)

        valid = cited_indices & set(range(1, total_sources + 1))
        invalid = cited_indices - set(range(1, total_sources + 1))
        uncited = set(range(1, total_sources + 1)) - cited_indices

        precision = len(valid) / max(len(cited_indices), 1)
        recall = len(valid) / max(total_sources, 1)

        return {
            "citation_precision": round(precision, 3),
            "citation_recall": round(recall, 3),
            "cited_sources": list(valid),
            "invalid_citations": list(invalid),
            "uncited_sources": list(uncited),
            "total_citations": len(cited_indices),
            "total_sources": total_sources,
        }

    # ------------------------------------------------------------------
    # Retrieval Quality
    # ------------------------------------------------------------------

    @staticmethod
    def retrieval_latency_report(
        retrieval_time_ms: float,
        generation_time_ms: float,
        total_time_ms: float,
    ) -> Dict[str, Any]:
        """Generate a latency breakdown report.

        Args:
            retrieval_time_ms: Time spent retrieving documents.
            generation_time_ms: Time spent generating the response.
            total_time_ms: End-to-end total time.

        Returns:
            Formatted latency report.
        """
        overhead = total_time_ms - retrieval_time_ms - generation_time_ms

        return {
            "retrieval_ms": round(retrieval_time_ms, 2),
            "generation_ms": round(generation_time_ms, 2),
            "overhead_ms": round(overhead, 2),
            "total_ms": round(total_time_ms, 2),
            "retrieval_pct": round(retrieval_time_ms / max(total_time_ms, 1) * 100, 1),
            "generation_pct": round(generation_time_ms / max(total_time_ms, 1) * 100, 1),
        }
