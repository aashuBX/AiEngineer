"""Tests for evaluation modules — custom metrics, evaluation pipeline."""
import pytest
from unittest.mock import MagicMock


class TestCustomMetrics:
    """Test custom evaluation metrics."""

    def test_latency_measurement(self):
        from src.evaluation.custom_metrics import CustomMetrics
        result = CustomMetrics.measure_latency(lambda: sum(range(1000)))
        assert "latency_ms" in result
        assert result["latency_ms"] >= 0
        assert result["result"] == sum(range(1000))

    def test_citation_accuracy_with_valid_citations(self):
        from src.evaluation.custom_metrics import CustomMetrics
        answer = "According to [1], FAISS is efficient. As stated in [2], it uses L2 distance."
        sources = [{"index": 1}, {"index": 2}]
        context = [{"content": "doc1"}, {"content": "doc2"}]
        result = CustomMetrics.check_citation_accuracy(answer, sources, context)
        assert result["citation_precision"] == 1.0
        assert result["citation_recall"] == 1.0

    def test_citation_accuracy_with_invalid_citations(self):
        from src.evaluation.custom_metrics import CustomMetrics
        answer = "According to [5], something is true."
        sources = []
        context = [{"content": "doc1"}]
        result = CustomMetrics.check_citation_accuracy(answer, sources, context)
        assert result["citation_precision"] == 0.0
        assert 5 in result["invalid_citations"]

    def test_latency_report(self):
        from src.evaluation.custom_metrics import CustomMetrics
        report = CustomMetrics.retrieval_latency_report(100, 200, 320)
        assert report["retrieval_ms"] == 100
        assert report["generation_ms"] == 200
        assert report["overhead_ms"] == 20


class TestEvaluationPipeline:
    """Test evaluation pipeline."""

    def test_pipeline_initialization(self):
        from src.evaluation.evaluation_pipeline import EvaluationPipeline
        pipeline = EvaluationPipeline()
        assert pipeline is not None

    def test_create_test_dataset(self):
        from src.evaluation.evaluation_pipeline import EvaluationPipeline
        dataset = EvaluationPipeline.create_test_dataset(
            queries=["q1", "q2"],
            ground_truths=["a1", "a2"],
        )
        assert len(dataset) == 2
        assert dataset[0]["query"] == "q1"
        assert dataset[0]["ground_truth"] == "a1"

    def test_pipeline_requires_rag_pipeline(self):
        from src.evaluation.evaluation_pipeline import EvaluationPipeline
        pipeline = EvaluationPipeline()
        with pytest.raises(ValueError, match="No RAG pipeline"):
            pipeline.run_evaluation([{"query": "test", "ground_truth": "test"}])
