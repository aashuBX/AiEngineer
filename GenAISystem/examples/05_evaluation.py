"""Example 05: RAG Evaluation with RAGAS and custom metrics."""
from src.evaluation.evaluation_pipeline import EvaluationPipeline
from src.evaluation.custom_metrics import CustomMetrics


def main():
    print("RAG Evaluation Demo")
    print("=" * 50)

    # Create a test dataset
    dataset = EvaluationPipeline.create_test_dataset(
        queries=[
            "What is machine learning?",
            "Explain the transformer architecture.",
            "What is FAISS used for?",
        ],
        ground_truths=[
            "Machine learning is a subset of AI that learns from data.",
            "Transformers use self-attention mechanisms for sequence processing.",
            "FAISS is a library for efficient similarity search.",
        ],
        output_path="test_dataset.json",
    )
    print(f"Created test dataset with {len(dataset)} cases")

    # Latency measurement example
    result = CustomMetrics.measure_latency(lambda: sum(range(1000000)))
    print(f"Latency example: {result['latency_ms']}ms")


if __name__ == "__main__":
    main()
