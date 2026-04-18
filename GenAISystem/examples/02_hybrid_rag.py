"""Example 02: Hybrid RAG — Vector + BM25 Keyword retrieval with fusion."""
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.keyword_retriever import KeywordRetriever
from src.retrieval.hybrid_retriever import HybridRetriever


def main():
    sample_docs = [
        "FAISS is a library for efficient similarity search developed by Meta.",
        "Qdrant is an open-source vector database with filtering and hybrid search.",
        "ChromaDB provides a simple, developer-friendly vector store API.",
        "Pinecone is a managed vector database service for production workloads.",
        "BM25 is a probabilistic retrieval model used in keyword search engines.",
    ]

    # Setup keyword retriever
    keyword = KeywordRetriever(documents=sample_docs)

    # Run keyword search
    print("=== Keyword (BM25) Results ===")
    results = keyword.retrieve("vector database filtering", top_k=3)
    for r in results:
        print(f"  [{r['score']:.3f}] {r['content'][:80]}")

    print("\nFor full hybrid retrieval, configure a vector store with embeddings.")


if __name__ == "__main__":
    main()
