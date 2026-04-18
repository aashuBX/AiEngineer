"""Example 06: Full Pipeline — End-to-end ingestion, embedding, storage, and query."""
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.data_loader import DataLoader


def main():
    print("Full Pipeline Demo")
    print("=" * 50)
    print()
    print("This example walks through the complete RAG pipeline:")
    print("  1. Load documents (PDF, TXT, CSV, etc.)")
    print("  2. Preprocess (clean, extract metadata)")
    print("  3. Chunk (recursive, semantic, agentic, or document-structure)")
    print("  4. Embed (OpenAI, SentenceTransformers, etc.)")
    print("  5. Store (FAISS, Qdrant, ChromaDB, Pinecone)")
    print("  6. Query with hybrid retrieval")
    print("  7. Generate response with citations")
    print("  8. Evaluate with RAGAS + custom metrics")
    print()

    # Quick data loader demo
    loader = DataLoader()
    print("Supported formats:", ["pdf", "txt", "csv", "docx", "json", "html", "md"])
    print("Place your documents in GenAISystem/data/ and run the pipeline.")


if __name__ == "__main__":
    main()
