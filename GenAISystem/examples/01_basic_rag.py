"""Example 01: Basic RAG — Ingest → Embed → Store → Query."""
from src.ingestion.chunking.recursive_chunker import RecursiveChunker
from src.embeddings.embedding_factory import EmbeddingFactory
from src.vectorstores.store_factory import VectorStoreFactory
from src.retrieval.vector_retriever import VectorRetriever
from src.generation.response_generator import ResponseGenerator


def main():
    sample_text = """
    Machine learning is a subset of artificial intelligence that focuses on building
    systems that can learn from data. Deep learning is a subset of machine learning
    that uses neural networks with multiple layers. Transformers are a type of
    neural network architecture introduced in the paper "Attention Is All You Need."
    """

    # 1. Chunk
    chunker = RecursiveChunker(chunk_size=200, overlap=50)
    chunks = chunker.split_text(sample_text)
    print(f"Created {len(chunks)} chunks")

    # 2. Embed + Store
    embeddings = EmbeddingFactory.get_embeddings("sentence-transformers", "minilm")
    store = VectorStoreFactory.get_store("faiss", embedding_function=embeddings)
    store.add_documents(chunks)
    print(f"Stored {store.count()} documents")

    # 3. Retrieve + Generate
    retriever = VectorRetriever(store, top_k=3)
    results = retriever.retrieve("What are transformers?")
    print(f"Retrieved {len(results)} relevant documents")

    for r in results:
        print(f"  [{r['score']:.3f}] {r['content'][:80]}...")


if __name__ == "__main__":
    main()
