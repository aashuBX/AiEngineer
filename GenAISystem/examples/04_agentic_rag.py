"""Example 04: Agentic RAG — Self-RAG with hallucination verification."""
from src.agentic_rag.self_rag import SelfRAG


def main():
    print("Agentic RAG Demo (Self-RAG)")
    print("=" * 50)
    print("Self-RAG: Retrieve → Grade → Generate → Self-Verify")
    print()
    print("This example requires:")
    print("  1. A configured LLM (Groq, OpenAI, etc.)")
    print("  2. A vector store with indexed documents")
    print()
    print("Usage:")
    print("  from src.agentic_rag.self_rag import SelfRAG")
    print("  rag = SelfRAG(retriever=my_retriever, llm=my_llm)")
    print("  result = rag.invoke('What is attention mechanism?')")
    print("  print(result['answer'])")


if __name__ == "__main__":
    main()
