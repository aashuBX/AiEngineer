"""Example 06: RAG Agent querying GenAISystem."""
from src.config.llm_providers import get_llm
from src.agents.graph_rag.graph_rag_agent import GraphRAGAgent


def main():
    """Demonstrate an agent that queries the GenAISystem for RAG-based answers."""
    llm = get_llm(provider="groq", model="llama-3.3-70b-versatile")

    agent = GraphRAGAgent(llm=llm)
    print("RAG Agent Demo")
    print("=" * 50)
    print("Ensure GenAISystem API is running on port 8002")
    print("Then query: agent.invoke('What is transformer architecture?')")


if __name__ == "__main__":
    main()
