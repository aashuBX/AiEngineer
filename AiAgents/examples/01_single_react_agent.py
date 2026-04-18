"""Example 01: Single ReAct Agent with Tools."""
from src.config.llm_providers import get_llm
from src.graphs.single_agent_graph import create_react_agent_graph


def main():
    """Demonstrate a basic ReAct agent with tool-calling."""
    llm = get_llm(provider="groq", model="llama-3.3-70b-versatile")

    # Create a simple ReAct agent (no external tools for this demo)
    graph = create_react_agent_graph(llm=llm, tools=[])

    # Run a conversation
    result = graph.invoke({
        "messages": [("user", "What are the key concepts in reinforcement learning?")]
    })

    for msg in result["messages"]:
        print(f"[{msg.type}]: {msg.content[:200]}")


if __name__ == "__main__":
    main()
