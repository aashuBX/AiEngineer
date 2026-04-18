"""Example 02: Multi-Agent Supervisor orchestrating specialist workers."""
from src.config.llm_providers import get_llm
from src.graphs.multi_agent_graph import create_supervisor_graph


def main():
    """Demonstrate supervisor dispatching to specialist agents."""
    llm = get_llm(provider="groq", model="llama-3.3-70b-versatile")
    graph = create_supervisor_graph(llm=llm, worker_names=["researcher", "writer"])

    result = graph.invoke({
        "messages": [("user", "Research the latest advances in AI agents and write a brief summary.")]
    })

    for msg in result["messages"]:
        role = getattr(msg, "name", msg.type)
        print(f"[{role}]: {msg.content[:300]}")


if __name__ == "__main__":
    main()
