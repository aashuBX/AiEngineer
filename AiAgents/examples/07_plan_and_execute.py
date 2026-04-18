"""Example 07: Plan-and-Execute pattern."""
from src.config.llm_providers import get_llm
from src.graphs.plan_execute_graph import create_plan_execute_graph


def main():
    """Demonstrate the Planner-Executor pattern with replanning."""
    llm = get_llm(provider="groq", model="llama-3.3-70b-versatile")
    graph = create_plan_execute_graph(llm=llm)

    result = graph.invoke({
        "messages": [("user", "Create a comprehensive comparison of FAISS vs Qdrant for vector search.")]
    })

    for msg in result["messages"]:
        print(f"[{msg.type}]: {msg.content[:300]}")


if __name__ == "__main__":
    main()
