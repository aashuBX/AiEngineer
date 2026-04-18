"""
Multi-Agent Supervisor Graph — Supervisor dispatches to specialist worker agents.
Implements the supervisor-worker orchestration pattern with conditional routing.
"""

from typing import Any, Literal

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.types import Command

from src.models.state import SupervisorState
from src.config.llm_providers import get_default_llm
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Available specialist workers
WORKERS = ["faq_agent", "crm_agent", "graph_rag_agent", "feedback_agent", "handoff_agent"]

_SUPERVISOR_PROMPT = """You are a Supervisor Agent coordinating a team of specialist AI agents.

Available workers:
- faq_agent       → General FAQ and policy questions
- crm_agent       → Customer data, orders, account queries
- graph_rag_agent → Deep domain knowledge requiring document synthesis
- feedback_agent  → User feedback and ratings
- handoff_agent   → Human escalation requests

Given the conversation history and current intent, decide which worker to call next.
If the task is complete, respond with FINISH.

Respond ONLY with the worker name or "FINISH".
"""


def build_supervisor_graph(
    worker_nodes: dict[str, Any],
    checkpointer=None,
) -> Any:
    """
    Build a Supervisor→Worker multi-agent graph.

    Args:
        worker_nodes: Dict mapping worker name → async callable (state) -> state update.
        checkpointer: LangGraph checkpointer for persistence.

    Returns:
        Compiled StateGraph.
    """
    llm = get_default_llm()

    # ── Supervisor node ────────────────────────────────────────────────────
    async def supervisor_node(state: SupervisorState) -> Command[Literal[*list(worker_nodes.keys()), "__end__"]]:
        messages = [
            SystemMessage(content=_SUPERVISOR_PROMPT),
            *state["messages"],
        ]
        # Add context about completed workers
        if state.get("worker_responses"):
            summary = "\n".join(
                f"- {w['worker']}: {w['result'][:100]}..."
                for w in state["worker_responses"]
            )
            messages.append(HumanMessage(content=f"Workers already called:\n{summary}"))

        response = await llm.ainvoke(messages)
        decision = response.content.strip().lower()

        logger.info(f"Supervisor decision: {decision!r}")

        if decision == "finish" or state.get("iteration_count", 0) >= 5:
            return Command(goto=END)

        if decision in worker_nodes:
            return Command(
                goto=decision,
                update={
                    "routing_decision": decision,
                    "iteration_count": state.get("iteration_count", 0) + 1,
                    "current_agent": "supervisor",
                },
            )

        logger.warning(f"Supervisor returned unknown worker: {decision!r} — finishing")
        return Command(goto=END)

    # ── Build graph ────────────────────────────────────────────────────────
    builder = StateGraph(SupervisorState)
    builder.add_node("supervisor", supervisor_node)

    for name, fn in worker_nodes.items():
        async def _worker_wrapper(state: SupervisorState, _fn=fn, _name=name) -> Command:
            result = await _fn(state)
            updated_responses = list(state.get("worker_responses", []))
            last_msg = result.get("messages", [])
            updated_responses.append({
                "worker": _name,
                "result": last_msg[-1].content if last_msg else "",
            })
            return Command(
                goto="supervisor",
                update={**result, "worker_responses": updated_responses},
            )
        builder.add_node(name, _worker_wrapper)

    builder.set_entry_point("supervisor")

    logger.info(f"Supervisor graph compiled with workers: {list(worker_nodes.keys())}")
    return builder.compile(checkpointer=checkpointer)
