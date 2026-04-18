"""
Map-Reduce Graph — Parallel fan-out to multiple agents, then aggregation.
Map: distributes sub-tasks. Reduce: synthesizes all results.
"""

from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.types import Send

from src.models.state import MapReduceState
from src.config.llm_providers import get_default_llm
from src.utils.logger import get_logger

logger = get_logger(__name__)

_DECOMPOSER_PROMPT = """You are a task decomposer. Break the user's request into independent
parallel sub-tasks that can each be solved separately.

Output ONLY a JSON array of strings, each being a distinct sub-task.
Example: ["Sub-task 1: ...", "Sub-task 2: ...", "Sub-task 3: ..."]

Keep sub-tasks focused, non-overlapping, and completable independently.
"""

_REDUCER_PROMPT = """You are a synthesis agent. You have received results from multiple
parallel workers. Combine all results into a single coherent, comprehensive response.

Eliminate redundancy, preserve key insights, and structure the final answer clearly.
"""


def build_map_reduce_graph(
    worker_fn: Any | None = None,
    checkpointer=None,
) -> Any:
    """
    Build a parallel Map-Reduce agent graph.

    Args:
        worker_fn:    Optional custom async worker function(state) -> dict.
                      Defaults to an LLM-based generic worker.
        checkpointer: LangGraph checkpointer.

    Returns:
        Compiled StateGraph.
    """
    llm = get_default_llm()

    # ── Decomposer (Map phase initiator) ───────────────────────────────────
    async def decompose_node(state: MapReduceState) -> dict:
        import json
        user_input = state["messages"][-1].content if state["messages"] else ""
        logger.info(f"MapReduce decomposer: breaking down task: {user_input!r}")

        response = await llm.ainvoke([
            SystemMessage(content=_DECOMPOSER_PROMPT),
            HumanMessage(content=user_input),
        ])
        raw = response.content.strip()
        # Strip markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            sub_tasks = json.loads(raw)
            if not isinstance(sub_tasks, list):
                sub_tasks = [user_input]
        except json.JSONDecodeError:
            # Fallback: one task = the whole input
            sub_tasks = [user_input]

        logger.info(f"MapReduce: {len(sub_tasks)} sub-tasks created")
        return {"sub_tasks": sub_tasks, "sub_results": []}

    # ── Fan-out router: sends each sub-task to a worker ───────────────────
    def fan_out(state: MapReduceState) -> list[Send]:
        return [
            Send("worker", {**state, "sub_tasks": [task], "sub_results": []})
            for task in state.get("sub_tasks", [])
        ]

    # ── Worker node ────────────────────────────────────────────────────────
    async def worker_node(state: MapReduceState) -> dict:
        task = state["sub_tasks"][0] if state.get("sub_tasks") else ""
        logger.debug(f"MapReduce worker: processing: {task!r}")

        if worker_fn:
            return await worker_fn({**state, "messages": [HumanMessage(content=task)]})

        # Default: generic LLM worker
        response = await llm.ainvoke([
            SystemMessage(content="You are a specialist agent. Complete the assigned sub-task thoroughly."),
            HumanMessage(content=task),
        ])
        return {"sub_results": [response.content]}

    # ── Reducer (Reduce phase) ─────────────────────────────────────────────
    async def reduce_node(state: MapReduceState) -> dict:
        results = state.get("sub_results", [])
        logger.info(f"MapReduce reducer: combining {len(results)} worker results")

        if not results:
            return {"final_summary": "No results to synthesize.", "task_status": "done"}

        combined = "\n\n".join(f"Worker {i+1} result:\n{r}" for i, r in enumerate(results))
        response = await llm.ainvoke([
            SystemMessage(content=_REDUCER_PROMPT),
            HumanMessage(content=combined),
        ])
        final = response.content
        return {
            "final_summary": final,
            "task_status": "done",
            "messages": [AIMessage(content=final)],
        }

    # ── Build graph ────────────────────────────────────────────────────────
    builder = StateGraph(MapReduceState)
    builder.add_node("decompose", decompose_node)
    builder.add_node("worker", worker_node)
    builder.add_node("reduce", reduce_node)

    builder.set_entry_point("decompose")
    builder.add_conditional_edges("decompose", fan_out, ["worker"])
    builder.add_edge("worker", "reduce")
    builder.add_edge("reduce", END)

    logger.info("Map-Reduce graph compiled")
    return builder.compile(checkpointer=checkpointer)
