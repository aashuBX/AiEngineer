"""
Plan-and-Execute Graph — Planner generates a step-by-step plan,
Executor runs each step, Replanner adjusts based on results.
"""

from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from src.models.state import PlanExecuteState
from src.config.llm_providers import get_default_llm
from src.utils.logger import get_logger

logger = get_logger(__name__)

_PLANNER_PROMPT = """You are a task planner. Given the user's objective, create a clear,
numbered step-by-step plan to accomplish it. Each step should be a single,
actionable task. Output ONLY the numbered list, nothing else.

Example:
1. Search for recent information about X
2. Summarize the key findings
3. Draft a response based on the summary
"""

_EXECUTOR_PROMPT = """You are a task executor. You will be given a specific step from a plan.
Execute it to the best of your ability using available tools and knowledge.
Provide a clear, concise result for this step only.
"""

_REPLANNER_PROMPT = """You are a task replanner. Review the original objective, the plan,
and the steps completed so far with their results.

Decide:
1. If the objective is fully achieved → respond with: COMPLETE: <final answer>
2. If adjustments are needed → respond with the REMAINING steps as a new numbered list

Output ONLY the decision, nothing else.
"""


def build_plan_execute_graph(
    executor_tools: list | None = None,
    checkpointer=None,
) -> Any:
    """
    Build a Plan → Execute → Replan workflow graph.

    Args:
        executor_tools: Optional tools for the executor agent.
        checkpointer:   LangGraph checkpointer.

    Returns:
        Compiled StateGraph.
    """
    llm = get_default_llm()
    exec_llm = llm.bind_tools(executor_tools) if executor_tools else llm

    # ── Planner node ───────────────────────────────────────────────────────
    async def planner_node(state: PlanExecuteState) -> dict:
        user_goal = state["messages"][-1].content if state["messages"] else ""
        logger.info(f"Planner: creating plan for: {user_goal!r}")

        response = await llm.ainvoke([
            SystemMessage(content=_PLANNER_PROMPT),
            HumanMessage(content=f"Objective: {user_goal}"),
        ])
        raw_plan = response.content.strip()

        # Parse numbered list into steps
        steps = []
        for line in raw_plan.split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                # Remove leading "1. ", "2. " etc.
                step_text = line.split(".", 1)[-1].strip()
                if step_text:
                    steps.append(step_text)

        logger.info(f"Planner: created {len(steps)} steps")
        return {"plan": steps, "current_step": 0, "past_steps": [], "execution_results": []}

    # ── Executor node ──────────────────────────────────────────────────────
    async def executor_node(state: PlanExecuteState) -> dict:
        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)

        if current_step >= len(plan):
            return {"task_status": "done"}

        step = plan[current_step]
        logger.info(f"Executor: running step {current_step + 1}/{len(plan)}: {step!r}")

        # Build context from past results
        past_context = ""
        for past_step, past_result in state.get("past_steps", []):
            past_context += f"Step: {past_step}\nResult: {past_result}\n\n"

        response = await exec_llm.ainvoke([
            SystemMessage(content=_EXECUTOR_PROMPT),
            HumanMessage(content=f"{past_context}Current step: {step}"),
        ])
        result = response.content

        past_steps = list(state.get("past_steps", []))
        past_steps.append((step, result))
        exec_results = list(state.get("execution_results", []))
        exec_results.append(result)

        return {
            "past_steps": past_steps,
            "execution_results": exec_results,
            "current_step": current_step + 1,
            "messages": [AIMessage(content=f"Step {current_step + 1} result: {result}")],
        }

    # ── Replanner node ─────────────────────────────────────────────────────
    async def replanner_node(state: PlanExecuteState) -> dict:
        goal = state["messages"][0].content if state["messages"] else ""
        past_summary = "\n".join(
            f"Step: {s}\nResult: {r}" for s, r in state.get("past_steps", [])
        )
        remaining_plan = "\n".join(
            f"{i+1}. {s}" for i, s in enumerate(state.get("plan", [])[state.get("current_step", 0):])
        )

        response = await llm.ainvoke([
            SystemMessage(content=_REPLANNER_PROMPT),
            HumanMessage(content=(
                f"Objective: {goal}\n\n"
                f"Completed steps:\n{past_summary}\n\n"
                f"Remaining planned steps:\n{remaining_plan}"
            )),
        ])
        decision = response.content.strip()

        if decision.startswith("COMPLETE:"):
            final = decision[len("COMPLETE:"):].strip()
            logger.info("Replanner: task complete")
            return {
                "final_response": final,
                "task_status": "done",
                "messages": [AIMessage(content=final)],
            }

        # Parse new remaining steps
        new_steps = []
        for line in decision.split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                new_steps.append(line.split(".", 1)[-1].strip())

        logger.info(f"Replanner: {len(new_steps)} steps remaining")
        return {"plan": new_steps, "current_step": 0}

    # ── Routing ────────────────────────────────────────────────────────────
    def should_continue(state: PlanExecuteState) -> str:
        if state.get("task_status") == "done":
            return "end"
        if state.get("current_step", 0) >= len(state.get("plan", [])):
            return "replan"
        return "execute"

    def after_replan(state: PlanExecuteState) -> str:
        if state.get("task_status") == "done" or not state.get("plan"):
            return "end"
        return "execute"

    # ── Build graph ────────────────────────────────────────────────────────
    builder = StateGraph(PlanExecuteState)
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("replanner", replanner_node)

    builder.set_entry_point("planner")
    builder.add_edge("planner", "executor")
    builder.add_conditional_edges("executor", should_continue, {
        "execute": "executor",
        "replan": "replanner",
        "end": END,
    })
    builder.add_conditional_edges("replanner", after_replan, {
        "execute": "executor",
        "end": END,
    })

    logger.info("Plan-and-Execute graph compiled")
    return builder.compile(checkpointer=checkpointer)
