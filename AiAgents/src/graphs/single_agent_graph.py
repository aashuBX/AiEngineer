"""
Single Agent ReAct Graph — Reasoning + Acting loop using LangGraph prebuilt.
The simplest agentic pattern: Agent → tools_condition → ToolNode → Agent → ...
"""

from typing import Any

from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from src.models.state import AgentState
from src.config.llm_providers import get_default_llm
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_react_graph(
    tools: list[BaseTool],
    llm: BaseChatModel | None = None,
    system_prompt: str = "You are a helpful AI assistant with access to tools.",
    checkpointer=None,
) -> Any:
    """
    Build a ReAct (Reason + Act) agent graph.

    Args:
        tools:         List of LangChain tools available to the agent.
        llm:           LLM to use. Defaults to get_default_llm().
        system_prompt: System-level instruction for the agent.
        checkpointer:  LangGraph checkpointer for persistence.

    Returns:
        Compiled LangGraph StateGraph.
    """
    _llm = llm or get_default_llm()
    _llm_with_tools = _llm.bind_tools(tools)

    # ── Agent node ─────────────────────────────────────────────────────────
    async def agent_node(state: AgentState) -> dict:
        from langchain_core.messages import SystemMessage
        messages = state["messages"]

        # Prepend system prompt if not already present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + list(messages)

        logger.debug(f"ReAct agent: invoking LLM with {len(messages)} messages")
        response = await _llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    # ── Tool node ──────────────────────────────────────────────────────────
    tool_node = ToolNode(tools=tools)

    # ── Build graph ────────────────────────────────────────────────────────
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)

    builder.set_entry_point("agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    logger.info(f"ReAct graph compiled with {len(tools)} tools")
    return builder.compile(checkpointer=checkpointer)


def build_simple_react_agent(tools: list[BaseTool], system_prompt: str = "") -> Any:
    """Convenience wrapper using LangGraph's prebuilt create_react_agent."""
    from langgraph.prebuilt import create_react_agent
    llm = get_default_llm()
    return create_react_agent(
        llm,
        tools=tools,
        state_modifier=system_prompt or None,
    )
