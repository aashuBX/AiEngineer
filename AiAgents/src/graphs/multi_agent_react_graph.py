"""
Multi-Agent ReAct Graph — Reasoning + Acting loop using specialized agents as tools.
Implements the Agent-as-a-Tool orchestration pattern.
"""

from typing import Any
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from src.models.state import SupervisorState
from src.config.llm_providers import get_default_llm
from src.utils.logger import get_logger

logger = get_logger(__name__)

_REACT_SYSTEM_PROMPT = """You are a highly capable Master Orchestrator Agent.
You have access to a team of specialist AI agents represented as tools.
Given a user request, you must reason about which specialist agent(s) to call to fulfill the request.
You can call multiple agents in sequence if needed.

Available specialists:
- faq_agent       → General FAQ and policy questions
- crm_agent       → Customer data, orders, account queries
- rag_agent       → Standard document searches across uploaded manuals/PDFs
- graph_rag_agent → Deep domain knowledge requiring entity connection or graph synthesis
- feedback_agent  → User feedback and ratings
- handoff_agent   → Human escalation requests

Analyze the user's intent, use the appropriate agent tools to gather information, and then synthesize a final response.
"""

class AgentToolInput(BaseModel):
    query: str = Field(description="The query or task to pass to the specialist agent.")

def _make_agent_tool(name: str, process_fn: Any) -> StructuredTool:
    """Wraps an agent's process function into a callable tool."""
    
    async def _tool_func(query: str) -> str:
        logger.info(f"ReAct router calling specialist {name} with query: {query!r}")
        # Build a temporary state for the worker
        worker_state = {"messages": [HumanMessage(content=query)]}
        result = await process_fn(worker_state)
        
        # Extract the last message content
        if "messages" in result and result["messages"]:
            return result["messages"][-1].content
        return "The agent did not return a valid response."

    return StructuredTool.from_function(
        func=None,
        coroutine=_tool_func,
        name=name,
        description=f"Call the {name} to handle queries suited for its domain.",
        args_schema=AgentToolInput,
    )


def build_multi_agent_react_graph(
    worker_nodes: dict[str, Any],
    checkpointer=None,
) -> Any:
    """
    Build a ReAct Multi-Agent graph.
    
    Args:
        worker_nodes: Dict mapping worker name → async callable (state) -> state update.
        checkpointer: LangGraph checkpointer for persistence.

    Returns:
        Compiled StateGraph.
    """
    llm = get_default_llm()
    
    # Convert worker nodes into tools
    agent_tools = []
    for name, fn in worker_nodes.items():
        agent_tools.append(_make_agent_tool(name, fn))
        
    llm_with_tools = llm.bind_tools(agent_tools)

    # ── Agent node ─────────────────────────────────────────────────────────
    async def agent_node(state: SupervisorState) -> dict:
        messages = state["messages"]

        # Prepend system prompt if not already present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=_REACT_SYSTEM_PROMPT)] + list(messages)

        logger.debug(f"Multi-Agent ReAct: invoking LLM with {len(messages)} messages")
        response = await llm_with_tools.ainvoke(messages)
        
        # Determine current agent based on tool calls
        current_agent = "react_orchestrator"
        if response.tool_calls:
            current_agent = response.tool_calls[-1]["name"]
            
        return {"messages": [response], "current_agent": current_agent, "routing_decision": current_agent}

    # ── Tool node ──────────────────────────────────────────────────────────
    tool_node = ToolNode(tools=agent_tools)

    # ── Build graph ────────────────────────────────────────────────────────
    builder = StateGraph(SupervisorState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)

    builder.set_entry_point("agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    logger.info(f"Multi-Agent ReAct graph compiled with tools: {[t.name for t in agent_tools]}")
    return builder.compile(checkpointer=checkpointer)
