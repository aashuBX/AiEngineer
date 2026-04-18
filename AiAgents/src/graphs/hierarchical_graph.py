"""
Hierarchical Multi-Agent Graph — Teams of agents with sub-supervisors.
Top-level supervisor coordinates between specialized teams (Research, Analysis, etc.)
"""

from typing import Any, Literal

from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.types import Command

from src.models.state import TeamState, SupervisorState
from src.config.llm_providers import get_default_llm
from src.utils.logger import get_logger

logger = get_logger(__name__)

_TOP_SUPERVISOR_PROMPT = """You are the Top-Level Supervisor coordinating multiple specialist teams.

Available teams:
- research_team   → Information gathering, web search, document retrieval
- analysis_team   → Data analysis, summarization, synthesis

Choose which team to activate next, or FINISH if the task is complete.
Respond ONLY with: research_team | analysis_team | FINISH
"""

_TEAM_SUPERVISOR_PROMPT = """You are a Team Supervisor managing a small team of specialist agents.

Team members available: {members}

Select the next team member to act, or FINISH if your team's task is complete.
Respond ONLY with the member name or FINISH.
"""


def _build_team_graph(
    team_name: str,
    members: dict[str, Any],
) -> Any:
    """Build a sub-graph for a single team with its own supervisor."""
    llm = get_default_llm()
    member_names = list(members.keys())

    async def team_supervisor(state: TeamState) -> Command:
        prompt = _TEAM_SUPERVISOR_PROMPT.format(members=", ".join(member_names))
        response = await llm.ainvoke([
            SystemMessage(content=prompt),
            *state["messages"],
        ])
        decision = response.content.strip().lower()

        if decision == "finish" or decision not in members:
            return Command(goto=END, update={"next_worker": None})

        return Command(goto=decision, update={"next_worker": decision})

    builder = StateGraph(TeamState)
    builder.add_node("team_supervisor", team_supervisor)

    for name, fn in members.items():
        async def _wrapper(state: TeamState, _fn=fn, _name=name) -> Command:
            result = await _fn(state)
            return Command(goto="team_supervisor", update=result)
        builder.add_node(name, _wrapper)

    builder.set_entry_point("team_supervisor")
    logger.info(f"Team graph '{team_name}' compiled with members: {member_names}")
    return builder.compile()


def build_hierarchical_graph(
    research_members: dict[str, Any],
    analysis_members: dict[str, Any],
    checkpointer=None,
) -> Any:
    """
    Build a hierarchical multi-agent graph with two teams.

    Args:
        research_members:  Dict of name → async callable for the research team.
        analysis_members:  Dict of name → async callable for the analysis team.
        checkpointer:      LangGraph checkpointer.

    Returns:
        Compiled top-level StateGraph.
    """
    llm = get_default_llm()

    # Compile sub-graphs
    research_graph = _build_team_graph("research_team", research_members)
    analysis_graph = _build_team_graph("analysis_team", analysis_members)

    # ── Top-level supervisor ───────────────────────────────────────────────
    async def top_supervisor(state: SupervisorState) -> Command[Literal["research_team", "analysis_team", "__end__"]]:
        response = await llm.ainvoke([
            SystemMessage(content=_TOP_SUPERVISOR_PROMPT),
            *state["messages"],
        ])
        decision = response.content.strip().lower()
        logger.info(f"Top Supervisor decision: {decision!r}")

        if decision == "finish" or state.get("iteration_count", 0) >= 6:
            return Command(goto=END)

        if decision in ("research_team", "analysis_team"):
            return Command(
                goto=decision,
                update={"iteration_count": state.get("iteration_count", 0) + 1},
            )
        return Command(goto=END)

    # ── Team adapter wrappers ──────────────────────────────────────────────
    async def run_research_team(state: SupervisorState) -> Command:
        team_state = TeamState(
            messages=state["messages"],
            team_name="research_team",
            task=state["messages"][-1].content if state["messages"] else "",
            team_members=list(research_members.keys()),
            next_worker=None,
            result=None,
        )
        result = await research_graph.ainvoke(team_state)
        return Command(goto="top_supervisor", update={"messages": result.get("messages", [])})

    async def run_analysis_team(state: SupervisorState) -> Command:
        team_state = TeamState(
            messages=state["messages"],
            team_name="analysis_team",
            task=state["messages"][-1].content if state["messages"] else "",
            team_members=list(analysis_members.keys()),
            next_worker=None,
            result=None,
        )
        result = await analysis_graph.ainvoke(team_state)
        return Command(goto="top_supervisor", update={"messages": result.get("messages", [])})

    # ── Build top-level graph ──────────────────────────────────────────────
    builder = StateGraph(SupervisorState)
    builder.add_node("top_supervisor", top_supervisor)
    builder.add_node("research_team", run_research_team)
    builder.add_node("analysis_team", run_analysis_team)
    builder.set_entry_point("top_supervisor")

    logger.info("Hierarchical graph compiled with research_team + analysis_team")
    return builder.compile(checkpointer=checkpointer)
