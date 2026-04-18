"""
LangGraph state schemas for all agent graphs.
Uses TypedDict + Annotated reducers for safe state merging.
"""

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# ── Base Agent State ───────────────────────────────────────────────────────────
class AgentState(TypedDict):
    """Shared base state for all single-agent and multi-agent graphs."""

    # Message history — add_messages reducer appends rather than overwrites
    messages: Annotated[list[BaseMessage], add_messages]

    # Routing & orchestration
    current_agent: str                     # Name of the currently active agent
    intent: Optional[str]                  # Classified user intent
    task_status: str                       # "pending" | "in_progress" | "done" | "error"

    # Tool execution results
    tool_results: list[dict[str, Any]]

    # Guardrail decisions
    input_safe: bool                       # True if input passed all guardrails
    output_safe: bool                      # True if output passed all guardrails

    # Human-in-the-loop
    awaiting_human: bool                   # True if graph is paused for human input
    human_feedback: Optional[str]          # Feedback/approval text from human

    # Metadata
    metadata: dict[str, Any]              # Arbitrary extra context
    error: Optional[str]                  # Error message if task_status == "error"
    session_id: str                        # Unique thread/conversation ID


# ── Supervisor State ───────────────────────────────────────────────────────────
class SupervisorState(AgentState):
    """State for supervisor-worker multi-agent graphs."""

    worker_responses: list[dict[str, Any]]   # Aggregated worker outputs
    routing_decision: Optional[str]           # Which worker to call next
    active_workers: list[str]                 # Currently dispatched worker agents
    iteration_count: int                      # Guard against infinite loops


# ── Plan-and-Execute State ─────────────────────────────────────────────────────
class PlanExecuteState(AgentState):
    """State for Planner-Executor-Replanner workflows."""

    plan: list[str]              # Ordered list of plan steps
    current_step: int            # Index of the step being executed
    past_steps: list[tuple[str, str]]  # (step, result) history
    execution_results: list[str] # Results from each executed step
    final_response: Optional[str]


# ── Hierarchical Team State ────────────────────────────────────────────────────
class TeamState(TypedDict):
    """State for a sub-team within a hierarchical graph."""

    messages: Annotated[list[BaseMessage], add_messages]
    team_name: str
    task: str
    team_members: list[str]
    next_worker: Optional[str]
    result: Optional[str]


# ── Map-Reduce State ───────────────────────────────────────────────────────────
class MapReduceState(AgentState):
    """State for parallel map-reduce graph patterns."""

    sub_tasks: list[str]         # Input tasks to fan-out
    sub_results: list[str]       # Collected results from parallel workers
    final_summary: Optional[str] # Reduced/aggregated output
