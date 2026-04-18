"""
Human-in-the-Loop Interrupt Handler.
Manages LangGraph interrupt points for human review and state serialization.
"""

import json
from typing import Any

from langgraph.types import interrupt

from src.utils.logger import get_logger

logger = get_logger(__name__)


def request_human_approval(
    state: dict[str, Any],
    question: str,
    context: dict | None = None,
) -> Any:
    """
    Pause the LangGraph graph and request human input.

    This uses LangGraph's `interrupt()` primitive — the graph will pause
    here and resume when .update_state() is called with human feedback.

    Args:
        state:    Current graph state dict.
        question: The question to present to the human reviewer.
        context:  Optional additional context for the reviewer.

    Returns:
        The human's response (after graph is resumed).
    """
    payload = {
        "question": question,
        "context": context or {},
        "session_id": state.get("session_id", "unknown"),
        "current_agent": state.get("current_agent", "unknown"),
    }
    logger.info(f"HITL: interrupting for human approval — session={payload['session_id']}")
    # This call suspends graph execution until resumed
    human_response = interrupt(payload)
    logger.info(f"HITL: graph resumed — human response received")
    return human_response


def serialize_interrupt_state(state: dict[str, Any]) -> str:
    """Serialize current graph state to JSON for external storage/review."""
    serializable = {}
    for key, value in state.items():
        try:
            json.dumps(value)  # Test serializability
            serializable[key] = value
        except (TypeError, ValueError):
            serializable[key] = str(value)
    return json.dumps(serializable, indent=2)


def parse_human_decision(response: Any) -> dict[str, Any]:
    """
    Parse a human approval response into a structured decision.

    Supports:
    - String: "approve" | "reject" | "modify: <new_instruction>"
    - Dict:   {"decision": "...", "comment": "...", "modified_input": "..."}

    Returns:
        Dict with keys: approved (bool), modified_input (str|None), comment (str).
    """
    if isinstance(response, dict):
        decision = response.get("decision", "").lower()
        return {
            "approved": decision in ("approve", "approved", "yes", "ok"),
            "modified_input": response.get("modified_input"),
            "comment": response.get("comment", ""),
        }

    if isinstance(response, str):
        lower = response.strip().lower()
        if lower.startswith("modify:"):
            modification = response[len("modify:"):].strip()
            return {"approved": True, "modified_input": modification, "comment": "Modified by human"}
        approved = lower in ("approve", "approved", "yes", "ok", "y")
        return {"approved": approved, "modified_input": None, "comment": response}

    return {"approved": False, "modified_input": None, "comment": "Unknown response format"}


def should_interrupt(state: dict[str, Any], trigger_keys: list[str] | None = None) -> bool:
    """
    Determine whether the graph should interrupt based on state conditions.

    Args:
        state:        Current graph state.
        trigger_keys: State keys that, if True, trigger interrupt.

    Returns:
        True if interrupt should be triggered.
    """
    if state.get("awaiting_human"):
        return True

    for key in (trigger_keys or []):
        if state.get(key):
            return True

    return False
