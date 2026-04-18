"""
HandOff Agent — detects user frustration or explicit escalation requests,
summarizes the conversation, and pauses the graph for human takeover.
"""

import re
from typing import Any

from langchain_core.messages import AIMessage

from src.agents.base_agent import BaseAgent
from src.models.schemas import AgentConfig, HandoffRequest
from src.utils.logger import get_logger

logger = get_logger(__name__)

_HANDOFF_SYSTEM_PROMPT = """You are a Human Escalation Agent.

Your role:
1. Detect frustration signals or explicit requests for human assistance
2. Empathize with the user in a calm, professional tone
3. Inform them that a human agent will take over shortly
4. Summarize the conversation for the incoming human agent

Frustration signals: "this is useless", "I want a human", "speak to a person",
"your bot is terrible", "escalate", "manager", "supervisor", repeated failures.

Response format:
- Acknowledge the user's frustration or request warmly
- Confirm a human agent is being notified
- Ask them to stay connected
"""

_ESCALATION_PATTERNS = [
    re.compile(r"\b(human|person|agent|manager|supervisor|escalate|real\s+person)\b", re.IGNORECASE),
    re.compile(r"\b(this\s+is\s+(useless|terrible|awful|horrible|garbage))\b", re.IGNORECASE),
    re.compile(r"\b(i\s+(give\s+up|can't\s+do|hate)\s+this)\b", re.IGNORECASE),
    re.compile(r"\b(not\s+helping|doesn't\s+work|stop\s+the\s+bot)\b", re.IGNORECASE),
]


def needs_handoff(text: str) -> bool:
    """Quick regex check for escalation signals."""
    return any(p.search(text) for p in _ESCALATION_PATTERNS)


class HandoffAgent(BaseAgent):
    """Manages human escalation by pausing the graph and notifying the human agent."""

    def __init__(self, webhook_url: str | None = None):
        super().__init__(
            config=AgentConfig(
                name="HandoffAgent",
                description="Detects escalation signals and transfers to human agents",
                temperature=0.3,
            )
        )
        self._webhook_url = webhook_url

    @property
    def system_prompt(self) -> str:
        return _HANDOFF_SYSTEM_PROMPT

    async def _notify_human_agent(self, request: HandoffRequest) -> bool:
        """Send escalation payload to a webhook (e.g., Slack, PlatformUI)."""
        if not self._webhook_url:
            logger.warning("HandoffAgent: no webhook URL configured — skipping notification")
            return False
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    self._webhook_url,
                    json=request.model_dump(mode="json"),
                )
                resp.raise_for_status()
                logger.info(f"HandoffAgent: webhook notified — status {resp.status_code}")
                return True
        except Exception as e:
            logger.error(f"HandoffAgent webhook failed: {e}")
            return False

    async def _summarize_conversation(self, messages: list) -> str:
        """LLM-summarize the conversation thread for the incoming human agent."""
        from src.config.llm_providers import get_default_llm
        llm = get_default_llm()
        summary_request = (
            "Summarize this conversation in 3-5 bullet points for a human agent "
            "who will take over. Focus on: user's original issue, agents attempted, "
            "what failed, and the user's current emotional state.\n\n"
            + "\n".join(
                f"{getattr(m, 'type', 'msg').upper()}: {m.content}"
                for m in messages[-10:]  # last 10 messages
            )
        )
        try:
            response = await llm.ainvoke([
                {"role": "system", "content": "You are a conversation summarizer."},
                {"role": "user", "content": summary_request},
            ])
            return response.content
        except Exception:
            return "Conversation summary unavailable."

    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        messages = state.get("messages", [])
        user_message = ""
        for msg in reversed(messages):
            if getattr(msg, "type", "") == "human":
                user_message = msg.content
                break

        logger.info(f"HandoffAgent: processing escalation for session {state.get('session_id')}")

        # Summarize conversation
        summary = await self._summarize_conversation(messages)

        # Build handoff request
        handoff = HandoffRequest(
            session_id=state.get("session_id", "unknown"),
            reason=user_message,
            conversation_summary=summary,
            user_message=user_message,
            urgency="high" if needs_handoff(user_message) else "normal",
        )

        # Notify human agent via webhook
        await self._notify_human_agent(handoff)

        # Generate user-facing response
        from src.config.llm_providers import get_default_llm
        llm = get_default_llm()
        prompt_msgs = self.build_messages(
            f"User message: {user_message}\nSituation: human escalation requested"
        )
        try:
            response = await llm.ainvoke(prompt_msgs)
            answer = response.content
        except Exception:
            answer = (
                "I understand your frustration and I'm sorry for the trouble. "
                "A human agent has been notified and will connect with you shortly. "
                "Please stay connected."
            )

        return {
            **state,
            "messages": [AIMessage(content=answer)],
            "task_status": "awaiting_human",
            "awaiting_human": True,
            "current_agent": "handoff",
            "metadata": {
                **state.get("metadata", {}),
                "handoff_summary": summary,
                "handoff_urgency": handoff.urgency,
            },
        }
