"""
Feedback Agent — collects user ratings, aggregates feedback,
and updates memory logs to guide future interactions.
"""

from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage

from src.agents.base_agent import BaseAgent
from src.models.schemas import AgentConfig, FeedbackEntry
from src.utils.logger import get_logger

logger = get_logger(__name__)

_FEEDBACK_SYSTEM_PROMPT = """You are a Feedback Collection Agent.

Your role:
1. Acknowledge the user's feedback warmly and professionally
2. Parse the feedback to identify: rating (1-5), sentiment, and specific issues mentioned
3. Thank the user for their time
4. If feedback is negative (rating 1-2), offer to escalate to a human agent

Extract and respond naturally. Do NOT ask the user to repeat themselves.
"""


class FeedbackAgent(BaseAgent):
    """Collects and processes user feedback, storing entries for analytics."""

    def __init__(self):
        super().__init__(
            config=AgentConfig(
                name="FeedbackAgent",
                description="Collects user ratings and feedback for continuous improvement",
                temperature=0.3,
            )
        )
        self._feedback_log: list[FeedbackEntry] = []

    @property
    def system_prompt(self) -> str:
        return _FEEDBACK_SYSTEM_PROMPT

    def _parse_rating(self, text: str) -> int | None:
        """Extract numeric rating from text."""
        import re
        # Look for explicit ratings: "3/5", "4 out of 5", "rating: 2", or just a number 1-5
        patterns = [
            r"(\d)/5",
            r"(\d)\s*out\s*of\s*5",
            r"rating[:\s]+(\d)",
            r"(?<!\d)([1-5])(?!\d)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                rating = int(match.group(1))
                if 1 <= rating <= 5:
                    return rating
        return None

    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        messages = state.get("messages", [])
        user_message = ""
        for msg in reversed(messages):
            if getattr(msg, "type", "") == "human":
                user_message = msg.content
                break

        if not user_message:
            return {**state, "task_status": "error", "error": "No feedback message found"}

        logger.info(f"FeedbackAgent: processing feedback: {user_message!r}")

        rating = self._parse_rating(user_message)

        # Store the feedback entry
        entry = FeedbackEntry(
            session_id=state.get("session_id", "unknown"),
            message_id=str(len(self._feedback_log)),
            rating=rating or 3,
            comment=user_message,
            agent_name="FeedbackAgent",
            timestamp=datetime.utcnow(),
        )
        self._feedback_log.append(entry)
        logger.info(f"FeedbackAgent: stored feedback — rating={entry.rating}")

        # Generate acknowledgement
        from src.config.llm_providers import get_default_llm
        llm = get_default_llm()
        context = f"User's feedback: {user_message}\nExtracted rating: {rating or 'not specified'}"
        prompt_msgs = self.build_messages(context)

        try:
            response = await llm.ainvoke(prompt_msgs)
            answer = response.content

            # If rating is very low, suggest escalation
            should_escalate = rating is not None and rating <= 2

            return {
                **state,
                "messages": [AIMessage(content=answer)],
                "task_status": "done",
                "current_agent": "feedback",
                "metadata": {
                    **state.get("metadata", {}),
                    "feedback_rating": rating,
                    "suggest_escalation": should_escalate,
                },
            }
        except Exception as e:
            logger.error(f"FeedbackAgent failed: {e}")
            return {**state, "task_status": "error", "error": str(e)}

    def get_feedback_summary(self) -> dict[str, Any]:
        """Return aggregate feedback statistics."""
        if not self._feedback_log:
            return {"total": 0, "average_rating": None}
        ratings = [e.rating for e in self._feedback_log]
        return {
            "total": len(self._feedback_log),
            "average_rating": sum(ratings) / len(ratings),
            "distribution": {i: ratings.count(i) for i in range(1, 6)},
        }
