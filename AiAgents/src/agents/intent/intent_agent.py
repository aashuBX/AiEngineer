"""
Intent Agent — entry point for all user queries.
Classifies the user's intent and routes to the appropriate specialist agent.
"""

from typing import Any

from langchain_core.messages import HumanMessage

from src.agents.base_agent import BaseAgent
from src.config.llm_providers import get_default_llm
from src.models.schemas import AgentConfig, IntentType
from src.utils.logger import get_logger

logger = get_logger(__name__)

_INTENT_SYSTEM_PROMPT = """You are an Intent Classification Agent for an AI customer service platform.

Your ONLY job is to read the user's latest message and classify it into ONE of these intent categories:

- faq         → General questions about policies, products, pricing, how-to guides
- crm         → Questions about a specific user's account, orders, history, or personal data
- rag         → Questions requiring search over standard uploaded documents and manuals
- graph_rag   → Deep domain questions requiring knowledge-graph reasoning or entity synthesis
- feedback    → User is rating, reviewing, or providing feedback on a past interaction
- handoff     → User is frustrated, asks for a human, or the situation requires escalation
- unknown     → Cannot be classified or is out of scope

Respond ONLY with a JSON object in this exact format:
{
  "intent": "<one of the six categories>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<one sentence explanation>"
}

Do NOT add any extra text before or after the JSON.
"""


class IntentAgent(BaseAgent):
    """Classifies user intent and populates the routing decision in graph state."""

    def __init__(self):
        super().__init__(
            config=AgentConfig(
                name="IntentAgent",
                description="Classifies user queries and determines routing path",
                system_prompt=_INTENT_SYSTEM_PROMPT,
                temperature=0.0,
            )
        )

    @property
    def system_prompt(self) -> str:
        return _INTENT_SYSTEM_PROMPT

    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Reads the last user message, classifies intent, updates state.
        Returns updated state with 'intent' and 'current_agent' set.
        """
        import json

        messages = state.get("messages", [])
        if not messages:
            logger.warning("IntentAgent: no messages in state")
            return {**state, "intent": IntentType.UNKNOWN, "current_agent": "intent"}

        # Get the last human message
        user_message = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
                user_message = msg.content
                break

        if not user_message:
            return {**state, "intent": IntentType.UNKNOWN, "current_agent": "intent"}

        llm = get_default_llm()
        prompt_msgs = self.build_messages(user_message)

        try:
            response = await llm.ainvoke(prompt_msgs)
            raw = response.content.strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            data = json.loads(raw)
            intent_str = data.get("intent", "unknown").lower()
            confidence = float(data.get("confidence", 0.5))
            reasoning = data.get("reasoning", "")

            # Validate against known intents
            try:
                intent = IntentType(intent_str)
            except ValueError:
                intent = IntentType.UNKNOWN

            logger.info(
                f"IntentAgent → intent={intent.value}, confidence={confidence:.2f}, "
                f"reason={reasoning!r}"
            )

            return {
                **state,
                "intent": intent.value,
                "current_agent": "intent",
                "metadata": {
                    **state.get("metadata", {}),
                    "intent_confidence": confidence,
                    "intent_reasoning": reasoning,
                },
            }

        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"IntentAgent failed to parse response: {e}")
            return {**state, "intent": IntentType.UNKNOWN.value, "current_agent": "intent"}
