"""
FAQ Agent — fast-path handler for frequently asked questions.
Uses BM25 / cached lookups to avoid heavy LLM calls for common queries.
"""

from typing import Any

from langchain_core.messages import AIMessage

from src.agents.base_agent import BaseAgent
from src.models.schemas import AgentConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

_FAQ_SYSTEM_PROMPT = """You are an FAQ Specialist Agent with deep knowledge of company policies,
product documentation, and standard procedures.

Your role:
- Answer common questions quickly and accurately
- Use the pre-loaded FAQ knowledge base as your primary source
- Provide clear, concise answers (2-4 sentences when possible)
- Include links or references to official documentation when relevant
- For questions outside your FAQ scope, indicate that and suggest alternatives

Be helpful, accurate, and efficient. Avoid unnecessary verbosity.
"""

# Static FAQ cache — populated at startup from a config file or DB
_FAQ_CACHE: dict[str, str] = {
    "what are your business hours": "We are open Monday to Friday, 9 AM to 6 PM IST.",
    "how do i reset my password": "Visit the login page and click 'Forgot Password'. Enter your registered email to receive a reset link.",
    "what is your refund policy": "We offer a 30-day full refund policy. Contact support@company.com with your order ID.",
    "how do i contact support": "You can reach us via email at support@company.com or call +91-XXXXXXXXXX during business hours.",
    "what payment methods do you accept": "We accept Credit/Debit cards, UPI, Net Banking, and PayPal.",
}


def _bm25_lookup(query: str, cache: dict[str, str], threshold: float = 0.5) -> str | None:
    """Simple keyword-overlap scoring for fast FAQ matching."""
    query_tokens = set(query.lower().split())
    best_score = 0.0
    best_answer = None

    for faq_question, answer in cache.items():
        faq_tokens = set(faq_question.lower().split())
        if not faq_tokens:
            continue
        overlap = len(query_tokens & faq_tokens) / len(faq_tokens)
        if overlap > best_score:
            best_score = overlap
            best_answer = answer

    return best_answer if best_score >= threshold else None


class FaqAgent(BaseAgent):
    """Handles FAQ queries with BM25 cache lookup + LLM fallback."""

    def __init__(self, faq_cache: dict[str, str] | None = None):
        super().__init__(
            config=AgentConfig(
                name="FaqAgent",
                description="Handles standard FAQ and policy questions with fast path caching",
                temperature=0.0,
            )
        )
        self._cache = faq_cache or _FAQ_CACHE

    @property
    def system_prompt(self) -> str:
        return _FAQ_SYSTEM_PROMPT

    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        messages = state.get("messages", [])
        user_query = ""
        for msg in reversed(messages):
            if getattr(msg, "type", "") == "human":
                user_query = msg.content
                break

        if not user_query:
            return {**state, "task_status": "error", "error": "No user query found"}

        logger.info(f"FaqAgent: handling query: {user_query!r}")

        # Fast-path: BM25 cache lookup
        cached_answer = _bm25_lookup(user_query, self._cache)
        if cached_answer:
            logger.info("FaqAgent: cache hit — returning cached answer")
            return {
                **state,
                "messages": [AIMessage(content=cached_answer)],
                "task_status": "done",
                "current_agent": "faq",
                "metadata": {**state.get("metadata", {}), "faq_cache_hit": True},
            }

        # LLM fallback for uncached FAQ queries
        logger.info("FaqAgent: cache miss — delegating to LLM")
        from src.config.llm_providers import get_default_llm
        llm = get_default_llm()

        # Build context from cache
        cache_context = "\n".join(f"Q: {q}\nA: {a}" for q, a in self._cache.items())
        augmented_query = f"FAQ Knowledge Base:\n{cache_context}\n\nUser Question: {user_query}"
        prompt_msgs = self.build_messages(augmented_query)

        try:
            response = await llm.ainvoke(prompt_msgs)
            return {
                **state,
                "messages": [AIMessage(content=response.content)],
                "task_status": "done",
                "current_agent": "faq",
                "metadata": {**state.get("metadata", {}), "faq_cache_hit": False},
            }
        except Exception as e:
            logger.error(f"FaqAgent LLM call failed: {e}")
            return {**state, "task_status": "error", "error": str(e)}
