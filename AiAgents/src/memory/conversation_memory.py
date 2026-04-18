"""
Conversation Memory — Buffer, Summary, and Window memory implementations.
Wraps LangChain memory classes with a unified interface.
"""

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConversationBufferMemory:
    """Stores the full conversation history in memory."""

    def __init__(self, max_messages: int = 100):
        self._history: list[BaseMessage] = []
        self.max_messages = max_messages

    def add_user_message(self, content: str) -> None:
        self._history.append(HumanMessage(content=content))
        self._trim()

    def add_ai_message(self, content: str) -> None:
        self._history.append(AIMessage(content=content))
        self._trim()

    def add_messages(self, messages: list[BaseMessage]) -> None:
        self._history.extend(messages)
        self._trim()

    def get_messages(self) -> list[BaseMessage]:
        return list(self._history)

    def clear(self) -> None:
        self._history.clear()

    def _trim(self) -> None:
        if len(self._history) > self.max_messages:
            self._history = self._history[-self.max_messages:]

    def __len__(self) -> int:
        return len(self._history)


class ConversationBufferWindowMemory(ConversationBufferMemory):
    """Keeps only the K most recent message pairs (sliding window)."""

    def __init__(self, k: int = 5):
        super().__init__(max_messages=k * 2)
        self.k = k


class ConversationSummaryMemory:
    """
    Maintains a rolling summary of long conversations.
    When message count exceeds the threshold, older messages are condensed.
    """

    def __init__(self, llm: BaseChatModel | None = None, summary_threshold: int = 20):
        self._recent: list[BaseMessage] = []
        self._summary: str = ""
        self.summary_threshold = summary_threshold
        self._llm = llm

    def add_user_message(self, content: str) -> None:
        self._recent.append(HumanMessage(content=content))
        self._maybe_summarize()

    def add_ai_message(self, content: str) -> None:
        self._recent.append(AIMessage(content=content))
        self._maybe_summarize()

    def _maybe_summarize(self) -> None:
        if len(self._recent) < self.summary_threshold:
            return
        if not self._llm:
            # Without LLM, just trim
            self._recent = self._recent[-10:]
            return

        import asyncio
        from langchain_core.messages import SystemMessage
        try:
            to_summarize = self._recent[:-10]
            history_text = "\n".join(
                f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
                for m in to_summarize
            )
            prompt = [
                SystemMessage(content="Summarize the following conversation concisely:"),
                HumanMessage(content=history_text),
            ]
            response = asyncio.run(self._llm.ainvoke(prompt))
            self._summary = (self._summary + "\n" + response.content).strip()
            self._recent = self._recent[-10:]
            logger.debug("ConversationSummaryMemory: summary updated")
        except Exception as e:
            logger.warning(f"ConversationSummaryMemory summarization failed: {e}")

    def get_messages(self) -> list[BaseMessage]:
        messages = []
        if self._summary:
            from langchain_core.messages import SystemMessage
            messages.append(SystemMessage(content=f"Previous conversation summary:\n{self._summary}"))
        messages.extend(self._recent)
        return messages

    def get_summary(self) -> str:
        return self._summary

    def clear(self) -> None:
        self._recent.clear()
        self._summary = ""
