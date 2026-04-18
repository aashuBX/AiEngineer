"""
Abstract BaseAgent — all specialized agents extend this class.
Provides unified invoke/ainvoke interface, tool binding, and prompt templating.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.tools import BaseTool

from src.config.llm_providers import get_llm
from src.models.schemas import AgentConfig, TaskResult, TaskStatus
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseAgent(ABC):
    """
    Abstract base for all AiAgents system agents.
    Subclasses must implement `system_prompt` and `process()`.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig(
            name=self.__class__.__name__,
            description="Generic AI agent",
        )
        self._llm: Optional[BaseChatModel] = None
        self._tools: list[BaseTool] = []
        logger.info(f"Initialised agent: {self.config.name}")

    # ── Abstract interface ─────────────────────────────────────────────────────
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the agent's system-level instruction."""

    @abstractmethod
    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Core processing logic — receives a LangGraph state dict,
        returns an updated state dict.
        """

    # ── LLM access ─────────────────────────────────────────────────────────────
    @property
    def llm(self) -> BaseChatModel:
        """Lazy-loaded LLM from the agent config."""
        if self._llm is None:
            self._llm = get_llm(
                provider=self.config.provider,
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            if self._tools:
                self._llm = self._llm.bind_tools(self._tools)
        return self._llm

    # ── Tool management ────────────────────────────────────────────────────────
    def register_tools(self, tools: list[BaseTool]) -> None:
        """Register tools and rebind to LLM."""
        self._tools = tools
        self._llm = None  # force rebuild with new tools

    # ── Prompt helpers ─────────────────────────────────────────────────────────
    def build_messages(self, user_input: str, history: Optional[list[BaseMessage]] = None) -> list[BaseMessage]:
        """Build a message list with system prompt + optional history + new user message."""
        messages: list[BaseMessage] = [SystemMessage(content=self.system_prompt)]
        if history:
            messages.extend(history)
        messages.append(HumanMessage(content=user_input))
        return messages

    # ── Synchronous wrapper ────────────────────────────────────────────────────
    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        """Synchronous wrapper — runs the async process() in an event loop."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.process(state))
                    return future.result()
            return loop.run_until_complete(self.process(state))
        except RuntimeError:
            return asyncio.run(self.process(state))

    # ── Async invoke ───────────────────────────────────────────────────────────
    async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
        """Async entry point — delegates to process()."""
        return await self.process(state)

    # ── Result helpers ─────────────────────────────────────────────────────────
    def success_result(self, output: str, **kwargs) -> TaskResult:
        return TaskResult(
            status=TaskStatus.DONE,
            agent_name=self.config.name,
            output=output,
            **kwargs,
        )

    def error_result(self, error: str) -> TaskResult:
        return TaskResult(
            status=TaskStatus.ERROR,
            agent_name=self.config.name,
            output="",
            metadata={"error": error},
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.config.name!r}>"
