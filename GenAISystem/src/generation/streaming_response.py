"""
Streaming Response — Async streaming token generation with SSE support.
"""

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

logger = logging.getLogger(__name__)


class StreamingResponder:
    """Stream LLM responses token-by-token with Server-Sent Events (SSE) support.

    Features:
    - Async streaming from any LangChain ChatModel
    - SSE-formatted output for web clients
    - Partial response buffering
    - Source metadata in final SSE event
    """

    DEFAULT_SYSTEM_PROMPT = """You are a knowledgeable AI assistant. Answer the user's question
based on the provided context. Be concise, accurate, and cite your sources."""

    def __init__(self, system_prompt: Optional[str] = None):
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    async def stream_generate(
        self,
        llm,
        query: str,
        context: List[Dict[str, Any]],
        additional_instructions: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream tokens from the LLM as they are generated.

        Args:
            llm: A LangChain ChatModel that supports astream().
            query: The user's question.
            context: List of retrieved document dicts.
            additional_instructions: Optional extra instructions.

        Yields:
            Individual tokens/chunks as strings.
        """
        from langchain_core.messages import SystemMessage, HumanMessage

        context_text = self._assemble_context(context)
        user_prompt = f"""Context:
{context_text}

---
Question: {query}
"""
        if additional_instructions:
            user_prompt += f"\nInstructions: {additional_instructions}"

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt),
        ]

        async for chunk in llm.astream(messages):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            if content:
                yield content

    async def stream_sse(
        self,
        llm,
        query: str,
        context: List[Dict[str, Any]],
    ) -> AsyncIterator[str]:
        """Stream responses in Server-Sent Events (SSE) format.

        Yields SSE-formatted strings:
        - `data: {"type": "token", "content": "..."}`
        - `data: {"type": "sources", "sources": [...]}`
        - `data: {"type": "done"}`

        Args:
            llm: A LangChain ChatModel.
            query: The user's question.
            context: List of retrieved document dicts.

        Yields:
            SSE-formatted strings.
        """
        # Stream tokens
        buffer = []
        async for token in self.stream_generate(llm, query, context):
            buffer.append(token)
            event = json.dumps({"type": "token", "content": token})
            yield f"data: {event}\n\n"

        # Send source metadata
        sources = [
            {
                "index": i + 1,
                "source": doc.get("metadata", {}).get("source", "Unknown"),
                "page": str(doc.get("metadata", {}).get("page", "")),
            }
            for i, doc in enumerate(context)
        ]
        source_event = json.dumps({"type": "sources", "sources": sources})
        yield f"data: {source_event}\n\n"

        # Send completion signal
        full_response = "".join(buffer)
        done_event = json.dumps({
            "type": "done",
            "total_tokens": len(full_response.split()),
        })
        yield f"data: {done_event}\n\n"

    async def stream_with_buffer(
        self,
        llm,
        query: str,
        context: List[Dict[str, Any]],
        buffer_size: int = 5,
    ) -> AsyncIterator[str]:
        """Stream with partial buffering — yields chunks of N tokens at a time.

        Useful for reducing the number of network events while still streaming.

        Args:
            llm: A LangChain ChatModel.
            query: The user's question.
            context: List of retrieved document dicts.
            buffer_size: Number of tokens to buffer before yielding.

        Yields:
            Buffered token chunks.
        """
        buffer = []
        async for token in self.stream_generate(llm, query, context):
            buffer.append(token)
            if len(buffer) >= buffer_size:
                yield "".join(buffer)
                buffer.clear()

        # Flush remaining
        if buffer:
            yield "".join(buffer)

    @staticmethod
    def _assemble_context(context: List[Dict[str, Any]]) -> str:
        """Assemble context documents into a single string."""
        parts = []
        for i, doc in enumerate(context):
            content = doc.get("content", "")
            source = doc.get("metadata", {}).get("source", f"Document {i+1}")
            parts.append(f"[Source {i+1}: {source}]\n{content}")
        return "\n\n".join(parts)
