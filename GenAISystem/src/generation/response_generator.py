"""
Response Generator — Assemble context + query into LLM responses with token management.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Generate RAG responses from retrieved context and user queries.

    Features:
    - System prompt customization
    - Context assembly and truncation
    - Multi-provider LLM support
    - Token limit management
    """

    DEFAULT_SYSTEM_PROMPT = """You are a knowledgeable AI assistant. Answer the user's question
based ONLY on the provided context. If the context doesn't contain enough information
to answer the question, say so honestly. Always be accurate and cite your sources."""

    RESPONSE_PROMPT = """Context:
{context}

---
Question: {query}

Instructions:
1. Answer based strictly on the provided context.
2. If the context is insufficient, acknowledge the limitation.
3. Be concise but thorough.
4. Reference specific parts of the context when possible.
"""

    def __init__(
        self,
        llm,
        system_prompt: Optional[str] = None,
        max_context_tokens: int = 3000,
        response_template: Optional[str] = None,
    ):
        """
        Args:
            llm: A LangChain ChatModel instance.
            system_prompt: Custom system prompt. Uses default if None.
            max_context_tokens: Approximate max tokens for context (conservative estimate).
            response_template: Custom template for the user prompt.
        """
        self.llm = llm
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.max_context_tokens = max_context_tokens
        self.response_template = response_template or self.RESPONSE_PROMPT

    def generate(
        self,
        query: str,
        context: List[Dict[str, Any]],
        additional_instructions: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a response from retrieved context.

        Args:
            query: The user's question.
            context: List of retrieved document dicts with 'content' and 'metadata'.
            additional_instructions: Optional extra instructions for the LLM.

        Returns:
            Dict with 'answer', 'sources', and 'model' info.
        """
        from langchain_core.messages import SystemMessage, HumanMessage

        # Assemble and truncate context
        context_text = self._assemble_context(context)

        # Build the prompt
        user_prompt = self.response_template.format(
            context=context_text,
            query=query,
        )
        if additional_instructions:
            user_prompt += f"\n\nAdditional Instructions: {additional_instructions}"

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            answer = response.content if hasattr(response, "content") else str(response)

            sources = self._extract_sources(context)

            return {
                "answer": answer,
                "sources": sources,
                "model": getattr(self.llm, "model_name", "unknown"),
                "context_docs_used": len(context),
            }
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return {
                "answer": f"I encountered an error generating the response: {e}",
                "sources": [],
                "model": "error",
                "context_docs_used": 0,
            }

    def _assemble_context(self, context: List[Dict[str, Any]]) -> str:
        """Assemble retrieved documents into a single context string with truncation."""
        context_parts = []
        estimated_tokens = 0

        for i, doc in enumerate(context):
            content = doc.get("content", "")
            source = doc.get("metadata", {}).get("source", f"Document {i+1}")

            chunk = f"[Source {i+1}: {source}]\n{content}\n"

            # Rough token estimate (1 token ≈ 4 chars)
            chunk_tokens = len(chunk) // 4
            if estimated_tokens + chunk_tokens > self.max_context_tokens:
                # Truncate this chunk to fit
                remaining_tokens = self.max_context_tokens - estimated_tokens
                remaining_chars = remaining_tokens * 4
                chunk = chunk[:remaining_chars] + "...[truncated]"
                context_parts.append(chunk)
                break

            context_parts.append(chunk)
            estimated_tokens += chunk_tokens

        return "\n".join(context_parts)

    @staticmethod
    def _extract_sources(context: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract source metadata from context documents."""
        sources = []
        for i, doc in enumerate(context):
            meta = doc.get("metadata", {})
            sources.append({
                "index": i + 1,
                "source": meta.get("source", "Unknown"),
                "page": str(meta.get("page", "")),
                "chunk_id": doc.get("id", ""),
            })
        return sources
