"""
RAG Response Generation — formatting contexts, generating cited answers.
"""

from typing import Any
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from src.utils.logger import get_logger

logger = get_logger(__name__)

QA_PROMPT = """You are a helpful and knowledgeable assistant. 
Use the following pieces of retrieved context to answer the question. 
If the answer is not in the context, explicitly say "I don't have enough information to answer." 
Do NOT hallucinate or make up facts.
Always cite your sources using bracketed numbers, e.g., [1], [2].

Context:
{context}
"""


class ResponseGenerator:
    """Generates LLM responses based on retrieved hybrid context."""

    def __init__(self, llm: Any):
        self.llm = llm

    def _format_context(self, docs: list[Document]) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown Source")
            content = doc.page_content.strip()
            parts.append(f"[{i}] (Source: {source})\n{content}")
        return "\n\n".join(parts)

    async def generate_answer(self, query: str, context_docs: list[Document]) -> dict:
        """
        Generate answer and return structured response with citations.
        """
        context_str = self._format_context(context_docs)
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=QA_PROMPT.format(context=context_str)),
            HumanMessage(content=query)
        ])

        try:
            logger.info("Generating final RAG response")
            chain = prompt | self.llm
            response = await chain.ainvoke({})
            
            return {
                "answer": response.content,
                "citations": [
                    {"id": i+1, "source": d.metadata.get("source"), "snippet": d.page_content[:150]}
                    for i, d in enumerate(context_docs)
                ]
            }
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {
                "answer": "Sorry, I encountered an error generating the response.",
                "error": str(e),
                "citations": []
            }
