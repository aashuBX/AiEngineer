"""
LLM-based Reranker for the Hybrid Retrieval Pipeline.
Scores the relevance of documents to a given query using an LLM.
"""

from typing import Any
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentScore(BaseModel):
    """Structure for the LLM to return relevance scores."""
    score: int = Field(
        description="A score from 0 to 10 indicating how relevant the document is to the query. 0 is completely irrelevant, 10 is a perfect match."
    )


class LLMReranker:
    """Reranks retrieved documents by asking an LLM to score their relevance."""

    def __init__(self, llm: Any):
        """
        Initialize the LLM reranker.

        Args:
            llm: A LangChain Chat model instance (e.g. ChatGroq, ChatOpenAI)
        """
        self.llm = llm
        
        # Configure LLM to output our structured score
        try:
            self.scorer_llm = self.llm.with_structured_output(DocumentScore)
        except NotImplementedError:
            logger.warning("Structured output not supported on this LLM, using standard invoke.")
            # Fallback if with_structured_output is not supported
            self.scorer_llm = None
        
        self.prompt = PromptTemplate(
            input_variables=["query", "context"],
            template=(
                "You are an expert relevance scorer for a search engine.\n"
                "Given the user query and the retrieved document context, "
                "score the relevance of the context to the query from 0 to 10.\n\n"
                "User Query:\n{query}\n\n"
                "Context:\n{context}\n\n"
                "Return only the numerical score in structured format."
            )
        )

    def _score_document(self, query: str, doc: Document) -> float:
        """Score a single document."""
        try:
            chain_input = {"query": query, "context": doc.page_content}
            if self.scorer_llm:
                chain_prompt = self.prompt.format(**chain_input)
                result = self.scorer_llm.invoke(chain_prompt)
                return float(result.score)
            else:
                # Raw fallback: prompt asks for "Score: X"
                fallback_prompt = self.prompt.format(**chain_input) + "\nOutput purely the number."
                res = self.llm.invoke(fallback_prompt)
                content = res.content.strip()
                # Basic parsing extraction
                import re
                match = re.search(r'\d+', content)
                return float(match.group()) if match else 0.0
                
        except Exception as e:
            logger.error(f"Error scoring document with LLM: {e}")
            return 0.0

    def rerank(self, query: str, documents: list[Document], top_k: int = 3) -> list[Document]:
        """
        Rerank documents by scoring them with the LLM.

        Args:
            query: The user query.
            documents: List of retrieved documents to rerank.
            top_k: Number of documents to return.
            
        Returns:
            The reranked top-k documents.
        """
        if not documents:
            return []

        logger.info(f"LLM Reranking {len(documents)} documents for query: '{query}'")

        scored_docs = []
        for doc in documents:
            score = self._score_document(query, doc)
            scored_docs.append((doc, score))

        # Sort descending by score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(f"LLM Reranking top score: {scored_docs[0][1]}")

        # Return the top_k docs
        return [doc for doc, score in scored_docs[:top_k]]
