"""
Cross-Encoder Reranker using Sentence Transformers.
Improves retrieval quality by generating a cross-encoded score between query and context.
"""

from typing import Any
from langchain_core.documents import Document

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    """Reranks documents based on cross-encoder similarity with the query."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize the cross-encoder model."""
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        """Lazy load the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading CrossEncoder model: {self.model_name}")
                self._model = CrossEncoder(self.model_name)
            except ImportError:
                logger.warning("sentence-transformers not installed. Skipping reranking.")
                return None
        return self._model

    def rerank(self, query: str, documents: list[Document], top_k: int = 3) -> list[Document]:
        """
        Rerank a list of documents against a query.

        Args:
            query: The user query.
            documents: List of retrieved documents.
            top_k: Number of documents to return after reranking.
        """
        model = self._get_model()
        if not model or not documents:
            return documents[:top_k]

        logger.info(f"Reranking {len(documents)} documents for query: '{query}'")

        # Create query-document pairs
        pairs = [[query, doc.page_content] for doc in documents]

        # Get scores
        try:
            scores = model.predict(pairs)
            
            # Combine docs with scores and sort
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            # Return top-k documents
            return [doc for doc, score in scored_docs[:top_k]]
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return documents[:top_k]
