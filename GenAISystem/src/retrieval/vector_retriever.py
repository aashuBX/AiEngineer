"""
Vector Retriever — Pure vector similarity search with configurable thresholds and metadata filtering.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class VectorRetriever:
    """Retrieve documents using vector similarity search against a vector store."""

    def __init__(
        self,
        vector_store,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        default_filters: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            vector_store: A BaseVectorStore implementation.
            top_k: Default number of results to return.
            score_threshold: If set, filter out results below this similarity score.
            default_filters: Default metadata filters applied to every search.
        """
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.default_filters = default_filters or {}

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query.

        Args:
            query: The search query text.
            top_k: Override default top_k.
            filters: Override or extend default metadata filters.

        Returns:
            List of result dicts: {'content', 'metadata', 'score', 'id'}.
        """
        k = top_k or self.top_k
        merged_filters = {**self.default_filters, **(filters or {})}
        active_filters = merged_filters if merged_filters else None

        results = self.vector_store.search(
            query=query,
            top_k=k,
            filters=active_filters,
        )

        # Apply score threshold filtering
        if self.score_threshold is not None:
            results = [r for r in results if r.get("score", 0) >= self.score_threshold]

        logger.info(f"VectorRetriever: {len(results)} results for query (top_k={k})")
        return results

    def retrieve_by_embedding(
        self,
        embedding: List[float],
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve using a pre-computed embedding vector."""
        k = top_k or self.top_k
        merged_filters = {**self.default_filters, **(filters or {})}
        active_filters = merged_filters if merged_filters else None

        results = self.vector_store.search_by_vector(
            embedding=embedding,
            top_k=k,
            filters=active_filters,
        )

        if self.score_threshold is not None:
            results = [r for r in results if r.get("score", 0) >= self.score_threshold]

        return results
