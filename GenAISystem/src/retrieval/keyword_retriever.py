"""
Keyword Retriever — BM25-based lexical retrieval with tokenization and exact match boosting.
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class KeywordRetriever:
    """BM25 keyword search retriever using rank-bm25."""

    def __init__(
        self,
        documents: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """
        Args:
            documents: List of text documents to index.
            metadata: Parallel list of metadata dicts.
            ids: Parallel list of document IDs.
            k1: BM25 term frequency saturation parameter.
            b: BM25 document length normalization parameter.
        """
        self.documents = documents or []
        self.metadata = metadata or [{} for _ in self.documents]
        self.ids = ids or [str(i) for i in range(len(self.documents))]
        self.k1 = k1
        self.b = b
        self._bm25 = None

        if self.documents:
            self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric, filter short tokens."""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return [t for t in tokens if len(t) > 1]

    def _build_index(self):
        """Build the BM25 index from the document corpus."""
        from rank_bm25 import BM25Okapi

        tokenized_corpus = [self._tokenize(doc) for doc in self.documents]
        self._bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        logger.info(f"Built BM25 index over {len(self.documents)} documents.")

    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ):
        """Add new documents and rebuild the index."""
        metadata = metadata or [{} for _ in documents]
        ids = ids or [str(len(self.documents) + i) for i in range(len(documents))]

        self.documents.extend(documents)
        self.metadata.extend(metadata)
        self.ids.extend(ids)
        self._build_index()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        boost_exact: bool = True,
    ) -> List[Dict[str, Any]]:
        """Retrieve documents ranked by BM25 relevance.

        Args:
            query: The search query.
            top_k: Number of results to return.
            boost_exact: If True, boost documents containing exact query match.

        Returns:
            List of result dicts: {'content', 'metadata', 'score', 'id'}.
        """
        if not self._bm25:
            logger.warning("BM25 index is empty.")
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Exact match boosting
        if boost_exact:
            query_lower = query.lower()
            for i, doc in enumerate(self.documents):
                if query_lower in doc.lower():
                    scores[i] *= 1.5

        # Sort by score descending
        scored_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []
        for idx in scored_indices:
            if scores[idx] > 0:
                results.append({
                    "id": self.ids[idx],
                    "content": self.documents[idx],
                    "metadata": self.metadata[idx],
                    "score": float(scores[idx]),
                })

        logger.info(f"KeywordRetriever: {len(results)} results for query")
        return results
