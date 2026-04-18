"""
Long-Term Memory — vector-backed semantic memory for cross-session recall.
Stores important facts/summaries and retrieves them by semantic similarity.
"""

from datetime import datetime
from typing import Any

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MemoryEntry:
    """A single long-term memory record."""
    def __init__(self, content: str, session_id: str, metadata: dict | None = None):
        self.content = content
        self.session_id = session_id
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow().isoformat()
        self.vector: list[float] | None = None


class LongTermVectorMemory:
    """
    Semantic vector memory store backed by FAISS (local) or Qdrant (scalable).
    Stores conversation highlights and retrieves them by semantic similarity.
    """

    def __init__(
        self,
        provider: str | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        collection_name: str = "long_term_memory",
    ):
        self.provider = provider or settings.vector_memory_provider
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self._entries: list[MemoryEntry] = []
        self._embedder = None
        self._index = None
        logger.info(f"LongTermVectorMemory: provider={self.provider}, model={embedding_model}")

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedding_model)
        return self._embedder

    def _embed(self, text: str) -> list[float]:
        embedder = self._get_embedder()
        return embedder.encode(text, normalize_embeddings=True).tolist()

    def store(self, content: str, session_id: str, metadata: dict | None = None) -> None:
        """Store a new memory entry with its embedding."""
        entry = MemoryEntry(content, session_id, metadata)
        entry.vector = self._embed(content)
        self._entries.append(entry)

        if self.provider == "qdrant":
            self._store_qdrant(entry)
        logger.debug(f"LongTermVectorMemory: stored memory for session {session_id}")

    def retrieve(self, query: str, top_k: int = 5, session_id: str | None = None) -> list[dict]:
        """
        Retrieve the most semantically similar memories to a query.

        Args:
            query:      Query string to search.
            top_k:      Number of results to return.
            session_id: Optional filter by session.

        Returns:
            List of dicts with 'content', 'score', 'metadata', 'created_at'.
        """
        if not self._entries:
            return []

        query_vec = self._embed(query)
        candidates = [
            e for e in self._entries
            if session_id is None or e.session_id == session_id
        ]

        if not candidates:
            return []

        # Cosine similarity
        import numpy as np
        q = np.array(query_vec)
        scores = []
        for entry in candidates:
            if entry.vector:
                v = np.array(entry.vector)
                score = float(np.dot(q, v))
                scores.append((score, entry))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "content": e.content,
                "score": s,
                "metadata": e.metadata,
                "session_id": e.session_id,
                "created_at": e.created_at,
            }
            for s, e in scores[:top_k]
        ]

    def _store_qdrant(self, entry: MemoryEntry) -> None:
        """Persist entry to Qdrant (non-blocking best-effort)."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import PointStruct
            import uuid

            client = QdrantClient(url=settings.qdrant_url)
            client.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(
                    id=str(uuid.uuid4()),
                    vector=entry.vector,
                    payload={
                        "content": entry.content,
                        "session_id": entry.session_id,
                        "metadata": entry.metadata,
                        "created_at": entry.created_at,
                    },
                )],
            )
        except Exception as e:
            logger.warning(f"LongTermVectorMemory: Qdrant store failed: {e}")

    def clear(self, session_id: str | None = None) -> None:
        """Clear all or session-specific memories."""
        if session_id:
            self._entries = [e for e in self._entries if e.session_id != session_id]
        else:
            self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
