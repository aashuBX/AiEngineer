"""
Qdrant Vector Store — Local and cloud client with collection management, batch upsert, and filtering.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

from .base_store import BaseVectorStore

logger = logging.getLogger(__name__)


class QdrantStore(BaseVectorStore):
    """Qdrant-backed vector store supporting local (in-memory / on-disk) and cloud modes."""

    def __init__(
        self,
        collection_name: str,
        embedding_function,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        path: Optional[str] = None,
        dimension: int = 384,
        distance: str = "cosine",
    ):
        """
        Args:
            collection_name: Name of the Qdrant collection.
            embedding_function: Callable or LangChain Embeddings.
            url: Qdrant server URL (for cloud/remote). If None, use local.
            api_key: API key for Qdrant Cloud.
            path: Path for on-disk storage (local mode). None = in-memory.
            dimension: Embedding dimensionality.
            distance: Distance metric ('cosine', 'euclid', 'dot').
        """
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.url = url
        self.api_key = api_key
        self.path = path
        self.dimension = dimension
        self.distance = distance
        self._client = None

    @property
    def client(self):
        """Lazy-init the Qdrant client."""
        if self._client is None:
            from qdrant_client import QdrantClient

            if self.url:
                self._client = QdrantClient(url=self.url, api_key=self.api_key)
            elif self.path:
                self._client = QdrantClient(path=self.path)
            else:
                self._client = QdrantClient(":memory:")

            self._ensure_collection()
        return self._client

    def _ensure_collection(self):
        """Create the collection if it doesn't already exist."""
        from qdrant_client.models import Distance, VectorParams

        distance_map = {
            "cosine": Distance.COSINE,
            "euclid": Distance.EUCLID,
            "dot": Distance.DOT,
        }

        collections = [c.name for c in self._client.get_collections().collections]
        if self.collection_name not in collections:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=distance_map.get(self.distance, Distance.COSINE),
                ),
            )
            logger.info(f"Created Qdrant collection '{self.collection_name}' (dim={self.dimension})")

    def _embed(self, texts: List[str]) -> List[List[float]]:
        if hasattr(self.embedding_function, "embed_documents"):
            return self.embedding_function.embed_documents(texts)
        return [self.embedding_function(t) for t in texts]

    def _embed_query(self, text: str) -> List[float]:
        if hasattr(self.embedding_function, "embed_query"):
            return self.embedding_function.embed_query(text)
        return self.embedding_function(text)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_documents(
        self,
        documents: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        from qdrant_client.models import PointStruct

        vectors = embeddings or self._embed(documents)
        ids = ids or [str(uuid.uuid4()) for _ in documents]
        metadata = metadata or [{} for _ in documents]

        points = []
        for doc_id, vector, doc, meta in zip(ids, vectors, documents, metadata):
            payload = {**meta, "_content": doc}
            points.append(PointStruct(id=doc_id, vector=vector, payload=payload))

        # Batch upsert for large datasets
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[i:i + batch_size],
            )

        logger.info(f"Upserted {len(points)} documents to Qdrant collection '{self.collection_name}'")
        return ids

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        query_vector = self._embed_query(query)
        return self.search_by_vector(query_vector, top_k, filters)

    def search_by_vector(
        self,
        embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        qdrant_filter = None
        if filters:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            qdrant_filter = Filter(must=conditions)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=top_k,
            query_filter=qdrant_filter,
        )

        return [
            {
                "id": str(hit.id),
                "content": hit.payload.get("_content", ""),
                "metadata": {k: v for k, v in hit.payload.items() if k != "_content"},
                "score": hit.score,
            }
            for hit in results
        ]

    def delete(self, ids: List[str]) -> bool:
        from qdrant_client.models import PointIdsList

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=ids),
        )
        logger.info(f"Deleted {len(ids)} points from Qdrant collection '{self.collection_name}'")
        return True

    def save(self, path: str) -> None:
        logger.info("Qdrant persistence is handled automatically for on-disk mode.")

    def load(self, path: str) -> None:
        logger.info("Qdrant loads automatically from its configured path.")

    def count(self) -> int:
        info = self.client.get_collection(self.collection_name)
        return info.points_count
