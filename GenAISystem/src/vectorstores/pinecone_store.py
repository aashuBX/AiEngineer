"""
Pinecone Vector Store — Managed cloud vector database with namespace partitioning.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

from .base_store import BaseVectorStore

logger = logging.getLogger(__name__)


class PineconeStore(BaseVectorStore):
    """Pinecone-backed vector store for production cloud deployments."""

    def __init__(
        self,
        index_name: str,
        embedding_function,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        namespace: str = "",
        dimension: int = 384,
        metric: str = "cosine",
    ):
        """
        Args:
            index_name: Pinecone index name.
            embedding_function: LangChain Embeddings or callable.
            api_key: Pinecone API key (or set PINECONE_API_KEY env var).
            environment: Pinecone environment (e.g., 'us-east-1-aws').
            namespace: Namespace for logical partitioning within the index.
            dimension: Embedding dimensionality.
            metric: Distance metric: 'cosine', 'euclidean', 'dotproduct'.
        """
        self.index_name = index_name
        self.embedding_function = embedding_function
        self.api_key = api_key
        self.environment = environment
        self.namespace = namespace
        self.dimension = dimension
        self.metric = metric
        self._index = None

    @property
    def index(self):
        """Lazy-init the Pinecone index."""
        if self._index is None:
            from pinecone import Pinecone, ServerlessSpec
            import os

            api_key = self.api_key or os.getenv("PINECONE_API_KEY")
            pc = Pinecone(api_key=api_key)

            # Create index if it doesn't exist
            existing = [idx.name for idx in pc.list_indexes()]
            if self.index_name not in existing:
                pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
                logger.info(f"Created Pinecone index '{self.index_name}' (dim={self.dimension})")

            self._index = pc.Index(self.index_name)
        return self._index

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
        vectors = embeddings or self._embed(documents)
        ids = ids or [str(uuid.uuid4()) for _ in documents]
        metadata = metadata or [{} for _ in documents]

        # Pinecone upsert format: list of (id, vector, metadata) tuples
        records = []
        for doc_id, vec, doc, meta in zip(ids, vectors, documents, metadata):
            payload = {**meta, "_content": doc}
            records.append({"id": doc_id, "values": vec, "metadata": payload})

        # Batch upsert (Pinecone recommends ≤100 per batch)
        batch_size = 100
        for i in range(0, len(records), batch_size):
            self.index.upsert(
                vectors=records[i:i + batch_size],
                namespace=self.namespace,
            )

        logger.info(f"Upserted {len(records)} vectors to Pinecone index '{self.index_name}'")
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
        kwargs = {
            "vector": embedding,
            "top_k": top_k,
            "include_metadata": True,
            "namespace": self.namespace,
        }
        if filters:
            kwargs["filter"] = filters

        response = self.index.query(**kwargs)

        results = []
        for match in response.get("matches", []):
            meta = match.get("metadata", {})
            results.append({
                "id": match["id"],
                "content": meta.pop("_content", ""),
                "metadata": meta,
                "score": match.get("score", 0.0),
            })
        return results

    def delete(self, ids: List[str]) -> bool:
        self.index.delete(ids=ids, namespace=self.namespace)
        logger.info(f"Deleted {len(ids)} vectors from Pinecone index '{self.index_name}'")
        return True

    def save(self, path: str) -> None:
        logger.info("Pinecone is a managed cloud service — no local save needed.")

    def load(self, path: str) -> None:
        logger.info("Pinecone is a managed cloud service — no local load needed.")

    def count(self) -> int:
        stats = self.index.describe_index_stats()
        ns_stats = stats.get("namespaces", {}).get(self.namespace, {})
        return ns_stats.get("vector_count", 0)
