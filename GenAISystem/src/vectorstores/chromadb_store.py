"""
ChromaDB Vector Store — Persistent and in-memory modes with metadata filtering.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

from .base_store import BaseVectorStore

logger = logging.getLogger(__name__)


class ChromaDBStore(BaseVectorStore):
    """ChromaDB-backed vector store with persistent and in-memory modes."""

    def __init__(
        self,
        collection_name: str = "default",
        embedding_function=None,
        persist_directory: Optional[str] = None,
    ):
        """
        Args:
            collection_name: ChromaDB collection name.
            embedding_function: LangChain Embeddings or callable. ChromaDB also supports
                built-in embedding functions.
            persist_directory: Path for persistent storage. None = ephemeral in-memory.
        """
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory
        self._client = None
        self._collection = None

    @property
    def collection(self):
        """Lazy-init ChromaDB client and collection."""
        if self._collection is None:
            import chromadb

            if self.persist_directory:
                self._client = chromadb.PersistentClient(path=self.persist_directory)
            else:
                self._client = chromadb.Client()

            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"ChromaDB collection '{self.collection_name}' ready "
                        f"(persistent={self.persist_directory is not None})")

        return self._collection

    def _embed(self, texts: List[str]) -> List[List[float]]:
        if self.embedding_function is None:
            return None  # Let ChromaDB use its default
        if hasattr(self.embedding_function, "embed_documents"):
            return self.embedding_function.embed_documents(texts)
        return [self.embedding_function(t) for t in texts]

    def _embed_query(self, text: str) -> List[float]:
        if self.embedding_function is None:
            return None
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
        ids = ids or [str(uuid.uuid4()) for _ in documents]
        metadata = metadata or [{}  for _ in documents]

        # Compute embeddings if not provided and we have an embedding function
        if embeddings is None and self.embedding_function is not None:
            embeddings = self._embed(documents)

        kwargs = {
            "ids": ids,
            "documents": documents,
            "metadatas": metadata,
        }
        if embeddings is not None:
            kwargs["embeddings"] = embeddings

        self.collection.upsert(**kwargs)
        logger.info(f"Added {len(documents)} documents to ChromaDB collection '{self.collection_name}'")
        return ids

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        query_embedding = self._embed_query(query)

        kwargs = {"n_results": top_k}
        if filters:
            kwargs["where"] = filters

        if query_embedding is not None:
            kwargs["query_embeddings"] = [query_embedding]
        else:
            kwargs["query_texts"] = [query]

        results = self.collection.query(**kwargs)
        return self._format_results(results)

    def search_by_vector(
        self,
        embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        kwargs = {
            "query_embeddings": [embedding],
            "n_results": top_k,
        }
        if filters:
            kwargs["where"] = filters

        results = self.collection.query(**kwargs)
        return self._format_results(results)

    def delete(self, ids: List[str]) -> bool:
        self.collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} documents from ChromaDB collection '{self.collection_name}'")
        return True

    def save(self, path: str) -> None:
        logger.info("ChromaDB PersistentClient auto-persists. No explicit save needed.")

    def load(self, path: str) -> None:
        """Reinitialize with a different persist directory."""
        self.persist_directory = path
        self._client = None
        self._collection = None
        _ = self.collection  # trigger re-initialization

    def count(self) -> int:
        return self.collection.count()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_results(results: dict) -> List[Dict[str, Any]]:
        """Convert ChromaDB query results to standard format."""
        formatted = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            formatted.append({
                "id": doc_id,
                "content": doc or "",
                "metadata": meta or {},
                "score": dist,
            })
        return formatted
