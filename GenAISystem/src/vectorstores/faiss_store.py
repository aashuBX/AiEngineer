"""
FAISS Vector Store — Local, high-performance vector similarity search.
Supports IndexFlatL2 and IndexIVFFlat, with metadata storage and persistence.
"""

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .base_store import BaseVectorStore

logger = logging.getLogger(__name__)


class FaissStore(BaseVectorStore):
    """FAISS-backed vector store with metadata support and disk persistence."""

    def __init__(
        self,
        embedding_function,
        dimension: Optional[int] = None,
        index_type: str = "flat",  # 'flat' or 'ivf'
        nlist: int = 100,
    ):
        """
        Args:
            embedding_function: Callable or LangChain Embeddings — converts text to vectors.
            dimension: Embedding dimensionality. Auto-detected if not provided.
            index_type: 'flat' (exact) or 'ivf' (approximate, faster for large datasets).
            nlist: Number of Voronoi cells for IVF index.
        """
        self.embedding_function = embedding_function
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        self.index = None
        self._documents: Dict[str, Dict[str, Any]] = {}  # id -> {content, metadata}
        self._id_to_idx: Dict[str, int] = {}
        self._idx_to_id: Dict[int, str] = {}

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Embed texts using the configured embedding function."""
        if hasattr(self.embedding_function, "embed_documents"):
            vectors = self.embedding_function.embed_documents(texts)
        else:
            vectors = [self.embedding_function(t) for t in texts]
        return np.array(vectors, dtype=np.float32)

    def _embed_query(self, text: str) -> np.ndarray:
        """Embed a single query text."""
        if hasattr(self.embedding_function, "embed_query"):
            vec = self.embedding_function.embed_query(text)
        else:
            vec = self.embedding_function(text)
        return np.array([vec], dtype=np.float32)

    def _init_index(self, dimension: int):
        """Lazily initialize the FAISS index."""
        import faiss

        self.dimension = dimension
        if self.index_type == "ivf":
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, self.nlist)
        else:
            self.index = faiss.IndexFlatL2(dimension)
        logger.info(f"Initialized FAISS {self.index_type} index with dimension={dimension}")

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def add_documents(
        self,
        documents: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        if embeddings is not None:
            vectors = np.array(embeddings, dtype=np.float32)
        else:
            vectors = self._embed(documents)

        if self.index is None:
            self._init_index(vectors.shape[1])

        # Train IVF index if needed
        if self.index_type == "ivf" and not self.index.is_trained:
            self.index.train(vectors)

        ids = ids or [str(uuid.uuid4()) for _ in documents]
        metadata = metadata or [{} for _ in documents]

        start_idx = self.index.ntotal
        self.index.add(vectors)

        for i, (doc_id, doc, meta) in enumerate(zip(ids, documents, metadata)):
            idx = start_idx + i
            self._documents[doc_id] = {"content": doc, "metadata": meta}
            self._id_to_idx[doc_id] = idx
            self._idx_to_id[idx] = doc_id

        logger.info(f"Added {len(documents)} documents to FAISS. Total: {self.index.ntotal}")
        return ids

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if self.index is None or self.index.ntotal == 0:
            return []

        query_vector = self._embed_query(query)
        return self._search_vectors(query_vector, top_k, filters)

    def search_by_vector(
        self,
        embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if self.index is None or self.index.ntotal == 0:
            return []

        query_vector = np.array([embedding], dtype=np.float32)
        return self._search_vectors(query_vector, top_k, filters)

    def _search_vectors(
        self,
        query_vector: np.ndarray,
        top_k: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Internal search with optional post-filtering."""
        # Fetch more than needed for post-filtering
        fetch_k = top_k * 3 if filters else top_k
        distances, indices = self.index.search(query_vector, min(fetch_k, self.index.ntotal))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            doc_id = self._idx_to_id.get(int(idx))
            if not doc_id or doc_id not in self._documents:
                continue

            doc_data = self._documents[doc_id]

            # Apply metadata filters
            if filters and not self._matches_filters(doc_data["metadata"], filters):
                continue

            results.append({
                "id": doc_id,
                "content": doc_data["content"],
                "metadata": doc_data["metadata"],
                "score": float(dist),
            })
            if len(results) >= top_k:
                break

        return results

    def delete(self, ids: List[str]) -> bool:
        """Remove documents by ID. Note: FAISS doesn't support true deletion,
        so we mark them as deleted in our metadata mapping."""
        for doc_id in ids:
            self._documents.pop(doc_id, None)
            idx = self._id_to_idx.pop(doc_id, None)
            if idx is not None:
                self._idx_to_id.pop(idx, None)
        logger.info(f"Soft-deleted {len(ids)} documents from FAISS store.")
        return True

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        import faiss

        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump({
                "documents": self._documents,
                "id_to_idx": self._id_to_idx,
                "idx_to_id": {str(k): v for k, v in self._idx_to_id.items()},
                "dimension": self.dimension,
            }, f)
        logger.info(f"FAISS store saved to {path}")

    def load(self, path: str) -> None:
        import faiss

        self.index = faiss.read_index(os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "metadata.json"), "r") as f:
            data = json.load(f)
        self._documents = data["documents"]
        self._id_to_idx = data["id_to_idx"]
        self._idx_to_id = {int(k): v for k, v in data["idx_to_id"].items()}
        self.dimension = data.get("dimension")
        logger.info(f"FAISS store loaded from {path}. Total: {self.index.ntotal}")

    def count(self) -> int:
        return self.index.ntotal if self.index else 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _matches_filters(metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if document metadata matches all filter conditions."""
        for key, value in filters.items():
            if metadata.get(key) != value:
                return False
        return True
