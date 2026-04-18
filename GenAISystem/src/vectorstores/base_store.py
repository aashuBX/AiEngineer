"""
Base Vector Store — Abstract interface for all vector store implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseVectorStore(ABC):
    """Abstract base class for vector database backends.

    All concrete implementations (FAISS, Qdrant, ChromaDB, Pinecone) must
    implement these methods to provide a unified retrieval interface.
    """

    @abstractmethod
    def add_documents(
        self,
        documents: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add documents (with optional pre-computed embeddings) to the store.

        Args:
            documents: List of text content.
            embeddings: Optional pre-computed embedding vectors.
            metadata: Optional list of metadata dicts per document.
            ids: Optional list of unique IDs per document.

        Returns:
            List of document IDs that were stored.
        """
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic similarity search.

        Args:
            query: The query text (will be embedded internally).
            top_k: Number of results to return.
            filters: Optional metadata filters.

        Returns:
            List of dicts with keys: 'content', 'metadata', 'score', 'id'.
        """
        pass

    @abstractmethod
    def search_by_vector(
        self,
        embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search with a pre-computed embedding vector.

        Args:
            embedding: The query vector.
            top_k: Number of results.
            filters: Optional metadata filters.

        Returns:
            List of result dicts.
        """
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> bool:
        """Delete documents by their IDs.

        Args:
            ids: List of document IDs to remove.

        Returns:
            True if deletion was successful.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the store to disk.

        Args:
            path: Directory path to save to.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load a persisted store from disk.

        Args:
            path: Directory path to load from.
        """
        pass

    def count(self) -> int:
        """Return the number of documents in the store."""
        raise NotImplementedError("count() not implemented for this store.")
