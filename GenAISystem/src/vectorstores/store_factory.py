"""
Vector Store Factory — Config-driven store selection with all providers.
"""

import logging
from typing import Any, Optional

from .base_store import BaseVectorStore

logger = logging.getLogger(__name__)


class VectorStoreFactory:
    """Factory to create vector store instances by provider name."""

    @staticmethod
    def get_store(
        provider: str,
        embedding_function=None,
        collection_name: str = "default",
        **kwargs: Any,
    ) -> BaseVectorStore:
        """Create a vector store instance.

        Args:
            provider: One of 'faiss', 'qdrant', 'chroma', 'pinecone'.
            embedding_function: Embeddings function or LangChain Embeddings object.
            collection_name: Collection/index name.
            **kwargs: Provider-specific configuration.

        Returns:
            An initialized BaseVectorStore implementation.
        """
        provider = provider.lower().strip()

        if provider == "faiss":
            from .faiss_store import FaissStore
            return FaissStore(
                embedding_function=embedding_function,
                dimension=kwargs.get("dimension"),
                index_type=kwargs.get("index_type", "flat"),
            )

        elif provider in ("qdrant", "qdrant_store"):
            from .qdrant_store import QdrantStore
            return QdrantStore(
                collection_name=collection_name,
                embedding_function=embedding_function,
                url=kwargs.get("url"),
                api_key=kwargs.get("api_key"),
                path=kwargs.get("path"),
                dimension=kwargs.get("dimension", 384),
                distance=kwargs.get("distance", "cosine"),
            )

        elif provider in ("chroma", "chromadb"):
            from .chromadb_store import ChromaDBStore
            return ChromaDBStore(
                collection_name=collection_name,
                embedding_function=embedding_function,
                persist_directory=kwargs.get("persist_directory"),
            )

        elif provider == "pinecone":
            from .pinecone_store import PineconeStore
            return PineconeStore(
                index_name=collection_name,
                embedding_function=embedding_function,
                api_key=kwargs.get("api_key"),
                namespace=kwargs.get("namespace", ""),
                dimension=kwargs.get("dimension", 384),
                metric=kwargs.get("metric", "cosine"),
            )

        else:
            raise ValueError(
                f"Unsupported vector store provider: '{provider}'. "
                f"Choose from: faiss, qdrant, chroma, pinecone"
            )
