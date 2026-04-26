"""
Vector Store Factory — provides FAISS (local), Qdrant, or Pinecone (production).
"""

import os
from typing import Any
from langchain_core.vectorstores import VectorStore

from src.config.settings import settings
from src.embeddings.factory import get_embeddings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default local paths for FAISS
FAISS_INDEX_PATH = "data/faiss_index"


def get_vector_store() -> VectorStore:
    """Factory to initialize or load the configured vector store."""
    provider = settings.vector_store_provider.lower()
    embeddings = get_embeddings()
    
    logger.info(f"Initializing Vector Store: {provider}")

    if provider == "faiss":
        from langchain_community.vectorstores import FAISS
        
        if os.path.exists(FAISS_INDEX_PATH) and os.listdir(FAISS_INDEX_PATH):
            logger.info("Loading existing FAISS index")
            return FAISS.load_local(
                folder_path=FAISS_INDEX_PATH,
                embeddings=embeddings,
                allow_dangerous_deserialization=True  # Required for FAISS locally
            )
        else:
            logger.info("Creating new FAISS index (in-memory)")
            import faiss
            from langchain_community.docstore.in_memory import InMemoryDocstore
            
            # Simple dimension check placeholder
            d = len(embeddings.embed_query("test"))
            index = faiss.IndexFlatL2(d)
            return FAISS(
                embedding_function=embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )

    elif provider == "qdrant":
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
        
        client = QdrantClient(url=settings.qdrant_url)
        return QdrantVectorStore(
            client=client,
            collection_name="genai_documents",
            embedding=embeddings,
        )

    elif provider == "pinecone":
        from langchain_pinecone import PineconeVectorStore

        logger.info(f"Connecting to Pinecone index: {settings.pinecone_index_name}")
        return PineconeVectorStore(
            index_name=settings.pinecone_index_name,
            embedding=embeddings,
            pinecone_api_key=settings.pinecone_api_key,
        )

    else:
        raise ValueError(f"Unknown vector store provider: {provider}")


def save_vector_store(store: VectorStore) -> None:
    """Save vector store to disk if applicable (like FAISS)."""
    if isinstance(store, __import__("langchain_community").vectorstores.FAISS):
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
        store.save_local(FAISS_INDEX_PATH)
        logger.info(f"Saved FAISS index to {FAISS_INDEX_PATH}")
