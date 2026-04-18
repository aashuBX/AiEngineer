"""
Embeddings Factory — provides SentenceTransformers or OpenAI embeddings.
"""

from functools import lru_cache
from typing import Any

from langchain_core.embeddings import Embeddings

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Factory to get the configured embedding model."""
    provider = settings.embedding_provider.lower()
    
    logger.info(f"Initializing Embeddings: {provider} ({settings.embedding_model})")
    
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=settings.embedding_model)
        
    elif provider == "sentence-transformers":
        # Using HuggingFace embeddings running locally
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={'device': 'cpu'},  # specify 'cuda' or 'mps' if available
            encode_kwargs={'normalize_embeddings': True}
        )
        
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
