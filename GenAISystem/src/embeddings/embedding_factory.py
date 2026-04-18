"""
Embedding Factory — Multi-provider embedding model factory.
Supports SentenceTransformers, OpenAI, Google, HuggingFace, and Cohere.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class EmbeddingFactory:
    """Factory for creating embedding model instances across providers."""

    SENTENCE_TRANSFORMER_MODELS = {
        "minilm": "all-MiniLM-L6-v2",
        "mpnet": "all-mpnet-base-v2",
        "bge-small": "BAAI/bge-small-en-v1.5",
        "bge-base": "BAAI/bge-base-en-v1.5",
    }

    @staticmethod
    def get_embeddings(
        provider: str = "openai",
        model_name: Optional[str] = None,
        **kwargs,
    ):
        """Create an embedding model instance.

        Args:
            provider: One of 'openai', 'sentence-transformers', 'huggingface',
                      'google', 'cohere'.
            model_name: Model identifier (provider-specific).
            **kwargs: Additional provider-specific arguments.

        Returns:
            A LangChain-compatible embeddings object.
        """
        provider = provider.lower().strip()

        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model=model_name or "text-embedding-3-small",
                **kwargs,
            )

        elif provider in ("sentence-transformers", "sentence_transformers", "st"):
            from langchain_community.embeddings import HuggingFaceEmbeddings
            resolved_model = EmbeddingFactory.SENTENCE_TRANSFORMER_MODELS.get(
                model_name, model_name or "all-MiniLM-L6-v2"
            )
            return HuggingFaceEmbeddings(
                model_name=resolved_model,
                model_kwargs={"device": kwargs.get("device", "cpu")},
                encode_kwargs={"normalize_embeddings": kwargs.get("normalize", True)},
            )

        elif provider in ("huggingface", "hf"):
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2",
                **kwargs,
            )

        elif provider == "google":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(
                model=model_name or "models/embedding-001",
                **kwargs,
            )

        elif provider == "cohere":
            from langchain_cohere import CohereEmbeddings
            return CohereEmbeddings(
                model=model_name or "embed-english-v3.0",
                **kwargs,
            )

        else:
            raise ValueError(
                f"Unsupported embedding provider: '{provider}'. "
                f"Choose from: openai, sentence-transformers, huggingface, google, cohere"
            )
