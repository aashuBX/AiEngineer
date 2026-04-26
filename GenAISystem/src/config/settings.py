"""
Configuration settings for GenAISystem.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── API Settings ───────────────────────────────────────────────────────
    port: int = 8001
    host: str = "0.0.0.0"

    # ── Dependencies ───────────────────────────────────────────────────────
    llm_provider: Literal["openai", "anthropic", "groq", "google", "ollama"] = "groq"
    llm_model: str = "qwen-qwq-32b"
    groq_api_key: str = ""

    embedding_provider: Literal["sentence-transformers", "openai"] = "sentence-transformers"
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── Neo4j Knowledge Graph ──────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "password"

    # ── Vector Store ───────────────────────────────────────────────────────
    vector_store_provider: Literal["faiss", "qdrant", "pinecone"] = "faiss"
    qdrant_url: str = "http://localhost:6333"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
