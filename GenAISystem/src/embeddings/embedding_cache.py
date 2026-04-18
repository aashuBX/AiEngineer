"""
Embedding Cache — Hash-based cache to avoid re-embedding unchanged documents.
Supports persistent storage via SQLite or JSON with cache invalidation.
"""

import hashlib
import json
import logging
import os
import sqlite3
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Cache embeddings to avoid re-computation for unchanged documents.

    Supports two storage backends:
    - JSON file (simple, good for small datasets)
    - SQLite (recommended for larger datasets)
    """

    def __init__(
        self,
        cache_path: str = "embedding_cache.db",
        backend: str = "sqlite",
    ):
        """
        Args:
            cache_path: Path to the cache file.
            backend: 'sqlite' or 'json'.
        """
        self.cache_path = cache_path
        self.backend = backend.lower()
        self._memory_cache: Dict[str, List[float]] = {}

        if self.backend == "sqlite":
            self._init_sqlite()
        elif self.backend == "json":
            self._load_json_cache()

    # ------------------------------------------------------------------
    # SQLite Backend
    # ------------------------------------------------------------------

    def _init_sqlite(self):
        """Initialize SQLite cache table."""
        self._conn = sqlite3.connect(self.cache_path)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS embedding_cache (
                text_hash TEXT PRIMARY KEY,
                embedding TEXT NOT NULL,
                model_name TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # JSON Backend
    # ------------------------------------------------------------------

    def _load_json_cache(self):
        """Load JSON cache from disk."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r") as f:
                    self._memory_cache = json.load(f)
                logger.info(f"Loaded {len(self._memory_cache)} cached embeddings from {self.cache_path}")
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Failed to load JSON cache: {e}. Starting fresh.")
                self._memory_cache = {}

    def _save_json_cache(self):
        """Persist JSON cache to disk."""
        with open(self.cache_path, "w") as f:
            json.dump(self._memory_cache, f)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_text(text: str, model_name: str = "") -> str:
        """Create a deterministic hash for a text + model combination."""
        content = f"{model_name}:{text}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get(self, text: str, model_name: str = "") -> Optional[List[float]]:
        """Retrieve a cached embedding.

        Args:
            text: The original text.
            model_name: Optional model identifier for cache key differentiation.

        Returns:
            The cached embedding vector, or None if not found.
        """
        key = self._hash_text(text, model_name)

        if self.backend == "sqlite":
            cursor = self._conn.execute(
                "SELECT embedding FROM embedding_cache WHERE text_hash = ?",
                (key,),
            )
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
            return None

        else:  # json
            return self._memory_cache.get(key)

    def set(self, text: str, embedding: List[float], model_name: str = "") -> None:
        """Store an embedding in the cache.

        Args:
            text: The original text.
            embedding: The embedding vector to cache.
            model_name: Optional model identifier.
        """
        key = self._hash_text(text, model_name)

        if self.backend == "sqlite":
            self._conn.execute(
                "INSERT OR REPLACE INTO embedding_cache (text_hash, embedding, model_name) VALUES (?, ?, ?)",
                (key, json.dumps(embedding), model_name),
            )
            self._conn.commit()
        else:  # json
            self._memory_cache[key] = embedding
            self._save_json_cache()

    def get_or_compute(
        self,
        text: str,
        embed_fn,
        model_name: str = "",
    ) -> List[float]:
        """Get from cache or compute and store.

        Args:
            text: Text to embed.
            embed_fn: Function that takes a string and returns an embedding vector.
            model_name: Optional model identifier.

        Returns:
            The embedding vector.
        """
        cached = self.get(text, model_name)
        if cached is not None:
            return cached

        embedding = embed_fn(text)
        self.set(text, embedding, model_name)
        return embedding

    def batch_get_or_compute(
        self,
        texts: List[str],
        embed_fn,
        model_name: str = "",
    ) -> List[List[float]]:
        """Batch version: check cache first, compute only uncached texts.

        Args:
            texts: List of texts to embed.
            embed_fn: Function that takes a list of strings and returns a list of embeddings.
            model_name: Optional model identifier.

        Returns:
            List of embedding vectors.
        """
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        for i, text in enumerate(texts):
            cached = self.get(text, model_name)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            logger.info(f"Cache hit: {len(texts) - len(uncached_texts)}/{len(texts)}. "
                        f"Computing {len(uncached_texts)} new embeddings.")
            new_embeddings = embed_fn(uncached_texts)
            for idx, text, emb in zip(uncached_indices, uncached_texts, new_embeddings):
                results[idx] = emb
                self.set(text, emb, model_name)
        else:
            logger.info(f"Cache hit: {len(texts)}/{len(texts)}. All embeddings cached.")

        return results

    # ------------------------------------------------------------------
    # Cache Management
    # ------------------------------------------------------------------

    def invalidate(self, text: str, model_name: str = "") -> None:
        """Remove a specific entry from the cache."""
        key = self._hash_text(text, model_name)
        if self.backend == "sqlite":
            self._conn.execute("DELETE FROM embedding_cache WHERE text_hash = ?", (key,))
            self._conn.commit()
        else:
            self._memory_cache.pop(key, None)
            self._save_json_cache()

    def clear(self) -> None:
        """Clear the entire cache."""
        if self.backend == "sqlite":
            self._conn.execute("DELETE FROM embedding_cache")
            self._conn.commit()
        else:
            self._memory_cache.clear()
            self._save_json_cache()
        logger.info("Embedding cache cleared.")

    def size(self) -> int:
        """Return the number of cached embeddings."""
        if self.backend == "sqlite":
            cursor = self._conn.execute("SELECT COUNT(*) FROM embedding_cache")
            return cursor.fetchone()[0]
        return len(self._memory_cache)

    def close(self):
        """Close the database connection (SQLite only)."""
        if self.backend == "sqlite" and hasattr(self, "_conn"):
            self._conn.close()
