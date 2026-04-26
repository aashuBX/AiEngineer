"""
Semantic Cache — caches LLM responses keyed by query embeddings.

Strategy:
  1. On each query, embed the query and compare against stored query embeddings.
  2. If cosine similarity > threshold → return cached answer (cache HIT).
  3. After LLM generates a new answer → store (embedding, answer, timestamp) in cache.

Backend:
  - Redis (RediSearch) if REDIS_URL is configured → persistent across restarts.
  - In-memory (numpy cosine) if Redis is unavailable → fast, no dependencies needed.

TTL:
  - Entries older than `semantic_cache_ttl_hours` are automatically expired.
"""

import time
from typing import Any, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SemanticCache:
    """
    Query-level semantic cache. Embeddings-based similarity matching
    with automatic in-memory fallback if Redis is unavailable.
    """

    def __init__(
        self,
        embeddings: Any,
        threshold: float = 0.95,
        ttl_hours: int = 24,
        redis_url: str = "",
    ):
        self.embeddings = embeddings
        self.threshold = threshold
        self.ttl_seconds = ttl_hours * 3600
        self._redis = self._connect_redis(redis_url)
        # In-memory fallback: list of (embedding, answer, timestamp)
        self._memory: list[tuple[list[float], str, float]] = []

    # ── Redis connection ───────────────────────────────────────────────────

    def _connect_redis(self, redis_url: str) -> Optional[Any]:
        if not redis_url:
            logger.info("SemanticCache: No REDIS_URL set — using in-memory fallback")
            return None
        try:
            import redis as redis_lib
            client = redis_lib.from_url(redis_url, decode_responses=False)
            client.ping()
            logger.info(f"SemanticCache: Connected to Redis at {redis_url}")
            return client
        except Exception as e:
            logger.warning(f"SemanticCache: Redis unavailable ({e}) — using in-memory fallback")
            return None

    # ── Public API ─────────────────────────────────────────────────────────

    def get(self, query: str) -> Optional[dict]:
        """
        Check if query hits the cache.

        Returns:
            dict with 'answer' and 'cache_hit'=True if hit, else None.
        """
        try:
            query_embedding = self.embeddings.embed_query(query)
            if self._redis:
                return self._redis_get(query_embedding)
            return self._memory_get(query_embedding)
        except Exception as e:
            logger.error(f"SemanticCache.get failed: {e}")
            return None

    def set(self, query: str, answer: str) -> None:
        """Store a (query, answer) pair in the cache."""
        try:
            query_embedding = self.embeddings.embed_query(query)
            if self._redis:
                self._redis_set(query_embedding, answer)
            else:
                self._memory_set(query_embedding, answer)
        except Exception as e:
            logger.error(f"SemanticCache.set failed: {e}")

    def clear(self) -> None:
        """Clear all cache entries."""
        self._memory.clear()
        logger.info("SemanticCache: Cleared in-memory cache")

    # ── In-memory backend ──────────────────────────────────────────────────

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        import numpy as np
        a_arr = np.array(a, dtype=np.float32)
        b_arr = np.array(b, dtype=np.float32)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))

    def _memory_get(self, query_embedding: list[float]) -> Optional[dict]:
        now = time.time()
        best_score = 0.0
        best_answer = None

        # Expire old entries inline
        self._memory = [
            (emb, ans, ts) for emb, ans, ts in self._memory
            if (now - ts) < self.ttl_seconds
        ]

        for emb, answer, _ in self._memory:
            score = self._cosine_similarity(query_embedding, emb)
            if score > best_score:
                best_score = score
                best_answer = answer

        if best_score >= self.threshold and best_answer is not None:
            logger.info(f"SemanticCache HIT (score={best_score:.4f})")
            return {"answer": best_answer, "cache_hit": True, "cache_score": best_score}

        logger.debug(f"SemanticCache MISS (best_score={best_score:.4f})")
        return None

    def _memory_set(self, query_embedding: list[float], answer: str) -> None:
        self._memory.append((query_embedding, answer, time.time()))
        # Keep cache bounded to 1000 entries (FIFO eviction)
        if len(self._memory) > 1000:
            self._memory = self._memory[-1000:]
        logger.debug(f"SemanticCache: stored entry (total={len(self._memory)})")

    # ── Redis backend ──────────────────────────────────────────────────────

    def _redis_get(self, query_embedding: list[float]) -> Optional[dict]:
        """
        Simple Redis scan-based similarity search.
        For production, upgrade to RediSearch VSS (vector similarity search).
        """
        import json
        try:
            keys = self._redis.keys("semcache:*")
            best_score = 0.0
            best_answer = None
            now = time.time()

            for key in keys:
                raw = self._redis.get(key)
                if not raw:
                    continue
                entry = json.loads(raw)
                # Skip expired
                if now - entry.get("timestamp", 0) > self.ttl_seconds:
                    self._redis.delete(key)
                    continue
                score = self._cosine_similarity(query_embedding, entry["embedding"])
                if score > best_score:
                    best_score = score
                    best_answer = entry["answer"]

            if best_score >= self.threshold and best_answer is not None:
                logger.info(f"SemanticCache Redis HIT (score={best_score:.4f})")
                return {"answer": best_answer, "cache_hit": True, "cache_score": best_score}
        except Exception as e:
            logger.error(f"Redis get failed: {e}")
        return None

    def _redis_set(self, query_embedding: list[float], answer: str) -> None:
        import json
        import hashlib
        key = "semcache:" + hashlib.sha256(str(query_embedding[:8]).encode()).hexdigest()[:16]
        entry = {
            "embedding": query_embedding,
            "answer": answer,
            "timestamp": time.time(),
        }
        self._redis.setex(key, self.ttl_seconds, json.dumps(entry))
        logger.debug(f"SemanticCache: stored in Redis key={key}")
