"""Tests for embedding modules — factory, batch embedder, and cache."""
import pytest
from unittest.mock import MagicMock


class TestEmbeddingFactory:
    """Test embedding model factory."""

    def test_factory_rejects_invalid_provider(self):
        from src.embeddings.embedding_factory import EmbeddingFactory
        with pytest.raises(ValueError, match="Unsupported"):
            EmbeddingFactory.get_embeddings(provider="invalid_provider")

    def test_factory_has_model_aliases(self):
        from src.embeddings.embedding_factory import EmbeddingFactory
        assert "minilm" in EmbeddingFactory.SENTENCE_TRANSFORMER_MODELS
        assert "bge-small" in EmbeddingFactory.SENTENCE_TRANSFORMER_MODELS


class TestBatchEmbedder:
    """Test batch embedding with progress tracking."""

    def test_batch_embedder_initialization(self):
        from src.embeddings.batch_embedder import BatchEmbedder
        mock_fn = MagicMock()
        embedder = BatchEmbedder(embedding_function=mock_fn, batch_size=10)
        assert embedder.batch_size == 10

    def test_batch_embedder_processes_documents(self):
        from src.embeddings.batch_embedder import BatchEmbedder
        mock_fn = MagicMock()
        mock_fn.embed_documents.return_value = [[0.1, 0.2]] * 3
        embedder = BatchEmbedder(embedding_function=mock_fn, batch_size=10, show_progress=False)
        results = embedder.embed_documents(["doc1", "doc2", "doc3"])
        assert len(results) == 3


class TestEmbeddingCache:
    """Test embedding cache with hash-based deduplication."""

    def test_cache_initialization(self):
        from src.embeddings.embedding_cache import EmbeddingCache
        import tempfile, os
        cache_path = os.path.join(tempfile.mkdtemp(), "test_cache.json")
        cache = EmbeddingCache(cache_path=cache_path, backend="json")
        assert cache.size() == 0

    def test_cache_set_and_get(self):
        from src.embeddings.embedding_cache import EmbeddingCache
        import tempfile, os
        cache_path = os.path.join(tempfile.mkdtemp(), "test_cache.json")
        cache = EmbeddingCache(cache_path=cache_path, backend="json")

        cache.set("hello", [0.1, 0.2, 0.3])
        result = cache.get("hello")
        assert result == [0.1, 0.2, 0.3]

    def test_cache_miss_returns_none(self):
        from src.embeddings.embedding_cache import EmbeddingCache
        import tempfile, os
        cache_path = os.path.join(tempfile.mkdtemp(), "test_cache.json")
        cache = EmbeddingCache(cache_path=cache_path, backend="json")
        assert cache.get("nonexistent") is None

    def test_cache_invalidation(self):
        from src.embeddings.embedding_cache import EmbeddingCache
        import tempfile, os
        cache_path = os.path.join(tempfile.mkdtemp(), "test_cache.json")
        cache = EmbeddingCache(cache_path=cache_path, backend="json")
        cache.set("hello", [0.1, 0.2])
        cache.invalidate("hello")
        assert cache.get("hello") is None
