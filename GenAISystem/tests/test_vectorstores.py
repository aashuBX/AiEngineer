"""Tests for vector store modules — base, factory, and individual stores."""
import pytest
from unittest.mock import MagicMock


class TestBaseVectorStore:
    """Test the abstract base store interface."""

    def test_base_store_is_abstract(self):
        from src.vectorstores.base_store import BaseVectorStore
        with pytest.raises(TypeError):
            BaseVectorStore()  # Cannot instantiate abstract class


class TestVectorStoreFactory:
    """Test factory-based store creation."""

    def test_factory_rejects_invalid_provider(self):
        from src.vectorstores.store_factory import VectorStoreFactory
        with pytest.raises(ValueError, match="Unsupported"):
            VectorStoreFactory.get_store("invalid_db")

    def test_factory_creates_faiss(self):
        from src.vectorstores.store_factory import VectorStoreFactory
        mock_fn = MagicMock()
        store = VectorStoreFactory.get_store("faiss", embedding_function=mock_fn)
        assert store is not None

    def test_factory_creates_chroma(self):
        from src.vectorstores.store_factory import VectorStoreFactory
        store = VectorStoreFactory.get_store("chroma")
        assert store is not None

    def test_factory_creates_qdrant(self):
        from src.vectorstores.store_factory import VectorStoreFactory
        mock_fn = MagicMock()
        store = VectorStoreFactory.get_store("qdrant", embedding_function=mock_fn)
        assert store is not None

    def test_factory_creates_pinecone(self):
        from src.vectorstores.store_factory import VectorStoreFactory
        mock_fn = MagicMock()
        store = VectorStoreFactory.get_store("pinecone", embedding_function=mock_fn)
        assert store is not None
