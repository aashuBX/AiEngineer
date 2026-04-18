"""Tests for retrieval modules — vector, keyword, hybrid, query router."""
import pytest
from unittest.mock import MagicMock


class TestKeywordRetriever:
    """Test BM25 keyword retriever."""

    def test_keyword_retriever_initialization(self):
        from src.retrieval.keyword_retriever import KeywordRetriever
        docs = ["Hello world", "Goodbye world"]
        retriever = KeywordRetriever(documents=docs)
        assert retriever is not None

    def test_keyword_retriever_search(self):
        from src.retrieval.keyword_retriever import KeywordRetriever
        docs = [
            "FAISS is a library for similarity search.",
            "BM25 is a ranking function used in information retrieval.",
            "Neural networks process data through layers.",
        ]
        retriever = KeywordRetriever(documents=docs)
        results = retriever.retrieve("similarity search", top_k=2)
        assert len(results) > 0
        # The FAISS doc should rank first
        assert "FAISS" in results[0]["content"]


class TestVectorRetriever:
    """Test vector similarity retriever."""

    def test_vector_retriever_initialization(self):
        from src.retrieval.vector_retriever import VectorRetriever
        mock_store = MagicMock()
        retriever = VectorRetriever(vector_store=mock_store, top_k=5)
        assert retriever.top_k == 5

    def test_vector_retriever_delegates_to_store(self):
        from src.retrieval.vector_retriever import VectorRetriever
        mock_store = MagicMock()
        mock_store.search.return_value = [{"content": "test", "score": 0.9}]
        retriever = VectorRetriever(vector_store=mock_store)
        results = retriever.retrieve("test query")
        mock_store.search.assert_called_once()
        assert len(results) == 1


class TestQueryRouter:
    """Test query routing logic."""

    def test_rule_based_routing_keyword(self):
        from src.retrieval.query_router import QueryRouter
        mock_llm = MagicMock()
        router = QueryRouter(llm=mock_llm, use_llm=False)
        assert router.route_query("Python 3.12 release date") == "keyword"

    def test_rule_based_routing_graph(self):
        from src.retrieval.query_router import QueryRouter
        mock_llm = MagicMock()
        router = QueryRouter(llm=mock_llm, use_llm=False)
        assert router.route_query("who owns Tesla?") == "graph"

    def test_rule_based_routing_short_query(self):
        from src.retrieval.query_router import QueryRouter
        mock_llm = MagicMock()
        router = QueryRouter(llm=mock_llm, use_llm=False)
        result = router.route_query("FAISS")
        assert result == "keyword"
