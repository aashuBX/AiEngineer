"""Tests for knowledge graph modules."""
import pytest
from unittest.mock import MagicMock


class TestNeo4jClient:
    """Test Neo4j client initialization (no live DB required)."""

    def test_client_initialization(self):
        from src.knowledge_graph.neo4j_client import Neo4jClient
        client = Neo4jClient(uri="bolt://localhost:7687")
        assert client.uri == "bolt://localhost:7687"
        assert client.database == "neo4j"

    def test_client_default_values(self):
        from src.knowledge_graph.neo4j_client import Neo4jClient
        client = Neo4jClient()
        assert client.max_connection_pool_size == 50


class TestEntityExtractor:
    """Test entity extraction schemas."""

    def test_extracted_entity_model(self):
        from src.knowledge_graph.entity_extractor import ExtractedEntity
        entity = ExtractedEntity(name="Python", entity_type="Technology")
        assert entity.name == "Python"
        assert entity.entity_type == "Technology"

    def test_extraction_result_model(self):
        from src.knowledge_graph.entity_extractor import ExtractionResult
        result = ExtractionResult()
        assert result.entities == []
        assert result.relationships == []

    def test_entity_extractor_initialization(self):
        from src.knowledge_graph.entity_extractor import EntityExtractor
        mock_llm = MagicMock()
        extractor = EntityExtractor(llm=mock_llm)
        assert extractor is not None
        assert len(extractor.allowed_entity_types) > 0


class TestTripletGenerator:
    """Test triplet generation schemas."""

    def test_triplet_model(self):
        from src.knowledge_graph.triplet_generator import Triplet
        triplet = Triplet(subject="Python", predicate="IS_A", object="Language")
        assert triplet.subject == "Python"
        assert triplet.confidence == 1.0

    def test_triplet_generator_initialization(self):
        from src.knowledge_graph.triplet_generator import TripletGenerator
        mock_llm = MagicMock()
        gen = TripletGenerator(llm=mock_llm)
        assert gen.min_confidence == 0.5

    def test_deduplication(self):
        from src.knowledge_graph.triplet_generator import TripletGenerator, Triplet
        t1 = Triplet(subject="A", predicate="REL", object="B", confidence=0.8)
        t2 = Triplet(subject="a", predicate="REL", object="b", confidence=0.9)
        deduped = TripletGenerator._deduplicate([t1, t2])
        assert len(deduped) == 1
        assert deduped[0].confidence == 0.9  # Higher confidence wins


class TestGraphRetriever:
    """Test graph retriever initialization."""

    def test_retriever_initialization(self):
        from src.knowledge_graph.graph_retriever import GraphRetriever
        mock_client = MagicMock()
        mock_llm = MagicMock()
        retriever = GraphRetriever(neo4j_client=mock_client, llm=mock_llm)
        assert retriever.max_results == 10
