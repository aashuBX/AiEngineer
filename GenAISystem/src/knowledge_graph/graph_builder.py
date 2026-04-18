"""
Knowledge Graph Builder — writes entities and relationships into Neo4j.
"""

from typing import Any
from langchain_core.documents import Document

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class KnowledgeGraphBuilder:
    """Extracts triples via LLM and builds a Neo4j knowledge graph."""

    def __init__(self, llm: Any):
        self.llm = llm
        self._graph = None

    def _get_graph(self):
        """Lazy load and return LangChain Neo4jGraph connection."""
        if self._graph is None:
            try:
                from langchain_neo4j import Neo4jGraph
                self._graph = Neo4jGraph(
                    url=settings.neo4j_uri,
                    username=settings.neo4j_username,
                    password=settings.neo4j_password,
                )
            except ImportError:
                logger.error("langchain-neo4j or neo4j driver not installed")
                raise
        return self._graph

    def process_documents(self, documents: list[Document]) -> None:
        """
        Extract graph triples from documents and load them into Neo4j.
        Uses LangChain's LLMGraphTransformer.
        """
        try:
            from langchain_experimental.graph_transformers import LLMGraphTransformer
        except ImportError:
            logger.warning("langchain-experimental not installed. Skipping Graph building.")
            return

        logger.info(f"Extracting entities and relationships for {len(documents)} docs")

        # Configure transformer
        transformer = LLMGraphTransformer(
            llm=self.llm,
            # We could constrain node/rel types here
            # allowed_nodes=["Person", "Organization", "Concept"],
            # allowed_relationships=["WORKS_AT", "USES", "RELATES_TO"],
        )

        try:
            graph_documents = transformer.convert_to_graph_documents(documents)
            logger.info(f"Generated {len(graph_documents)} graph documents")

            # Persist to Neo4j
            graph = self._get_graph()
            graph.add_graph_documents(graph_documents, baseEntityLabel=True)
            logger.info("Successfully persisted graph data to Neo4j")
        except Exception as e:
            logger.error(f"Failed to process graph documents: {e}")

    def query_schema(self) -> str:
        """Get the current schema of the graph database."""
        graph = self._get_graph()
        graph.refresh_schema()
        return graph.schema
