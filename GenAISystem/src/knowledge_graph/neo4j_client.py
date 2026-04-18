"""
Neo4j Client — Connection management, session helpers, and vector index support.
"""

import logging
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Manages Neo4j driver lifecycle, sessions, and vector index operations."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
        max_connection_pool_size: int = 50,
    ):
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.max_connection_pool_size = max_connection_pool_size
        self._driver = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self):
        """Establish a connection to Neo4j with connection pooling."""
        try:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                max_connection_pool_size=self.max_connection_pool_size,
            )
            # Verify connectivity
            self._driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        """Close the driver and release all pooled connections."""
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed.")

    @contextmanager
    def session(self):
        """Yield a Neo4j session that auto-closes on exit."""
        if not self._driver:
            self.connect()
        session = self._driver.session(database=self.database)
        try:
            yield session
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        write: bool = False,
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results as list of dicts.

        Args:
            query: Cypher query string.
            parameters: Optional dict of query parameters.
            write: If True, run inside a write transaction; else read transaction.
        """
        parameters = parameters or {}

        with self.session() as session:
            if write:
                result = session.run(query, parameters)
            else:
                result = session.run(query, parameters)

            return [record.data() for record in result]

    def execute_write(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Shortcut for write transactions (MERGE, CREATE, SET)."""
        return self.execute_query(query, parameters, write=True)

    # ------------------------------------------------------------------
    # Vector index management
    # ------------------------------------------------------------------

    def create_vector_index(
        self,
        index_name: str,
        label: str,
        property_name: str,
        dimensions: int,
        similarity_function: str = "cosine",
    ):
        """Create a vector index in Neo4j 5.x+ for hybrid graph+vector retrieval.

        Args:
            index_name: Name of the vector index.
            label: Node label to index.
            property_name: Property containing the embedding vector.
            dimensions: Dimensionality of the embedding vectors.
            similarity_function: 'cosine', 'euclidean', or 'dotProduct'.
        """
        query = f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (n:{label})
        ON (n.{property_name})
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {dimensions},
                `vector.similarity_function`: '{similarity_function}'
            }}
        }}
        """
        self.execute_write(query)
        logger.info(f"Vector index '{index_name}' created on :{label}.{property_name}")

    def vector_search(
        self,
        index_name: str,
        query_vector: List[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Perform a vector similarity search on a Neo4j vector index.

        Args:
            index_name: Name of the vector index.
            query_vector: The query embedding vector.
            top_k: Number of nearest neighbors to return.
        """
        query = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
        YIELD node, score
        RETURN node, score
        ORDER BY score DESC
        """
        return self.execute_query(query, {
            "index_name": index_name,
            "top_k": top_k,
            "query_vector": query_vector,
        })

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """Quick connectivity check."""
        try:
            result = self.execute_query("RETURN 1 AS healthy")
            return bool(result)
        except Exception:
            return False

    def clear_database(self):
        """Delete all nodes and relationships — use with caution!"""
        self.execute_write("MATCH (n) DETACH DELETE n")
        logger.warning("All nodes and relationships have been removed from the database.")
