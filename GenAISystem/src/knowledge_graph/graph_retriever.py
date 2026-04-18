"""
Graph Retriever — Natural-language-to-Cypher query generation and multi-hop traversal.
Supports both text-to-Cypher and vector-enriched graph retrieval.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class GraphRetriever:
    """Retrieve relevant context from a Neo4j knowledge graph.

    Supports:
    - LLM-generated Cypher from natural language queries
    - Multi-hop relationship traversal
    - Vector-enriched graph retrieval (embedding search + neighbor traversal)
    - GraphCypherQAChain integration
    """

    CYPHER_GENERATION_PROMPT = """You are a Neo4j Cypher expert. Convert the user's natural language
question into a valid Cypher query.

Graph Schema:
{schema}

Rules:
1. Only use node labels and relationship types from the schema above.
2. Always LIMIT results to {max_results}.
3. Return meaningful properties, not just node IDs.
4. Use case-insensitive matching with toLower() where appropriate.

Question: {question}

Respond with ONLY the Cypher query, no explanations.
"""

    def __init__(
        self,
        neo4j_client,
        llm,
        max_results: int = 10,
        vector_index_name: Optional[str] = None,
    ):
        self.client = neo4j_client
        self.llm = llm
        self.max_results = max_results
        self.vector_index_name = vector_index_name
        self._graph_schema: Optional[str] = None

    # ------------------------------------------------------------------
    # Schema Discovery
    # ------------------------------------------------------------------

    def get_graph_schema(self) -> str:
        """Discover the current graph schema (labels, relationships, properties)."""
        if self._graph_schema:
            return self._graph_schema

        try:
            # Get node labels
            labels = self.client.execute_query("CALL db.labels()")
            label_list = [r.get("label", "") for r in labels]

            # Get relationship types
            rels = self.client.execute_query("CALL db.relationshipTypes()")
            rel_list = [r.get("relationshipType", "") for r in rels]

            # Get property keys
            props = self.client.execute_query("CALL db.propertyKeys()")
            prop_list = [r.get("propertyKey", "") for r in props]

            self._graph_schema = (
                f"Node Labels: {', '.join(label_list)}\n"
                f"Relationship Types: {', '.join(rel_list)}\n"
                f"Property Keys: {', '.join(prop_list)}"
            )
        except Exception as e:
            logger.warning(f"Failed to discover schema: {e}")
            self._graph_schema = "Schema unavailable"

        return self._graph_schema

    # ------------------------------------------------------------------
    # Text-to-Cypher Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> str:
        """Convert a natural language query to Cypher and execute it.

        Args:
            query: Natural language question.

        Returns:
            Formatted context string from graph results.
        """
        cypher = self._generate_cypher(query)
        logger.info(f"Generated Cypher: {cypher}")

        try:
            results = self.client.execute_query(cypher)
            if not results:
                return "No relevant information found in the knowledge graph."
            return self._format_results(results)
        except Exception as e:
            logger.error(f"Cypher execution failed: {e}")
            return f"Graph query failed: {e}"

    def _generate_cypher(self, question: str) -> str:
        """Use the LLM to generate a Cypher query from natural language."""
        schema = self.get_graph_schema()
        prompt = self.CYPHER_GENERATION_PROMPT.format(
            schema=schema,
            question=question,
            max_results=self.max_results,
        )

        response = self.llm.invoke(prompt)
        # Extract the content string from the LLM response
        cypher = response.content if hasattr(response, "content") else str(response)
        return cypher.strip().strip("`").strip()

    # ------------------------------------------------------------------
    # Multi-hop Traversal
    # ------------------------------------------------------------------

    def multi_hop_retrieve(self, entity: str, hops: int = 2) -> str:
        """Retrieve context by traversing N hops from a starting entity.

        Args:
            entity: The starting entity name.
            hops: Number of relationship hops to traverse (1-3 recommended).

        Returns:
            Formatted context from the traversal.
        """
        hops = min(hops, 3)  # Safety limit
        relationship_pattern = "-[r]->".join(["(n" + str(i) + ")" for i in range(hops + 1)])

        cypher = f"""
        MATCH path = (n0)-[*1..{hops}]-(connected)
        WHERE toLower(n0.name) CONTAINS toLower($entity)
        RETURN DISTINCT connected.name AS name,
               labels(connected) AS labels,
               [r IN relationships(path) | type(r)] AS relationships
        LIMIT {self.max_results}
        """

        try:
            results = self.client.execute_query(cypher, {"entity": entity})
            if not results:
                return f"No connections found for entity '{entity}' within {hops} hops."
            return self._format_results(results)
        except Exception as e:
            logger.error(f"Multi-hop traversal failed: {e}")
            return f"Multi-hop query failed: {e}"

    # ------------------------------------------------------------------
    # Vector-enriched Graph Retrieval
    # ------------------------------------------------------------------

    def vector_graph_retrieve(
        self,
        query_embedding: List[float],
        expand_hops: int = 1,
        top_k: int = 5,
    ) -> str:
        """Find nearest nodes by vector similarity, then traverse their neighbors.

        Args:
            query_embedding: The query vector.
            expand_hops: How many hops to expand from the vector-matched nodes.
            top_k: Number of vector nearest neighbors.

        Returns:
            Combined context from vector matches and their graph neighbors.
        """
        if not self.vector_index_name:
            return "No vector index configured for graph retrieval."

        try:
            # Step 1: Vector search
            vector_results = self.client.vector_search(
                self.vector_index_name, query_embedding, top_k
            )

            if not vector_results:
                return "No vector matches found in the graph."

            # Step 2: Expand each matched node's neighborhood
            context_parts = []
            for vr in vector_results:
                node = vr.get("node", {})
                score = vr.get("score", 0)
                node_name = node.get("name", "Unknown")
                context_parts.append(f"[Vector Match | score={score:.3f}] {node_name}")

                if expand_hops > 0 and node_name:
                    neighbors = self.multi_hop_retrieve(node_name, hops=expand_hops)
                    context_parts.append(neighbors)

            return "\n\n".join(context_parts)
        except Exception as e:
            logger.error(f"Vector-graph retrieval failed: {e}")
            return f"Vector-graph retrieval failed: {e}"

    # ------------------------------------------------------------------
    # LangChain GraphCypherQAChain
    # ------------------------------------------------------------------

    def get_cypher_qa_chain(self):
        """Return a LangChain GraphCypherQAChain for direct Q&A over the graph.

        Requires `langchain-community` with Neo4j support.
        """
        from langchain_community.graphs import Neo4jGraph
        from langchain.chains import GraphCypherQAChain

        graph = Neo4jGraph(
            url=self.client.uri,
            username=self.client.username,
            password=self.client.password,
            database=self.client.database,
        )

        return GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=graph,
            verbose=True,
            return_intermediate_steps=True,
        )

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_results(results: List[Dict[str, Any]]) -> str:
        """Format Neo4j query results into a readable context string."""
        lines = []
        for record in results:
            parts = [f"{k}: {v}" for k, v in record.items() if v is not None]
            lines.append(" | ".join(parts))
        return "\n".join(lines)
