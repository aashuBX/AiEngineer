"""
Graph Visualizer — Render knowledge graph subgraphs as interactive HTML.
Uses pyvis (if available) with networkx fallback.
"""

import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class GraphVisualizer:
    """Visualize knowledge graph data as interactive network diagrams."""

    # Color palette for different entity types
    ENTITY_COLORS = {
        "Person": "#4CAF50",
        "Organization": "#2196F3",
        "Technology": "#FF9800",
        "Concept": "#9C27B0",
        "Product": "#F44336",
        "Location": "#00BCD4",
        "Event": "#FFEB3B",
        "Document": "#795548",
    }
    DEFAULT_COLOR = "#607D8B"

    def __init__(self, neo4j_client=None):
        self.client = neo4j_client

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def fetch_subgraph(
        self,
        center_entity: Optional[str] = None,
        hops: int = 2,
        limit: int = 50,
    ) -> Dict[str, List]:
        """Fetch nodes and edges from Neo4j for visualization.

        Args:
            center_entity: Optional entity name to center the view on.
            hops: Number of hops to expand from center.
            limit: Max nodes to return.

        Returns:
            Dict with 'nodes' and 'edges' lists.
        """
        if not self.client:
            return {"nodes": [], "edges": []}

        if center_entity:
            query = f"""
            MATCH path = (start)-[*1..{hops}]-(end)
            WHERE toLower(start.name) CONTAINS toLower($entity)
            WITH nodes(path) AS ns, relationships(path) AS rs
            UNWIND ns AS n
            WITH COLLECT(DISTINCT n) AS nodes,
                 COLLECT(DISTINCT rs) AS all_rels
            UNWIND all_rels AS rel_list
            UNWIND rel_list AS r
            RETURN nodes, COLLECT(DISTINCT r) AS edges
            LIMIT 1
            """
            params = {"entity": center_entity}
        else:
            query = f"""
            MATCH (n)-[r]->(m)
            RETURN n, r, m
            LIMIT {limit}
            """
            params = {}

        try:
            results = self.client.execute_query(query, params)
            return self._parse_graph_results(results)
        except Exception as e:
            logger.error(f"Failed to fetch subgraph: {e}")
            return {"nodes": [], "edges": []}

    # ------------------------------------------------------------------
    # Visualization with pyvis
    # ------------------------------------------------------------------

    def render_subgraph(
        self,
        entities: Optional[List[Dict[str, Any]]] = None,
        relationships: Optional[List[Dict[str, Any]]] = None,
        output_path: str = "knowledge_graph.html",
        center_entity: Optional[str] = None,
        height: str = "700px",
        width: str = "100%",
    ) -> str:
        """Render a knowledge graph as an interactive HTML file.

        Can accept pre-extracted data or fetch from Neo4j.

        Args:
            entities: List of dicts with 'name' and 'entity_type'.
            relationships: List of dicts with 'source', 'target', 'relationship_type'.
            output_path: Path for the output HTML file.
            center_entity: If entities/relationships not provided, fetch from Neo4j.
            height: Chart height.
            width: Chart width.

        Returns:
            Path to the generated HTML file.
        """
        try:
            from pyvis.network import Network
        except ImportError:
            logger.warning("pyvis not installed. Falling back to networkx matplotlib export.")
            return self._render_with_networkx(entities, relationships, output_path)

        net = Network(
            height=height,
            width=width,
            bgcolor="#1a1a2e",
            font_color="white",
            directed=True,
        )
        net.barnes_hut(
            gravity=-8000,
            central_gravity=0.3,
            spring_length=200,
            spring_strength=0.01,
        )

        # If no data provided, fetch from Neo4j
        if entities is None or relationships is None:
            data = self.fetch_subgraph(center_entity=center_entity)
            entities = entities or data.get("nodes", [])
            relationships = relationships or data.get("edges", [])

        # Add nodes
        for entity in entities:
            name = entity.get("name", "Unknown")
            etype = entity.get("entity_type", "Unknown")
            color = self.ENTITY_COLORS.get(etype, self.DEFAULT_COLOR)
            net.add_node(
                name,
                label=name,
                title=f"{etype}: {name}",
                color=color,
                size=25,
                shape="dot",
            )

        # Add edges
        for rel in relationships:
            source = rel.get("source", "")
            target = rel.get("target", "")
            rtype = rel.get("relationship_type", "RELATED_TO")
            if source and target:
                net.add_edge(source, target, title=rtype, label=rtype, arrows="to")

        # Save
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        net.save_graph(str(output))
        logger.info(f"Graph visualization saved to {output}")
        return str(output)

    # ------------------------------------------------------------------
    # Networkx fallback
    # ------------------------------------------------------------------

    def _render_with_networkx(
        self,
        entities: Optional[List[Dict[str, Any]]],
        relationships: Optional[List[Dict[str, Any]]],
        output_path: str,
    ) -> str:
        """Fallback renderer using networkx + matplotlib."""
        try:
            import networkx as nx
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("Neither pyvis nor networkx/matplotlib available.")
            return ""

        G = nx.DiGraph()
        entities = entities or []
        relationships = relationships or []

        for entity in entities:
            name = entity.get("name", "Unknown")
            G.add_node(name, entity_type=entity.get("entity_type", "Unknown"))

        for rel in relationships:
            source = rel.get("source", "")
            target = rel.get("target", "")
            rtype = rel.get("relationship_type", "RELATED_TO")
            if source and target:
                G.add_edge(source, target, label=rtype)

        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(G, k=2, iterations=50)

        # Color nodes by type
        colors = [
            self.ENTITY_COLORS.get(G.nodes[n].get("entity_type", ""), self.DEFAULT_COLOR)
            for n in G.nodes
        ]

        nx.draw(
            G, pos,
            with_labels=True,
            node_color=colors,
            node_size=800,
            font_size=8,
            font_weight="bold",
            edge_color="#888888",
            arrows=True,
        )

        # Edge labels
        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

        png_path = output_path.replace(".html", ".png")
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Graph visualization saved to {png_path}")
        return png_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_graph_results(results: List[Dict]) -> Dict[str, List]:
        """Parse raw Neo4j results into nodes and edges."""
        nodes = []
        edges = []
        seen_nodes = set()

        for record in results:
            for key, value in record.items():
                if isinstance(value, dict) and "name" in value:
                    name = value["name"]
                    if name not in seen_nodes:
                        seen_nodes.add(name)
                        nodes.append({
                            "name": name,
                            "entity_type": value.get("type", "Unknown"),
                        })

        return {"nodes": nodes, "edges": edges}
