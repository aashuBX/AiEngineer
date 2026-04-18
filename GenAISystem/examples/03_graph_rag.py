"""Example 03: Graph RAG — Entity extraction + Neo4j knowledge graph."""
from src.knowledge_graph.neo4j_client import Neo4jClient
from src.knowledge_graph.entity_extractor import EntityExtractor
from src.knowledge_graph.graph_visualizer import GraphVisualizer


def main():
    print("Graph RAG Demo")
    print("=" * 50)
    print("Prerequisites: Neo4j running on bolt://localhost:7687")
    print()

    # Initialize client
    client = Neo4jClient(uri="bolt://localhost:7687", username="neo4j", password="password")

    try:
        client.connect()
        healthy = client.health_check()
        print(f"Neo4j connection: {'✓ Healthy' if healthy else '✗ Failed'}")

        if healthy:
            # Visualize existing graph
            viz = GraphVisualizer(client)
            output = viz.render_subgraph(output_path="knowledge_graph.html")
            print(f"Graph visualization: {output}")
    except Exception as e:
        print(f"Could not connect to Neo4j: {e}")
        print("Please start Neo4j and update connection settings.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
