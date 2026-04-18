"""
Hybrid Retriever — Combines Vector Store and Neo4j Graph queries.
"""

from typing import Any
from langchain_core.documents import Document

from src.utils.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    """Retrieves and merges contexts from both Vector DB and Knowledge Graph."""

    def __init__(self, vector_store: Any, llm: Any, reranker: Any = None):
        self.vector_store = vector_store
        self.llm = llm
        self.reranker = reranker
        self._graph = None

    def _get_graph(self):
        from src.config.settings import settings
        if self._graph is None:
            try:
                from langchain_neo4j import Neo4jGraph
                self._graph = Neo4jGraph(
                    url=settings.neo4j_uri,
                    username=settings.neo4j_username,
                    password=settings.neo4j_password,
                )
            except Exception as e:
                logger.warning(f"Neo4j unavailable for retrieval: {e}")
        return self._graph

    def async_retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Perform unified retrieval sequentially (could be async in prod)."""
        logger.info(f"Hybrid retrieval for query: {query}")
        
        # 1. Vector Search
        vector_docs = []
        try:
            vector_docs = self.vector_store.similarity_search(query, k=top_k)
            logger.debug(f"Vector search returned {len(vector_docs)} docs")
        except Exception as e:
            logger.error(f"Vector search failed: {e}")

        # 2. Graph Search (GraphQA chain / Extract)
        graph_docs = []
        graph = self._get_graph()
        if graph:
            try:
                # Basic unstructured graph querying using GraphQA
                from langchain_neo4j import GraphCypherQAChain
                chain = GraphCypherQAChain.from_llm(
                    self.llm,
                    graph=graph,
                    verbose=True,
                    return_direct=True, # We just want the context, not final answer
                )
                graph_res = chain.invoke({"query": query})
                raw_graph_res = graph_res.get("result", "")
                if raw_graph_res:
                    graph_docs.append(Document(
                        page_content=str(raw_graph_res),
                        metadata={"source": "Neo4j_Cypher_QA"}
                    ))
            except Exception as e:
                logger.error(f"Graph retrieval failed: {e}")

        # 3. Reciprocal Rank Fusion / De-duplication
        all_docs = vector_docs + graph_docs
        
        seen = set()
        unique_docs = []
        for d in all_docs:
            if d.page_content not in seen:
                seen.add(d.page_content)
                unique_docs.append(d)

        # Apply reranker if configured
        if self.reranker and unique_docs:
            return self.reranker.rerank(query, unique_docs, top_k=top_k)

        return unique_docs[:top_k]
