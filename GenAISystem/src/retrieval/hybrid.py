"""
Hybrid Retriever — Combines Vector Store and Neo4j Graph queries.
"""

from typing import Any
from langchain_core.documents import Document

from src.utils.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    """Retrieves and merges contexts from both Vector DB and Knowledge Graph."""

    def __init__(self, vector_store: Any, llm: Any, llm_reranker: Any = None, cross_encoder: Any = None):
        self.vector_store = vector_store
        self.llm = llm
        self.llm_reranker = llm_reranker
        self.cross_encoder = cross_encoder
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

    def async_retrieve(self, query: str, top_k: int = 5, rerank_strategy: str = "auto") -> list[Document]:
        """Perform unified retrieval sequentially (could be async in prod)."""
        logger.info(f"Hybrid retrieval for query: {query} (Strategy: {rerank_strategy})")
        
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

        # Determine Routing Strategy
        if rerank_strategy == "auto":
            query_lower = query.lower()
            complex_keywords = ["compare", "difference", "analyze", "why", "summarize", "pros and cons", "evaluate", "reason"]
            if any(k in query_lower for k in complex_keywords) or len(query.split()) > 15:
                rerank_strategy = "llm_reranker"
            else:
                rerank_strategy = "cross_encoder"
                
        # Apply reranker based on strategy
        if rerank_strategy == "llm_reranker" and self.llm_reranker and unique_docs:
            logger.info("Routing to LLM Reranker")
            return self.llm_reranker.rerank(query, unique_docs, top_k=top_k)
        elif rerank_strategy == "cross_encoder" and self.cross_encoder and unique_docs:
            logger.info("Routing to Cross-Encoder")
            return self.cross_encoder.rerank(query, unique_docs, top_k=top_k)

        return unique_docs[:top_k]
