"""
Hybrid Retriever — Combines Vector Store and Neo4j Graph queries.
"""

from typing import Any
from langchain_core.documents import Document

from src.utils.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    """Retrieves and merges contexts from Vector DB, Knowledge Graph, and BM25."""

    def __init__(
        self,
        vector_store: Any,
        llm: Any,
        llm_reranker: Any = None,
        cross_encoder: Any = None,
        keyword_retriever: Any = None,
    ):
        self.vector_store = vector_store
        self.llm = llm
        self.llm_reranker = llm_reranker
        self.cross_encoder = cross_encoder
        self.keyword_retriever = keyword_retriever
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

    def async_retrieve(
        self,
        query: str,
        strategy: str = "hybrid",
        top_k: int = 5,
        rerank_strategy: str = "auto"
    ) -> list[Document]:
        """Perform unified retrieval using specified strategy."""
        logger.info(f"Retrieval strategy='{strategy}', rerank='{rerank_strategy}' for query: {query}")
        
        all_docs = []

        # ── 1. Vector & BM25 Search (Reciprocal Rank Fusion) ──
        if strategy in ("vector", "hybrid"):
            # Dense Vector Search
            dense_docs = []
            try:
                # Overfetch for fusion and reranking
                dense_docs = self.vector_store.similarity_search(query, k=top_k * 3)
                logger.debug(f"Dense search returned {len(dense_docs)} docs")
            except Exception as e:
                logger.error(f"Dense search failed: {e}")

            # Sparse BM25 Search
            sparse_docs = []
            if self.keyword_retriever:
                try:
                    kw_results = self.keyword_retriever.retrieve(query, top_k=top_k * 3)
                    sparse_docs = [
                        Document(page_content=r["content"], metadata={**r["metadata"], "source": "BM25"})
                        for r in kw_results
                    ]
                    logger.debug(f"BM25 search returned {len(sparse_docs)} docs")
                except Exception as e:
                    logger.error(f"BM25 search failed: {e}")

            # Fuse Dense and Sparse
            from src.retrieval.fusion import ReciprocalRankFusion
            fusion = ReciprocalRankFusion()
            fused_vector_docs = fusion.fuse(dense_docs, sparse_docs)
            all_docs.extend(fused_vector_docs)

        # ── 2. Graph Search ──
        if strategy in ("graph", "hybrid"):
            graph = self._get_graph()
            if graph:
                try:
                    from langchain_neo4j import GraphCypherQAChain
                    chain = GraphCypherQAChain.from_llm(
                        self.llm,
                        graph=graph,
                        verbose=False,
                        return_direct=True, # We just want the context
                    )
                    graph_res = chain.invoke({"query": query})
                    raw_graph_res = graph_res.get("result", "")
                    if raw_graph_res:
                        all_docs.append(Document(
                            page_content=str(raw_graph_res),
                            metadata={"source": "Neo4j_Cypher_QA"}
                        ))
                        logger.debug("Graph search returned 1 result")
                except Exception as e:
                    logger.error(f"Graph retrieval failed: {e}")

        # Deduplicate
        seen = set()
        unique_docs = []
        for d in all_docs:
            if d.page_content not in seen:
                seen.add(d.page_content)
                unique_docs.append(d)

        if not unique_docs:
            return []

        # ── 3. Routing & Reranking ──
        if rerank_strategy == "auto":
            query_lower = query.lower()
            complex_keywords = ["compare", "difference", "analyze", "why", "summarize", "evaluate", "reason"]
            word_count = len(query.split())
            
            if any(k in query_lower for k in complex_keywords) or word_count > 15:
                rerank_strategy = "llm_reranker"
            else:
                rerank_strategy = "cross_encoder"
                
        # Apply reranker based on strategy
        if rerank_strategy == "llm_reranker" and self.llm_reranker:
            logger.info("Routing to LLM Reranker")
            unique_docs = self.llm_reranker.rerank(query, unique_docs, top_k=top_k)
        elif rerank_strategy == "cross_encoder" and self.cross_encoder:
            logger.info("Routing to Cross-Encoder")
            unique_docs = self.cross_encoder.rerank(query, unique_docs, top_k=top_k)
        else:
            unique_docs = unique_docs[:top_k]

        return unique_docs
