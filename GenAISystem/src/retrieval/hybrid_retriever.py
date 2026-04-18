"""
Hybrid Retriever — Combines Vector + Keyword + Graph retrieval with configurable weights.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combine multiple retrieval strategies with weighted fusion.

    Supports:
    - Vector (semantic) retrieval
    - Keyword (BM25) retrieval
    - Graph (knowledge graph) retrieval
    - Parallel execution of all retrievers
    - Configurable weights per retrieval method
    """

    def __init__(
        self,
        vector_retriever=None,
        keyword_retriever=None,
        graph_retriever=None,
        fusion_algorithm=None,
        weights: Optional[Dict[str, float]] = None,
        parallel: bool = True,
    ):
        """
        Args:
            vector_retriever: VectorRetriever instance.
            keyword_retriever: KeywordRetriever instance.
            graph_retriever: GraphRetriever instance (from knowledge_graph or retrieval).
            fusion_algorithm: Fusion strategy (e.g., RRF). If None, simple interleave.
            weights: Dict of weights per method, e.g. {"vector": 0.5, "keyword": 0.3, "graph": 0.2}.
            parallel: If True, execute retrievers in parallel using threads.
        """
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.graph_retriever = graph_retriever
        self.fusion_algorithm = fusion_algorithm
        self.weights = weights or {"vector": 0.5, "keyword": 0.3, "graph": 0.2}
        self.parallel = parallel
        self._executor = ThreadPoolExecutor(max_workers=3) if parallel else None

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        methods: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve from all configured sources and fuse results.

        Args:
            query: The search query.
            top_k: Number of final results after fusion.
            methods: Which methods to use. Default: all configured ones.
                     Options: ['vector', 'keyword', 'graph'].

        Returns:
            Fused and ranked list of results.
        """
        methods = methods or self._active_methods()
        fetch_k = top_k * 3  # Over-fetch for fusion

        if self.parallel and len(methods) > 1:
            all_results = self._parallel_retrieve(query, fetch_k, methods)
        else:
            all_results = self._sequential_retrieve(query, fetch_k, methods)

        # Fuse results
        if self.fusion_algorithm and len(all_results) > 1:
            fused = self.fusion_algorithm.fuse(
                *all_results.values(),
                weights={k: self.weights.get(k, 1.0) for k in all_results.keys()},
            )
        else:
            fused = self._simple_merge(all_results)

        return fused[:top_k]

    def _active_methods(self) -> List[str]:
        """Determine which retrieval methods are available."""
        methods = []
        if self.vector_retriever:
            methods.append("vector")
        if self.keyword_retriever:
            methods.append("keyword")
        if self.graph_retriever:
            methods.append("graph")
        return methods

    def _sequential_retrieve(
        self, query: str, fetch_k: int, methods: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Execute retrievers sequentially."""
        results = {}
        for method in methods:
            results[method] = self._run_retriever(method, query, fetch_k)
        return results

    def _parallel_retrieve(
        self, query: str, fetch_k: int, methods: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Execute retrievers in parallel using threads."""
        futures = {
            method: self._executor.submit(self._run_retriever, method, query, fetch_k)
            for method in methods
        }
        results = {}
        for method, future in futures.items():
            try:
                results[method] = future.result(timeout=30)
            except Exception as e:
                logger.error(f"{method} retriever failed: {e}")
                results[method] = []
        return results

    def _run_retriever(self, method: str, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Run a single retriever by method name."""
        try:
            if method == "vector" and self.vector_retriever:
                return self.vector_retriever.retrieve(query, top_k=top_k)
            elif method == "keyword" and self.keyword_retriever:
                return self.keyword_retriever.retrieve(query, top_k=top_k)
            elif method == "graph" and self.graph_retriever:
                # Graph retriever may return a string; wrap it
                result = self.graph_retriever.retrieve(query)
                if isinstance(result, str):
                    return [{"id": "graph_0", "content": result, "metadata": {"source": "knowledge_graph"}, "score": 1.0}]
                return result
        except Exception as e:
            logger.error(f"Error in {method} retriever: {e}")
        return []

    @staticmethod
    def _simple_merge(all_results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Simple round-robin merge without sophisticated fusion."""
        seen_ids = set()
        merged = []

        # Interleave results from all sources
        max_len = max((len(v) for v in all_results.values()), default=0)
        for i in range(max_len):
            for method, results in all_results.items():
                if i < len(results):
                    result = results[i]
                    rid = result.get("id", f"{method}_{i}")
                    if rid not in seen_ids:
                        seen_ids.add(rid)
                        result["retrieval_method"] = method
                        merged.append(result)

        return merged
