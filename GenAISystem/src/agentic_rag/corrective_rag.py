"""
Corrective RAG (CRAG) — Detect retrieval failures and fall back to web search.
Implements: Retrieve → Grade Quality → (Web Fallback if needed) → Generate.
"""

import logging
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# State
# ------------------------------------------------------------------

class CRAGState(TypedDict):
    """State for the Corrective RAG workflow."""
    query: str
    documents: List[Dict[str, Any]]
    web_results: List[Dict[str, Any]]
    retrieval_quality: str  # "HIGH", "LOW", "AMBIGUOUS"
    final_context: List[Dict[str, Any]]
    answer: str


# ------------------------------------------------------------------
# CRAG Workflow
# ------------------------------------------------------------------

class CorrectiveRAG:
    """Corrective RAG (CRAG): Grade retrieval quality and fall back to web search.

    Flow:
    1. Retrieve from local knowledge base
    2. Grade retrieval quality (HIGH / AMBIGUOUS / LOW)
    3. If HIGH → use retrieved context directly
    4. If AMBIGUOUS → supplement with web search results
    5. If LOW → replace with web search results
    6. Generate answer from best available context
    """

    def __init__(
        self,
        retriever,
        llm,
        web_search_tool=None,
    ):
        """
        Args:
            retriever: Local knowledge base retriever.
            llm: LangChain ChatModel.
            web_search_tool: Optional async/sync web search callable.
                If None, web fallback is disabled.
        """
        self.retriever = retriever
        self.llm = llm
        self.web_search_tool = web_search_tool

    def build_graph(self) -> StateGraph:
        """Build the CRAG LangGraph workflow."""
        workflow = StateGraph(CRAGState)

        # Nodes
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("grade_quality", self._grade_quality)
        workflow.add_node("web_search", self._web_search)
        workflow.add_node("prepare_context", self._prepare_context)
        workflow.add_node("generate", self._generate)

        # Entry
        workflow.set_entry_point("retrieve")

        # Edges
        workflow.add_edge("retrieve", "grade_quality")
        workflow.add_conditional_edges(
            "grade_quality",
            self._route_by_quality,
            {
                "use_local": "prepare_context",
                "supplement": "web_search",
                "web_only": "web_search",
            },
        )
        workflow.add_edge("web_search", "prepare_context")
        workflow.add_edge("prepare_context", "generate")
        workflow.add_edge("generate", END)

        return workflow

    def compile(self):
        """Compile the graph."""
        return self.build_graph().compile()

    def invoke(self, query: str) -> Dict[str, Any]:
        """Run CRAG on a query."""
        app = self.compile()
        initial_state: CRAGState = {
            "query": query,
            "documents": [],
            "web_results": [],
            "retrieval_quality": "HIGH",
            "final_context": [],
            "answer": "",
        }
        result = app.invoke(initial_state)
        return {
            "answer": result["answer"],
            "retrieval_quality": result["retrieval_quality"],
            "sources": result["final_context"],
            "used_web_search": len(result.get("web_results", [])) > 0,
        }

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def _retrieve(self, state: CRAGState) -> Dict:
        """Retrieve from local knowledge base."""
        docs = self.retriever.retrieve(state["query"], top_k=5)
        logger.info(f"CRAG: Retrieved {len(docs)} local documents")
        return {"documents": docs}

    def _grade_quality(self, state: CRAGState) -> Dict:
        """Grade the overall quality of retrieved documents."""
        query = state["query"]
        docs = state["documents"]

        if not docs:
            return {"retrieval_quality": "LOW"}

        # Use LLM to assess retrieval quality
        doc_summaries = "\n".join(
            f"- {doc.get('content', '')[:200]}" for doc in docs[:5]
        )

        prompt = f"""Evaluate whether the following retrieved documents adequately answer the query.

Query: {query}

Retrieved Documents:
{doc_summaries}

Rate the retrieval quality:
- HIGH: Documents directly and thoroughly address the query
- AMBIGUOUS: Documents are partially relevant but may be insufficient
- LOW: Documents are not relevant to the query

Answer with just: HIGH, AMBIGUOUS, or LOW"""

        try:
            response = self.llm.invoke(prompt)
            quality = response.content.strip().upper() if hasattr(response, "content") else "AMBIGUOUS"
            # Extract just the quality label
            for label in ["HIGH", "LOW", "AMBIGUOUS"]:
                if label in quality:
                    quality = label
                    break
            else:
                quality = "AMBIGUOUS"
        except Exception as e:
            logger.warning(f"Quality grading failed: {e}")
            quality = "AMBIGUOUS"

        logger.info(f"CRAG: Retrieval quality = {quality}")
        return {"retrieval_quality": quality}

    def _web_search(self, state: CRAGState) -> Dict:
        """Fall back to web search for additional or replacement context."""
        if not self.web_search_tool:
            logger.warning("CRAG: No web search tool configured. Skipping web fallback.")
            return {"web_results": []}

        query = state["query"]
        try:
            # Support both sync and async web search
            import asyncio
            if asyncio.iscoroutinefunction(self.web_search_tool):
                results = asyncio.get_event_loop().run_until_complete(
                    self.web_search_tool(query)
                )
            else:
                results = self.web_search_tool(query)

            # Normalize results
            if isinstance(results, str):
                web_docs = [{"content": results, "metadata": {"source": "web_search"}}]
            elif isinstance(results, list):
                web_docs = results
            else:
                web_docs = [{"content": str(results), "metadata": {"source": "web_search"}}]

            logger.info(f"CRAG: Web search returned {len(web_docs)} results")
            return {"web_results": web_docs}
        except Exception as e:
            logger.error(f"CRAG: Web search failed: {e}")
            return {"web_results": []}

    def _prepare_context(self, state: CRAGState) -> Dict:
        """Prepare the final context based on retrieval quality."""
        quality = state["retrieval_quality"]
        local_docs = state["documents"]
        web_docs = state.get("web_results", [])

        if quality == "HIGH":
            final = local_docs
        elif quality == "AMBIGUOUS":
            # Combine local + web
            final = local_docs + web_docs
        else:  # LOW
            # Prefer web results, include local as backup
            final = web_docs + local_docs if web_docs else local_docs

        return {"final_context": final}

    def _generate(self, state: CRAGState) -> Dict:
        """Generate the final answer from the best available context."""
        query = state["query"]
        context = state["final_context"]

        if not context:
            return {"answer": "I couldn't find relevant information to answer your question."}

        context_text = "\n\n".join(
            doc.get("content", "") for doc in context
        )

        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content="You are a helpful assistant. Answer based on the context provided."),
            HumanMessage(content=f"Context:\n{context_text}\n\nQuestion: {query}"),
        ]

        response = self.llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)

        return {"answer": answer}

    # ------------------------------------------------------------------
    # Conditional
    # ------------------------------------------------------------------

    @staticmethod
    def _route_by_quality(state: CRAGState) -> str:
        """Route based on retrieval quality assessment."""
        quality = state["retrieval_quality"]
        if quality == "HIGH":
            return "use_local"
        elif quality == "AMBIGUOUS":
            return "supplement"
        else:
            return "web_only"
