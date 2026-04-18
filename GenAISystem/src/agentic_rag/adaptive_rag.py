"""
Adaptive RAG workflow using LangGraph.
Routes between Web Search, Vector RAG, and direct LLM generation based on the query.
"""

from typing import TypedDict, Literal

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AdaptiveRagState(TypedDict):
    """State for the Adaptive RAG graph."""
    query: str
    route: str
    retrieved_docs: list
    generation: str
    messages: list[BaseMessage]


class AdaptiveRagRouter:
    """Implements an adaptive RAG LangGraph."""

    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def _route_query(self, state: AdaptiveRagState) -> AdaptiveRagState:
        """Route the query to the correct strategy."""
        query = state["query"]
        logger.info(f"AdaptiveRAG: routing query '{query}'")
        
        # Simple heuristic router for demonstration
        # In prod, this uses an LLM structured output classification
        query_lower = query.lower()
        if "weather" in query_lower or "news" in query_lower:
            route = "web_search"
        elif "hello" in query_lower or "hi " in query_lower:
            route = "direct_llm"
        else:
            route = "vector_rag"
            
        return {"route": route}

    def _retrieve(self, state: AdaptiveRagState) -> AdaptiveRagState:
        """Retrieve documents using the configured retriever."""
        docs = self.retriever.async_retrieve(state["query"])
        return {"retrieved_docs": docs}

    def _generate(self, state: AdaptiveRagState) -> AdaptiveRagState:
        """Generate response based on retrieved docs."""
        # Assume response generator handles context + llm call
        return {"generation": "Generated answer based on vector docs."}

    def _fallback_web(self, state: AdaptiveRagState) -> AdaptiveRagState:
        """Fallback to web search."""
        return {"generation": "Generated answer based on live web search."}
        
    def _direct_answer(self, state: AdaptiveRagState) -> AdaptiveRagState:
        """Answer directly without context."""
        return {"generation": "Direct LLM answer without retrieval."}

    def determine_edge(self, state: AdaptiveRagState) -> Literal["retrieve", "fallback_web", "direct_answer"]:
        """Determine which node to execute next based on the route."""
        route = state["route"]
        if route == "vector_rag":
            return "retrieve"
        elif route == "web_search":
            return "fallback_web"
        else:
            return "direct_answer"

    def build_graph(self) -> StateGraph:
        """Compile the LangGraph workflow."""
        builder = StateGraph(AdaptiveRagState)

        # Add nodes
        builder.add_node("router", self._route_query)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("generate", self._generate)
        builder.add_node("fallback_web", self._fallback_web)
        builder.add_node("direct_answer", self._direct_answer)

        # Bind edges
        builder.set_entry_point("router")
        
        builder.add_conditional_edges(
            "router",
            self.determine_edge,
            {
                "retrieve": "retrieve",
                "fallback_web": "fallback_web",
                "direct_answer": "direct_answer"
            }
        )
        
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", END)
        builder.add_edge("fallback_web", END)
        builder.add_edge("direct_answer", END)

        return builder.compile()
