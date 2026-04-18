"""
Agentic RAG Graph — LangGraph-based agentic RAG workflow.
Route Query → Retrieve → Grade Documents → Generate → Check Hallucination → Check Relevance → END.
"""

import logging
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# State
# ------------------------------------------------------------------

class RAGGraphState(TypedDict):
    """State for the agentic RAG workflow graph."""
    query: str
    retrieval_strategy: str
    documents: List[Dict[str, Any]]
    relevant_documents: List[Dict[str, Any]]
    answer: str
    hallucination_check: bool
    relevance_check: bool
    retry_count: int


# ------------------------------------------------------------------
# RAG Graph
# ------------------------------------------------------------------

class AgenticRAGGraph:
    """Full agentic RAG workflow as a LangGraph state machine.

    Flow:
    1. Route query → determine retrieval strategy
    2. Retrieve documents using selected strategy
    3. Grade document relevance
    4. Generate answer from relevant docs
    5. Check for hallucination
    6. Check answer relevance to query
    7. If checks pass → END; else → retry (up to max)
    """

    def __init__(
        self,
        retriever,
        llm,
        query_router=None,
        max_retries: int = 2,
    ):
        """
        Args:
            retriever: A retriever with .retrieve(query, top_k) method.
            llm: LangChain ChatModel.
            query_router: Optional QueryRouter for strategy selection.
            max_retries: Maximum number of retry loops.
        """
        self.retriever = retriever
        self.llm = llm
        self.query_router = query_router
        self.max_retries = max_retries

    def build_graph(self) -> StateGraph:
        """Build the complete agentic RAG graph."""
        workflow = StateGraph(RAGGraphState)

        # Add all nodes
        workflow.add_node("route_query", self._route_query)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("grade_documents", self._grade_documents)
        workflow.add_node("generate", self._generate)
        workflow.add_node("check_hallucination", self._check_hallucination)
        workflow.add_node("check_relevance", self._check_relevance)

        # Entry
        workflow.set_entry_point("route_query")

        # Linear edges
        workflow.add_edge("route_query", "retrieve")
        workflow.add_edge("retrieve", "grade_documents")

        # After grading: if relevant docs found → generate; else → END with fallback
        workflow.add_conditional_edges(
            "grade_documents",
            self._has_relevant_docs,
            {
                "has_docs": "generate",
                "no_docs": END,
            },
        )

        workflow.add_edge("generate", "check_hallucination")
        workflow.add_edge("check_hallucination", "check_relevance")

        # After relevance check: accept or retry
        workflow.add_conditional_edges(
            "check_relevance",
            self._should_accept,
            {
                "accept": END,
                "retry": "retrieve",
            },
        )

        return workflow

    def compile(self):
        """Compile the graph."""
        return self.build_graph().compile()

    def invoke(self, query: str) -> Dict[str, Any]:
        """Run the agentic RAG pipeline."""
        app = self.compile()
        initial_state: RAGGraphState = {
            "query": query,
            "retrieval_strategy": "hybrid",
            "documents": [],
            "relevant_documents": [],
            "answer": "",
            "hallucination_check": True,
            "relevance_check": True,
            "retry_count": 0,
        }
        result = app.invoke(initial_state)
        return {
            "answer": result.get("answer", "Unable to generate an answer."),
            "strategy_used": result["retrieval_strategy"],
            "sources": result.get("relevant_documents", []),
            "retries": result["retry_count"],
        }

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def _route_query(self, state: RAGGraphState) -> Dict:
        """Route the query to the best retrieval strategy."""
        if self.query_router:
            strategy = self.query_router.route_query(state["query"])
        else:
            strategy = "hybrid"
        logger.info(f"RAGGraph: Query routed to '{strategy}' strategy")
        return {"retrieval_strategy": strategy}

    def _retrieve(self, state: RAGGraphState) -> Dict:
        """Retrieve documents using the selected strategy."""
        docs = self.retriever.retrieve(state["query"], top_k=8)
        count = state["retry_count"] + (1 if state["retry_count"] > 0 else 0)
        logger.info(f"RAGGraph: Retrieved {len(docs)} documents")
        return {"documents": docs, "retry_count": count}

    def _grade_documents(self, state: RAGGraphState) -> Dict:
        """Grade each document for relevance."""
        query = state["query"]
        relevant = []

        for doc in state["documents"]:
            content = doc.get("content", "")[:300]
            prompt = f"Is this document relevant to '{query}'?\nDocument: {content}\nAnswer yes or no."
            try:
                response = self.llm.invoke(prompt)
                text = response.content.strip().lower() if hasattr(response, "content") else "yes"
                if "yes" in text:
                    relevant.append(doc)
            except Exception:
                relevant.append(doc)  # Include by default on error

        logger.info(f"RAGGraph: {len(relevant)}/{len(state['documents'])} documents relevant")
        return {"relevant_documents": relevant}

    def _generate(self, state: RAGGraphState) -> Dict:
        """Generate answer from relevant documents."""
        context = "\n\n".join(doc.get("content", "") for doc in state["relevant_documents"])

        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content="Answer based only on the provided context. Be accurate and concise."),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['query']}"),
        ]

        response = self.llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)
        return {"answer": answer}

    def _check_hallucination(self, state: RAGGraphState) -> Dict:
        """Check if the answer is grounded in the context."""
        context = " ".join(doc.get("content", "") for doc in state["relevant_documents"])[:2000]
        prompt = f"Is this answer grounded in the context (no fabricated facts)?\nContext: {context}\nAnswer: {state['answer']}\nRespond yes or no."

        try:
            response = self.llm.invoke(prompt)
            text = response.content.strip().lower() if hasattr(response, "content") else "yes"
            is_grounded = "yes" in text
        except Exception:
            is_grounded = True

        logger.info(f"RAGGraph: Hallucination check = {'PASS' if is_grounded else 'FAIL'}")
        return {"hallucination_check": is_grounded}

    def _check_relevance(self, state: RAGGraphState) -> Dict:
        """Check if the answer is relevant to the original query."""
        prompt = f"Does this answer address the question?\nQuestion: {state['query']}\nAnswer: {state['answer']}\nRespond yes or no."

        try:
            response = self.llm.invoke(prompt)
            text = response.content.strip().lower() if hasattr(response, "content") else "yes"
            is_relevant = "yes" in text
        except Exception:
            is_relevant = True

        logger.info(f"RAGGraph: Relevance check = {'PASS' if is_relevant else 'FAIL'}")
        return {"relevance_check": is_relevant}

    # ------------------------------------------------------------------
    # Conditional Edges
    # ------------------------------------------------------------------

    @staticmethod
    def _has_relevant_docs(state: RAGGraphState) -> str:
        if state["relevant_documents"]:
            return "has_docs"
        return "no_docs"

    def _should_accept(self, state: RAGGraphState) -> str:
        if state["hallucination_check"] and state["relevance_check"]:
            return "accept"
        if state["retry_count"] >= self.max_retries:
            return "accept"  # Give up after max retries
        return "retry"
