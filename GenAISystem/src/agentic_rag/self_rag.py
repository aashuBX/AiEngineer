"""
Self-RAG — Self-reflective RAG with document grading and hallucination verification.
Implements: Retrieve → Grade → Generate → Self-Verify loop.
"""

import logging
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# State
# ------------------------------------------------------------------

class SelfRAGState(TypedDict):
    """State for the Self-RAG workflow."""
    query: str
    documents: List[Dict[str, Any]]
    relevant_documents: List[Dict[str, Any]]
    answer: str
    is_hallucinated: bool
    is_relevant: bool
    generation_count: int
    max_generations: int


# ------------------------------------------------------------------
# Self-RAG Workflow
# ------------------------------------------------------------------

class SelfRAG:
    """Self-RAG: Retrieve, grade relevance, generate, then self-verify.

    Flow:
    1. Retrieve candidate documents
    2. Grade each document for relevance to the query
    3. Generate answer from relevant documents
    4. Self-verify: check for hallucinations (LLM-as-judge)
    5. If hallucinated → re-retrieve with refined query or regenerate
    """

    def __init__(
        self,
        retriever,
        llm,
        max_generations: int = 3,
    ):
        """
        Args:
            retriever: Any retriever with a `.retrieve(query, top_k)` method.
            llm: LangChain ChatModel.
            max_generations: Max retry loops before giving up.
        """
        self.retriever = retriever
        self.llm = llm
        self.max_generations = max_generations

    def build_graph(self) -> StateGraph:
        """Build the Self-RAG LangGraph workflow."""
        workflow = StateGraph(SelfRAGState)

        # Add nodes
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("grade_documents", self._grade_documents)
        workflow.add_node("generate", self._generate)
        workflow.add_node("verify", self._verify)

        # Set entry point
        workflow.set_entry_point("retrieve")

        # Add edges
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_edge("grade_documents", "generate")
        workflow.add_edge("generate", "verify")

        # Conditional: if hallucinated and under max retries → re-retrieve
        workflow.add_conditional_edges(
            "verify",
            self._should_retry,
            {
                "retry": "retrieve",
                "accept": END,
            },
        )

        return workflow

    def compile(self):
        """Compile and return the runnable graph."""
        graph = self.build_graph()
        return graph.compile()

    def invoke(self, query: str) -> Dict[str, Any]:
        """Run the Self-RAG pipeline on a query."""
        app = self.compile()
        initial_state: SelfRAGState = {
            "query": query,
            "documents": [],
            "relevant_documents": [],
            "answer": "",
            "is_hallucinated": False,
            "is_relevant": True,
            "generation_count": 0,
            "max_generations": self.max_generations,
        }
        result = app.invoke(initial_state)
        return {
            "answer": result["answer"],
            "sources": result["relevant_documents"],
            "generations_used": result["generation_count"],
            "hallucination_detected": result["is_hallucinated"],
        }

    # ------------------------------------------------------------------
    # Graph Nodes
    # ------------------------------------------------------------------

    def _retrieve(self, state: SelfRAGState) -> Dict:
        """Retrieve candidate documents."""
        query = state["query"]
        documents = self.retriever.retrieve(query, top_k=10)
        logger.info(f"Self-RAG: Retrieved {len(documents)} documents")
        return {"documents": documents}

    def _grade_documents(self, state: SelfRAGState) -> Dict:
        """Grade each document for relevance to the query using LLM."""
        query = state["query"]
        documents = state["documents"]
        relevant = []

        for doc in documents:
            content = doc.get("content", "")
            grade_prompt = f"""Is the following document relevant to the query?
Query: {query}
Document: {content[:500]}

Answer with just 'yes' or 'no'."""

            try:
                response = self.llm.invoke(grade_prompt)
                answer = response.content.strip().lower() if hasattr(response, "content") else str(response).lower()
                if "yes" in answer:
                    relevant.append(doc)
            except Exception as e:
                logger.warning(f"Grading failed: {e}. Including document by default.")
                relevant.append(doc)

        logger.info(f"Self-RAG: {len(relevant)}/{len(documents)} documents graded as relevant")
        return {"relevant_documents": relevant}

    def _generate(self, state: SelfRAGState) -> Dict:
        """Generate answer from relevant documents."""
        query = state["query"]
        docs = state["relevant_documents"]
        count = state["generation_count"] + 1

        if not docs:
            return {
                "answer": "I don't have enough relevant information to answer this question.",
                "generation_count": count,
            }

        context = "\n\n".join(doc.get("content", "") for doc in docs)
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Provide a thorough, accurate answer based only on the context above."""

        from langchain_core.messages import HumanMessage
        response = self.llm.invoke([HumanMessage(content=prompt)])
        answer = response.content if hasattr(response, "content") else str(response)

        logger.info(f"Self-RAG: Generated answer (attempt {count})")
        return {"answer": answer, "generation_count": count}

    def _verify(self, state: SelfRAGState) -> Dict:
        """Verify that the answer is grounded in the context (hallucination check)."""
        answer = state["answer"]
        docs = state["relevant_documents"]
        context = "\n".join(doc.get("content", "") for doc in docs)

        verify_prompt = f"""You are a factual accuracy judge.

Context:
{context[:2000]}

Answer:
{answer}

Is the answer fully supported by the context? Answer 'yes' if it is grounded, 'no' if it contains unsupported claims."""

        try:
            response = self.llm.invoke(verify_prompt)
            verdict = response.content.strip().lower() if hasattr(response, "content") else "yes"
            is_hallucinated = "no" in verdict
        except Exception as e:
            logger.warning(f"Verification failed: {e}. Assuming not hallucinated.")
            is_hallucinated = False

        logger.info(f"Self-RAG: Hallucination check = {'FAIL' if is_hallucinated else 'PASS'}")
        return {"is_hallucinated": is_hallucinated}

    # ------------------------------------------------------------------
    # Conditional Edge
    # ------------------------------------------------------------------

    @staticmethod
    def _should_retry(state: SelfRAGState) -> str:
        """Decide whether to retry generation or accept the answer."""
        if state["is_hallucinated"] and state["generation_count"] < state["max_generations"]:
            return "retry"
        return "accept"
