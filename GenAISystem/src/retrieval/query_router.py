"""
Query Router — LLM-based query classification to select the best retrieval strategy.
"""

import logging
from typing import Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RouteDecision(BaseModel):
    """Structured output for query routing."""
    strategy: str = Field(
        ...,
        description="Retrieval strategy: 'vector', 'keyword', 'graph', or 'hybrid'"
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of why this strategy was chosen"
    )


class QueryRouter:
    """Route queries to the optimal retrieval strategy using LLM classification.

    Strategies:
    - **vector**: Semantic/conceptual queries → vector similarity search
    - **keyword**: Exact/factual queries → BM25 keyword search
    - **graph**: Relationship queries → knowledge graph traversal
    - **hybrid**: Complex/multi-faceted queries → all methods combined
    """

    ROUTING_PROMPT = """Classify the following search query and determine the best retrieval strategy.

Strategies:
- "vector": For conceptual, semantic, or natural language questions (e.g., "How does machine learning work?")
- "keyword": For exact term lookups, technical terms, or specific facts (e.g., "Python 3.12 release date")
- "graph": For relationship questions between entities (e.g., "What companies does Elon Musk own?")
- "hybrid": For complex multi-part questions that need multiple sources (e.g., "Compare the architectures of GPT-4 and Claude and their parent companies")

Query: "{query}"

Classify this query and explain your reasoning briefly.
"""

    def __init__(
        self,
        llm,
        default_strategy: str = "hybrid",
        use_llm: bool = True,
    ):
        """
        Args:
            llm: LangChain LLM for query classification.
            default_strategy: Fallback strategy if classification fails.
            use_llm: If False, use rule-based routing instead of LLM.
        """
        self.llm = llm
        self.default_strategy = default_strategy
        self.use_llm = use_llm

    def route_query(self, query: str) -> str:
        """Determine the best retrieval strategy for a query.

        Args:
            query: The user's search query.

        Returns:
            Strategy name: 'vector', 'keyword', 'graph', or 'hybrid'.
        """
        if not self.use_llm:
            return self._rule_based_route(query)

        try:
            structured_llm = self.llm.with_structured_output(RouteDecision)
            prompt = self.ROUTING_PROMPT.format(query=query)
            decision: RouteDecision = structured_llm.invoke(prompt)
            strategy = decision.strategy.lower().strip()

            valid_strategies = {"vector", "keyword", "graph", "hybrid"}
            if strategy not in valid_strategies:
                logger.warning(f"LLM returned invalid strategy '{strategy}', using default.")
                return self.default_strategy

            logger.info(f"QueryRouter: '{query[:50]}...' → {strategy} ({decision.reasoning})")
            return strategy

        except Exception as e:
            logger.warning(f"LLM routing failed: {e}. Falling back to rule-based routing.")
            return self._rule_based_route(query)

    def _rule_based_route(self, query: str) -> str:
        """Simple rule-based routing as fallback."""
        query_lower = query.lower()

        # Graph indicators: relationship words
        graph_keywords = [
            "who owns", "related to", "connected to", "works for",
            "depends on", "part of", "relationship between",
            "how is .* related", "what companies",
        ]
        for kw in graph_keywords:
            if kw in query_lower:
                return "graph"

        # Keyword indicators: specific terms, codes, exact lookups
        keyword_indicators = [
            "exact", "specific", "error code", "version", "release date",
            "definition of", "what is the",
        ]
        for kw in keyword_indicators:
            if kw in query_lower:
                return "keyword"

        # Short queries → keyword, longer → vector
        word_count = len(query.split())
        if word_count <= 3:
            return "keyword"
        elif word_count >= 15:
            return "hybrid"

        return "vector"

    def route_with_explanation(self, query: str) -> dict:
        """Route and return both the strategy and explanation."""
        if not self.use_llm:
            strategy = self._rule_based_route(query)
            return {"strategy": strategy, "reasoning": "Rule-based classification"}

        try:
            structured_llm = self.llm.with_structured_output(RouteDecision)
            prompt = self.ROUTING_PROMPT.format(query=query)
            decision: RouteDecision = structured_llm.invoke(prompt)
            return {"strategy": decision.strategy, "reasoning": decision.reasoning}
        except Exception as e:
            strategy = self._rule_based_route(query)
            return {"strategy": strategy, "reasoning": f"Fallback (LLM error: {e})"}
