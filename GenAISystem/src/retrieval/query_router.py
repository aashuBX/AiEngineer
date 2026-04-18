class QueryRouter:
    def __init__(self, llm):
        self.llm = llm

    def route_query(self, query: str) -> str:
        # Route logic: "vector", "keyword", "graph", or "hybrid"
        return "hybrid"
