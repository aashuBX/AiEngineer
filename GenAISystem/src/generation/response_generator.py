class ResponseGenerator:
    def __init__(self, llm):
        self.llm = llm

    def generate(self, query: str, context: list):
        prompt = f"Answer this query based on context:\nContext: {context}\nQuery: {query}"
        return self.llm.invoke(prompt)
