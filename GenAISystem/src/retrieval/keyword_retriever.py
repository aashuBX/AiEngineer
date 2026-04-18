class KeywordRetriever:
    def __init__(self, documents):
        self.documents = documents
        # Build BM25 index here

    def retrieve(self, query: str, top_k: int = 5):
        # Execute BM25 ranking
        return []
