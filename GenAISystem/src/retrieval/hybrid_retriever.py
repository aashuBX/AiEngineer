class HybridRetriever:
    def __init__(self, vector_retriever, keyword_retriever, fusion_algorithm):
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.fusion_algorithm = fusion_algorithm

    def retrieve(self, query: str, top_k: int = 5):
        v_results = self.vector_retriever.retrieve(query, top_k*2)
        k_results = self.keyword_retriever.retrieve(query, top_k*2)
        return self.fusion_algorithm.fuse(v_results, k_results)[:top_k]
