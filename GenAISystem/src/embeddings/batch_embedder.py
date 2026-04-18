class BatchEmbedder:
    def __init__(self, embedder_func, batch_size=100):
        self.embedder_func = embedder_func
        self.batch_size = batch_size

    def embed_documents(self, documents: list):
        results = []
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            results.extend(self.embedder_func(batch))
        return results
