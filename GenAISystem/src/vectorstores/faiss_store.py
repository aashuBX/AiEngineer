from .base_store import BaseVectorStore

class FaissStore(BaseVectorStore):
    def __init__(self, embedding_function):
        self.embedding_function = embedding_function
        self.index = None # Lazy load FAISS index

    def add_documents(self, documents, metadata=None):
        pass
        
    def search(self, query, top_k=5):
        return []
