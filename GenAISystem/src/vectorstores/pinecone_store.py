from .base_store import BaseVectorStore

class PineconeStore(BaseVectorStore):
    def __init__(self, index_name: str):
        self.index_name = index_name

    def add_documents(self, documents, metadata=None):
        pass
        
    def search(self, query, top_k=5):
        return []
