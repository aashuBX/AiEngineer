from .base_store import BaseVectorStore

class QdrantStore(BaseVectorStore):
    def __init__(self, collection_name: str, url: str):
        self.collection_name = collection_name
        self.url = url

    def add_documents(self, documents, metadata=None):
        pass
        
    def search(self, query, top_k=5):
        return []
