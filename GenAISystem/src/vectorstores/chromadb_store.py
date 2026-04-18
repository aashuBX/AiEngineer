from .base_store import BaseVectorStore

class ChromaDBStore(BaseVectorStore):
    def __init__(self, persist_directory: str):
        self.persist_directory = persist_directory

    def add_documents(self, documents, metadata=None):
        pass
        
    def search(self, query, top_k=5):
        return []
