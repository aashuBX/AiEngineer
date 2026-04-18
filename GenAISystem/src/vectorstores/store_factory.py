from .faiss_store import FaissStore
from .chromadb_store import ChromaDBStore

class VectorStoreFactory:
    @staticmethod
    def get_store(provider: str, **kwargs):
        if provider == "faiss":
            return FaissStore(kwargs.get("embedding_function"))
        elif provider == "chroma":
            return ChromaDBStore(kwargs.get("persist_directory", "./chroma_db"))
        raise ValueError("Unsupported vector store provider")
