from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseVectorStore(ABC):
    @abstractmethod
    def add_documents(self, documents: List[str], metadata: List[Dict[str, Any]] = None):
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[Any]:
        pass
