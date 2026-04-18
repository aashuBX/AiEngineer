import hashlib
import json

class EmbeddingCache:
    def __init__(self, cache_file: str = "embedding_cache.json"):
        self.cache_file = cache_file
        self.cache = self._load_cache()

    def _load_cache(self):
        try:
            with open(self.cache_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _hash_text(self, text: str):
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, text: str):
        return self.cache.get(self._hash_text(text))

    def set(self, text: str, embedding: list):
        self.cache[self._hash_text(text)] = embedding
