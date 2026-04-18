"""
Semantic Chunker — Split text into semantically coherent chunks using embedding similarity.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SemanticChunker:
    """Group semantically similar sentences together by detecting topic boundaries.

    Uses embedding cosine similarity between consecutive sentences to find
    natural breakpoints where the topic shifts.
    """

    def __init__(
        self,
        embedding_function,
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
    ):
        """
        Args:
            embedding_function: LangChain Embeddings or callable.
            similarity_threshold: Cosine similarity below this triggers a chunk boundary.
            min_chunk_size: Minimum characters per chunk.
            max_chunk_size: Maximum characters per chunk (hard split if exceeded).
        """
        self.embedding_function = embedding_function
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def split_text(self, text: str) -> List[str]:
        """Split text into semantically coherent chunks.

        Args:
            text: The full text to split.

        Returns:
            List of text chunks.
        """
        # Split into sentences
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [text]

        # Embed all sentences
        embeddings = self._embed_sentences(sentences)
        if embeddings is None:
            return [text]

        # Find boundary points based on similarity drops
        boundaries = self._find_boundaries(embeddings)

        # Create chunks from boundaries
        chunks = self._create_chunks(sentences, boundaries)

        logger.info(f"SemanticChunker: {len(sentences)} sentences → {len(chunks)} chunks")
        return chunks

    def split_documents(
        self,
        documents: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Split a list of documents, preserving metadata.

        Args:
            documents: List of dicts with 'content' and optional 'metadata'.

        Returns:
            List of chunked documents with inherited metadata.
        """
        chunked_docs = []
        for doc in documents:
            text = doc.get("content", "")
            metadata = doc.get("metadata", {})
            chunks = self.split_text(text)
            for i, chunk in enumerate(chunks):
                chunked_docs.append({
                    "content": chunk,
                    "metadata": {**metadata, "chunk_index": i, "total_chunks": len(chunks)},
                })
        return chunked_docs

    # ------------------------------------------------------------------
    # Internal Methods
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        import re
        # Split on sentence-ending punctuation followed by whitespace
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _embed_sentences(self, sentences: List[str]) -> Optional[np.ndarray]:
        """Embed all sentences."""
        try:
            if hasattr(self.embedding_function, "embed_documents"):
                vectors = self.embedding_function.embed_documents(sentences)
            else:
                vectors = [self.embedding_function(s) for s in sentences]
            return np.array(vectors)
        except Exception as e:
            logger.error(f"Failed to embed sentences: {e}")
            return None

    def _find_boundaries(self, embeddings: np.ndarray) -> List[int]:
        """Find chunk boundaries based on cosine similarity between consecutive sentences."""
        boundaries = [0]

        for i in range(1, len(embeddings)):
            similarity = self._cosine_similarity(embeddings[i - 1], embeddings[i])
            if similarity < self.similarity_threshold:
                boundaries.append(i)

        return boundaries

    def _create_chunks(self, sentences: List[str], boundaries: List[int]) -> List[str]:
        """Create chunks from sentences and boundary indices."""
        chunks = []
        boundaries.append(len(sentences))  # Add end boundary

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            chunk_text = " ".join(sentences[start:end])

            # Handle min/max size constraints
            if len(chunk_text) < self.min_chunk_size and chunks:
                # Merge with previous chunk if too small
                chunks[-1] = chunks[-1] + " " + chunk_text
            elif len(chunk_text) > self.max_chunk_size:
                # Hard split if too large
                while len(chunk_text) > self.max_chunk_size:
                    split_point = chunk_text[:self.max_chunk_size].rfind(". ")
                    if split_point == -1:
                        split_point = self.max_chunk_size
                    chunks.append(chunk_text[:split_point + 1].strip())
                    chunk_text = chunk_text[split_point + 1:].strip()
                if chunk_text:
                    chunks.append(chunk_text)
            else:
                chunks.append(chunk_text)

        return chunks

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
