"""
Batch Embedder — Efficient batch embedding with progress tracking and error handling.
"""

import logging
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)


class BatchEmbedder:
    """Embed large document collections in configurable batches with progress tracking."""

    def __init__(
        self,
        embedding_function,
        batch_size: int = 100,
        show_progress: bool = True,
    ):
        """
        Args:
            embedding_function: A LangChain Embeddings object or callable.
            batch_size: Number of documents per batch.
            show_progress: If True, display a tqdm progress bar.
        """
        self.embedding_function = embedding_function
        self.batch_size = batch_size
        self.show_progress = show_progress

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a single batch of texts."""
        if hasattr(self.embedding_function, "embed_documents"):
            return self.embedding_function.embed_documents(texts)
        elif callable(self.embedding_function):
            return [self.embedding_function(t) for t in texts]
        else:
            raise TypeError("embedding_function must be a LangChain Embeddings or callable.")

    def embed_documents(
        self,
        documents: List[str],
        retry_failed: bool = True,
        max_retries: int = 3,
    ) -> List[Optional[List[float]]]:
        """Embed a list of documents in batches.

        Args:
            documents: List of text strings to embed.
            retry_failed: Whether to retry failed batches.
            max_retries: Maximum number of retries per failed batch.

        Returns:
            List of embedding vectors (None for any that failed all retries).
        """
        results: List[Optional[List[float]]] = [None] * len(documents)
        total_batches = (len(documents) + self.batch_size - 1) // self.batch_size

        # Optional tqdm progress bar
        iterator = range(0, len(documents), self.batch_size)
        if self.show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, total=total_batches, desc="Embedding", unit="batch")
            except ImportError:
                logger.info("tqdm not installed. Progress bar disabled.")

        failed_batches = []

        for batch_start in iterator:
            batch_end = min(batch_start + self.batch_size, len(documents))
            batch = documents[batch_start:batch_end]

            try:
                embeddings = self._embed_batch(batch)
                for i, emb in enumerate(embeddings):
                    results[batch_start + i] = emb
            except Exception as e:
                logger.warning(f"Batch [{batch_start}:{batch_end}] failed: {e}")
                failed_batches.append((batch_start, batch_end))

        # Retry failed batches
        if retry_failed and failed_batches:
            for attempt in range(1, max_retries + 1):
                if not failed_batches:
                    break
                logger.info(f"Retrying {len(failed_batches)} failed batch(es), attempt {attempt}/{max_retries}")
                still_failed = []
                for batch_start, batch_end in failed_batches:
                    batch = documents[batch_start:batch_end]
                    try:
                        embeddings = self._embed_batch(batch)
                        for i, emb in enumerate(embeddings):
                            results[batch_start + i] = emb
                    except Exception as e:
                        logger.warning(f"Retry attempt {attempt} failed for batch [{batch_start}:{batch_end}]: {e}")
                        still_failed.append((batch_start, batch_end))
                failed_batches = still_failed

        if failed_batches:
            total_failed = sum(end - start for start, end in failed_batches)
            logger.error(f"{total_failed} documents failed to embed after {max_retries} retries.")

        success_count = sum(1 for r in results if r is not None)
        logger.info(f"Embedded {success_count}/{len(documents)} documents successfully.")
        return results

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""
        if hasattr(self.embedding_function, "embed_query"):
            return self.embedding_function.embed_query(query)
        return self._embed_batch([query])[0]
