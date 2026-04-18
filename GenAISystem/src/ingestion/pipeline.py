"""
Full Ingestion Pipeline.
Chains together loader -> cleaner -> chunker -> (vectorstore / knowledge graph).
"""

import logging
from pathlib import Path

from src.ingestion.data_loader import DataLoader
from src.ingestion.preprocessing.text_cleaner import TextCleaner
from src.ingestion.chunking.semantic import recursive_chunk, semantic_chunk

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Orchestrates the ingestion of documents into vector stores and knowledge graphs."""

    def __init__(
        self,
        vector_store: Any | None = None,
        knowledge_graph: Any | None = None,
        embedder: Any | None = None,
    ):
        """
        Args:
            vector_store: Target vector store instance.
            knowledge_graph: Target graph builder instance.
            embedder: Embedding model (optional, required if using semantic chunking).
        """
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.embedder = embedder

    def run_directory_ingestion(
        self,
        directory: str | Path,
        strategy: str = "recursive",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> dict[str, int]:
        """
        Run the full ingestion pipeline on a directory.

        Returns:
            Dict containing stats about the ingestion.
        """
        logger.info(f"Starting ingestion pipeline for directory: {directory}")

        # 1. Load
        raw_docs = list(DataLoader.load_directory(directory))
        if not raw_docs:
            logger.warning("No documents found to ingest.")
            return {"docs_loaded": 0, "chunks_created": 0}

        # 2. Clean
        clean_docs = TextCleaner.process(raw_docs, strip_html=False)

        # 3. Chunk
        if strategy == "semantic" and self.embedder:
            chunks = semantic_chunk(clean_docs, self.embedder)
        else:
            chunks = recursive_chunk(clean_docs, chunk_size, chunk_overlap)

        logger.info(f"Generated {len(chunks)} chunks from {len(clean_docs)} documents.")

        # 4. Store in Vector DB
        if self.vector_store:
            logger.info("Adding documents to Vector Store...")
            self.vector_store.add_documents(chunks)

        # 5. Extract Entities to Knowledge Graph
        if self.knowledge_graph:
            logger.info("Extracting graph triples to Knowledge Graph...")
            self.knowledge_graph.process_documents(chunks)

        return {
            "docs_loaded": len(clean_docs),
            "chunks_created": len(chunks),
        }
