"""
Semantic and Recursive Chunking Strategies.
"""

import logging
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def recursive_chunk(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """
    Standard recursive character text splitting.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    logger.info(f"Recursive chunking: size={chunk_size}, overlap={chunk_overlap}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def semantic_chunk(
    documents: list[Document],
    embeddings_model: Any | None = None,
    breakpoint_threshold: float = 0.95,
) -> list[Document]:
    """
    Semantic chunking splits text based on semantic similarity of sentences,
    using an embedding model to find natural topic boundaries.
    """
    try:
        from langchain_experimental.text_splitter import SemanticChunker
    except ImportError:
        logger.warning("langchain-experimental not installed. Falling back to recursive chunking.")
        return recursive_chunk(documents)

    if not embeddings_model:
        raise ValueError("Semantic chunking requires an embeddings model")

    logger.info(f"Semantic chunking: threshold={breakpoint_threshold}")
    splitter = SemanticChunker(
        embeddings=embeddings_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=breakpoint_threshold * 100,
    )
    return splitter.split_documents(documents)
