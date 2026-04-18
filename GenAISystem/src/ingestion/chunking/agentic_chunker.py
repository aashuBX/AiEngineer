"""
Agentic Chunker — LLM-guided chunking that uses the LLM to determine chunk boundaries.
Generates chunk summaries and metadata automatically.
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ChunkDecision(BaseModel):
    """Structured output for the LLM's chunking decisions."""
    chunks: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of chunks, each with 'content' and 'summary' keys"
    )


class AgenticChunker:
    """Use an LLM to intelligently chunk text based on topic/concept boundaries.

    The LLM analyzes the text and determines where natural topic shifts occur,
    then splits accordingly. Each chunk also gets an auto-generated summary.
    """

    CHUNKING_PROMPT = """You are a document analysis expert. Split the following text into
logically coherent chunks based on topic/concept boundaries.

Rules:
1. Each chunk should cover a single topic or closely related topics.
2. Chunks should be self-contained and understandable on their own.
3. Aim for chunks between {min_size} and {max_size} characters.
4. For each chunk, provide a brief summary (1-2 sentences).

Text:
\"\"\"
{text}
\"\"\"

Return your response as JSON:
{{
    "chunks": [
        {{"content": "...chunk text...", "summary": "...brief summary..."}}
    ]
}}
"""

    def __init__(
        self,
        llm,
        min_chunk_size: int = 200,
        max_chunk_size: int = 1500,
        overlap_sentences: int = 1,
    ):
        """
        Args:
            llm: LangChain ChatModel for intelligent chunking decisions.
            min_chunk_size: Target minimum chunk size in characters.
            max_chunk_size: Target maximum chunk size in characters.
            overlap_sentences: Number of sentences to overlap between chunks.
        """
        self.llm = llm
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_sentences = overlap_sentences

    def chunk_document(self, text: str) -> List[Dict[str, str]]:
        """Split text into topic-coherent chunks using LLM guidance.

        Args:
            text: Full document text.

        Returns:
            List of dicts with 'content' and 'summary' keys.
        """
        # For very short texts, don't chunk
        if len(text) < self.min_chunk_size:
            return [{"content": text, "summary": text[:100]}]

        # For very long texts, split into windows first to fit context
        windows = self._split_into_windows(text)
        all_chunks = []

        for window in windows:
            chunks = self._llm_chunk(window)
            all_chunks.extend(chunks)

        logger.info(f"AgenticChunker: Split into {len(all_chunks)} chunks")
        return all_chunks

    def _llm_chunk(self, text: str) -> List[Dict[str, str]]:
        """Use the LLM to chunk a single window of text."""
        prompt = self.CHUNKING_PROMPT.format(
            text=text,
            min_size=self.min_chunk_size,
            max_size=self.max_chunk_size,
        )

        try:
            structured_llm = self.llm.with_structured_output(ChunkDecision)
            result: ChunkDecision = structured_llm.invoke(prompt)
            return [c for c in result.chunks if c.get("content")]
        except Exception as e:
            logger.warning(f"LLM chunking failed: {e}. Falling back to paragraph splitting.")
            return self._fallback_chunk(text)

    def _split_into_windows(self, text: str, window_size: int = 6000) -> List[str]:
        """Split very long text into windows for LLM processing."""
        if len(text) <= window_size:
            return [text]

        windows = []
        paragraphs = text.split("\n\n")
        current_window = ""

        for para in paragraphs:
            if len(current_window) + len(para) > window_size and current_window:
                windows.append(current_window.strip())
                current_window = para
            else:
                current_window += "\n\n" + para if current_window else para

        if current_window.strip():
            windows.append(current_window.strip())

        return windows

    @staticmethod
    def _fallback_chunk(text: str) -> List[Dict[str, str]]:
        """Fallback: split by paragraphs if LLM fails."""
        paragraphs = text.split("\n\n")
        chunks = []
        for para in paragraphs:
            para = para.strip()
            if para:
                chunks.append({
                    "content": para,
                    "summary": para[:100] + "..." if len(para) > 100 else para,
                })
        return chunks
