"""
Document Structure Chunker — Respects document structure (headings, sections, tables).
Preserves parent-child relationships between sections.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DocumentStructureChunker:
    """Document-structure-aware chunker that respects headings, sections, and tables.

    Supports:
    - Markdown heading-based splitting
    - HTML heading-based splitting
    - Parent-child section relationships
    - Table preservation (keeps tables as single chunks)
    """

    def __init__(
        self,
        max_chunk_size: int = 1500,
        min_chunk_size: int = 100,
        preserve_tables: bool = True,
    ):
        """
        Args:
            max_chunk_size: Hard limit on chunk size in characters.
            min_chunk_size: Minimum chunk size (merge with parent if too small).
            preserve_tables: If True, keep tables as single chunks.
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.preserve_tables = preserve_tables

    def parse_and_chunk(self, text: str, format: str = "markdown") -> List[Dict[str, Any]]:
        """Parse a document and split into structure-aware chunks.

        Args:
            text: The full document text.
            format: 'markdown' or 'html'.

        Returns:
            List of chunk dicts with 'content', 'metadata' (section info, heading hierarchy).
        """
        if format == "html":
            return self._chunk_html(text)
        else:
            return self._chunk_markdown(text)

    # ------------------------------------------------------------------
    # Markdown Chunking
    # ------------------------------------------------------------------

    def _chunk_markdown(self, text: str) -> List[Dict[str, Any]]:
        """Split markdown by heading structure."""
        sections = self._parse_markdown_sections(text)
        chunks = []

        for section in sections:
            content = section["content"].strip()
            if not content:
                continue

            # If section is too large, split further
            if len(content) > self.max_chunk_size:
                sub_chunks = self._split_long_section(content)
                for i, sub in enumerate(sub_chunks):
                    chunks.append({
                        "content": sub,
                        "metadata": {
                            **section["metadata"],
                            "sub_chunk": i,
                            "total_sub_chunks": len(sub_chunks),
                        },
                    })
            elif len(content) < self.min_chunk_size and chunks:
                # Merge with previous chunk
                chunks[-1]["content"] += "\n\n" + content
            else:
                chunks.append({
                    "content": content,
                    "metadata": section["metadata"],
                })

        logger.info(f"DocumentChunker: {len(chunks)} chunks from markdown")
        return chunks

    def _parse_markdown_sections(self, text: str) -> List[Dict[str, Any]]:
        """Parse markdown into sections based on headings."""
        # Pattern for markdown headings
        heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        matches = list(heading_pattern.finditer(text))

        if not matches:
            return [{"content": text, "metadata": {"heading": "Document", "level": 0}}]

        sections = []
        heading_stack: List[Tuple[int, str]] = []  # (level, heading text)

        for i, match in enumerate(matches):
            level = len(match.group(1))
            heading = match.group(2).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            # Update heading hierarchy
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading))

            # Build breadcrumb path
            breadcrumb = " > ".join(h[1] for h in heading_stack)

            content = text[start:end].strip()
            sections.append({
                "content": f"{heading}\n\n{content}" if content else heading,
                "metadata": {
                    "heading": heading,
                    "level": level,
                    "breadcrumb": breadcrumb,
                    "parent_heading": heading_stack[-2][1] if len(heading_stack) > 1 else None,
                },
            })

        return sections

    # ------------------------------------------------------------------
    # HTML Chunking
    # ------------------------------------------------------------------

    def _chunk_html(self, html: str) -> List[Dict[str, Any]]:
        """Split HTML by heading elements (h1-h6)."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.warning("BeautifulSoup not installed. Falling back to markdown chunking.")
            return self._chunk_markdown(html)

        soup = BeautifulSoup(html, "html.parser")
        chunks = []
        heading_tags = ["h1", "h2", "h3", "h4", "h5", "h6"]

        current_heading = "Document"
        current_level = 0
        current_content = []

        for element in soup.children:
            tag_name = element.name if hasattr(element, "name") else None

            if tag_name in heading_tags:
                # Flush previous section
                if current_content:
                    content = "\n".join(current_content).strip()
                    if content:
                        chunks.append({
                            "content": content,
                            "metadata": {"heading": current_heading, "level": current_level},
                        })
                    current_content = []

                current_heading = element.get_text().strip()
                current_level = int(tag_name[1])
            elif tag_name == "table" and self.preserve_tables:
                # Keep table as a single unit
                current_content.append(str(element))
            else:
                text = element.get_text() if hasattr(element, "get_text") else str(element)
                if text.strip():
                    current_content.append(text.strip())

        # Flush last section
        if current_content:
            content = "\n".join(current_content).strip()
            if content:
                chunks.append({
                    "content": content,
                    "metadata": {"heading": current_heading, "level": current_level},
                })

        return chunks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_long_section(self, text: str) -> List[str]:
        """Split a section that exceeds max_chunk_size."""
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) > self.max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks
