"""
Text Preprocessing — clean text, strip HTML, normalize whitespace.
"""

import re
from typing import Any

from langchain_core.documents import Document


def clean_text(text: str) -> str:
    """Normalize whitespace and remove zero-width characters."""
    # Remove null bytes and zero-width spaces
    text = text.replace("\x00", "").replace("\u200b", "")
    # Normalize unicode to replace smart quotes, etc. left to basic encoding
    text = text.encode("ascii", "ignore").decode("ascii")
    # Compress multiple whitespaces (excluding newlines)
    text = re.sub(r'[ \t]+', ' ', text)
    # Compress multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def remove_html_tags(text: str) -> str:
    """Strip HTML tags from text."""
    import html
    text = html.unescape(text)
    return re.sub(r'<[^>]+>', ' ', text).strip()


class TextCleaner:
    """Pre-processor class to clean Document contents before chunking."""

    @staticmethod
    def process(documents: list[Document], strip_html: bool = False) -> list[Document]:
        """Apply text cleaning to a list of Documents."""
        processed = []
        for doc in documents:
            content = doc.page_content
            if strip_html:
                content = remove_html_tags(content)
            content = clean_text(content)

            # Only keep documents that still have content after parsing
            if content.strip():
                processed.append(
                    Document(
                        page_content=content,
                        metadata=doc.metadata,
                    )
                )
        return processed
