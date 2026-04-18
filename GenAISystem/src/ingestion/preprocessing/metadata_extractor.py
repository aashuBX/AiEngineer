"""
Metadata Extractor — Extract structured metadata from documents (title, author, date, keywords).
Supports multiple document formats and LLM-assisted extraction.
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Extract metadata from documents using pattern matching and optional LLM assistance.

    Extracts:
    - Title, author, date
    - Keywords / key phrases
    - Document type classification
    - Language detection
    - Word and character counts
    """

    # Common date patterns
    DATE_PATTERNS = [
        r'\b(\d{4}-\d{2}-\d{2})\b',                    # 2024-01-15
        r'\b(\d{2}/\d{2}/\d{4})\b',                    # 01/15/2024
        r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4})\b',  # 15 January 2024
        r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s+\d{4})\b',  # January 15, 2024
    ]

    # Common metadata field patterns
    FIELD_PATTERNS = {
        "title": [
            r'^Title:\s*(.+)$',
            r'^#\s+(.+)$',  # Markdown H1
            r'<title>(.+?)</title>',  # HTML title
        ],
        "author": [
            r'^Author:\s*(.+)$',
            r'^By\s+(.+)$',
            r'^Written by:\s*(.+)$',
        ],
        "date": [
            r'^Date:\s*(.+)$',
            r'^Published:\s*(.+)$',
            r'^Created:\s*(.+)$',
        ],
    }

    def __init__(self, llm=None):
        """
        Args:
            llm: Optional LangChain ChatModel for enhanced extraction.
        """
        self.llm = llm

    def extract(self, text: str, source: str = "") -> Dict[str, Any]:
        """Extract metadata from document text.

        Args:
            text: Document text content.
            source: Original file path or URL (for format detection).

        Returns:
            Dict of extracted metadata.
        """
        metadata = {}

        # Basic stats
        metadata["char_count"] = len(text)
        metadata["word_count"] = len(text.split())
        metadata["line_count"] = text.count("\n") + 1

        # Pattern-based extraction
        metadata.update(self.extract_basics(text))

        # Date extraction
        dates = self.extract_dates(text)
        if dates:
            metadata["dates_found"] = dates
            metadata["primary_date"] = dates[0]

        # Source-based metadata
        if source:
            metadata["source"] = source
            metadata["file_type"] = self._detect_file_type(source)

        # Keywords
        metadata["keywords"] = self.extract_keywords(text)

        # Language hint
        metadata["language"] = self._detect_language_hint(text)

        return metadata

    def extract_basics(self, text: str) -> Dict[str, str]:
        """Extract title, author, and date using regex patterns."""
        result = {}

        for field, patterns in self.FIELD_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
                if match:
                    result[field] = match.group(1).strip()
                    break

        return result

    def extract_dates(self, text: str) -> List[str]:
        """Extract all dates from text."""
        dates = []
        for pattern in self.DATE_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.extend(matches)
        return list(set(dates))  # Deduplicate

    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Extract key terms using simple TF-based approach.

        Args:
            text: Document text.
            top_k: Number of keywords to extract.

        Returns:
            List of top keywords.
        """
        # Tokenize and count
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

        # Common stopwords
        stopwords = {
            "the", "and", "for", "are", "but", "not", "you", "all", "can",
            "had", "her", "was", "one", "our", "out", "has", "his", "how",
            "its", "let", "may", "new", "now", "old", "see", "way", "who",
            "did", "get", "got", "him", "have", "from", "with", "this",
            "that", "they", "been", "each", "make", "like", "than", "them",
            "then", "very", "when", "what", "your", "will", "more", "also",
            "into", "some", "such", "just", "over", "only", "other", "about",
            "these", "those", "their", "which", "would", "there", "could",
        }

        word_counts = {}
        for word in words:
            if word not in stopwords:
                word_counts[word] = word_counts.get(word, 0) + 1

        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:top_k]]

    def extract_with_llm(self, text: str) -> Dict[str, Any]:
        """Use LLM for enhanced metadata extraction.

        Args:
            text: Document text (first 2000 chars used).

        Returns:
            LLM-extracted metadata dict.
        """
        if not self.llm:
            return self.extract(text)

        prompt = f"""Extract structured metadata from the following document text.

Text (first 2000 chars):
\"\"\"
{text[:2000]}
\"\"\"

Extract:
1. Title (if apparent)
2. Author (if mentioned)
3. Date (if mentioned)
4. Summary (2-3 sentences)
5. Key topics (list of 5-10 keywords)
6. Document type (article, research paper, manual, email, etc.)

Return as JSON.
"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            import json
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                llm_metadata = json.loads(json_match.group())
                # Merge with basic extraction
                basic = self.extract(text)
                basic.update({k: v for k, v in llm_metadata.items() if v})
                return basic
        except Exception as e:
            logger.warning(f"LLM metadata extraction failed: {e}")

        return self.extract(text)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_file_type(path: str) -> str:
        """Detect file type from path/extension."""
        path_lower = path.lower()
        type_map = {
            ".pdf": "pdf", ".txt": "text", ".md": "markdown",
            ".csv": "csv", ".json": "json", ".html": "html",
            ".htm": "html", ".docx": "docx", ".doc": "doc",
            ".xlsx": "excel", ".py": "python", ".yaml": "yaml",
            ".yml": "yaml",
        }
        for ext, ftype in type_map.items():
            if path_lower.endswith(ext):
                return ftype
        return "unknown"

    @staticmethod
    def _detect_language_hint(text: str) -> str:
        """Simple language detection heuristic."""
        # Count common English function words
        english_markers = ["the", "is", "and", "of", "to", "in", "for", "with"]
        words = text.lower().split()[:200]
        english_count = sum(1 for w in words if w in english_markers)

        if english_count > 5:
            return "en"
        return "unknown"
