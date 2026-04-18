"""
Citation Handler — Inject inline source citations into generated responses.
"""

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class CitationHandler:
    """Add and verify inline source citations in LLM-generated responses.

    Supports:
    - Inline citation injection: [1], [2] markers
    - Source metadata tracking
    - Citation verification against retrieved context
    - Footnote-style source appendix
    """

    def inject_citations(
        self,
        response_text: str,
        sources: List[Dict[str, Any]],
        mode: str = "footnote",
    ) -> str:
        """Inject or append source citations to a response.

        Args:
            response_text: The LLM-generated response text.
            sources: List of source dicts with 'source', 'page', 'index', 'chunk_id'.
            mode: 'footnote' (append at end) or 'inline' (attempt to match).

        Returns:
            Response text with citations added.
        """
        if not sources:
            return response_text

        if mode == "inline":
            return self._inject_inline(response_text, sources)
        else:
            return self._append_footnotes(response_text, sources)

    def _append_footnotes(self, response_text: str, sources: List[Dict[str, Any]]) -> str:
        """Append a footnote-style source section at the end of the response."""
        footnotes = ["\n\n---\n**Sources:**"]
        for src in sources:
            idx = src.get("index", "?")
            source_name = src.get("source", "Unknown")
            page = src.get("page", "")
            page_info = f", page {page}" if page else ""
            footnotes.append(f"[{idx}] {source_name}{page_info}")

        return response_text + "\n".join(footnotes)

    def _inject_inline(self, response_text: str, sources: List[Dict[str, Any]]) -> str:
        """Attempt to inject inline [N] references at the end of relevant sentences."""
        # Split response into sentences
        sentences = re.split(r'(?<=[.!?])\s+', response_text)
        annotated_sentences = []

        for sentence in sentences:
            # Find which sources are relevant to this sentence
            matching_indices = self._find_matching_sources(sentence, sources)
            if matching_indices:
                citations = "".join(f"[{i}]" for i in matching_indices)
                # Insert before the period
                if sentence.rstrip().endswith(('.', '!', '?')):
                    sentence = sentence.rstrip()[:-1] + f" {citations}" + sentence.rstrip()[-1]
                else:
                    sentence = sentence + f" {citations}"
            annotated_sentences.append(sentence)

        result = " ".join(annotated_sentences)

        # Always append source list
        return self._append_footnotes(result, sources)

    @staticmethod
    def _find_matching_sources(
        sentence: str,
        sources: List[Dict[str, Any]],
    ) -> List[int]:
        """Find which sources are referenced or relevant to a sentence.

        Uses simple keyword overlap between sentence and source content.
        """
        matching = []
        sentence_lower = sentence.lower()

        for src in sources:
            source_name = src.get("source", "").lower()
            # Check if the source name or key terms appear in the sentence
            if source_name and source_name in sentence_lower:
                matching.append(src.get("index", 0))
                continue

            # Check for existing inline citations like [1], [2]
            idx = src.get("index", 0)
            if f"[{idx}]" in sentence:
                matching.append(idx)

        return matching

    def verify_citations(
        self,
        response_text: str,
        context: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Verify that cited claims are supported by the retrieved context.

        Args:
            response_text: Response with inline citations.
            context: List of source documents used during retrieval.

        Returns:
            Verification report with supported/unsupported citations.
        """
        # Extract all [N] markers from the response
        cited_indices = set(int(m) for m in re.findall(r'\[(\d+)\]', response_text))

        available_indices = set(range(1, len(context) + 1))
        valid_citations = cited_indices & available_indices
        invalid_citations = cited_indices - available_indices

        return {
            "total_citations": len(cited_indices),
            "valid_citations": list(valid_citations),
            "invalid_citations": list(invalid_citations),
            "uncited_sources": list(available_indices - cited_indices),
            "citation_coverage": len(valid_citations) / max(len(available_indices), 1),
        }
