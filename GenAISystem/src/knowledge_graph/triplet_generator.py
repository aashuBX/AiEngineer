"""
Triplet Generator — Generates (Subject, Predicate, Object) triplets from text.
Includes deduplication and confidence scoring.
"""

import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Triplet(BaseModel):
    """A single (Subject)-[Predicate]->(Object) triplet."""
    subject: str = Field(..., description="Source entity")
    predicate: str = Field(..., description="Relationship type")
    object: str = Field(..., description="Target entity")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence")
    source_chunk_id: Optional[str] = Field(default=None, description="ID of the source text chunk")
    properties: Dict[str, Any] = Field(default_factory=dict)


class TripletList(BaseModel):
    """Container for structured output parsing."""
    triplets: List[Triplet] = Field(default_factory=list)


class TripletGenerator:
    """Generate knowledge graph triplets from text using an LLM.

    Features:
    - LLM-based triplet extraction with confidence scoring
    - Cross-chunk entity deduplication
    - Configurable predicate normalization
    """

    EXTRACTION_PROMPT = """Extract all factual relationships from the text below as
(Subject, Predicate, Object) triplets.

Rules:
1. Subjects and Objects should be named entities or key concepts.
2. Predicates should be concise relationship labels in UPPER_SNAKE_CASE.
3. Assign a confidence score (0.0-1.0) based on how explicit the relationship is.
4. Only extract relationships that are clearly stated or strongly implied.

Text:
\"\"\"
{text}
\"\"\"

Return JSON:
{{
    "triplets": [
        {{"subject": "...", "predicate": "...", "object": "...", "confidence": 0.95}},
        ...
    ]
}}
"""

    def __init__(self, llm, min_confidence: float = 0.5):
        self.llm = llm
        self.min_confidence = min_confidence
        self._entity_canonical: Dict[str, str] = {}  # lowercase -> canonical

    def generate_triplets(self, text: str, chunk_id: Optional[str] = None) -> List[Triplet]:
        """Generate triplets from a single text chunk.

        Args:
            text: Raw text to extract triplets from.
            chunk_id: Optional identifier for source tracking.

        Returns:
            List of Triplet objects.
        """
        prompt = self.EXTRACTION_PROMPT.format(text=text)

        try:
            structured_llm = self.llm.with_structured_output(TripletList)
            result: TripletList = structured_llm.invoke(prompt)
        except Exception as e:
            logger.warning(f"Structured extraction failed: {e}. Attempting raw parse.")
            return []

        # Post-process: filter by confidence, normalize, deduplicate
        triplets = []
        for t in result.triplets:
            if t.confidence < self.min_confidence:
                continue

            t.subject = self._canonicalize(t.subject)
            t.object = self._canonicalize(t.object)
            t.predicate = t.predicate.upper().replace(" ", "_")
            t.source_chunk_id = chunk_id
            triplets.append(t)

        logger.info(f"Generated {len(triplets)} triplets from chunk {chunk_id or '(unknown)'}")
        return triplets

    def generate_from_documents(
        self,
        texts: List[str],
        chunk_ids: Optional[List[str]] = None,
    ) -> List[Triplet]:
        """Generate triplets from multiple text chunks with cross-chunk deduplication.

        Args:
            texts: List of text chunks.
            chunk_ids: Optional parallel list of chunk identifiers.

        Returns:
            Deduplicated list of all triplets.
        """
        chunk_ids = chunk_ids or [f"chunk_{i}" for i in range(len(texts))]
        all_triplets: List[Triplet] = []

        for text, cid in zip(texts, chunk_ids):
            triplets = self.generate_triplets(text, chunk_id=cid)
            all_triplets.extend(triplets)

        deduplicated = self._deduplicate(all_triplets)
        logger.info(f"Total triplets: {len(all_triplets)} → after dedup: {len(deduplicated)}")
        return deduplicated

    # ------------------------------------------------------------------
    # Deduplication & normalization
    # ------------------------------------------------------------------

    def _canonicalize(self, entity: str) -> str:
        """Normalize entity names for consistency (case-insensitive dedup)."""
        key = entity.strip().lower()
        if key in self._entity_canonical:
            return self._entity_canonical[key]
        # Store the first variant as canonical
        canonical = entity.strip()
        self._entity_canonical[key] = canonical
        return canonical

    @staticmethod
    def _deduplicate(triplets: List[Triplet]) -> List[Triplet]:
        """Remove duplicate triplets, keeping the one with highest confidence."""
        seen: Dict[tuple, Triplet] = {}
        for t in triplets:
            key = (t.subject.lower(), t.predicate, t.object.lower())
            if key not in seen or t.confidence > seen[key].confidence:
                seen[key] = t
        return list(seen.values())
