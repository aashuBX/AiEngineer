"""
Entity Extractor — LLM-based Named Entity Recognition for Knowledge Graph construction.
Uses LangChain's LLMGraphTransformer for schema-guided extraction.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.documents import Document
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Pydantic schemas for structured extraction output
# ------------------------------------------------------------------

class ExtractedEntity(BaseModel):
    """A single entity extracted from text."""
    name: str = Field(..., description="Canonical name of the entity")
    entity_type: str = Field(..., description="Type/label, e.g. Person, Organization, Technology")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties")


class ExtractedRelationship(BaseModel):
    """A relationship between two entities."""
    source: str
    target: str
    relationship_type: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class ExtractionResult(BaseModel):
    """Complete extraction result for a chunk of text."""
    entities: List[ExtractedEntity] = Field(default_factory=list)
    relationships: List[ExtractedRelationship] = Field(default_factory=list)


# ------------------------------------------------------------------
# Entity Extractor
# ------------------------------------------------------------------

class EntityExtractor:
    """Extract entities and relationships from text using an LLM.

    Supports two modes:
    1. **LLMGraphTransformer** (recommended) — uses LangChain's experimental
       graph transformer for schema-guided extraction.
    2. **Structured Output** — prompts the LLM with a Pydantic schema and
       parses the response into entities and relationships.
    """

    # Default entity and relationship types to guide extraction
    DEFAULT_ENTITY_TYPES = [
        "Person", "Organization", "Technology", "Concept",
        "Product", "Location", "Event", "Document",
    ]
    DEFAULT_RELATIONSHIP_TYPES = [
        "WORKS_FOR", "DEVELOPED_BY", "RELATED_TO", "PART_OF",
        "USES", "LOCATED_IN", "AUTHORED", "DEPENDS_ON",
    ]

    def __init__(
        self,
        llm,
        allowed_entity_types: Optional[List[str]] = None,
        allowed_relationship_types: Optional[List[str]] = None,
        use_graph_transformer: bool = True,
    ):
        self.llm = llm
        self.allowed_entity_types = allowed_entity_types or self.DEFAULT_ENTITY_TYPES
        self.allowed_relationship_types = allowed_relationship_types or self.DEFAULT_RELATIONSHIP_TYPES
        self.use_graph_transformer = use_graph_transformer
        self._transformer = None

    # ------------------------------------------------------------------
    # LLMGraphTransformer mode
    # ------------------------------------------------------------------

    def _get_transformer(self):
        """Lazy-init the LLMGraphTransformer."""
        if self._transformer is None:
            from langchain_experimental.graph_transformers import LLMGraphTransformer

            self._transformer = LLMGraphTransformer(
                llm=self.llm,
                allowed_nodes=self.allowed_entity_types,
                allowed_relationships=self.allowed_relationship_types,
            )
        return self._transformer

    def extract_graph_documents(self, documents: Sequence[Document]):
        """Use LLMGraphTransformer to extract graph documents.

        Returns LangChain GraphDocument objects (nodes + relationships).
        """
        transformer = self._get_transformer()
        graph_docs = transformer.convert_to_graph_documents(documents)
        logger.info(f"Extracted {sum(len(gd.nodes) for gd in graph_docs)} nodes, "
                     f"{sum(len(gd.relationships) for gd in graph_docs)} relationships "
                     f"from {len(documents)} documents.")
        return graph_docs

    # ------------------------------------------------------------------
    # Structured-output mode
    # ------------------------------------------------------------------

    def extract(self, text: str) -> ExtractionResult:
        """Extract entities and relationships from raw text using structured output.

        Args:
            text: The raw text chunk to analyze.

        Returns:
            ExtractionResult with entities and relationships.
        """
        if self.use_graph_transformer:
            # Wrap text in a Document and use the transformer
            docs = [Document(page_content=text)]
            graph_docs = self.extract_graph_documents(docs)
            return self._graph_docs_to_result(graph_docs)

        # Fall back to direct structured output
        return self._extract_with_structured_output(text)

    def _extract_with_structured_output(self, text: str) -> ExtractionResult:
        """Prompt the LLM with a schema and parse the structured response."""
        prompt = f"""Extract all entities and relationships from the following text.

Entity types allowed: {', '.join(self.allowed_entity_types)}
Relationship types allowed: {', '.join(self.allowed_relationship_types)}

Text:
\"\"\"
{text}
\"\"\"

Return your answer as JSON matching this schema:
{{
    "entities": [{{"name": "...", "entity_type": "...", "properties": {{}}}}],
    "relationships": [{{"source": "...", "target": "...", "relationship_type": "...", "properties": {{}}}}]
}}
"""
        try:
            structured_llm = self.llm.with_structured_output(ExtractionResult)
            result = structured_llm.invoke(prompt)
            logger.info(f"Extracted {len(result.entities)} entities, {len(result.relationships)} relationships")
            return result
        except Exception as e:
            logger.warning(f"Structured extraction failed, returning empty result: {e}")
            return ExtractionResult()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _graph_docs_to_result(graph_docs) -> ExtractionResult:
        """Convert LangChain GraphDocuments to our ExtractionResult schema."""
        entities = []
        relationships = []
        seen_entities = set()

        for gd in graph_docs:
            for node in gd.nodes:
                key = (node.id, node.type)
                if key not in seen_entities:
                    seen_entities.add(key)
                    entities.append(ExtractedEntity(
                        name=node.id,
                        entity_type=node.type,
                        properties=node.properties if hasattr(node, "properties") else {},
                    ))
            for rel in gd.relationships:
                relationships.append(ExtractedRelationship(
                    source=rel.source.id,
                    target=rel.target.id,
                    relationship_type=rel.type,
                    properties=rel.properties if hasattr(rel, "properties") else {},
                ))

        return ExtractionResult(entities=entities, relationships=relationships)

    def extract_batch(self, texts: List[str]) -> List[ExtractionResult]:
        """Extract entities from multiple text chunks."""
        return [self.extract(text) for text in texts]
