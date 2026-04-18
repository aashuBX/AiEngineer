"""Tests for ingestion modules — data loading, chunking, and preprocessing."""
import pytest


class TestRecursiveChunker:
    """Test recursive text chunker."""

    def test_recursive_chunker_import(self):
        from src.ingestion.chunking.recursive_chunker import RecursiveChunker
        chunker = RecursiveChunker(chunk_size=200, overlap=50)
        assert chunker is not None

    def test_chunking_basic_text(self):
        from src.ingestion.chunking.recursive_chunker import RecursiveChunker
        chunker = RecursiveChunker(chunk_size=50, overlap=10)
        text = "This is a test. " * 20
        chunks = chunker.split_text(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) > 0


class TestDocumentChunker:
    """Test document-structure-aware chunker."""

    def test_document_chunker_initialization(self):
        from src.ingestion.chunking.document_chunker import DocumentStructureChunker
        chunker = DocumentStructureChunker()
        assert chunker is not None

    def test_markdown_chunking(self):
        from src.ingestion.chunking.document_chunker import DocumentStructureChunker
        chunker = DocumentStructureChunker()
        md_text = "# Title\n\nParagraph one.\n\n## Section 1\n\nContent here.\n\n## Section 2\n\nMore content."
        chunks = chunker.parse_and_chunk(md_text, format="markdown")
        assert len(chunks) >= 2


class TestMetadataExtractor:
    """Test metadata extraction from text."""

    def test_extract_basics(self):
        from src.ingestion.preprocessing.metadata_extractor import MetadataExtractor
        extractor = MetadataExtractor()
        text = "Title: My Document\nAuthor: John Doe\nDate: 2024-01-15\n\nContent here."
        result = extractor.extract(text)
        assert "title" in result
        assert result["title"] == "My Document"
        assert result["word_count"] > 0

    def test_extract_dates(self):
        from src.ingestion.preprocessing.metadata_extractor import MetadataExtractor
        extractor = MetadataExtractor()
        dates = extractor.extract_dates("Published on 2024-03-15 and updated January 20, 2025.")
        assert len(dates) >= 1

    def test_extract_keywords(self):
        from src.ingestion.preprocessing.metadata_extractor import MetadataExtractor
        extractor = MetadataExtractor()
        text = "Machine learning uses neural networks for pattern recognition in data."
        keywords = extractor.extract_keywords(text, top_k=5)
        assert len(keywords) > 0


class TestTextCleaner:
    """Test text cleaning utilities."""

    def test_text_cleaner_import(self):
        from src.ingestion.preprocessing.text_cleaner import TextCleaner
        assert TextCleaner is not None
