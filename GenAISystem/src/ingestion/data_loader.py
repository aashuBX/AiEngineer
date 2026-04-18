"""
Data Loader — unified interface for loading documents from various sources.
Wraps LangChain community document loaders and Unstructured.
"""

from typing import Iterator
from pathlib import Path
import logging

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads documents from local directories or specific files."""

    @staticmethod
    def load_directory(
        directory_path: str | Path,
        glob_pattern: str = "**/*",
        use_unstructured: bool = True,
    ) -> Iterator[Document]:
        """
        Load all matching files in a directory.

        Args:
            directory_path: Root directory to search.
            glob_pattern: Pattern to match files (e.g., "**/*.pdf").
            use_unstructured: If True, uses DirectoryLoader with Unstructured.
                              Otherwise uses generic TextLoader.
        """
        path = str(directory_path)
        logger.info(f"Loading directory '{path}' with pattern '{glob_pattern}'")

        if use_unstructured:
            from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
            loader = DirectoryLoader(
                path,
                glob=glob_pattern,
                loader_cls=UnstructuredFileLoader,
                recursive=True,
                show_progress=True,
            )
        else:
            from langchain_community.document_loaders import DirectoryLoader, TextLoader
            loader = DirectoryLoader(
                path,
                glob=glob_pattern,
                loader_cls=TextLoader,
                recursive=True,
                show_progress=True,
                loader_kwargs={"autodetect_encoding": True},
            )

        yield from loader.lazy_load()

    @staticmethod
    def load_web_url(url: str) -> Iterator[Document]:
        """Load content from a webpage URL."""
        logger.info(f"Loading URL: {url}")
        from langchain_community.document_loaders import WebBaseLoader
        loader = WebBaseLoader(url)
        yield from loader.lazy_load()
