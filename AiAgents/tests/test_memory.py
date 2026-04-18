"""Tests for memory modules."""
import pytest
from unittest.mock import MagicMock


class TestConversationMemory:
    """Test conversation buffer and summary memory."""

    def test_memory_initialization(self):
        from src.memory.conversation_memory import ConversationMemory
        memory = ConversationMemory()
        assert memory is not None


class TestLongTermMemory:
    """Test long-term vector-backed memory."""

    def test_long_term_memory_initialization(self):
        from src.memory.long_term_memory import LongTermMemory
        memory = LongTermMemory()
        assert memory is not None


class TestCheckpointer:
    """Test graph state checkpointing."""

    def test_checkpointer_initialization(self):
        from src.memory.checkpointer import create_checkpointer
        checkpointer = create_checkpointer()
        assert checkpointer is not None
