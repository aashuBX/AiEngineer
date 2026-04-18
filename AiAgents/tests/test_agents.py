"""Tests for agent modules."""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch


class TestIntentAgent:
    """Test the IntentAgent classification logic."""

    def test_intent_agent_initialization(self):
        """Verify IntentAgent can be instantiated."""
        from src.agents.intent.intent_agent import IntentAgent
        mock_llm = MagicMock()
        agent = IntentAgent(llm=mock_llm)
        assert agent is not None
        assert agent.llm == mock_llm

    def test_intent_agent_has_process_method(self):
        """Ensure process method exists."""
        from src.agents.intent.intent_agent import IntentAgent
        mock_llm = MagicMock()
        agent = IntentAgent(llm=mock_llm)
        assert hasattr(agent, "process")


class TestGuardrailAgent:
    """Test the GuardrailAgent safety checks."""

    def test_guardrail_agent_initialization(self):
        from src.agents.guardrail.guardrail_agent import GuardrailAgent
        mock_llm = MagicMock()
        agent = GuardrailAgent(llm=mock_llm)
        assert agent is not None

    def test_guardrail_agent_has_process_method(self):
        from src.agents.guardrail.guardrail_agent import GuardrailAgent
        mock_llm = MagicMock()
        agent = GuardrailAgent(llm=mock_llm)
        assert hasattr(agent, "process")


class TestFAQAgent:
    """Test the FAQAgent search logic."""

    def test_faq_agent_initialization(self):
        from src.agents.faq.faq_agent import FAQAgent
        mock_llm = MagicMock()
        agent = FAQAgent(llm=mock_llm)
        assert agent is not None


class TestCRMAgent:
    """Test the CRMAgent data retrieval."""

    def test_crm_agent_initialization(self):
        from src.agents.crm.crm_agent import CRMAgent
        mock_llm = MagicMock()
        agent = CRMAgent(llm=mock_llm)
        assert agent is not None


class TestFeedbackAgent:
    """Test the FeedbackAgent feedback processing."""

    def test_feedback_agent_initialization(self):
        from src.agents.feedback.feedback_agent import FeedbackAgent
        mock_llm = MagicMock()
        agent = FeedbackAgent(llm=mock_llm)
        assert agent is not None


class TestHandoffAgent:
    """Test the HandoffAgent escalation logic."""

    def test_handoff_agent_initialization(self):
        from src.agents.handoff.handoff_agent import HandoffAgent
        mock_llm = MagicMock()
        agent = HandoffAgent(llm=mock_llm)
        assert agent is not None
