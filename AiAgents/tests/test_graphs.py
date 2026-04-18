"""Tests for LangGraph workflow patterns."""
import pytest
from unittest.mock import MagicMock


class TestSingleAgentGraph:
    """Test single ReAct agent graph construction."""

    def test_graph_creation(self):
        from src.graphs.single_agent_graph import create_react_agent_graph
        mock_llm = MagicMock()
        # Should not raise
        graph = create_react_agent_graph(llm=mock_llm, tools=[])
        assert graph is not None

    def test_graph_has_invoke(self):
        from src.graphs.single_agent_graph import create_react_agent_graph
        mock_llm = MagicMock()
        graph = create_react_agent_graph(llm=mock_llm, tools=[])
        assert hasattr(graph, "invoke")


class TestMultiAgentGraph:
    """Test supervisor-worker graph construction."""

    def test_supervisor_graph_creation(self):
        from src.graphs.multi_agent_graph import create_supervisor_graph
        mock_llm = MagicMock()
        graph = create_supervisor_graph(llm=mock_llm, worker_names=["a", "b"])
        assert graph is not None


class TestPlanExecuteGraph:
    """Test plan-and-execute graph construction."""

    def test_plan_execute_creation(self):
        from src.graphs.plan_execute_graph import create_plan_execute_graph
        mock_llm = MagicMock()
        graph = create_plan_execute_graph(llm=mock_llm)
        assert graph is not None


class TestMapReduceGraph:
    """Test map-reduce graph construction."""

    def test_map_reduce_creation(self):
        from src.graphs.map_reduce_graph import create_map_reduce_graph
        mock_llm = MagicMock()
        graph = create_map_reduce_graph(llm=mock_llm)
        assert graph is not None
