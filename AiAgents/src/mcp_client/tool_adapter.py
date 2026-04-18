"""
MCP Tool Adapter — converts MCP tools to LangGraph-compatible ToolNode.
Handles schema validation and result formatting.
"""

from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.prebuilt import ToolNode

from src.utils.logger import get_logger

logger = get_logger(__name__)


def adapt_mcp_tools_to_tool_node(mcp_tools: list[BaseTool]) -> ToolNode:
    """
    Wrap MCP tools into a LangGraph ToolNode.

    Args:
        mcp_tools: List of tools from MCPClientManager.tools

    Returns:
        A LangGraph ToolNode ready to be added to a StateGraph.
    """
    logger.info(f"Adapting {len(mcp_tools)} MCP tools to ToolNode")
    return ToolNode(tools=mcp_tools)


def validate_tool_input(tool: BaseTool, inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Validate and coerce tool inputs against the tool's args schema.

    Args:
        tool:   The target tool.
        inputs: Raw input dict.

    Returns:
        Validated input dict.

    Raises:
        ValueError: If required fields are missing or types are wrong.
    """
    schema = getattr(tool, "args_schema", None)
    if schema is None:
        return inputs

    try:
        validated = schema(**inputs)
        return validated.model_dump()
    except Exception as e:
        raise ValueError(f"Tool '{tool.name}' input validation failed: {e}") from e


def format_tool_result(tool_name: str, result: Any) -> dict[str, Any]:
    """
    Standardize a tool execution result for inclusion in AgentState.tool_results.

    Args:
        tool_name: Name of the invoked tool.
        result:    Raw result from tool execution.

    Returns:
        Normalized result dict.
    """
    if isinstance(result, dict):
        content = result
    elif isinstance(result, str):
        content = {"output": result}
    else:
        content = {"output": str(result)}

    return {
        "tool_name": tool_name,
        "result": content,
        "success": True,
    }


def create_structured_tool(
    name: str,
    description: str,
    fn: Any,
    args_schema: Any = None,
) -> StructuredTool:
    """
    Helper to wrap a plain Python function into a LangChain StructuredTool.

    Args:
        name:        Tool name (used by LLM to reference it).
        description: What the tool does (shown to LLM).
        fn:          Async or sync function to execute.
        args_schema: Optional Pydantic model for input validation.

    Returns:
        A StructuredTool instance.
    """
    return StructuredTool.from_function(
        func=fn,
        name=name,
        description=description,
        args_schema=args_schema,
        coroutine=fn if __import__("asyncio").iscoroutinefunction(fn) else None,
    )
