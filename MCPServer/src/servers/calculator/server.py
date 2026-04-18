"""
Calculator MCP Server.
Provides secure mathematical operations to AI Agents.
"""

from mcp.server.fastmcp import FastMCP

# Create a FastMCP app
mcp = FastMCP("calculator_server", description="Secure mathematical and logical operations")


@mcp.tool()
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    Supports basic arithmetic (+, -, *, /, **).
    Do NOT use this for code execution.

    Args:
        expression: The math expression to evaluate, e.g., "(45 * 3) / 2.5".
    """
    # Restrict characters to math symbols to prevent code injection
    import re
    allowed = re.compile(r"^[\d\.\s\+\-\*\/\(\)]+$")
    if not allowed.match(expression):
        return "Error: Invalid characters in expression. Only numbers and basic math operators are allowed."

    try:
        # Use Python's built-in eval safely since we regex-checked it
        result = eval(expression, {"__builtins__": None}, {})
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as e:
        return f"Error: Failed to evaluate expression. {e}"


@mcp.tool()
def fibonacci(n: int) -> str:
    """
    Calculate the nth Fibonacci number.

    Args:
        n: The index of the Fibonacci sequence (must be 0-100).
    """
    if n < 0 or n > 100:
        return "Error: n must be between 0 and 100."

    if n == 0:
        return "0"
    if n == 1:
        return "1"

    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return str(b)
