"""
File System MCP Server.
Provides safe operations on permitted directories.
"""

import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP

from src.config.settings import settings

mcp = FastMCP("file_system_server")


def _is_safe_path(requested_path: str) -> bool:
    """Check if the path is within the allowed directories."""
    # If no restrictions configured, allow all (dev only!)
    if not settings.fs_allowed_dirs:
        return True

    target = Path(requested_path).resolve()
    for allowed in settings.fs_allowed_dirs:
        allowed_path = Path(allowed).resolve()
        try:
            # Check if target is relative to allowed_path
            target.relative_to(allowed_path)
            return True
        except ValueError:
            pass
    return False


@mcp.tool()
def list_directory(directory_path: str) -> str:
    """
    List contents of a directory.

    Args:
        directory_path: Absolute path to the directory.
    """
    if not _is_safe_path(directory_path):
        return f"Error: Access to '{directory_path}' is denied."

    try:
        items = os.listdir(directory_path)
        return "\n".join(items) if items else "Directory is empty."
    except Exception as e:
        return f"Error reading directory: {e}"


@mcp.tool()
def read_file(file_path: str) -> str:
    """
    Read text content from a file.

    Args:
        file_path: Absolute path to the file.
    """
    if not _is_safe_path(file_path):
        return f"Error: Access to '{file_path}' is denied."

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        return "Error: File appears to be binary or not UTF-8 encoded."
    except Exception as e:
        return f"Error reading file: {e}"


@mcp.tool()
def write_file(file_path: str, content: str) -> str:
    """
    Write text content to a file (overwrites existing).

    Args:
        file_path: Absolute path to the file.
        content: The text content to write.
    """
    if not _is_safe_path(file_path):
        return f"Error: Access to '{file_path}' is denied."

    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error writing file: {e}"
