#!/bin/bash
# Test MCP servers with MCP Inspector
#
# Prerequisites:
#   npm install -g @modelcontextprotocol/inspector
#   OR use npx (no install needed)
#
# Usage:
#   chmod +x scripts/test_with_inspector.sh
#   ./scripts/test_with_inspector.sh

set -e

echo "==================================="
echo "  MCP Inspector — Tool Testing"
echo "==================================="
echo ""

# Check if npx is available
if ! command -v npx &> /dev/null; then
    echo "Error: npx not found. Install Node.js first."
    exit 1
fi

echo "Starting MCP Inspector..."
echo "This will open a browser-based UI to test your MCP tools."
echo ""

# Run the unified server through the inspector
npx -y @modelcontextprotocol/inspector python -m src.gateway.unified_server
