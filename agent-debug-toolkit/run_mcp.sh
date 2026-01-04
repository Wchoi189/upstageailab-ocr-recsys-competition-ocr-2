#!/usr/bin/env bash
# Agent Debug Toolkit MCP Server Launcher
# Usage: ./run_mcp.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set project root for the MCP server
export PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Run the MCP server
exec uv run python -m agent_debug_toolkit.mcp_server
