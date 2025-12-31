#!/bin/bash
# Agent-Only Compliance Wrapper
# This script is ONLY for AI agents - humans should not use this

echo "🤖 Agent Compliance (AGENT-ONLY)"
echo "================================"
echo ""
echo "⚠️  WARNING: This tool is for AI agents only!"
echo "   Humans should use the main project tools."
echo ""

# Check if we're in the agent directory
if [ ! -f "Makefile" ]; then
    echo "❌ Error: This script must be run from the agent/ directory"
    echo "   Current directory: $(pwd)"
    echo "   Expected: agent/"
    exit 1
fi

# Run the compliance command (containerized implementation layer)
# Note: Set PYTHONPATH to project root for proper imports
SCRIPT_DIR="$(dirname "$0")"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHONPATH="$PROJECT_ROOT" python "$PROJECT_ROOT/AgentQMS/agent_tools/compliance/monitor_artifacts.py" "$@"
