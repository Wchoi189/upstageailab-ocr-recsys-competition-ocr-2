#!/bin/bash
# Agent-Only Artifact Creation Wrapper
# This script is ONLY for AI agents - humans should not use this

echo "🤖 Agent Artifact Creation (AGENT-ONLY)"
echo "======================================"
echo ""
echo "⚠️  WARNING: This tool is for AI agents only!"
echo "   Humans should use the main project tools."
echo ""
echo "📏 Reminder for agents: Keep generated documentation ultra concise."
echo "   - Use bullet points, not paragraphs."
echo "   - 1–3 lines per concept; no tutorials."
echo ""

# Check if we're in the agent directory
if [ ! -f "Makefile" ]; then
    echo "❌ Error: This script must be run from the agent/ directory"
    echo "   Current directory: $(pwd)"
    echo "   Expected: agent/"
    exit 1
fi

# Run the artifact creation command (containerized implementation layer)
# Note: Set PYTHONPATH to project root for proper imports
SCRIPT_DIR="$(dirname "$0")"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHONPATH="$PROJECT_ROOT" python "$PROJECT_ROOT/AgentQMS/agent_tools/core/artifact_workflow.py" "$@"
