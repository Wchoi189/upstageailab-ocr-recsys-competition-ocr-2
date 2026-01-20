#!/bin/bash
# AgentQMS Environment Setup
# Source this file to add aqms to PATH
# Usage: source setup-env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Add bin to PATH
export PATH="$SCRIPT_DIR/bin:$PATH"

# Set PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Verify
echo "✅ AgentQMS environment configured"
echo "   CLI: $(which aqms 2>/dev/null || echo 'aqms (in $SCRIPT_DIR/bin)')"
echo "   Version: $(aqms --version 2>/dev/null || echo 'Run: aqms --version')"
echo ""
echo "Quick commands:"
echo "  aqms --help              Show all commands"
echo "  aqms validate --all      Validate all artifacts"
echo "  aqms context \"task\"      Get context for task"
