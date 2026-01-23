#!/bin/bash
# AgentQMS Environment Setup
# Source this file to add aqms to PATH
# Usage: source setup-env.sh

# Resolve absolute path of the script, handling symlinks
if command -v readlink >/dev/null 2>&1; then
    SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
else
    # Fallback for systems without readlink -f (unlikely on Linux but good practice)
    SCRIPT_PATH="${BASH_SOURCE[0]}"
fi
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"

# Add bin to PATH
export PATH="$SCRIPT_DIR/bin:$PATH"

# Set PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Project Artifacts - External Storage
export WANDB_DIR="/workspaces/project-artifacts/ocr-external-storage/wandb"
export PYTHONDONTWRITEBYTECODE=1

# Verify
echo "✅ AgentQMS environment configured"
echo "   CLI: $(which aqms 2>/dev/null || echo 'aqms (in $SCRIPT_DIR/bin)')"
echo "   Version: $(aqms --version 2>/dev/null || echo 'Run: aqms --version')"
echo ""
echo "Quick commands:"
echo "  aqms --help              Show all commands"
echo "  aqms validate --all      Validate all artifacts"
echo "  aqms context \"task\"      Get context for task"
