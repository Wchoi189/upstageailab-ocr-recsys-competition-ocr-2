#!/bin/bash
# Gemini Agent Self-Validation Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "🔍 Validating Gemini agent configuration..."

python3 "$ROOT_DIR/AgentQMS/standards/schemas/compliance-checker.py" \
  "$SCRIPT_DIR/config.yaml" \
  "$SCRIPT_DIR/quick-reference.yaml"

echo "✅ All Gemini agent files validated successfully"
exit 0
