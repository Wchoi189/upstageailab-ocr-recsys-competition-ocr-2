#!/bin/bash
# Cursor Agent Self-Validation Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "🔍 Validating Cursor agent configuration..."

python3 "$ROOT_DIR/AgentQMS/standards/schema/compliance-checker.py" \
  "$SCRIPT_DIR/config.yaml" \
  "$SCRIPT_DIR/quick-reference.yaml"

echo "✅ All Cursor agent files validated successfully"
exit 0
