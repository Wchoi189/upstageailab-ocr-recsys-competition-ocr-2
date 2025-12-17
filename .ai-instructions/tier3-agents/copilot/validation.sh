#!/bin/bash
# GitHub Copilot Agent Self-Validation Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "🔍 Validating GitHub Copilot agent configuration..."

python3 "$ROOT_DIR/.ai-instructions/schema/compliance-checker.py" \
  "$SCRIPT_DIR/config.yaml" \
  "$SCRIPT_DIR/quick-reference.yaml"

echo "🔗 Checking dependencies..."
for dep in naming-conventions.yaml file-placement-rules.yaml workflow-requirements.yaml; do
  if [ ! -f "$ROOT_DIR/.ai-instructions/tier1-sst/$dep" ]; then
    echo "❌ Missing dependency: tier1-sst/$dep"
    exit 1
  fi
done

if [ ! -f "$ROOT_DIR/.ai-instructions/tier2-framework/tool-catalog.yaml" ]; then
  echo "❌ Missing dependency: tier2-framework/tool-catalog.yaml"
  exit 1
fi

echo "✅ All GitHub Copilot agent files validated successfully"
exit 0
