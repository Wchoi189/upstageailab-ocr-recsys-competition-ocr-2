#!/bin/bash
# Claude Agent Self-Validation Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "🔍 Validating Claude agent configuration..."

# 1. Validate config.yaml against ADS v1.0
echo "📋 Checking config.yaml compliance..."
python3 "$ROOT_DIR/AgentQMS/standards/schema/compliance-checker.py" \
  "$SCRIPT_DIR/config.yaml"

# 2. Validate quick-reference.yaml
echo "📋 Checking quick-reference.yaml compliance..."
python3 "$ROOT_DIR/AgentQMS/standards/schema/compliance-checker.py" \
  "$SCRIPT_DIR/quick-reference.yaml"

# 3. Check dependencies exist
echo "🔗 Checking dependencies..."
for dep in naming-conventions.yaml file-placement-rules.yaml workflow-requirements.yaml validation-protocols.yaml prohibited-actions.yaml; do
  if [ ! -f "$ROOT_DIR/AgentQMS/standards/tier1-sst/$dep" ]; then
    echo "❌ Missing dependency: tier1-sst/$dep"
    exit 1
  fi
done

echo "✅ All Claude agent files validated successfully"
exit 0
