#!/bin/bash
# Pre-commit hook: ADS v1.0 compliance validation

# Check YAML files in AgentQMS/standards/ for compliance
YAML_FILES=$(git diff --cached --name-only --diff-filter=ACM | \
    grep -E '^\AgentQMS/standards/.*\.yaml$' || true)

if [ -n "$YAML_FILES" ]; then
    echo "🔍 Validating ADS v1.0 compliance for YAML files..."

    for file in $YAML_FILES; do
        if [ -f "$file" ]; then
            if ! uv run python AgentQMS/standards/schemas/compliance-checker.py "$file"; then
                echo ""
                echo "❌ ADS v1.0 compliance validation failed"
                echo "See: AgentQMS/standards/schemas/ads-v2.0-spec.yaml"
                exit 1
            fi
        fi
    done

    echo "✅ ADS v1.0 compliance passed for all YAML files"
fi

exit 0
