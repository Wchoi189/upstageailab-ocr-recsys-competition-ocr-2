#!/bin/bash
# Pre-commit hook: Naming validation

# Check for ALL-CAPS filenames at docs/ root (except README, CHANGELOG)
CAPS_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep -E '^docs/[A-Z_]+\.md$' | grep -v -E 'README|CHANGELOG' || true)

if [ -n "$CAPS_FILES" ]; then
    echo "❌ ERROR: ALL-CAPS filenames detected at docs/ root:"
    echo "$CAPS_FILES"
    echo ""
    echo "💡 Use proper naming: YYYY-MM-DD_HHMM_{TYPE}_slug.md"
    echo "💡 Create via: cd AgentQMS/interface && make create-{TYPE}"
    echo ""
    echo "See: AgentQMS/standards/tier1-sst/naming-conventions.yaml"
    exit 1
fi

# Check for underscore_case in artifact slugs (should be kebab-case)
UNDERSCORE_SLUGS=$(git diff --cached --name-only --diff-filter=ACM | \
    grep -E '^docs/artifacts/.*\.md$' | \
    grep -E '_[a-z]+_[a-z]+_[a-z_]+\.md$' || true)

if [ -n "$UNDERSCORE_SLUGS" ]; then
    echo "⚠️  WARNING: underscore_case detected in artifact slugs (prefer kebab-case):"
    echo "$UNDERSCORE_SLUGS"
    echo ""
    echo "💡 Prefer: YYYY-MM-DD_HHMM_type_kebab-case-slug.md"
fi

echo "✅ Naming validation passed"
exit 0
