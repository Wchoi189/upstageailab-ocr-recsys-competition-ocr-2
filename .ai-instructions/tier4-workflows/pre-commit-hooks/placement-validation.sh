#!/bin/bash
# Pre-commit hook: Placement validation

# Check for markdown files at docs/ root (except allowed)
ROOT_FILES=$(git diff --cached --name-only --diff-filter=ACM | \
    grep -E '^docs/[^/]+\.md$' | \
    grep -v -E 'README|CHANGELOG|CONTRIBUTING' || true)

if [ -n "$ROOT_FILES" ]; then
    echo "❌ ERROR: Files detected at docs/ root (prohibited):"
    echo "$ROOT_FILES"
    echo ""
    echo "💡 Place in: docs/artifacts/{TYPE}/"
    echo "💡 Create via: cd AgentQMS/interface && make create-{TYPE}"
    echo ""
    echo "Allowed at docs/ root: README.md, CHANGELOG.md, CONTRIBUTING.md"
    echo "See: .ai-instructions/tier1-sst/file-placement-rules.yaml"
    exit 1
fi

# Check artifacts are in correct type directories
MISPLACED_ARTIFACTS=$(git diff --cached --name-only --diff-filter=ACM | \
    grep -E '^docs/artifacts/[^/]+\.md$' | \
    grep -v 'INDEX\.md' || true)

if [ -n "$MISPLACED_ARTIFACTS" ]; then
    echo "❌ ERROR: Artifacts at docs/artifacts/ root (must be in type subdirectories):"
    echo "$MISPLACED_ARTIFACTS"
    echo ""
    echo "💡 Valid locations:"
    echo "   - docs/artifacts/implementation_plans/"
    echo "   - docs/artifacts/assessments/"
    echo "   - docs/artifacts/design_documents/"
    echo "   - docs/artifacts/research/"
    echo "   - docs/artifacts/audits/"
    echo "   - docs/artifacts/bug_reports/"
    echo ""
    echo "See: .ai-instructions/tier1-sst/file-placement-rules.yaml"
    exit 1
fi

echo "✅ Placement validation passed"
exit 0
