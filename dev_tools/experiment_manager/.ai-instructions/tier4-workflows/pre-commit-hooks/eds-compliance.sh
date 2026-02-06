#!/usr/bin/env bash
# EDS v1.0 Compliance Validation Hook
# Validates YAML frontmatter against EDS v1.0 specification

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
COMPLIANCE_CHECKER="$REPO_ROOT/experiment-tracker/.ai-instructions/schema/compliance-checker.py"

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get staged .md files in experiment-tracker/experiments/.metadata/ directories only
staged_files=$(git diff --cached --name-only --diff-filter=ACM | grep "^experiment-tracker/experiments/.*/.metadata/.*\.md$" || true)

if [ -z "$staged_files" ]; then
    exit 0  # No experiment markdown files to validate
fi

# Check if compliance checker exists
if [ ! -f "$COMPLIANCE_CHECKER" ]; then
    echo -e "${RED}❌ ERROR: Compliance checker not found${NC}"
    echo -e "   Expected: $COMPLIANCE_CHECKER"
    echo -e "   Cannot validate EDS v1.0 compliance"
    exit 1
fi

violations_found=false

echo "🔍 EDS v1.0: Validating frontmatter compliance..."

for file in $staged_files; do
    filename=$(basename "$file")

    # Skip README.md (allowed exception)
    if [ "$filename" == "README.md" ]; then
        continue
    fi

    # Run compliance checker on file
    if python3 "$COMPLIANCE_CHECKER" "$REPO_ROOT/$file" >/dev/null 2>&1; then
        echo -e "${GREEN}✅${NC} $filename"
    else
        echo -e "${RED}❌ CRITICAL: Compliance validation failed${NC}"
        echo -e "   File: ${YELLOW}$file${NC}"
        echo ""
        # Run again to show detailed errors
        python3 "$COMPLIANCE_CHECKER" "$REPO_ROOT/$file" 2>&1 | sed 's/^/   /'
        echo ""
        violations_found=true
    fi
done

if [ "$violations_found" = true ]; then
    echo ""
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}❌ COMMIT BLOCKED: Compliance violations detected${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "EDS v1.0 requires valid YAML frontmatter with correct fields."
    echo ""
    echo "Common violations:"
    echo "  - Missing required fields (ads_version, type, experiment_id, etc.)"
    echo "  - Invalid field formats (dates, patterns, enums)"
    echo "  - Prohibited content (user-oriented phrases, emoji)"
    echo "  - Type-specific field mismatches"
    echo ""
    echo "Fix options:"
    echo "  1. Use CLI to generate artifacts (ensures correct frontmatter):"
    echo "     eds generate-assessment --experiment <id> --slug <slug>"
    echo ""
    echo "  2. Manually fix frontmatter to match EDS v1.0 specification:"
    echo "     See: experiment-tracker/.ai-instructions/schema/eds-v1.0-spec.yaml"
    echo ""
    echo "  3. Validate before committing:"
    echo "     python3 $COMPLIANCE_CHECKER <file>"
    echo ""
    echo "No bypass mechanism available - violations MUST be fixed."
    echo ""
    exit 1
fi

echo -e "${GREEN}✅ All artifacts comply with EDS v1.0 frontmatter requirements${NC}"
exit 0
