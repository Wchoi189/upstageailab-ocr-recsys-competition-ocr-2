#!/usr/bin/env bash
# EDS v1.0 Naming Validation Hook
# Blocks ALL-CAPS filenames and validates YYYYMMDD_HHMM_{TYPE}_{slug}.md pattern

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get staged .md files in experiment_manager/experiments/.metadata/ directories only
staged_files=$(git diff --cached --name-only --diff-filter=ACM | grep "^experiment_manager/experiments/.*/.metadata/.*\.md$" || true)

if [ -z "$staged_files" ]; then
    exit 0  # No experiment markdown files to validate
fi

violations_found=false

echo "🔍 EDS v1.0: Validating artifact naming..."

for file in $staged_files; do
    filename=$(basename "$file")

    # Skip README.md (allowed exception)
    if [ "$filename" == "README.md" ]; then
        continue
    fi

    # Skip state.json (not markdown, but check just in case)
    if [ "$filename" == "state.json" ]; then
        continue
    fi

    # Check for ALL-CAPS violations (CRITICAL)
    if echo "$filename" | grep -qE '^[A-Z_]+\.md$'; then
        echo -e "${RED}❌ CRITICAL: ALL-CAPS filename detected${NC}"
        echo -e "   File: ${YELLOW}$file${NC}"
        echo -e "   Violation: ALL-CAPS filenames are PROHIBITED"
        echo -e "   Example: MASTER_ROADMAP.md → 20251217_1530_guide_roadmap.md"
        violations_found=true
        continue
    fi

    # Check for camelCase violations
    if echo "$filename" | grep -qE '[a-z][A-Z]'; then
        echo -e "${RED}❌ CRITICAL: camelCase filename detected${NC}"
        echo -e "   File: ${YELLOW}$file${NC}"
        echo -e "   Violation: camelCase filenames are PROHIBITED"
        echo -e "   Example: failureAnalysis.md → 20251217_1530_assessment_failure_analysis.md"
        violations_found=true
        continue
    fi

    # Check for PascalCase violations
    if echo "$filename" | grep -qE '^[A-Z][a-z]+([A-Z][a-z]+)+\.md$'; then
        echo -e "${RED}❌ CRITICAL: PascalCase filename detected${NC}"
        echo -e "   File: ${YELLOW}$file${NC}"
        echo -e "   Violation: PascalCase filenames are PROHIBITED"
        echo -e "   Example: FailureAnalysis.md → 20251217_1530_assessment_failure_analysis.md"
        violations_found=true
        continue
    fi

    # Check for spaces
    if echo "$filename" | grep -q ' '; then
        echo -e "${RED}❌ CRITICAL: Spaces in filename detected${NC}"
        echo -e "   File: ${YELLOW}$file${NC}"
        echo -e "   Violation: Spaces in filenames are PROHIBITED"
        echo -e "   Example: failure analysis.md → 20251217_1530_assessment_failure_analysis.md"
        violations_found=true
        continue
    fi

    # Validate YYYYMMDD_HHMM_{TYPE}_{slug}.md pattern
    if ! echo "$filename" | grep -qE '^[0-9]{8}_[0-9]{4}_(assessment|report|guide|script|manifest|plan)_[a-z0-9]+(-[a-z0-9]+)*\.md$'; then
        echo -e "${RED}❌ CRITICAL: Invalid naming pattern${NC}"
        echo -e "   File: ${YELLOW}$file${NC}"
        echo -e "   Required: YYYYMMDD_HHMM_{TYPE}_{slug}.md"
        echo -e "   Example: 20251217_1530_assessment_failure_analysis.md"
        echo -e "   Types: assessment, report, guide, plan, script, manifest"
        echo -e "   Slug: lowercase-hyphenated (max 50 chars)"
        violations_found=true
        continue
    fi

    echo -e "${GREEN}✅${NC} $filename"
done

if [ "$violations_found" = true ]; then
    echo ""
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}❌ COMMIT BLOCKED: Naming violations detected${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "EDS v1.0 requires CLI-generated artifacts to prevent naming chaos."
    echo ""
    echo "Fix options:"
    echo "  1. Use CLI to generate artifacts:"
    echo "     eds generate-assessment --experiment <id> --slug <slug>"
    echo "     eds generate-report --experiment <id> --slug <slug>"
    echo "     eds generate-guide --experiment <id> --slug <slug>"
    echo "     eds generate-script --experiment <id> --slug <slug>"
    echo ""
    echo "  2. Rename manually to match pattern:"
    echo "     YYYYMMDD_HHMM_{TYPE}_{slug}.md"
    echo "     Example: 20251217_1530_assessment_failure_analysis.md"
    echo ""
    echo "No bypass mechanism available - violations MUST be fixed."
    echo ""
    exit 1
fi

echo -e "${GREEN}✅ All artifact names comply with EDS v1.0${NC}"
exit 0
