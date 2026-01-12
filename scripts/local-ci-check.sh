#!/bin/bash
# Local CI validation script - Run this before pushing to catch failures early
# Usage: ./scripts/local-ci-check.sh [--quick|--full]

set -e

MODE="${1:-quick}"
FAILED_CHECKS=()

echo "🔍 Running Local CI Checks (${MODE} mode)..."
echo "================================================"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

run_check() {
    local name="$1"
    local cmd="$2"

    echo -n "▶ ${name}... "

    if eval "$cmd" > /tmp/check_output.log 2>&1; then
        echo -e "${GREEN}✅ PASS${NC}"
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}"
        FAILED_CHECKS+=("$name")
        echo "  Error details:"
        tail -n 10 /tmp/check_output.log | sed 's/^/    /'
        return 1
    fi
}

# 1. Workflow YAML Validation
echo "📋 GitHub Actions Workflows"
echo "----------------------------"
run_check "Workflow YAML syntax" "uv run python3 -c 'import yaml; [yaml.safe_load(open(f)) for f in __import__(\"glob\").glob(\".github/workflows/*.yml\")]'"

# 2. Python Code Quality
echo ""
echo "🐍 Python Code Quality"
echo "----------------------"
if command -v uv &> /dev/null; then
    run_check "Ruff linting" "uv run ruff check . --quiet"
    run_check "Ruff formatting" "uv run ruff format . --check --quiet"
else
    echo -e "${YELLOW}⚠️  UV not installed, skipping Python checks${NC}"
fi

# 3. AgentQMS Validation
echo ""
echo "📚 AgentQMS Artifacts"
echo "---------------------"
if [ -f "AgentQMS/tools/compliance/validate_artifacts.py" ]; then
    export PYTHONPATH=$PWD

    # Quick mode: only changed files
    if [ "$MODE" = "quick" ]; then
        CHANGED_FILES=$(git diff --name-only HEAD docs/artifacts/ 2>/dev/null || echo "")
        if [ -n "$CHANGED_FILES" ]; then
            run_check "AgentQMS validation (changed only)" "cd AgentQMS/bin && uv run python ../tools/compliance/validate_artifacts.py --changed-only"
        else
            echo "  No artifact changes detected, skipping"
        fi
    else
        run_check "AgentQMS validation (all)" "cd AgentQMS/bin && uv run python ../tools/compliance/validate_artifacts.py --all"
    fi
else
    echo -e "${YELLOW}⚠️  AgentQMS validator not found${NC}"
fi

# 4. Test Isolation Check
echo ""
echo "🧪 Test Quality Checks"
echo "----------------------"
if [ -f "scripts/ci/check_test_isolation.py" ]; then
    run_check "Test isolation (mock cleanup)" "uv run python scripts/ci/check_test_isolation.py"
fi

# 5. Quick Test Run (if --full mode)
if [ "$MODE" = "full" ] && command -v pytest &> /dev/null; then
    echo ""
    echo "🧪 Running Quick Tests"
    echo "----------------------"
    run_check "Fast unit tests" "uv run pytest tests/unit -x -q -m 'not slow' --tb=short"
fi

# Summary
echo ""
echo "================================================"
if [ ${#FAILED_CHECKS[@]} -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed!${NC} Safe to push."
    echo ""
    echo "💡 Next steps:"
    echo "  git add ."
    echo "  git commit -m 'your message'"
    echo "  git push"
    exit 0
else
    echo -e "${RED}❌ ${#FAILED_CHECKS[@]} check(s) failed:${NC}"
    for check in "${FAILED_CHECKS[@]}"; do
        echo "  - $check"
    done
    echo ""
    echo "💡 Fix these issues before pushing to avoid CI failures"
    exit 1
fi
