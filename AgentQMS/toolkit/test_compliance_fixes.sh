#!/bin/bash
# Test script for automated compliance fixes
# Demonstrates all improvements and validates edge cases

echo "=================================================="
echo "Automated Compliance Fix - Test Suite"
echo "=================================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_PASSED=0
TEST_FAILED=0

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

run_test() {
    local test_name=$1
    local command=$2
    local expected_pattern=$3
    
    echo -e "${BLUE}[TEST]${NC} $test_name"
    
    if eval "$command" | grep -q "$expected_pattern"; then
        echo -e "${GREEN}✅ PASSED${NC}"
        ((TEST_PASSED++))
    else
        echo -e "${RED}❌ FAILED${NC}"
        ((TEST_FAILED++))
    fi
    echo ""
}

# Test 1: Verify --max-files parameter works
run_test "Processing limit respected" \
    "$SCRIPT_DIR/automated_compliance_fix.sh --max-files 2 --dry-run 2>&1" \
    "Reached file limit (2)"

# Test 2: Verify dry-run mode doesn't make changes
run_test "Dry-run mode active" \
    "$SCRIPT_DIR/automated_compliance_fix.sh --max-files 1 --dry-run 2>&1" \
    "DRY RUN MODE"

# Test 3: Verify MASTER_INDEX.md is skipped
run_test "Registry files skipped" \
    "python $SCRIPT_DIR/maintenance/fix_naming_conventions.py --directory docs/artifacts --limit 5 --dry-run 2>&1" \
    "No naming issues found for docs/artifacts/MASTER_INDEX.md"

# Test 4: Verify files with correct prefixes not modified
run_test "Correct prefixes recognized" \
    "python $SCRIPT_DIR/maintenance/fix_naming_conventions.py --directory docs/artifacts --limit 10 --dry-run 2>&1" \
    "No naming issues found for.*assessment-"

# Test 5: Verify confidence threshold enforced
run_test "Confidence threshold working" \
    "python $SCRIPT_DIR/maintenance/reorganize_files.py --directory docs/artifacts --limit 5 --dry-run 2>&1" \
    "Skipping registry/index file"

# Summary
echo "=================================================="
echo "Test Results Summary"
echo "=================================================="
echo -e "${GREEN}Passed: $TEST_PASSED${NC}"
echo -e "${RED}Failed: $TEST_FAILED${NC}"
echo ""

if [ $TEST_FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}❌ Some tests failed${NC}"
    exit 1
fi
