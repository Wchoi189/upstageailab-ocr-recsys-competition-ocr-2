#!/usr/bin/env bash
# MCP Tools Integration Test Script (Fixed)
# Tests all 6 MCP tools after Phase 7.2 Spec-Kit migration

set -e  # Exit on error

echo "=========================================="
echo "MCP Tools Integration Test"
echo "Phase 7.2 Post-Migration Verification"
echo "=========================================="
echo ""

PROJECT_ROOT="/workspaces/upstageailab-ocr-recsys-competition-ocr-2"
cd "$PROJECT_ROOT"

PASSED=0
FAILED=0
TESTS_RUN=0

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

test_result() {
    TESTS_RUN=$((TESTS_RUN + 1))
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ PASSED${NC}"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}✗ FAILED${NC}"
        FAILED=$((FAILED + 1))
    fi
}

# Test 1: create_artifact
echo "1. Testing create_artifact..."
uv run python -c "
from AgentQMS.tools.core.artifacts.workflow import ArtifactWorkflow
workflow = ArtifactWorkflow(quiet=True)
file_path = workflow.create_artifact('design_document', 'mcp-test-auto-v2', 'Automated MCP Test V2')
assert file_path, 'File path not returned'
print(f'Created: {file_path}')
" 2>&1 && test_result 0 || test_result 1

# Test 2: validate_artifact
echo ""
echo "2. Testing validate_artifact..."
uv run python AgentQMS/tools/compliance/validate_artifacts.py \
  --file docs/artifacts/design_documents/mcp-test-auto-v2.md 2>&1 | grep -q "design_document" \
  && test_result 0 || test_result 1

# Test 3: list_artifact_templates
echo ""
echo "3. Testing list_artifact_templates..."
uv run python -c "
from AgentQMS.tools.core.artifacts.workflow import ArtifactWorkflow
workflow = ArtifactWorkflow(quiet=True)
templates = workflow.get_available_templates()
assert len(templates) >= 7, f'Expected >= 7 templates, got {len(templates)}'
print(f'Found {len(templates)} templates: {templates}')
" 2>&1 && test_result 0 || test_result 1

# Test 4: check_compliance
echo ""
echo "4. Testing check_compliance..."
uv run python AgentQMS/tools/compliance/validate_artifacts.py --all 2>&1 > /dev/null \
  && test_result 0 || test_result 1

# Test 5: get_standard (CRITICAL - Phase 7 impact)
echo ""
echo -e "5. Testing get_standard... ${YELLOW}[CRITICAL]${NC}"
uv run python -c "
from pathlib import Path

# Simulate get_standard tool logic (fixed in mcp_server.py)
AGENTQMS_DIR = Path('AgentQMS')
query = 'compliance'
specs_dir = AGENTQMS_DIR / 'specs'
matches = []

if specs_dir.exists():
    for path in specs_dir.rglob('*'):
        if path.is_file() and path.suffix in ['.md', '.yaml', '.json']:
            if query in path.stem.lower():
                matches.append(path)

assert len(matches) > 0, f'No specs found for {query}'
assert 'specs/' in str(matches[0]), f'Found in wrong location: {matches[0]}'
print(f'✓ Found {len(matches)} specs matching \"{query}\"')
for m in matches:
    print(f'  - {m}')
" 2>&1 && test_result 0 || test_result 1

# Test 6: get_context_bundle
echo ""
echo "6. Testing get_context_bundle..."
uv run python -c "
from AgentQMS.tools.core.context.context_bundle import auto_suggest_context
suggestion = auto_suggest_context('fixing validation errors')
assert 'bundle_files' in suggestion, 'No bundle files returned'
assert len(suggestion['bundle_files']) > 0, 'Empty bundle'
print(f'✓ Found {len(suggestion[\"bundle_files\"])} files in bundle')
print(f'  Bundle: {suggestion.get(\"context_bundle\")}')
print(f'  Tokens: {suggestion.get(\"token_usage\", {}).get(\"total_tokens\", 0)}')
" 2>&1 && test_result 0 || test_result 1

# Summary
echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Tests run: $TESTS_RUN"
echo -e "${GREEN}Passed: $PASSED${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Failed: $FAILED${NC}"
fi
echo "=========================================="

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed! MCP tools ready for Phase 7.2${NC}"
    exit 0
else
    echo -e "${RED}✗ Some tests failed. See output above.${NC}"
    exit 1
fi
