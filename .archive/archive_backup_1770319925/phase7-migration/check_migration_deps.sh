#!/bin/bash
# Migration Dependency Checker
#
# Comprehensive scan for migration-related issues across AgentQMS/
#
# Usage:
#   bash scripts/utils/check_migration_deps.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "Migration Impact Scan"
echo "============================================================"
echo "Scanning: AgentQMS/"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Check for deleted directory references
echo "1. Checking for deleted path references..."
echo "   ├─ 'standards/schemas' references:"
SCHEMA_COUNT=$(grep -r "standards/schemas" AgentQMS --include="*.py" 2>/dev/null | wc -l || echo "0")
echo "   │  Found: $SCHEMA_COUNT"

echo "   ├─ 'standards/tier1-sst' references:"
TIER1_COUNT=$(grep -r "standards/tier1-sst" AgentQMS --include="*.py" 2>/dev/null | wc -l || echo "0")
echo "   │  Found: $TIER1_COUNT"

echo "   └─ 'standards/registry.yaml' references:"
REGISTRY_COUNT=$(grep -r "standards/registry.yaml" AgentQMS --include="*.py" 2>/dev/null | wc -l || echo "0")
echo "      Found: $REGISTRY_COUNT"

TOTAL_PATH_REFS=$((SCHEMA_COUNT + TIER1_COUNT + REGISTRY_COUNT))

# 2. Check for syntax errors
echo ""
echo "2. Running Python syntax checks..."
SYNTAX_ERRORS=0
SYNTAX_OUTPUT=$(mktemp)
find AgentQMS -name "*.py" -type f -exec python3 -m py_compile {} \; 2>&1 | tee "$SYNTAX_OUTPUT" | grep -c "SyntaxError" || true
SYNTAX_ERRORS=$?
if [ -s "$SYNTAX_OUTPUT" ]; then
    echo -e "   ${RED}✗ Syntax errors found${NC}"
    head -10 "$SYNTAX_OUTPUT"
else
    echo -e "   ${GREEN}✓ No syntax errors${NC}"
fi
rm -f "$SYNTAX_OUTPUT"

# 3. Check for import issues (lightweight check)
echo ""
echo "3. Checking for potential import issues..."
IMPORT_ISSUES=0
for py_file in $(find AgentQMS -name "*.py" -type f | grep -v "__pycache__" | head -20); do
    if ! python3 -c "import py_compile; py_compile.compile('$py_file', doraise=True)" 2>/dev/null; then
        IMPORT_ISSUES=$((IMPORT_ISSUES + 1))
    fi
done
if [ "$IMPORT_ISSUES" -eq 0 ]; then
    echo -e "   ${GREEN}✓ No import issues detected (sample check)${NC}"
else
    echo -e "   ${YELLOW}⚠ Found $IMPORT_ISSUES potential import issues${NC}"
fi

# 4. Check plugin system
echo ""
echo "4. Checking plugin system..."
if uv run python -m AgentQMS.tools.core.plugins --list > /dev/null 2>&1; then
    echo -e "   ${GREEN}✓ Plugin system operational${NC}"
else
    echo -e "   ${RED}✗ Plugin system has errors${NC}"
fi

# 5. Check validation system
echo ""
echo "5. Checking validation system..."
if uv run python AgentQMS/tools/compliance/validate_artifacts.py --help > /dev/null 2>&1; then
    echo -e "   ${GREEN}✓ Validation system operational${NC}"
else
    echo -e "   ${RED}✗ Validation system has errors${NC}"
fi

# Summary
echo ""
echo "============================================================"
echo "Summary"
echo "============================================================"
echo "Deleted path references found: $TOTAL_PATH_REFS"

if [ "$TOTAL_PATH_REFS" -gt 0 ]; then
    echo -e "${YELLOW}⚠ Action required: Run fix_migration_paths.py${NC}"
    echo "   uv run python scripts/utils/fix_migration_paths.py --dry-run"
fi

if [ "$SYNTAX_ERRORS" -eq 0 ] && [ "$IMPORT_ISSUES" -eq 0 ]; then
    echo -e "\n${GREEN}✅ No critical issues detected${NC}"
    exit 0
else
    echo -e "\n${YELLOW}⚠ Issues detected - review output above${NC}"
    exit 1
fi
