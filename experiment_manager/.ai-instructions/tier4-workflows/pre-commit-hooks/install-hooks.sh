#!/usr/bin/env bash
# EDS v1.0 Pre-Commit Hook Installer
# Sets up all pre-commit hooks for experiment-tracker

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
GIT_HOOKS_DIR="$REPO_ROOT/.git/hooks"

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  EDS v1.0 Pre-Commit Hook Installation${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check if .git directory exists
if [ ! -d "$REPO_ROOT/.git" ]; then
    echo -e "${RED}❌ ERROR: Not a git repository${NC}"
    echo -e "   Expected: $REPO_ROOT/.git"
    echo -e "   Cannot install hooks"
    exit 1
fi

# Create hooks directory if it doesn't exist
mkdir -p "$GIT_HOOKS_DIR"

# Create pre-commit orchestrator
PRE_COMMIT_HOOK="$GIT_HOOKS_DIR/pre-commit"

echo "📝 Creating pre-commit orchestrator..."

cat > "$PRE_COMMIT_HOOK" << 'EOF'
#!/usr/bin/env bash
# EDS v1.0 Pre-Commit Orchestrator
# Runs all validation hooks before commit

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
HOOKS_DIR="$REPO_ROOT/experiment-tracker/.ai-instructions/tier4-workflows/pre-commit-hooks"

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  EDS v1.0 Pre-Commit Validation${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check if staged files include experiment-tracker artifacts
staged_experiment_files=$(git diff --cached --name-only --diff-filter=ACM | grep "^experiment-tracker/experiments/" || true)

if [ -z "$staged_experiment_files" ]; then
    echo "ℹ️  No experiment artifacts staged - skipping EDS validation"
    exit 0
fi

echo "🔍 Running EDS v1.0 validation checks..."
echo ""

# Run all validation hooks in sequence
hooks=(
    "naming-validation.sh"
    "metadata-validation.sh"
    "eds-compliance.sh"
)

for hook in "${hooks[@]}"; do
    hook_path="$HOOKS_DIR/$hook"

    if [ ! -f "$hook_path" ]; then
        echo -e "${RED}❌ ERROR: Hook not found: $hook${NC}"
        exit 1
    fi

    if ! bash "$hook_path"; then
        echo ""
        echo -e "${RED}❌ Validation failed: $hook${NC}"
        echo -e "${RED}Commit blocked by EDS v1.0 enforcement${NC}"
        exit 1
    fi

    echo ""
done

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ All EDS v1.0 validation checks passed${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

exit 0
EOF

chmod +x "$PRE_COMMIT_HOOK"

echo -e "${GREEN}✅${NC} Created pre-commit orchestrator: $PRE_COMMIT_HOOK"
echo ""

# Make individual hooks executable
echo "🔧 Setting execute permissions on validation hooks..."

hooks=(
    "naming-validation.sh"
    "metadata-validation.sh"
    "eds-compliance.sh"
)

for hook in "${hooks[@]}"; do
    hook_path="$SCRIPT_DIR/$hook"

    if [ -f "$hook_path" ]; then
        chmod +x "$hook_path"
        echo -e "${GREEN}✅${NC} $hook"
    else
        echo -e "${RED}❌${NC} $hook (not found)"
    fi
done

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ EDS v1.0 Pre-Commit Hooks Installed Successfully${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Installed hooks:"
echo "  • naming-validation.sh (blocks ALL-CAPS, validates pattern)"
echo "  • metadata-validation.sh (requires .metadata/ structure)"
echo "  • eds-compliance.sh (validates YAML frontmatter)"
echo ""
echo "Next steps:"
echo "  1. Test hooks: git add <file> && git commit"
echo "  2. Validate existing artifacts: python3 compliance-checker.py <directory>"
echo "  3. Generate new artifacts using CLI tools (coming soon)"
echo ""
