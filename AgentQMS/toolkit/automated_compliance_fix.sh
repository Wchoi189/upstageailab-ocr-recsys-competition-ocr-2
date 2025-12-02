#!/bin/bash
# Automated Compliance Fix Script
# Master script that runs all automated fix operations

# Default values
ARTIFACTS_ROOT="docs/artifacts"
MAX_FILES=""
DRY_RUN=false

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-files)
            MAX_FILES="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --directory)
            ARTIFACTS_ROOT="$2"
            shift 2
            ;;
        *)
            ARTIFACTS_ROOT="$1"
            shift
            ;;
    esac
done

echo "🔧 Starting automated compliance fixes..."

# Set script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if artifacts directory exists
if [ ! -d "$ARTIFACTS_ROOT" ]; then
    print_error "Artifacts directory not found: $ARTIFACTS_ROOT"
    exit 1
fi

print_status "Processing artifacts in: $ARTIFACTS_ROOT"
if [ -n "$MAX_FILES" ]; then
    print_status "Maximum files to process: $MAX_FILES"
fi
if [ "$DRY_RUN" = true ]; then
    print_warning "DRY RUN MODE - No changes will be made"
fi

# Create backup directory
BACKUP_DIR="backups/automated_fixes_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
print_status "Created backup directory: $BACKUP_DIR"

# Build common arguments
COMMON_ARGS="--directory $ARTIFACTS_ROOT"
if [ -n "$MAX_FILES" ]; then
    COMMON_ARGS="$COMMON_ARGS --limit $MAX_FILES"
fi
if [ "$DRY_RUN" = true ]; then
    COMMON_ARGS="$COMMON_ARGS --dry-run"
fi

# Step 1: Fix naming convention violations
print_status "Step 1: Fixing naming convention violations..."
if python "$SCRIPT_DIR/maintenance/fix_naming_conventions.py" $COMMON_ARGS --auto-fix; then
    print_success "Naming convention fixes completed"
else
    print_warning "Some naming convention fixes may have failed"
fi

# Step 2: Add missing frontmatter
print_status "Step 2: Adding missing frontmatter..."
if python "$SCRIPT_DIR/maintenance/add_frontmatter.py" $COMMON_ARGS --batch-process; then
    print_success "Frontmatter generation completed"
else
    print_warning "Some frontmatter generation may have failed"
fi

# Step 3: Fix invalid categories and types
print_status "Step 3: Fixing invalid category/type values..."
if python "$SCRIPT_DIR/maintenance/fix_categories.py" $COMMON_ARGS --auto-correct; then
    print_success "Category/type fixes completed"
else
    print_warning "Some category/type fixes may have failed"
fi

# Step 4: Reorganize misplaced files
print_status "Step 4: Reorganizing misplaced files..."
if python "$SCRIPT_DIR/maintenance/reorganize_files.py" $COMMON_ARGS --move-to-correct-dirs; then
    print_success "File reorganization completed"
else
    print_warning "Some file reorganization may have failed"
fi

# Step 5: Validate all artifacts
print_status "Step 5: Validating all artifacts..."
if python "$SCRIPT_DIR/compliance/validate_artifacts.py" --all --artifacts-root "$ARTIFACTS_ROOT"; then
    print_success "All artifacts are now compliant!"
else
    print_warning "Some artifacts may still have compliance issues"
fi

# Generate final report
print_status "Generating final compliance report..."
REPORT_FILE="$BACKUP_DIR/compliance_fix_report_$(date +%Y%m%d_%H%M%S).txt"

{
    echo "Automated Compliance Fix Report"
    echo "Generated: $(date)"
    echo "Artifacts Root: $ARTIFACTS_ROOT"
    echo "Backup Directory: $BACKUP_DIR"
    echo "========================================"
    echo ""

    echo "Naming Convention Fixes:"
    python "$SCRIPT_DIR/maintenance/fix_naming_conventions.py" --directory "$ARTIFACTS_ROOT" --dry-run 2>/dev/null || echo "Naming fix script not available"
    echo ""

    echo "Frontmatter Status:"
    python "$SCRIPT_DIR/maintenance/add_frontmatter.py" --directory "$ARTIFACTS_ROOT" --dry-run 2>/dev/null || echo "Frontmatter script not available"
    echo ""

    echo "Category/Type Validation:"
    python "$SCRIPT_DIR/maintenance/fix_categories.py" --directory "$ARTIFACTS_ROOT" --validate-only 2>/dev/null || echo "Category fix script not available"
    echo ""

    echo "File Organization Status:"
    python "$SCRIPT_DIR/maintenance/reorganize_files.py" --directory "$ARTIFACTS_ROOT" --validate-only 2>/dev/null || echo "Reorganization script not available"
    echo ""

    echo "Final Validation Results:"
    python "$SCRIPT_DIR/compliance/validate_artifacts.py" --all --artifacts-root "$ARTIFACTS_ROOT" 2>/dev/null || echo "Validation script not available"

} > "$REPORT_FILE"

print_success "Final report saved to: $REPORT_FILE"

# Summary
echo ""
echo "========================================"
echo "🎉 Automated Compliance Fix Complete!"
echo "========================================"
echo "Backup directory: $BACKUP_DIR"
echo "Report file: $REPORT_FILE"
echo ""
echo "Next steps:"
echo "1. Review the generated report"
echo "2. Test the fixed artifacts"
echo "3. Commit changes if satisfied"
echo "4. Update compliance monitoring"
echo ""

# Check if we should run in dry-run mode for verification
if [ "${2:-}" = "--verify" ]; then
    print_status "Running verification checks..."
    python "$SCRIPT_DIR/compliance/validate_artifacts.py" --all --artifacts-root "$ARTIFACTS_ROOT"
fi
