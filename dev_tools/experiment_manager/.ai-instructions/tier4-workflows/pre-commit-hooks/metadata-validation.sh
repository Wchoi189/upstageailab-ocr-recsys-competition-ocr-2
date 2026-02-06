#!/usr/bin/env bash
# EDS v1.0 Metadata Directory Validation Hook
# Ensures .metadata/ directory exists and contains artifacts

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get staged files in experiment-tracker/experiments/
staged_files=$(git diff --cached --name-only --diff-filter=ACM | grep "^experiment-tracker/experiments/" || true)

if [ -z "$staged_files" ]; then
    exit 0  # No experiment files to validate
fi

# Extract unique experiment IDs from staged files
experiment_ids=$(echo "$staged_files" | grep -oP 'experiment-tracker/experiments/\K[^/]+' | sort -u)

violations_found=false

echo "🔍 EDS v1.0: Validating .metadata/ directories..."

for experiment_id in $experiment_ids; do
    experiment_path="$REPO_ROOT/experiment-tracker/experiments/$experiment_id"
    metadata_path="$experiment_path/.metadata"

    # Check if .metadata/ directory exists
    if [ ! -d "$metadata_path" ]; then
        echo -e "${RED}❌ CRITICAL: Missing .metadata/ directory${NC}"
        echo -e "   Experiment: ${YELLOW}$experiment_id${NC}"
        echo -e "   Required: $metadata_path"
        echo -e "   Violation: ALL experiments MUST have .metadata/ directory"
        violations_found=true
        continue
    fi

    # Check if .metadata/ has required subdirectories
    required_subdirs=("assessments" "reports" "guides" "plans")
    missing_subdirs=()

    for subdir in "${required_subdirs[@]}"; do
        if [ ! -d "$metadata_path/$subdir" ]; then
            missing_subdirs+=("$subdir")
        fi
    done

    if [ ${#missing_subdirs[@]} -gt 0 ]; then
        echo -e "${RED}❌ CRITICAL: Missing .metadata/ subdirectories${NC}"
        echo -e "   Experiment: ${YELLOW}$experiment_id${NC}"
        echo -e "   Missing: ${missing_subdirs[*]}"
        echo -e "   Required structure:"
        echo -e "     .metadata/"
        echo -e "       assessments/"
        echo -e "       reports/"
        echo -e "       guides/"
        echo -e "       plans/"
        violations_found=true
        continue
    fi

    # Check if .metadata/ contains at least one artifact
    artifact_count=$(find "$metadata_path" -name "*.md" -type f | wc -l)

    if [ "$artifact_count" -eq 0 ]; then
        echo -e "${YELLOW}⚠️  WARNING: Empty .metadata/ directory${NC}"
        echo -e "   Experiment: ${YELLOW}$experiment_id${NC}"
        echo -e "   .metadata/ exists but contains no artifacts"
        echo -e "   Consider creating at least one artifact using CLI tools"
        # Not blocking - this is a warning
    fi

    echo -e "${GREEN}✅${NC} $experiment_id (.metadata/ structure valid)"
done

if [ "$violations_found" = true ]; then
    echo ""
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}❌ COMMIT BLOCKED: .metadata/ violations detected${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "EDS v1.0 requires .metadata/ directory with proper structure."
    echo ""
    echo "Fix options:"
    echo "  1. Create .metadata/ structure manually:"
    echo "     cd experiment-tracker/experiments/<experiment_id>"
    echo "     mkdir -p .metadata/{assessments,reports,guides,plans}"
    echo ""
    echo "  2. Use CLI to initialize experiment:"
    echo "     eds init-experiment --id <experiment_id>"
    echo ""
    echo "No bypass mechanism available - violations MUST be fixed."
    echo ""
    exit 1
fi

echo -e "${GREEN}✅ All .metadata/ directories comply with EDS v1.0${NC}"
exit 0
