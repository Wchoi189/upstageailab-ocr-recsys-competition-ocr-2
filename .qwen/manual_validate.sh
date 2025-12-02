#!/bin/bash
# Manual Document Validation Script for AgentQMS
# This script provides an alternative to Qwen for document validation

set -e

PROJECT_ROOT="/workspaces/upstageailab-ocr-recsys-competition-ocr-2"
ARTIFACTS_DIR="$PROJECT_ROOT/docs/artifacts"

echo "🔍 Manual AgentQMS Document Validation"
echo "======================================"
echo "Artifacts Directory: $ARTIFACTS_DIR"
echo ""

# Function to validate filename format
validate_filename() {
    local file="$1"
    local filename=$(basename "$file")

    # Check if it matches the pattern: YYYY-MM-DD_HHMM_{ARTIFACT_TYPE}_descriptive-name.md
    if [[ ! $filename =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{4}_(.+)\.md$ ]]; then
        echo "❌ INVALID: $filename"
        echo "   Expected format: YYYY-MM-DD_HHMM_{ARTIFACT_TYPE}_descriptive-name.md"
        return 1
    fi

    echo "✅ VALID: $filename"
    return 0
}

# Function to check frontmatter
check_frontmatter() {
    local file="$1"

    # Check if file starts with ---
    if ! head -1 "$file" | grep -q "^---$"; then
        echo "❌ MISSING FRONTMATTER: $file"
        return 1
    fi

    # Check for required fields (basic check)
    if ! grep -q "^type:" "$file" || ! grep -q "^title:" "$file"; then
        echo "❌ INCOMPLETE FRONTMATTER: $file"
        return 1
    fi

    echo "✅ FRONTMATTER OK: $(basename "$file")"
    return 0
}

# Function to check directory structure
check_directory() {
    local file="$1"
    local relative_path="${file#$ARTIFACTS_DIR/}"

    # This is a basic check - could be enhanced based on artifact_categories
    echo "📁 STRUCTURE: $relative_path"
}

echo "🔍 Scanning artifacts directory..."
echo ""

# Find all .md files in artifacts directory
find "$ARTIFACTS_DIR" -name "*.md" -type f | while read -r file; do
    echo "📄 Checking: $(basename "$file")"

    # Validate filename
    if ! validate_filename "$file"; then
        continue
    fi

    # Check frontmatter
    if ! check_frontmatter "$file"; then
        continue
    fi

    # Check directory structure
    check_directory "$file"

    echo ""
done

echo "✅ Manual validation complete!"
echo ""
echo "💡 To fix issues:"
echo "   1. Rename files to: YYYY-MM-DD_HHMM_{ARTIFACT_TYPE}_descriptive-name.md"
echo "   2. Add proper YAML frontmatter at the top"
echo "   3. Move files to correct subdirectories"
echo ""
echo "📖 See AgentQMS/knowledge/agent/system.md for detailed guidelines"
