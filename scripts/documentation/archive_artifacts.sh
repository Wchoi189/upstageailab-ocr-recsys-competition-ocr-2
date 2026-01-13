#!/bin/bash
# Archive artifacts older than 30 days
set -e

CUTOFF_DATE=$(date -d "30 days ago" +%Y-%m-%d)
ARCHIVE_ROOT="archive/artifacts"
CURRENT_ARCHIVE_DIR="$ARCHIVE_ROOT/$(date +%Y%m)"

echo "🧹 Pruning artifacts older than $CUTOFF_DATE..."

# Ensure archive directory exists
mkdir -p "$CURRENT_ARCHIVE_DIR"

# Find old artifacts
find docs/artifacts -type f -name "*.md" | while read file; do
    # Extract date from filename (YYYY-MM-DD)
    # Assumes format like: 2026-01-12_... or similar.
    # If filename doesn't start with date, we skip or need smarter logic.
    # Grep for first occurrence of date pattern
    file_date=$(basename "$file" | grep -oP '\d{4}-\d{2}-\d{2}' | head -1)

    if [ -n "$file_date" ] && [[ "$file_date" < "$CUTOFF_DATE" ]]; then
        echo "📦 Archiving $file ($file_date)..."

        # Create parent dir structure in archive
        # e.g. docs/artifacts/audits/foo.md -> archive/artifacts/202601/audits/foo.md
        REL_DIR=$(dirname "${file#docs/artifacts/}")
        mkdir -p "$CURRENT_ARCHIVE_DIR/$REL_DIR"

        mv "$file" "$CURRENT_ARCHIVE_DIR/$REL_DIR/"
    fi
done

# Compress archive if not empty
if [ -n "$(ls -A $CURRENT_ARCHIVE_DIR 2>/dev/null)" ]; then
    echo "🗜️ Compressing archive..."
    TAR_NAME="artifacts_$(date +%Y%m).tar.gz"
    tar -czf "$ARCHIVE_ROOT/$TAR_NAME" -C "$ARCHIVE_ROOT" "$(date +%Y%m)"
    rm -rf "$CURRENT_ARCHIVE_DIR"
    echo "✅ Archived to $ARCHIVE_ROOT/$TAR_NAME"
else
    echo "✨ No artifacts to archive."
    rm -rf "$CURRENT_ARCHIVE_DIR"
fi
