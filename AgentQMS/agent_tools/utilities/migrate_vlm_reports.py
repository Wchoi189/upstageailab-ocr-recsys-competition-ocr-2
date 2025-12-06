#!/usr/bin/env python3
"""
VLM Reports Migration Script

Migrates VLM reports from BUG prefix to vlm_report_ prefix.

This script:
1. Renames files from BUG-XXX pattern to vlm_report_ pattern
2. Updates frontmatter type from bug_report to vlm_report
3. Adds default timestamp (1200) if not present in filename

Usage:
    python migrate_vlm_reports.py [--dry-run]
"""

import argparse
import re
import sys
from pathlib import Path

# Project root detection
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VLM_DIR = PROJECT_ROOT / "docs" / "artifacts" / "vlm_reports"


def migrate_vlm_report(file_path: Path, dry_run: bool = False) -> bool:
    """
    Migrate a single VLM report file.

    Args:
        file_path: Path to VLM report file
        dry_run: If True, only preview changes

    Returns:
        True if migration successful, False otherwise
    """
    # Pattern: 2025-12-03_BUG-001_inference-overlay-misalignment_69.md
    # Target:  2025-12-03_1200_vlm_report_inference-overlay-misalignment_69.md

    match = re.match(r'(\d{4}-\d{2}-\d{2})_BUG-(\d+)_(.+)\.md', file_path.name)
    if not match:
        print(f"⏭️  SKIP: {file_path.name} (doesn't match BUG pattern)")
        return False

    date, bug_num, rest = match.groups()
    # Use 1200 as default time if none exists
    new_name = f"{date}_1200_vlm_report_{rest}.md"
    new_path = file_path.parent / new_name

    if new_path.exists():
        print(f"⚠️  SKIP: {file_path.name} (target already exists: {new_name})")
        return False

    print(f"📝 Rename: {file_path.name}")
    print(f"       → {new_name}")

    if not dry_run:
        try:
            # Read content and update frontmatter type
            content = file_path.read_text(encoding="utf-8")

            # Update type in frontmatter
            content = re.sub(
                r'type:\s*["\']?bug_report["\']?',
                'type: "vlm_report"',
                content,
                flags=re.IGNORECASE
            )

            # Also update category if it's "troubleshooting"
            content = re.sub(
                r'category:\s*["\']?troubleshooting["\']?',
                'category: "evaluation"',
                content,
                flags=re.IGNORECASE
            )

            # Rename file
            file_path.rename(new_path)

            # Write updated content
            new_path.write_text(content, encoding="utf-8")

            print("   ✅ Migrated successfully")
            return True
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    else:
        print("   🔍 (dry-run, no changes made)")
        return True


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Migrate VLM reports from BUG prefix to vlm_report_ prefix"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying files"
    )
    args = parser.parse_args()

    if not VLM_DIR.exists():
        print(f"❌ VLM reports directory not found: {VLM_DIR}")
        return 1

    # Find all markdown files in vlm_reports directory
    files = list(VLM_DIR.glob("*.md"))

    if not files:
        print(f"ℹ️  No files found in {VLM_DIR}")
        return 0

    print(f"🎯 Found {len(files)} file(s) in vlm_reports directory")

    if args.dry_run:
        print("🔍 DRY RUN mode (no changes will be made)\n")
    else:
        print("⚠️  This will rename files and modify frontmatter!")
        response = input("Continue? [y/N]: ").strip().lower()
        if response != 'y':
            print("❌ Aborted by user")
            return 1
        print()

    success_count = 0
    skip_count = 0
    error_count = 0

    for file_path in files:
        result = migrate_vlm_report(file_path, dry_run=args.dry_run)
        if result:
            success_count += 1
        else:
            # Check if it was skipped (not matching pattern)
            if not re.match(r'.*_BUG-\d+_.*\.md', file_path.name):
                skip_count += 1
            else:
                error_count += 1
        print()

    print("📊 Migration Summary:")
    print(f"   Migrated: {success_count}")
    print(f"   Skipped: {skip_count}")
    print(f"   Errors: {error_count}")

    if args.dry_run:
        print("\n💡 Run without --dry-run to apply changes")

    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
