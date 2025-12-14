#!/usr/bin/env python3
"""
Archive Artifacts Script

Moves artifacts marked with status="archived" from docs/artifacts/ to docs/archive/artifacts/
while preserving the directory structure.

Usage:
    python archive_artifacts.py [--dry-run] [--all]
"""

import argparse
import shutil
import sys
from pathlib import Path

import yaml

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "docs" / "artifacts"
ARCHIVE_DIR = PROJECT_ROOT / "docs" / "archive" / "artifacts"


def extract_frontmatter(file_path: Path) -> dict | None:
    """Extract YAML frontmatter from a markdown file."""
    try:
        content = file_path.read_text(encoding="utf-8")
        if not content.startswith("---"):
            return None

        # Find the closing ---
        parts = content.split("---", 2)
        if len(parts) < 3:
            return None

        frontmatter_str = parts[1].strip()
        return yaml.safe_load(frontmatter_str)
    except Exception as e:
        print(f"⚠️  Error reading {file_path.name}: {e}")
        return None


def find_archived_artifacts() -> list[Path]:
    """Find all artifacts with status='archived'."""
    archived = []

    for md_file in ARTIFACTS_DIR.rglob("*.md"):
        # Skip INDEX.md files
        if md_file.name == "INDEX.md":
            continue

        frontmatter = extract_frontmatter(md_file)
        if frontmatter and frontmatter.get("status") == "archived":
            archived.append(md_file)

    return archived


def archive_file(file_path: Path, dry_run: bool = False) -> bool:
    """Move a file from artifacts/ to archive/artifacts/ preserving structure."""
    try:
        # Calculate relative path from artifacts dir
        relative_path = file_path.relative_to(ARTIFACTS_DIR)

        # Destination in archive
        dest_path = ARCHIVE_DIR / relative_path

        if dry_run:
            print("  [DRY-RUN] Would move:")
            print(f"    From: {file_path.relative_to(PROJECT_ROOT)}")
            print(f"    To:   {dest_path.relative_to(PROJECT_ROOT)}")
            return True

        # Create destination directory
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Move the file
        shutil.move(str(file_path), str(dest_path))
        print(f"  ✅ Archived: {file_path.name}")
        print(f"     To: {dest_path.relative_to(PROJECT_ROOT)}")
        return True

    except Exception as e:
        print(f"  ❌ Failed to archive {file_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Archive artifacts marked with status='archived'"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be archived without moving files",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Archive all files marked as archived (default behavior)",
    )

    args = parser.parse_args()

    print("🗄️  Artifact Archival System")
    print("=" * 60)

    # Find archived artifacts
    print(f"\n🔍 Scanning for archived artifacts in {ARTIFACTS_DIR.relative_to(PROJECT_ROOT)}...")
    archived_files = find_archived_artifacts()

    if not archived_files:
        print("\n✨ No artifacts marked for archival (status='archived')")
        return 0

    print(f"\n📋 Found {len(archived_files)} artifact(s) marked for archival:")
    for f in archived_files:
        relative = f.relative_to(ARTIFACTS_DIR)
        print(f"   • {relative}")

    if args.dry_run:
        print("\n🔍 DRY RUN MODE - No files will be moved\n")
    else:
        print(f"\n⚠️  Files will be moved to: {ARCHIVE_DIR.relative_to(PROJECT_ROOT)}")
        response = input("\nContinue? [y/N]: ").strip().lower()
        if response != "y":
            print("❌ Cancelled")
            return 1
        print()

    # Archive files
    print("📦 Archiving files...\n")
    success_count = 0
    fail_count = 0

    for file_path in archived_files:
        if archive_file(file_path, dry_run=args.dry_run):
            success_count += 1
        else:
            fail_count += 1
        print()

    # Summary
    print("=" * 60)
    print("📊 Summary:")
    print(f"   ✅ Successfully archived: {success_count}")
    if fail_count > 0:
        print(f"   ❌ Failed: {fail_count}")

    if args.dry_run:
        print("\n💡 Run without --dry-run to actually move the files")
    else:
        print(f"\n✨ Files archived to: {ARCHIVE_DIR.relative_to(PROJECT_ROOT)}")
        print("   The archive/ directory is excluded from AgentQMS validation")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
