#!/usr/bin/env python3
"""
Automated Migration Path Fixer

Scans AgentQMS/ for references to deleted paths and applies automated fixes.
Phase 7+ Migration Cleanup Tool

Usage:
    uv run python scripts/utils/fix_migration_paths.py
    uv run python scripts/utils/fix_migration_paths.py --dry-run
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict


# Path replacement map: old_pattern -> new_path
REPLACEMENTS: Dict[str, str] = {
    r'AgentQMS/standards/registry\.yaml': 'AgentQMS/.agentqms/registry.yaml',
    r'AgentQMS/standards/schemas': 'AgentQMS/.agentqms/schemas',
    r'AgentQMS/standards/tier1-sst/artifact-types\.yaml': 'AgentQMS/specs/tier1-contracts/compliance.spec.md',
    r'AgentQMS/standards/tier1-sst/naming-conventions\.yaml': 'AgentQMS/specs/tier1-contracts/compliance.spec.md',
    r'AgentQMS/standards/tier1-sst/artifact-rules\.yaml': 'AgentQMS/.agentqms/standards_db.json',
}


def get_project_root() -> Path:
    """Find project root by looking for .git directory."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find project root (.git directory)")


def fix_file(file_path: Path, dry_run: bool = False) -> tuple[bool, list[str]]:
    """
    Apply automated fixes to a file.

    Args:
        file_path: Path to the file to fix
        dry_run: If True, don't write changes

    Returns:
        Tuple of (was_modified, list_of_changes)
    """
    try:
        content = file_path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError):
        return False, []

    original_content = content
    changes = []

    for pattern, replacement in REPLACEMENTS.items():
        matches = list(re.finditer(pattern, content))
        if matches:
            content = re.sub(pattern, replacement, content)
            changes.append(f"  {len(matches)}x: {pattern} → {replacement}")

    if content != original_content:
        if not dry_run:
            file_path.write_text(content, encoding='utf-8')
        return True, changes

    return False, []


def scan_and_fix(project_root: Path, dry_run: bool = False) -> tuple[int, int]:
    """
    Scan AgentQMS directory and fix migration paths.

    Args:
        project_root: Project root directory
        dry_run: If True, don't write changes

    Returns:
        Tuple of (files_scanned, files_modified)
    """
    agentqms_dir = project_root / "AgentQMS"
    if not agentqms_dir.exists():
        print(f"ERROR: AgentQMS directory not found: {agentqms_dir}", file=sys.stderr)
        return 0, 0

    files_scanned = 0
    files_modified = 0

    print("=" * 60)
    print("Migration Path Fixer")
    print("=" * 60)
    if dry_run:
        print("**DRY RUN** - No files will be modified")
    print()

    for py_file in sorted(agentqms_dir.rglob("*.py")):
        # Skip __pycache__ and hidden files
        if "__pycache__" in str(py_file) or py_file.name.startswith("."):
            continue

        files_scanned += 1
        was_modified, changes = fix_file(py_file, dry_run)

        if was_modified:
            files_modified += 1
            relative_path = py_file.relative_to(project_root)
            status = "Would fix" if dry_run else "✓ Fixed"
            print(f"{status}: {relative_path}")
            for change in changes:
                print(change)
            print()

    print("=" * 60)
    print(f"Scanned: {files_scanned} files")
    print(f"{'Would modify' if dry_run else 'Modified'}: {files_modified} files")
    print("=" * 60)

    return files_scanned, files_modified


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fix migration-related path references in AgentQMS/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )

    args = parser.parse_args()

    try:
        project_root = get_project_root()
        files_scanned, files_modified = scan_and_fix(project_root, dry_run=args.dry_run)

        # Summary
        if files_modified > 0:
            if args.dry_run:
                print("\n💡 Run without --dry-run to apply changes")
            else:
                print(f"\n✅ Successfully fixed {files_modified} files")
                print("💡 Recommended: Run tests to verify changes")
                print("   uv run pytest AgentQMS/tests/")
            return 0
        else:
            print("\n✅ No migration path issues found")
            return 0

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
