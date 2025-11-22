#!/usr/bin/env python3
"""
Artifact Frontmatter Fix Script

This script fixes common frontmatter validation issues in artifact files.
It maps invalid categories, statuses, and types to valid values.

Usage:
    python fix_artifacts.py --all
    python fix_artifacts.py --file path/to/artifact.md
"""

import argparse
import sys
from pathlib import Path

# Valid values from validate_artifacts.py
VALID_CATEGORIES = [
    "development",
    "architecture",
    "evaluation",
    "compliance",
    "reference",
    "planning",
    "research",
    "troubleshooting",
]

VALID_STATUSES = ["active", "draft", "completed", "archived", "deprecated"]

VALID_TYPES = [
    "implementation_plan",
    "assessment",
    "design",
    "research",
    "template",
    "bug_report",
    "session_note",
    "completion_summary",
]

# Mapping of invalid values to valid ones
CATEGORY_MAPPING = {
    "debugging": "troubleshooting",
    "completion_report": "evaluation",
    "bug": "troubleshooting",
    "semantic_search": "development",
    "performance_optimization": "development",
    "functionality": "troubleshooting",
    "recovery": "troubleshooting",
    "architecture-audit": "architecture",
    "architecture_audit": "architecture",
    "root-cause": "troubleshooting",
    "root-cause-analysis": "troubleshooting",
    "completion": "evaluation",
    "fix": "troubleshooting",
    "summary": "evaluation",
    "code-review": "evaluation",
    "feature_catalog": "reference",
    "performance": "evaluation",
    "bug-fix": "troubleshooting",
    "refactoring": "development",
    "migration": "development",
    "remediation": "troubleshooting",
    "bug-fixes": "troubleshooting",
    "session-handover": "planning",
    "dependency_verification": "compliance",
    "progress_report": "planning",
    "development_environment": "development",
    "quality_assurance": "evaluation",
}

STATUS_MAPPING = {
    "open": "active",
    "complete": "completed",
    "resolved": "completed",
    "in-progress": "active",
    "in_progress": "active",
    "ready": "draft",
    "critical": "active",
    "final": "completed",
    "analysis_complete": "completed",
}

TYPE_MAPPING = {
    "solution": "implementation_plan",
    "assessment_results": "assessment",
    "completion_report": "completion_summary",
    "session_handover": "session_note",
    "living_blueprint": "implementation_plan",
    "test_report": "assessment",
    "archive_index": "reference",
}


def fix_frontmatter(content: str) -> tuple[str, bool]:
    """Fix frontmatter in content. Returns (fixed_content, was_changed)."""
    if not content.startswith("---"):
        return content, False

    # Extract frontmatter
    frontmatter_end = content.find("---", 3)
    if frontmatter_end == -1:
        return content, False

    frontmatter_content = content[3:frontmatter_end]
    rest_content = content[frontmatter_end + 3 :]

    # Parse frontmatter
    lines = frontmatter_content.split("\n")
    fixed_lines = []
    changed = False

    for line in lines:
        original_line = line
        line = line.strip()

        # Skip comments and empty lines
        if not line or line.startswith("#"):
            fixed_lines.append(original_line)
            continue

        # Check for key: value pattern
        if ":" in line and not line.startswith("#"):
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip("\"'")

            # Fix category
            if key == "category" and value in CATEGORY_MAPPING:
                value = CATEGORY_MAPPING[value]
                changed = True
                fixed_lines.append(f'{key}: "{value}"')

            # Fix status
            elif key == "status" and value in STATUS_MAPPING:
                value = STATUS_MAPPING[value]
                changed = True
                fixed_lines.append(f'{key}: "{value}"')

            # Fix type
            elif key == "type" and value in TYPE_MAPPING:
                value = TYPE_MAPPING[value]
                changed = True
                fixed_lines.append(f'{key}: "{value}"')

            else:
                fixed_lines.append(original_line)
        else:
            fixed_lines.append(original_line)

    if changed:
        fixed_frontmatter = "\n".join(fixed_lines)
        return f"---{fixed_frontmatter}---{rest_content}", True

    return content, False


def fix_file(file_path: Path, dry_run: bool = False) -> bool:
    """Fix a single file. Returns True if changes were made."""
    try:
        content = file_path.read_text(encoding="utf-8")
        fixed_content, was_changed = fix_frontmatter(content)

        if was_changed:
            if not dry_run:
                file_path.write_text(fixed_content, encoding="utf-8")
            print(f"{'[DRY RUN] ' if dry_run else ''}Fixed: {file_path}")
            return True
        return False
    except Exception as e:
        print(f"Error fixing {file_path}: {e}", file=sys.stderr)
        return False


def fix_directory(directory: Path, dry_run: bool = False) -> int:
    """Fix all markdown files in a directory. Returns count of fixed files."""
    fixed_count = 0
    for file_path in directory.rglob("*.md"):
        if file_path.is_file() and file_path.name != "INDEX.md":
            if fix_file(file_path, dry_run):
                fixed_count += 1
    return fixed_count


def main():
    parser = argparse.ArgumentParser(description="Fix artifact frontmatter issues")
    parser.add_argument("--file", help="Fix a specific file")
    parser.add_argument("--directory", help="Fix all files in a directory")
    parser.add_argument("--all", action="store_true", help="Fix all artifacts")
    parser.add_argument(
        "--artifacts-root",
        default="artifacts",
        help="Root directory for artifacts",
    )
    parser.add_argument("--dry-run", action="store_true", help="Don't write changes")

    args = parser.parse_args()

    if args.file:
        file_path = Path(args.file)
        if file_path.exists():
            fix_file(file_path, args.dry_run)
        else:
            print(f"File not found: {file_path}", file=sys.stderr)
            sys.exit(1)
    elif args.directory:
        dir_path = Path(args.directory)
        if dir_path.exists():
            fixed_count = fix_directory(dir_path, args.dry_run)
            print(f"\nFixed {fixed_count} file(s)")
        else:
            print(f"Directory not found: {dir_path}", file=sys.stderr)
            sys.exit(1)
    elif args.all:
        artifacts_root = Path(args.artifacts_root)
        if artifacts_root.exists():
            fixed_count = fix_directory(artifacts_root, args.dry_run)
            print(f"\nFixed {fixed_count} file(s)")
        else:
            print(f"Artifacts root not found: {artifacts_root}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
