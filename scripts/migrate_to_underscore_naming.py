#!/usr/bin/env python3
"""
Migrate artifact filenames from hyphen to underscore convention.

This script:
1. Renames files: type-name.md -> type_name.md
2. Updates all references in markdown files
3. Updates INDEX.md files
4. Generates a rollback script
"""

import re
import sys
from pathlib import Path

# Artifact types to migrate
ARTIFACT_TYPES = ["assessment", "design", "audit", "research"]


def find_files_to_rename(artifacts_dir: Path) -> list[tuple[Path, Path]]:
    """Find all files that need renaming."""
    renames = []

    for artifact_type in ARTIFACT_TYPES:
        type_dir = artifacts_dir / f"{artifact_type}s" if artifact_type != "design" else artifacts_dir / "design_documents"

        if not type_dir.exists():
            continue

        # Find files with pattern: YYYY-MM-DD_HHMM_type-name.md
        for file_path in type_dir.glob("*.md"):
            if file_path.name == "INDEX.md":
                continue

            # Check if it matches the old pattern (type-name)
            pattern = rf"(\d{{4}}-\d{{2}}-\d{{2}}_\d{{4}})_{artifact_type}-(.+\.md)"
            match = re.match(pattern, file_path.name)

            if match:
                timestamp = match.group(1)
                description = match.group(2)
                new_name = f"{timestamp}_{artifact_type}_{description}"
                new_path = file_path.parent / new_name
                renames.append((file_path, new_path))

    return renames


def find_references(content: str, old_filename: str) -> list[tuple[int, str]]:
    """Find all references to a filename in content."""
    references = []
    lines = content.split('\n')

    for i, line in enumerate(lines, 1):
        if old_filename in line:
            references.append((i, line))

    return references


def update_references(docs_dir: Path, renames: list[tuple[Path, Path]]) -> dict[Path, list[str]]:
    """Update all references to renamed files."""
    updates = {}

    # Create mapping of old basename to new basename
    rename_map = {old.name: new.name for old, new in renames}

    # Also handle .ko.md versions
    for old_name, new_name in list(rename_map.items()):
        if not old_name.endswith('.ko.md'):
            ko_old = old_name.replace('.md', '.ko.md')
            ko_new = new_name.replace('.md', '.ko.md')
            rename_map[ko_old] = ko_new

    # Search all markdown files
    for md_file in docs_dir.rglob("*.md"):
        try:
            content = md_file.read_text(encoding='utf-8')
            updated_content = content
            changes = []

            for old_name, new_name in rename_map.items():
                if old_name in content:
                    updated_content = updated_content.replace(old_name, new_name)
                    changes.append(f"  {old_name} -> {new_name}")

            if changes:
                updates[md_file] = (content, updated_content, changes)
        except Exception as e:
            print(f"Warning: Could not read {md_file}: {e}", file=sys.stderr)

    return updates


def generate_rollback_script(renames: list[tuple[Path, Path]], output_path: Path):
    """Generate a rollback script to undo the migration."""
    script = """#!/bin/bash
# Rollback script - reverts underscore naming back to hyphen naming
set -e

echo "Rolling back filename migration..."

"""
    for old_path, new_path in renames:
        script += f'mv "{new_path}" "{old_path}"\n'

    script += '\necho "Rollback complete!"'

    output_path.write_text(script)
    output_path.chmod(0o755)


def main():
    project_root = Path(__file__).resolve().parent.parent
    artifacts_dir = project_root / "docs" / "artifacts"
    docs_dir = project_root / "docs"

    print("🔍 Scanning for files to migrate...")
    renames = find_files_to_rename(artifacts_dir)

    if not renames:
        print("✓ No files need migration!")
        return 0

    print(f"\n📝 Found {len(renames)} files to rename:")
    for old, new in renames:
        print(f"  {old.name}")
        print(f"  -> {new.name}")
        print()

    # Find and update references
    print("🔍 Searching for references in markdown files...")
    updates = update_references(docs_dir, renames)

    if updates:
        print(f"\n📝 Found references in {len(updates)} files:")
        for file_path, (_, _, changes) in updates.items():
            print(f"\n  {file_path.relative_to(project_root)}:")
            for change in changes:
                print(f"    {change}")
    else:
        print("✓ No references found (no updates needed)")

    # Confirm before proceeding
    print("\n" + "="*70)
    response = input("Proceed with migration? (yes/no): ").strip().lower()

    if response != "yes":
        print("Migration cancelled.")
        return 1

    # Generate rollback script first
    rollback_path = project_root / "scripts" / "rollback_naming_migration.sh"
    print(f"\n📝 Generating rollback script: {rollback_path.relative_to(project_root)}")
    generate_rollback_script(renames, rollback_path)

    # Perform renames
    print("\n🔄 Renaming files...")
    for old_path, new_path in renames:
        print(f"  {old_path.name} -> {new_path.name}")
        old_path.rename(new_path)

        # Also rename .ko.md version if it exists
        old_ko = old_path.with_suffix('.ko.md')
        if old_ko.exists():
            new_ko = new_path.with_suffix('.ko.md')
            print(f"  {old_ko.name} -> {new_ko.name}")
            old_ko.rename(new_ko)

    # Update references
    if updates:
        print("\n🔄 Updating references...")
        for file_path, (original, updated, changes) in updates.items():
            print(f"  {file_path.relative_to(project_root)}")
            file_path.write_text(updated, encoding='utf-8')

    print("\n✅ Migration complete!")
    print(f"   Renamed: {len(renames)} files")
    print(f"   Updated: {len(updates)} files with references")
    print(f"\n💡 Rollback script: {rollback_path.relative_to(project_root)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
