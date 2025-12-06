#!/usr/bin/env python3
"""
Safe autofix pipeline: apply validator suggestions with limits.
Consumes validation JSON, performs git mv, updates indexes and links.

Phase 4 enhancements:
- Link rewriting after file moves/renames
- Duplicate conflict resolution policy
- Post-fix re-validation loop
- Support for --update-links flag
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def run_git_command(cmd: list[str], dry_run: bool = False) -> bool:
    """Execute git command, optionally in dry-run mode."""
    if dry_run:
        print(f"  [dry-run] git {' '.join(cmd)}")
        return True
    try:
        result = subprocess.run(["git"] + cmd, check=True, capture_output=True, text=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"  ❌ git command failed: {e.stderr}")
        return False


def find_links_to_file(search_dir: Path, old_path: Path, project_root: Path) -> list[tuple[Path, int, str, str]]:
    """Find all markdown links that reference a specific file.

    Returns list of (file_path, line_num, link_text, link_url) tuples.
    """
    links_found = []
    old_name = old_path.name

    # Get relative path from project root for matching
    try:
        old_rel = str(old_path.relative_to(project_root))
    except ValueError:
        old_rel = str(old_path)

    for md_file in search_dir.rglob("*.md"):
        if ".git" in str(md_file):
            continue

        try:
            content = md_file.read_text(encoding="utf-8")
            lines = content.split("\n")

            for i, line in enumerate(lines, 1):
                # Check for markdown links
                for match in re.finditer(r'\[([^\]]+)\]\(([^)]+)\)', line):
                    text, url = match.groups()

                    # Check if this link references the old file
                    if old_name in url or old_rel in url:
                        links_found.append((md_file, i, text, url))
        except Exception:
            continue

    return links_found


def update_links_in_file(file_path: Path, old_url: str, new_url: str, dry_run: bool = False) -> bool:
    """Update markdown link occurrences of old_url to new_url in a file.

    Uses regex to match proper markdown link syntax [text](url) to avoid
    unintended replacements in other contexts.
    """
    try:
        content = file_path.read_text(encoding="utf-8")

        # Escape special regex characters in old_url
        escaped_old = re.escape(old_url)

        # Match markdown links containing the old URL
        # Pattern: [any text](old_url) or [any text](old_url#anchor)
        pattern = r'(\[[^\]]+\]\()' + escaped_old + r'([#)])'

        if not re.search(pattern, content):
            return False

        # Replace only within markdown link context
        new_content = re.sub(pattern, r'\g<1>' + new_url.replace('\\', '\\\\') + r'\2', content)

        if dry_run:
            print(f"    [dry-run] Would update links in {file_path.name}")
            return True

        file_path.write_text(new_content, encoding="utf-8")
        return True
    except Exception as e:
        print(f"    ❌ Error updating links in {file_path}: {e}")
        return False


def calculate_relative_path(from_file: Path, to_file: Path) -> str:
    """Calculate the relative path from one file to another."""
    from_dir = from_file.parent
    try:
        rel_path = to_file.relative_to(from_dir)
        return str(rel_path)
    except ValueError:
        # Need to use ../ notation
        # Find common ancestor
        from_parts = from_dir.parts
        to_parts = to_file.parts

        common_length = 0
        for i in range(min(len(from_parts), len(to_parts))):
            if from_parts[i] == to_parts[i]:
                common_length += 1
            else:
                break

        # Calculate ups and downs
        ups = len(from_parts) - common_length
        downs = to_parts[common_length:]

        path_parts = [".."] * ups + list(downs)
        return "/".join(path_parts)


def rewrite_links_after_move(
    old_path: Path,
    new_path: Path,
    project_root: Path,
    dry_run: bool = False
) -> int:
    """Rewrite all links pointing to old_path to point to new_path.

    Returns number of files updated.
    """
    docs_dir = project_root / "docs"
    if not docs_dir.exists():
        return 0

    links = find_links_to_file(docs_dir, old_path, project_root)

    if not links:
        return 0

    updated_files: set[Path] = set()

    for link_file, line_num, text, url in links:
        # Calculate new relative URL
        new_rel_url = calculate_relative_path(link_file, new_path)

        # Preserve any anchors
        if "#" in url:
            anchor = url.split("#", 1)[1]
            new_rel_url = f"{new_rel_url}#{anchor}"

        if update_links_in_file(link_file, url, new_rel_url, dry_run):
            updated_files.add(link_file)
            if not dry_run:
                print(f"    📝 Updated link in {link_file.name}: {url} → {new_rel_url}")

    return len(updated_files)


def check_for_duplicates(target_path: Path, source_path: Path) -> bool:
    """Check if target already exists and handle conflicts.

    Returns True if it's safe to proceed, False if there's a conflict.
    """
    if not target_path.exists():
        return True

    # Target exists - check if it's the same file or a different one
    if target_path == source_path:
        return True

    # Different file exists at target - this is a conflict
    print(f"  ⚠️  Conflict: Target already exists: {target_path}")
    print(f"      Source: {source_path}")

    # Policy: prefer canonical target, skip move if conflict
    return False


def extract_suggestions_from_violations(violations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse validation violations and extract actionable rename/move suggestions."""
    suggestions = []

    for item in violations:
        if item.get("valid"):
            continue

        file_path = Path(item["file"])
        errors = item.get("errors", [])

        for error in errors:
            # Parse naming violations for renames
            if "Missing or invalid timestamp" in error or "Missing valid artifact type" in error:
                # Suggest rename based on pattern
                suggestion = {
                    "type": "rename",
                    "source": str(file_path),
                    "target": None,  # To be computed based on rules
                    "reason": error
                }
                suggestions.append(suggestion)

            # Parse directory violations for moves (use new error format)
            elif "Directory:" in error or "[E004]" in error:
                # Try both old and new error formats
                match = re.search(r"should be in '([^']+)' directory", error)
                if not match:
                    match = re.search(r"Expected directory: ([^\s|]+)", error)

                if match:
                    target_dir = match.group(1).rstrip("/")
                    suggestion = {
                        "type": "move",
                        "source": str(file_path),
                        "target_dir": target_dir,
                        "reason": error
                    }
                    suggestions.append(suggestion)

    return suggestions


def apply_fixes(
    suggestions: list[dict[str, Any]],
    limit: int,
    dry_run: bool,
    project_root: Path,
    update_links: bool = False
) -> tuple[int, list[tuple[Path, Path]]]:
    """Apply fixes with limit.

    Returns (applied_count, list of (old_path, new_path) for moves).
    """
    applied = 0
    moves: list[tuple[Path, Path]] = []

    for i, suggestion in enumerate(suggestions[:limit]):
        if suggestion["type"] == "move":
            source = Path(suggestion["source"])
            target_dir = project_root / "docs" / "artifacts" / suggestion["target_dir"]
            target = target_dir / source.name

            # Check for duplicates/conflicts
            if not check_for_duplicates(target, source):
                print(f"\n{i+1}. Skip (conflict): {source.name}")
                continue

            print(f"\n{i+1}. Move: {source.name}")
            print(f"   From: {source.parent}")
            print(f"   To:   {target_dir}")

            if not dry_run:
                target_dir.mkdir(parents=True, exist_ok=True)

            if run_git_command(["mv", str(source), str(target)], dry_run):
                applied += 1
                moves.append((source, target))

                # Update links if requested
                if update_links:
                    print("   🔗 Updating links...")
                    updated = rewrite_links_after_move(source, target, project_root, dry_run)
                    if updated > 0:
                        print(f"   ✅ Updated {updated} files with new links")

        elif suggestion["type"] == "rename":
            # For now, just report (actual rename logic requires pattern detection)
            print(f"\n{i+1}. Rename needed: {suggestion['source']}")
            print(f"   Reason: {suggestion['reason']}")

    return applied, moves


def run_reindex(project_root: Path, dry_run: bool = False) -> bool:
    """Run the reindex command to regenerate indexes."""
    if dry_run:
        print("  [dry-run] Would regenerate indexes")
        return True

    try:
        from AgentQMS.agent_tools.documentation.reindex_artifacts import main as reindex_main
        result = reindex_main()
        return result == 0
    except Exception as e:
        print(f"  ⚠️  Reindex failed: {e}")
        return False


def run_validation() -> list[dict[str, Any]]:
    """Run validation and return results."""
    from AgentQMS.agent_tools.compliance.validate_artifacts import ArtifactValidator
    validator = ArtifactValidator()
    return validator.validate_all()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Safe autofix pipeline for artifacts")
    parser.add_argument("--limit", type=int, default=10, help="Max fixes to apply")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, no changes")
    parser.add_argument("--commit", action="store_true", help="Auto-commit changes")
    parser.add_argument("--validation-json", help="Path to validation JSON output")
    parser.add_argument("--update-links", action="store_true",
                        help="Rewrite links in other files after moves")
    parser.add_argument("--revalidate", action="store_true",
                        help="Run validation again after fixes to verify")
    parser.add_argument("--force-large-batch", action="store_true",
                        help="Allow batch operations >30 files (use with caution)")

    args = parser.parse_args()

    from AgentQMS.agent_tools.utils.paths import get_project_root
    project_root = get_project_root()

    # Run validation to get current state
    if args.validation_json and Path(args.validation_json).exists():
        with open(args.validation_json) as f:
            violations = json.load(f)
    else:
        print("🔍 Running validation...")
        violations = run_validation()

    # Extract actionable suggestions
    suggestions = extract_suggestions_from_violations(violations)

    if not suggestions:
        print("✅ No fixes needed")
        return 0

    # DISASTER PREVENTION SAFEGUARD (added 2025-12-07 after catastrophic migration incident)
    # Reject bulk operations >30 files without explicit override flag
    actual_limit = min(args.limit, len(suggestions))
    if actual_limit > 30 and not args.force_large_batch and not args.dry_run:
        print(f"\n🚨 SAFETY CHECK FAILED: Attempting to modify {actual_limit} files")
        print("   This exceeds the 30-file safety threshold.")
        print("   ")
        print("   ⚠️  LESSON FROM 2025-12-06 CATASTROPHIC MIGRATION:")
        print("      Bulk operations >30 files destroyed 103 artifact filenames by")
        print("      overwriting all dates to present (2025-12-06_0000), losing all")
        print("      historical context. This safeguard prevents similar disasters.")
        print("   ")
        print("   Options to proceed:")
        print("     1. Use --limit 30 to process in safer batches")
        print("     2. Use --dry-run to preview changes first")
        print("     3. Use --force-large-batch if you're absolutely certain")
        print("   ")
        print("   Recommended: make fix ARGS='--limit 30'")
        return 1

    print(f"\n📋 Found {len(suggestions)} potential fixes")
    print(f"   Applying up to {args.limit} fixes {'(DRY RUN)' if args.dry_run else ''}")
    if args.update_links:
        print("   🔗 Link rewriting enabled")

    # Apply fixes
    applied, moves = apply_fixes(
        suggestions, args.limit, args.dry_run, project_root, args.update_links
    )

    print(f"\n✨ Applied {applied} fixes")

    # Regenerate indexes after moves
    if applied > 0 and moves:
        print("\n🔄 Regenerating indexes...")
        run_reindex(project_root, args.dry_run)

    # Post-fix re-validation
    if args.revalidate and applied > 0 and not args.dry_run:
        print("\n🔍 Re-validating after fixes...")
        new_violations = run_validation()
        invalid_count = sum(1 for v in new_violations if not v.get("valid"))
        if invalid_count == 0:
            print("✅ All artifacts now valid!")
        else:
            print(f"⚠️  {invalid_count} artifacts still have issues")

    if not args.dry_run and args.commit and applied > 0:
        print("\n📝 Committing changes...")
        run_git_command(["add", "-A"], False)
        run_git_command(["commit", "-m", f"AgentQMS: autofix {applied} artifacts"], False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
