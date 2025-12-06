#!/usr/bin/env python3
"""
Legacy Artifact Migrator for AgentQMS

Finds artifacts that don't follow current naming conventions and
helps migrate them to compliant format.

Expected convention: YYYY-MM-DD_HHMM_[type]_name.md
Example: 2025-12-06_0112_implementation_plan_agentqms-framework-enhancement.md

Usage:
    python legacy_migrator.py find --limit 10
    python legacy_migrator.py migrate --file path/to/legacy.md --autofix
    python legacy_migrator.py migrate --all --dry-run
    python legacy_migrator.py migrate --directory docs/artifacts/ --limit 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from AgentQMS.agent_tools.utils.runtime import ensure_project_root_on_sys_path

ensure_project_root_on_sys_path()

from AgentQMS.agent_tools.utils.paths import get_project_root


class LegacyArtifactMigrator:
    """Find and migrate legacy artifacts to current naming convention."""

    CONVENTION = r"^\d{4}-\d{2}-\d{2}_\d{4}_\w+_.+\.md$"

    def __init__(self, project_root: Path | None = None):
        """Initialize the migrator.

        Args:
            project_root: Project root path (auto-detected if not provided)
        """
        if project_root is None:
            project_root = get_project_root()

        self.project_root = Path(project_root)
        self.artifacts_dir = self.project_root / "docs" / "artifacts"
        self._migration_state = self._load_migration_state()

    def _load_migration_state(self) -> dict[str, Any]:
        """Load migration state from .agentqms/state/migration_state.json"""
        state_dir = self.project_root / ".agentqms" / "state"
        state_file = state_dir / "migration_state.json"

        if state_file.exists():
            try:
                return json.loads(state_file.read_text(encoding="utf-8"))
            except Exception:
                return {}

        return {}

    def _save_migration_state(self, state: dict[str, Any]) -> None:
        """Save migration state."""
        state_dir = self.project_root / ".agentqms" / "state"
        state_dir.mkdir(parents=True, exist_ok=True)

        state_file = state_dir / "migration_state.json"
        state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def is_compliant(self, filename: str) -> bool:
        """Check if filename follows current convention.

        Args:
            filename: Filename to check

        Returns:
            True if compliant
        """
        return bool(re.match(self.CONVENTION, filename))

    def find_legacy_artifacts(
        self,
        directory: Path | None = None,
        limit: int | None = None,
    ) -> list[Path]:
        """Find artifacts that don't follow current convention.

        Args:
            directory: Directory to search (defaults to artifacts/)
            limit: Maximum number to return

        Returns:
            List of legacy artifact paths
        """
        if directory is None:
            directory = self.artifacts_dir

        if not directory.exists():
            return []

        legacy = []
        for artifact_file in directory.rglob("*.md"):
            if not self.is_compliant(artifact_file.name):
                legacy.append(artifact_file)
                if limit and len(legacy) >= limit:
                    break

        return legacy

    def extract_metadata(self, filepath: Path) -> dict[str, Any]:
        """Extract frontmatter from artifact file.

        Args:
            filepath: Path to artifact file

        Returns:
            Dictionary with extracted metadata
        """
        content = filepath.read_text(encoding="utf-8")

        # Extract YAML frontmatter
        if content.startswith("---"):
            match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
            if match:
                try:
                    metadata = yaml.safe_load(match.group(1)) or {}
                    return metadata
                except Exception:
                    pass

        return {}

    def generate_new_filename(
        self,
        filepath: Path,
        artifact_type: str | None = None,
    ) -> str:
        """Generate compliant filename for legacy artifact.

        Args:
            filepath: Path to legacy artifact
            artifact_type: Artifact type (extracted from frontmatter if not provided)

        Returns:
            New compliant filename
        """
        metadata = self.extract_metadata(filepath)

        # Use provided type or extract from metadata
        if artifact_type is None:
            artifact_type = metadata.get("type", "artifact")

        # Use date from metadata or current time
        date_str = metadata.get("date", "")
        if date_str:
            # Try to parse the date string
            try:
                date_obj = datetime.fromisoformat(date_str.split()[0])
                timestamp = date_obj.strftime("%Y-%m-%d_%H%M")
            except Exception:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")

        # Generate name from filename (remove extensions, spaces, etc.)
        base_name = filepath.stem
        # Remove common prefixes and suffixes
        base_name = re.sub(r"^\d{4}-\d{2}-\d{2}[_\-]", "", base_name)
        base_name = re.sub(r"[^\w\-]+", "_", base_name.lower())
        base_name = re.sub(r"_+", "_", base_name).strip("_")

        return f"{timestamp}_{artifact_type}_{base_name}.md"

    def migrate_artifact(
        self,
        filepath: Path,
        autofix: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Migrate a single artifact to compliant naming.

        Args:
            filepath: Path to legacy artifact
            autofix: If True, actually rename the file
            dry_run: If True, don't make changes

        Returns:
            Dictionary with migration result
        """
        if self.is_compliant(filepath.name):
            return {
                "status": "skip",
                "reason": "Already compliant",
                "file": str(filepath),
            }

        # Extract metadata to get artifact type
        metadata = self.extract_metadata(filepath)
        artifact_type = metadata.get("type", "artifact")

        # Generate new filename
        new_filename = self.generate_new_filename(filepath, artifact_type)
        new_path = filepath.parent / new_filename

        # Check if target already exists
        if new_path.exists():
            return {
                "status": "error",
                "reason": f"Target file already exists: {new_filename}",
                "file": str(filepath),
            }

        result = {
            "status": "success",
            "file": str(filepath),
            "new_name": new_filename,
            "new_path": str(new_path),
        }

        if not dry_run and autofix:
            try:
                # Use git mv if available for better history tracking
                import subprocess
                subprocess.run(
                    ["git", "mv", str(filepath), str(new_path)],
                    cwd=self.project_root,
                    check=True,
                    capture_output=True,
                )
                result["migrated"] = True
            except Exception:
                # Fallback to regular file move
                try:
                    filepath.rename(new_path)
                    result["migrated"] = True
                except Exception as move_error:
                    result["status"] = "error"
                    result["reason"] = str(move_error)
                    result["migrated"] = False

        return result

    def migrate_batch(
        self,
        limit: int | None = None,
        directory: Path | None = None,
        autofix: bool = False,
        dry_run: bool = False,
        force_large_batch: bool = False,
    ) -> list[dict[str, Any]]:
        """Migrate multiple artifacts.

        Args:
            limit: Maximum number to migrate
            directory: Directory to search
            autofix: Actually rename files
            dry_run: Don't make changes
            force_large_batch: Allow operations >30 files

        Returns:
            List of migration results
        """
        legacy_files = self.find_legacy_artifacts(directory=directory, limit=limit)

        # DISASTER PREVENTION SAFEGUARD (added 2025-12-07 after catastrophic migration incident)
        # Reject bulk operations >30 files without explicit override flag
        if len(legacy_files) > 30 and autofix and not dry_run and not force_large_batch:
            print(f"\n🚨 SAFETY CHECK FAILED: Attempting to migrate {len(legacy_files)} files")
            print("   This exceeds the 30-file safety threshold.")
            print("   ")
            print("   ⚠️  LESSON FROM 2025-12-06 CATASTROPHIC MIGRATION:")
            print("      Bulk operation destroyed 103 artifact filenames by overwriting")
            print("      all dates to present (2025-12-06_0000), losing all historical")
            print("      context. This safeguard prevents similar disasters.")
            print("   ")
            print("   Options to proceed:")
            print("     1. Use --limit 30 to process in safer batches")
            print("     2. Use --dry-run to preview changes first")
            print("     3. Add explicit override if you're absolutely certain")
            print("   ")
            print("   Aborting migration.")
            return []

        results = []

        for filepath in legacy_files:
            result = self.migrate_artifact(
                filepath,
                autofix=autofix,
                dry_run=dry_run,
            )
            results.append(result)

        return results

    def compute_content_hash(self, filepath: Path) -> str:
        """Compute SHA-256 hash of file content (excluding frontmatter).

        Args:
            filepath: Path to file

        Returns:
            Hex digest of content hash
        """
        import hashlib

        content = filepath.read_text(encoding="utf-8")

        # Strip frontmatter for consistent hashing
        if content.startswith("---"):
            match = re.match(r"^---\n.*?\n---\n(.*)$", content, re.DOTALL)
            if match:
                content = match.group(1)

        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def detect_manual_moves(
        self,
        directory: Path | None = None,
    ) -> list[dict[str, Any]]:
        """Detect artifacts that were manually moved/renamed without git tracking.

        Uses content hashing to identify files that may have been renamed
        but still have broken references in the link graph.

        Args:
            directory: Directory to scan (defaults to artifacts/)

        Returns:
            List of potential manual moves with old/new paths
        """
        if directory is None:
            directory = self.artifacts_dir

        # Build content hash index
        hash_index: dict[str, list[Path]] = {}

        print("🔍 Building content hash index...")
        for artifact_file in directory.rglob("*.md"):
            if artifact_file.name in ["INDEX.md", "README.md", "MASTER_INDEX.md"]:
                continue

            try:
                content_hash = self.compute_content_hash(artifact_file)
                if content_hash not in hash_index:
                    hash_index[content_hash] = []
                hash_index[content_hash].append(artifact_file)
            except Exception as e:
                print(f"  ⚠️  Error hashing {artifact_file.name}: {e}")

        # Find duplicates (potential moves)
        duplicates = [paths for paths in hash_index.values() if len(paths) > 1]

        if not duplicates:
            print("✅ No duplicate content detected")
            return []

        print(f"🔍 Found {len(duplicates)} potential manual move(s)")

        manual_moves = []
        for paths in duplicates:
            # Sort by modification time (oldest first)
            paths_sorted = sorted(paths, key=lambda p: p.stat().st_mtime)

            manual_moves.append({
                "original": str(paths_sorted[0]),
                "duplicates": [str(p) for p in paths_sorted[1:]],
                "content_hash": list(hash_index.keys())[
                    list(hash_index.values()).index(paths)
                ],
            })

        return manual_moves

    def repair_manual_moves(
        self,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Detect and repair manually moved artifacts.

        This will:
        1. Detect files with duplicate content (manual moves)
        2. Remove older duplicates
        3. Update link references using check_links logic

        Args:
            dry_run: Preview changes without applying

        Returns:
            Repair summary
        """
        manual_moves = self.detect_manual_moves()

        if not manual_moves:
            return {
                "status": "success",
                "manual_moves_found": 0,
                "duplicates_removed": 0,
                "links_updated": 0,
            }

        print()
        print("=" * 60)
        print("MANUAL MOVE DETECTION RESULTS")
        print("=" * 60)

        for i, move in enumerate(manual_moves, 1):
            print(f"\n{i}. Duplicate Content Detected:")
            print(f"   Original:   {Path(move['original']).name}")
            print("   Duplicates:")
            for dup in move["duplicates"]:
                print(f"     - {Path(dup).name}")

        duplicates_removed = 0
        if not dry_run:
            print("\n🔧 Removing duplicate files...")
            for move in manual_moves:
                for dup_path_str in move["duplicates"]:
                    dup_path = Path(dup_path_str)
                    try:
                        dup_path.unlink()
                        duplicates_removed += 1
                        print(f"  ✅ Removed: {dup_path.name}")
                    except Exception as e:
                        print(f"  ❌ Failed to remove {dup_path.name}: {e}")

        result = {
            "status": "success",
            "manual_moves_found": len(manual_moves),
            "duplicates_removed": duplicates_removed,
            "links_updated": 0,  # Would integrate with check_links.py
            "dry_run": dry_run,
        }

        if dry_run:
            print("\nℹ️  DRY RUN mode - no files were removed")

        return result


def main() -> int:
    """Command-line interface for artifact migration."""
    parser = argparse.ArgumentParser(
        description="Migrate legacy artifacts to current naming convention",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find legacy artifacts
  %(prog)s find --limit 10

  # Migrate single artifact (dry-run)
  %(prog)s migrate --file docs/artifacts/old_artifact.md

  # Migrate with autofix (actually rename)
  %(prog)s migrate --file docs/artifacts/old_artifact.md --autofix

  # Batch migrate
  %(prog)s migrate --directory docs/artifacts/ --limit 5 --dry-run

  # Migrate all (carefully!)
  %(prog)s migrate --all --autofix
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Find command
    find_parser = subparsers.add_parser("find", help="Find legacy artifacts")
    find_parser.add_argument("--limit", "-l", type=int, help="Limit results")
    find_parser.add_argument(
        "--directory",
        "-d",
        type=Path,
        help="Directory to search",
    )

    # Migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Migrate artifacts")
    migrate_group = migrate_parser.add_mutually_exclusive_group(required=True)
    migrate_group.add_argument(
        "--file",
        "-f",
        type=Path,
        help="Path to single artifact to migrate",
    )
    migrate_group.add_argument(
        "--all",
        action="store_true",
        help="Migrate all legacy artifacts",
    )
    migrate_group.add_argument(
        "--directory",
        "-d",
        type=Path,
        help="Directory to search",
    )
    migrate_parser.add_argument("--limit", "-l", type=int, help="Limit batch size")
    migrate_parser.add_argument(
        "--autofix",
        action="store_true",
        help="Actually rename files (default is dry-run)",
    )
    migrate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without applying",
    )
    migrate_parser.add_argument(
        "--force-large-batch",
        action="store_true",
        help="Allow batch operations >30 files (use with extreme caution)",
    )

    # Detect manual moves command
    detect_parser = subparsers.add_parser(
        "detect-moves",
        help="Detect manually moved/renamed artifacts",
    )
    detect_parser.add_argument(
        "--directory",
        "-d",
        type=Path,
        help="Directory to scan",
    )
    detect_parser.add_argument(
        "--repair",
        action="store_true",
        help="Remove duplicate files (keep oldest)",
    )
    detect_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without applying",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    try:
        migrator = LegacyArtifactMigrator()

        if args.command == "find":
            legacy = migrator.find_legacy_artifacts(
                directory=args.directory,
                limit=args.limit,
            )

            if not legacy:
                print("✅ No legacy artifacts found!")
                return 0

            print(f"Found {len(legacy)} legacy artifact(s):")
            print("-" * 60)
            for i, path in enumerate(legacy, 1):
                print(f"{i}. {path.relative_to(migrator.project_root)}")

        elif args.command == "migrate":
            if args.file:
                if not args.file.exists():
                    print(f"❌ File not found: {args.file}", file=sys.stderr)
                    return 1

                result = migrator.migrate_artifact(
                    args.file,
                    autofix=args.autofix,
                    dry_run=args.dry_run or not args.autofix,
                )

                print(f"Status: {result['status']}")
                if result["status"] == "success":
                    print(f"Old name: {args.file.name}")
                    print(f"New name: {result['new_name']}")
                    if result.get("migrated"):
                        print("✅ Migration completed!")
                    elif args.autofix:
                        print("⚠️  Migration skipped (dry-run)")
                else:
                    print(f"Error: {result.get('reason')}")

            else:
                directory = args.directory if args.directory else None
                results = migrator.migrate_batch(
                    limit=args.limit,
                    directory=directory,
                    autofix=args.autofix,
                    dry_run=args.dry_run or not args.autofix,
                    force_large_batch=args.force_large_batch,
                )

                success = [r for r in results if r["status"] == "success"]
                errors = [r for r in results if r["status"] == "error"]

                print(f"Processed {len(results)} artifact(s)")
                print(f"  ✅ Success: {len(success)}")
                print(f"  ⚠️  Skipped: {len([r for r in results if r['status'] == 'skip'])}")
                print(f"  ❌ Errors: {len(errors)}")

                if success:
                    print("\nMigrations:")
                    for result in success:
                        print(f"  {Path(result['file']).name} → {result['new_name']}")

                if errors:
                    print("\nErrors:")
                    for result in errors:
                        print(f"  {Path(result['file']).name}: {result['reason']}")

        elif args.command == "detect-moves":
            if args.repair:
                result = migrator.repair_manual_moves(dry_run=args.dry_run)
                print()
                print("=" * 60)
                print("REPAIR SUMMARY")
                print("=" * 60)
                print(f"Manual moves found:    {result['manual_moves_found']}")
                print(f"Duplicates removed:    {result['duplicates_removed']}")
                print(f"Links updated:         {result['links_updated']}")

                if result['dry_run']:
                    print("\nℹ️  DRY RUN mode - no changes were applied")
            else:
                manual_moves = migrator.detect_manual_moves(directory=args.directory)

                if not manual_moves:
                    print("✅ No manually moved artifacts detected")
                else:
                    print("\n💡 Run with --repair to remove duplicates")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
