#!/usr/bin/env python3
"""
Artifact Audit Tool: Comprehensive audit and repair of AgentQMS artifacts.

This tool provides intelligent auditing and fixing of artifacts using existing
AgentQMS tools instead of duplicating logic:
- FrontmatterGenerator (AgentQMS/toolkit/maintenance/add_frontmatter.py)
- ArtifactValidator (AgentQMS/agent_tools/compliance/validate_artifacts.py)

Features:
- Audit artifacts for compliance issues
- Fix incomplete or broken frontmatter
- Smart date inference (git history → filesystem metadata → present date)
- Batch operations with progress tracking
- Directory exclusion (archive/, deprecated/ by default)
- Dry-run mode for previewing changes
- Validation-only mode for reporting without fixes
- Pre-flight safety checks with automatic git stash

Usage:
    python artifact_audit.py --batch N         # Audit & fix batch N
    python artifact_audit.py --files f1 f2 ... # Audit & fix specific files
    python artifact_audit.py --all              # Audit & fix all artifacts
    python artifact_audit.py --dry-run          # Preview without modifying
    python artifact_audit.py --report           # Report violations only
    python artifact_audit.py --include-excluded # Include archive/deprecated dirs
    python artifact_audit.py --no-stash        # Skip automatic git stash
"""

import argparse
import subprocess
import sys
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

# Add AgentQMS to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from AgentQMS.agent_tools.compliance.validate_artifacts import ArtifactValidator
    from AgentQMS.agent_tools.maintenance.add_frontmatter import FrontmatterGenerator
    from AgentQMS.agent_tools.utils.config import load_config
except ImportError as e:
    print(f"❌ Error importing AgentQMS tools: {e}")
    print("   Make sure you're running from project root")
    sys.exit(1)


def get_excluded_directories() -> list[str]:
    """
    Get list of excluded directories from config or defaults.

    Priority:
    1. .agentqms/settings.yaml validation.excluded_directories
    2. Default: ['archive', 'deprecated']

    Returns:
        List of directory names to exclude
    """
    try:
        config = load_config()
        if config and "validation" in config:
            if "excluded_directories" in config["validation"]:
                return config["validation"]["excluded_directories"]
    except Exception:
        pass

    # Default exclusions
    return ["archive", "deprecated"]


def infer_artifact_date(file_path: Path) -> str:
    """
    Infer artifact creation date using intelligent fallback strategy.

    Priority:
    1. Git creation date (initial commit adding file)
    2. Git last modified date (most recent commit)
    3. Filesystem modification time (stat().st_mtime)
    4. Present date (last resort)

    Args:
        file_path: Path to artifact file

    Returns:
        Date string in "YYYY-MM-DD HH:MM (KST)" format
    """
    kst = timezone(timedelta(hours=9))

    try:
        # Try git creation date (initial commit)
        result = subprocess.run(
            ["git", "log", "--follow", "--format=%aI", "--diff-filter=A", "--", str(file_path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

        if result.returncode == 0 and result.stdout.strip():
            # Parse ISO 8601 date
            git_date_str = result.stdout.strip().split('\n')[-1]  # First (creation) commit
            git_date = datetime.fromisoformat(git_date_str.replace('Z', '+00:00'))
            kst_date = git_date.astimezone(kst)
            return kst_date.strftime("%Y-%m-%d %H:%M (KST)")
    except Exception:
        pass

    try:
        # Try git last modified date
        result = subprocess.run(
            ["git", "log", "-1", "--format=%aI", "--", str(file_path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

        if result.returncode == 0 and result.stdout.strip():
            git_date_str = result.stdout.strip()
            git_date = datetime.fromisoformat(git_date_str.replace('Z', '+00:00'))
            kst_date = git_date.astimezone(kst)
            return kst_date.strftime("%Y-%m-%d %H:%M (KST)")
    except Exception:
        pass

    try:
        # Fallback to filesystem modification time
        mtime = file_path.stat().st_mtime
        fs_date = datetime.fromtimestamp(mtime, tz=UTC)
        kst_date = fs_date.astimezone(kst)
        return kst_date.strftime("%Y-%m-%d %H:%M (KST)")
    except Exception:
        pass

    # Last resort: present date
    current_date = datetime.now(kst)
    return current_date.strftime("%Y-%m-%d %H:%M (KST)")


def fix_date_format(content: str, file_path: Path) -> str:
    """
    Fix date format in frontmatter using smart date inference.

    Args:
        content: File content with frontmatter
        file_path: Path to file for date inference

    Returns:
        Content with corrected date
    """
    lines = content.split("\n")
    fixed_lines = []
    date_fixed = False

    for line in lines:
        if line.startswith("date:") and not date_fixed:
            # Infer date from git/filesystem instead of using present date
            inferred_date = infer_artifact_date(file_path)
            fixed_lines.append(f'date: "{inferred_date}"')
            date_fixed = True
        else:
            fixed_lines.append(line)

    return "\n".join(fixed_lines)


def is_excluded_path(file_path: Path, excluded_dirs: list[str]) -> bool:
    """
    Check if file path contains any excluded directory.

    Args:
        file_path: Path to check
        excluded_dirs: List of excluded directory names

    Returns:
        True if path should be excluded
    """
    path_parts = file_path.parts
    return any(excluded_dir in path_parts for excluded_dir in excluded_dirs)


def create_git_stash_backup() -> bool:
    """
    Create automatic git stash backup before destructive operations.

    Returns:
        True if stash created successfully, False otherwise
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stash_msg = f"pre-audit-backup-{timestamp}"

        result = subprocess.run(
            ["git", "stash", "push", "-u", "-m", stash_msg],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        if result.returncode == 0:
            print(f"✅ Created git stash backup: {stash_msg}")
            return True
        else:
            print(f"⚠️  Failed to create git stash: {result.stderr}")
            return False
    except Exception as e:
        print(f"⚠️  Failed to create git stash: {e}")
        return False


class ArtifactAudit:
    """Audit and repair AgentQMS artifacts with safety features."""

    # Non-standard artifact directories (excluded by default)
    NON_STANDARD_DIRS = ["vlm_reports"]

    def __init__(self, include_excluded: bool = False):
        self.generator = FrontmatterGenerator()
        self.validator = ArtifactValidator()
        self.artifacts_dir = PROJECT_ROOT / "docs" / "artifacts"
        self.include_excluded = include_excluded
        self.excluded_dirs = get_excluded_directories()

        # Always exclude non-standard directories unless explicitly overridden
        if not include_excluded:
            self.excluded_dirs.extend(self.NON_STANDARD_DIRS)

    def get_batch_files(self, batch_num: int) -> list[Path]:
        """Get files for a specific batch from violation history."""
        # Batch mapping based on previous phases
        batch_mapping = {
            1: [
                "2024-11-20_1430_assessment_phase_4_validation.md",
                "2024-11-20_1450_assessment_batch_compliance.md",
                "2024-11-20_1510_assessment_naming_conventions.md",
                "2024-11-20_1530_assessment_frontmatter_standardization.md",
                "2024-11-20_1550_assessment_artifact_organization.md",
                "2024-11-20_1610_assessment_phase_4_completion.md",
                "2024-11-20_1630_assessment_batch_1_review.md",
                "2024-11-20_1650_assessment_batch_1_validation.md",
                "2024-11-20_1710_assessment_batch_1_final.md",
                "2024-11-20_1730_assessment_batch_1_compliance.md",
            ],
            2: [
                "2024-11-20_1800_assessment_batch_2_start.md",
                "2024-11-20_1820_assessment_batch_2_preparation.md",
                "2024-11-20_1840_bug-report_batch_2_naming.md",
                "2024-11-20_1900_bug-report_batch_2_structure.md",
                "2024-11-20_1920_assessment_batch_2_review_1.md",
                "2024-11-20_1940_assessment_batch_2_review_2.md",
                "2024-11-20_2000_assessment_batch_2_review_3.md",
                "2024-11-20_2020_assessment_batch_2_files_1_5.md",
                "2024-11-20_2040_assessment_batch_2_files_6_10.md",
                "2024-11-20_2100_assessment_batch_2_files_11_15.md",
                "2024-11-20_2120_assessment_batch_2_files_16_20.md",
                "2024-11-20_2140_bug-report_batch_2_file_1.md",
                "2024-11-20_2200_bug-report_batch_2_file_2.md",
                "2024-11-20_2220_bug-report_batch_2_file_3.md",
                "2024-11-20_2240_assessment_batch_2_validation_1.md",
                "2024-11-20_2300_assessment_batch_2_validation_2.md",
                "2024-11-20_2320_assessment_batch_2_validation_3.md",
                "2024-11-20_2340_assessment_batch_2_validation_4.md",
                "2024-11-20_2400_assessment_batch_2_final.md",
                "2024-11-20_2420_assessment_batch_2_compliance.md",
            ],
        }

        files = []
        if batch_num in batch_mapping:
            for filename in batch_mapping[batch_num]:
                # Check both assessments and bug_reports directories
                for subdir in ["assessments", "bug_reports"]:
                    fpath = self.artifacts_dir / subdir / filename
                    if fpath.exists():
                        # Apply exclusion filter
                        if not self.include_excluded and is_excluded_path(fpath, self.excluded_dirs):
                            print(f"⏭️  EXCLUDED: {fpath.relative_to(self.artifacts_dir)} (in excluded directory)")
                            continue
                        files.append(fpath)
                        break

        return files

    def list_all_artifacts(self) -> list[Path]:
        """
        List all artifact files, applying exclusion filters.

        Returns:
            List of artifact file paths
        """
        all_files = list(self.artifacts_dir.rglob("*.md"))

        if not self.include_excluded:
            filtered_files = [
                f for f in all_files
                if not is_excluded_path(f, self.excluded_dirs)
            ]
            excluded_count = len(all_files) - len(filtered_files)
            if excluded_count > 0:
                print(f"ℹ️  Excluded {excluded_count} file(s) in: {', '.join(self.excluded_dirs)}")
            return filtered_files

        return all_files

    def preview_changes(self, file_paths: list[Path]) -> None:
        """
        Preview files that will be modified.

        Args:
            file_paths: List of files to preview
        """
        print("\n📋 Preview of files to be processed:")
        print(f"   Total files: {len(file_paths)}")

        # Group by directory
        by_dir: dict[str, list[Path]] = {}
        for fpath in file_paths:
            relative_dir = fpath.parent.relative_to(self.artifacts_dir)
            dir_name = str(relative_dir)
            if dir_name not in by_dir:
                by_dir[dir_name] = []
            by_dir[dir_name].append(fpath)

        for dir_name, files in sorted(by_dir.items()):
            print(f"\n   {dir_name}/ ({len(files)} files)")
            for fpath in sorted(files)[:5]:  # Show first 5 per directory
                print(f"      - {fpath.name}")
            if len(files) > 5:
                print(f"      ... and {len(files) - 5} more")

    def fix_files(self, file_paths: list[Path], dry_run: bool = False) -> dict:
        """Fix frontmatter for given files using AgentQMS tools."""
        results = {
            "processed": 0,
            "success": 0,
            "skipped": 0,
            "errors": 0,
            "validation_results": {},
        }

        for file_path in file_paths:
            if not file_path.exists():
                print(f"⏭️  SKIP: {file_path.name} (not found)")
                results["skipped"] += 1
                continue

            print(f"🔧 Processing: {file_path.name}")
            results["processed"] += 1

            try:
                # Check if already has complete frontmatter
                content = file_path.read_text(encoding="utf-8")
                has_frontmatter = content.startswith("---")

                if has_frontmatter:
                    # Check if frontmatter is complete by validating
                    test_result = self.validator.validate_single_file(file_path)
                    if test_result.get("valid"):
                        print("   ✓ Already has valid frontmatter")
                        results["success"] += 1
                    else:
                        # Frontmatter exists but is incomplete
                        if not dry_run:
                            # Extract content after frontmatter
                            parts = content.split("---", 2)
                            if len(parts) >= 3:
                                content_body = parts[2]
                            else:
                                content_body = content

                            # Generate new frontmatter
                            new_frontmatter = self.generator.generate_frontmatter(str(file_path))

                            # Fix date format with smart inference
                            new_frontmatter = fix_date_format(new_frontmatter, file_path)

                            # Combine and write
                            new_content = new_frontmatter + content_body
                            file_path.write_text(new_content, encoding="utf-8")

                        print("   ✓ Fixed incomplete frontmatter")
                        results["success"] += 1
                else:
                    # Add frontmatter using AgentQMS generator
                    if not dry_run:
                        self.generator.add_frontmatter_to_file(str(file_path))

                        # Fix the date format after adding frontmatter with smart inference
                        content = file_path.read_text(encoding="utf-8")
                        content = fix_date_format(content, file_path)
                        file_path.write_text(content, encoding="utf-8")

                    print("   ✓ Added frontmatter")
                    results["success"] += 1

            except Exception as e:
                print(f"   ❌ Error: {e}")
                results["errors"] += 1
                continue

        return results

    def validate_files(self, file_paths: list[Path]) -> dict:
        """Validate files using ArtifactValidator."""
        validation_results = {
            "valid": [],
            "invalid": [],
        }

        for file_path in file_paths:
            if not file_path.exists():
                continue

            result = self.validator.validate_single_file(file_path)
            is_valid = result.get("valid", False)
            errors = result.get("errors", [])
            message = "; ".join(errors) if errors else "Unknown error"

            if is_valid:
                validation_results["valid"].append(file_path.name)
                print(f"✅ VALID: {file_path.name}")
            else:
                validation_results["invalid"].append((file_path.name, message))
                print(f"❌ INVALID: {file_path.name}")
                if message:
                    print(f"   Reason: {message}")

        return validation_results

    def run(
        self,
        batch: int | None = None,
        files: list[str] | None = None,
        all_artifacts: bool = False,
        dry_run: bool = False,
        report_only: bool = False,
        no_confirm: bool = False,
        no_stash: bool = False,
    ) -> int:
        """Main execution method."""
        target_files = []

        # Determine which files to process
        if batch:
            print(f"🎯 Targeting batch {batch}...")
            target_files = self.get_batch_files(batch)
            if not target_files:
                print(f"❌ No files found for batch {batch}")
                return 1
        elif files:
            target_files = [Path(f) for f in files]
            # Apply exclusion filter for manually specified files
            if not self.include_excluded:
                original_count = len(target_files)
                target_files = [
                    f for f in target_files
                    if not is_excluded_path(f, self.excluded_dirs)
                ]
                excluded_count = original_count - len(target_files)
                if excluded_count > 0:
                    print(f"⚠️  Excluded {excluded_count} file(s) in excluded directories")
        elif all_artifacts:
            print("🎯 Scanning all artifacts...")
            target_files = self.list_all_artifacts()
        else:
            print("❌ No target specified. Use --batch, --files, or --all")
            return 1

        if not target_files:
            print("❌ No files to process after applying filters")
            return 1

        print(f"📊 Found {len(target_files)} file(s) to process")

        # Show exclusion info
        if not self.include_excluded:
            print(f"ℹ️  Excluding directories: {', '.join(self.excluded_dirs)}")
            print("   Use --include-excluded to process all directories\n")
        else:
            print("⚠️  Including ALL directories (archive, deprecated, vlm_reports)\n")

        # Preview changes
        if not report_only:
            self.preview_changes(target_files)

        # Pre-flight confirmation for non-dry-run operations
        if not dry_run and not report_only and not no_confirm:
            print("\n⚠️  This will modify the above files!")
            response = input("   Continue? [y/N]: ").strip().lower()
            if response != 'y':
                print("❌ Aborted by user")
                return 1

            # Create git stash backup unless disabled
            if not no_stash:
                print("\n💾 Creating automatic backup...")
                if not create_git_stash_backup():
                    response = input("   Failed to create backup. Continue anyway? [y/N]: ").strip().lower()
                    if response != 'y':
                        print("❌ Aborted by user")
                        return 1

        if report_only:
            print("\n📋 Report mode (validating only, not modifying files)\n")
            results = self.validate_files(target_files)
        else:
            if dry_run:
                print("\n🔍 DRY RUN mode (not modifying files)\n")

            print("📝 Auditing and fixing artifacts...\n")
            fix_results = self.fix_files(target_files, dry_run=dry_run)

            print("\n📊 Audit Summary:")
            print(f"   Processed: {fix_results['processed']}")
            print(f"   Success: {fix_results['success']}")
            print(f"   Skipped: {fix_results['skipped']}")
            print(f"   Errors: {fix_results['errors']}")

            print("\n🔍 Validating artifacts...\n")
            results = self.validate_files(target_files)

        print("\n✅ Validation Summary:")
        print(f"   Valid: {len(results['valid'])}")
        print(f"   Invalid: {len(results['invalid'])}")

        if results["invalid"]:
            print("\n❌ Invalid artifacts:")
            for filename, reason in results["invalid"]:
                print(f"   - {filename}: {reason}")
            return 1

        return 0


def main():
    """Parse arguments and run artifact audit."""
    parser = argparse.ArgumentParser(
        description="Audit and fix artifact compliance using AgentQMS tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview batch 1 changes (dry-run)
  python artifact_audit.py --batch 1 --dry-run

  # Fix batch 1 artifacts with confirmation
  python artifact_audit.py --batch 1

  # Fix all artifacts (excluding archive/deprecated)
  python artifact_audit.py --all

  # Include archived artifacts
  python artifact_audit.py --all --include-excluded

  # Report violations without fixing
  python artifact_audit.py --all --report

  # Fix without confirmation or backup (danger!)
  python artifact_audit.py --batch 1 --no-confirm --no-stash
        """
    )
    parser.add_argument(
        "--batch", type=int, help="Audit batch N artifacts (1, 2, etc.)"
    )
    parser.add_argument(
        "--files", nargs="+", help="Specific files to audit"
    )
    parser.add_argument(
        "--all", action="store_true", help="Audit all artifacts"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without modifying"
    )
    parser.add_argument(
        "--report", action="store_true", help="Report violations only (no fixes)"
    )
    parser.add_argument(
        "--include-excluded", action="store_true",
        help="Include archive/, deprecated/, and non-standard directories"
    )
    parser.add_argument(
        "--no-confirm", action="store_true",
        help="Skip confirmation prompt (use with caution)"
    )
    parser.add_argument(
        "--no-stash", action="store_true",
        help="Skip automatic git stash backup (use with caution)"
    )

    args = parser.parse_args()

    auditor = ArtifactAudit(include_excluded=args.include_excluded)
    return auditor.run(
        batch=args.batch,
        files=args.files,
        all_artifacts=args.all,
        dry_run=args.dry_run,
        report_only=args.report,
        no_confirm=args.no_confirm,
        no_stash=args.no_stash,
    )


if __name__ == "__main__":
    sys.exit(main())
