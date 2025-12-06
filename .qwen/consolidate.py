#!/usr/bin/env python3
"""
Consolidate Script: Wrapper for AgentQMS frontmatter & validation tools.

This script reuses existing AgentQMS tools instead of duplicating logic:
- FrontmatterGenerator (AgentQMS/toolkit/maintenance/add_frontmatter.py)
- ArtifactValidator (AgentQMS/agent_tools/compliance/validate_artifacts.py)

Usage:
    python consolidate.py --batch N         # Fix batch N files
    python consolidate.py --files f1 f2 ... # Fix specific files
    python consolidate.py --all              # Fix all artifacts missing frontmatter
    python consolidate.py --dry-run          # Preview changes without modifying
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add AgentQMS to path for imports
AGENTQMS_PATH = Path(__file__).parent.parent / "AgentQMS"
sys.path.insert(0, str(AGENTQMS_PATH))

try:
    from agent_tools.compliance.validate_artifacts import ArtifactValidator
    from toolkit.maintenance.add_frontmatter import FrontmatterGenerator
except ImportError as e:
    print(f"❌ Error importing AgentQMS tools: {e}")
    print("   Make sure you're running from project root")
    sys.exit(1)


def fix_date_format(content: str) -> str:
    """Fix date format in frontmatter to KST format."""
    from datetime import timedelta
    lines = content.split("\n")
    fixed_lines = []
    date_fixed = False

    for line in lines:
        if line.startswith("date:") and not date_fixed:
            # Get current date in KST (UTC+9)
            kst = timezone(timedelta(hours=9))
            current_date = datetime.now(kst).strftime("%Y-%m-%d %H:%M (KST)")
            fixed_lines.append(f'date: "{current_date}"')
            date_fixed = True
        else:
            fixed_lines.append(line)

    return "\n".join(fixed_lines)


class ConsolidateQMS:
    """Wrapper around AgentQMS tools for batch operations."""

    def __init__(self):
        self.generator = FrontmatterGenerator()
        self.validator = ArtifactValidator()
        self.artifacts_dir = Path("docs/artifacts")

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
                        files.append(fpath)
                        break

        return files

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

                            # Fix date format to KST
                            new_frontmatter = fix_date_format(new_frontmatter)

                            # Combine and write
                            new_content = new_frontmatter + content_body
                            file_path.write_text(new_content, encoding="utf-8")

                        print("   ✓ Fixed incomplete frontmatter")
                        results["success"] += 1
                else:
                    # Add frontmatter using AgentQMS generator
                    if not dry_run:
                        self.generator.add_frontmatter_to_file(str(file_path))

                        # Fix the date format after adding frontmatter
                        content = file_path.read_text(encoding="utf-8")
                        content = fix_date_format(content)
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
        validate_only: bool = False,
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
        elif all_artifacts:
            print("🎯 Scanning all artifacts...")
            target_files = list(self.artifacts_dir.rglob("*.md"))
        else:
            print("❌ No target specified. Use --batch, --files, or --all")
            return 1

        print(f"📊 Found {len(target_files)} file(s) to process\n")

        if validate_only:
            print("🔍 Validation mode (not modifying files)\n")
            results = self.validate_files(target_files)
        else:
            if dry_run:
                print("🔍 DRY RUN mode (not modifying files)\n")

            print("📝 Fixing frontmatter...\n")
            fix_results = self.fix_files(target_files, dry_run=dry_run)

            print("\n📊 Fix Summary:")
            print(f"   Processed: {fix_results['processed']}")
            print(f"   Success: {fix_results['success']}")
            print(f"   Skipped: {fix_results['skipped']}")
            print(f"   Errors: {fix_results['errors']}")

            print("\n🔍 Validating files...\n")
            results = self.validate_files(target_files)

        print("\n✅ Validation Summary:")
        print(f"   Valid: {len(results['valid'])}")
        print(f"   Invalid: {len(results['invalid'])}")

        if results["invalid"]:
            print("\n❌ Invalid files:")
            for filename, reason in results["invalid"]:
                print(f"   - {filename}: {reason}")
            return 1

        return 0


def main():
    """Parse arguments and run consolidation."""
    parser = argparse.ArgumentParser(
        description="Consolidate & fix artifact frontmatter using AgentQMS tools"
    )
    parser.add_argument(
        "--batch", type=int, help="Fix batch N files (1, 2, etc.)"
    )
    parser.add_argument(
        "--files", nargs="+", help="Specific files to process"
    )
    parser.add_argument(
        "--all", action="store_true", help="Process all artifacts"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview without modifying"
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="Only validate, don't fix"
    )

    args = parser.parse_args()

    consolidator = ConsolidateQMS()
    return consolidator.run(
        batch=args.batch,
        files=args.files,
        all_artifacts=args.all,
        dry_run=args.dry_run,
        validate_only=args.validate_only,
    )


if __name__ == "__main__":
    sys.exit(main())
