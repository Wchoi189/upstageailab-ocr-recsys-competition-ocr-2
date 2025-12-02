#!/usr/bin/env python3
"""
Automated Fix Script for Naming Convention Violations

This script automatically fixes naming convention violations in artifact files.
It handles:
- Missing timestamp prefixes
- Missing type prefixes
- Incorrect directory placement
- Invalid descriptive naming (underscores vs hyphens)

Usage:
    python fix_naming_conventions.py --auto-fix
    python fix_naming_conventions.py --dry-run
    python fix_naming_conventions.py --file path/to/file.md
    python fix_naming_conventions.py --directory docs/artifacts/
"""

import argparse
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class RenameOperation:
    """Represents a file rename operation"""

    old_path: str
    new_path: str
    reason: str
    confidence: float


class NamingConventionFixer:
    """Automatically fixes naming convention violations"""

    def __init__(self, artifacts_root: str = "docs/artifacts"):
        self.artifacts_root = Path(artifacts_root)
        self.backup_dir = Path("backups/naming_fixes")

        # Define valid prefixes and their expected directories
        self.valid_prefixes = {
            "implementation_plan_": "implementation_plans/",
            "assessment-": "assessments/",
            "design-": "design_documents/",
            "research-": "research/",
            "template-": "templates/",
            "BUG_": "bug_reports/",
            "SESSION_": "completed_plans/completion_summaries/session_notes/",
        }

        # Type detection patterns for content analysis
        self.type_patterns = {
            "implementation_plan": [
                r"implementation\s+plan",
                r"blueprint",
                r"roadmap",
                r"development\s+plan",
                r"project\s+plan",
                r"technical\s+plan",
                r"architecture\s+plan",
            ],
            "assessment": [
                r"assessment",
                r"evaluation",
                r"audit",
                r"review",
                r"analysis",
                r"compliance\s+check",
                r"quality\s+assessment",
            ],
            "design": [
                r"design\s+document",
                r"architecture",
                r"technical\s+design",
                r"system\s+design",
                r"component\s+design",
                r"interface\s+design",
            ],
            "research": [
                r"research",
                r"investigation",
                r"study",
                r"findings",
                r"analysis",
                r"exploration",
                r"discovery",
            ],
            "template": [
                r"template",
                r"example",
                r"sample",
                r"boilerplate",
                r"reference",
                r"guide",
                r"how.to",
            ],
            "bug_report": [
                r"bug\s+report",
                r"issue",
                r"problem",
                r"error",
                r"defect",
                r"troubleshooting",
                r"fix",
            ],
        }

    def analyze_naming_issues(self, file_path: Path) -> list[RenameOperation]:
        """Analyze naming issues and generate fix operations"""
        operations = []
        filename = file_path.name

        # Skip INDEX.md and registry files
        skip_patterns = [
            "INDEX.md",
            "MASTER_INDEX.md",
            "REGISTRY.md",
            "README.md"
        ]
        if any(filename.upper() == pattern.upper() for pattern in skip_patterns):
            return operations

        # Check timestamp format
        timestamp_match = re.match(r"^(\d{4}-\d{2}-\d{2}_\d{4}_)", filename)
        if not timestamp_match:
            operation = self._fix_timestamp_issue(file_path, filename)
            if operation:
                operations.append(operation)

        # Check type prefix (after timestamp if present)
        prefix_part = filename[len(timestamp_match.group(1)):] if timestamp_match else filename
        has_valid_prefix = any(
            prefix_part.startswith(prefix) for prefix in self.valid_prefixes
        )
        if not has_valid_prefix:
            operation = self._fix_type_prefix_issue(file_path, filename)
            if operation:
                operations.append(operation)

        # Check descriptive naming (kebab-case)
        operation = self._fix_descriptive_naming(file_path, filename)
        if operation:
            operations.append(operation)

        # Check directory placement
        operation = self._fix_directory_placement(file_path, filename)
        if operation:
            operations.append(operation)

        return operations

    def _fix_timestamp_issue(
        self, file_path: Path, filename: str
    ) -> RenameOperation | None:
        """Fix missing or invalid timestamp"""
        # Try to extract timestamp from filename
        timestamp_match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{4})", filename)
        if timestamp_match:
            # Timestamp exists but not at start
            timestamp = timestamp_match.group(1)
            new_filename = f"{timestamp}_{filename.replace(timestamp + '_', '')}"
        else:
            # No timestamp found, use current time
            current_time = datetime.now().strftime("%Y-%m-%d_%H%M")
            new_filename = f"{current_time}_{filename}"

        new_path = file_path.parent / new_filename
        return RenameOperation(
            old_path=str(file_path),
            new_path=str(new_path),
            reason="Fix timestamp format",
            confidence=0.9,
        )

    def _fix_type_prefix_issue(
        self, file_path: Path, filename: str
    ) -> RenameOperation | None:
        """Fix missing type prefix by analyzing content and directory"""
        # Determine type from directory structure
        directory_type = self._detect_type_from_directory(file_path)

        # Determine type from content analysis
        content_type = self._detect_type_from_content(file_path)

        # Use content analysis if available, otherwise use directory
        detected_type = content_type or directory_type

        if detected_type:
            # Find appropriate prefix for the type
            prefix = self._get_prefix_for_type(detected_type)
            if prefix:
                # Remove existing timestamp if present
                timestamp_match = re.search(r"^\d{4}-\d{2}-\d{2}_\d{4}_", filename)
                if timestamp_match:
                    timestamp = timestamp_match.group(0)
                    descriptive_part = filename[len(timestamp) :]
                else:
                    timestamp = ""
                    descriptive_part = filename

                new_filename = f"{timestamp}{prefix}{descriptive_part}"
                new_path = file_path.parent / new_filename

                return RenameOperation(
                    old_path=str(file_path),
                    new_path=str(new_path),
                    reason=f"Add {detected_type} type prefix",
                    confidence=0.8,
                )

        return None

    def _fix_descriptive_naming(
        self, file_path: Path, filename: str
    ) -> RenameOperation | None:
        """Fix descriptive naming to use kebab-case"""
        # Extract descriptive part (after prefix and timestamp)
        descriptive_part = filename

        # Remove timestamp
        timestamp_match = re.search(r"^\d{4}-\d{2}-\d{2}_\d{4}_", descriptive_part)
        if timestamp_match:
            descriptive_part = descriptive_part[len(timestamp_match.group(0)) :]

        # Remove type prefix
        for prefix in self.valid_prefixes:
            if descriptive_part.startswith(prefix):
                descriptive_part = descriptive_part[len(prefix) :]
                break

        # Remove .md extension
        if descriptive_part.endswith(".md"):
            descriptive_part = descriptive_part[:-3]

        # Check if needs kebab-case conversion
        if (
            "_" in descriptive_part
            and not descriptive_part.replace("-", "").replace("_", "").isalnum()
        ):
            # Convert underscores to hyphens
            kebab_case = descriptive_part.replace("_", "-")

            # Reconstruct filename
            timestamp_part = timestamp_match.group(0) if timestamp_match else ""
            prefix_part = ""
            for prefix in self.valid_prefixes:
                if filename.startswith(prefix):
                    prefix_part = prefix
                    break

            new_filename = f"{timestamp_part}{prefix_part}{kebab_case}.md"
            new_path = file_path.parent / new_filename

            return RenameOperation(
                old_path=str(file_path),
                new_path=str(new_path),
                reason="Convert to kebab-case",
                confidence=0.95,
            )

        return None

    def _fix_directory_placement(
        self, file_path: Path, filename: str
    ) -> RenameOperation | None:
        """Fix incorrect directory placement"""
        # Determine expected directory from filename prefix
        expected_dir = None
        for prefix, directory in self.valid_prefixes.items():
            if filename.startswith(prefix):
                expected_dir = directory.rstrip("/")
                break

        if expected_dir:
            current_dir = str(file_path.parent.relative_to(self.artifacts_root))
            if current_dir != expected_dir:
                # Create new path in correct directory
                new_path = self.artifacts_root / expected_dir / filename

                return RenameOperation(
                    old_path=str(file_path),
                    new_path=str(new_path),
                    reason=f"Move to correct directory: {expected_dir}",
                    confidence=0.9,
                )

        return None

    def _detect_type_from_directory(self, file_path: Path) -> str | None:
        """Detect artifact type from directory structure"""
        relative_path = file_path.relative_to(self.artifacts_root)
        directory = str(relative_path.parent)

        directory_type_mapping = {
            "implementation_plans": "implementation_plan",
            "assessments": "assessment",
            "design_documents": "design",
            "research": "research",
            "templates": "template",
            "bug_reports": "bug_report",
            "session_notes": "session_note",
        }

        return directory_type_mapping.get(directory)

    def _detect_type_from_content(self, file_path: Path) -> str | None:
        """Detect artifact type from content analysis"""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read().lower()

            # Check for type patterns
            for artifact_type, patterns in self.type_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        return artifact_type

            return None
        except Exception:
            return None

    def _get_prefix_for_type(self, artifact_type: str) -> str | None:
        """Get the appropriate prefix for an artifact type"""
        type_to_prefix = {
            "implementation_plan": "implementation_plan_",
            "assessment": "assessment-",
            "design": "design-",
            "research": "research-",
            "template": "template-",
            "bug_report": "BUG_",
            "session_note": "SESSION_",
        }

        return type_to_prefix.get(artifact_type)

    def create_backup(self, file_path: Path) -> bool:
        """Create backup of file before renaming"""
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = self.backup_dir / file_path.name
            shutil.copy2(file_path, backup_path)
            return True
        except Exception as e:
            print(f"Warning: Could not create backup for {file_path}: {e}")
            return False

    def execute_rename_operation(
        self, operation: RenameOperation, dry_run: bool = False
    ) -> bool:
        """Execute a rename operation"""
        old_path = Path(operation.old_path)
        new_path = Path(operation.new_path)

        if dry_run:
            print(f"[DRY RUN] Would rename: {old_path} -> {new_path}")
            print(f"         Reason: {operation.reason}")
            return True

        try:
            # Create backup
            self.create_backup(old_path)

            # Create target directory if it doesn't exist
            new_path.parent.mkdir(parents=True, exist_ok=True)

            # Perform rename
            old_path.rename(new_path)
            print(f"✅ Renamed: {old_path} -> {new_path}")
            print(f"   Reason: {operation.reason}")
            return True

        except Exception as e:
            print(f"❌ Failed to rename {old_path}: {e}")
            return False

    def fix_file(self, file_path: Path, dry_run: bool = False) -> list[RenameOperation]:
        """Fix naming issues for a single file"""
        operations = self.analyze_naming_issues(file_path)

        if not operations:
            print(f"✅ No naming issues found for {file_path}")
            return operations

        print(f"🔧 Found {len(operations)} naming issues for {file_path}")

        for operation in operations:
            success = self.execute_rename_operation(operation, dry_run)
            if success and not dry_run:
                # Update file_path for next operation
                file_path = Path(operation.new_path)

        return operations

    def validate_operations(self, operations: dict[str, list[RenameOperation]]) -> tuple[bool, list[str]]:
        """Validate all operations before execution
        
        Returns:
            tuple: (all_valid, list_of_issues)
        """
        issues = []
        final_paths = {}
        
        for old_path, ops in operations.items():
            # Trace through sequential operations to find final path
            current_path = old_path
            for op in ops:
                current_path = op.new_path
            
            # Check for duplicate final paths
            if current_path in final_paths:
                issues.append(f"Conflict: {old_path} and {final_paths[current_path]} both rename to {current_path}")
            else:
                final_paths[current_path] = old_path
            
            # Check if any intermediate target exists
            for op in ops:
                if Path(op.new_path).exists() and op.new_path != op.old_path:
                    issues.append(f"Target exists: {op.new_path}")
        
        return len(issues) == 0, issues

    def fix_directory(
        self, directory: Path, dry_run: bool = False, limit: int | None = None, validate: bool = True
    ) -> dict[str, list[RenameOperation]]:
        """Fix naming issues for all files in a directory"""
        results = {}
        files_processed = 0

        for file_path in directory.rglob("*.md"):
            if file_path.is_file():
                operations = self.fix_file(file_path, dry_run)
                if operations:
                    results[str(file_path)] = operations
                    files_processed += 1
                    
                    # Check limit
                    if limit is not None and files_processed >= limit:
                        print(f"✋ Reached file limit ({limit}). Stopping.")
                        break

        # Validate operations before returning
        if validate and results and not dry_run:
            valid, issues = self.validate_operations(results)
            if not valid:
                print("\n⚠️  Validation issues detected:")
                for issue in issues:
                    print(f"   • {issue}")
                print("\nℹ️  No changes will be made. Use --dry-run to preview.")
                return {}
        
        return results

    def generate_fix_report(self, results: dict[str, list[RenameOperation]]) -> str:
        """Generate a comprehensive fix report"""
        report = []
        report.append("🔧 Naming Convention Fix Report")
        report.append("=" * 50)

        total_files = len(results)
        total_operations = sum(len(ops) for ops in results.values())

        report.append(f"Files processed: {total_files}")
        report.append(f"Total operations: {total_operations}")
        report.append("")

        if results:
            report.append("Files with fixes applied:")
            report.append("-" * 30)

            for file_path, operations in results.items():
                report.append(f"\n📁 {file_path}")
                for operation in operations:
                    report.append(
                        f"   • {operation.reason} (confidence: {operation.confidence:.1f})"
                    )
                    report.append(f"     {operation.old_path} -> {operation.new_path}")
        else:
            report.append("✅ No naming issues found!")

        return "\n".join(report)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Fix naming convention violations")
    parser.add_argument("--file", help="Fix specific file")
    parser.add_argument("--directory", default="docs/artifacts", help="Fix directory")
    parser.add_argument("--auto-fix", action="store_true", help="Apply automatic fixes")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fixed without making changes",
    )
    parser.add_argument("--output", help="Output file for report")
    parser.add_argument(
        "--artifacts-root",
        default="docs/artifacts",
        help="Root directory for artifacts",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of files to process",
    )

    args = parser.parse_args()

    fixer = NamingConventionFixer(args.artifacts_root)

    if args.file:
        file_path = Path(args.file)
        operations = fixer.fix_file(file_path, dry_run=args.dry_run)
        results = {str(file_path): operations} if operations else {}
    else:
        directory = Path(args.directory)
        results = fixer.fix_directory(directory, dry_run=args.dry_run, limit=args.limit)

    # Generate report
    report = fixer.generate_fix_report(results)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report saved to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
