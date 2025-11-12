#!/usr/bin/env python3
"""
Automated File Reorganization Script

This script automatically reorganizes misplaced artifact files by moving them
to the correct directories based on their naming conventions and content analysis.

Usage:
    python reorganize_files.py --move-to-correct-dirs
    python reorganize_files.py --dry-run
    python reorganize_files.py --file path/to/file.md
    python reorganize_files.py --directory docs/artifacts/
"""

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MoveOperation:
    """Represents a file move operation"""

    old_path: str
    new_path: str
    reason: str
    confidence: float


class FileReorganizer:
    """Automatically reorganizes misplaced artifact files"""

    def __init__(self, artifacts_root: str = "docs/artifacts"):
        self.artifacts_root = Path(artifacts_root)
        self.backup_dir = Path("backups/file_reorganization")

        # Define valid prefixes and their expected directories
        self.valid_prefixes = {
            "IMPLEMENTATION_PLAN_": "implementation_plans/",
            "assessment-": "assessments/",
            "design-": "design_documents/",
            "research-": "research/",
            "template-": "templates/",
            "BUG_": "bug_reports/",
            "SESSION_": "completed_plans/completion_summaries/session_notes/",
        }

        # Directory structure mapping
        self.directory_structure = {
            "implementation_plans": {
                "description": "Implementation plans and blueprints",
                "prefixes": ["IMPLEMENTATION_PLAN_"],
                "types": ["implementation_plan"],
            },
            "assessments": {
                "description": "Assessments and evaluations",
                "prefixes": ["assessment-"],
                "types": ["assessment"],
            },
            "design_documents": {
                "description": "Design documents and architecture",
                "prefixes": ["design-"],
                "types": ["design"],
            },
            "research": {
                "description": "Research documents and findings",
                "prefixes": ["research-"],
                "types": ["research"],
            },
            "templates": {
                "description": "Templates and examples",
                "prefixes": ["template-"],
                "types": ["template"],
            },
            "bug_reports": {
                "description": "Bug reports and troubleshooting",
                "prefixes": ["BUG_"],
                "types": ["bug_report"],
            },
            "completed_plans": {
                "description": "Completed plans and summaries",
                "prefixes": ["SESSION_"],
                "types": ["session_note", "completion_summary"],
            },
        }

        # Content-based type detection patterns
        self.type_patterns = {
            "implementation_plan": [
                r"implementation\s+plan",
                r"blueprint",
                r"roadmap",
                r"development\s+plan",
                r"project\s+plan",
                r"technical\s+plan",
                r"architecture\s+plan",
                r"phase\s+\d+",
                r"task\s+\d+",
                r"milestone",
                r"timeline",
            ],
            "assessment": [
                r"assessment",
                r"evaluation",
                r"audit",
                r"review",
                r"analysis",
                r"compliance\s+check",
                r"quality\s+assessment",
                r"performance\s+review",
                r"risk\s+assessment",
                r"security\s+audit",
            ],
            "design": [
                r"design\s+document",
                r"architecture",
                r"technical\s+design",
                r"system\s+design",
                r"component\s+design",
                r"interface\s+design",
                r"uml",
                r"diagram",
                r"flowchart",
                r"wireframe",
            ],
            "research": [
                r"research",
                r"investigation",
                r"study",
                r"findings",
                r"analysis",
                r"exploration",
                r"discovery",
                r"hypothesis",
                r"methodology",
                r"experiment",
                r"results",
            ],
            "template": [
                r"template",
                r"example",
                r"sample",
                r"boilerplate",
                r"reference",
                r"guide",
                r"how.to",
                r"checklist",
                r"format",
                r"structure",
            ],
            "bug_report": [
                r"bug\s+report",
                r"issue",
                r"problem",
                r"error",
                r"defect",
                r"troubleshooting",
                r"fix",
                r"resolution",
                r"bug\s+fix",
                r"patch",
            ],
            "session_note": [
                r"session\s+note",
                r"meeting\s+note",
                r"discussion",
                r"conversation",
                r"chat",
                r"log",
            ],
            "completion_summary": [
                r"completion\s+summary",
                r"wrap.up",
                r"final\s+report",
                r"project\s+summary",
                r"results",
                r"outcome",
            ],
        }

    def analyze_file_placement(self, file_path: Path) -> MoveOperation | None:
        """Analyze if file is in correct directory and suggest move if needed"""
        filename = file_path.name

        # Skip INDEX.md files
        if filename == "INDEX.md":
            return None

        # Determine expected directory from filename prefix
        expected_dir = None
        for prefix, directory in self.valid_prefixes.items():
            if filename.startswith(prefix):
                expected_dir = directory.rstrip("/")
                break

        if not expected_dir:
            # No prefix found, try to determine from content
            expected_dir = self._determine_directory_from_content(file_path)
            if expected_dir:
                return MoveOperation(
                    old_path=str(file_path),
                    new_path=str(self.artifacts_root / expected_dir / filename),
                    reason=f"Move to {expected_dir} based on content analysis",
                    confidence=0.7,
                )
            return None

        # Check current directory
        current_dir = str(file_path.parent.relative_to(self.artifacts_root))

        if current_dir != expected_dir:
            return MoveOperation(
                old_path=str(file_path),
                new_path=str(self.artifacts_root / expected_dir / filename),
                reason=f"Move to correct directory: {expected_dir}",
                confidence=0.9,
            )

        return None

    def _determine_directory_from_content(self, file_path: Path) -> str | None:
        """Determine correct directory from file content analysis"""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
        except Exception:
            return None

        # Check frontmatter first
        frontmatter_type = self._extract_type_from_frontmatter(content)
        if frontmatter_type:
            directory = self._get_directory_for_type(frontmatter_type)
            if directory:
                return directory

        # Analyze content for type patterns
        content_lower = content.lower()
        for artifact_type, patterns in self.type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_lower, re.IGNORECASE):
                    directory = self._get_directory_for_type(artifact_type)
                    if directory:
                        return directory

        return None

    def _extract_type_from_frontmatter(self, content: str) -> str | None:
        """Extract type from frontmatter"""
        frontmatter_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
        if not frontmatter_match:
            return None

        frontmatter_content = frontmatter_match.group(1)
        type_match = re.search(r'type:\s*["\']?([^"\'\n]+)["\']?', frontmatter_content)

        if type_match:
            return type_match.group(1).strip()

        return None

    def _get_directory_for_type(self, artifact_type: str) -> str | None:
        """Get directory name for artifact type"""
        type_to_directory = {
            "implementation_plan": "implementation_plans",
            "assessment": "assessments",
            "design": "design_documents",
            "research": "research",
            "template": "templates",
            "bug_report": "bug_reports",
            "session_note": "completed_plans/completion_summaries/session_notes",
            "completion_summary": "completed_plans/completion_summaries",
        }

        return type_to_directory.get(artifact_type)

    def create_backup(self, file_path: Path) -> bool:
        """Create backup of file before moving"""
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = self.backup_dir / file_path.name
            shutil.copy2(file_path, backup_path)
            return True
        except Exception as e:
            print(f"Warning: Could not create backup for {file_path}: {e}")
            return False

    def execute_move_operation(
        self, operation: MoveOperation, dry_run: bool = False
    ) -> bool:
        """Execute a move operation"""
        old_path = Path(operation.old_path)
        new_path = Path(operation.new_path)

        if dry_run:
            print(f"[DRY RUN] Would move: {old_path} -> {new_path}")
            print(f"         Reason: {operation.reason}")
            return True

        try:
            # Create backup
            self.create_backup(old_path)

            # Create target directory if it doesn't exist
            new_path.parent.mkdir(parents=True, exist_ok=True)

            # Check if target file already exists
            if new_path.exists():
                print(f"⚠️  Target file already exists: {new_path}")
                # Generate unique name
                counter = 1
                while new_path.exists():
                    stem = new_path.stem
                    suffix = new_path.suffix
                    new_path = new_path.parent / f"{stem}_{counter}{suffix}"
                    counter += 1
                print(f"   Using unique name: {new_path}")

            # Perform move
            shutil.move(str(old_path), str(new_path))
            print(f"✅ Moved: {old_path} -> {new_path}")
            print(f"   Reason: {operation.reason}")
            return True

        except Exception as e:
            print(f"❌ Failed to move {old_path}: {e}")
            return False

    def reorganize_file(
        self, file_path: Path, dry_run: bool = False
    ) -> MoveOperation | None:
        """Reorganize a single file"""
        operation = self.analyze_file_placement(file_path)

        if not operation:
            print(f"✅ {file_path} is already in correct directory")
            return None

        print(f"🔧 Found misplaced file: {file_path}")
        success = self.execute_move_operation(operation, dry_run)

        if success:
            return operation
        else:
            return None

    def reorganize_directory(
        self, directory: Path, dry_run: bool = False
    ) -> dict[str, MoveOperation]:
        """Reorganize all files in a directory"""
        results = {}

        for file_path in directory.rglob("*.md"):
            if file_path.is_file():
                operation = self.reorganize_file(file_path, dry_run)
                if operation:
                    results[str(file_path)] = operation

        return results

    def generate_reorganization_report(self, results: dict[str, MoveOperation]) -> str:
        """Generate a comprehensive reorganization report"""
        report = []
        report.append("📁 File Reorganization Report")
        report.append("=" * 50)

        total_files = len(results)

        if total_files == 0:
            report.append("✅ All files are already in correct directories!")
            return "\n".join(report)

        report.append(f"Files reorganized: {total_files}")
        report.append("")

        # Group by destination directory
        by_destination = {}
        for file_path, operation in results.items():
            dest_dir = Path(operation.new_path).parent.name
            if dest_dir not in by_destination:
                by_destination[dest_dir] = []
            by_destination[dest_dir].append((file_path, operation))

        report.append("Files moved by destination:")
        report.append("-" * 30)

        for dest_dir, files in by_destination.items():
            report.append(f"\n📂 {dest_dir}/ ({len(files)} files)")
            for file_path, operation in files:
                old_path = Path(operation.old_path)
                report.append(f"   • {old_path.name}")
                report.append(f"     From: {old_path.parent.name}/")
                report.append(f"     Reason: {operation.reason}")
                report.append(f"     Confidence: {operation.confidence:.1f}")

        return "\n".join(report)

    def validate_directory_structure(self, directory: Path) -> dict[str, list[str]]:
        """Validate directory structure and identify misplaced files"""
        issues = {}

        for file_path in directory.rglob("*.md"):
            if file_path.is_file() and file_path.name != "INDEX.md":
                file_issues = []

                # Check if file is in correct directory
                operation = self.analyze_file_placement(file_path)
                if operation:
                    file_issues.append(
                        f"Should be in {Path(operation.new_path).parent.name}/"
                    )

                # Check if directory exists and is valid
                current_dir = str(file_path.parent.relative_to(self.artifacts_root))
                if current_dir not in self.directory_structure:
                    file_issues.append(f"Invalid directory: {current_dir}")

                if file_issues:
                    issues[str(file_path)] = file_issues

        return issues

    def create_directory_structure(self, dry_run: bool = False) -> bool:
        """Create missing directory structure"""
        if dry_run:
            print("[DRY RUN] Would create directory structure:")
            for dir_name, info in self.directory_structure.items():
                dir_path = self.artifacts_root / dir_name
                if not dir_path.exists():
                    print(f"  - {dir_path}")
            return True

        created_dirs = []
        for dir_name, info in self.directory_structure.items():
            dir_path = self.artifacts_root / dir_name
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                created_dirs.append(str(dir_path))
                print(f"✅ Created directory: {dir_path}")

        if created_dirs:
            print(f"Created {len(created_dirs)} directories")
        else:
            print("All directories already exist")

        return True

    def generate_directory_structure_report(self) -> str:
        """Generate a report of the directory structure"""
        report = []
        report.append("📁 Directory Structure Report")
        report.append("=" * 50)

        for dir_name, info in self.directory_structure.items():
            dir_path = self.artifacts_root / dir_name
            exists = dir_path.exists()
            status = "✅" if exists else "❌"

            report.append(f"{status} {dir_name}/")
            report.append(f"   Description: {info['description']}")
            report.append(f"   Prefixes: {', '.join(info['prefixes'])}")
            report.append(f"   Types: {', '.join(info['types'])}")

            if exists:
                # Count files in directory
                file_count = len(list(dir_path.rglob("*.md")))
                report.append(f"   Files: {file_count}")

            report.append("")

        return "\n".join(report)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Reorganize misplaced artifact files")
    parser.add_argument("--file", help="Reorganize specific file")
    parser.add_argument(
        "--directory", default="docs/artifacts", help="Directory to process"
    )
    parser.add_argument(
        "--move-to-correct-dirs",
        action="store_true",
        help="Move files to correct directories",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be moved without making changes",
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="Only validate, do not move files"
    )
    parser.add_argument(
        "--create-dirs", action="store_true", help="Create missing directory structure"
    )
    parser.add_argument(
        "--structure-report",
        action="store_true",
        help="Generate directory structure report",
    )
    parser.add_argument("--output", help="Output file for report")
    parser.add_argument(
        "--artifacts-root",
        default="docs/artifacts",
        help="Root directory for artifacts",
    )

    args = parser.parse_args()

    reorganizer = FileReorganizer(args.artifacts_root)

    if args.structure_report:
        # Generate directory structure report
        report = reorganizer.generate_directory_structure_report()
        print(report)
        return

    if args.create_dirs:
        # Create directory structure
        reorganizer.create_directory_structure(dry_run=args.dry_run)
        return

    if args.validate_only:
        # Validation only mode
        directory = Path(args.directory)
        issues = reorganizer.validate_directory_structure(directory)

        if issues:
            print("❌ Misplaced files found:")
            for file_path, file_issues in issues.items():
                print(f"\n📁 {file_path}")
                for issue in file_issues:
                    print(f"   • {issue}")
        else:
            print("✅ All files are in correct directories!")
        return

    if args.file:
        # Process specific file
        file_path = Path(args.file)
        operation = reorganizer.reorganize_file(file_path, dry_run=args.dry_run)
        results = {str(file_path): operation} if operation else {}
    else:
        # Process directory
        directory = Path(args.directory)
        results = reorganizer.reorganize_directory(directory, dry_run=args.dry_run)

    # Generate report
    report = reorganizer.generate_reorganization_report(results)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report saved to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
