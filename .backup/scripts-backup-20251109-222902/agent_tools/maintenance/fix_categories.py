#!/usr/bin/env python3
"""
Automated Category and Type Fix Script

This script automatically fixes invalid category and type values in artifact frontmatter.
It uses intelligent mapping and content analysis to suggest appropriate corrections.

Usage:
    python fix_categories.py --auto-correct
    python fix_categories.py --dry-run
    python fix_categories.py --file path/to/file.md
    python fix_categories.py --directory docs/artifacts/
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CategoryFix:
    """Represents a category/type fix operation"""

    file_path: str
    field: str  # 'category' or 'type'
    old_value: str
    new_value: str
    reason: str
    confidence: float


class CategoryTypeFixer:
    """Automatically fixes invalid category and type values"""

    def __init__(self, artifacts_root: str = "docs/artifacts"):
        self.artifacts_root = Path(artifacts_root)

        # Valid categories
        self.valid_categories = [
            "development",
            "architecture",
            "evaluation",
            "compliance",
            "reference",
            "planning",
            "research",
            "troubleshooting",
        ]

        # Valid types
        self.valid_types = [
            "implementation_plan",
            "assessment",
            "design",
            "research",
            "template",
            "bug_report",
            "session_note",
            "completion_summary",
        ]

        # Category mapping for common invalid values
        self.category_mapping = {
            # Common misspellings and variations
            "dev": "development",
            "devlopment": "development",
            "developement": "development",
            "programming": "development",
            "coding": "development",
            "implementation": "development",
            "integration": "development",
            "api": "development",
            "backend": "development",
            "frontend": "development",
            # Architecture related
            "arch": "architecture",
            "architectural": "architecture",
            "design_doc": "architecture",
            "system_design": "architecture",
            "technical_design": "architecture",
            "component_design": "architecture",
            "interface_design": "architecture",
            "framework": "architecture",
            "pattern": "architecture",
            "model": "architecture",
            # Evaluation related
            "eval": "evaluation",
            "evaluaton": "evaluation",
            "assessment": "evaluation",
            "analysis": "evaluation",
            "review": "evaluation",
            "audit": "evaluation",
            "performance": "evaluation",
            "quality": "evaluation",
            "metrics": "evaluation",
            "benchmark": "evaluation",
            "testing": "evaluation",
            "test": "evaluation",
            # Compliance related
            "compl": "compliance",
            "validation": "compliance",
            "check": "compliance",
            "standard": "compliance",
            "policy": "compliance",
            "requirement": "compliance",
            "convention": "compliance",
            "rule": "compliance",
            "guideline": "compliance",
            "naming-conventions": "compliance",
            "naming_conventions": "compliance",
            # Reference related
            "ref": "reference",
            "doc": "reference",
            "documentation": "reference",
            "guide": "reference",
            "manual": "reference",
            "tutorial": "reference",
            "example": "reference",
            "sample": "reference",
            "boilerplate": "reference",
            "template": "reference",
            "format": "reference",
            "structure": "reference",
            "historical": "reference",
            "completed": "reference",
            "archive": "reference",
            # Planning related
            "plan": "planning",
            "roadmap": "planning",
            "timeline": "planning",
            "milestone": "planning",
            "schedule": "planning",
            "project": "planning",
            "strategy": "planning",
            "goal": "planning",
            "objective": "planning",
            "task": "planning",
            "phase": "planning",
            "iteration": "planning",
            "sprint": "planning",
            # Research related
            "res": "research",
            "investigation": "research",
            "study": "research",
            "exploration": "research",
            "discovery": "research",
            "experiment": "research",
            "hypothesis": "research",
            "findings": "research",
            "methodology": "research",
            "survey": "research",
            "interview": "research",
            # Troubleshooting related
            "troubleshoot": "troubleshooting",
            "debug": "troubleshooting",
            "fix": "troubleshooting",
            "issue": "troubleshooting",
            "problem": "troubleshooting",
            "error": "troubleshooting",
            "bug": "troubleshooting",
            "resolution": "troubleshooting",
            "bug_fix": "troubleshooting",
            "patch": "troubleshooting",
            "hotfix": "troubleshooting",
            "support": "troubleshooting",
        }

        # Type mapping for common invalid values
        self.type_mapping = {
            # Implementation plan variations
            "implementation_guide": "implementation_plan",
            "implementation_plan": "implementation_plan",
            "development_plan": "implementation_plan",
            "project_plan": "implementation_plan",
            "technical_plan": "implementation_plan",
            "architecture_plan": "implementation_plan",
            "blueprint": "implementation_plan",
            "roadmap": "implementation_plan",
            "migration_plan": "implementation_plan",
            "deployment_plan": "implementation_plan",
            "phase_plan": "implementation_plan",
            "milestone_plan": "implementation_plan",
            # Assessment variations
            "evaluation": "assessment",
            "audit": "assessment",
            "review": "assessment",
            "analysis": "assessment",
            "compliance_check": "assessment",
            "quality_assessment": "assessment",
            "performance_review": "assessment",
            "risk_assessment": "assessment",
            "security_audit": "assessment",
            "code_review": "assessment",
            "technical_review": "assessment",
            # Design variations
            "design_document": "design",
            "architecture": "design",
            "technical_design": "design",
            "system_design": "design",
            "component_design": "design",
            "interface_design": "design",
            "api_design": "design",
            "database_design": "design",
            "user_interface": "design",
            "ui_design": "design",
            "ux_design": "design",
            # Research variations
            "investigation": "research",
            "study": "research",
            "findings": "research",
            "exploration": "research",
            "discovery": "research",
            "experiment": "research",
            "hypothesis": "research",
            "methodology": "research",
            "survey": "research",
            "interview": "research",
            # Template variations
            "example": "template",
            "sample": "template",
            "boilerplate": "template",
            "reference": "template",
            "guide": "template",
            "how_to": "template",
            "checklist": "template",
            "format": "template",
            "structure": "template",
            # Bug report variations
            "issue": "bug_report",
            "problem": "bug_report",
            "error": "bug_report",
            "defect": "bug_report",
            "troubleshooting": "bug_report",
            "fix": "bug_report",
            "resolution": "bug_report",
            "bug_fix": "bug_report",
            "patch": "bug_report",
            "hotfix": "bug_report",
            # Session note variations
            "session": "session_note",
            "meeting": "session_note",
            "notes": "session_note",
            "discussion": "session_note",
            "conversation": "session_note",
            "chat": "session_note",
            "log": "session_note",
            # Completion summary variations
            "completion": "completion_summary",
            "summary": "completion_summary",
            "wrap_up": "completion_summary",
            "final": "completion_summary",
            "end": "completion_summary",
            "conclusion": "completion_summary",
            "results": "completion_summary",
            "outcome": "completion_summary",
        }

    def analyze_file(self, file_path: Path) -> list[CategoryFix]:
        """Analyze file for invalid category/type values"""
        fixes = []

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return fixes

        # Extract frontmatter
        frontmatter_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
        if not frontmatter_match:
            return fixes

        frontmatter_content = frontmatter_match.group(1)

        # Check category
        category_fix = self._check_category(file_path, frontmatter_content)
        if category_fix:
            fixes.append(category_fix)

        # Check type
        type_fix = self._check_type(file_path, frontmatter_content)
        if type_fix:
            fixes.append(type_fix)

        return fixes

    def _check_category(
        self, file_path: Path, frontmatter_content: str
    ) -> CategoryFix | None:
        """Check for invalid category value"""
        category_match = re.search(
            r'category:\s*["\']?([^"\'\n]+)["\']?', frontmatter_content
        )
        if not category_match:
            return None

        current_category = category_match.group(1).strip()

        # Check if category is valid
        if current_category in self.valid_categories:
            return None

        # Try to map to valid category
        mapped_category = self.category_mapping.get(current_category.lower())
        if mapped_category:
            return CategoryFix(
                file_path=str(file_path),
                field="category",
                old_value=current_category,
                new_value=mapped_category,
                reason=f"Mapped '{current_category}' to '{mapped_category}'",
                confidence=0.9,
            )

        # Try to suggest based on directory structure
        suggested_category = self._suggest_category_from_directory(file_path)
        if suggested_category:
            return CategoryFix(
                file_path=str(file_path),
                field="category",
                old_value=current_category,
                new_value=suggested_category,
                reason=f"Suggested '{suggested_category}' based on directory structure",
                confidence=0.7,
            )

        # Default fallback
        return CategoryFix(
            file_path=str(file_path),
            field="category",
            old_value=current_category,
            new_value="reference",
            reason="Default fallback to 'reference'",
            confidence=0.5,
        )

    def _check_type(
        self, file_path: Path, frontmatter_content: str
    ) -> CategoryFix | None:
        """Check for invalid type value"""
        type_match = re.search(r'type:\s*["\']?([^"\'\n]+)["\']?', frontmatter_content)
        if not type_match:
            return None

        current_type = type_match.group(1).strip()

        # Check if type is valid
        if current_type in self.valid_types:
            return None

        # Try to map to valid type
        mapped_type = self.type_mapping.get(current_type.lower())
        if mapped_type:
            return CategoryFix(
                file_path=str(file_path),
                field="type",
                old_value=current_type,
                new_value=mapped_type,
                reason=f"Mapped '{current_type}' to '{mapped_type}'",
                confidence=0.9,
            )

        # Try to suggest based on directory structure
        suggested_type = self._suggest_type_from_directory(file_path)
        if suggested_type:
            return CategoryFix(
                file_path=str(file_path),
                field="type",
                old_value=current_type,
                new_value=suggested_type,
                reason=f"Suggested '{suggested_type}' based on directory structure",
                confidence=0.7,
            )

        # Default fallback
        return CategoryFix(
            file_path=str(file_path),
            field="type",
            old_value=current_type,
            new_value="template",
            reason="Default fallback to 'template'",
            confidence=0.5,
        )

    def _suggest_category_from_directory(self, file_path: Path) -> str | None:
        """Suggest category based on directory structure"""
        relative_path = file_path.relative_to(self.artifacts_root)
        directory = str(relative_path.parent)

        directory_category_mapping = {
            "implementation_plans": "development",
            "assessments": "evaluation",
            "design_documents": "architecture",
            "research": "research",
            "templates": "reference",
            "bug_reports": "troubleshooting",
            "session_notes": "planning",
            "completion_summaries": "reference",
        }

        return directory_category_mapping.get(directory)

    def _suggest_type_from_directory(self, file_path: Path) -> str | None:
        """Suggest type based on directory structure"""
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
            "completion_summaries": "completion_summary",
        }

        return directory_type_mapping.get(directory)

    def apply_fix(self, fix: CategoryFix, dry_run: bool = False) -> bool:
        """Apply a category/type fix to a file"""
        if dry_run:
            print(f"[DRY RUN] Would fix {fix.field} in {fix.file_path}")
            print(f"         {fix.old_value} -> {fix.new_value}")
            print(f"         Reason: {fix.reason}")
            return True

        try:
            # Read file
            with open(fix.file_path, encoding="utf-8") as f:
                content = f.read()

            # Create pattern for the field
            if fix.field == "category":
                pattern = r'category:\s*["\']?[^"\'\n]+["\']?'
                replacement = f'category: "{fix.new_value}"'
            elif fix.field == "type":
                pattern = r'type:\s*["\']?[^"\'\n]+["\']?'
                replacement = f'type: "{fix.new_value}"'
            else:
                return False

            # Apply replacement
            new_content = re.sub(pattern, replacement, content)

            if new_content != content:
                # Write updated content
                with open(fix.file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)

                print(f"✅ Fixed {fix.field} in {fix.file_path}")
                print(f"   {fix.old_value} -> {fix.new_value}")
                print(f"   Reason: {fix.reason}")
                return True
            else:
                print(f"⚠️  No changes made to {fix.file_path}")
                return False

        except Exception as e:
            print(f"❌ Failed to fix {fix.file_path}: {e}")
            return False

    def fix_file(self, file_path: Path, dry_run: bool = False) -> list[CategoryFix]:
        """Fix category/type issues for a single file"""
        fixes = self.analyze_file(file_path)

        if not fixes:
            print(f"✅ No category/type issues found for {file_path}")
            return fixes

        print(f"🔧 Found {len(fixes)} category/type issues for {file_path}")

        for fix in fixes:
            self.apply_fix(fix, dry_run)

        return fixes

    def fix_directory(
        self, directory: Path, dry_run: bool = False
    ) -> dict[str, list[CategoryFix]]:
        """Fix category/type issues for all files in a directory"""
        results = {}

        for file_path in directory.rglob("*.md"):
            if file_path.is_file() and file_path.name != "INDEX.md":
                fixes = self.fix_file(file_path, dry_run)
                if fixes:
                    results[str(file_path)] = fixes

        return results

    def generate_fix_report(self, results: dict[str, list[CategoryFix]]) -> str:
        """Generate a comprehensive fix report"""
        report = []
        report.append("🔧 Category/Type Fix Report")
        report.append("=" * 50)

        total_files = len(results)
        total_fixes = sum(len(fixes) for fixes in results.values())

        # Count fixes by field
        category_fixes = sum(
            1 for fixes in results.values() for fix in fixes if fix.field == "category"
        )
        type_fixes = sum(
            1 for fixes in results.values() for fix in fixes if fix.field == "type"
        )

        report.append(f"Files processed: {total_files}")
        report.append(f"Total fixes: {total_fixes}")
        report.append(f"Category fixes: {category_fixes}")
        report.append(f"Type fixes: {type_fixes}")
        report.append("")

        if results:
            report.append("Files with fixes applied:")
            report.append("-" * 30)

            for file_path, fixes in results.items():
                report.append(f"\n📁 {file_path}")
                for fix in fixes:
                    report.append(
                        f"   • {fix.field}: {fix.old_value} -> {fix.new_value}"
                    )
                    report.append(
                        f"     Reason: {fix.reason} (confidence: {fix.confidence:.1f})"
                    )
        else:
            report.append("✅ No category/type issues found!")

        return "\n".join(report)

    def validate_all_categories_types(self, directory: Path) -> dict[str, list[str]]:
        """Validate all category/type values in directory"""
        issues = {}

        for file_path in directory.rglob("*.md"):
            if file_path.is_file() and file_path.name != "INDEX.md":
                file_issues = []

                try:
                    with open(file_path, encoding="utf-8") as f:
                        content = f.read()

                    # Check frontmatter
                    frontmatter_match = re.match(
                        r"^---\n(.*?)\n---", content, re.DOTALL
                    )
                    if frontmatter_match:
                        frontmatter_content = frontmatter_match.group(1)

                        # Check category
                        category_match = re.search(
                            r'category:\s*["\']?([^"\'\n]+)["\']?', frontmatter_content
                        )
                        if category_match:
                            category = category_match.group(1).strip()
                            if category not in self.valid_categories:
                                file_issues.append(f"Invalid category: '{category}'")

                        # Check type
                        type_match = re.search(
                            r'type:\s*["\']?([^"\'\n]+)["\']?', frontmatter_content
                        )
                        if type_match:
                            file_type = type_match.group(1).strip()
                            if file_type not in self.valid_types:
                                file_issues.append(f"Invalid type: '{file_type}'")

                except Exception as e:
                    file_issues.append(f"Error reading file: {e}")

                if file_issues:
                    issues[str(file_path)] = file_issues

        return issues


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Fix invalid category and type values")
    parser.add_argument("--file", help="Fix specific file")
    parser.add_argument(
        "--directory", default="docs/artifacts", help="Directory to process"
    )
    parser.add_argument(
        "--auto-correct", action="store_true", help="Apply automatic corrections"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fixed without making changes",
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="Only validate, do not fix"
    )
    parser.add_argument("--output", help="Output file for report")
    parser.add_argument(
        "--artifacts-root",
        default="docs/artifacts",
        help="Root directory for artifacts",
    )

    args = parser.parse_args()

    fixer = CategoryTypeFixer(args.artifacts_root)

    if args.validate_only:
        # Validation only mode
        directory = Path(args.directory)
        issues = fixer.validate_all_categories_types(directory)

        if issues:
            print("❌ Invalid category/type values found:")
            for file_path, file_issues in issues.items():
                print(f"\n📁 {file_path}")
                for issue in file_issues:
                    print(f"   • {issue}")
        else:
            print("✅ All category/type values are valid!")
        return

    if args.file:
        # Process specific file
        file_path = Path(args.file)
        fixes = fixer.fix_file(file_path, dry_run=args.dry_run)
        results = {str(file_path): fixes} if fixes else {}
    else:
        # Process directory
        directory = Path(args.directory)
        results = fixer.fix_directory(directory, dry_run=args.dry_run)

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
