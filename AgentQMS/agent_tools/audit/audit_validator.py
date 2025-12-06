#!/usr/bin/env python3
"""
Audit Framework Document Validator

Validates audit documents for completeness, structure, and required sections.

Usage:
    python audit_validator.py validate --audit-dir "docs/audit"
    python audit_validator.py validate --document "docs/audit/00_audit_summary.md"
"""

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from AgentQMS.agent_tools.utils.runtime import ensure_project_root_on_sys_path

ensure_project_root_on_sys_path()


@dataclass
class ValidationError:
    """Represents a validation error."""
    document: str
    section: str
    message: str
    severity: str = "error"  # error, warning


@dataclass
class ValidationResult:
    """Result of document validation."""
    document_path: Path
    valid: bool
    errors: list[ValidationError]
    warnings: list[ValidationError]


# Required sections for each document type
REQUIRED_SECTIONS = {
    "00_audit_summary.md": [
        "Executive Summary",
        "Key Findings",
        "Deliverables",
        "Framework Assessment",
        "Implementation Roadmap",
    ],
    "01_removal_candidates.md": [
        "Executive Summary",
        "CRITICAL: Broken Dependencies",
        "HIGH PRIORITY: Project-Specific Content",
        "Implementation Priority Matrix",
    ],
    "02_workflow_analysis.md": [
        "Executive Summary",
        "Current Workflow Map",
        "Pain Points Analysis",
        "Goal vs. Implementation Alignment",
    ],
    "03_restructure_proposal.md": [
        "Executive Summary",
        "Restructure Principles",
        "Phase 1: Critical Fixes",
        "Success Criteria",
    ],
    "04_standards_specification.md": [
        "Executive Summary",
        "Output Directory Structure",
        "File Naming Conventions",
        "Frontmatter Schemas",
    ],
    "05_automation_recommendations.md": [
        "Executive Summary",
        "Self-Enforcing Compliance",
        "Automated Validation",
        "Implementation Plan",
    ],
}


def check_required_sections(content: str, document_name: str) -> list[ValidationError]:
    """
    Check if document contains all required sections.

    Args:
        content: Document content
        document_name: Name of the document file

    Returns:
        List of validation errors for missing sections
    """
    errors = []
    required = REQUIRED_SECTIONS.get(document_name, [])

    for section in required:
        # Look for section headers (## or ###)
        pattern = rf"^#{{2,3}}\s+{re.escape(section)}"
        if not re.search(pattern, content, re.MULTILINE):
            errors.append(ValidationError(
                document=document_name,
                section=section,
                message=f"Missing required section: {section}",
                severity="error"
            ))

    return errors


def check_placeholders(content: str, document_name: str) -> list[ValidationError]:
    """
    Check for unreplaced placeholders.

    Args:
        content: Document content
        document_name: Name of the document file

    Returns:
        List of validation errors for unreplaced placeholders
    """
    errors = []
    placeholders = re.findall(r'\{\{([A-Z_]+)\}\}', content)

    for placeholder in placeholders:
        errors.append(ValidationError(
            document=document_name,
            section="Placeholders",
            message=f"Unreplaced placeholder: {{{{ {placeholder} }}}}",
            severity="error"
        ))

    return errors


DATE_FORMAT = "%Y-%m-%d %H:%M (KST)"


def check_frontmatter(content: str, document_name: str) -> list[ValidationError]:
    """
    Check if document has valid frontmatter.

    Args:
        content: Document content
        document_name: Name of the document file

    Returns:
        List of validation errors for frontmatter issues
    """
    errors = []

    # Check for frontmatter block
    if not content.startswith("---"):
        errors.append(ValidationError(
            document=document_name,
            section="Frontmatter",
            message="Missing frontmatter block (should start with ---)",
            severity="error"
        ))
        return errors

    # Extract frontmatter
    frontmatter_match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
    if not frontmatter_match:
        errors.append(ValidationError(
            document=document_name,
            section="Frontmatter",
            message="Invalid frontmatter format",
            severity="error"
        ))
        return errors

    frontmatter = frontmatter_match.group(1)

    # Check for required fields
    required_fields = ["type", "category", "title", "date"]
    for field in required_fields:
        if f"{field}:" not in frontmatter:
            errors.append(ValidationError(
                document=document_name,
                section="Frontmatter",
                message=f"Missing required frontmatter field: {field}",
                severity="error"
            ))

    date_match = re.search(r'date:\s*["\']?([^\n"\']+)["\']?', frontmatter)
    if date_match:
        date_value = date_match.group(1).strip()
        try:
            datetime.strptime(date_value, DATE_FORMAT)
        except ValueError:
            errors.append(ValidationError(
                document=document_name,
                section="Frontmatter",
                message="Date must use 'YYYY-MM-DD HH:MM (KST)' format.",
                severity="error",
            ))

    return errors


def validate_document(document_path: Path) -> ValidationResult:
    """
    Validate a single audit document.

    Args:
        document_path: Path to the document

    Returns:
        ValidationResult with validation status and errors
    """
    if not document_path.exists():
        return ValidationResult(
            document_path=document_path,
            valid=False,
            errors=[ValidationError(
                document=document_path.name,
                section="File",
                message=f"Document not found: {document_path}",
                severity="error"
            )],
            warnings=[]
        )

    content = document_path.read_text(encoding="utf-8")
    document_name = document_path.name

    errors = []

    # Check required sections
    errors.extend(check_required_sections(content, document_name))

    # Check for unreplaced placeholders
    errors.extend(check_placeholders(content, document_name))

    # Check frontmatter
    errors.extend(check_frontmatter(content, document_name))

    # Separate errors and warnings
    error_list = [e for e in errors if e.severity == "error"]
    warning_list = [e for e in errors if e.severity == "warning"]

    return ValidationResult(
        document_path=document_path,
        valid=len(error_list) == 0,
        errors=error_list,
        warnings=warning_list
    )


def validate_completeness(audit_dir: Path) -> dict:
    """
    Validate that all required audit documents exist.

    Args:
        audit_dir: Directory containing audit documents

    Returns:
        Dictionary with completeness status
    """
    required_docs = [
        "00_audit_summary.md",
        "01_removal_candidates.md",
        "02_workflow_analysis.md",
        "03_restructure_proposal.md",
        "04_standards_specification.md",
        "05_automation_recommendations.md",
    ]

    missing = []
    present = []

    for doc in required_docs:
        doc_path = audit_dir / doc
        if doc_path.exists():
            present.append(doc)
        else:
            missing.append(doc)

    return {
        "complete": len(missing) == 0,
        "present": present,
        "missing": missing,
        "total": len(required_docs),
        "found": len(present),
    }


def validate_audit(audit_dir: Path) -> None:
    """
    Validate all documents in an audit directory.

    Args:
        audit_dir: Directory containing audit documents
    """
    if not audit_dir.exists():
        print(f"❌ Audit directory not found: {audit_dir}")
        return

    print(f"🔍 Validating audit documents in: {audit_dir}")
    print()

    # Check completeness
    completeness = validate_completeness(audit_dir)
    print(f"📋 Completeness: {completeness['found']}/{completeness['total']} documents")

    if completeness["missing"]:
        print("⚠️  Missing documents:")
        for doc in completeness["missing"]:
            print(f"   - {doc}")
    print()

    # Validate each document
    all_valid = True
    total_errors = 0
    total_warnings = 0

    for doc_name in completeness["present"]:
        doc_path = audit_dir / doc_name
        result = validate_document(doc_path)

        if result.valid:
            print(f"✅ {doc_name}")
        else:
            all_valid = False
            print(f"❌ {doc_name}")
            for error in result.errors:
                print(f"   Error: {error.message}")
                total_errors += 1
            for warning in result.warnings:
                print(f"   Warning: {warning.message}")
                total_warnings += 1

    print()
    if all_valid and completeness["complete"]:
        print("✅ All documents valid and complete!")
    else:
        print("❌ Validation failed:")
        if not completeness["complete"]:
            print(f"   - {len(completeness['missing'])} missing document(s)")
        if total_errors > 0:
            print(f"   - {total_errors} error(s) found")
        if total_warnings > 0:
            print(f"   - {total_warnings} warning(s) found")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate audit framework documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate all documents in audit directory
  python audit_validator.py validate --audit-dir "docs/audit"

  # Validate single document
  python audit_validator.py validate --document "docs/audit/00_audit_summary.md"
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate audit documents")
    validate_parser.add_argument(
        "--audit-dir",
        type=Path,
        help="Directory containing audit documents"
    )
    validate_parser.add_argument(
        "--document",
        type=Path,
        help="Single document to validate"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "validate":
            if args.document:
                result = validate_document(args.document)
                if result.valid:
                    print(f"✅ {args.document.name} is valid")
                else:
                    print(f"❌ {args.document.name} has errors:")
                    for error in result.errors:
                        print(f"   - {error.message}")
            elif args.audit_dir:
                validate_audit(args.audit_dir)
            else:
                parser.error("Must specify either --audit-dir or --document")
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

