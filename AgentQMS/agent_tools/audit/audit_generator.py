#!/usr/bin/env python3
"""
Audit Framework Document Generator

Generates audit documents from templates by replacing placeholders.

Usage:
    python audit_generator.py init --framework-name "Framework Name" --audit-date "2025-11-09"
    python audit_generator.py generate --template "00_audit_summary_template.md" --output "docs/audit/00_audit_summary.md"
"""

import argparse
import re
from pathlib import Path

from AgentQMS.agent_tools.utils.paths import get_project_conventions_dir
from AgentQMS.agent_tools.utils.runtime import ensure_project_root_on_sys_path

ensure_project_root_on_sys_path()


def get_templates_dir() -> Path:
    """Get the audit framework templates directory."""
    templates_dir = get_project_conventions_dir() / "audit_framework" / "templates"
    if templates_dir.exists():
        return templates_dir
    raise RuntimeError(f"Templates directory not found: {templates_dir}")


def load_template(template_name: str) -> str:
    """
    Load a template file.

    Args:
        template_name: Name of the template file (e.g., "00_audit_summary_template.md")

    Returns:
        Template content as string
    """
    templates_dir = get_templates_dir()
    template_path = templates_dir / template_name

    if not template_path.exists():
        available = [f.name for f in templates_dir.glob("*_template.md")]
        raise FileNotFoundError(
            f"Template not found: {template_name}\n"
            f"Available templates: {', '.join(available)}"
        )

    return template_path.read_text(encoding="utf-8")


def replace_placeholders(content: str, values: dict[str, str]) -> str:
    """
    Replace placeholders in template content.

    Args:
        content: Template content with placeholders like {{PLACEHOLDER_NAME}}
        values: Dictionary mapping placeholder names to values

    Returns:
        Content with placeholders replaced
    """
    result = content

    # Find all placeholders in the format {{PLACEHOLDER_NAME}}
    placeholders = re.findall(r'\{\{([A-Z_]+)\}\}', content)

    for placeholder in placeholders:
        if placeholder not in values:
            raise ValueError(
                f"Missing value for placeholder: {placeholder}\n"
                f"Required placeholders: {', '.join(set(placeholders))}\n"
                f"Provided values: {', '.join(values.keys())}"
            )
        result = result.replace(f"{{{{{placeholder}}}}}", values[placeholder])

    return result


def generate_document(
    template_name: str,
    values: dict[str, str],
    output_path: Path,
    overwrite: bool = False
) -> Path:
    """
    Generate an audit document from a template.

    Args:
        template_name: Name of the template file
        values: Dictionary of placeholder values
        output_path: Path where document should be written
        overwrite: Whether to overwrite existing file

    Returns:
        Path to generated document
    """
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}\n"
            f"Use --overwrite to replace it"
        )

    # Load template
    template_content = load_template(template_name)

    # Replace placeholders
    document_content = replace_placeholders(template_content, values)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write document
    output_path.write_text(document_content, encoding="utf-8")

    print(f"✅ Generated: {output_path}")
    return output_path


def init_audit(
    framework_name: str,
    audit_date: str,
    audit_scope: str,
    output_dir: Path,
    status: str = "Draft",
    overwrite: bool = False
) -> None:
    """
    Initialize a complete audit by generating all documents.

    Args:
        framework_name: Name of the framework being audited
        audit_date: Date of audit (YYYY-MM-DD)
        audit_scope: Scope description
        output_dir: Directory where audit documents should be created
        status: Initial status for documents
        overwrite: Whether to overwrite existing files
    """
    templates_dir = get_templates_dir()
    templates = sorted(templates_dir.glob("*_template.md"))

    if not templates:
        raise RuntimeError(f"No templates found in {templates_dir}")

    # Common values for all documents
    common_values = {
        "FRAMEWORK_NAME": framework_name,
        "AUDIT_DATE": audit_date,
        "AUDIT_SCOPE": audit_scope,
        "STATUS": status,
    }

    print(f"📋 Initializing audit for: {framework_name}")
    print(f"📅 Date: {audit_date}")
    print(f"📁 Output directory: {output_dir}")
    print()

    generated = []

    for template_path in templates:
        template_name = template_path.name
        # Generate output filename by removing "_template" from name
        output_name = template_name.replace("_template", "")
        output_path = output_dir / output_name

        try:
            generate_document(template_name, common_values, output_path, overwrite)
            generated.append(output_name)
        except FileExistsError as e:
            print(f"⚠️  Skipped (already exists): {output_name}")
            if not overwrite:
                print(f"   {e}")

    print()
    print(f"✅ Generated {len(generated)} audit documents:")
    for doc in generated:
        print(f"   - {doc}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate audit framework documents from templates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize complete audit
  python audit_generator.py init \\
      --framework-name "My Framework" \\
      --audit-date "2025-11-09" \\
      --audit-scope "Complete Framework Audit" \\
      --output-dir "docs/audit"

  # Generate single document
  python audit_generator.py generate \\
      --template "00_audit_summary_template.md" \\
      --framework-name "My Framework" \\
      --audit-date "2025-11-09" \\
      --output "docs/audit/00_audit_summary.md"
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize complete audit")
    init_parser.add_argument(
        "--framework-name",
        required=True,
        help="Name of the framework being audited"
    )
    init_parser.add_argument(
        "--audit-date",
        required=True,
        help="Date of audit (YYYY-MM-DD)"
    )
    init_parser.add_argument(
        "--audit-scope",
        required=True,
        help="Scope description of the audit"
    )
    init_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/audit"),
        help="Directory where audit documents should be created (default: docs/audit)"
    )
    init_parser.add_argument(
        "--status",
        default="Draft",
        help="Initial status for documents (default: Draft)"
    )
    init_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files"
    )

    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate single document")
    generate_parser.add_argument(
        "--template",
        required=True,
        help="Template file name (e.g., 00_audit_summary_template.md)"
    )
    generate_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output file path"
    )
    generate_parser.add_argument(
        "--framework-name",
        help="Framework name (for {{FRAMEWORK_NAME}} placeholder)"
    )
    generate_parser.add_argument(
        "--audit-date",
        help="Audit date (for {{AUDIT_DATE}} placeholder)"
    )
    generate_parser.add_argument(
        "--audit-scope",
        help="Audit scope (for {{AUDIT_SCOPE}} placeholder)"
    )
    generate_parser.add_argument(
        "--status",
        help="Status (for {{STATUS}} placeholder)"
    )
    generate_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing file"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "init":
            init_audit(
                framework_name=args.framework_name,
                audit_date=args.audit_date,
                audit_scope=args.audit_scope,
                output_dir=args.output_dir,
                status=args.status,
                overwrite=args.overwrite
            )
        elif args.command == "generate":
            values = {}
            if args.framework_name:
                values["FRAMEWORK_NAME"] = args.framework_name
            if args.audit_date:
                values["AUDIT_DATE"] = args.audit_date
            if args.audit_scope:
                values["AUDIT_SCOPE"] = args.audit_scope
            if args.status:
                values["STATUS"] = args.status

            generate_document(
                template_name=args.template,
                values=values,
                output_path=args.output,
                overwrite=args.overwrite
            )
    except (FileNotFoundError, ValueError, FileExistsError, RuntimeError) as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

