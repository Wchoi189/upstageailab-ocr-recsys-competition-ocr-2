#!/usr/bin/env python3
"""
Export Framework Utility

Automated export tool for creating project-agnostic AI agent framework packages.
Excludes project-specific content and Streamlit UI dashboard features.

Usage:
    python export_framework.py --output export_package/
    python export_framework.py --output export_package/ --validate
    python export_framework.py --output export_package/ --dry-run
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    import yaml  # noqa: F401
except ImportError:
    print(
        "ERROR: PyYAML not installed. Install with: pip install pyyaml", file=sys.stderr
    )
    sys.exit(1)


class FrameworkExporter:
    """Exports AI agent framework to project-agnostic package."""

    # Files/directories to exclude (project-specific or Streamlit UI)
    EXCLUDE_PATTERNS = [
        # Streamlit UI dashboard features
        "compliance/compliance_dashboard.py",
        "documentation/validate_coordinate_consistency.py",
        # Project-specific paths
        "streamlit_app",
        "__pycache__",
        "*.pyc",
        ".pytest_cache",
        ".mypy_cache",
        ".git",
        ".vscode",
        ".idea",
        # Project-specific data
        "data/",
        "outputs/",
        "logs/",
        "results/",
        "visualizations/",
        "experiments/",
        "backups/",
        "dist/",
        "build/",
        "htmlcov/",
        "node_modules/",
        # Project-specific config
        "project_config.yaml",
        "project_version.yaml",
        "setup-utility.log",
        "test_tone.wav",
        "output.mp3",
    ]

    # Files to exclude by name pattern
    EXCLUDE_FILES = [
        "compliance_dashboard.py",  # Streamlit UI dashboard
        "validate_coordinate_consistency.py",  # Streamlit UI validation
    ]

    def __init__(self, project_root: Path, output_dir: Path):
        self.project_root = Path(project_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.stats = {
            "files_copied": 0,
            "files_excluded": 0,
            "directories_created": 0,
            "errors": [],
        }

    def should_exclude(self, path: Path) -> bool:
        """Check if path should be excluded from export."""
        relative_path = path.relative_to(self.project_root)

        # Exclude the output directory itself to prevent recursion
        try:
            if self.output_dir.resolve() in path.resolve().parents or path.resolve() == self.output_dir.resolve():
                return True
        except (ValueError, OSError):
            # If paths can't be compared (different drives, etc.), check by name
            if path.name == self.output_dir.name and str(self.output_dir) in str(path):
                return True

        # Check exclude patterns
        for pattern in self.EXCLUDE_PATTERNS:
            if pattern in str(relative_path) or path.match(pattern):
                return True

        # Check exclude files
        if path.name in self.EXCLUDE_FILES:
            return True

        # Check if path contains streamlit_app
        return "streamlit_app" in str(relative_path)

    def copy_directory(self, src: Path, dst: Path, dry_run: bool = False) -> None:
        """Copy directory structure, excluding specified patterns."""
        if not src.exists():
            self.stats["errors"].append(f"Source does not exist: {src}")
            return

        if self.should_exclude(src):
            if not dry_run:
                print(f"⏭️  Excluding: {src.relative_to(self.project_root)}")
            self.stats["files_excluded"] += 1
            return

        # Create destination directory
        if not dry_run:
            dst.mkdir(parents=True, exist_ok=True)
            self.stats["directories_created"] += 1

        # Copy files and subdirectories
        try:
            for item in src.iterdir():
                item_dst = dst / item.name

                if self.should_exclude(item):
                    if not dry_run:
                        print(f"⏭️  Excluding: {item.relative_to(self.project_root)}")
                    self.stats["files_excluded"] += 1
                    continue

                if item.is_dir():
                    self.copy_directory(item, item_dst, dry_run=dry_run)
                elif item.is_file():
                    if not dry_run:
                        shutil.copy2(item, item_dst)
                        self.stats["files_copied"] += 1
                    else:
                        print(f"📄 Would copy: {item.relative_to(self.project_root)}")
                        self.stats["files_copied"] += 1

        except Exception as e:
            error_msg = f"Error copying {src}: {e}"
            self.stats["errors"].append(error_msg)
            if not dry_run:
                print(f"❌ {error_msg}")

    def copy_file(self, src: Path, dst: Path, dry_run: bool = False) -> None:
        """Copy a single file."""
        if not src.exists():
            self.stats["errors"].append(f"Source does not exist: {src}")
            return

        if self.should_exclude(src):
            if not dry_run:
                print(f"⏭️  Excluding: {src.relative_to(self.project_root)}")
            self.stats["files_excluded"] += 1
            return

        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            self.stats["files_copied"] += 1
        else:
            print(f"📄 Would copy: {src.relative_to(self.project_root)}")
            self.stats["files_copied"] += 1

    def export_agent_tools(self, dry_run: bool = False) -> None:
        """Export scripts/agent_tools/ directory."""
        src = self.project_root / "scripts" / "agent_tools"
        dst = self.output_dir / "agent_tools"

        print("\n📦 Exporting agent tools...")
        print(f"   From: {src.relative_to(self.project_root)}")
        print(f"   To: {dst}")

        self.copy_directory(src, dst, dry_run=dry_run)

    def export_agent_interface(self, dry_run: bool = False) -> None:
        """Export agent_interface/ directory (interface layer)."""
        src = self.project_root / "agent_interface"
        dst = self.output_dir / "agent_interface"

        print("\n📦 Exporting agent interface...")
        print(f"   From: {src.relative_to(self.project_root)}")
        print(f"   To: {dst}")

        # Exclude project-specific agent data
        exclude_in_agent = ["data", "artifacts_violations_history.json", "8"]

        if not dry_run:
            dst.mkdir(parents=True, exist_ok=True)

        for item in src.iterdir():
            # Exclude the output directory to prevent recursion
            if item.resolve() == self.output_dir.resolve():
                if not dry_run:
                    print(f"⏭️  Excluding output directory: {item.relative_to(self.project_root)}")
                self.stats["files_excluded"] += 1
                continue

            if item.name in exclude_in_agent:
                if not dry_run:
                    print(f"⏭️  Excluding: {item.relative_to(self.project_root)}")
                self.stats["files_excluded"] += 1
                continue

            item_dst = dst / item.name
            if item.is_dir():
                self.copy_directory(item, item_dst, dry_run=dry_run)
            elif item.is_file():
                self.copy_file(item, item_dst, dry_run=dry_run)

    def export_ai_handbook(self, dry_run: bool = False) -> None:
        """Export docs/ai_handbook/ directory."""
        src = self.project_root / "docs" / "ai_handbook"
        dst = self.output_dir / "ai_handbook"

        print("\n📦 Exporting AI handbook...")
        print(f"   From: {src.relative_to(self.project_root)}")
        print(f"   To: {dst}")

        self.copy_directory(src, dst, dry_run=dry_run)

    def export_ai_agent_docs(self, dry_run: bool = False) -> None:
        """Export docs/ai_handbook/04_agent_system/ directory."""
        src = self.project_root / "docs" / "ai_handbook" / "04_agent_system"
        dst = self.output_dir / "ai_handbook" / "04_agent_system"

        print("\n📦 Exporting AI agent system documentation...")
        if src.exists():
            print(f"   From: {src.relative_to(self.project_root)}")
            print(f"   To: {dst}")
            self.copy_directory(src, dst, dry_run=dry_run)
        else:
            print(f"⚠️  Source does not exist: {src}")

    def export_templates(self, dry_run: bool = False) -> None:
        """Export template directories."""
        template_locations = [
            ("docs/ai_handbook/templates", "ai_handbook/templates"),
            ("docs/artifacts/templates", "docs/artifacts/templates"),
            ("docs/artifacts/templates/agent_workflows", "docs/artifacts/templates/agent_workflows"),
        ]

        print("\n📦 Exporting templates...")

        for src_rel, dst_rel in template_locations:
            src = self.project_root / src_rel
            dst = self.output_dir / dst_rel

            if src.exists():
                print(f"   From: {src_rel}")
                print(f"   To: {dst_rel}")
                self.copy_directory(src, dst, dry_run=dry_run)
            else:
                print(f"⚠️  Template location does not exist: {src_rel}")

    def export_adaptation_tool(self, dry_run: bool = False) -> None:
        """Export adaptation tool to scripts/ directory."""
        src = (
            self.project_root
            / "scripts"
            / "agent_tools"
            / "utilities"
            / "adapt_project.py"
        )
        dst = self.output_dir / "scripts" / "adapt_project.py"

        print("\n📦 Exporting adaptation tool...")
        print(f"   From: {src.relative_to(self.project_root)}")
        print(f"   To: {dst}")

        if src.exists():
            self.copy_file(src, dst, dry_run=dry_run)
        else:
            print(f"⚠️  Adaptation tool does not exist: {src}")

    def export_documentation(self, dry_run: bool = False) -> None:
        """Export export documentation."""
        export_docs = [
            "docs/user_guides/export_documentation/export_guide.md",
            "docs/user_guides/export_documentation/quick_start_export.md",
            "docs/user_guides/export_documentation/resources.md",
        ]

        print("\n📦 Exporting export documentation...")

        for doc_rel in export_docs:
            src = self.project_root / doc_rel
            dst = self.output_dir / "docs" / Path(doc_rel).name

            if src.exists():
                print(f"   Copying: {Path(doc_rel).name}")
                self.copy_file(src, dst, dry_run=dry_run)
            else:
                print(f"⚠️  Documentation does not exist: {doc_rel}")

    def create_readme(self, dry_run: bool = False) -> None:
        """Create README for export package."""
        readme_content = """# AI Agent Framework Export

This package contains the AI collaboration and documentation management framework.

## Contents

- `agent_tools/` - Implementation layer (Python automation scripts)
- `agent_interface/` - Interface layer (Makefile, wrappers, config)
- `ai_handbook/` - AI agent documentation and protocols (includes \`04_agent_system/\`)
- `scripts/` - Adaptation scripts
- `docs/` - Export documentation

## Installation

See `docs/export_guide.md` for detailed installation instructions.

## Quick Start

1. Copy framework directories to your project:
   ```bash
   cp -r agent_tools/ your_project/scripts/
   cp -r agent_interface/ your_project/
   cp -r ai_handbook/ your_project/docs/
   ```

2. Run adaptation script:
   ```bash
   python scripts/adapt_project.py --interactive
   ```

3. Verify installation:
   ```bash
   cd AgentQMS/agent_interface/
   make discover
   make status
   ```

## Excluded Components

The following project-specific components are excluded from this export:
- Streamlit UI dashboard features
- Project-specific data and outputs
- Project-specific configuration files
- Development artifacts (logs, caches, etc.)

## License

[Add your license information here]
"""

        readme_path = self.output_dir / "README.md"
        if not dry_run:
            readme_path.write_text(readme_content, encoding="utf-8")
            print("\n📝 Created README.md")
        else:
            print("\n📝 Would create README.md")

    def validate_export(self) -> dict[str, Any]:
        """Validate export completeness."""
        validation_results = {
            "passed": True,
            "issues": [],
            "warnings": [],
        }

        # Check required directories
        required_dirs = [
            "agent_tools",
            "agent_interface",
            "ai_handbook",
            "ai_handbook/04_agent_system",
        ]

        for dir_name in required_dirs:
            dir_path = self.output_dir / dir_name
            if not dir_path.exists():
                validation_results["passed"] = False
                validation_results["issues"].append(
                    f"Missing required directory: {dir_name}"
                )

        # Check required files
        required_files = [
            "agent_tools/core/artifact_workflow.py",
            "agent_tools/core/discover.py",
            "agent_interface/Makefile",
            "ai_handbook/index.md",
            "ai_handbook/04_agent_system/system.md",
            "scripts/adapt_project.py",
        ]

        for file_rel in required_files:
            file_path = self.output_dir / file_rel
            if not file_path.exists():
                validation_results["passed"] = False
                validation_results["issues"].append(
                    f"Missing required file: {file_rel}"
                )

        # Check for excluded files (should not be present)
        excluded_files = [
            "agent_tools/compliance/compliance_dashboard.py",
            "agent_tools/documentation/validate_coordinate_consistency.py",
        ]

        for file_rel in excluded_files:
            file_path = self.output_dir / file_rel
            if file_path.exists():
                validation_results["warnings"].append(
                    f"Excluded file found in export: {file_rel}"
                )

        return validation_results

    def export_all(self, dry_run: bool = False, validate: bool = False) -> None:
        """Export entire framework."""
        print("🚀 Starting framework export...")
        print(f"   Project root: {self.project_root}")
        print(f"   Output directory: {self.output_dir}")
        if dry_run:
            print("   Mode: DRY RUN (no files will be copied)")
        print()

        # Create output directory
        if not dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Export components
        self.export_agent_tools(dry_run=dry_run)
        self.export_agent_interface(dry_run=dry_run)
        self.export_ai_handbook(dry_run=dry_run)
        self.export_ai_agent_docs(dry_run=dry_run)
        self.export_templates(dry_run=dry_run)
        self.export_adaptation_tool(dry_run=dry_run)
        self.export_documentation(dry_run=dry_run)
        self.create_readme(dry_run=dry_run)

        # Print summary
        print("\n" + "=" * 60)
        print("📊 Export Summary")
        print("=" * 60)
        print(f"   Files copied: {self.stats['files_copied']}")
        print(f"   Files excluded: {self.stats['files_excluded']}")
        print(f"   Directories created: {self.stats['directories_created']}")
        if self.stats["errors"]:
            print(f"   Errors: {len(self.stats['errors'])}")
            for error in self.stats["errors"]:
                print(f"      - {error}")

        # Validate if requested
        if validate and not dry_run:
            print("\n" + "=" * 60)
            print("🔍 Validating Export")
            print("=" * 60)
            validation = self.validate_export()
            if validation["passed"]:
                print("✅ Export validation passed!")
            else:
                print("❌ Export validation failed!")
                for issue in validation["issues"]:
                    print(f"   - {issue}")
            if validation["warnings"]:
                print("\n⚠️  Warnings:")
                for warning in validation["warnings"]:
                    print(f"   - {warning}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export AI agent framework to project-agnostic package",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export to export_package directory
  python export_framework.py --output export_package/

  # Dry run (see what would be exported)
  python export_framework.py --output export_package/ --dry-run

  # Export with validation
  python export_framework.py --output export_package/ --validate
        """,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for export package",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=".",
        help="Project root directory (default: current directory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be exported without copying files",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate export completeness after export",
    )

    args = parser.parse_args()

    exporter = FrameworkExporter(args.project_root, args.output)
    exporter.export_all(dry_run=args.dry_run, validate=args.validate)

    return 0 if exporter.stats["errors"] == [] else 1


if __name__ == "__main__":
    sys.exit(main())
