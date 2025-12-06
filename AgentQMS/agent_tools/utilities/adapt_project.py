#!/usr/bin/env python3
"""
Project Adaptation Script
Automates the process of adapting the AI agent framework to a new project.

Usage:
    python adapt_project.py --config project_config.yaml --project-root .
    python adapt_project.py --interactive  # Interactive mode
"""

import sys
from pathlib import Path

import yaml


class ProjectAdapter:
    """Handles adaptation of AI agent framework to new projects."""

    def __init__(self, project_root: Path, config: dict):
        self.project_root = Path(project_root).resolve()
        self.config = config
        self.replacements = self._build_replacements()

    def _build_replacements(self) -> dict[str, str]:
        """Build replacement dictionary from config."""
        return {
            "{{PROJECT_NAME}}": self.config.get("project", {}).get("name", ""),
            "{{PROJECT_PURPOSE}}": self.config.get("project", {}).get("purpose", ""),
            "{{PROJECT_DESCRIPTION}}": self.config.get("project", {}).get(
                "description", ""
            ),
            "{{PROJECT_ROOT}}": self.config.get("project", {})
            .get("structure", {})
            .get("root", "."),
            "Korean Grammar Correction Project": self.config.get("project", {}).get(
                "name", ""
            ),
            "Korean Grammar Error Correction": self.config.get("project", {}).get(
                "purpose", ""
            ),
            "upstage-prompt-a-thon-project": self.config.get("project", {})
            .get("structure", {})
            .get("root", "."),
        }

    def load_config(self, config_path: Path) -> dict:
        """Load project configuration from YAML file."""
        with open(config_path) as f:
            return yaml.safe_load(f)

    def replace_placeholders(self, content: str) -> str:
        """Replace placeholders in content with config values."""
        result = content
        for placeholder, value in self.replacements.items():
            if value:  # Only replace if value is not empty
                result = result.replace(placeholder, value)
        return result

    def adapt_file(self, file_path: Path, dry_run: bool = False) -> bool:
        """Adapt a single file by replacing placeholders."""
        if not file_path.exists():
            return False

        try:
            content = file_path.read_text(encoding="utf-8")
            adapted = self.replace_placeholders(content)

            if adapted != content:
                if not dry_run:
                    file_path.write_text(adapted, encoding="utf-8")
                return True
        except Exception as e:
            print(f"⚠️  Error adapting {file_path}: {e}")
            return False

        return False

    def get_files_to_adapt(self) -> list[Path]:
        """Get list of files that need adaptation."""
        files = []

        # Handbook files
        handbook_root = self.project_root / "docs" / "ai_handbook"
        if handbook_root.exists():
            files.extend(
                [
                    handbook_root / "index.md",
                    handbook_root / "01_onboarding" / "01_project_overview.md",
                    handbook_root / "01_onboarding" / "03_data_overview.md",
                    handbook_root
                    / "03_references"
                    / "development"
                    / "ai_agent_context.md",
                ]
            )

        # Agent config files
        agent_root = self.project_root / "agent"
        if agent_root.exists():
            files.extend(
                [
                    agent_root / "config" / "agent_config.yaml",
                    agent_root / "README.md",
                ]
            )

        return [f for f in files if f.exists()]

    def adapt_all(self, dry_run: bool = False) -> dict[str, int]:
        """Adapt all files in the project."""
        files_to_adapt = self.get_files_to_adapt()
        stats = {"total": len(files_to_adapt), "adapted": 0, "skipped": 0, "errors": 0}

        print(f"\n🔍 Found {stats['total']} files to adapt\n")

        for file_path in files_to_adapt:
            relative_path = file_path.relative_to(self.project_root)

            if self.adapt_file(file_path, dry_run=dry_run):
                stats["adapted"] += 1
                print(f"✅ {'[DRY RUN] ' if dry_run else ''}Adapted: {relative_path}")
            else:
                stats["skipped"] += 1
                if dry_run:
                    print(f"⏭️  [DRY RUN] No changes needed: {relative_path}")

        return stats

    def validate_paths(self) -> list[str]:
        """Validate that required paths exist."""
        issues = []

        required_paths = [
            self.project_root / "docs" / "ai_handbook",
            self.project_root / "agent",
            self.project_root / "scripts" / "agent_tools",
        ]

        for path in required_paths:
            if not path.exists():
                issues.append(f"Missing: {path.relative_to(self.project_root)}")

        return issues


def create_config_template(output_path: Path):
    """Create a template configuration file."""
    template = """# Project Configuration Template
# Fill in your project details and save as project_config.yaml

project:
  name: "Your Project Name"
  purpose: "Brief project purpose"
  description: |
    Detailed project description.
    This can be multiple lines.

    Describe what your project does,
    its goals, and key features.

  structure:
    root: "."  # Project root directory
    docs_path: "docs"
    artifacts_path: "docs/artifacts"
    scripts_path: "scripts"

  # Optional: Project-specific settings
  technology_stack:
    - "Python 3.x"
    - "Streamlit"
    # Add your technologies here

  key_files:
    - "main.py"
    # Add your key files here
"""

    output_path.write_text(template)
    print(f"✅ Created config template: {output_path}")


def interactive_setup():
    """Interactive setup wizard."""
    print("🚀 AI Agent Framework Adaptation Wizard\n")
    print("This wizard will help you configure the framework for your project.\n")

    config = {
        "project": {
            "name": input("Project Name: ").strip() or "My Project",
            "purpose": input("Project Purpose (brief): ").strip() or "Project purpose",
            "description": input(
                "Project Description (multi-line, end with empty line):\n"
            ).strip()
            or "Project description",
            "structure": {
                "root": input("Project Root Directory [.]: ").strip() or ".",
                "docs_path": input("Documentation Path [docs]: ").strip() or "docs",
                "artifacts_path": input("Artifacts Path [docs/artifacts]: ").strip()
                or "docs/artifacts",
                "scripts_path": input("Scripts Path [scripts]: ").strip() or "scripts",
            },
        }
    }

    # Save config
    config_path = Path("project_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\n✅ Configuration saved to: {config_path}\n")
    return config_path, config


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Adapt AI agent framework to a new project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python adapt_project.py --interactive

  # Using config file
  python adapt_project.py --config project_config.yaml

  # Dry run (see what would change)
  python adapt_project.py --config project_config.yaml --dry-run

  # Create config template
  python adapt_project.py --create-config
        """,
    )

    parser.add_argument(
        "--config", type=Path, help="Path to project configuration YAML file"
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
        help="Show what would be changed without making changes",
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Run interactive setup wizard"
    )
    parser.add_argument(
        "--create-config", action="store_true", help="Create a config template file"
    )

    args = parser.parse_args()

    # Create config template
    if args.create_config:
        create_config_template(Path("project_config.yaml.template"))
        return 0

    # Interactive mode
    if args.interactive:
        config_path, config = interactive_setup()
        project_root = Path(config["project"]["structure"]["root"])
    elif args.config:
        project_root = args.project_root
        adapter = ProjectAdapter(project_root, {})
        config = adapter.load_config(args.config)
    else:
        print("❌ Error: Must specify --config or --interactive")
        parser.print_help()
        return 1

    # Validate paths
    adapter = ProjectAdapter(project_root, config)
    issues = adapter.validate_paths()

    if issues:
        print("⚠️  Validation Issues:")
        for issue in issues:
            print(f"  - {issue}")
        print("\n⚠️  Some paths are missing. Adaptation may be incomplete.")
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response != "y":
            return 1

    # Adapt project
    print(f"\n📦 Adapting project: {config['project']['name']}")
    print(f"   Root: {project_root}\n")

    stats = adapter.adapt_all(dry_run=args.dry_run)

    # Summary
    print("\n📊 Adaptation Summary:")
    print(f"   Total files: {stats['total']}")
    print(f"   Adapted: {stats['adapted']}")
    print(f"   Skipped: {stats['skipped']}")
    print(f"   Errors: {stats['errors']}")

    if args.dry_run:
        print("\n💡 This was a dry run. Use without --dry-run to apply changes.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
