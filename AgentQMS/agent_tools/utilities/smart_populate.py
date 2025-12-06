#!/usr/bin/env python3
"""
Smart Auto-Population for Artifact Creation

Analyzes project context (git, codebase, related files) to intelligently
pre-populate artifact fields like author, branch, tags, and related files.

Usage:
    python smart_populate.py analyze --type implementation_plan
    python smart_populate.py suggest-tags --type assessment
    python smart_populate.py suggest-files --type bug_report
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from AgentQMS.agent_tools.utils.paths import get_project_root
from AgentQMS.agent_tools.utils.runtime import ensure_project_root_on_sys_path

ensure_project_root_on_sys_path()


class SmartPopulator:
    """Analyzes context and suggests artifact field values."""

    def __init__(self, project_root: Path | None = None):
        """Initialize the populator.

        Args:
            project_root: Project root path (auto-detected if not provided)
        """
        if project_root is None:
            project_root = get_project_root()

        self.project_root = Path(project_root)
        self.rules_file = self.project_root / "AgentQMS" / "knowledge" / "agent" / "artifact_rules.yaml"
        self._rules = self._load_rules()
        self._git_context = self._analyze_git()

    def _load_rules(self) -> dict[str, Any]:
        """Load artifact rules to understand type-specific fields."""
        if not self.rules_file.exists():
            return {}

        try:
            with open(self.rules_file, encoding="utf-8") as f:
                rules = yaml.safe_load(f) or {}
                return rules.get("artifact_types", {})
        except Exception:
            return {}

    def _analyze_git(self) -> dict[str, Any]:
        """Analyze git context for current branch and author."""
        context = {}

        try:
            # Get current branch
            branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            context["branch"] = branch
        except Exception:
            context["branch"] = "unknown"

        try:
            # Get current user
            author = subprocess.run(
                ["git", "config", "user.name"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            context["author"] = author
        except Exception:
            context["author"] = "Unknown"

        try:
            # Get user email
            email = subprocess.run(
                ["git", "config", "user.email"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            context["email"] = email
        except Exception:
            context["email"] = ""

        try:
            # Get recent changed files
            changed_files = subprocess.run(
                ["git", "diff", "--name-only", "HEAD~5..HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            ).stdout.strip().split("\n")
            context["recent_files"] = [f for f in changed_files if f][:10]
        except Exception:
            context["recent_files"] = []

        return context

    def analyze_artifact_type(self, artifact_type: str) -> dict[str, Any]:
        """Analyze what fields are expected for an artifact type.

        Args:
            artifact_type: Type of artifact

        Returns:
            Dictionary with field suggestions
        """
        type_config = self._rules.get(artifact_type, {})
        required_fields = type_config.get("required_fields", [])
        optional_fields = type_config.get("optional_fields", [])

        return {
            "artifact_type": artifact_type,
            "required_fields": required_fields,
            "optional_fields": optional_fields,
            "expected_frontmatter": type_config.get("frontmatter_template", {}),
        }

    def suggest_author(self) -> str:
        """Suggest artifact author from git config.

        Returns:
            Author name
        """
        return self._git_context.get("author", "Unknown")

    def suggest_branch(self) -> str:
        """Suggest artifact branch from git context.

        Returns:
            Current branch name
        """
        return self._git_context.get("branch", "main")

    def suggest_tags(self, artifact_type: str) -> list[str]:
        """Suggest tags based on artifact type and rules.

        Args:
            artifact_type: Type of artifact

        Returns:
            List of suggested tags
        """
        type_config = self._rules.get(artifact_type, {})
        default_tags = type_config.get("default_tags", [])
        categories = type_config.get("categories", [])

        tags = list(default_tags)
        if categories:
            tags.extend(categories)

        # Add artifact type itself as tag
        tags.append(artifact_type)

        # Deduplicate and sort
        return sorted(set(tags))

    def suggest_related_files(self, artifact_type: str, limit: int = 5) -> list[str]:
        """Suggest related files based on git history and artifact type.

        Args:
            artifact_type: Type of artifact
            limit: Maximum number of files to suggest

        Returns:
            List of related file paths
        """
        recent = self._git_context.get("recent_files", [])[:limit]

        # Filter to likely relevant files
        relevant = []
        for filepath in recent:
            # Skip test files, cache, etc. for now
            if any(skip in filepath for skip in ["__pycache__", ".git", "node_modules"]):
                continue
            relevant.append(filepath)

        return relevant[:limit]

    def suggest_metadata(self, artifact_type: str) -> dict[str, Any]:
        """Generate complete metadata suggestions for artifact.

        Args:
            artifact_type: Type of artifact

        Returns:
            Dictionary with suggested metadata
        """
        return {
            "type": artifact_type,
            "author": self.suggest_author(),
            "branch": self.suggest_branch(),
            "tags": self.suggest_tags(artifact_type),
            "related_files": self.suggest_related_files(artifact_type),
            "status": "active",  # Default for new artifacts
        }

    def format_frontmatter(self, artifact_type: str) -> str:
        """Generate suggested frontmatter for artifact.

        Args:
            artifact_type: Type of artifact

        Returns:
            YAML frontmatter string
        """
        metadata = self.suggest_metadata(artifact_type)

        # Build minimal frontmatter
        frontmatter = {
            "type": metadata["type"],
            "status": metadata["status"],
        }

        # Add author if available
        if metadata["author"] != "Unknown":
            frontmatter["author"] = metadata["author"]

        # Add tags
        if metadata["tags"]:
            frontmatter["tags"] = metadata["tags"]

        # Format as YAML
        yaml_str = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
        return f"---\n{yaml_str}---\n"


def main() -> int:
    """Command-line interface for smart population."""
    parser = argparse.ArgumentParser(
        description="Smart auto-population for artifact creation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze artifact type requirements
  %(prog)s analyze --type implementation_plan

  # Suggest tags for an artifact type
  %(prog)s suggest-tags --type assessment

  # Suggest related files based on git history
  %(prog)s suggest-files --type bug_report

  # Generate complete metadata suggestion
  %(prog)s suggest-metadata --type design
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze artifact type")
    analyze_parser.add_argument("--type", "-t", required=True, help="Artifact type")

    # Suggest tags command
    tags_parser = subparsers.add_parser("suggest-tags", help="Suggest tags")
    tags_parser.add_argument("--type", "-t", required=True, help="Artifact type")

    # Suggest files command
    files_parser = subparsers.add_parser("suggest-files", help="Suggest related files")
    files_parser.add_argument("--type", "-t", required=True, help="Artifact type")
    files_parser.add_argument("--limit", "-l", type=int, default=5, help="Max files")

    # Suggest metadata command
    metadata_parser = subparsers.add_parser("suggest-metadata", help="Suggest all metadata")
    metadata_parser.add_argument("--type", "-t", required=True, help="Artifact type")
    metadata_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Suggest frontmatter command
    frontmatter_parser = subparsers.add_parser("frontmatter", help="Generate frontmatter")
    frontmatter_parser.add_argument("--type", "-t", required=True, help="Artifact type")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    try:
        populator = SmartPopulator()

        if args.command == "analyze":
            result = populator.analyze_artifact_type(args.type)
            print(f"📋 Artifact Type: {result['artifact_type']}")
            print(f"   Required fields: {', '.join(result['required_fields'])}")
            print(f"   Optional fields: {', '.join(result['optional_fields'])}")

        elif args.command == "suggest-tags":
            tags = populator.suggest_tags(args.type)
            print(f"🏷️  Suggested tags for {args.type}:")
            for tag in tags:
                print(f"   - {tag}")

        elif args.command == "suggest-files":
            files = populator.suggest_related_files(args.type, limit=args.limit)
            if files:
                print("📁 Suggested related files:")
                for filepath in files:
                    print(f"   - {filepath}")
            else:
                print("ℹ️  No recent files found")

        elif args.command == "suggest-metadata":
            metadata = populator.suggest_metadata(args.type)
            if args.json:
                print(json.dumps(metadata, indent=2))
            else:
                print(f"📝 Suggested metadata for {args.type}:")
                for key, value in metadata.items():
                    if isinstance(value, list):
                        print(f"   {key}: {', '.join(str(v) for v in value)}")
                    else:
                        print(f"   {key}: {value}")

        elif args.command == "frontmatter":
            frontmatter = populator.format_frontmatter(args.type)
            print("📄 Suggested frontmatter:")
            print(frontmatter)

        return 0

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
