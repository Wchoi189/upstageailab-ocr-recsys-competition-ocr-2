#!/usr/bin/env python3
"""
Context Suggestion Tool for AgentQMS

Analyzes task descriptions and suggests appropriate context bundles
based on keyword matching against workflow-triggers.yaml.

Usage:
    python suggest_context.py "implement new feature for OCR"
    python suggest_context.py --file task_description.txt
    python suggest_context.py --json "debug memory leak"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from AgentQMS.agent_tools.utils.paths import get_project_root
from AgentQMS.agent_tools.utils.runtime import ensure_project_root_on_sys_path

ensure_project_root_on_sys_path()


class ContextSuggester:
    """Suggests context bundles based on task keywords."""

    def __init__(self, project_root: Path | None = None):
        """Initialize the context suggester.

        Args:
            project_root: Project root path (auto-detected if not provided)
        """
        if project_root is None:
            project_root = get_project_root()

        self.project_root = Path(project_root)
        self.triggers_file = (
            self.project_root / ".copilot" / "context" / "workflow-triggers.yaml"
        )

        self._task_types: dict[str, Any] = {}
        self._load_triggers()

    def _load_triggers(self) -> None:
        """Load workflow triggers configuration."""
        if not self.triggers_file.exists():
            raise FileNotFoundError(
                f"Workflow triggers file not found: {self.triggers_file}"
            )

        try:
            with open(self.triggers_file, encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
                self._task_types = config.get("task_types", {})
        except Exception as e:
            raise RuntimeError(f"Failed to load workflow triggers: {e}") from e

    def suggest(self, task_description: str) -> dict[str, Any]:
        """Suggest context bundles for a task description.

        Args:
            task_description: Description of the task

        Returns:
            Dictionary with suggested bundles and ranking
        """
        task_lower = task_description.lower()
        scores: dict[str, int] = {}
        matched_keywords: dict[str, list[str]] = {}

        # Score each task type based on keyword matches
        for task_type, config in self._task_types.items():
            keywords = config.get("keywords", [])
            matches = []

            for keyword in keywords:
                if keyword.lower() in task_lower:
                    matches.append(keyword)

            if matches:
                # Weight: number of matches + length of matched keywords
                score = len(matches) + sum(len(k.split()) for k in matches)
                scores[task_type] = score
                matched_keywords[task_type] = matches

        # Sort by score (descending)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Build suggestions
        suggestions = []
        for task_type, score in ranked:
            config = self._task_types[task_type]
            suggestions.append({
                "task_type": task_type,
                "score": score,
                "context_bundle": config.get("context_bundle"),
                "matched_keywords": matched_keywords.get(task_type, []),
                "suggested_workflows": config.get("suggested_workflows", []),
                "suggested_tools": config.get("suggested_tools", []),
            })

        # Always include general as fallback if nothing matched
        if not suggestions:
            config = self._task_types.get("general", {})
            suggestions.append({
                "task_type": "general",
                "score": 0,
                "context_bundle": config.get("context_bundle"),
                "matched_keywords": [],
                "suggested_workflows": config.get("suggested_workflows", []),
                "suggested_tools": config.get("suggested_tools", []),
            })

        return {
            "task_description": task_description,
            "suggestions": suggestions,
            "primary_bundle": suggestions[0]["context_bundle"] if suggestions else None,
        }

    def format_output(self, result: dict[str, Any], json_format: bool = False) -> str:
        """Format suggestion result for display.

        Args:
            result: Suggestion result from suggest()
            json_format: If True, return JSON; otherwise return human-readable text

        Returns:
            Formatted output string
        """
        if json_format:
            return json.dumps(result, indent=2)

        # Human-readable format
        lines = []
        lines.append(f"📋 Task: {result['task_description']}")
        lines.append("")
        lines.append("Suggested Context Bundles:")
        lines.append("-" * 50)

        for i, suggestion in enumerate(result["suggestions"], 1):
            bundle = suggestion["context_bundle"]
            task_type = suggestion["task_type"]
            matched = suggestion["matched_keywords"]
            workflows = suggestion["suggested_workflows"]

            lines.append(f"{i}. {bundle.upper()} (type: {task_type}, score: {suggestion['score']})")

            if matched:
                lines.append(f"   📌 Matched keywords: {', '.join(matched)}")

            if workflows:
                workflow_cmds = [f"make {w}" for w in workflows]
                lines.append(f"   💡 Try: {' | '.join(workflow_cmds)}")

            lines.append(f"   🔧 Usage: make context TASK=\"{result['task_description']}\"")
            lines.append("")

        return "\n".join(lines)


def main() -> int:
    """Command-line interface for context suggestion."""
    parser = argparse.ArgumentParser(
        description="Suggest context bundles for AgentQMS tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Direct task description
  %(prog)s "implement new feature for OCR"

  # From file
  %(prog)s --file task_description.txt

  # JSON output
  %(prog)s --json "debug memory leak"
        """,
    )

    parser.add_argument(
        "task",
        nargs="?",
        help="Task description to analyze",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=Path,
        help="Read task description from file",
    )
    parser.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Output result as JSON",
    )
    parser.add_argument(
        "--project-root",
        "-p",
        type=Path,
        help="Project root directory",
    )

    args = parser.parse_args()

    # Get task description
    task_description = None
    if args.file:
        if not args.file.exists():
            print(f"❌ File not found: {args.file}", file=sys.stderr)
            return 1
        task_description = args.file.read_text(encoding="utf-8").strip()
    elif args.task:
        task_description = args.task
    else:
        parser.print_help()
        return 0

    # Run suggestion
    try:
        suggester = ContextSuggester(project_root=args.project_root)
        result = suggester.suggest(task_description)
        output = suggester.format_output(result, json_format=args.json)
        print(output)
        return 0
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
