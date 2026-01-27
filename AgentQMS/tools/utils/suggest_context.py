#!/usr/bin/env python3
"""
Context Suggestion Tool for AgentQMS with AST-Based Analysis

Analyzes task descriptions and suggests appropriate context bundles
based on keyword matching and AST-based code pattern analysis.

Features:
- Keyword-based bundle suggestion (traditional)
- AST pattern detection for debugging bundles (new)
- Debugging context detection (debug, refactor, audit tasks)
- Dynamic bundle prioritization based on task type

Usage:
    uv run python AgentQMS/tools/utilities/suggest_context.py "implement new feature for OCR"
    uv run python AgentQMS/tools/utilities/suggest_context.py --file task_description.txt
    uv run python AgentQMS/tools/utilities/suggest_context.py --json "debug memory leak"
    uv run python AgentQMS/tools/utilities/suggest_context.py --analyze-patterns "debug config merge issue"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


from AgentQMS.tools.core.plugins import get_plugin_registry
from AgentQMS.tools.utils.system.paths import get_project_root
from AgentQMS.tools.utils.system.runtime import ensure_project_root_on_sys_path

ensure_project_root_on_sys_path()


class DebugPatternAnalyzer:
    """Analyzes task descriptions for debugging patterns and AST integration needs."""

    # Patterns that indicate debugging-related tasks
    DEBUG_PATTERNS = [
        r"\bdebug(?:ging)?\b",
        r"\btroubleshoot(?:ing)?\b",
        r"\brefactor(?:ing)?\b",
        r"\b(?:code\s+)?audit\b",
        r"\btrace\b.*(?:merge|flow|path)",
        r"\bwhy\b.*(?:override|fail|not work)",
        r"\b(?:find|search)\b.*(?:config|component|instantiat)",
        r"\b(?:analyze|understand)\b.*(?:flow|precedence|order)",
        r"\bmerge.*order\b",
        r"\bconfig.*access\b",
        r"\bhydra.*(?:pattern|issue)\b",
    ]

    # Patterns indicating AST/code analysis would be helpful
    AST_BENEFICIAL_PATTERNS = [
        r"\bmerge\b",
        r"\boverride\b",
        r"\binstantiata\b",
        r"\bfactory\b",
        r"\bcomponent\b",
        r"\bconfig\b.*(?:precedence|order|flow)",
    ]

    @classmethod
    def is_debugging_task(cls, task_description: str) -> bool:
        """Check if task is debugging-related.

        Args:
            task_description: Task description to analyze

        Returns:
            True if task appears to be debugging-related
        """
        task_lower = task_description.lower()
        for pattern in cls.DEBUG_PATTERNS:
            if re.search(pattern, task_lower):
                return True
        return False

    @classmethod
    def needs_ast_analysis(cls, task_description: str) -> bool:
        """Check if task would benefit from AST analysis.

        Args:
            task_description: Task description to analyze

        Returns:
            True if AST-based debugging would be helpful
        """
        task_lower = task_description.lower()
        for pattern in cls.AST_BENEFICIAL_PATTERNS:
            if re.search(pattern, task_lower):
                return True
        return False

    @classmethod
    def detect_analysis_type(cls, task_description: str) -> str:
        """Detect what kind of analysis is needed.

        Returns: One of 'config_access', 'merge_order', 'hydra_usage', 'instantiation', 'general'
        """
        task_lower = task_description.lower()

        if re.search(r"\b(?:cfg\.|config\[|access)\b", task_lower):
            return "config_access"
        if re.search(r"\bmerge.*order\b|\bprecedence\b", task_lower):
            return "merge_order"
        if re.search(r"\bhydra\b|\b@hydra\b", task_lower):
            return "hydra_usage"
        if re.search(r"\binstantiata\b|\bfactory\b|\bget_.*_by\b", task_lower):
            return "instantiation"

        return "general"


class StandardsSuggester:
    """Suggests applicable standards based on task description using standards-router.yaml."""

    def __init__(self, project_root: Path | None = None):
        """Initialize the standards suggester.

        Args:
            project_root: Project root path (auto-detected if not provided)
        """
        if project_root is None:
            project_root = get_project_root()
        self.project_root = Path(project_root)
        self._router: dict[str, Any] = {}
        self._load_router()

    def _load_router(self) -> None:
        """Load standards router configuration."""
        router_path = self.project_root / "AgentQMS/standards/standards-router.yaml"
        if router_path.exists():
            import yaml
            with open(router_path, encoding="utf-8") as f:
                self._router = yaml.safe_load(f) or {}

    def suggest(self, task_description: str) -> list[dict[str, Any]]:
        """Suggest applicable standards for a task.

        Args:
            task_description: Description of the task

        Returns:
            List of matching task mappings with their standards
        """
        task_lower = task_description.lower()
        matches = []

        task_mappings = self._router.get("task_mappings", {})
        for task_type, config in task_mappings.items():
            triggers = config.get("triggers", {})
            keywords = triggers.get("keywords", [])
            patterns = triggers.get("patterns", [])

            # Check keyword matches
            keyword_matches = [k for k in keywords if k.lower() in task_lower]

            # Check pattern matches
            pattern_matches = []
            for pattern in patterns:
                if re.search(pattern, task_lower):
                    pattern_matches.append(pattern)

            if keyword_matches or pattern_matches:
                matches.append({
                    "task_type": task_type,
                    "description": config.get("description", ""),
                    "standards": config.get("standards", []),
                    "priority": config.get("priority", 3),
                    "matched_keywords": keyword_matches,
                    "matched_patterns": pattern_matches,
                })

        # Sort by priority (lower is higher priority)
        matches.sort(key=lambda x: x["priority"])
        return matches


class ContextSuggester:
    """Suggests context bundles based on task keywords and AST pattern analysis."""

    def __init__(self, project_root: Path | None = None):
        """Initialize the context suggester.

        Args:
            project_root: Project root path (auto-detected if not provided)
        """
        if project_root is None:
            project_root = get_project_root()

        self.project_root = Path(project_root)
        self._bundles: dict[str, Any] = {}
        self._debug_analyzer = DebugPatternAnalyzer()
        self._standards_suggester = StandardsSuggester(project_root)
        self._load_bundles_from_registry()

    def _load_bundles_from_registry(self) -> None:
        """Load context bundles from plugin registry."""
        try:
            registry = get_plugin_registry()
            raw_bundles = registry.get_context_bundles()

            # Transform plugin bundles into task type format for scoring
            for bundle_name, bundle_config in raw_bundles.items():
                # Extract keywords from tags and triggers
                keywords = []
                patterns = []

                # Get tags (always present)
                if "tags" in bundle_config:
                    keywords.extend(bundle_config["tags"])

                # Get trigger keywords if defined
                if "triggers" in bundle_config:
                    trigger_keywords = bundle_config["triggers"].get("keywords", [])
                    keywords.extend(trigger_keywords)
                    trigger_patterns = bundle_config["triggers"].get("patterns", [])
                    patterns.extend(trigger_patterns)

                # Store bundle with metadata
                self._bundles[bundle_name] = {
                    "keywords": keywords,
                    "patterns": patterns,
                    "context_bundle": bundle_name,
                    "description": bundle_config.get("description", ""),
                    "title": bundle_config.get("title", bundle_name),
                }

        except Exception as e:
            raise RuntimeError(f"Failed to load context bundles from plugin registry: {e}") from e

    def suggest(self, task_description: str, include_debugging: bool = True) -> dict[str, Any]:
        """Suggest context bundles for a task description.

        Args:
            task_description: Description of the task
            include_debugging: If True, automatically include debugging bundles for debug tasks

        Returns:
            Dictionary with suggested bundles and ranking
        """
        task_lower = task_description.lower()
        scores: dict[str, int] = {}
        matched_keywords: dict[str, list[str]] = {}
        analysis_metadata: dict[str, Any] = {}

        # Check if this is a debugging task
        is_debug_task = include_debugging and self._debug_analyzer.is_debugging_task(task_description)
        needs_ast = self._debug_analyzer.needs_ast_analysis(task_description)
        analysis_type = self._debug_analyzer.detect_analysis_type(task_description) if needs_ast else None

        # Score each bundle based on keyword matches
        for bundle_name, config in self._bundles.items():
            keywords = config.get("keywords", [])
            patterns = config.get("patterns", [])
            matches = []
            pattern_matches = []

            for keyword in keywords:
                if keyword.lower() in task_lower:
                    matches.append(keyword)

            for pattern in patterns:
                if re.search(pattern, task_lower, re.IGNORECASE):
                    pattern_matches.append(pattern)

            if matches or pattern_matches:
                # Weight: number of matches + length of matched keywords + pattern matches
                score = len(matches) + len(pattern_matches) + sum(len(k.split()) for k in matches)
                scores[bundle_name] = score
                matched_keywords[bundle_name] = matches + pattern_matches

        # Boost debugging bundle if this is a debugging task
        if is_debug_task and "ocr-debugging" in self._bundles:
            # Add base score if not already matched
            if "ocr-debugging" not in scores:
                scores["ocr-debugging"] = 5
            else:
                # Boost existing score
                scores["ocr-debugging"] = int(scores["ocr-debugging"] * 1.5)

            # Add analysis metadata
            analysis_metadata["is_debug_task"] = True
            analysis_metadata["ast_beneficial"] = needs_ast
            analysis_metadata["analysis_type"] = analysis_type
            analysis_metadata["recommended_tools"] = self._get_recommended_ast_tools(analysis_type)

        # Sort by score (descending)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Build suggestions
        suggestions = []
        for i, (bundle_name, score) in enumerate(ranked):
            config = self._bundles[bundle_name]
            is_primary = i == 0

            suggestion = {
                "task_type": bundle_name,
                "score": score,
                "context_bundle": bundle_name,
                "title": config.get("title", bundle_name),
                "description": config.get("description", ""),
                "matched_keywords": matched_keywords.get(bundle_name, []),
                "is_primary": is_primary,
            }

            # Add AST tool recommendations for debugging bundle
            if bundle_name == "ocr-debugging" and analysis_metadata:
                suggestion["debug_context"] = {
                    "is_debug_task": analysis_metadata.get("is_debug_task"),
                    "recommended_tools": analysis_metadata.get("recommended_tools", []),
                }

            suggestions.append(suggestion)

        # Include fallback message if nothing matched
        if not suggestions:
            suggestions.append(
                {
                    "task_type": "none",
                    "score": 0,
                    "context_bundle": None,
                    "title": "No matching bundles",
                    "description": "Consider creating a custom context bundle for this task",
                    "matched_keywords": [],
                    "is_primary": False,
                }
            )

        # Get standards suggestions
        standards_suggestions = self._standards_suggester.suggest(task_description)

        return {
            "task_description": task_description,
            "suggestions": suggestions,
            "primary_bundle": suggestions[0]["context_bundle"] if suggestions else None,
            "is_debugging_task": is_debug_task,
            "analysis_metadata": analysis_metadata if analysis_metadata else None,
            "standards": standards_suggestions,
        }

    def _get_recommended_ast_tools(self, analysis_type: str | None) -> list[str]:
        """Get recommended AST tools for the analysis type.

        Args:
            analysis_type: Type of analysis (config_access, merge_order, etc.)

        Returns:
            List of recommended tool commands
        """
        tools = {
            "config_access": [
                "adt analyze-config <file>",
                "adt analyze-config <file> --component <name>",
            ],
            "merge_order": [
                "adt trace-merges <file> --output markdown",
                "adt trace-merges <file> --output json",
            ],
            "hydra_usage": [
                "adt find-hydra <path>",
                "adt full-analysis <path>",
            ],
            "instantiation": [
                "adt find-instantiations <path> --component <type>",
                "adt full-analysis <path>",
            ],
            "general": [
                "adt full-analysis <path>",
                "adt context-tree <path>",
            ],
        }
        return tools.get(analysis_type or "general", tools["general"])

    def format_output(self, result: dict[str, Any], json_format: bool = False, verbose: bool = False) -> str:
        """Format suggestion result for display.

        Args:
            result: Suggestion result from suggest()
            json_format: If True, return JSON; otherwise return human-readable text
            verbose: If True, include additional details

        Returns:
            Formatted output string
        """
        if json_format:
            return json.dumps(result, indent=2)

        # Human-readable format
        lines = []
        lines.append(f"📋 Task: {result['task_description']}")
        lines.append("")

        # Show debugging context if available
        if result.get("is_debugging_task"):
            lines.append("🔍 DEBUGGING TASK DETECTED")
            if result.get("analysis_metadata"):
                meta = result["analysis_metadata"]
                if meta.get("recommended_tools"):
                    lines.append("   Recommended AST tools:")
                    for tool in meta["recommended_tools"][:3]:
                        lines.append(f"   • {tool}")
            lines.append("")

        lines.append("Suggested Context Bundles:")
        lines.append("-" * 50)

        for i, suggestion in enumerate(result["suggestions"], 1):
            bundle = suggestion["context_bundle"]
            task_type = suggestion["task_type"]
            title = suggestion.get("title", bundle)
            matched = suggestion["matched_keywords"]
            is_primary = suggestion.get("is_primary", False)

            if bundle:
                primary_marker = " ⭐ PRIMARY" if is_primary else ""
                lines.append(f"{i}. {bundle.upper()} - {title} (score: {suggestion['score']}){primary_marker}")
            else:
                lines.append(f"{i}. {title}")

            if matched:
                lines.append(f"   📌 Matched keywords: {', '.join(matched)}")

            # Show debug context if available
            if suggestion.get("debug_context"):
                debug = suggestion["debug_context"]
                if debug.get("recommended_tools"):
                    lines.append("   🛠️  AST Tools:")
                    for tool in debug["recommended_tools"][:2]:
                        lines.append(f"      $ {tool}")

            if bundle:
                lines.append(f'   🔧 Usage: uv run python AgentQMS/tools/utilities/suggest_context.py --analyze-patterns "{result["task_description"]}"')

            lines.append("")

        # Show standards suggestions if available
        standards = result.get("standards", [])
        if standards:
            lines.append("")
            lines.append("📚 Applicable Standards:")
            lines.append("-" * 50)
            for std in standards:
                lines.append(f"  📖 {std['task_type'].upper()}: {std['description']}")
                for s in std.get("standards", [])[:3]:
                    lines.append(f"     • {s}")
                if std.get("matched_keywords"):
                    lines.append(f"     📌 Matched: {', '.join(std['matched_keywords'][:3])}")
                lines.append("")

        return "\n".join(lines)


def main() -> int:
    """Command-line interface for context suggestion."""
    parser = argparse.ArgumentParser(
        description="Suggest context bundles for AgentQMS tasks with AST analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Direct task description
  %(prog)s "implement new feature for OCR"

  # From file
  %(prog)s --file task_description.txt

  # JSON output
  %(prog)s --json "debug memory leak"

  # Analyze patterns and show AST tools
  %(prog)s --analyze-patterns "debug config merge issue"

  # Verbose output with debug context
  %(prog)s -v "trace OmegaConf.merge precedence"
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
        "--analyze-patterns",
        "-a",
        action="store_true",
        help="Enable pattern analysis and show AST tool recommendations",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output including debug context",
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
        output = suggester.format_output(result, json_format=args.json, verbose=args.verbose or args.analyze_patterns)
        print(output)
        return 0
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
