#!/usr/bin/env python3
"""
Workflow Detection and Auto-Suggestion

Analyzes task descriptions and suggests appropriate workflows, tools, and context bundles.

Usage:
    from AgentQMS.tools.core.workflow_detector import suggest_workflows

    suggestions = suggest_workflows("implement new feature")
    print(suggestions)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. Fix with: uv sync", file=sys.stderr)
    sys.exit(1)

from AgentQMS.tools.core.context_bundle import TASK_KEYWORDS, analyze_task_type
from AgentQMS.tools.utils.config.loader import ConfigLoader
from AgentQMS.tools.utils.paths import get_project_root
from AgentQMS.tools.utils.system.runtime import ensure_project_root_on_sys_path

ensure_project_root_on_sys_path()

PROJECT_ROOT = get_project_root()

STANDARDS_CONFIG_DIR = PROJECT_ROOT / "AgentQMS" / "standards" / "tier4-workflows"
CONFIG_PATH = STANDARDS_CONFIG_DIR / "workflow-detector.yaml"
_CONFIG_CACHE: dict[str, Any] | None = None
_CONFIG_LOADER = ConfigLoader(cache_size=5)

# DEFAULT_CONFIG removed - all workflow detection config now loaded from:
# AgentQMS/standards/tier1-foundations/workflow-detector.yaml
# This enforces the plugin/YAML-only architecture with no hardcoded fallbacks.
# System will fail loudly if config is missing, making issues immediately visible.


def _get_config() -> dict[str, Any]:
    """Load workflow detection config from YAML (no hardcoded defaults)."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        # Fail-fast: no defaults, must load from YAML
        _CONFIG_CACHE = _CONFIG_LOADER.get_config(CONFIG_PATH)
        if not _CONFIG_CACHE:
            raise ValueError(
                f"Workflow detection config missing: {CONFIG_PATH}. "
                "No hardcoded fallbacks - config must be explicitly defined in YAML."
            )
    return _CONFIG_CACHE


def detect_artifact_type(task_description: str) -> str | None:
    """Detect if task should create an artifact and which type."""
    task_lower = task_description.lower()

    for artifact_type, keywords in _get_config().get("artifact_triggers", {}).items():
        for keyword in keywords:
            if keyword in task_lower:
                return artifact_type

    return None


def suggest_workflows(task_description: str) -> dict[str, Any]:
    """Suggest workflows, tools, and context based on task description.

    Args:
        task_description: Description of the task

    Returns:
        Dictionary with suggestions including:
        - task_type: Detected task type
        - context_bundle: Suggested context bundle
        - workflows: List of suggested workflow names
        - tools: List of suggested tool names
        - artifact_type: Suggested artifact type if applicable
        - make_commands: List of make commands to run
    """
    task_type = analyze_task_type(task_description)
    artifact_type = detect_artifact_type(task_description)

    config = _get_config()
    task_types = config.get("task_types", {})
    base_suggestions = task_types.get(task_type, task_types.get("general", {}))

    suggestions: dict[str, Any] = {
        "task_type": task_type,
        "context_bundle": base_suggestions["context_bundle"],
        "workflows": list(base_suggestions["suggested_workflows"]),
        "tools": list(base_suggestions["suggested_tools"]),
        "artifact_type": artifact_type,
        "make_commands": [],
    }

    # Add artifact creation workflow if detected
    if artifact_type:
        cmd_templates = config.get("command_templates", {}).get("artifact", {})
        if artifact_type == "plan":
            suggestions["workflows"].insert(0, "create-plan")
            template = cmd_templates.get("plan")
        elif artifact_type == "bug_report":
            suggestions["workflows"].insert(0, "create-bug-report")
            template = cmd_templates.get("bug_report")
        elif artifact_type == "design":
            suggestions["workflows"].insert(0, "create-design")
            template = cmd_templates.get("design")
        elif artifact_type == "assessment":
            suggestions["workflows"].insert(0, "create-assessment")
            template = cmd_templates.get("assessment")
        elif artifact_type == "research":
            suggestions["workflows"].insert(0, "create-research")
            template = cmd_templates.get("research")
        else:
            template = None

        if template:
            suggestions["make_commands"].append(template.format(artifact_type=artifact_type))

    # Add context loading command
    context_template = config.get("command_templates", {}).get("context", "make context-{context_bundle}")
    context_cmd = context_template.format(context_bundle=suggestions["context_bundle"])
    suggestions["make_commands"].insert(0, context_cmd)

    return suggestions


def generate_workflow_triggers_yaml(output_path: Path) -> None:
    """Generate workflow-triggers.yaml mapping task patterns to workflows."""
    config = _get_config()
    triggers: dict[str, Any] = {
        "version": config.get("version", "1.0"),
        "task_types": {},
        "artifact_triggers": config.get("artifact_triggers", {}),
    }

    for task_type, entry in config.get("task_types", {}).items():
        triggers["task_types"][task_type] = {
            "keywords": TASK_KEYWORDS.get(task_type, []),
            "context_bundle": entry.get("context_bundle"),
            "suggested_workflows": entry.get("suggested_workflows", []),
            "suggested_tools": entry.get("suggested_tools", []),
        }

    with output_path.open("w", encoding="utf-8") as f:
        yaml.dump(triggers, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Generated {output_path}")


def main() -> int:
    """CLI entry point for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Workflow detection and suggestion")
    parser.add_argument("task", nargs="?", help="Task description to analyze")
    parser.add_argument(
        "--generate-triggers",
        action="store_true",
        help="Generate workflow-triggers.yaml file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "AgentQMS" / "config" / "workflow-triggers.yaml"),
        help="Output path for triggers file",
    )

    args = parser.parse_args()

    if args.generate_triggers:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        generate_workflow_triggers_yaml(output_path)
        return 0

    if not args.task:
        parser.print_help()
        return 1

    suggestions = suggest_workflows(args.task)

    print(f"Task: {args.task}")
    print(f"Detected Type: {suggestions['task_type']}")
    print(f"Context Bundle: {suggestions['context_bundle']}")

    if suggestions["artifact_type"]:
        print(f"Suggested Artifact: {suggestions['artifact_type']}")

    print("\nSuggested Workflows:")
    for wf in suggestions["workflows"]:
        print(f"  - {wf}")

    print("\nSuggested Tools:")
    for tool in suggestions["tools"]:
        print(f"  - {tool}")

    print("\nMake Commands:")
    for cmd in suggestions["make_commands"]:
        print(f"  cd AgentQMS/bin && {cmd}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
