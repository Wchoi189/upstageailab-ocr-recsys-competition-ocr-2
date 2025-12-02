#!/usr/bin/env python3
"""
Workflow Detection and Auto-Suggestion

Analyzes task descriptions and suggests appropriate workflows, tools, and context bundles.

Usage:
    from AgentQMS.agent_tools.core.workflow_detector import suggest_workflows
    
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
    print("ERROR: PyYAML not installed. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

from AgentQMS.agent_tools.core.context_bundle import TASK_KEYWORDS, analyze_task_type
from AgentQMS.agent_tools.utils.paths import get_project_root
from AgentQMS.agent_tools.utils.runtime import ensure_project_root_on_sys_path

ensure_project_root_on_sys_path()

PROJECT_ROOT = get_project_root()


# Workflow suggestions based on task keywords
WORKFLOW_SUGGESTIONS = {
    "development": {
        "context_bundle": "development",
        "suggested_workflows": [
            "create-plan",
            "validate",
            "compliance",
        ],
        "suggested_tools": [
            "artifact_workflow",
            "validate_artifacts",
            "context_bundle",
        ],
    },
    "documentation": {
        "context_bundle": "documentation",
        "suggested_workflows": [
            "docs-generate",
            "docs-validate-links",
            "docs-update-indexes",
        ],
        "suggested_tools": [
            "auto_generate_index",
            "validate_links",
            "update_artifact_indexes",
        ],
    },
    "debugging": {
        "context_bundle": "debugging",
        "suggested_workflows": [
            "create-bug-report",
            "validate",
            "context-debug",
        ],
        "suggested_tools": [
            "artifact_workflow",
            "validate_artifacts",
            "context_bundle",
        ],
    },
    "planning": {
        "context_bundle": "planning",
        "suggested_workflows": [
            "create-plan",
            "create-design",
            "context-plan",
        ],
        "suggested_tools": [
            "artifact_workflow",
            "context_bundle",
        ],
    },
    "general": {
        "context_bundle": "general",
        "suggested_workflows": [
            "discover",
            "status",
            "validate",
        ],
        "suggested_tools": [
            "discover",
            "validate_artifacts",
        ],
    },
}

# Artifact creation triggers
ARTIFACT_TRIGGERS = {
    "plan": ["plan", "design", "architecture", "blueprint", "strategy", "roadmap"],
    "assessment": ["assess", "evaluate", "review", "analysis"],
    "audit": ["audit", "compliance", "framework audit", "quality audit", "security audit", "accessibility audit", "performance audit", "code quality audit"],
    "bug_report": ["bug", "error", "issue", "fix", "broken", "crash", "exception"],
    "design": ["design", "spec", "specification", "schema"],
    "research": ["research", "investigate", "explore", "study"],
}


def detect_artifact_type(task_description: str) -> str | None:
    """Detect if task should create an artifact and which type."""
    task_lower = task_description.lower()
    
    for artifact_type, keywords in ARTIFACT_TRIGGERS.items():
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
    
    base_suggestions = WORKFLOW_SUGGESTIONS.get(task_type, WORKFLOW_SUGGESTIONS["general"])
    
    suggestions: dict[str, Any] = {
        "task_type": task_type,
        "context_bundle": base_suggestions["context_bundle"],
        "workflows": base_suggestions["suggested_workflows"].copy(),
        "tools": base_suggestions["suggested_tools"].copy(),
        "artifact_type": artifact_type,
        "make_commands": [],
    }
    
    # Add artifact creation workflow if detected
    if artifact_type:
        if artifact_type == "plan":
            suggestions["workflows"].insert(0, "create-plan")
            suggestions["make_commands"].append(f'make create-plan NAME=my-{artifact_type} TITLE="..."')
        elif artifact_type == "bug_report":
            suggestions["workflows"].insert(0, "create-bug-report")
            suggestions["make_commands"].append(f'make create-bug-report NAME=my-{artifact_type} TITLE="..."')
        elif artifact_type == "design":
            suggestions["workflows"].insert(0, "create-design")
            suggestions["make_commands"].append(f'make create-design NAME=my-{artifact_type} TITLE="..."')
        elif artifact_type == "assessment":
            suggestions["workflows"].insert(0, "create-assessment")
            suggestions["make_commands"].append(f'make create-assessment NAME=my-{artifact_type} TITLE="..."')
        elif artifact_type == "research":
            suggestions["workflows"].insert(0, "create-research")
            suggestions["make_commands"].append(f'make create-research NAME=my-{artifact_type} TITLE="..."')
    
    # Add context loading command
    context_cmd = f'make context-{suggestions["context_bundle"]}'
    suggestions["make_commands"].insert(0, context_cmd)
    
    return suggestions


def generate_workflow_triggers_yaml(output_path: Path) -> None:
    """Generate workflow-triggers.yaml mapping task patterns to workflows."""
    triggers: dict[str, Any] = {
        "version": "1.0",
        "task_types": {},
        "artifact_triggers": ARTIFACT_TRIGGERS,
    }
    
    for task_type, config in WORKFLOW_SUGGESTIONS.items():
        triggers["task_types"][task_type] = {
            "keywords": TASK_KEYWORDS.get(task_type, []),
            "context_bundle": config["context_bundle"],
            "suggested_workflows": config["suggested_workflows"],
            "suggested_tools": config["suggested_tools"],
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
        default=str(PROJECT_ROOT / ".copilot" / "context" / "workflow-triggers.yaml"),
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
        print(f"  cd AgentQMS/interface && {cmd}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

