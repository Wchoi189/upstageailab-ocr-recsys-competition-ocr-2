#!/usr/bin/env python3
"""
Context Bundle Generator Core

Generates task-specific context bundles from YAML definitions.
Supports automatic task type detection, glob patterns, and freshness checking.

Supports extension via plugin system - see .agentqms/plugins/context_bundles/

This is the canonical implementation in agent_tools.

Usage:
    from AgentQMS.agent_tools.core.context_bundle import get_context_bundle

    # Automatic task type detection
    files = get_context_bundle("implement new feature")

    # Explicit task type
    files = get_context_bundle("fix bug", task_type="debugging")
    
    # Plugin-registered bundle
    files = get_context_bundle("security review", task_type="security-review")
"""

import glob
import sys
import time
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print(
        "ERROR: PyYAML not installed. Install with: pip install pyyaml", file=sys.stderr
    )
    sys.exit(1)

from AgentQMS.agent_tools.utils.paths import get_project_root
from AgentQMS.agent_tools.utils.runtime import ensure_project_root_on_sys_path

ensure_project_root_on_sys_path()
PROJECT_ROOT = get_project_root()

# Default bundle directory - canonical location
BUNDLES_DIR = PROJECT_ROOT / "AgentQMS" / "knowledge" / "context_bundles"

# Try to import plugin registry for extensibility
try:
    from AgentQMS.agent_tools.core.plugins import get_plugin_registry

    PLUGINS_AVAILABLE = True
except ImportError:
    PLUGINS_AVAILABLE = False

# Task type keywords for automatic classification
TASK_KEYWORDS = {
    "development": [
        "implement",
        "code",
        "develop",
        "feature",
        "function",
        "class",
        "module",
        "refactor",
        "rewrite",
        "build",
        "create",
        "add",
        "fix bug",
        "bug fix",
    ],
    "documentation": [
        "document",
        "doc",
        "write docs",
        "readme",
        "guide",
        "manual",
        "tutorial",
        "update docs",
        "documentation",
    ],
    "debugging": [
        "debug",
        "troubleshoot",
        "error",
        "fix",
        "broken",
        "issue",
        "problem",
        "crash",
        "exception",
        "traceback",
        "log",
        "investigate",
    ],
    "planning": [
        "plan",
        "design",
        "architecture",
        "blueprint",
        "strategy",
        "assess",
        "evaluate",
        "analysis",
        "proposal",
        "roadmap",
    ],
}


def analyze_task_type(description: str) -> str:
    """
    Analyze task description and classify task type based on keywords.

    Args:
        description: Task description text

    Returns:
        Task type: 'development', 'documentation', 'debugging', 'planning', or 'general'
    """
    description_lower = description.lower()

    # Count keyword matches for each task type
    scores = {}
    for task_type, keywords in TASK_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in description_lower)
        if score > 0:
            scores[task_type] = score

    # Return task type with highest score, or 'general' if no matches
    if scores:
        return max(scores.items(), key=lambda x: x[1])[0]

    return "general"


def load_bundle_definition(bundle_name: str) -> dict[str, Any]:
    """
    Load bundle definition from YAML file or plugin registry.

    Searches in order:
    1. Framework bundles: AgentQMS/knowledge/context_bundles/
    2. Plugin bundles: .agentqms/plugins/context_bundles/ (via registry)

    Args:
        bundle_name: Name of bundle (without .yaml extension)

    Returns:
        Bundle definition dictionary

    Raises:
        FileNotFoundError: If bundle file doesn't exist
        yaml.YAMLError: If YAML is invalid
    """
    # First, check framework bundles directory
    bundle_path = BUNDLES_DIR / f"{bundle_name}.yaml"

    if bundle_path.exists():
        with bundle_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # Second, check plugin registry
    if PLUGINS_AVAILABLE:
        try:
            registry = get_plugin_registry()
            plugin_bundle = registry.get_context_bundle(bundle_name)
            if plugin_bundle:
                return plugin_bundle
        except Exception:
            pass  # Fall through to error

    # Not found in either location
    available = list_available_bundles()
    available_str = ", ".join(available) if available else "none"
    raise FileNotFoundError(
        f"Bundle '{bundle_name}' not found. "
        f"Available bundles: {available_str}"
    )


def is_fresh(path: Path | str, days: int = 30) -> bool:
    """
    Check if file/directory was modified within specified days.

    Args:
        path: File or directory path
        days: Number of days to check (default: 30)

    Returns:
        True if file is fresh (modified within days), False otherwise
    """
    path_obj = Path(path)

    if not path_obj.exists():
        return False

    # Get modification time
    mtime = path_obj.stat().st_mtime

    # Calculate days since modification
    days_since_modification = (time.time() - mtime) / (24 * 60 * 60)

    return days_since_modification <= days


def expand_glob_pattern(pattern: str, max_files: int | None = None) -> list[Path]:
    """
    Expand glob pattern and return matching files.

    Args:
        pattern: Glob pattern (e.g., "docs/**/*.md")
        max_files: Maximum number of files to return (None = all)

    Returns:
        List of matching file paths
    """
    # Make pattern relative to project root
    if not Path(pattern).is_absolute():
        pattern = str(PROJECT_ROOT / pattern)

    matches = [Path(p) for p in glob.glob(pattern, recursive=True) if Path(p).is_file()]

    # Sort by modification time (newest first)
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    if max_files:
        matches = matches[:max_files]

    return matches


def validate_bundle_files(bundle_def: dict[str, Any]) -> list[str]:
    """
    Validate bundle definition and return list of valid file paths.

    Args:
        bundle_def: Bundle definition dictionary

    Returns:
        List of valid file paths (as strings)
    """
    valid_paths: list[str] = []

    tiers = bundle_def.get("tiers", {})

    for tier_key in sorted(tiers.keys()):  # tier1, tier2, tier3...
        tier = tiers[tier_key]
        max_files = tier.get("max_files")
        files = tier.get("files", [])

        tier_paths: list[Path] = []

        for file_spec in files:
            if isinstance(file_spec, str):
                file_path_str = file_spec
            elif isinstance(file_spec, dict):
                file_path_str = file_spec.get("path", "")
            else:
                continue

            # Handle glob patterns
            if "*" in file_path_str or "**" in file_path_str:
                expanded = expand_glob_pattern(file_path_str, max_files=None)
                tier_paths.extend(expanded)
            else:
                # Regular file path
                file_path = PROJECT_ROOT / file_path_str

                if file_path.exists() and is_fresh(file_path):
                    tier_paths.append(file_path)

        # Apply tier limit and add to valid paths
        if max_files and len(tier_paths) > max_files:
            tier_paths = tier_paths[:max_files]

        for p in tier_paths:
            try:
                valid_paths.append(str(p.relative_to(PROJECT_ROOT)))
            except ValueError:
                valid_paths.append(str(p))

    return valid_paths


def get_context_bundle(
    task_description: str, task_type: str | None = None
) -> list[str]:
    """
    Get context bundle for a task.

    Args:
        task_description: Description of the task
        task_type: Explicit task type ('development', 'documentation', 'debugging',
                  'planning', 'general'). If None, will auto-detect from description.

    Returns:
        List of file paths to include in context bundle

    Raises:
        FileNotFoundError: If bundle definition file not found
        yaml.YAMLError: If bundle YAML is invalid
    """
    # Determine task type
    if task_type is None:
        task_type = analyze_task_type(task_description)

    # Load bundle definition
    bundle_def = load_bundle_definition(task_type)

    # Validate and get file paths
    file_paths = validate_bundle_files(bundle_def)

    return file_paths


def print_context_bundle(task_description: str, task_type: str | None = None) -> None:
    """
    Print context bundle file paths to stdout (one per line).

    Args:
        task_description: Description of the task
        task_type: Explicit task type (optional)
    """
    file_paths = get_context_bundle(task_description, task_type)

    for path in file_paths:
        print(path)


def list_available_bundles() -> list[str]:
    """
    List all available bundle names from framework and plugins.

    Searches:
    1. Framework bundles: AgentQMS/knowledge/context_bundles/
    2. Plugin bundles: .agentqms/plugins/context_bundles/ (via registry)

    Returns:
        List of bundle names (without .yaml extension), sorted and deduplicated
    """
    bundles: set[str] = set()

    # Framework bundles
    if BUNDLES_DIR.exists():
        for f in BUNDLES_DIR.glob("*.yaml"):
            if f.stem != "README":
                bundles.add(f.stem)

    # Plugin-registered bundles
    if PLUGINS_AVAILABLE:
        try:
            registry = get_plugin_registry()
            plugin_bundles = registry.get_context_bundles()
            bundles.update(plugin_bundles.keys())
        except Exception:
            pass  # Continue with framework bundles only

    return sorted(bundles)


def auto_suggest_context(task_description: str) -> dict[str, Any]:
    """
    Automatically suggest context bundle and related information based on task description.
    
    This function analyzes the task description and returns suggestions for:
    - Which context bundle to load
    - Related workflows to consider
    - Tools that might be useful
    
    Args:
        task_description: Description of the current task
        
    Returns:
        Dictionary with suggestions:
        - task_type: Detected task type
        - context_bundle: Suggested bundle name
        - bundle_files: List of files in the suggested bundle
        - suggested_workflows: List of workflow names
        - suggested_tools: List of tool names
    """
    # Detect task type
    detected_type = analyze_task_type(task_description)
    
    # Get context bundle files
    try:
        bundle_files = get_context_bundle(task_description, detected_type)
    except FileNotFoundError:
        bundle_files = []
    
    # Try to import workflow detector for enhanced suggestions
    suggestions: dict[str, Any] = {
        "task_type": detected_type,
        "context_bundle": detected_type,
        "bundle_files": bundle_files,
        "suggested_workflows": [],
        "suggested_tools": [],
    }
    
    try:
        from AgentQMS.agent_tools.core.workflow_detector import suggest_workflows
        
        workflow_suggestions = suggest_workflows(task_description)
        suggestions["suggested_workflows"] = workflow_suggestions.get("workflows", [])
        suggestions["suggested_tools"] = workflow_suggestions.get("tools", [])
        if workflow_suggestions.get("artifact_type"):
            suggestions["artifact_type"] = workflow_suggestions["artifact_type"]
    except ImportError:
        # Workflow detector not available, use basic suggestions
        if detected_type == "development":
            suggestions["suggested_workflows"] = ["create-plan", "validate"]
            suggestions["suggested_tools"] = ["artifact_workflow", "validate_artifacts"]
        elif detected_type == "documentation":
            suggestions["suggested_workflows"] = ["docs-generate", "docs-validate-links"]
            suggestions["suggested_tools"] = ["auto_generate_index", "validate_links"]
        elif detected_type == "debugging":
            suggestions["suggested_workflows"] = ["create-bug-report", "validate"]
            suggestions["suggested_tools"] = ["artifact_workflow", "validate_artifacts"]
        elif detected_type == "planning":
            suggestions["suggested_workflows"] = ["create-plan", "create-design"]
            suggestions["suggested_tools"] = ["artifact_workflow"]
    
    return suggestions


def main():
    """Main entry point for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate context bundles for AI agent tasks"
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Task description (will auto-detect task type)",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["development", "documentation", "debugging", "planning", "general"],
        help="Explicit task type",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available bundles",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-detect task type and suggest context bundle with workflows",
    )

    args = parser.parse_args()

    if args.list:
        bundles = list_available_bundles()
        if bundles:
            print("Available bundles:")
            for bundle in bundles:
                print(f"  - {bundle}")
        else:
            print("No bundles found. Create YAML bundle definitions in:")
            print(f"  {BUNDLES_DIR}/")
    elif args.auto and args.task:
        # Auto-suggest mode
        suggestions = auto_suggest_context(args.task)
        print(f"Task Type: {suggestions['task_type']}")
        print(f"Context Bundle: {suggestions['context_bundle']}")
        print(f"\nBundle Files ({len(suggestions['bundle_files'])}):")
        for f in suggestions['bundle_files']:
            print(f"  - {f}")
        if suggestions.get("suggested_workflows"):
            print("\nSuggested Workflows:")
            for wf in suggestions["suggested_workflows"]:
                print(f"  - {wf}")
        if suggestions.get("suggested_tools"):
            print("\nSuggested Tools:")
            for tool in suggestions["suggested_tools"]:
                print(f"  - {tool}")
    elif args.task:
        print_context_bundle(args.task, args.type)
    elif args.type:
        # Use a generic description if only type is provided
        print_context_bundle(f"task type: {args.type}", args.type)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

