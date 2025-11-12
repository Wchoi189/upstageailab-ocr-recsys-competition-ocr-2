#!/usr/bin/env python3
"""
Context Bundle Generator Core

Generates task-specific context bundles from YAML definitions.
Supports automatic task type detection, glob patterns, and freshness checking.

Usage:
    from core.context_bundle import get_context_bundle

    # Automatic task type detection
    files = get_context_bundle("implement new feature in streamlit app")

    # Explicit task type
    files = get_context_bundle("fix bug", task_type="debugging")
"""

import glob
import importlib.util
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print(
        "ERROR: PyYAML not installed. Install with: pip install pyyaml", file=sys.stderr
    )
    sys.exit(1)


def _load_bootstrap():
    if "scripts._bootstrap" in sys.modules:
        return sys.modules["scripts._bootstrap"]

    current_dir = Path(__file__).resolve().parent
    for directory in (current_dir, *tuple(current_dir.parents)):
        candidate = directory / "_bootstrap.py"
        if candidate.exists():
            spec = importlib.util.spec_from_file_location(
                "scripts._bootstrap", candidate
            )
            if spec is None or spec.loader is None:  # pragma: no cover - defensive
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    raise RuntimeError("Could not locate scripts/_bootstrap.py")


_BOOTSTRAP = _load_bootstrap()
setup_project_paths = _BOOTSTRAP.setup_project_paths
get_path_resolver = _BOOTSTRAP.get_path_resolver

setup_project_paths()
_RESOLVER = get_path_resolver()
PROJECT_ROOT = _RESOLVER.config.project_root

# Default bundle directory
# Try to use docs_dir if available, otherwise use project_root / "docs"
try:
    docs_dir = _RESOLVER.config.docs_dir
except AttributeError:
    docs_dir = PROJECT_ROOT / "docs"
BUNDLES_DIR = docs_dir / "context_bundles"

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
    Load bundle definition from YAML file.

    Args:
        bundle_name: Name of bundle (without .yaml extension)

    Returns:
        Bundle definition dictionary

    Raises:
        FileNotFoundError: If bundle file doesn't exist
        yaml.YAMLError: If YAML is invalid
    """
    bundle_path = BUNDLES_DIR / f"{bundle_name}.yaml"

    if not bundle_path.exists():
        available = ", ".join([f.stem for f in BUNDLES_DIR.glob("*.yaml")])
        raise FileNotFoundError(
            f"Bundle '{bundle_name}' not found at {bundle_path}. "
            f"Available bundles: {available}"
        )

    with bundle_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
    import time

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

    Validates:
    - Files exist
    - Files are fresh (modified within 30 days)
    - Glob patterns are expanded correctly
    - Tier limits are respected

    Args:
        bundle_def: Bundle definition dictionary

    Returns:
        List of valid file paths (as strings)
    """
    valid_paths: list[str] = []

    tiers = bundle_def.get("tiers", {})

    for tier_key in sorted(tiers.keys()):  # tier1, tier2, tier3...
        tier = tiers[tier_key]
        tier.get("name", tier_key)
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
                project_root = Path(__file__).parent.parent.parent
                file_path = project_root / file_path_str

                if file_path.exists() and is_fresh(file_path):
                    tier_paths.append(file_path)

        # Apply tier limit and add to valid paths
        if max_files and len(tier_paths) > max_files:
            tier_paths = tier_paths[:max_files]

        valid_paths.extend([str(p.relative_to(project_root)) for p in tier_paths])

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
    List all available bundle names.

    Returns:
        List of bundle names (without .yaml extension)
    """
    if not BUNDLES_DIR.exists():
        return []

    return [f.stem for f in BUNDLES_DIR.glob("*.yaml") if f.stem != "README"]


if __name__ == "__main__":
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

    args = parser.parse_args()

    if args.list:
        bundles = list_available_bundles()
        print("Available bundles:")
        for bundle in bundles:
            print(f"  - {bundle}")
    elif args.task:
        print_context_bundle(args.task, args.type)
    elif args.type:
        # Use a generic description if only type is provided
        print_context_bundle(f"task type: {args.type}", args.type)
    else:
        parser.print_help()
        sys.exit(1)
