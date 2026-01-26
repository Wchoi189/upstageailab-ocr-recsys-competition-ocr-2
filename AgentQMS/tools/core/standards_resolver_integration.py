#!/usr/bin/env python3
"""
Standards Resolver Integration
Bridges ADS v2.0 resolver with context loading system.

This module integrates the new registry-based resolver (resolve_standards.py)
with the existing context bundle loading system, enabling:
- Automatic standard resolution based on task/path/keywords
- Seamless integration with token budgeting
- Backward compatibility with existing context bundles
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path if needed
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import resolver functions
try:
    from AgentQMS.tools.resolve_standards import (
        load_registry,
        resolve_by_task,
        resolve_by_path,
        resolve_by_keywords,
        expand_dependencies,
        get_standard_paths,
    )
    RESOLVER_AVAILABLE = True
except ImportError:
    RESOLVER_AVAILABLE = False


class StandardsResolverIntegration:
    """
    Integration layer between ADS v2.0 resolver and context loading.
    """

    def __init__(self, use_cache: bool = True):
        """
        Initialize resolver integration.

        Args:
            use_cache: Whether to use binary cache for registry loading
        """
        self.use_cache = use_cache
        self._registry: Optional[Dict[str, Any]] = None

    def _ensure_registry(self) -> Dict[str, Any]:
        """Lazy-load registry."""
        if self._registry is None and RESOLVER_AVAILABLE:
            self._registry = load_registry(use_cache=self.use_cache)
        return self._registry or {}

    def resolve_standards_for_task(
        self,
        task_description: str,
        include_deps: bool = True,
        include_tier1: bool = True,
    ) -> List[str]:
        """
        Resolve standards relevant to a task description.

        Args:
            task_description: Description of the task
            include_deps: Whether to include dependencies
            include_tier1: Whether to auto-include Tier 1 standards

        Returns:
            List of standard file paths (relative to project root)
        """
        if not RESOLVER_AVAILABLE:
            return []

        registry = self._ensure_registry()
        if not registry:
            return []

        # Try multiple resolution strategies
        standard_ids = set()

        # 1. Keyword-based resolution (extract keywords from description)
        keywords = self._extract_keywords(task_description)
        if keywords:
            ids = resolve_by_keywords(registry, keywords, fuzzy=True, threshold=0.7)
            standard_ids.update(ids)

        # 2. If no matches, try analyzing task type
        if not standard_ids:
            task_type = self._infer_task_type(task_description)
            if task_type:
                ids = resolve_by_task(registry, task_type)
                standard_ids.update(ids)

        # Expand dependencies if requested
        if standard_ids and include_deps:
            standard_ids = expand_dependencies(
                registry,
                list(standard_ids),
                include_tier1=include_tier1
            )

        # Get file paths
        if standard_ids:
            return get_standard_paths(registry, standard_ids)

        return []

    def resolve_standards_for_path(
        self,
        file_path: str,
        include_deps: bool = True,
        include_tier1: bool = True,
    ) -> List[str]:
        """
        Resolve standards relevant to a file path.

        Args:
            file_path: File path being modified
            include_deps: Whether to include dependencies
            include_tier1: Whether to auto-include Tier 1 standards

        Returns:
            List of standard file paths
        """
        if not RESOLVER_AVAILABLE:
            return []

        registry = self._ensure_registry()
        if not registry:
            return []

        # Resolve by path pattern
        standard_ids = resolve_by_path(registry, file_path)

        # Expand dependencies
        if standard_ids and include_deps:
            standard_ids = expand_dependencies(
                registry,
                standard_ids,
                include_tier1=include_tier1
            )

        # Get file paths
        if standard_ids:
            return get_standard_paths(registry, standard_ids)

        return []

    def _extract_keywords(self, description: str) -> List[str]:
        """
        Extract potential keywords from task description.

        Args:
            description: Task description

        Returns:
            List of keywords
        """
        # Simple keyword extraction - split and normalize
        description_lower = description.lower()

        # Common keywords to look for
        keyword_patterns = [
            "config", "hydra", "model", "training", "inference",
            "dataset", "architecture", "ocr", "detection", "recognition",
            "debug", "test", "documentation", "artifact", "workflow",
            "standard", "compliance", "validation"
        ]

        keywords = [kw for kw in keyword_patterns if kw in description_lower]
        return keywords

    def _infer_task_type(self, description: str) -> Optional[str]:
        """
        Infer task type from description.

        Args:
            description: Task description

        Returns:
            Task type or None
        """
        description_lower = description.lower()

        # Map keywords to task types
        task_type_map = {
            "config": "config_files",
            "hydra": "config_files",
            "model": "code_changes",
            "architecture": "code_changes",
            "document": "documentation",
            "artifact": "documentation",
            "test": "code_changes",
            "debug": "code_changes",
        }

        for keyword, task_type in task_type_map.items():
            if keyword in description_lower:
                return task_type

        return None

    def get_standards_as_bundle_files(
        self,
        standard_paths: List[str],
        mode: str = "full"
    ) -> List[Dict[str, Any]]:
        """
        Convert standard file paths to context bundle format.

        Args:
            standard_paths: List of standard file paths
            mode: Loading mode ("full", "structure", "reference")

        Returns:
            List of file specs compatible with context bundle system
        """
        bundle_files = []

        for path in standard_paths:
            # Infer tier from path
            tier = self._infer_tier_from_path(path)

            file_spec = {
                "path": path,
                "mode": mode,
                "tier": tier,
                "description": f"Standard: {Path(path).stem}",
            }
            bundle_files.append(file_spec)

        return bundle_files

    def _infer_tier_from_path(self, path: str) -> str:
        """
        Infer tier from file path.

        Args:
            path: File path

        Returns:
            Tier string (e.g., "tier1", "tier2")
        """
        path_lower = path.lower()

        if "tier1" in path_lower:
            return "tier1"
        elif "tier2" in path_lower:
            return "tier2"
        elif "tier3" in path_lower:
            return "tier3"
        elif "tier4" in path_lower:
            return "tier4"

        return "tier2"  # Default to tier2


# Singleton instance
_INTEGRATION = StandardsResolverIntegration()


# Convenience functions for backward compatibility
def resolve_standards_for_task(
    task_description: str,
    include_deps: bool = True,
    include_tier1: bool = True,
) -> List[str]:
    """
    Resolve standards for a task description.

    Args:
        task_description: Description of the task
        include_deps: Whether to include dependencies
        include_tier1: Whether to auto-include Tier 1 standards

    Returns:
        List of standard file paths
    """
    return _INTEGRATION.resolve_standards_for_task(
        task_description,
        include_deps=include_deps,
        include_tier1=include_tier1,
    )


def resolve_standards_for_path(
    file_path: str,
    include_deps: bool = True,
    include_tier1: bool = True,
) -> List[str]:
    """
    Resolve standards for a file path.

    Args:
        file_path: File path being modified
        include_deps: Whether to include dependencies
        include_tier1: Whether to auto-include Tier 1 standards

    Returns:
        List of standard file paths
    """
    return _INTEGRATION.resolve_standards_for_path(
        file_path,
        include_deps=include_deps,
        include_tier1=include_tier1,
    )


def get_standards_as_bundle_files(
    standard_paths: List[str],
    mode: str = "full"
) -> List[Dict[str, Any]]:
    """
    Convert standard paths to context bundle format.

    Args:
        standard_paths: List of standard file paths
        mode: Loading mode

    Returns:
        List of file specs for context bundle system
    """
    return _INTEGRATION.get_standards_as_bundle_files(standard_paths, mode=mode)


def main():
    """CLI for testing resolver integration."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Standards Resolver Integration - Bridge to Context Loading"
    )
    parser.add_argument("--task", help="Task description to resolve standards for")
    parser.add_argument("--path", help="File path to resolve standards for")
    parser.add_argument("--no-deps", action="store_true", help="Don't expand dependencies")
    parser.add_argument("--no-tier1", action="store_true", help="Don't include Tier 1")
    parser.add_argument("--mode", default="full", choices=["full", "structure", "reference"],
                       help="Loading mode for standards")

    args = parser.parse_args()

    if not RESOLVER_AVAILABLE:
        print("Error: Resolver not available. Check imports.", file=sys.stderr)
        return 1

    standard_paths = []

    if args.task:
        print(f"Resolving standards for task: {args.task}")
        standard_paths = resolve_standards_for_task(
            args.task,
            include_deps=not args.no_deps,
            include_tier1=not args.no_tier1,
        )
    elif args.path:
        print(f"Resolving standards for path: {args.path}")
        standard_paths = resolve_standards_for_path(
            args.path,
            include_deps=not args.no_deps,
            include_tier1=not args.no_tier1,
        )
    else:
        parser.print_help()
        return 1

    if standard_paths:
        print(f"\nResolved {len(standard_paths)} standard(s):")
        for path in standard_paths:
            print(f"  - {path}")

        # Convert to bundle format
        bundle_files = get_standards_as_bundle_files(standard_paths, mode=args.mode)
        print(f"\nBundle format ({len(bundle_files)} files):")
        for file_spec in bundle_files:
            print(f"  - {file_spec['path']} [{file_spec['mode']}] (Tier: {file_spec['tier']})")
    else:
        print("No standards matched the query.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
