#!/usr/bin/env python3
"""
ADS v2.0 Standards Resolver
High-performance resolution tool with fuzzy search, binary caching, and dependency expansion.

Features:
- Task-based resolution (e.g., --task config_files)
- Path-based resolution (e.g., --path ocr/models/vgg.py)
- Keyword search with fuzzy matching (e.g., --keywords "hydra config")
- Binary caching for sub-10ms queries
- Automatic Tier 1 dependency inclusion
- Shell piping support (--paths-only)
"""

import argparse
import pickle
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

# Project root and paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STANDARDS_DIR = PROJECT_ROOT / "AgentQMS" / "standards"
REGISTRY_PATH = STANDARDS_DIR / "registry.yaml"
CACHE_PATH = STANDARDS_DIR / ".ads_cache.pickle"

# Cache TTL in seconds (default: 1 hour)
CACHE_TTL = 3600


class ResolverError(Exception):
    """Raised when resolution fails."""
    pass


class RegistryCache:
    """Binary cache for registry data with TTL."""

    def __init__(self, cache_path: Path, ttl: int = CACHE_TTL):
        self.cache_path = cache_path
        self.ttl = ttl
        self._data: Optional[Dict[str, Any]] = None
        self._timestamp: float = 0.0

    def load(self) -> Optional[Dict[str, Any]]:
        """Load cache if valid, otherwise return None."""
        if not self.cache_path.exists():
            return None

        try:
            with open(self.cache_path, "rb") as f:
                cached = pickle.load(f)
                cache_age = time.time() - cached.get("timestamp", 0)

                if cache_age < self.ttl:
                    self._data = cached.get("data")
                    self._timestamp = cached.get("timestamp", 0)
                    return self._data
                else:
                    # Cache expired
                    return None
        except Exception:
            return None

    def save(self, data: Dict[str, Any]) -> None:
        """Save data to cache with timestamp."""
        try:
            with open(self.cache_path, "wb") as f:
                pickle.dump(
                    {"data": data, "timestamp": time.time()},
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL
                )
        except Exception as e:
            # Cache write failures should not break the tool
            print(f"Warning: Failed to write cache: {e}", file=sys.stderr)

    def is_valid(self) -> bool:
        """Check if cache is loaded and valid."""
        return self._data is not None

    @property
    def age(self) -> float:
        """Get cache age in seconds."""
        return time.time() - self._timestamp


def load_registry(use_cache: bool = True) -> Dict[str, Any]:
    """
    Load registry with optional caching.

    Args:
        use_cache: Whether to use binary cache

    Returns:
        Registry dictionary
    """
    if use_cache:
        cache = RegistryCache(CACHE_PATH)
        cached_data = cache.load()

        if cached_data:
            return cached_data

    # Load from YAML
    if not REGISTRY_PATH.exists():
        raise ResolverError(f"Registry not found: {REGISTRY_PATH}")

    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = yaml.safe_load(f)

    # Save to cache
    if use_cache:
        cache = RegistryCache(CACHE_PATH)
        cache.save(registry)

    return registry


def fuzzy_match(query: str, target: str, threshold: float = 0.8) -> float:
    """
    Compute fuzzy match score between query and target.

    Args:
        query: Search query string
        target: Target string to match against
        threshold: Minimum score threshold (0.0-1.0)

    Returns:
        Match score (0.0-1.0), or 0.0 if below threshold
    """
    score = SequenceMatcher(None, query.lower(), target.lower()).ratio()
    return score if score >= threshold else 0.0


def resolve_by_task(registry: Dict[str, Any], task: str) -> List[str]:
    """
    Resolve standards by task type.

    Args:
        registry: Registry dictionary
        task: Task identifier (e.g., "config_files", "code_changes")

    Returns:
        List of standard IDs
    """
    # Check if registry has task_mappings (v1.0 format)
    if "task_mappings" in registry:
        task_data = registry["task_mappings"].get(task)
        if task_data:
            # Extract standard IDs from paths
            paths = task_data.get("standards", [])
            # Map paths to IDs (this is a heuristic for v1.0 compatibility)
            return [Path(p).stem for p in paths]

    # Check standards with triggers (v2.0 format)
    standards = registry.get("standards", {})
    matched_ids = []

    for std_id, std_data in standards.items():
        triggers = std_data.get("triggers", {})
        keywords = triggers.get("keywords", [])

        # Check if task matches any keyword
        if any(task.lower() in kw.lower() for kw in keywords):
            matched_ids.append(std_id)

    return matched_ids


def resolve_by_path(registry: Dict[str, Any], file_path: str) -> List[str]:
    """
    Resolve standards by file path pattern.

    Args:
        registry: Registry dictionary
        file_path: File path or pattern

    Returns:
        List of standard IDs
    """
    standards = registry.get("standards", {})
    matched_ids = []
    file_path_lower = file_path.lower()

    for std_id, std_data in standards.items():
        triggers = std_data.get("triggers", {})
        path_patterns = triggers.get("path_patterns", [])

        # Check if file path matches any pattern
        for pattern in path_patterns:
            pattern_lower = pattern.lower()

            # Simple pattern matching (glob-style ** and *)
            if "**" in pattern_lower:
                # Match anywhere in path
                parts = pattern_lower.split("**")
                if all(part in file_path_lower for part in parts if part):
                    matched_ids.append(std_id)
                    break
            elif "*" in pattern_lower:
                # Basic wildcard matching
                parts = pattern_lower.split("*")
                if all(part in file_path_lower for part in parts if part):
                    matched_ids.append(std_id)
                    break
            elif pattern_lower in file_path_lower:
                matched_ids.append(std_id)
                break

    return matched_ids


def resolve_by_keywords(
    registry: Dict[str, Any],
    keywords: List[str],
    fuzzy: bool = False,
    threshold: float = 0.8
) -> List[str]:
    """
    Resolve standards by keyword search.

    Args:
        registry: Registry dictionary
        keywords: List of keywords to search
        fuzzy: Enable fuzzy matching
        threshold: Fuzzy match threshold (0.0-1.0)

    Returns:
        List of standard IDs with match scores
    """
    standards = registry.get("standards", {})
    keyword_index = registry.get("keyword_index", {})
    matched_ids = set()

    for query_kw in keywords:
        query_lower = query_kw.lower()

        # Exact match from index
        if query_lower in keyword_index:
            matched_ids.update(keyword_index[query_lower])

        # Fuzzy matching
        if fuzzy:
            for std_id, std_data in standards.items():
                std_keywords = std_data.get("keywords", [])
                std_threshold = std_data.get("fuzzy_threshold", threshold)

                for std_kw in std_keywords:
                    score = fuzzy_match(query_lower, std_kw, std_threshold)
                    if score > 0:
                        matched_ids.add(std_id)

    return list(matched_ids)


def expand_dependencies(
    registry: Dict[str, Any],
    standard_ids: List[str],
    include_tier1: bool = True
) -> List[str]:
    """
    Expand standard IDs to include dependencies.

    Args:
        registry: Registry dictionary
        standard_ids: Initial list of standard IDs
        include_tier1: Auto-include all Tier 1 standards

    Returns:
        Expanded list of standard IDs (deduplicated, sorted)
    """
    standards = registry.get("standards", {})
    expanded = set(standard_ids)

    # Add Tier 1 standards (constitutional laws apply to everything)
    if include_tier1:
        tier1_ids = registry.get("tier_index", {}).get("tier1", [])
        expanded.update(tier1_ids)

    # Recursively add dependencies
    def add_deps(std_id: str, visited: Set[str]):
        if std_id in visited or std_id not in standards:
            return
        visited.add(std_id)

        deps = standards[std_id].get("dependencies", [])
        for dep_id in deps:
            expanded.add(dep_id)
            add_deps(dep_id, visited)

    for std_id in list(standard_ids):
        add_deps(std_id, set())

    return sorted(expanded)


def get_standard_paths(registry: Dict[str, Any], standard_ids: List[str]) -> List[str]:
    """
    Get file paths for standard IDs.

    Args:
        registry: Registry dictionary
        standard_ids: List of standard IDs

    Returns:
        List of relative file paths
    """
    standards = registry.get("standards", {})
    paths = []

    for std_id in standard_ids:
        std_data = standards.get(std_id, {})
        file_path = std_data.get("file_path", std_data.get("_file_path"))
        if file_path:
            paths.append(file_path)

    return paths


def display_results(
    registry: Dict[str, Any],
    standard_ids: List[str],
    paths_only: bool = False,
    verbose: bool = False
) -> None:
    """
    Display resolution results.

    Args:
        registry: Registry dictionary
        standard_ids: Resolved standard IDs
        paths_only: Only output file paths (for shell piping)
        verbose: Show detailed information
    """
    standards = registry.get("standards", {})

    if paths_only:
        # Shell-friendly output
        paths = get_standard_paths(registry, standard_ids)
        for path in paths:
            print(path)
        return

    if not standard_ids:
        print("No standards matched the query.")
        return

    print(f"\n📋 Resolved {len(standard_ids)} standard(s):\n")

    for std_id in standard_ids:
        std_data = standards.get(std_id, {})
        tier = std_data.get("tier", "?")
        priority = std_data.get("priority", "unknown")
        description = std_data.get("description", "")
        file_path = std_data.get("file_path", std_data.get("_file_path", ""))

        # Priority emoji
        priority_emoji = {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🟢",
        }.get(priority, "⚪")

        print(f"{priority_emoji} {std_id} [Tier {tier}] - {description}")

        if verbose:
            print(f"   Path: {file_path}")
            keywords = std_data.get("keywords", [])
            if keywords:
                print(f"   Keywords: {', '.join(keywords[:5])}")
            deps = std_data.get("dependencies", [])
            if deps:
                print(f"   Dependencies: {', '.join(deps)}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="ADS v2.0 Standards Resolver - High-performance standard resolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --task config_files
  %(prog)s --path ocr/models/vgg.py
  %(prog)s --keywords "hydra configuration"
  %(prog)s --query "hidra config" --fuzzy
  %(prog)s --task config_files --paths-only | xargs cat
        """
    )

    parser.add_argument("--task", help="Resolve by task type")
    parser.add_argument("--path", help="Resolve by file path")
    parser.add_argument("--keywords", help="Resolve by keywords (space-separated)")
    parser.add_argument("--query", help="Shorthand for --keywords (single query)")
    parser.add_argument("--fuzzy", action="store_true", help="Enable fuzzy keyword matching")
    parser.add_argument("--threshold", type=float, default=0.8,
                       help="Fuzzy match threshold (0.0-1.0, default: 0.8)")
    parser.add_argument("--no-deps", action="store_true",
                       help="Don't auto-expand dependencies")
    parser.add_argument("--no-tier1", action="store_true",
                       help="Don't auto-include Tier 1 standards")
    parser.add_argument("--paths-only", action="store_true",
                       help="Output only file paths (for shell piping)")
    parser.add_argument("--no-cache", action="store_true", help="Disable binary cache")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Show detailed information")

    args = parser.parse_args()

    # Validate arguments
    if not any([args.task, args.path, args.keywords, args.query]):
        parser.error("Must specify --task, --path, --keywords, or --query")

    try:
        # Load registry (with timing)
        start_time = time.time()
        registry = load_registry(use_cache=not args.no_cache)
        load_time_ms = (time.time() - start_time) * 1000

        if args.verbose and not args.paths_only:
            cache_status = "cached" if load_time_ms < 10 else "loaded"
            print(f"Registry {cache_status} in {load_time_ms:.2f}ms\n")

        # Resolve standards
        standard_ids = []

        if args.task:
            standard_ids.extend(resolve_by_task(registry, args.task))

        if args.path:
            standard_ids.extend(resolve_by_path(registry, args.path))

        if args.keywords or args.query:
            keywords = args.keywords.split() if args.keywords else [args.query]
            standard_ids.extend(
                resolve_by_keywords(registry, keywords, args.fuzzy, args.threshold)
            )

        # Remove duplicates
        standard_ids = list(dict.fromkeys(standard_ids))

        # Expand dependencies
        if not args.no_deps:
            standard_ids = expand_dependencies(
                registry, standard_ids, include_tier1=not args.no_tier1
            )

        # Display results
        display_results(registry, standard_ids, args.paths_only, args.verbose)

        return 0

    except ResolverError as e:
        print(f"❌ Resolution failed: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"💥 Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
