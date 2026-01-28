#!/usr/bin/env python3
"""
Context Engine 2.0

High-performance context bundling with caching, parallel I/O, and token budgeting.
Refs: AgentQMS/tools/core/context_bundle.py
"""

import glob
import heapq
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

try:
    import yaml
    try:
        from yaml import CSafeLoader as Loader
    except ImportError:
        from yaml import SafeLoader as Loader
except ImportError:
    print("ERROR: PyYAML not installed. Fix with: uv sync", file=sys.stderr)
    sys.exit(1)

from AgentQMS.tools.utils.paths import get_project_root
from AgentQMS.tools.utils.system.runtime import ensure_project_root_on_sys_path
from AgentQMS.tools.utils.config.loader import ConfigLoader

ensure_project_root_on_sys_path()
PROJECT_ROOT = get_project_root()
BUNDLES_DIR = PROJECT_ROOT / "AgentQMS" / ".agentqms" / "plugins" / "context_bundles"
CONFIG_LOADER = ConfigLoader()

# Try to import plugin registry
try:
    from AgentQMS.tools.core.plugins import get_plugin_registry
    PLUGINS_AVAILABLE = True
except ImportError:
    PLUGINS_AVAILABLE = False

# Try to import ADS v2.0 resolver integration
try:
    from AgentQMS.tools.core.standards_resolver_integration import (
        resolve_standards_for_task,
        resolve_standards_for_path,
        get_standards_as_bundle_files,
    )
    RESOLVER_AVAILABLE = True
except ImportError:
    RESOLVER_AVAILABLE = False


class ContextEngine:
    """
    Stateful context engine with caching and token budgeting.
    """

    def __init__(self, max_tokens: int = 32000):
        self.max_tokens = max_tokens
        self.stat_cache: dict[str, os.stat_result] = {}
        self.keyword_cache: dict[str, list[str]] | None = None
        self.bundle_cache: dict[str, Any] = {}
        self.timestamp = time.time()
        self._thread_pool: ThreadPoolExecutor | None = None

    def clear_cache(self):
        """Reset per-request caches."""
        self.stat_cache.clear()
        self.bundle_cache.clear() # Keywords are persistent usually, but bundles might change?
        # Actually bundle defs likely don't change fast. But file existence does.
        # stat_cache is key for freshness.
        self.timestamp = time.time()

    def get_executor(self) -> ThreadPoolExecutor:
        if self._thread_pool is None:
            self._thread_pool = ThreadPoolExecutor(max_workers=8)
        return self._thread_pool

    def _get_stat(self, path: Path) -> os.stat_result | None:
        path_str = str(path)
        if path_str in self.stat_cache:
            return self.stat_cache[path_str]
        try:
            st = path.stat()
            self.stat_cache[path_str] = st
            return st
        except FileNotFoundError:
            self.stat_cache[path_str] = None # Negative cache
            return None

    def load_task_keywords(self) -> dict[str, list[str]]:
        if self.keyword_cache is not None:
             return self.keyword_cache

        config_path = PROJECT_ROOT / "AgentQMS/standards/tier2-framework/discovery/discovery-rules.yaml"
        if not config_path.exists():
            return {}

        # relying on ConfigLoader for parsing
        raw_config = CONFIG_LOADER.get_config(config_path, defaults={})
        
        # Filter metadata keys from discovery-rules which contains other fields
        # valid task types should have a list of strings as values
        self.keyword_cache = {}
        excluded_keys = {"keywords", "dependencies"}
        for k, v in raw_config.items():
            if k not in excluded_keys and isinstance(v, list):
                self.keyword_cache[k] = v
                
        return self.keyword_cache

    def analyze_task_type(self, description: str) -> str:
        """
        Analyze task description task type based on keywords.
        """
        description_lower = description.lower()
        scores = {}
        keywords_map = self.load_task_keywords()

        for task_type, keywords in keywords_map.items():
            score = sum(1 for keyword in keywords if keyword in description_lower)
            if score > 0:
                scores[task_type] = score

        if scores:
            task_type = max(scores.items(), key=lambda x: x[1])[0]

            # Map task types to specialized bundles
            TASK_TO_BUNDLE_MAP = {
                "development": "pipeline-development",
                "documentation": "documentation-update",
                "debugging": "ocr-debugging",
                "planning": "project-compass",
                "hydra-configuration": "hydra-configuration",
                "hydra-v5-patterns": "hydra-configuration",
                "ocr-architecture": "pipeline-development",
                "agent-configuration": "agent-configuration",
            }
            return TASK_TO_BUNDLE_MAP.get(task_type, task_type)

        return "compliance-check"

    def load_bundle_definition(self, bundle_name: str) -> dict[str, Any]:
        """
        Load bundle definition with caching.
        """
        if bundle_name in self.bundle_cache:
            return self.bundle_cache[bundle_name]

        # 1. Framework bundles
        bundle_path = BUNDLES_DIR / f"{bundle_name}.yaml"
        if bundle_path.exists():
            try:
                with bundle_path.open("r", encoding="utf-8") as f:
                    data = yaml.load(f, Loader=Loader)
                    self.bundle_cache[bundle_name] = data
                    return data
            except Exception as e:
                # If error, try plugins
                print(f"Warning: Failed to load {bundle_path}: {e}", file=sys.stderr)

        # 2. Plugin bundles
        if PLUGINS_AVAILABLE:
            try:
                registry = get_plugin_registry()
                plugin_bundle = registry.get_context_bundle(bundle_name)
                if plugin_bundle:
                    self.bundle_cache[bundle_name] = plugin_bundle
                    return plugin_bundle
            except Exception:
                pass

        # Not found fallback
        available = self.list_available_bundles()
        available_str = ", ".join(available) if available else "none"
        raise FileNotFoundError(f"Bundle '{bundle_name}' not found. Available bundles: {available_str}")

    def list_available_bundles(self) -> list[str]:
        bundles = set()
        if BUNDLES_DIR.exists():
            for f in BUNDLES_DIR.glob("*.yaml"):
                 if f.stem != "README":
                     bundles.add(f.stem)
        if PLUGINS_AVAILABLE:
             try:
                 registry = get_plugin_registry()
                 bundles.update(registry.get_context_bundles().keys())
             except Exception:
                 pass
        return sorted(bundles)

    def is_fresh(self, path: Path, days: int = 30) -> bool:
        """
        Check if file is fresh using cached stat.
        """
        st = self._get_stat(path)
        if not st:
            return False
        # Calculate days since modification
        days_since = (self.timestamp - st.st_mtime) / (24 * 60 * 60)
        return days_since <= days

    def expand_glob_pattern(self, pattern: str, max_files: int | None = None) -> list[Path]:
        """
        Expand glob with efficient sorting and limits.
        """
        if not Path(pattern).is_absolute():
            pattern_full = str(PROJECT_ROOT / pattern)
        else:
            pattern_full = pattern

        # Use iglob for iterator to avoid building full list in memory if possible
        # recursive=True matches **
        matches_iter = glob.iglob(pattern_full, recursive=True)

        # Filter for files and cache stat results
        valid_matches = []

        for p_str in matches_iter:
            p = Path(p_str)
            # is_file checks os.stat, but we want to cache it
            # But we can't easily cache "is_file" without statting.
            # _get_stat does the stat.
            st = self._get_stat(p)
            if st and not os.path.isdir(p_str): # st.st_mode checking is cleaner but isdir works
                 # Actually os.stat_result doesn't tell us if it's a dir easily without mode check
                 # stat.S_ISREG(st.st_mode)
                 import stat
                 if stat.S_ISREG(st.st_mode):
                     valid_matches.append((p, st.st_mtime))

        if not valid_matches:
            return []

        # Optimization: Use nlargest if limiting
        if max_files and len(valid_matches) > max_files:
            # Sort by mtime descending (newest first)
            top_matches = heapq.nlargest(max_files, valid_matches, key=lambda x: x[1])
            return [x[0] for x in top_matches]
        else:
            # Full sort needed
            valid_matches.sort(key=lambda x: x[1], reverse=True)
            return [x[0] for x in valid_matches]

    def _estimate_file_tokens(self, file_info: dict[str, Any]) -> int:
        """Helper for parallel token estimation."""
        path_str = file_info.get("path")
        if not path_str: return 0

        mode = file_info.get("mode", "full")
        path = PROJECT_ROOT / path_str

        if mode == "reference":
            desc_len = len(file_info.get("description", ""))
            return (desc_len + len(path_str) + 50) // 4

        # Check stat cache first
        st = self._get_stat(path)
        if not st:
            return 0

        try:
             # Just read the file - I/O bound
             content = path.read_text(encoding="utf-8")
             raw_count = len(content) // 4
             if mode == "structure":
                 return int(raw_count * 0.2)
             return raw_count
        except Exception:
            return 0

    def validate_bundle_files(self, bundle_def: dict[str, Any]) -> list[dict[str, Any]]:
        """Validate and resolve file paths from bundle definition."""
        valid_files = []
        tiers = bundle_def.get("tiers", {})

        # Process tiers in order (tier1, tier2...)
        for tier_key in sorted(tiers.keys()):
             tier = tiers[tier_key]
             tier_max_files = tier.get("max_files")
             files_spec = tier.get("files", [])
             tier_files = []

             for file_spec in files_spec:
                 # Normalize spec
                 if isinstance(file_spec, str):
                     spec = {
                         "path": file_spec,
                         "mode": "full",
                         "description": "",
                         "tier": tier_key
                     }
                 elif isinstance(file_spec, dict):
                     spec = file_spec.copy()
                     if "path" not in spec: continue
                     if "mode" not in spec: spec["mode"] = "full"
                     spec["tier"] = tier_key
                 else:
                     continue

                 path_str = spec["path"]

                 # Handle globs
                 if "*" in path_str or "**" in path_str:
                     # Glob expansion with NO LIMIT here, limit applied at Tier level usually
                     # But we should apply freshness per file if possible?
                     # Expanded uses is_fresh internally? No, expand_glob just returns sorted list.
                     # We need to apply freshness check.

                     expanded = self.expand_glob_pattern(path_str)

                     for p in expanded:
                         entry = spec.copy()
                         try:
                             # Make relative
                             rel_path = p.relative_to(PROJECT_ROOT)
                             entry["path"] = str(rel_path)
                         except ValueError:
                             entry["path"] = str(p)

                         if self.is_fresh(p):
                             tier_files.append(entry)
                 else:
                     # Direct file
                     path = PROJECT_ROOT / path_str
                     if self.is_fresh(path):
                         tier_files.append(spec)

             # Apply Tier-level max_files limit
             # Since files are populated in order of 'files' list, and globs are sorted by time...
             # We effectively keep explicit files first, then sorted glob results.
             if tier_max_files and len(tier_files) > tier_max_files:
                 tier_files = tier_files[:tier_max_files]

             valid_files.extend(tier_files)

        return valid_files

    def estimate_token_count_batch(self, files: list[dict[str, Any]]) -> tuple[int, dict[str, int]]:
        """Estimate tokens for all files in parallel."""
        path_counts = {}
        total_tokens = 0

        # Parallel execution
        executor = self.get_executor()
        results = list(executor.map(self._estimate_file_tokens, files))

        for f, count in zip(files, results):
            f["tokens"] = count
            path_counts[f["path"]] = count
            total_tokens += count

        return total_tokens, path_counts

    def apply_budget(self, files: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Enforce token budget by dropping lower priority files."""
        total_tokens, _ = self.estimate_token_count_batch(files)

        if total_tokens <= self.max_tokens:
            return files

        # Over budget strategy:
        # 1. Sort by Tier (tier1 > tier2 > ...)
        # 2. Within Tier, sort by Tokens (heuristic: keep smaller?) or Input Order?
        #    Usually input order (which implies relevance/recency for globs) is best.
        #    Let's preserve stability.

        # We need a stable sort that groups by tier.
        # Tiers are string keys "tier1", "tier2". Ascending string order works.

        # Group by tier
        files_by_tier: dict[str, list[dict[str, Any]]] = {}
        for f in files:
            t = f.get("tier", "tier99")
            if t not in files_by_tier:
                files_by_tier[t] = []
            files_by_tier[t].append(f)

        kept_files = []
        current_tokens = 0

        # Iterate tiers
        for tier in sorted(files_by_tier.keys()):
            for f in files_by_tier[tier]:
                if current_tokens + f["tokens"] <= self.max_tokens:
                    kept_files.append(f)
                    current_tokens += f["tokens"]
                else:
                    # Budget exceeded.
                    # If it's a critical tier (tier1), we might want to warn?
                    # For now, just drop.
                    # Optional: Could try to switch to 'structure' mode here if 'full'?
                    # That is complex (requires re-estimation).
                    pass

        return kept_files

    def get_context_bundle(
        self,
        description: str,
        task_type: str | None = None,
        use_resolver: bool = False,
        include_bundle: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Main method to generate bundle.

        Args:
            description: Task description
            task_type: Explicit task type (if known)
            use_resolver: Use ADS v2.0 resolver for standard resolution
            include_bundle: Include traditional bundle files (when use_resolver=True)

        Returns:
            List of file specifications with token estimates
        """
        self.clear_cache()

        files = []

        # Use ADS v2.0 resolver if requested and available
        if use_resolver and RESOLVER_AVAILABLE:
            try:
                # Resolve standards using ADS v2.0 registry
                standard_paths = resolve_standards_for_task(
                    description,
                    include_deps=True,
                    include_tier1=True,
                )

                if standard_paths:
                    # Convert to bundle format
                    resolver_files = get_standards_as_bundle_files(
                        standard_paths,
                        mode="full"
                    )
                    files.extend(resolver_files)
                    print(f"[ContextBundle] ADS v2.0 resolver: {len(resolver_files)} standards", file=sys.stderr)
            except Exception as e:
                print(f"[ContextBundle] Resolver error: {e}", file=sys.stderr)

        # Include traditional bundle if requested
        if include_bundle or not use_resolver:
            if task_type is None:
                task_type = self.analyze_task_type(description)

            try:
                bundle_def = self.load_bundle_definition(task_type)
                bundle_files = self.validate_bundle_files(bundle_def)
                files.extend(bundle_files)
            except FileNotFoundError:
                # Bundle not found, continue with resolver results only
                if not use_resolver:
                    # If not using resolver and bundle not found, this is an error
                    raise

        # Apply token budget
        final_files = self.apply_budget(files)

        return final_files


# Singleton instance for backward compatibility
_ENGINE = ContextEngine()

# Backward compatible exports
TASK_KEYWORDS = _ENGINE.load_task_keywords()

def analyze_task_type(description: str) -> str:
    return _ENGINE.analyze_task_type(description)

def get_context_bundle(
    task_description: str,
    task_type: str | None = None,
    use_resolver: bool = False,
    include_bundle: bool = True,
) -> list[dict[str, Any]]:
    return _ENGINE.get_context_bundle(
        task_description,
        task_type,
        use_resolver=use_resolver,
        include_bundle=include_bundle,
    )

def list_available_bundles() -> list[str]:
    return _ENGINE.list_available_bundles()

def estimate_token_count(files: list[dict[str, Any]]) -> dict[str, int]:
    # Compatibility wrapper - slightly inefficient as it re-creates executor potentially
    # or re-calculates. But `_ENGINE` is persistent.
    total, counts = _ENGINE.estimate_token_count_batch(files)
    return {
        "total_tokens": total,
        "file_counts": counts
    }


def auto_suggest_context(task_description: str) -> dict[str, Any]:
    """
    Automatically suggest context bundle and related information.
    """
    detected_type = _ENGINE.analyze_task_type(task_description)
    print(f"[ContextBundle] analyzing task: '{task_description}' -> detected type: '{detected_type}'", file=sys.stderr)

    try:
        # Use Engine directly
        bundle_files = _ENGINE.get_context_bundle(task_description, detected_type)
        print(f"[ContextBundle] found {len(bundle_files)} files in bundle '{detected_type}'", file=sys.stderr)
    except FileNotFoundError:
        print(f"[ContextBundle] warning: bundle '{detected_type}' not found", file=sys.stderr)
        bundle_files = []

    suggestions = {
        "task_type": detected_type,
        "context_bundle": detected_type,
        "bundle_files": bundle_files,
        "suggested_workflows": [],
        "suggested_tools": [],
    }

    # Workflow detection
    try:
        from AgentQMS.tools.core.workflow_detector import suggest_workflows
        workflow_suggestions = suggest_workflows(task_description)
        suggestions["suggested_workflows"] = workflow_suggestions.get("workflows", [])
        suggestions["suggested_tools"] = workflow_suggestions.get("tools", [])
        if workflow_suggestions.get("artifact_type"):
            suggestions["artifact_type"] = workflow_suggestions["artifact_type"]
    except ImportError:
        pass

    # Universal Utils Injection
    universal_utils = [{"path": "AgentQMS/tools/utils/paths.py", "mode": "structure", "description": "Path utilities", "tier": "tier0"}]
    current_paths = {f["path"] for f in suggestions["bundle_files"]}

    for util in universal_utils:
        if util["path"] not in current_paths:
            suggestions["bundle_files"].append(util)

    # Recalculate tokens (files might have changed)
    total, counts = _ENGINE.estimate_token_count_batch(suggestions["bundle_files"])
    suggestions["token_usage"] = {"total_tokens": total, "file_counts": counts}
    print(f"[ContextBundle] estimated context size: {total} tokens", file=sys.stderr)

    return suggestions


def print_context_bundle(task_description: str, task_type: str | None = None) -> None:
    files = get_context_bundle(task_description, task_type)
    for f in files:
        mode_str = f" [{f['mode']}]" if f.get('mode') != 'full' else ""
        print(f"{f['path']}{mode_str}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Context Engine 2.0 - High Performance Context Bundler")
    parser.add_argument("--task", type=str, help="Task description")
    parser.add_argument("--type", type=str, help="Explicit task type")
    parser.add_argument("--list", action="store_true", help="List available bundles")
    parser.add_argument("--auto", action="store_true", help="Auto-suggest mode")
    parser.add_argument("--budget", type=int, default=32000, help="Token budget (default: 32000)")

    args = parser.parse_args()

    # Configure global engine
    _ENGINE.max_tokens = args.budget

    if args.list:
        bundles = list_available_bundles()
        if bundles:
            print("Available bundles:")
            for b in bundles: print(f"  - {b}")
        else:
            print("No bundles found.")

    elif args.auto and args.task:
        suggestions = auto_suggest_context(args.task)
        print(f"Task Type: {suggestions['task_type']}")
        print(f"Context Bundle: {suggestions['context_bundle']}")
        print(f"\nBundle Files ({len(suggestions['bundle_files'])}):")
        for f in suggestions["bundle_files"]:
            mode_tag = f" [{f['mode']}]" if f.get('mode') != 'full' else ""
            print(f"  - {f['path']}{mode_tag}")

        token_stats = suggestions.get("token_usage", {})
        print(f"\nEstimated Token Usage: {token_stats.get('total_tokens', 0)}")

        if suggestions.get("suggested_workflows"):
            print("\nSuggested Workflows:")
            for wf in suggestions["suggested_workflows"]: print(f"  - {wf}")

    elif args.task:
        print_context_bundle(args.task, args.type)

    elif args.type:
        print_context_bundle(f"task type: {args.type}", args.type)

    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
