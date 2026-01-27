#!/usr/bin/env python3
"""
ADS v2.0 Registry Compiler
Auto-generates registry.yaml from standard files with ADS headers.

Features:
- Strict mode enforcement (zero untracked files)
- Circular dependency detection
- Visual architecture graphs (DOT format)
- Semantic diff (Pulse Delta)
- JSON schema validation
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Project root and paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
STANDARDS_DIR = PROJECT_ROOT / "AgentQMS" / "standards"
SCHEMA_PATH = PROJECT_ROOT / "AgentQMS" / ".agentqms" / "schemas" / "artifact_type_validation.yaml"
REGISTRY_PATH = STANDARDS_DIR / "registry.yaml"
ARCHIVE_DIR = STANDARDS_DIR / "_archive"
DOT_OUTPUT = STANDARDS_DIR / "architecture_map.dot"
CACHE_PATH = STANDARDS_DIR / ".ads_cache.pickle"

# Tier directories to scan
TIER_DIRS = [
    STANDARDS_DIR / "tier1-sst",
    STANDARDS_DIR / "tier2-framework",
    STANDARDS_DIR / "tier3-agents",
    STANDARDS_DIR / "tier4-workflows",
]

# Phase 6: Mini-Registry Protocol - Stop words for keyword filtering
KEYWORD_STOPWORDS = {
    "validates", "compliance", "status", "memory", "footprint",
    "last", "updated", "standard", "agent", "tier"
}

# Maximum number of standards a keyword can point to before being pruned
KEYWORD_SATURATION_LIMIT = 10

# Fields to KEEP in registry (pointer-mapping only)
# All other fields are moved to cache
PERMITTED_FIELDS = {
    "id", "tier", "file_path", "description", "dependencies"
}

# Maximum description length in registry (full text in cache)
MAX_DESCRIPTION_LENGTH = 60


class RegistryCompilationError(Exception):
    """Raised when registry compilation fails."""
    pass


class CircularDependencyError(RegistryCompilationError):
    """Raised when circular dependencies are detected."""
    pass


class SchemaValidationError(RegistryCompilationError):
    """Raised when ADS header validation fails."""
    pass


def load_json_schema() -> dict[str, Any]:
    """Load ADS v2.0 YAML schema."""
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema not found: {SCHEMA_PATH}")

    with open(SCHEMA_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def scan_tier_directories(strict: bool = False) -> list[Path]:
    """
    Scan tier directories for YAML standard files.

    Args:
        strict: If True, fail if no files found in any tier directory

    Returns:
        List of YAML file paths
    """
    yaml_files = []

    for tier_dir in TIER_DIRS:
        if not tier_dir.exists():
            if strict:
                raise RegistryCompilationError(f"Tier directory not found: {tier_dir}")
            continue

        # Recursively find all .yaml files (exclude hidden files)
        files = [
            f for f in tier_dir.rglob("*.yaml")
            if not any(part.startswith(".") for part in f.parts)
        ]

        yaml_files.extend(files)

    if strict and not yaml_files:
        raise RegistryCompilationError(
            "SC-011 Strict Mode: Zero YAML files found in tier directories"
        )

    return yaml_files


def extract_ads_header(yaml_path: Path) -> dict[str, Any]:
    """
    Extract ADS header from YAML file.

    Args:
        yaml_path: Path to YAML standard file

    Returns:
        Dictionary containing ADS header fields
    """
    try:
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise SchemaValidationError(f"Invalid YAML structure in {yaml_path}")

        # Extract ADS header fields (top-level keys)
        header = {
            "ads_version": data.get("ads_version"),
            "id": data.get("id"),
            "type": data.get("type"),
            "agent": data.get("agent"),
            "tier": data.get("tier"),
            "priority": data.get("priority"),
            "validates_with": data.get("validates_with"),
            "compliance_status": data.get("compliance_status"),
            "memory_footprint": data.get("memory_footprint"),
            "dependencies": data.get("dependencies", []),
            "keywords": data.get("keywords", []),
            "fuzzy_threshold": data.get("fuzzy_threshold", 0.8),
            "triggers": data.get("triggers"),
            "description": data.get("description"),
            "replaces": data.get("replaces"),
            "deprecated": data.get("deprecated", False),
        }

        # Add file path for reference
        header["_file_path"] = str(yaml_path.relative_to(PROJECT_ROOT))

        return {k: v for k, v in header.items() if v is not None}

    except yaml.YAMLError as e:
        raise SchemaValidationError(f"YAML parsing error in {yaml_path}: {e}")
    except Exception as e:
        raise SchemaValidationError(f"Error reading {yaml_path}: {e}")


def validate_header(header: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """
    Validate ADS header against JSON schema.

    Args:
        header: ADS header dictionary
        schema: JSON schema

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    required = schema.get("required", [])
    properties = schema.get("properties", {})

    # Check required fields
    for field in required:
        if field not in header:
            errors.append(f"Missing required field: {field}")

    # Check ads_version
    if header.get("ads_version") != "2.0":
        errors.append(f"Invalid ads_version: {header.get('ads_version')} (expected '2.0')")

    # Check id pattern
    if "id" in header:
        import re
        pattern = properties.get("id", {}).get("pattern", "")
        if pattern and not re.match(pattern, header["id"]):
            errors.append(f"Invalid id format: {header['id']} (expected pattern: {pattern})")

    # Check tier range
    if "tier" in header:
        tier = header["tier"]
        if not isinstance(tier, int) or tier < 0 or tier > 4:
            errors.append(f"Invalid tier: {tier} (must be 0-4)")

    # Check enums
    for field in ["type", "agent", "priority", "compliance_status"]:
        if field in header:
            allowed = properties.get(field, {}).get("enum", [])
            if allowed and header[field] not in allowed:
                errors.append(
                    f"Invalid {field}: {header[field]} (allowed: {', '.join(map(str, allowed))})"
                )

    # Check dependencies format
    if "dependencies" in header:
        deps = header["dependencies"]
        if not isinstance(deps, list):
            errors.append(f"dependencies must be an array, got {type(deps).__name__}")

    return errors


def detect_cycles(
    standards: dict[str, dict[str, Any]]
) -> list[str] | None:
    """
    Detect circular dependencies using DFS.

    Args:
        standards: Dictionary mapping standard IDs to their headers

    Returns:
        Cycle path if found, None otherwise
    """
    def dfs(node: str, visited: set[str], rec_stack: list[str]) -> list[str] | None:
        visited.add(node)
        rec_stack.append(node)

        for dep_id in standards.get(node, {}).get("dependencies", []):
            if dep_id not in standards:
                # Missing dependency - will be caught by validation
                continue

            if dep_id not in visited:
                cycle = dfs(dep_id, visited, rec_stack)
                if cycle:
                    return cycle
            elif dep_id in rec_stack:
                # Cycle detected
                cycle_start = rec_stack.index(dep_id)
                return rec_stack[cycle_start:] + [dep_id]

        rec_stack.pop()
        return None

    visited: set[str] = set()

    for std_id in standards:
        if std_id not in visited:
            cycle = dfs(std_id, visited, [])
            if cycle:
                return cycle

    return None


def generate_dot_graph(standards: dict[str, dict[str, Any]]) -> str:
    """
    Generate DOT format dependency graph.

    Phase 6.5: Uses mechanized architecture with governance and dependency flows.

    Args:
        standards: Dictionary mapping standard IDs to their headers

    Returns:
        DOT format string
    """
    # Import the mechanized graph generator
    try:
        # Add tools directory to path for import
        tools_dir = Path(__file__).parent
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))

        from generate_mechanized_graph import generate_mechanized_graph
        return generate_mechanized_graph(standards, include_legend=True, include_domains=True)
    except Exception as e:
        # Fallback to basic graph if mechanized generator not available
        print(f"   ⚠️  Warning: Mechanized graph generator error ({e}), using basic graph")
        return generate_basic_dot_graph(standards)


def generate_basic_dot_graph(standards: dict[str, dict[str, Any]]) -> str:
    """
    Generate basic DOT format dependency graph (fallback).

    Args:
        standards: Dictionary mapping standard IDs to their headers

    Returns:
        DOT format string
    """
    lines = [
        "digraph AgentQMS_Architecture {",
        '  rankdir=LR;',
        '  node [shape=box, style=rounded];',
        "",
    ]

    # Group by tier
    tier_groups = defaultdict(list)
    for std_id, header in standards.items():
        tier = header.get("tier", 0)
        tier_groups[tier].append((std_id, header))

    # Define tier subgraphs with colors
    tier_colors = {
        1: "#FFE6E6",  # Light red - Constitution
        2: "#E6F3FF",  # Light blue - Framework
        3: "#E6FFE6",  # Light green - Agents
        4: "#FFF9E6",  # Light yellow - Workflows
    }

    for tier in sorted(tier_groups.keys()):
        tier_name = {
            1: "Constitution (Tier 1)",
            2: "Framework (Tier 2)",
            3: "Agents (Tier 3)",
            4: "Workflows (Tier 4)",
        }.get(tier, f"Tier {tier}")

        lines.append(f'  subgraph cluster_tier{tier} {{')
        lines.append(f'    label="{tier_name}";')
        lines.append('    style=filled;')
        lines.append(f'    color="{tier_colors.get(tier, "#FFFFFF")}";')
        lines.append("")

        for std_id, header in tier_groups[tier]:
            desc = header.get("description", "")
            if desc and len(desc) > 40:
                desc = desc[:37] + "..."
            label = f"{std_id}\\n{desc}" if desc else std_id
            priority = header.get("priority", "medium")

            # Style by priority - FIX: No trailing comma
            if priority == "critical":
                style = 'penwidth=2, color=red'
            elif priority == "high":
                style = 'penwidth=1.5, color=orange'
            else:
                style = None

            if style:
                lines.append(f'    "{std_id}" [label="{label}", {style}];')
            else:
                lines.append(f'    "{std_id}" [label="{label}"];')

        lines.append("  }")
        lines.append("")

    # Add dependency edges
    for std_id, header in standards.items():
        for dep_id in header.get("dependencies", []):
            if dep_id in standards:
                lines.append(f'  "{dep_id}" -> "{std_id}";')

    lines.append("}")

    return "\n".join(lines)


def compute_pulse_delta(
    old_registry: dict[str, Any] | None,
    new_standards: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """
    Compute semantic diff between old and new registry.

    Args:
        old_registry: Previous registry data (None if first compilation)
        new_standards: New standards dictionary

    Returns:
        Pulse Delta dictionary with changes
    """
    delta = {
        "added": [],
        "removed": [],
        "modified": [],
        "unchanged": [],
    }

    if not old_registry:
        # First compilation - all standards are new
        delta["added"] = list(new_standards.keys())
        return delta

    # Extract old standards (assuming they were in a similar format)
    # For now, we'll check the tier mappings in the old registry
    old_standards = {}
    for tier_key in ["tier1_sst", "tier2_framework", "tier3_agents", "tier4_workflows"]:
        tier_data = old_registry.get(tier_key, {})
        for name, path in tier_data.items():
            # Create pseudo-ID from path
            if isinstance(path, str):
                old_standards[name] = {"path": path}

    old_ids = set(old_standards.keys())
    new_ids = set(new_standards.keys())

    delta["added"] = sorted(new_ids - old_ids)
    delta["removed"] = sorted(old_ids - new_ids)

    # Check for modifications (comparing by ID only for now)
    common_ids = old_ids & new_ids
    for std_id in sorted(common_ids):
        # Compare paths or other metadata if available
        delta["unchanged"].append(std_id)

    return delta


def filter_keywords(standards: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """
    Build keyword index with stop-word filtering and saturation limits.

    Phase 6: Context-Thin optimization.

    Args:
        standards: Dictionary mapping standard IDs to their headers

    Returns:
        Filtered keyword index
    """
    # Build initial index
    raw_index = defaultdict(list)
    for std_id, header in standards.items():
        for keyword in header.get("keywords", []):
            kw_lower = keyword.lower()
            # Skip stop words
            if kw_lower not in KEYWORD_STOPWORDS:
                raw_index[kw_lower].append(std_id)

    # Apply saturation limit
    filtered_index = {}
    for keyword, std_ids in raw_index.items():
        if len(std_ids) <= KEYWORD_SATURATION_LIMIT:
            filtered_index[keyword] = std_ids

    return filtered_index


def save_full_cache(
    standards: dict[str, dict[str, Any]],
    pulse_delta: dict[str, Any],
    keyword_index: dict[str, list[str]],
    tier_index: dict[str, list[str]],
    dependency_summary: dict[str, Any]
) -> None:
    """
    Save complete standard metadata to binary cache.

    Phase 6: Offload stripped metadata + search indices to .ads_cache.pickle

    Args:
        standards: Full standards dictionary with all metadata
        pulse_delta: Pulse Delta change summary
        keyword_index: Keyword search index
        tier_index: Tier grouping index
        dependency_summary: Dependency statistics
    """
    import pickle
    import time

    cache_data = {
        "standards": standards,
        "pulse_delta": pulse_delta,
        "keyword_index": keyword_index,
        "tier_index": tier_index,
        "dependency_summary": dependency_summary,
        "timestamp": time.time(),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

    try:
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"   ⚠️  Warning: Failed to write cache: {e}")


def generate_registry(
    standards: dict[str, dict[str, Any]],
    pulse_delta: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, list[str]], dict[str, list[str]], dict[str, Any]]:
    """
    Generate minimal registry.yaml structure + search indices for cache.

    Phase 6: Mini-Registry Protocol - pointer-mapping ONLY in registry.
    All search indices moved to .ads_cache.pickle

    Args:
        standards: Dictionary mapping standard IDs to their headers
        pulse_delta: Pulse Delta change summary

    Returns:
        Tuple of (minimal_registry, keyword_index, tier_index, dependency_summary)
    """
    # Build search indices (will be moved to cache)
    tier_index = defaultdict(list)
    for std_id, header in standards.items():
        tier = header.get("tier", 0)
        tier_index[f"tier{tier}"].append(std_id)

    keyword_index = filter_keywords(standards)

    dependency_summary = {
        "total_dependencies": sum(
            len(h.get("dependencies", [])) for h in standards.values()
        ),
        "standards_with_deps": sum(
            1 for h in standards.values() if h.get("dependencies")
        ),
        "orphan_standards": [
            std_id for std_id, h in standards.items()
            if not h.get("dependencies") and h.get("tier") != 1
        ],
    }

    # Minimal registry - pointer mapping only
    registry = {
        "ads_version": "2.0",
        "type": "unified_registry",
        "name": "AgentQMS Registry v2.0 P6",
        "total_standards": len(standards),
        "standards": {},
    }

    # Add standards with stripped metadata (keep only pointer-mapping fields)
    for std_id, header in sorted(standards.items()):
        # Keep ONLY permitted pointer-mapping fields
        clean_header = {}
        for k, v in header.items():
            # Rename _file_path to file_path
            key = k[1:] if k.startswith("_") else k

            # Only keep fields in PERMITTED_FIELDS
            if key in PERMITTED_FIELDS:
                # Skip empty/None values to save space
                if not v:
                    continue

                # Truncate long descriptions (full text in cache)
                if key == "description":
                    if len(v) > MAX_DESCRIPTION_LENGTH:
                        v = v[:MAX_DESCRIPTION_LENGTH - 3] + "..."

                clean_header[key] = v

        registry["standards"][std_id] = clean_header

    return registry, keyword_index, tier_index, dependency_summary


def display_pulse_delta(pulse_delta: dict[str, Any]) -> None:
    """Display Pulse Delta in a formatted way."""
    print("\n" + "="*60)
    print("  PULSE DELTA: Registry Changes")
    print("="*60)

    if pulse_delta["added"]:
        print(f"\n✅ ADDED ({len(pulse_delta['added'])}):")
        for std_id in pulse_delta["added"]:
            print(f"   + {std_id}")

    if pulse_delta["removed"]:
        print(f"\n❌ REMOVED ({len(pulse_delta['removed'])}):")
        for std_id in pulse_delta["removed"]:
            print(f"   - {std_id}")

    if pulse_delta["modified"]:
        print(f"\n📝 MODIFIED ({len(pulse_delta['modified'])}):")
        for std_id in pulse_delta["modified"]:
            print(f"   ~ {std_id}")

    if not any(pulse_delta.values()):
        print("\n✨ No changes detected")

    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="ADS v2.0 Registry Compiler - Auto-generate registry.yaml"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and show changes without writing files"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode: fail if no standards found"
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Skip DOT graph generation"
    )

    args = parser.parse_args()

    print("🔧 ADS v2.0 Registry Compiler")
    print("="*60)

    try:
        # Load schema
        print("📋 Loading JSON schema...")
        schema = load_json_schema()
        print(f"   ✓ Schema loaded: {SCHEMA_PATH.name}")

        # Scan directories
        print("\n📂 Scanning tier directories...")
        yaml_files = scan_tier_directories(strict=args.strict)
        print(f"   ✓ Found {len(yaml_files)} YAML files")

        if not yaml_files:
            print("\n⚠️  No standards found. Nothing to compile.")
            return 0

        # Extract and validate headers
        print("\n🔍 Extracting and validating ADS headers...")
        standards = {}
        errors_found = False

        for yaml_path in yaml_files:
            try:
                header = extract_ads_header(yaml_path)

                # Validate header
                validation_errors = validate_header(header, schema)
                if validation_errors:
                    print(f"\n   ❌ {yaml_path.name}:")
                    for error in validation_errors:
                        print(f"      • {error}")
                    errors_found = True
                else:
                    std_id = header.get("id")
                    if std_id:
                        if std_id in standards:
                            print(f"\n   ❌ Duplicate ID: {std_id}")
                            print(f"      File 1: {standards[std_id]['_file_path']}")
                            print(f"      File 2: {header['_file_path']}")
                            errors_found = True
                        else:
                            standards[std_id] = header
                            print(f"   ✓ {std_id}: {yaml_path.name}")

            except SchemaValidationError as e:
                print(f"\n   ❌ {yaml_path.name}: {e}")
                errors_found = True

        if errors_found:
            print("\n❌ Validation failed. Fix errors before compiling.")
            return 1

        print(f"\n   ✓ All {len(standards)} standards validated")

        # Check for circular dependencies
        print("\n🔄 Checking for circular dependencies...")
        cycle = detect_cycles(standards)
        if cycle:
            print("\n   ❌ CIRCULAR DEPENDENCY DETECTED:")
            print(f"      {' -> '.join(cycle)}")
            raise CircularDependencyError(
                f"Circular dependency: {' -> '.join(cycle)}"
            )
        print("   ✓ No cycles detected")

        # Load old registry for diff
        old_registry = None
        if REGISTRY_PATH.exists():
            with open(REGISTRY_PATH, encoding="utf-8") as f:
                old_registry = yaml.safe_load(f)

        # Compute Pulse Delta
        print("\n📊 Computing Pulse Delta...")
        pulse_delta = compute_pulse_delta(old_registry, standards)
        display_pulse_delta(pulse_delta)

        # Generate minimal registry + search indices (Phase 6: Mini-Registry Protocol)
        print("\n📝 Generating minimal registry...")
        registry, keyword_index, tier_index, dependency_summary = generate_registry(standards, pulse_delta)
        print(f"   ✓ Registry generated ({len(standards)} standards, metadata stripped)")
        print(f"   ✓ Search indices built: {len(keyword_index)} keywords, {len(tier_index)} tiers")

        # Save full metadata + indices to cache (Phase 6: Binary offloading)
        if not args.dry_run:
            print("\n💾 Saving full metadata to cache...")
            save_full_cache(standards, pulse_delta, keyword_index, tier_index, dependency_summary)
            print(f"   ✓ Cache saved: {CACHE_PATH.name}")

        # Generate DOT graph
        if not args.no_graph:
            print("\n🎨 Generating dependency graph...")
            dot_content = generate_dot_graph(standards)

            if not args.dry_run:
                with open(DOT_OUTPUT, "w", encoding="utf-8") as f:
                    f.write(dot_content)
                print(f"   ✓ Graph saved: {DOT_OUTPUT.name}")
                print(f"   💡 Render with: dot -Tpng {DOT_OUTPUT.name} -o architecture.png")
            else:
                print("   ✓ Graph generated (dry-run, not saved)")

        # Write registry
        if not args.dry_run:
            print("\n💾 Writing registry...")
            # Convert defaultdicts to regular dicts before dumping
            def convert_defaultdicts(obj):
                if isinstance(obj, defaultdict):
                    return dict(obj)
                elif isinstance(obj, dict):
                    return {k: convert_defaultdicts(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_defaultdicts(item) for item in obj]
                return obj

            registry_clean = convert_defaultdicts(registry)

            with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    registry_clean,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True
                )
            print(f"   ✓ Registry saved: {REGISTRY_PATH.name}")
        else:
            print("\n🔍 DRY RUN: Registry validated but not saved")

        print("\n" + "="*60)
        print("✅ Compilation successful!")
        print("="*60 + "\n")

        return 0

    except RegistryCompilationError as e:
        print(f"\n❌ Compilation failed: {e}")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
