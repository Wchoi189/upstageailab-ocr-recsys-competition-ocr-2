#!/usr/bin/env python3
"""
ADS v2.0 Header Migration Tool
ML-aided tool to migrate v1.0 standards to v2.0 format with automatic header generation.

Features:
- Automatic ID generation from filename/path
- Tier inference from directory structure
- Keyword extraction from content
- Dependency suggestion based on Tier 1 references
- Preview mode (default) and --apply mode
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

# Project root and paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STANDARDS_DIR = PROJECT_ROOT / "AgentQMS" / "standards"
ARCHIVE_DIR = STANDARDS_DIR / "_archive"

# ID prefix mapping by tier
TIER_PREFIXES = {
    1: "SC",  # Standards/Constitution
    2: "FW",  # Framework
    3: "AG",  # Agents
    4: "WF",  # Workflows
}

# Common stopwords to filter from keywords
STOPWORDS = {
    "the", "is", "at", "which", "on", "a", "an", "and", "or", "but", "in", "with",
    "to", "for", "of", "as", "by", "from", "that", "this", "these", "those",
    "be", "are", "was", "were", "been", "being", "have", "has", "had", "do",
    "does", "did", "will", "would", "should", "could", "may", "might", "must",
}


class MigrationError(Exception):
    """Raised when migration fails."""
    pass


def infer_tier_from_path(file_path: Path) -> int:
    """
    Infer tier from file path.

    Args:
        file_path: Path to YAML file

    Returns:
        Tier number (1-4)
    """
    path_str = str(file_path).lower()

    if "tier1" in path_str or "sst" in path_str:
        return 1
    elif "tier2" in path_str or "framework" in path_str:
        return 2
    elif "tier3" in path_str or "agent" in path_str:
        return 3
    elif "tier4" in path_str or "workflow" in path_str:
        return 4
    else:
        # Default to Tier 2 (Framework) for ambiguous files
        return 2


def generate_id(file_path: Path, tier: int, existing_ids: Set[str]) -> str:
    """
    Generate unique ID for standard.

    Args:
        file_path: Path to YAML file
        tier: Tier number (1-4)
        existing_ids: Set of existing IDs to avoid collisions

    Returns:
        Unique ID (e.g., SC-001, FW-042)
    """
    prefix = TIER_PREFIXES[tier]

    # Try to extract number from filename if present
    filename = file_path.stem
    match = re.search(r'(\d+)', filename)

    if match:
        number = int(match.group(1))
        candidate = f"{prefix}-{number:03d}"
        if candidate not in existing_ids:
            return candidate

    # Generate sequential ID
    for num in range(1, 1000):
        candidate = f"{prefix}-{num:03d}"
        if candidate not in existing_ids:
            return candidate

    raise MigrationError(f"Could not generate unique ID for {file_path}")


def extract_keywords(data: Dict[str, Any], max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from YAML content using heuristics.

    Args:
        data: Parsed YAML data
        max_keywords: Maximum number of keywords to extract

    Returns:
        List of keywords
    """
    keywords = []

    # Extract from existing keywords if present
    if "keywords" in data:
        keywords.extend(data["keywords"])

    # Extract from description
    if "description" in data:
        desc = str(data["description"]).lower()
        words = re.findall(r'\b[a-z]{3,}\b', desc)
        keywords.extend([w for w in words if w not in STOPWORDS])

    # Extract from keys (field names often indicate content)
    for key in data.keys():
        if key not in ["ads_version", "type", "agent", "tier", "priority"]:
            # Split camelCase and snake_case
            parts = re.findall(r'[a-z]+', key.lower())
            keywords.extend([p for p in parts if p not in STOPWORDS and len(p) > 2])

    # Count frequency and take most common
    keyword_counts = Counter(keywords)
    return [kw for kw, _ in keyword_counts.most_common(max_keywords)]


def suggest_dependencies(
    data: Dict[str, Any],
    tier: int,
    existing_standards: Dict[str, Dict[str, Any]]
) -> List[str]:
    """
    Suggest dependencies based on content analysis.

    Args:
        data: Parsed YAML data
        tier: Tier number
        existing_standards: Dictionary of existing standards

    Returns:
        List of suggested dependency IDs
    """
    if tier == 1:
        # Tier 1 standards have no dependencies (they ARE the dependencies)
        return []

    dependencies = []

    # Auto-include naming conventions for most standards
    for std_id, std_data in existing_standards.items():
        if std_data.get("tier") == 1:
            keywords = std_data.get("keywords", [])
            if "naming" in keywords or "conventions" in keywords:
                dependencies.append(std_id)
                break

    # TODO: More sophisticated dependency inference could be added here
    # - NLP-based similarity matching
    # - Keyword overlap analysis
    # - Reference detection in content

    return dependencies


def migrate_standard(
    file_path: Path,
    existing_ids: Set[str],
    existing_standards: Dict[str, Dict[str, Any]],
    force: bool = False
) -> Dict[str, Any]:
    """
    Migrate a v1.0 standard to v2.0 format.

    Args:
        file_path: Path to v1.0 YAML file
        existing_ids: Set of existing IDs
        existing_standards: Dictionary of existing standards
        force: Force migration even if already v2.0

    Returns:
        Migrated header dictionary
    """
    # Load existing file
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise MigrationError(f"Invalid YAML structure in {file_path}")

    # Check if already v2.0
    if data.get("ads_version") == "2.0" and not force:
        raise MigrationError(f"File already has ADS v2.0 header: {file_path}")

    # Infer tier
    tier = data.get("tier", infer_tier_from_path(file_path))

    # Generate ID
    std_id = generate_id(file_path, tier, existing_ids)

    # Extract keywords
    keywords = extract_keywords(data)

    # Suggest dependencies
    dependencies = data.get("dependencies", suggest_dependencies(data, tier, existing_standards))

    # Build v2.0 header
    header = {
        "ads_version": "2.0",
        "id": std_id,
        "type": data.get("type", "rule_set"),
        "agent": data.get("agent", "all"),
        "tier": tier,
        "priority": data.get("priority", "medium"),
        "validates_with": data.get("validates_with", "AgentQMS/standards/schemas/compliance-checker.py"),
        "compliance_status": data.get("compliance_status", "unknown"),
        "memory_footprint": data.get("memory_footprint", 100),
        "dependencies": dependencies if isinstance(dependencies, list) else [],
        "keywords": keywords,
        "fuzzy_threshold": 0.8,
    }

    # Add optional fields if present
    if "description" in data:
        header["description"] = data["description"]

    if "triggers" in data:
        header["triggers"] = data["triggers"]

    return header


def apply_migration(
    file_path: Path,
    header: Dict[str, Any],
    output_dir: Path
) -> Path:
    """
    Apply migration by writing v2.0 file.

    Args:
        file_path: Original file path
        header: Migrated v2.0 header
        output_dir: Output directory for migrated file

    Returns:
        Path to migrated file
    """
    # Load original content
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Merge header into data
    for key, value in header.items():
        data[key] = value

    # Determine output path (use tier directory)
    tier = header["tier"]
    tier_dir = output_dir / f"tier{tier}-{'sst' if tier == 1 else ['', '', 'framework', 'agents', 'workflows'][tier]}"
    tier_dir.mkdir(parents=True, exist_ok=True)

    output_path = tier_dir / file_path.name

    # Write migrated file
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True
        )

    return output_path


def display_header(header: Dict[str, Any], file_path: Path) -> None:
    """Display suggested header in a formatted way."""
    print(f"\n{'='*60}")
    print(f"File: {file_path.name}")
    print(f"{'='*60}")
    print(yaml.dump(header, default_flow_style=False, sort_keys=False))
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="ADS v2.0 Header Migration Tool - ML-aided migration from v1.0 to v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s path/to/standard.yaml                    # Preview migration
  %(prog)s path/to/standard.yaml --apply            # Apply migration
  %(prog)s _archive/naming-conventions.yaml --apply # Migrate from archive
        """
    )

    parser.add_argument("file", type=Path, help="Path to v1.0 standard file")
    parser.add_argument("--apply", action="store_true",
                       help="Apply migration and write v2.0 file")
    parser.add_argument("--output-dir", type=Path, default=STANDARDS_DIR,
                       help="Output directory for migrated files")
    parser.add_argument("--force", action="store_true",
                       help="Force migration even if already v2.0")

    args = parser.parse_args()

    if not args.file.exists():
        print(f"❌ File not found: {args.file}", file=sys.stderr)
        return 1

    try:
        # Load existing standards to avoid ID collisions
        existing_ids: Set[str] = set()
        existing_standards: Dict[str, Dict[str, Any]] = {}

        # TODO: Load from registry if available
        # For now, scan tier directories
        for tier_dir in [STANDARDS_DIR / f"tier{i}-{name}"
                        for i, name in [(1, "sst"), (2, "framework"),
                                       (3, "agents"), (4, "workflows")]]:
            if tier_dir.exists():
                for yaml_file in tier_dir.rglob("*.yaml"):
                    try:
                        with open(yaml_file, "r", encoding="utf-8") as f:
                            data = yaml.safe_load(f)
                            if isinstance(data, dict) and "id" in data:
                                std_id = data["id"]
                                existing_ids.add(std_id)
                                existing_standards[std_id] = data
                    except Exception:
                        pass

        # Migrate standard
        header = migrate_standard(args.file, existing_ids, existing_standards, args.force)

        # Display or apply
        if args.apply:
            output_path = apply_migration(args.file, header, args.output_dir)
            print(f"✅ Migrated: {output_path}")
            print(f"   ID: {header['id']}")
            print(f"   Tier: {header['tier']}")
            print(f"   Keywords: {', '.join(header['keywords'][:5])}")
        else:
            print("\n🔍 PREVIEW MODE (use --apply to write file)")
            display_header(header, args.file)
            print("💡 Run with --apply to write the migrated file")

        return 0

    except MigrationError as e:
        print(f"❌ Migration failed: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"💥 Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
