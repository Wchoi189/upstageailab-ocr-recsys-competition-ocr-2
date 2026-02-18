#!/usr/bin/env python3
"""
Generate Registry from Specs

Auto-generates AgentQMS/.agentqms/registry.yaml from AgentQMS/specs/ directory.
This restores the "single source of truth" registry architecture.

Usage:
    uv run python scripts/utils/generate_registry.py
"""

from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. Run: uv sync", file=sys.stderr)
    sys.exit(1)


def get_project_root() -> Path:
    """Find project root by looking for .git directory."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find project root (.git directory)")


def parse_frontmatter(content: str) -> dict[str, Any]:
    """Extract YAML frontmatter from markdown file."""
    # Match YAML frontmatter between --- delimiters
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if not match:
        return {}

    try:
        return yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return {}


def extract_description(content: str, frontmatter: dict[str, Any]) -> str:
    """Extract description from frontmatter or first paragraph."""
    # Priority 1: frontmatter description
    if "description" in frontmatter:
        return frontmatter["description"]

    # Priority 2: first paragraph after frontmatter
    # Remove frontmatter
    content_no_fm = re.sub(r"^---\s*\n.*?\n---\s*\n", "", content, flags=re.DOTALL)

    # Find first non-header paragraph
    lines = content_no_fm.strip().split("\n")
    for i, line in enumerate(lines):
        line = line.strip()
        if line and not line.startswith("#"):
            # Take first sentence or first 100 chars
            sentence_end = re.search(r"[.!?]\s", line)
            if sentence_end:
                return line[:sentence_end.end()].strip()
            return line[:100] + ("..." if len(line) > 100 else "")

    return ""


def scan_specs(specs_dir: Path) -> dict[str, dict[str, Any]]:
    """Scan all .spec.md files and extract metadata."""
    specs = {}

    for spec_file in specs_dir.rglob("*.spec.md"):
        try:
            content = spec_file.read_text(encoding="utf-8")
            frontmatter = parse_frontmatter(content)

            # Extract spec ID (required)
            spec_id = frontmatter.get("id")
            if not spec_id:
                # Try to infer from filename
                spec_id = spec_file.stem.replace(".spec", "").upper()

            # Extract tier from directory structure
            relative_path = spec_file.relative_to(specs_dir)
            tier_match = re.match(r"tier(\d+)-", str(relative_path))
            tier = int(tier_match.group(1)) if tier_match else None

            # Override with frontmatter tier if present
            if "tier" in frontmatter:
                try:
                    tier = int(frontmatter["tier"])
                except (ValueError, TypeError):
                    pass

            # Build spec entry
            spec_entry = {
                "id": spec_id,
                "file_path": str(spec_file.relative_to(get_project_root())),
            }

            # Add optional fields
            if tier:
                spec_entry["tier"] = tier

            description = extract_description(content, frontmatter)
            if description:
                spec_entry["description"] = description

            # Extract dependencies if present
            dependencies = frontmatter.get("dependencies", [])
            if dependencies:
                spec_entry["dependencies"] = dependencies

            # Add priority if present
            priority = frontmatter.get("priority")
            if priority:
                spec_entry["priority"] = priority

            specs[spec_id] = spec_entry

        except Exception as e:
            print(f"Warning: Failed to parse {spec_file}: {e}", file=sys.stderr)
            continue

    return specs


def generate_registry(specs_dir: Path, output_path: Path) -> None:
    """Generate registry.yaml from specs directory."""
    # Scan specs
    specs = scan_specs(specs_dir)

    # Check if specs content changed vs existing registry (ignore generated_at)
    if output_path.exists():
        try:
            existing = yaml.safe_load(output_path.read_text(encoding="utf-8")) or {}
            if existing.get("specs") == specs:
                print(f"✓ Registry up to date with {len(specs)} specs (no changes)")
                return
        except yaml.YAMLError:
            pass  # Corrupted file — regenerate

    # Build registry structure
    registry = {
        "ads_version": "2.0",
        "type": "unified_registry",
        "name": "AgentQMS Registry v2.0 P7",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_specs": len(specs),
        "specs": specs,
    }

    # Write registry
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.dump(
            registry,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=100,
        )

    print(f"✓ Generated registry with {len(specs)} specs")
    print(f"✓ Written to: {output_path}")


def main() -> int:
    """Main entry point."""
    try:
        project_root = get_project_root()
        specs_dir = project_root / "AgentQMS" / "specs"
        output_path = project_root / "AgentQMS" / ".agentqms" / "registry.yaml"

        if not specs_dir.exists():
            print(f"ERROR: Specs directory not found: {specs_dir}", file=sys.stderr)
            return 1

        generate_registry(specs_dir, output_path)
        return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
