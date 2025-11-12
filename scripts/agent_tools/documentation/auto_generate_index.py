#!/usr/bin/env python3
"""Auto-generate and update the AI agents documentation index.json from directory structure."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def extract_title_from_file(file_path: Path) -> str:
    """Extract title from markdown file (frontmatter or first heading)."""
    try:
        with file_path.open("r", encoding="utf-8") as f:
            content = f.read()

            # Try to extract from frontmatter
            frontmatter_match = re.search(r"^---\s*\ntitle:\s*[\"'](.+?)[\"']", content, re.MULTILINE)
            if frontmatter_match:
                return frontmatter_match.group(1)

            # Fallback to first heading
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("# "):
                    title = line[2:].strip()
                    # Remove markdown formatting
                    title = re.sub(r"\*\*([^*]+)\*\*", r"\1", title)
                    return title
    except Exception:
        pass
    # Fallback to filename
    return file_path.stem.replace("_", " ").replace("-", " ").title()


def extract_metadata_from_file(file_path: Path) -> dict[str, Any]:
    """Extract metadata from markdown file frontmatter."""
    metadata = {
        "title": None,
        "date": None,
        "type": None,
        "category": None,
        "status": None,
        "version": None,
        "tags": [],
    }

    try:
        with file_path.open("r", encoding="utf-8") as f:
            content = f.read()

            # Extract frontmatter
            frontmatter_match = re.search(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
            if frontmatter_match:
                frontmatter = frontmatter_match.group(1)

                # Extract title
                title_match = re.search(r"title:\s*[\"'](.+?)[\"']", frontmatter)
                if title_match:
                    metadata["title"] = title_match.group(1)

                # Extract date
                date_match = re.search(r"date:\s*[\"'](.+?)[\"']", frontmatter)
                if date_match:
                    metadata["date"] = date_match.group(1)

                # Extract type
                type_match = re.search(r"type:\s*[\"'](.+?)[\"']", frontmatter)
                if type_match:
                    metadata["type"] = type_match.group(1)

                # Extract category
                category_match = re.search(r"category:\s*[\"'](.+?)[\"']", frontmatter)
                if category_match:
                    metadata["category"] = category_match.group(1)

                # Extract status
                status_match = re.search(r"status:\s*[\"'](.+?)[\"']", frontmatter)
                if status_match:
                    metadata["status"] = status_match.group(1)

                # Extract version
                version_match = re.search(r"version:\s*[\"'](.+?)[\"']", frontmatter)
                if version_match:
                    metadata["version"] = version_match.group(1)

                # Extract tags
                tags_match = re.search(r"tags:\s*\[(.*?)\]", frontmatter, re.DOTALL)
                if tags_match:
                    tags_str = tags_match.group(1)
                    tags = [t.strip().strip('"\'') for t in tags_str.split(",") if t.strip()]
                    metadata["tags"] = tags
    except Exception:
        pass

    return metadata


def scan_agents_directory(agents_dir: Path) -> dict[str, Any]:
    """Scan docs/agents/ directory and build index structure."""
    entries = []
    bundles: dict[str, list[str]] = defaultdict(list)

    if not agents_dir.exists():
        return {
            "source": "docs/agents/",
            "version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "entries": [],
            "bundles": {},
        }

    # Scan for markdown files
    for md_file in agents_dir.rglob("*.md"):
        if md_file.name.startswith(".") or md_file.name == "README.md":
            continue

        # Calculate relative path
        rel_path = md_file.relative_to(agents_dir)
        path_str = f"docs/agents/{rel_path}"

        # Extract metadata
        file_metadata = extract_metadata_from_file(md_file)
        title = file_metadata.get("title") or extract_title_from_file(md_file)

        # Determine category and priority
        category = file_metadata.get("category") or "general"
        priority = "medium"

        # Determine category from path
        if "protocols" in str(rel_path):
            category = "protocols"
            priority = "high"
        elif "references" in str(rel_path):
            category = "references"
            priority = "medium"
        elif "tracking" in str(rel_path):
            category = "tracking"
            priority = "medium"
        elif "automation" in str(rel_path):
            category = "automation"
            priority = "medium"
        elif "coding_protocols" in str(rel_path):
            category = "coding"
            priority = "high"
        elif md_file.name == "system.md":
            category = "core"
            priority = "critical"
        elif md_file.name == "index.md":
            category = "core"
            priority = "high"

        # Create entry ID from filename
        entry_id = md_file.stem.lower().replace(" ", "_").replace("-", "_")

        # Determine tags
        tags = file_metadata.get("tags", [])
        if not tags:
            # Auto-detect tags from filename
            filename_lower = md_file.stem.lower()
            if "debug" in filename_lower:
                tags.append("debugging")
            if "train" in filename_lower or "experiment" in filename_lower:
                tags.append("training")
            if "ui" in filename_lower or "streamlit" in filename_lower:
                tags.append("ui")
            if "config" in filename_lower or "hydra" in filename_lower:
                tags.append("configuration")
            if not tags:
                tags.append(category)

        entry = {
            "id": entry_id,
            "title": title,
            "path": path_str,
            "category": category,
            "tags": tags,
            "priority": priority,
            "status": file_metadata.get("status") or "active",
            "version": file_metadata.get("version") or "1.0",
            "date": file_metadata.get("date") or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        }

        entries.append(entry)

        # Add to bundles based on tags
        for tag in tags:
            if tag not in bundles:
                bundles[tag] = []
            bundles[tag].append(entry_id)

    # Convert bundles to the expected format
    formatted_bundles = {}
    for bundle_name, entry_ids in bundles.items():
        formatted_bundles[bundle_name] = {"entries": entry_ids}

    return {
        "source": "docs/agents/",
        "version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
        "bundles": formatted_bundles,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-generate agents documentation index.json from directory structure"
    )
    parser.add_argument(
        "--agents-dir",
        type=Path,
        default=Path("docs/agents"),
        help="Path to the docs/agents directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/agents/index.json"),
        help="Output path for the generated index.json",
    )
    parser.add_argument(
        "--validate", action="store_true", help="Run validation after generation"
    )

    args = parser.parse_args()

    # Generate the index
    print(f"Scanning {args.agents_dir}...")
    index_data = scan_agents_directory(args.agents_dir)

    # Write to file
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Generated index with {len(index_data['entries'])} entries")
    print(f"   Output: {args.output}")

    if args.validate:
        print("🔍 Validating index...")
        # Basic validation
        if not index_data.get("entries"):
            print("⚠️  Warning: No entries found")
        else:
            print("✅ Index validation passed")


if __name__ == "__main__":
    main()
