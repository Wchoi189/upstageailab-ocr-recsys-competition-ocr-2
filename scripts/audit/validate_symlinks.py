#!/usr/bin/env python3
"""Audit all symbolic links in data/ directory and document structure.

This script:
1. Finds all symbolic links in data/
2. Checks if targets exist and are accessible
3. Identifies broken links
4. Documents the structure
5. Evaluates pros/cons of current approach
"""

import json
import sys
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def find_symlinks(directory: Path) -> list[dict[str, Any]]:
    """Find all symbolic links in a directory tree."""
    symlinks = []

    if not directory.exists():
        return symlinks

    for item in directory.rglob("*"):
        if item.is_symlink():
            try:
                target = item.readlink()
                target_path = item.parent / target if not target.is_absolute() else Path(target)
                target_resolved = target_path.resolve()

                symlinks.append({
                    "link_path": str(item.relative_to(project_root)),
                    "link_absolute": str(item),
                    "target": str(target),
                    "target_resolved": str(target_resolved),
                    "target_exists": target_resolved.exists(),
                    "target_is_directory": target_resolved.is_dir() if target_resolved.exists() else None,
                    "is_broken": not target_resolved.exists(),
                })
            except Exception as e:
                symlinks.append({
                    "link_path": str(item.relative_to(project_root)),
                    "link_absolute": str(item),
                    "error": str(e),
                    "is_broken": True,
                })

    return symlinks


def analyze_structure(symlinks: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze the symbolic link structure."""
    analysis = {
        "total_links": len(symlinks),
        "broken_links": sum(1 for s in symlinks if s.get("is_broken")),
        "valid_links": sum(1 for s in symlinks if not s.get("is_broken")),
        "link_patterns": {},
        "potential_issues": [],
    }

    # Group by target patterns
    for link in symlinks:
        if "target" in link:
            target = link["target"]
            # Extract pattern (e.g., ../raw/competition/...)
            if target.startswith("../"):
                pattern = "relative_up"
            elif target.startswith("/"):
                pattern = "absolute"
            else:
                pattern = "relative"

            if pattern not in analysis["link_patterns"]:
                analysis["link_patterns"][pattern] = []
            analysis["link_patterns"][pattern].append(link["link_path"])

    # Check for potential issues
    link_paths = {s["link_path"] for s in symlinks}
    for link in symlinks:
        if link.get("is_broken"):
            analysis["potential_issues"].append({
                "type": "broken_link",
                "link": link["link_path"],
                "target": link.get("target", "unknown"),
            })

        # Check for circular references (simplified check)
        if "target_resolved" in link and link["target_resolved"]:
            target_res = Path(link["target_resolved"])
            if target_res.is_symlink():
                # Could be circular, but would need deeper analysis
                pass

    return analysis


def main():
    """Main validation function."""
    output_dir = project_root / "data" / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = project_root / "data"

    print("=" * 80)
    print("Symbolic Link Validation")
    print("=" * 80)
    print()

    # Find all symlinks
    print(f"Scanning {data_dir} for symbolic links...")
    symlinks = find_symlinks(data_dir)

    print(f"Found {len(symlinks)} symbolic links")
    print()

    # Display results
    for link in symlinks:
        status = "✓" if not link.get("is_broken") else "✗"
        print(f"{status} {link['link_path']}")
        print(f"    → {link.get('target', 'unknown')}")
        if link.get("is_broken"):
            print(f"    ⚠ BROKEN")
        elif "target_resolved" in link:
            print(f"    → {link['target_resolved']}")
        print()

    # Analyze structure
    analysis = analyze_structure(symlinks)

    print("=" * 80)
    print("Structure Analysis")
    print("=" * 80)
    print(f"Total links: {analysis['total_links']}")
    print(f"Valid links: {analysis['valid_links']}")
    print(f"Broken links: {analysis['broken_links']}")
    print()

    if analysis["potential_issues"]:
        print("Potential Issues:")
        for issue in analysis["potential_issues"]:
            print(f"  - {issue['type']}: {issue['link']}")
        print()

    # Generate documentation
    structure_doc = f"""# Symbolic Link Structure Documentation

## Overview

This document describes the symbolic link structure in the `data/` directory.

## Summary

- **Total symbolic links**: {analysis['total_links']}
- **Valid links**: {analysis['valid_links']}
- **Broken links**: {analysis['broken_links']}

## Symbolic Links

"""

    for link in symlinks:
        structure_doc += f"### `{link['link_path']}`\n\n"
        structure_doc += f"- **Target**: `{link.get('target', 'unknown')}`\n"
        if "target_resolved" in link:
            structure_doc += f"- **Resolved**: `{link['target_resolved']}`\n"
        structure_doc += f"- **Status**: {'✓ Valid' if not link.get('is_broken') else '✗ Broken'}\n\n"

    structure_doc += """## Link Patterns

"""

    for pattern, links in analysis["link_patterns"].items():
        structure_doc += f"### {pattern}\n\n"
        for link_path in links:
            structure_doc += f"- `{link_path}`\n"
        structure_doc += "\n"

    # Save structure documentation
    structure_path = output_dir / "symlink_structure.md"
    with open(structure_path, "w") as f:
        f.write(structure_doc)

    # Generate evaluation
    evaluation_doc = f"""# Symbolic Link Structure Evaluation

## Current Approach

The `data/` directory uses symbolic links to logically map data to locations without duplicating files.

## Pros

1. **No Duplication**: Physical files exist in one location, reducing storage
2. **Flexible Organization**: Can reorganize logical structure without moving files
3. **Multiple Views**: Same data can appear in multiple logical locations

## Cons

1. **Complexity**: Symbolic links can be confusing and hard to track
2. **Broken Links**: Links can break if target directories are moved
3. **Tool Compatibility**: Some tools may not follow symlinks correctly
4. **Debugging Difficulty**: Harder to understand actual file locations

## Current Issues

"""

    if analysis["broken_links"] > 0:
        evaluation_doc += f"- **{analysis['broken_links']} broken link(s)** need to be fixed\n"

    evaluation_doc += """
## Recommendations

1. **Document all symlinks** in a central location (this document)
2. **Validate symlinks** in CI/CD to catch broken links early
3. **Consider alternatives**:
   - Real directory structure (simpler, but requires duplication or reorganization)
   - Configuration-based path mapping (more flexible, but requires code changes)
   - Hybrid approach (use symlinks for large datasets, real structure for small ones)

## Alternative Approaches

### Real Directory Structure
- **Pros**: Simple, no symlink issues, clear file locations
- **Cons**: Requires file duplication or reorganization, larger storage footprint

### Configuration-Based Path Mapping
- **Pros**: Very flexible, no symlink issues, easy to change
- **Cons**: Requires code changes, may need path resolution utilities

### Hybrid Approach
- **Pros**: Best of both worlds
- **Cons**: More complex to maintain, need clear guidelines on when to use each
"""

    # Save evaluation
    evaluation_path = output_dir / "symlink_evaluation.md"
    with open(evaluation_path, "w") as f:
        f.write(evaluation_doc)

    # Generate report
    report = {
        "audit_timestamp": str(Path(__file__).stat().st_mtime),
        "symlinks": symlinks,
        "analysis": analysis,
    }

    report_path = output_dir / "symlink_validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Structure documentation: {structure_path}")
    print(f"Evaluation document: {evaluation_path}")
    print(f"Full report: {report_path}")

    return report


if __name__ == "__main__":
    main()
