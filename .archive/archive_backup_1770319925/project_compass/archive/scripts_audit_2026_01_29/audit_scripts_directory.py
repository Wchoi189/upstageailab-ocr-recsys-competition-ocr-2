#!/usr/bin/env python3
"""
Script audit and categorization tool.

Analyzes scripts/ directory to categorize files for:
- Archive (obsolete, unused)
- Remove (broken beyond repair, truly legacy)
- Review (needs manual inspection before decision)
- Keep (critical, actively used)
- Refactor (needs updates but valuable)
"""

import ast
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def analyze_script_complexity(file_path: Path) -> Dict:
    """Quick complexity analysis of a script."""
    try:
        with open(file_path) as f:
            content = f.read()
            tree = ast.parse(content)

        # Count key elements
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]

        # Check for main execution
        has_main = any(
            isinstance(node, ast.If) and
            isinstance(node.test, ast.Compare) and
            any("__name__" in ast.dump(n) for n in ast.walk(node.test))
            for node in tree.body
        )

        return {
            "functions": len(functions),
            "classes": len(classes),
            "imports": len(imports),
            "has_main": has_main,
            "lines": len(content.splitlines()),
        }
    except Exception as e:
        return {"error": str(e)}


def categorize_script(file_path: Path, stats: Dict) -> str:
    """Categorize script based on location and complexity."""
    parts = file_path.relative_to(SCRIPTS_DIR).parts

    if not parts:
        return "review"

    subdir = parts[0]
    filename = file_path.name

    # Categorization logic
    if subdir == "audit":
        return "keep"  # Audit tools are critical

    if subdir in ["archive", "deprecated", "old"]:
        return "archive"  # Already marked as archive

    if subdir == "troubleshooting":
        # Troubleshooting scripts - review individually
        if "test_" in filename:
            return "review"  # Test scripts may be outdated
        return "keep"

    if subdir in ["demos", "examples"]:
        # Demos - keep if simple, review if complex
        if stats.get("lines", 0) < 100:
            return "keep"
        return "review"

    if subdir == "performance":
        return "refactor"  # Performance scripts likely need updates

    if subdir in ["data", "etl"]:
        # Data processing - keep if has main, otherwise review
        if stats.get("has_main"):
            return "keep"
        return "review"

    if subdir in ["validation", "checkpoints"]:
        return "refactor"  # Validation scripts valuable but may need refactor

    if subdir == "utils":
        return "keep"  # Utilities are generally valuable

    return "review"  # Default to manual review


def audit_scripts_directory():
    """Full audit of scripts/ directory."""

    categorization = defaultdict(list)

    for script_file in SCRIPTS_DIR.rglob("*.py"):
        if script_file.name.startswith("__"):
            continue  # Skip __init__.py, __pycache__

        rel_path = script_file.relative_to(PROJECT_ROOT)
        stats = analyze_script_complexity(script_file)
        category = categorize_script(script_file, stats)

        categorization[category].append({
            "file": str(rel_path),
            "complexity": stats,
            "size_bytes": script_file.stat().st_size,
        })

    return dict(categorization)


def print_categorization(categorization: Dict):
    """Print human-readable categorization."""

    print("=" * 80)
    print("SCRIPTS DIRECTORY AUDIT")
    print("=" * 80)
    print()

    total = sum(len(files) for files in categorization.values())
    print(f"Total scripts analyzed: {total}\n")

    category_descriptions = {
        "keep": "✅ KEEP - Critical, actively used",
        "refactor": "🔧 REFACTOR - Valuable but needs updates",
        "review": "👀 REVIEW - Needs manual inspection",
        "archive": "📦 ARCHIVE - Move to archive/ directory",
        "remove": "🗑️  REMOVE - Delete entirely",
    }

    for category in ["keep", "refactor", "review", "archive", "remove"]:
        files = categorization.get(category, [])
        if not files:
            continue

        desc = category_descriptions.get(category, category.upper())
        print(f"\n{desc} ({len(files)} files)")
        print("-" * 80)

        # Group by subdirectory
        by_subdir = defaultdict(list)
        for item in files:
            parts = Path(item["file"]).parts
            if len(parts) > 1:
                subdir = parts[1]  # scripts/subdir/file.py
                by_subdir[subdir].append(item)
            else:
                by_subdir["root"].append(item)

        for subdir, items in sorted(by_subdir.items()):
            print(f"\n  📁 scripts/{subdir}/ ({len(items)} files)")
            for item in sorted(items, key=lambda x: x["file"])[:5]:
                filename = Path(item["file"]).name
                lines = item["complexity"].get("lines", "?")
                print(f"     - {filename:50s} ({lines:4} lines)")
            if len(items) > 5:
                print(f"     ... and {len(items)-5} more files")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    keep_count = len(categorization.get("keep", []))
    refactor_count = len(categorization.get("refactor", []))
    review_count = len(categorization.get("review", []))
    archive_count = len(categorization.get("archive", []))

    print(f"✅ Keep {keep_count} critical scripts")
    print(f"🔧 Refactor {refactor_count} valuable scripts")
    print(f"👀 Manually review {review_count} scripts")
    print(f"📦 Archive {archive_count} obsolete scripts")
    print()


def main():
    """Main entrypoint."""
    print(f"Auditing: {SCRIPTS_DIR}\n")

    categorization = audit_scripts_directory()
    print_categorization(categorization)

    # Export results
    output_file = PROJECT_ROOT / "scripts/audit/scripts_categorization.json"
    with open(output_file, "w") as f:
        json.dump({
            "audit_date": datetime.now().isoformat(),
            "total_scripts": sum(len(v) for v in categorization.values()),
            "categorization": categorization,
        }, f, indent=2)

    print(f"📄 Full categorization exported to: {output_file}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
