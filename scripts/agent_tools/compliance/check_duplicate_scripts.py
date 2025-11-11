#!/usr/bin/env python3
"""
Check for Duplicate Scripts

This script checks for duplicate or similar scripts before allowing new script creation.
It validates that agents have checked existing scripts before creating new ones.

Usage:
    python scripts/agent_tools/compliance/check_duplicate_scripts.py --file path/to/new_script.py
    python scripts/agent_tools/compliance/check_duplicate_scripts.py --all
"""

import argparse
import ast
import importlib.util
import re
import sys
from pathlib import Path


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
            if spec is None or spec.loader is None:
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


def extract_script_info(script_path: Path) -> dict:
    """Extract information from a Python script."""
    try:
        with open(script_path, encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return {}

    info = {
        "name": script_path.stem,
        "path": str(script_path),
        "docstring": "",
        "functions": [],
        "classes": [],
        "keywords": [],
    }

    # Extract docstring
    docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
    if docstring_match:
        info["docstring"] = docstring_match.group(1).strip()

    # Extract function names
    func_matches = re.findall(r"def\s+(\w+)\s*\(", content)
    info["functions"] = func_matches

    # Extract class names
    class_matches = re.findall(r"class\s+(\w+)\s*[(:]", content)
    info["classes"] = class_matches

    # Extract keywords from docstring and content
    keywords = []
    if info["docstring"]:
        keywords.extend(re.findall(r"\b\w{4,}\b", info["docstring"].lower()))
    keywords.extend([f.lower() for f in info["functions"]])
    keywords.extend([c.lower() for c in info["classes"]])
    info["keywords"] = list(set(keywords))

    return info


def find_similar_scripts(new_script_path: Path, search_dirs: list[Path]) -> list[dict]:
    """Find scripts with similar functionality."""
    new_info = extract_script_info(new_script_path)
    if not new_info:
        return []

    similar = []
    new_name = new_info["name"].lower()
    new_keywords = set(new_info["keywords"])

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        for script_path in search_dir.rglob("*.py"):
            if script_path == new_script_path or script_path.name == "__init__.py":
                continue

            existing_info = extract_script_info(script_path)
            if not existing_info:
                continue

            existing_name = existing_info["name"].lower()
            existing_keywords = set(existing_info["keywords"])

            # Check for similar names
            similarity_score = 0
            reasons = []

            # Name similarity
            if new_name in existing_name or existing_name in new_name:
                similarity_score += 50
                reasons.append(f"Similar name: {existing_info['name']}")

            # Keyword overlap
            common_keywords = new_keywords & existing_keywords
            if len(common_keywords) >= 3:
                similarity_score += 30
                reasons.append(f"Shared keywords: {', '.join(list(common_keywords)[:3])}")

            # Function name overlap
            common_functions = set(new_info["functions"]) & set(existing_info["functions"])
            if common_functions:
                similarity_score += 20
                reasons.append(f"Shared functions: {', '.join(list(common_functions)[:3])}")

            if similarity_score >= 30:
                similar.append(
                    {
                        "script": str(script_path),
                        "name": existing_info["name"],
                        "score": similarity_score,
                        "reasons": reasons,
                    }
                )

    return sorted(similar, key=lambda x: x["score"], reverse=True)


def check_duplicate_scripts(script_path: Path) -> dict:
    """Check if a script duplicates existing functionality."""
    result = {
        "file": str(script_path),
        "has_duplicates": False,
        "similar_scripts": [],
        "errors": [],
    }

    if not script_path.exists():
        result["errors"].append("Script file does not exist")
        return result

    # Search in agent_tools directory
    agent_tools_dir = Path("scripts/agent_tools")
    search_dirs = [agent_tools_dir]

    # Also check scripts root if script is in agent_tools
    if "agent_tools" not in str(script_path):
        search_dirs.append(Path("scripts"))

    similar = find_similar_scripts(script_path, search_dirs)

    if similar:
        result["has_duplicates"] = True
        result["similar_scripts"] = similar

    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check for duplicate scripts before creation"
    )
    parser.add_argument("--file", help="Check specific script file")
    parser.add_argument(
        "files", nargs="*", help="Files to check (for pre-commit hooks)"
    )

    args = parser.parse_args()

    if args.files:
        results = []
        for file_path_str in args.files:
            file_path = Path(file_path_str)
            if file_path.is_file() and file_path.suffix == ".py":
                results.append(check_duplicate_scripts(file_path))
    elif args.file:
        file_path = Path(args.file)
        results = [check_duplicate_scripts(file_path)]
    else:
        print("❌ Please specify --file or provide files as arguments")
        return 1

    has_duplicates = False
    for result in results:
        if result["has_duplicates"]:
            has_duplicates = True
            print(f"\n⚠️  DUPLICATE FUNCTIONALITY DETECTED: {result['file']}")
            print("=" * 70)
            print("Similar existing scripts found:")
            for similar in result["similar_scripts"][:5]:  # Show top 5
                print(f"\n  📁 {similar['script']}")
                print(f"     Name: {similar['name']}")
                print(f"     Similarity: {similar['score']}%")
                for reason in similar["reasons"]:
                    print(f"     • {reason}")
            print("\n❌ ACTION REQUIRED:")
            print("   1. Review existing scripts above")
            print("   2. Update existing script instead of creating new one")
            print("   3. Or use: python scripts/agent_tools/core/discover.py to find existing tools")
            print("   4. If truly new functionality, document why it's different")

    if has_duplicates:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

