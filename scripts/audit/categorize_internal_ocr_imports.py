#!/usr/bin/env python3
"""
Categorize internal OCR imports into actionable fix groups.
Separates core ocr/ package fixes from scripts/ directory candidates.
"""

import json
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent
ANALYSIS_FILE = PROJECT_ROOT / "scripts/audit/broken_imports_analysis.json"


def load_analysis():
    """Load the broken imports analysis."""
    with open(ANALYSIS_FILE) as f:
        return json.load(f)


def categorize_internal_imports(internal_imports):
    """Categorize internal OCR imports by location and fix strategy."""

    core_fixes = defaultdict(list)  # Core ocr/ package - fix immediately
    script_candidates = defaultdict(list)  # scripts/ - defer for review

    for item in internal_imports:
        file_path = item["file"]

        # Categorize by location
        if file_path.startswith("scripts/"):
            # Categorize scripts by subdirectory
            parts = file_path.split("/")
            if len(parts) > 1:
                script_type = parts[1]  # audit, demos, performance, etc.
                script_candidates[script_type].append(item)
            else:
                script_candidates["root"].append(item)

        elif file_path.startswith("ocr/"):
            # Categorize core ocr/ package by import type
            module = item.get("module", "")

            if "ocr.core.validation" in module:
                core_fixes["validation_module"].append(item)
            elif "ocr.core.interfaces" in module:
                core_fixes["base_interfaces"].append(item)
            elif "ocr.core.lightning" in module:
                core_fixes["lightning_utils"].append(item)
            elif "ocr.core.utils.registry" in module:
                core_fixes["registry"].append(item)
            elif "ocr.core.models" in module:
                core_fixes["model_components"].append(item)
            elif "ocr.domains.detection.metrics" in module or "ocr.domains.detection.evaluation" in module:
                core_fixes["metrics_evaluation"].append(item)
            else:
                core_fixes["other_core"].append(item)

    return dict(core_fixes), dict(script_candidates)


def print_categorization(core_fixes, script_candidates):
    """Print categorization summary."""

    print("=" * 80)
    print("INTERNAL OCR IMPORT CATEGORIZATION")
    print("=" * 80)
    print()

    # Core package fixes
    total_core = sum(len(items) for items in core_fixes.values())
    print(f"📦 CORE OCR/ PACKAGE IMPORTS (FIX IMMEDIATELY): {total_core}")
    print("-" * 80)
    for category, items in sorted(core_fixes.items(), key=lambda x: -len(x[1])):
        print(f"  {category:25s}: {len(items):3d} imports")
        # Show sample files
        files = list(set(item["file"] for item in items))
        for f in files[:3]:
            print(f"    - {f}")
        if len(files) > 3:
            print(f"    ... and {len(files)-3} more files")

    print()

    # Scripts candidates
    total_scripts = sum(len(items) for items in script_candidates.values())
    print(f"📝 SCRIPTS/ DIRECTORY IMPORTS (DEFER FOR REVIEW): {total_scripts}")
    print("-" * 80)
    for script_type, items in sorted(script_candidates.items(), key=lambda x: -len(x[1])):
        print(f"  scripts/{script_type:20s}: {len(items):3d} imports")
        files = list(set(item["file"] for item in items))
        for f in files[:2]:
            print(f"    - {f}")

    print()
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print(f"✅ Fix {total_core} core ocr/ package imports immediately")
    print(f"⏸️  Defer {total_scripts} scripts/ imports for manual review and categorization")
    print()


def generate_fix_plan(core_fixes):
    """Generate fix plan for core imports."""

    print("FIX PLAN (Priority Order)")
    print("-" * 80)

    priority_order = [
        ("validation_module", "ocr.core.validation -> Check if module exists or was moved"),
        ("base_interfaces", "ocr.core.interfaces -> Verify base classes are in correct location"),
        ("lightning_utils", "ocr.core.lightning.utils -> Fix callback and utility imports"),
        ("registry", "ocr.core.utils.registry -> Verify registry module location"),
        ("model_components", "ocr.core.models -> Fix encoder/decoder/head factory functions"),
        ("metrics_evaluation", "ocr.domains.*.metrics -> Fix metric and evaluator imports"),
        ("other_core", "Other core imports -> Review individually"),
    ]

    for i, (category, description) in enumerate(priority_order, 1):
        count = len(core_fixes.get(category, []))
        if count > 0:
            print(f"{i}. {description}")
            print(f"   Count: {count} imports")

    print()


def main():
    """Main entrypoint."""
    analysis = load_analysis()
    internal_imports = analysis["details"].get("internal_ocr", [])

    print(f"Total internal OCR imports: {len(internal_imports)}\n")

    core_fixes, script_candidates = categorize_internal_imports(internal_imports)
    print_categorization(core_fixes, script_candidates)
    generate_fix_plan(core_fixes)

    # Export categorization
    output = {
        "core_fixes": core_fixes,
        "script_candidates": script_candidates,
        "summary": {
            "total_internal": len(internal_imports),
            "core_to_fix": sum(len(items) for items in core_fixes.values()),
            "scripts_deferred": sum(len(items) for items in script_candidates.values()),
        }
    }

    output_file = PROJECT_ROOT / "scripts/audit/internal_import_categorization.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"📄 Categorization exported to: {output_file}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
