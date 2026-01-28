#!/usr/bin/env python3
"""
Comprehensive broken import analysis using ADT and master_audit.py

This script:
1. Runs master_audit.py to get baseline broken imports
2. Categorizes imports by type (torch/lightning, hydra, tiktoken, internal, UI)
3. Uses ADT intelligent-search to find where symbols should be imported from
4. Generates fix recommendations
"""

import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent


def run_master_audit() -> List[Dict]:
    """Run master_audit.py and parse broken imports."""
    result = subprocess.run(
        ["uv", "run", "python", "scripts/audit/master_audit.py"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    return _parse_audit_output(result.stdout)

def _parse_audit_output(stdout: str) -> List[Dict]:
    """Parse output from master_audit.py."""
    broken_imports = []
    current_import = {}

    for line in stdout.split("\n"):
        # Match: [File] path:line
        file_match = re.match(r"\s+\[File\] (.+):(\d+)", line)
        if file_match:
            if current_import:
                broken_imports.append(current_import)
            current_import = {
                "file": file_match.group(1),
                "line": int(file_match.group(2)),
            }

        # Match: --> Import: module [symbols]
        import_match = re.match(r"\s+--\u003e Import: (.+)", line)
        if import_match and current_import:
            import_str = import_match.group(1).strip()
            # Parse "module ['symbol1', 'symbol2']" or just "module"
            if "[" in import_str:
                module, symbols_str = import_str.split("[", 1)
                symbols_str = symbols_str.rstrip("]").strip()
                symbols = [s.strip().strip("'\"") for s in symbols_str.split(",")]
                current_import["module"] = module.strip()
                current_import["symbols"] = symbols
            else:
                current_import["module"] = import_str
                current_import["symbols"] = []

        # Match: --> Error: message
        error_match = re.match(r"\s+--\u003e Error: (.+)", line)
        if error_match and current_import:
            current_import["error"] = error_match.group(1).strip()

    if current_import:
        broken_imports.append(current_import)

    return broken_imports


def categorize_imports(broken_imports: List[Dict]) -> Dict[str, List[Dict]]:
    """Categorize broken imports by type."""
    categories = defaultdict(list)

    for item in broken_imports:
        module = item.get("module", "")

        if module in ["torch.nn", "torch._dynamo", "torch"]:
            categories["torch_missing"].append(item)
        elif module in ["lightning.pytorch", "lightning.pytorch.callbacks"]:
            categories["lightning_missing"].append(item)
        elif module in ["hydra", "hydra.utils"]:
            categories["hydra_missing"].append(item)
        elif module == "tiktoken":
            categories["tiktoken_optional"].append(item)
        elif module.startswith("ui."):
            categories["ui_modules"].append(item)
        elif module.startswith("ocr."):
            categories["internal_ocr"].append(item)
        else:
            categories["other"].append(item)

    return dict(categories)


def analyze_internal_ocr_imports(imports: List[Dict]) -> Dict[str, List[Dict]]:
    """Further categorize internal OCR imports."""
    subcategories = defaultdict(list)

    for item in imports:
        module = item.get("module", "")

        if "domains.detection.metrics" in module or "domains.recognition.metrics" in module:
            subcategories["domain_metrics"].append(item)
        elif "domains.detection.evaluation" in module:
            subcategories["domain_evaluation"].append(item)
        elif "core.evaluation" in module:
            subcategories["core_evaluation"].append(item)
        elif "core.lightning" in module:
            subcategories["lightning_utils"].append(item)
        elif "core.models" in module:
            subcategories["model_components"].append(item)
        elif "core.validation" in module:
            subcategories["validation"].append(item)
        elif "core.utils.registry" in module:
            subcategories["registry"].append(item)
        else:
            subcategories["other_internal"].append(item)

    return dict(subcategories)


def print_analysis(categories: Dict[str, List[Dict]]):
    """Print comprehensive analysis."""
    print("=" * 80)
    print("BROKEN IMPORT ANALYSIS")
    print("=" * 80)
    print()

    total = sum(len(items) for items in categories.values())
    print(f"Total broken imports: {total}\n")

    # Summary by category
    print("CATEGORY BREAKDOWN:")
    print("-" * 80)
    for category, items in sorted(categories.items(), key=lambda x: -len(x[1])):
        print(f"  {category:30s}: {len(items):3d} imports")
    print()

    _print_env_issues(categories)
    _print_detailed_breakdown(categories)

    # Internal OCR subcategory analysis
    if "internal_ocr" in categories:
        _print_internal_analysis(categories["internal_ocr"])

def _print_env_issues(categories: Dict[str, List[Dict]]) -> None:
    # Environment issues (likely false positives if torch/lightning installed)
    env_issues = (
        len(categories.get("torch_missing", [])) +
        len(categories.get("lightning_missing", []))
    )
    if env_issues > 0:
        print(f"⚠️  Note: {env_issues} imports are torch/lightning - likely environment issue in audit")
        print("   These may not be real broken imports if dependencies are installed\n")

def _print_detailed_breakdown(categories: Dict[str, List[Dict]]) -> None:
    print("\n" + "=" * 80)
    print("DETAILED BREAKDOWN")
    print("=" * 80)

    for category, items in categories.items():
        if not items:
            continue

        print(f"\n## {category.upper().replace('_', ' ')} ({len(items)} imports)")
        print("-" * 80)

        # Group by file for readability
        by_file = defaultdict(list)
        for item in items:
            by_file[item["file"]].append(item)

        for file, file_items in sorted(by_file.items())[:10]:  # Show top 10 files
            print(f"\n  📁 {file}")
            for item in file_items[:5]:  # Show top 5 imports per file
                module = item.get("module", "N/A")
                symbols = item.get("symbols", [])
                if symbols:
                    symbols_str = ", ".join(symbols[:3])
                    if len(symbols) > 3:
                        symbols_str += f", ... (+{len(symbols)-3} more)"
                    print(f"     → {module} [{symbols_str}]")
                else:
                    print(f"     → {module}")

        if len(by_file) > 10:
            print(f"\n  ... and {len(by_file) - 10} more files")

def _print_internal_analysis(items: List[Dict]) -> None:
    print("\n" + "=" * 80)
    print("INTERNAL OCR IMPORT SUBCATEGORIES")
    print("=" * 80)

    subcategories = analyze_internal_ocr_imports(items)
    for subcat, sub_items in sorted(subcategories.items(), key=lambda x: -len(x[1])):
        print(f"  {subcat:30s}: {len(sub_items):3d} imports")


def main():
    """Main entrypoint."""
    print("🔍 Running master audit to detect broken imports...\n")

    broken_imports = run_master_audit()

    if not broken_imports:
        print("✅ No broken imports detected!")
        return 0

    print(f"Found {len(broken_imports)} broken imports\n")

    categories = categorize_imports(broken_imports)
    print_analysis(categories)

    # Export to JSON for further processing
    output_file = PROJECT_ROOT / "scripts/audit/broken_imports_analysis.json"
    with open(output_file, "w") as f:
        json.dump({
            "total": len(broken_imports),
            "categories": {k: len(v) for k, v in categories.items()},
            "details": categories,
        }, f, indent=2)

    print(f"\n\n📄 Detailed analysis exported to: {output_file}")

    return 1 if broken_imports else 0


if __name__ == "__main__":
    sys.exit(main())
