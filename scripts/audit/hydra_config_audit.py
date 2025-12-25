#!/usr/bin/env python3
"""
Hydra Configuration System Audit Script

This script helps audit the Hydra configuration system to identify:
- Legacy vs new architecture configurations
- Override pattern requirements
- Configuration references
- Removal impact assessment
"""

import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
BASE_DIR = CONFIGS_DIR / "_base"


def find_config_files() -> dict[str, Path]:
    """Find all YAML config files."""
    configs = {}
    for yaml_file in CONFIGS_DIR.rglob("*.yaml"):
        rel_path = yaml_file.relative_to(CONFIGS_DIR)
        configs[str(rel_path)] = yaml_file
    return configs


def analyze_config_architecture(config_path: Path) -> dict[str, Any]:
    """Analyze a config file to determine its architecture."""
    analysis = {
        "path": str(config_path.relative_to(CONFIGS_DIR)),
        "architecture": "unknown",
        "uses_base": False,
        "has_defaults": False,
        "dependencies": [],
        "package_directive": None,
    }

    try:
        content = config_path.read_text()

        # Check for _base references
        if "_base" in content or "/_base/" in content:
            analysis["uses_base"] = True
            analysis["architecture"] = "new"

        # Check for defaults section
        if "defaults:" in content:
            analysis["has_defaults"] = True
            # Extract dependencies from defaults
            defaults_match = re.search(r"defaults:\s*\n((?:\s*-\s*[^\n]+\n?)+)", content)
            if defaults_match:
                defaults_lines = defaults_match.group(1)
                deps = re.findall(r"-\s*([^\s:]+)", defaults_lines)
                analysis["dependencies"] = [d for d in deps if d != "_self"]

        # Check for @package directive
        package_match = re.search(r"#\s*@package\s+([^\n]+)", content)
        if package_match:
            analysis["package_directive"] = package_match.group(1).strip()

        # Classify architecture
        if analysis["uses_base"]:
            analysis["architecture"] = "new"
        elif config_path.parent.name == "_base":
            analysis["architecture"] = "new"
        elif not analysis["has_defaults"] and not analysis["package_directive"]:
            analysis["architecture"] = "legacy"
        else:
            analysis["architecture"] = "hybrid"

    except Exception as e:
        analysis["error"] = str(e)

    return analysis


def find_references(config_name: str) -> dict[str, list[str]]:
    """Find all references to a config in the codebase."""
    references = {
        "code": [],
        "scripts": [],
        "docs": [],
        "ui": [],
    }

    # Search patterns

    search_dirs = {
        "code": ["ocr", "runners"],
        "scripts": ["scripts"],
        "docs": ["docs"],
        "ui": ["ui", "apps"],
    }

    for search_type, dirs in search_dirs.items():
        for search_dir in dirs:
            search_path = PROJECT_ROOT / search_dir
            if not search_path.exists():
                continue

            for file_path in search_path.rglob("*.py"):
                try:
                    content = file_path.read_text()
                    if config_name in content or f"configs/{config_name}" in content:
                        references[search_type].append(str(file_path.relative_to(PROJECT_ROOT)))
                except Exception:
                    pass

    return references


def check_override_pattern(config_path: Path, config_name: str) -> dict[str, Any]:
    """Check if config requires + prefix for overrides."""
    override_info = {
        "in_defaults": False,
        "requires_plus": None,
        "notes": [],
    }

    # Check if it's in base.yaml defaults
    base_yaml = CONFIGS_DIR / "base.yaml"
    if base_yaml.exists():
        content = base_yaml.read_text()
        if f"- {config_name.split('/')[0]}:" in content or f"- {config_name}:" in content:
            override_info["in_defaults"] = True
            override_info["requires_plus"] = False
            override_info["notes"].append("In base.yaml defaults - use without +")
        else:
            override_info["requires_plus"] = True
            override_info["notes"].append("Not in base.yaml defaults - use with +")

    return override_info


def generate_audit_report() -> dict[str, Any]:
    """Generate comprehensive audit report."""
    configs = find_config_files()
    report = {
        "summary": {
            "total_configs": len(configs),
            "new_architecture": 0,
            "legacy_architecture": 0,
            "hybrid": 0,
            "unknown": 0,
        },
        "configs": {},
    }

    for config_name, config_path in configs.items():
        analysis = analyze_config_architecture(config_path)
        references = find_references(config_name)
        override_info = check_override_pattern(config_path, config_name)

        report["configs"][config_name] = {
            **analysis,
            "references": references,
            "override_pattern": override_info,
            "has_references": any(references.values()),
        }

        # Update summary
        arch = analysis["architecture"]
        if arch in report["summary"]:
            report["summary"][arch] += 1

    return report


def main():
    """Main audit function."""
    print("🔍 Starting Hydra Configuration Audit...")
    print(f"📁 Configs directory: {CONFIGS_DIR}")
    print()

    report = generate_audit_report()

    # Print summary
    print("=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    print(f"Total configs: {report['summary']['total_configs']}")
    print(f"New architecture: {report['summary']['new_architecture']}")
    print(f"Legacy architecture: {report['summary']['legacy_architecture']}")
    print(f"Hybrid: {report['summary']['hybrid']}")
    print(f"Unknown: {report['summary']['unknown']}")
    print()

    # Print legacy configs
    print("=" * 60)
    print("LEGACY CONFIGS")
    print("=" * 60)
    legacy_configs = [
        (name, data) for name, data in report["configs"].items()
        if data["architecture"] == "legacy"
    ]

    for name, data in sorted(legacy_configs):
        ref_count = sum(len(refs) for refs in data["references"].values())
        status = "⚠️ HAS REFERENCES" if ref_count > 0 else "✅ NO REFERENCES"
        print(f"{status} {name}")
        if ref_count > 0:
            for ref_type, refs in data["references"].items():
                if refs:
                    print(f"  {ref_type}: {len(refs)} references")

    print()

    # Save full report
    report_path = PROJECT_ROOT / "hydra_config_audit_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"📄 Full report saved to: {report_path}")
    print()
    print("✅ Audit complete! Review the report for detailed findings.")


if __name__ == "__main__":
    main()
