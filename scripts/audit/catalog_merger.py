#!/usr/bin/env python3
"""Audit and compare all data catalog files.

This script:
1. Reads all catalog files
2. Identifies duplicates, conflicts, and missing information
3. Generates comparison report
"""

import json
import sys
from pathlib import Path
from typing import Any

import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def read_yaml_file(filepath: Path) -> dict[str, Any] | None:
    """Read a YAML file and return its contents."""
    if not filepath.exists():
        return None

    try:
        with open(filepath, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        return {"error": str(e)}


def audit_catalog_file(filepath: Path) -> dict[str, Any]:
    """Audit a single catalog file."""
    result = {
        "path": str(filepath),
        "exists": filepath.exists(),
        "error": None,
        "content": None,
        "structure": {},
    }

    if not filepath.exists():
        result["error"] = "File does not exist"
        return result

    content = read_yaml_file(filepath)
    if content is None:
        result["error"] = "Could not read file"
        return result

    if isinstance(content, dict) and "error" in content:
        result["error"] = content["error"]
        return result

    result["content"] = content

    # Analyze structure
    if isinstance(content, dict):
        result["structure"] = {
            "top_level_keys": list(content.keys()),
            "has_datasets": "datasets" in content,
            "has_exports": "exports" in content,
            "has_version": "version" in content,
        }

        # Extract dataset information if present
        if "datasets" in content:
            if isinstance(content["datasets"], dict):
                result["structure"]["dataset_keys"] = list(content["datasets"].keys())
            elif isinstance(content["datasets"], list):
                result["structure"]["dataset_count"] = len(content["datasets"])
                result["structure"]["dataset_names"] = [
                    d.get("name", "unnamed") for d in content["datasets"] if isinstance(d, dict)
                ]

    return result


def compare_catalogs(catalog_audits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Compare multiple catalog files."""
    comparison = {
        "total_catalogs": len(catalog_audits),
        "accessible_catalogs": sum(1 for a in catalog_audits.values() if a["exists"] and not a.get("error")),
        "duplicates": [],
        "conflicts": [],
        "missing_information": [],
        "unique_entries": {},
    }

    # Find duplicates (identical content)
    catalog_contents = {}
    for name, audit in catalog_audits.items():
        if audit["exists"] and not audit.get("error") and audit.get("content"):
            content_str = json.dumps(audit["content"], sort_keys=True)
            if content_str in catalog_contents:
                comparison["duplicates"].append({
                    "file1": catalog_contents[content_str],
                    "file2": name,
                    "path1": catalog_audits[catalog_contents[content_str]]["path"],
                    "path2": audit["path"],
                })
            else:
                catalog_contents[content_str] = name

    # Find conflicts (same dataset, different information)
    dataset_info = {}
    for name, audit in catalog_audits.items():
        if audit["exists"] and not audit.get("error") and audit.get("content"):
            content = audit["content"]
            if "datasets" in content:
                datasets = content["datasets"]
                if isinstance(datasets, dict):
                    for ds_name, ds_info in datasets.items():
                        if ds_name not in dataset_info:
                            dataset_info[ds_name] = []
                        dataset_info[ds_name].append({
                            "catalog": name,
                            "path": audit["path"],
                            "info": ds_info,
                        })
                elif isinstance(datasets, list):
                    for ds in datasets:
                        if isinstance(ds, dict) and "name" in ds:
                            ds_name = ds["name"]
                            if ds_name not in dataset_info:
                                dataset_info[ds_name] = []
                            dataset_info[ds_name].append({
                                "catalog": name,
                                "path": audit["path"],
                                "info": ds,
                            })

    # Check for conflicts
    for ds_name, entries in dataset_info.items():
        if len(entries) > 1:
            # Compare entries
            first_info = entries[0]["info"]
            for entry in entries[1:]:
                if json.dumps(first_info, sort_keys=True) != json.dumps(entry["info"], sort_keys=True):
                    comparison["conflicts"].append({
                        "dataset": ds_name,
                        "catalog1": entries[0]["catalog"],
                        "catalog2": entry["catalog"],
                        "path1": entries[0]["path"],
                        "path2": entry["path"],
                    })

    return comparison


def main():
    """Main audit function."""
    output_dir = project_root / "data" / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define catalog files to audit
    catalog_files = {
        "data_catalog": project_root / "data" / "data_catalog.yaml",
        "export_data_catalog": project_root / "data" / "export" / "data_catalog.yaml",
        "metadata_dataset_catalog": project_root / "data" / "metadata" / "dataset_catalog.yml",
        "aws_batch_processor_catalog": project_root / "aws-batch-processor" / "docs" / "artifacts" / "data_catalog.yaml",
    }

    print("=" * 80)
    print("Data Catalog Audit")
    print("=" * 80)
    print()

    # Audit all files
    catalog_audits = {}
    for name, filepath in catalog_files.items():
        print(f"Auditing {name}...")
        audit = audit_catalog_file(filepath)
        catalog_audits[name] = audit

        if audit["exists"]:
            if audit.get("error"):
                print(f"  ✗ Error: {audit['error']}")
            else:
                print(f"  ✓ Found")
                if audit.get("structure"):
                    struct = audit["structure"]
                    if "top_level_keys" in struct:
                        print(f"    Top-level keys: {', '.join(struct['top_level_keys'])}")
                    if "dataset_count" in struct:
                        print(f"    Datasets: {struct['dataset_count']}")
                    elif "dataset_keys" in struct:
                        print(f"    Dataset keys: {', '.join(struct['dataset_keys'])}")
        else:
            print(f"  ✗ File not found: {filepath}")
        print()

    # Compare catalogs
    print("=" * 80)
    print("Catalog Comparison")
    print("=" * 80)
    print()

    comparison = compare_catalogs(catalog_audits)

    print(f"Total catalogs: {comparison['total_catalogs']}")
    print(f"Accessible catalogs: {comparison['accessible_catalogs']}")
    print(f"Duplicates found: {len(comparison['duplicates'])}")
    print(f"Conflicts found: {len(comparison['conflicts'])}")
    print()

    if comparison["duplicates"]:
        print("Duplicates:")
        for dup in comparison["duplicates"]:
            print(f"  - {dup['file1']} and {dup['file2']} are identical")
            print(f"    {dup['path1']}")
            print(f"    {dup['path2']}")
        print()

    if comparison["conflicts"]:
        print("Conflicts:")
        for conflict in comparison["conflicts"]:
            print(f"  - Dataset '{conflict['dataset']}' differs between:")
            print(f"    {conflict['catalog1']}: {conflict['path1']}")
            print(f"    {conflict['catalog2']}: {conflict['path2']}")
        print()

    # Generate report
    report = {
        "audit_timestamp": str(Path(__file__).stat().st_mtime),
        "catalog_audits": catalog_audits,
        "comparison": comparison,
        "recommendations": [],
    }

    # Generate recommendations
    if comparison["duplicates"]:
        report["recommendations"].append(
            "Consolidate duplicate catalog files into a single source of truth in data/metadata/"
        )
    if comparison["conflicts"]:
        report["recommendations"].append(
            "Resolve conflicts in dataset information across catalogs"
        )
    if comparison["accessible_catalogs"] < comparison["total_catalogs"]:
        report["recommendations"].append(
            "Some catalog files are missing or inaccessible - verify paths"
        )

    # Save report
    report_path = output_dir / "catalog_comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print("Recommendations:")
    for rec in report["recommendations"]:
        print(f"  - {rec}")
    print()
    print(f"Full report saved to: {report_path}")

    return report


if __name__ == "__main__":
    main()
