#!/usr/bin/env python3
"""Audit script to examine KIE/DP parquet files and identify corruption.

This script checks:
- Column schema (DP should have polygons, KIE should not)
- Sample rows to verify data structure
- Compare DP vs KIE for each split (train/val/test)
- Generate comprehensive audit report
"""

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def check_file_exists(filepath: Path) -> dict[str, Any]:
    """Check if file exists and return basic info."""
    result = {
        "exists": filepath.exists(),
        "path": str(filepath),
    }
    if filepath.exists():
        result["size_bytes"] = filepath.stat().st_size
    return result


def audit_parquet_file(filepath: Path, project_root: Path) -> dict[str, Any]:
    """Audit a single parquet file and return detailed information."""
    result = {
        "path": str(filepath),
        "exists": False,
        "error": None,
        "row_count": 0,
        "columns": [],
        "has_polygons": False,
        "polygon_sample": None,
        "sample_row": None,
        "null_counts": {},
        "data_types": {},
        "image_path_validation": {
            "valid_paths": 0,
            "invalid_paths": 0,
            "missing_images": 0,
            "sample_invalid": [],
        },
        "data_integrity": {
            "critical_nulls": {},
            "empty_lists": {},
            "length_mismatches": [],
        },
    }

    if not filepath.exists():
        result["error"] = "File does not exist"
        return result

    try:
        df = pd.read_parquet(filepath)
        result["exists"] = True
        result["row_count"] = len(df)
        result["columns"] = list(df.columns)
        result["data_types"] = {col: str(dtype) for col, dtype in df.dtypes.items()}

        # Check for polygons column
        if "polygons" in df.columns:
            result["has_polygons"] = True
            # Get a sample of non-null polygons
            non_null_polygons = df[df["polygons"].notna()]["polygons"]
            if len(non_null_polygons) > 0:
                sample_polygon = non_null_polygons.iloc[0]
                result["polygon_sample"] = (
                    str(sample_polygon)[:200] if isinstance(sample_polygon, list) else str(sample_polygon)[:200]
                )

        # Check for null values in critical columns
        critical_cols = ["id", "image_path", "texts", "labels"]
        for col in critical_cols:
            if col in df.columns:
                null_count = int(df[col].isna().sum())
                result["null_counts"][col] = null_count
                result["data_integrity"]["critical_nulls"][col] = null_count

        # Validate image paths
        if "image_path" in df.columns:
            valid_count = 0
            invalid_count = 0
            missing_count = 0
            sample_invalid = []

            for idx, img_path in enumerate(df["image_path"]):
                if pd.isna(img_path):
                    invalid_count += 1
                    if len(sample_invalid) < 5:
                        sample_invalid.append({"row": idx, "reason": "null_path"})
                    continue

                # Try to resolve path
                img_path_str = str(img_path)
                # Handle relative paths
                if not img_path_str.startswith("/") and not img_path_str.startswith("s3://"):
                    # Try relative to project root
                    resolved_path = project_root / img_path_str
                else:
                    resolved_path = Path(img_path_str)

                if resolved_path.exists():
                    valid_count += 1
                else:
                    missing_count += 1
                    if len(sample_invalid) < 5:
                        sample_invalid.append({"row": idx, "path": img_path_str, "reason": "file_not_found"})

            result["image_path_validation"]["valid_paths"] = valid_count
            result["image_path_validation"]["invalid_paths"] = invalid_count
            result["image_path_validation"]["missing_images"] = missing_count
            result["image_path_validation"]["sample_invalid"] = sample_invalid[:5]

        # Check data integrity: empty lists, length mismatches
        if "texts" in df.columns and "polygons" in df.columns:
            empty_texts = df[df["texts"].apply(lambda x: isinstance(x, list) and len(x) == 0)].index.tolist()
            empty_polygons = df[df["polygons"].apply(lambda x: isinstance(x, list) and len(x) == 0)].index.tolist()
            result["data_integrity"]["empty_lists"]["texts"] = len(empty_texts)
            result["data_integrity"]["empty_lists"]["polygons"] = len(empty_polygons)

            # Check length mismatches between texts and polygons
            mismatches = []
            for idx in df.index:
                texts = df.loc[idx, "texts"]
                polygons = df.loc[idx, "polygons"]
                if isinstance(texts, list) and isinstance(polygons, list):
                    if len(texts) != len(polygons):
                        mismatches.append({"row": int(idx), "texts_len": len(texts), "polygons_len": len(polygons)})
                        if len(mismatches) >= 10:
                            break
            result["data_integrity"]["length_mismatches"] = mismatches

        # Get a sample row (first non-null row)
        if len(df) > 0:
            sample = df.iloc[0].to_dict()
            # Truncate long values for readability
            sample_row = {}
            for key, value in sample.items():
                if isinstance(value, (list, dict)):
                    sample_row[key] = str(value)[:200] + "..." if len(str(value)) > 200 else value
                else:
                    sample_row[key] = value
            result["sample_row"] = sample_row

    except Exception as e:
        result["error"] = str(e)

    return result


def compare_dp_kie(dp_result: dict[str, Any], kie_result: dict[str, Any], split: str) -> dict[str, Any]:
    """Compare DP and KIE results for a given split."""
    comparison = {
        "split": split,
        "dp_exists": dp_result["exists"],
        "kie_exists": kie_result["exists"],
        "identical_files": False,
        "row_count_match": False,
        "column_differences": [],
        "schema_issues": [],
        "corruption_detected": False,
        "corruption_details": [],
    }

    if not dp_result["exists"] or not kie_result["exists"]:
        comparison["corruption_detected"] = True
        comparison["corruption_details"].append("One or both files are missing")
        return comparison

    # Check if files are identical (data corruption)
    if dp_result["row_count"] == kie_result["row_count"]:
        comparison["row_count_match"] = True
        # Try to read and compare actual data
        try:
            dp_df = pd.read_parquet(dp_result["path"])
            kie_df = pd.read_parquet(kie_result["path"])
            if dp_df.equals(kie_df):
                comparison["identical_files"] = True
                comparison["corruption_detected"] = True
                comparison["corruption_details"].append(
                    "DP and KIE files are identical - KIE should not contain polygons"
                )
        except Exception as e:
            comparison["corruption_details"].append(f"Could not compare files: {e}")

    # Check column differences
    dp_cols = set(dp_result.get("columns", []))
    kie_cols = set(kie_result.get("columns", []))
    comparison["column_differences"] = {
        "only_in_dp": list(dp_cols - kie_cols),
        "only_in_kie": list(kie_cols - dp_cols),
        "common": list(dp_cols & kie_cols),
    }

    # Schema validation
    # DP should have polygons
    if not dp_result.get("has_polygons", False):
        comparison["schema_issues"].append("DP file missing polygons column (DP should have polygons)")
        comparison["corruption_detected"] = True

    # KIE should NOT have polygons (according to user requirements)
    if kie_result.get("has_polygons", False):
        comparison["schema_issues"].append("KIE file contains polygons (KIE should only have key-value pairs)")
        comparison["corruption_detected"] = True

    return comparison


def main():
    """Main audit function."""
    base_path = project_root / "data" / "export"
    output_dir = project_root / "data" / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define files to audit
    files_to_audit = {
        "dp_train": base_path / "baseline_dp" / "train.parquet",
        "dp_val": base_path / "baseline_dp" / "val.parquet",
        "dp_test": base_path / "baseline_dp" / "test.parquet",
        "kie_train": base_path / "baseline_kie" / "train.parquet",
        "kie_val": base_path / "baseline_kie" / "val.parquet",
        "kie_test": base_path / "baseline_kie" / "test.parquet",
    }

    print("=" * 80)
    print("KIE/DP Dataset Corruption Audit")
    print("=" * 80)
    print()

    # Audit all files
    audit_results = {}
    for name, filepath in files_to_audit.items():
        print(f"Auditing {name}...")
        audit_results[name] = audit_parquet_file(filepath, project_root)
        if audit_results[name]["exists"]:
            print(f"  ✓ Found {audit_results[name]['row_count']} rows")
            print(f"  ✓ Columns: {', '.join(audit_results[name]['columns'])}")
            if audit_results[name]["has_polygons"]:
                print(f"  ⚠ Contains polygons column")
            else:
                print(f"  ✓ No polygons column")

            # Data integrity checks
            img_val = audit_results[name].get("image_path_validation", {})
            if img_val.get("missing_images", 0) > 0:
                print(f"  ⚠ {img_val['missing_images']} missing image files")

            nulls = audit_results[name].get("null_counts", {})
            if any(count > 0 for count in nulls.values()):
                print(f"  ⚠ Null values found: {nulls}")
        else:
            print(f"  ✗ File not found: {filepath}")
        print()

    # Compare DP vs KIE for each split
    print("=" * 80)
    print("DP vs KIE Comparisons")
    print("=" * 80)
    print()

    comparisons = {}
    for split in ["train", "val", "test"]:
        dp_key = f"dp_{split}"
        kie_key = f"kie_{split}"
        print(f"Comparing {split} split...")
        comparison = compare_dp_kie(audit_results[dp_key], audit_results[kie_key], split)
        comparisons[split] = comparison

        if comparison["corruption_detected"]:
            print(f"  ⚠ CORRUPTION DETECTED")
            for detail in comparison["corruption_details"]:
                print(f"    - {detail}")
            for issue in comparison["schema_issues"]:
                print(f"    - {issue}")
        else:
            print(f"  ✓ No corruption detected")
        print()

    # Generate recommendations
    recommendations = []
    corrupt_splits = []
    for split, comp in comparisons.items():
        if comp["corruption_detected"]:
            corrupt_splits.append(split)
            if comp["identical_files"]:
                recommendations.append(
                    f"{split.upper()}: KIE and DP files are identical. KIE needs API re-call to generate key-value pairs without polygons."
                )
            elif comp["schema_issues"]:
                for issue in comp["schema_issues"]:
                    if "KIE file contains polygons" in issue:
                        recommendations.append(
                            f"{split.upper()}: KIE file contains polygons. Needs API re-call to generate proper KIE data (key-value pairs only)."
                        )
                    elif "DP file missing polygons" in issue:
                        recommendations.append(
                            f"{split.upper()}: DP file missing polygons. Needs API re-call to generate proper DP data (with layout/polygons)."
                        )

    # Compile final report
    report = {
        "audit_timestamp": pd.Timestamp.now().isoformat(),
        "files_audited": {name: {"path": str(path), "exists": path.exists()} for name, path in files_to_audit.items()},
        "audit_results": audit_results,
        "comparisons": comparisons,
        "summary": {
            "total_files": len(files_to_audit),
            "files_found": sum(1 for r in audit_results.values() if r["exists"]),
            "corrupt_splits": corrupt_splits,
            "corruption_detected": len(corrupt_splits) > 0,
        },
        "recommendations": recommendations,
    }

    # Save detailed corruption report
    report_path = output_dir / "kie_dp_corruption_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Save comprehensive full audit report
    full_report_path = output_dir / "full_dataset_audit_report.json"
    with open(full_report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Files audited: {report['summary']['total_files']}")
    print(f"Files found: {report['summary']['files_found']}")
    print(f"Corrupt splits: {', '.join(corrupt_splits) if corrupt_splits else 'None'}")
    print()
    print("Recommendations:")
    for rec in recommendations:
        print(f"  - {rec}")
    print()
    print(f"Full report saved to: {report_path}")

    return report


if __name__ == "__main__":
    main()
