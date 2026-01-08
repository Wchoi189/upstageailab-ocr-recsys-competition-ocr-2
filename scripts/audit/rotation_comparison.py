#!/usr/bin/env python3
"""Compare original vs optimized train images to determine rotation issue root cause.

This script checks EXIF orientation tags in both original and optimized images
to determine if the issue is:
1. Original train data has rotation that wasn't corrected
2. optimize_images.py incorrectly preserves/applies EXIF tags
3. Both
"""

import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import orientation utilities
try:
    from ocr.core.utils.orientation import get_exif_orientation, normalize_pil_image
    from ocr.core.utils.orientation_constants import EXIF_ORIENTATION_TAG
except ImportError:
    print("Warning: Could not import ocr.core.utils.orientation, using basic PIL methods")
    get_exif_orientation = None
    normalize_pil_image = None
    EXIF_ORIENTATION_TAG = 274  # Standard EXIF orientation tag


def get_exif_orientation_basic(image: Image.Image) -> int:
    """Get EXIF orientation tag using basic PIL methods."""
    try:
        exif = image.getexif()
        if exif is not None:
            orientation = exif.get(EXIF_ORIENTATION_TAG, 1)
            return orientation
    except Exception:
        pass
    return 1


def check_image_rotation(image_path: Path) -> dict[str, Any]:
    """Check rotation information for a single image."""
    result = {
        "path": str(image_path),
        "exists": False,
        "error": None,
        "exif_orientation": 1,
        "image_size": None,
        "has_exif": False,
    }

    if not image_path.exists():
        result["error"] = "File does not exist"
        return result

    try:
        with Image.open(image_path) as img:
            result["exists"] = True
            result["image_size"] = img.size  # (width, height)

            # Get EXIF orientation
            if get_exif_orientation:
                try:
                    orientation = get_exif_orientation(img)
                    result["exif_orientation"] = orientation
                    result["has_exif"] = orientation != 1
                except Exception as e:
                    result["error"] = f"Error getting orientation: {e}"
                    result["exif_orientation"] = get_exif_orientation_basic(img)
            else:
                result["exif_orientation"] = get_exif_orientation_basic(img)
                result["has_exif"] = result["exif_orientation"] != 1

    except Exception as e:
        result["error"] = str(e)

    return result


def compare_images(original_path: Path, optimized_path: Path) -> dict[str, Any]:
    """Compare original and optimized images."""
    comparison = {
        "original": check_image_rotation(original_path),
        "optimized": check_image_rotation(optimized_path),
        "rotation_issue_detected": False,
        "issue_type": None,
        "details": [],
    }

    orig = comparison["original"]
    opt = comparison["optimized"]

    if not orig["exists"] or not opt["exists"]:
        comparison["details"].append("One or both images missing")
        return comparison

    # Check if orientations differ
    if orig["exif_orientation"] != opt["exif_orientation"]:
        comparison["rotation_issue_detected"] = True
        comparison["issue_type"] = "orientation_mismatch"
        comparison["details"].append(
            f"Orientation differs: original={orig['exif_orientation']}, optimized={opt['exif_orientation']}"
        )

    # Check if optimized has rotation when original doesn't
    if orig["exif_orientation"] == 1 and opt["exif_orientation"] != 1:
        comparison["rotation_issue_detected"] = True
        comparison["issue_type"] = "optimized_added_rotation"
        comparison["details"].append(
            f"Optimized image has rotation (orientation={opt['exif_orientation']}) but original doesn't"
        )

    # Check if original has rotation that should have been normalized
    if orig["exif_orientation"] != 1:
        comparison["rotation_issue_detected"] = True
        comparison["issue_type"] = "original_has_rotation"
        comparison["details"].append(
            f"Original image has rotation (orientation={orig['exif_orientation']}) that may need correction"
        )

    # Check if sizes differ (rotation can change dimensions)
    if orig["image_size"] and opt["image_size"]:
        if orig["image_size"] != opt["image_size"]:
            comparison["details"].append(
                f"Size differs: original={orig['image_size']}, optimized={opt['image_size']}"
            )

    return comparison


def main():
    """Main comparison function."""
    output_dir = project_root / "data" / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Example image from user's report
    example_original = project_root / "data/raw/competition/baseline_text_detection/images/train/drp.en_ko.in_house.selectstar_000006.jpg"
    example_optimized = project_root / "data/optimized_images/baseline_train/train_drp.en_ko.in_house.selectstar_000006.jpg"

    print("=" * 80)
    print("Rotation Comparison Audit")
    print("=" * 80)
    print()

    # Check example image
    print("Checking example image...")
    print(f"  Original: {example_original}")
    print(f"  Optimized: {example_optimized}")
    print()

    example_comparison = compare_images(example_original, example_optimized)

    print("Example Image Results:")
    print(f"  Original orientation: {example_comparison['original']['exif_orientation']}")
    print(f"  Optimized orientation: {example_comparison['optimized']['exif_orientation']}")
    if example_comparison["rotation_issue_detected"]:
        print(f"  ⚠ ROTATION ISSUE DETECTED: {example_comparison['issue_type']}")
        for detail in example_comparison["details"]:
            print(f"    - {detail}")
    else:
        print(f"  ✓ No rotation issue detected")
    print()

    # Sample more images from optimized directory
    optimized_dir = project_root / "data/optimized_images/baseline_train"
    original_dir = project_root / "data/raw/competition/baseline_text_detection/images/train"

    comparisons = []
    if optimized_dir.exists() and original_dir.exists():
        print("Sampling additional images...")
        optimized_images = list(optimized_dir.glob("*.jpg"))[:10]  # Sample first 10

        for opt_img in optimized_images:
            # Try to find corresponding original
            # Optimized filename format: train_<original_filename>
            orig_filename = opt_img.name.replace("train_", "")
            orig_path = original_dir / orig_filename

            if orig_path.exists():
                comp = compare_images(orig_path, opt_img)
                comparisons.append(comp)
                if comp["rotation_issue_detected"]:
                    print(f"  ⚠ {orig_filename}: {comp['issue_type']}")
            else:
                # Try without train_ prefix
                orig_path2 = original_dir / opt_img.name
                if orig_path2.exists():
                    comp = compare_images(orig_path2, opt_img)
                    comparisons.append(comp)
                    if comp["rotation_issue_detected"]:
                        print(f"  ⚠ {opt_img.name}: {comp['issue_type']}")

    print()

    # Analyze results
    total_checked = len(comparisons) + 1  # +1 for example
    issues_found = sum(1 for c in comparisons + [example_comparison] if c["rotation_issue_detected"])

    issue_types = {}
    for comp in comparisons + [example_comparison]:
        if comp["rotation_issue_detected"]:
            issue_type = comp["issue_type"]
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1

    # Generate report
    report = {
        "audit_timestamp": str(Path(__file__).stat().st_mtime),
        "example_comparison": example_comparison,
        "sample_comparisons": comparisons,
        "summary": {
            "total_checked": total_checked,
            "issues_found": issues_found,
            "issue_types": issue_types,
        },
        "recommendations": [],
    }

    # Generate recommendations
    if "optimized_added_rotation" in issue_types:
        report["recommendations"].append(
            "optimize_images.py is adding rotation. Fix by normalizing EXIF orientation before saving."
        )
    if "original_has_rotation" in issue_types:
        report["recommendations"].append(
            "Original train images contain EXIF rotation tags. Consider normalizing source data."
        )
    if "orientation_mismatch" in issue_types:
        report["recommendations"].append(
            "Orientation mismatch between original and optimized. optimize_images.py needs EXIF handling."
        )

    # Save report
    report_path = output_dir / "optimized_image_rotation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Images checked: {total_checked}")
    print(f"Issues found: {issues_found}")
    print(f"Issue types: {issue_types}")
    print()
    print("Recommendations:")
    for rec in report["recommendations"]:
        print(f"  - {rec}")
    print()
    print(f"Full report saved to: {report_path}")

    return report


if __name__ == "__main__":
    main()
