#!/usr/bin/env python3
"""
Data Cleaning Script for OCR Training Dataset

Scans the dataset for problematic samples and identifies:
- Corrupted or missing images
- Invalid or degenerate polygons
- Shape mismatches
- Missing annotations
- Invalid coordinate ranges
- Images with no valid polygons after filtering

Usage:
    # Scan and report issues (dry run)
    python scripts/data/clean_dataset.py --image-dir data/datasets/images/train --annotation-file data/datasets/jsons/train.json

    # Get simple list of images with bad polygons (one per line)
    python scripts/data/clean_dataset.py --image-dir data/datasets/images/train --annotation-file data/datasets/jsons/train.json --list-bad

    # Generate report and optionally remove bad samples
    python scripts/data/clean_dataset.py --image-dir data/datasets/images/train --annotation-file data/datasets/jsons/train.json --output-report reports/data_cleaning_report.json

    # Remove problematic samples from dataset
    python scripts/data/clean_dataset.py --image-dir data/datasets/images/train --annotation-file data/datasets/jsons/train.json --remove-bad --backup
"""

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFile

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Ensure repository root is importable
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ocr.core.utils.polygon_utils import filter_degenerate_polygons


class DatasetCleaner:
    """Scans and cleans OCR training dataset for problematic samples."""

    def __init__(self, image_dir: Path, annotation_file: Path | None = None, verbose: bool = False):
        self.image_dir = Path(image_dir)
        self.annotation_file = Path(annotation_file) if annotation_file else None
        self.verbose = verbose

        self.issues: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.stats = Counter()
        self.annotations: dict[str, Any] = {}

        if not self.image_dir.exists():
            raise ValueError(f"Image directory does not exist: {self.image_dir}")

        if self.annotation_file and not self.annotation_file.exists():
            raise ValueError(f"Annotation file does not exist: {self.annotation_file}")

    def log(self, message: str) -> None:
        """Log message if verbose mode."""
        if self.verbose:
            print(message)

    def load_annotations(self) -> None:
        """Load annotations from JSON file."""
        if not self.annotation_file:
            self.log("No annotation file provided - will only check images")
            return

        self.log(f"Loading annotations from {self.annotation_file}")
        try:
            with open(self.annotation_file, encoding="utf-8") as f:
                data = json.load(f)
            self.annotations = data.get("images", {})
            self.stats["total_annotations"] = len(self.annotations)
            self.log(f"Loaded {len(self.annotations)} image annotations")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in annotation file: {e}") from e
        except Exception as e:
            raise ValueError(f"Error loading annotation file: {e}") from e

    def validate_image(self, image_path: Path) -> dict[str, Any]:
        """Validate a single image file."""
        issues = []

        # Check if file exists
        if not image_path.exists():
            issues.append({"type": "missing_file", "message": "Image file does not exist"})
            return {"valid": False, "issues": issues}

        # Try to load image
        try:
            with Image.open(image_path) as img:
                # Check image format
                if img.format not in ("JPEG", "PNG", "JPG"):
                    issues.append(
                        {
                            "type": "invalid_format",
                            "message": f"Unsupported format: {img.format}",
                            "format": img.format,
                        }
                    )

                # Check image size
                width, height = img.size
                if width == 0 or height == 0:
                    issues.append({"type": "zero_size", "message": f"Image has zero size: {width}x{height}"})

                if width > 10000 or height > 10000:
                    issues.append(
                        {
                            "type": "extremely_large",
                            "message": f"Image is extremely large: {width}x{height}",
                            "width": width,
                            "height": height,
                        }
                    )

                # Try to convert to RGB array (this also validates the image is not corrupted)
                try:
                    img_array = np.array(img.convert("RGB"))
                    if img_array.size == 0:
                        issues.append({"type": "empty_array", "message": "Image array is empty"})
                except Exception as e:
                    issues.append({"type": "conversion_error", "message": f"Cannot convert to RGB: {e}"})

        except Exception as e:
            issues.append({"type": "load_error", "message": f"Cannot load image: {e}"})

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "width": width if "width" in locals() else None,
            "height": height if "height" in locals() else None,
        }

    def validate_polygon(
        self, polygon: list[list[float]], image_width: int | None = None, image_height: int | None = None
    ) -> dict[str, Any]:
        """Validate a single polygon."""
        issues = []

        # Check polygon structure
        if not isinstance(polygon, list):
            issues.append({"type": "invalid_structure", "message": "Polygon is not a list"})
            return {"valid": False, "issues": issues}

        if len(polygon) < 3:
            issues.append(
                {
                    "type": "too_few_points",
                    "message": f"Polygon has fewer than 3 points: {len(polygon)}",
                    "num_points": len(polygon),
                }
            )
            return {"valid": False, "issues": issues}

        # Convert to numpy array for validation
        try:
            poly_array = np.array(polygon, dtype=np.float32)
            if poly_array.shape != (len(polygon), 2):
                issues.append(
                    {
                        "type": "invalid_shape",
                        "message": f"Polygon shape is invalid: {poly_array.shape}, expected ({len(polygon)}, 2)",
                        "shape": poly_array.shape.tolist(),
                    }
                )
                return {"valid": False, "issues": issues}

            # Check for NaN or Inf values
            if np.any(np.isnan(poly_array)) or np.any(np.isinf(poly_array)):
                issues.append({"type": "nan_or_inf", "message": "Polygon contains NaN or Inf values"})

            # Check coordinate ranges
            x_coords = poly_array[:, 0]
            y_coords = poly_array[:, 1]

            if image_width is not None:
                if np.any(x_coords < 0) or np.any(x_coords > image_width):
                    issues.append(
                        {
                            "type": "out_of_bounds_x",
                            "message": f"X coordinates out of bounds [0, {image_width}]",
                            "min_x": float(np.min(x_coords)),
                            "max_x": float(np.max(x_coords)),
                        }
                    )

            if image_height is not None:
                if np.any(y_coords < 0) or np.any(y_coords > image_height):
                    issues.append(
                        {
                            "type": "out_of_bounds_y",
                            "message": f"Y coordinates out of bounds [0, {image_height}]",
                            "min_y": float(np.min(y_coords)),
                            "max_y": float(np.max(y_coords)),
                        }
                    )

            # Check for degenerate polygons
            width_span = float(np.max(x_coords) - np.min(x_coords))
            height_span = float(np.max(y_coords) - np.min(y_coords))

            if width_span < 1.0 or height_span < 1.0:
                issues.append(
                    {
                        "type": "too_small",
                        "message": f"Polygon is too small: {width_span:.2f}x{height_span:.2f}",
                        "width_span": width_span,
                        "height_span": height_span,
                    }
                )

            # Check if polygon would be filtered by filter_degenerate_polygons
            filtered = filter_degenerate_polygons([poly_array], min_side=1.0)
            if len(filtered) == 0:
                issues.append(
                    {
                        "type": "would_be_filtered",
                        "message": "Polygon would be filtered by degenerate polygon filter",
                    }
                )

        except Exception as e:
            issues.append({"type": "conversion_error", "message": f"Cannot convert polygon to array: {e}"})
            return {"valid": False, "issues": issues}

        return {"valid": len(issues) == 0, "issues": issues}

    def validate_annotations(self, filename: str, image_width: int | None = None, image_height: int | None = None) -> dict[str, Any]:
        """Validate annotations for a single image."""
        issues = []

        if filename not in self.annotations:
            issues.append({"type": "missing_annotation", "message": "No annotation found for image"})
            return {"valid": False, "issues": issues, "num_polygons": 0}

        image_annotation = self.annotations[filename]
        words = image_annotation.get("words", {})

        if not words:
            issues.append({"type": "empty_annotation", "message": "Annotation has no words"})
            return {"valid": True, "issues": issues, "num_polygons": 0}  # Empty annotation is valid

        polygons = []
        for word_id, word_data in words.items():
            points = word_data.get("points")
            if not points:
                issues.append({"type": "missing_points", "message": f"Word {word_id} has no points"})
                continue

            poly_result = self.validate_polygon(points, image_width, image_height)
            if not poly_result["valid"]:
                issues.extend(poly_result["issues"])
            else:
                polygons.append(points)

        # Check if image would have no valid polygons after filtering
        if polygons:
            try:
                poly_arrays = [np.array(p, dtype=np.float32) for p in polygons]
                filtered = filter_degenerate_polygons(poly_arrays, min_side=1.0)
                if len(filtered) == 0:
                    issues.append(
                        {
                            "type": "all_polygons_filtered",
                            "message": f"All {len(polygons)} polygons would be filtered (image would have no training data)",
                            "original_count": len(polygons),
                        }
                    )
            except Exception as e:
                issues.append({"type": "filtering_error", "message": f"Error checking polygon filtering: {e}"})

        return {"valid": len(issues) == 0, "issues": issues, "num_polygons": len(polygons)}

    def scan_dataset(self) -> dict[str, Any]:
        """Scan the entire dataset for issues."""
        self.log("Starting dataset scan...")

        # Load annotations if available
        if self.annotation_file:
            self.load_annotations()

        # Get all image files
        image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
        image_files = [f for f in self.image_dir.iterdir() if f.is_file() and f.suffix in image_extensions]

        self.stats["total_images"] = len(image_files)
        self.log(f"Found {len(image_files)} image files")

        # Scan each image
        for image_file in image_files:
            filename = image_file.name
            self.log(f"Scanning {filename}...")

            # Validate image
            image_result = self.validate_image(image_file)
            if not image_result["valid"]:
                self.issues["image_errors"].append(
                    {
                        "filename": filename,
                        "path": str(image_file),
                        "issues": image_result["issues"],
                    }
                )
                self.stats["images_with_errors"] += 1
                continue

            self.stats["valid_images"] += 1

            # Validate annotations if available
            if self.annotation_file:
                annotation_result = self.validate_annotations(filename, image_result.get("width"), image_result.get("height"))
                if not annotation_result["valid"]:
                    self.issues["annotation_errors"].append(
                        {
                            "filename": filename,
                            "path": str(image_file),
                            "issues": annotation_result["issues"],
                            "num_polygons": annotation_result["num_polygons"],
                        }
                    )
                    self.stats["images_with_annotation_errors"] += 1
                else:
                    self.stats["images_with_valid_annotations"] += 1
                    if annotation_result["num_polygons"] == 0:
                        self.stats["images_with_no_polygons"] += 1

            # Check if image has annotation but annotation file doesn't reference it
            if self.annotation_file and filename not in self.annotations:
                self.issues["orphaned_images"].append(
                    {
                        "filename": filename,
                        "path": str(image_file),
                        "message": "Image exists but has no annotation",
                    }
                )
                self.stats["orphaned_images"] += 1

        # Check for annotations referencing missing images
        if self.annotation_file:
            for filename in self.annotations.keys():
                image_path = self.image_dir / filename
                if not image_path.exists():
                    self.issues["missing_images"].append(
                        {
                            "filename": filename,
                            "path": str(image_path),
                            "message": "Annotation references missing image",
                        }
                    )
                    self.stats["missing_images"] += 1

        return self.generate_report()

    def generate_report(self) -> dict[str, Any]:
        """Generate a summary report of all issues."""
        total_issues = sum(len(issue_list) for issue_list in self.issues.values())

        report = {
            "summary": {
                "total_images": self.stats["total_images"],
                "valid_images": self.stats["valid_images"],
                "images_with_errors": self.stats["images_with_errors"],
                "total_issues": total_issues,
                "issues_by_category": {category: len(issue_list) for category, issue_list in self.issues.items()},
            },
            "statistics": dict(self.stats),
            "issues": {category: issue_list for category, issue_list in self.issues.items() if issue_list},
        }

        return report

    def print_report(self, report: dict[str, Any]) -> None:
        """Print a human-readable report."""
        summary = report["summary"]
        stats = report["statistics"]

        print("\n" + "=" * 80)
        print("📊 DATASET CLEANING REPORT")
        print("=" * 80)

        print(f"\n📁 Dataset: {self.image_dir}")
        if self.annotation_file:
            print(f"📄 Annotations: {self.annotation_file}")

        print("\n📈 Summary:")
        print(f"  Total images: {summary['total_images']}")
        print(f"  Valid images: {summary['valid_images']}")
        print(f"  Images with errors: {summary['images_with_errors']}")

        if self.annotation_file:
            print(f"  Images with valid annotations: {stats.get('images_with_valid_annotations', 0)}")
            print(f"  Images with annotation errors: {stats.get('images_with_annotation_errors', 0)}")
            print(f"  Images with no polygons: {stats.get('images_with_no_polygons', 0)}")
            print(f"  Orphaned images (no annotation): {stats.get('orphaned_images', 0)}")
            print(f"  Missing images (annotation but no file): {stats.get('missing_images', 0)}")

        print("\n⚠️  Issues by Category:")
        for category, count in summary["issues_by_category"].items():
            print(f"  {category}: {count}")

        # Print sample issues
        if report["issues"]:
            print("\n🔍 Sample Issues (showing first 5 per category):")
            for category, issue_list in report["issues"].items():
                print(f"\n  {category.upper()}:")
                for issue in issue_list[:5]:
                    print(f"    - {issue['filename']}")
                    for issue_detail in issue.get("issues", [])[:2]:
                        print(f"      • {issue_detail.get('type', 'unknown')}: {issue_detail.get('message', 'No message')}")
                if len(issue_list) > 5:
                    print(f"    ... and {len(issue_list) - 5} more")

        print("\n" + "=" * 80)

    def remove_bad_samples(self, report: dict[str, Any], backup: bool = False) -> None:
        """Remove problematic samples from the dataset."""
        removed_count = 0

        # Create backup if requested
        if backup:
            backup_dir = self.image_dir.parent / f"{self.image_dir.name}_backup"
            if not backup_dir.exists():
                self.log(f"Creating backup at {backup_dir}")
                shutil.copytree(self.image_dir, backup_dir)
                self.log("Backup created")

        # Remove images with errors
        for issue in self.issues.get("image_errors", []):
            image_path = Path(issue["path"])
            if image_path.exists():
                self.log(f"Removing corrupted image: {image_path.name}")
                image_path.unlink()
                removed_count += 1

        # Remove images with all polygons filtered
        for issue in self.issues.get("annotation_errors", []):
            # Check if all polygons would be filtered
            if any(i.get("type") == "all_polygons_filtered" for i in issue.get("issues", [])):
                image_path = Path(issue["path"])
                if image_path.exists():
                    self.log(f"Removing image with no valid polygons: {image_path.name}")
                    image_path.unlink()
                    removed_count += 1

        # Remove orphaned images (no annotation)
        for issue in self.issues.get("orphaned_images", []):
            image_path = Path(issue["path"])
            if image_path.exists():
                self.log(f"Removing orphaned image: {image_path.name}")
                image_path.unlink()
                removed_count += 1

        # Update annotation file if it exists
        if self.annotation_file and self.annotations:
            # Remove entries for missing images and problematic images
            images_to_remove = set()

            # Add missing images
            for issue in self.issues.get("missing_images", []):
                images_to_remove.add(issue["filename"])

            # Add images with critical annotation errors
            for issue in self.issues.get("annotation_errors", []):
                if any(i.get("type") == "all_polygons_filtered" for i in issue.get("issues", [])):
                    images_to_remove.add(issue["filename"])

            if images_to_remove:
                original_count = len(self.annotations)
                for filename in images_to_remove:
                    self.annotations.pop(filename, None)

                # Write updated annotations
                updated_data = {"images": self.annotations}
                backup_annotation = self.annotation_file.with_suffix(self.annotation_file.suffix + ".backup")
                if backup and not backup_annotation.exists():
                    shutil.copy2(self.annotation_file, backup_annotation)
                    self.log(f"Backed up annotation file to {backup_annotation}")

                with open(self.annotation_file, "w", encoding="utf-8") as f:
                    json.dump(updated_data, f, indent=2, ensure_ascii=False)

                removed_annotations = original_count - len(self.annotations)
                self.log(f"Removed {removed_annotations} entries from annotation file")

        print(f"\n✅ Removed {removed_count} problematic image files")
        if self.annotation_file:
            print(f"✅ Updated annotation file: {self.annotation_file}")


def main():
    parser = argparse.ArgumentParser(description="Clean OCR training dataset by identifying and removing problematic samples")
    parser.add_argument("--image-dir", type=Path, required=True, help="Directory containing training images")
    parser.add_argument("--annotation-file", type=Path, help="Path to annotation JSON file")
    parser.add_argument("--output-report", type=Path, help="Path to save JSON report")
    parser.add_argument("--list-bad", action="store_true", help="Output simple list of images with bad polygons (one per line)")
    parser.add_argument("--remove-bad", action="store_true", help="Remove problematic samples from dataset")
    parser.add_argument("--backup", action="store_true", help="Create backup before removing files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Create cleaner
    try:
        cleaner = DatasetCleaner(args.image_dir, args.annotation_file, verbose=args.verbose)
    except ValueError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Scan dataset
    try:
        report = cleaner.scan_dataset()
    except Exception as e:
        print(f"❌ Error during scan: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)

    # Output simple list of bad images if requested
    if args.list_bad:
        bad_images = []
        # Collect all images with annotation errors (bad polygons)
        for issue in cleaner.issues.get("annotation_errors", []):
            bad_images.append(issue["filename"])
        # Also include images with image errors
        for issue in cleaner.issues.get("image_errors", []):
            bad_images.append(issue["filename"])

        if bad_images:
            for filename in sorted(bad_images):
                print(filename)
            sys.exit(0)
        else:
            print("No problematic images found.", file=sys.stderr)
            sys.exit(0)

    # Print report
    cleaner.print_report(report)

    # Save report if requested
    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Report saved to: {args.output_report}")

    # Remove bad samples if requested
    if args.remove_bad:
        if not args.backup:
            response = input("\n⚠️  WARNING: This will permanently delete files. Create backup? (y/n): ")
            if response.lower() == "y":
                args.backup = True

        try:
            cleaner.remove_bad_samples(report, backup=args.backup)
        except Exception as e:
            print(f"❌ Error during removal: {e}", file=sys.stderr)
            if args.verbose:
                import traceback

                traceback.print_exc()
            sys.exit(1)
    else:
        print("\n💡 Tip: Use --remove-bad to automatically remove problematic samples")
        print("💡 Tip: Use --backup to create a backup before removal")

    # Exit with error code if issues found
    if report["summary"]["total_issues"] > 0:
        sys.exit(1)
    else:
        print("\n✅ No issues found! Dataset is clean.")
        sys.exit(0)


if __name__ == "__main__":
    main()
