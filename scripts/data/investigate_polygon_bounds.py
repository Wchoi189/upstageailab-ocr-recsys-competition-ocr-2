#!/usr/bin/env python3
"""
Investigate Root Cause of Out-of-Bounds Polygon Coordinates

BUG-20251116-001: This script investigates why polygons have out-of-bounds coordinates by:
1. Analyzing coordinate distributions
2. Checking EXIF orientation remapping effects
3. Identifying patterns in out-of-bounds coordinates
4. Comparing raw vs canonical dimensions

Usage:
    python scripts/data/investigate_polygon_bounds.py \
        --annotation-file data/datasets/jsons/train.json \
        --image-dir data/datasets/images/train \
        --output-report reports/polygon_bounds_investigation.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Ensure repository root is importable
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ocr.utils.orientation import get_exif_orientation, normalize_pil_image, remap_polygons


class PolygonBoundsInvestigator:
    """Investigate root cause of out-of-bounds polygon coordinates."""

    def __init__(self, annotation_file: Path, image_dir: Path, verbose: bool = False):
        self.annotation_file = Path(annotation_file)
        self.image_dir = Path(image_dir)
        self.verbose = verbose

        self.stats = defaultdict(int)
        self.analysis = defaultdict(list)

        if not self.annotation_file.exists():
            raise ValueError(f"Annotation file does not exist: {self.annotation_file}")
        if not self.image_dir.exists():
            raise ValueError(f"Image directory does not exist: {self.image_dir}")

    def log(self, message: str) -> None:
        """Log message if verbose mode."""
        if self.verbose:
            print(message)

    def load_annotations(self) -> dict[str, Any]:
        """Load annotations from JSON file."""
        self.log(f"Loading annotations from {self.annotation_file}")
        with open(self.annotation_file, encoding="utf-8") as f:
            data = json.load(f)
        return data

    def get_image_info(self, filename: str) -> dict[str, Any] | None:
        """Get comprehensive image information."""
        image_path = self.image_dir / filename
        if not image_path.exists():
            return None

        try:
            with Image.open(image_path) as img:
                raw_width, raw_height = img.size
                orientation = get_exif_orientation(img)

                # Get canonical dimensions after EXIF normalization
                normalized_img, _ = normalize_pil_image(img)
                canonical_width, canonical_height = normalized_img.size
                normalized_img.close()

                return {
                    "raw_width": raw_width,
                    "raw_height": raw_height,
                    "canonical_width": canonical_width,
                    "canonical_height": canonical_height,
                    "orientation": orientation,
                    "dimensions_swapped": (raw_width, raw_height) != (canonical_width, canonical_height),
                }
        except Exception as e:
            self.log(f"Error loading image {filename}: {e}")
            return None

    def analyze_polygon_coordinates(self, polygon: list[list[float]], image_info: dict[str, Any]) -> dict[str, Any]:
        """
        Analyze polygon coordinates for out-of-bounds issues.

        BUG-20251116-001: Investigates coordinate patterns to identify root cause.
        """
        poly_array = np.array(polygon, dtype=np.float32)
        x_coords = poly_array[:, 0]
        y_coords = poly_array[:, 1]

        canonical_width = image_info["canonical_width"]
        canonical_height = image_info["canonical_height"]
        raw_width = image_info["raw_width"]
        raw_height = image_info["raw_height"]
        orientation = image_info["orientation"]

        # Check bounds violations
        x_min, x_max = float(np.min(x_coords)), float(np.max(x_coords))
        y_min, y_max = float(np.min(y_coords)), float(np.max(y_coords))

        x_out_of_bounds = x_min < 0 or x_max > canonical_width
        y_out_of_bounds = y_min < 0 or y_max > canonical_height

        # Calculate how far out of bounds
        x_under = min(0, x_min)
        x_over = max(0, x_max - canonical_width)
        y_under = min(0, y_min)
        y_over = max(0, y_max - canonical_height)

        # Check if coordinates match raw dimensions (might indicate wrong frame)
        x_matches_raw = abs(x_max - raw_width) < 5 or abs(x_min) < 5
        y_matches_raw = abs(y_max - raw_height) < 5 or abs(y_min) < 5

        # Check if coordinates match canonical dimensions
        x_matches_canonical = abs(x_max - canonical_width) < 5 or abs(x_min) < 5
        y_matches_canonical = abs(y_max - canonical_height) < 5 or abs(y_min) < 5

        return {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "x_out_of_bounds": x_out_of_bounds,
            "y_out_of_bounds": y_out_of_bounds,
            "x_under": x_under,
            "x_over": x_over,
            "y_under": y_under,
            "y_over": y_over,
            "x_matches_raw": x_matches_raw,
            "y_matches_raw": y_matches_raw,
            "x_matches_canonical": x_matches_canonical,
            "y_matches_canonical": y_matches_canonical,
            "orientation": orientation,
            "dimensions_swapped": image_info["dimensions_swapped"],
        }

    def test_exif_remapping(self, polygon: list[list[float]], image_info: dict[str, Any]) -> dict[str, Any]:
        """
        Test what happens when polygon is remapped with EXIF orientation.

        BUG-20251116-001: Checks if EXIF remapping produces out-of-bounds coordinates.
        """
        poly_array = np.array(polygon, dtype=np.float32)
        raw_width = image_info["raw_width"]
        raw_height = image_info["raw_height"]
        orientation = image_info["orientation"]
        canonical_width = image_info["canonical_width"]
        canonical_height = image_info["canonical_height"]

        # Test remapping
        try:
            remapped = remap_polygons([poly_array], raw_width, raw_height, orientation)
            if remapped:
                remapped_array = remapped[0]
                remapped_x = remapped_array[:, 0]
                remapped_y = remapped_array[:, 1]

                remapped_x_min, remapped_x_max = float(np.min(remapped_x)), float(np.max(remapped_x))
                remapped_y_min, remapped_y_max = float(np.min(remapped_y)), float(np.max(remapped_y))

                remapped_x_out = remapped_x_min < 0 or remapped_x_max > canonical_width
                remapped_y_out = remapped_y_min < 0 or remapped_y_max > canonical_height

                return {
                    "remapped_x_min": remapped_x_min,
                    "remapped_x_max": remapped_x_max,
                    "remapped_y_min": remapped_y_min,
                    "remapped_y_max": remapped_y_max,
                    "remapped_x_out_of_bounds": remapped_x_out,
                    "remapped_y_out_of_bounds": remapped_y_out,
                    "remapping_causes_issue": remapped_x_out or remapped_y_out,
                }
        except Exception as e:
            return {"remapping_error": str(e)}

        return {}

    def investigate(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Investigate out-of-bounds polygon coordinates.

        BUG-20251116-001: Comprehensive investigation of coordinate issues.
        """
        images = data.get("images", {})
        self.log(f"Investigating {len(images)} images...")

        results = {
            "summary": {},
            "out_of_bounds_cases": [],
            "patterns": defaultdict(int),
            "orientation_analysis": defaultdict(int),
        }

        for filename, image_data in images.items():
            # Get image info
            image_info = self.get_image_info(filename)
            if image_info is None:
                self.stats["images_skipped"] += 1
                continue

            self.stats["images_analyzed"] += 1
            orientation = image_info["orientation"]
            results["orientation_analysis"][f"orientation_{orientation}"] += 1

            # Process polygons
            words = image_data.get("words", {})
            if not words:
                continue

            for word_id, word_data in words.items():
                points = word_data.get("points")
                if not points or not isinstance(points, list):
                    continue

                # Convert to list of lists if needed
                if isinstance(points[0], int | float):
                    if len(points) % 2 != 0:
                        continue
                    polygon = [[points[i], points[i + 1]] for i in range(0, len(points), 2)]
                else:
                    polygon = points

                if len(polygon) < 3:
                    continue

                self.stats["polygons_analyzed"] += 1

                # Analyze coordinates
                analysis = self.analyze_polygon_coordinates(polygon, image_info)

                # Test EXIF remapping
                remap_test = self.test_exif_remapping(polygon, image_info)

                # Check if out of bounds
                if analysis["x_out_of_bounds"] or analysis["y_out_of_bounds"]:
                    self.stats["polygons_out_of_bounds"] += 1

                    case = {
                        "filename": filename,
                        "word_id": word_id,
                        "image_info": image_info,
                        "coordinate_analysis": analysis,
                        "remap_test": remap_test,
                    }
                    results["out_of_bounds_cases"].append(case)

                    # Track patterns
                    if analysis["x_under"] < 0:
                        results["patterns"]["negative_x"] += 1
                    if analysis["x_over"] > 0:
                        results["patterns"]["x_exceeds_width"] += 1
                    if analysis["y_under"] < 0:
                        results["patterns"]["negative_y"] += 1
                    if analysis["y_over"] > 0:
                        results["patterns"]["y_exceeds_height"] += 1

                    if analysis["x_matches_raw"] and image_info["dimensions_swapped"]:
                        results["patterns"]["x_matches_raw_when_swapped"] += 1
                    if analysis["y_matches_raw"] and image_info["dimensions_swapped"]:
                        results["patterns"]["y_matches_raw_when_swapped"] += 1

                    if remap_test.get("remapping_causes_issue"):
                        results["patterns"]["remapping_causes_issue"] += 1

        # Generate summary
        results["summary"] = {
            "total_images": len(images),
            "images_analyzed": self.stats["images_analyzed"],
            "images_skipped": self.stats["images_skipped"],
            "polygons_analyzed": self.stats["polygons_analyzed"],
            "polygons_out_of_bounds": self.stats["polygons_out_of_bounds"],
            "patterns": dict(results["patterns"]),
            "orientation_distribution": dict(results["orientation_analysis"]),
        }

        return results

    def print_summary(self, results: dict[str, Any]) -> None:
        """Print investigation summary."""
        print("\n" + "=" * 80)
        print("🔍 POLYGON BOUNDS INVESTIGATION SUMMARY")
        print("=" * 80)

        summary = results["summary"]
        print("\n📊 Statistics:")
        print(f"  Total images: {summary['total_images']}")
        print(f"  Images analyzed: {summary['images_analyzed']}")
        print(f"  Images skipped: {summary['images_skipped']}")
        print(f"  Polygons analyzed: {summary['polygons_analyzed']}")
        print(f"  Polygons out of bounds: {summary['polygons_out_of_bounds']}")

        if summary["polygons_out_of_bounds"] > 0:
            print("\n📈 Patterns:")
            for pattern, count in summary["patterns"].items():
                print(f"  {pattern}: {count}")

            print("\n🔄 Orientation Distribution:")
            for orientation, count in summary["orientation_distribution"].items():
                print(f"  {orientation}: {count}")

            print("\n🔍 Sample Out-of-Bounds Cases (first 5):")
            for case in results["out_of_bounds_cases"][:5]:
                print(f"\n  {case['filename']} (word {case['word_id']}):")
                info = case["image_info"]
                analysis = case["coordinate_analysis"]
                print(f"    Orientation: {info['orientation']}")
                print(f"    Raw dimensions: {info['raw_width']}x{info['raw_height']}")
                print(f"    Canonical dimensions: {info['canonical_width']}x{info['canonical_height']}")
                print(f"    Dimensions swapped: {info['dimensions_swapped']}")
                print(f"    X range: [{analysis['x_min']:.1f}, {analysis['x_max']:.1f}]")
                print(f"    Y range: [{analysis['y_min']:.1f}, {analysis['y_max']:.1f}]")
                if analysis["x_out_of_bounds"]:
                    print(f"    ⚠️  X out of bounds: under={analysis['x_under']:.1f}, over={analysis['x_over']:.1f}")
                if analysis["y_out_of_bounds"]:
                    print(f"    ⚠️  Y out of bounds: under={analysis['y_under']:.1f}, over={analysis['y_over']:.1f}")
                if case["remap_test"].get("remapping_causes_issue"):
                    print("    ⚠️  EXIF remapping causes out-of-bounds coordinates")

        print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Investigate root cause of out-of-bounds polygon coordinates (BUG-20251116-001)")
    parser.add_argument("--annotation-file", type=Path, required=True, help="Path to annotation JSON file")
    parser.add_argument("--image-dir", type=Path, required=True, help="Directory containing images")
    parser.add_argument("--output-report", type=Path, help="Path to save investigation report (JSON)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Create investigator
    try:
        investigator = PolygonBoundsInvestigator(args.annotation_file, args.image_dir, verbose=args.verbose)
    except ValueError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Load annotations
    try:
        data = investigator.load_annotations()
    except Exception as e:
        print(f"❌ Error loading annotations: {e}", file=sys.stderr)
        sys.exit(1)

    # Investigate
    try:
        results = investigator.investigate(data)
    except Exception as e:
        print(f"❌ Error during investigation: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)

    # Print summary
    investigator.print_summary(results)

    # Save report if requested
    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_report, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Investigation report saved to: {args.output_report}")

    sys.exit(0)


if __name__ == "__main__":
    main()
