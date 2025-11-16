#!/usr/bin/env python3
"""
Fix Out-of-Bounds Polygon Coordinates in Annotation Files

BUG-20251116-001: This script fixes polygon coordinates that are out of bounds by:
1. Clamping coordinates to valid range [0, width] x [0, height]
2. Handling EXIF orientation if needed
3. Preserving polygon geometry while ensuring bounds compliance

Usage:
    # Dry run - show what would be fixed
    python scripts/data/fix_polygon_coordinates.py \
        --annotation-file data/datasets/jsons/train.json \
        --image-dir data/datasets/images/train \
        --tolerance 3.0

    # Fix coordinates and save to new file
    python scripts/data/fix_polygon_coordinates.py \
        --annotation-file data/datasets/jsons/train.json \
        --image-dir data/datasets/images/train \
        --output-file data/datasets/jsons/train_fixed.json \
        --tolerance 3.0 \
        --backup

    # Fix in place (overwrites original)
    python scripts/data/fix_polygon_coordinates.py \
        --annotation-file data/datasets/jsons/train.json \
        --image-dir data/datasets/images/train \
        --in-place \
        --backup
"""

import argparse
import json
import shutil
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


class PolygonCoordinateFixer:
    """Fix out-of-bounds polygon coordinates in annotation files."""

    def __init__(
        self,
        annotation_file: Path,
        image_dir: Path,
        tolerance: float = 3.0,
        verbose: bool = False,
    ):
        self.annotation_file = Path(annotation_file)
        self.image_dir = Path(image_dir)
        self.tolerance = tolerance
        self.verbose = verbose

        self.stats = defaultdict(int)
        self.fixes_applied = []

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

    def get_image_dimensions(self, filename: str) -> tuple[int, int, int] | None:
        """Get image dimensions and EXIF orientation."""
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
                return (canonical_width, canonical_height, orientation)
        except Exception as e:
            self.log(f"Error loading image {filename}: {e}")
            return None

    def clamp_polygon_coordinates(
        self,
        polygon: list[list[float]],
        image_width: int,
        image_height: int,
        tolerance: float,
    ) -> tuple[list[list[float]], dict[str, Any]]:
        """
        Clamp polygon coordinates to valid range.

        BUG-20251116-001: Clamps coordinates within tolerance to [0, width] x [0, height].
        Coordinates outside tolerance are clamped but logged as warnings.

        Returns:
            Tuple of (fixed_polygon, fix_info)
        """
        poly_array = np.array(polygon, dtype=np.float32)
        original = poly_array.copy()

        # Clamp coordinates to valid range
        x_coords = poly_array[:, 0]
        y_coords = poly_array[:, 1]

        # Track what was fixed
        fix_info = {
            "original_min_x": float(np.min(x_coords)),
            "original_max_x": float(np.max(x_coords)),
            "original_min_y": float(np.min(y_coords)),
            "original_max_y": float(np.max(y_coords)),
            "clamped_x": False,
            "clamped_y": False,
            "out_of_tolerance": False,
        }

        # Check if coordinates are out of tolerance
        x_out_of_tolerance = np.any((x_coords < -tolerance) | (x_coords > image_width + tolerance))
        y_out_of_tolerance = np.any((y_coords < -tolerance) | (y_coords > image_height + tolerance))

        if x_out_of_tolerance or y_out_of_tolerance:
            fix_info["out_of_tolerance"] = True

        # Clamp coordinates
        x_clamped = np.clip(x_coords, 0.0, float(image_width))
        y_clamped = np.clip(y_coords, 0.0, float(image_height))

        if not np.allclose(x_coords, x_clamped, atol=1e-6):
            fix_info["clamped_x"] = True
            poly_array[:, 0] = x_clamped

        if not np.allclose(y_coords, y_clamped, atol=1e-6):
            fix_info["clamped_y"] = True
            poly_array[:, 1] = y_clamped

        fixed_polygon = poly_array.tolist()

        return fixed_polygon, fix_info

    def fix_annotations(self, data: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
        """
        Fix out-of-bounds coordinates in annotations.

        BUG-20251116-001: Processes each polygon and clamps coordinates to valid bounds.
        """
        fixed_data = json.loads(json.dumps(data))  # Deep copy
        images = fixed_data.get("images", {})

        self.log(f"Processing {len(images)} images...")

        for filename, image_data in images.items():
            # Get image dimensions
            dims = self.get_image_dimensions(filename)
            if dims is None:
                self.stats["images_skipped_missing"] += 1
                continue

            canonical_width, canonical_height, orientation = dims
            self.stats["images_processed"] += 1

            # Process polygons
            words = image_data.get("words", {})
            if not words:
                continue

            image_fixes = []
            for word_id, word_data in words.items():
                points = word_data.get("points")
                if not points or not isinstance(points, list):
                    continue

                # Convert to list of lists if needed
                if isinstance(points[0], (int, float)):
                    # Flattened format - reshape to (N, 2)
                    if len(points) % 2 != 0:
                        self.stats["polygons_skipped_invalid"] += 1
                        continue
                    polygon = [[points[i], points[i + 1]] for i in range(0, len(points), 2)]
                else:
                    polygon = points

                if len(polygon) < 3:
                    self.stats["polygons_skipped_too_few_points"] += 1
                    continue

                # Check if polygon needs fixing
                poly_array = np.array(polygon, dtype=np.float32)
                x_coords = poly_array[:, 0]
                y_coords = poly_array[:, 1]

                needs_fixing = (
                    np.any(x_coords < 0)
                    or np.any(x_coords > canonical_width)
                    or np.any(y_coords < 0)
                    or np.any(y_coords > canonical_height)
                )

                if needs_fixing:
                    # Clamp coordinates
                    fixed_polygon, fix_info = self.clamp_polygon_coordinates(
                        polygon, canonical_width, canonical_height, self.tolerance
                    )

                    if not dry_run:
                        # Update annotation
                        word_data["points"] = fixed_polygon

                    # Record fix
                    fix_record = {
                        "filename": filename,
                        "word_id": word_id,
                        "fix_info": fix_info,
                    }
                    image_fixes.append(fix_record)
                    self.stats["polygons_fixed"] += 1

                    if fix_info["out_of_tolerance"]:
                        self.stats["polygons_out_of_tolerance"] += 1

            if image_fixes:
                self.fixes_applied.append(
                    {
                        "filename": filename,
                        "num_fixes": len(image_fixes),
                        "fixes": image_fixes,
                    }
                )
                self.stats["images_with_fixes"] += 1

        return fixed_data

    def print_summary(self, dry_run: bool = False) -> None:
        """Print summary of fixes applied."""
        print("\n" + "=" * 80)
        print("📊 POLYGON COORDINATE FIXING SUMMARY")
        print("=" * 80)
        print(f"\n📁 Annotation file: {self.annotation_file}")
        print(f"📁 Image directory: {self.image_dir}")
        print(f"🔧 Tolerance: {self.tolerance} pixels")
        print(f"🔍 Mode: {'DRY RUN' if dry_run else 'FIXING'}")

        print(f"\n📈 Statistics:")
        print(f"  Images processed: {self.stats['images_processed']}")
        print(f"  Images skipped (missing): {self.stats['images_skipped_missing']}")
        print(f"  Images with fixes: {self.stats['images_with_fixes']}")
        print(f"  Polygons fixed: {self.stats['polygons_fixed']}")
        print(f"  Polygons out of tolerance: {self.stats['polygons_out_of_tolerance']}")
        print(f"  Polygons skipped (invalid): {self.stats['polygons_skipped_invalid']}")
        print(f"  Polygons skipped (too few points): {self.stats['polygons_skipped_too_few_points']}")

        if self.fixes_applied:
            print(f"\n🔧 Sample Fixes (first 5):")
            for fix_record in self.fixes_applied[:5]:
                print(f"\n  {fix_record['filename']} ({fix_record['num_fixes']} polygons fixed):")
                for fix in fix_record["fixes"][:2]:
                    info = fix["fix_info"]
                    print(f"    Word {fix['word_id']}:")
                    print(f"      X: [{info['original_min_x']:.1f}, {info['original_max_x']:.1f}] -> clamped: {info['clamped_x']}")
                    print(f"      Y: [{info['original_min_y']:.1f}, {info['original_max_y']:.1f}] -> clamped: {info['clamped_y']}")
                    if info["out_of_tolerance"]:
                        print(f"      ⚠️  Out of tolerance (> {self.tolerance} pixels)")

            if len(self.fixes_applied) > 5:
                print(f"\n    ... and {len(self.fixes_applied) - 5} more images with fixes")

        print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Fix out-of-bounds polygon coordinates in annotation files (BUG-20251116-001)"
    )
    parser.add_argument("--annotation-file", type=Path, required=True, help="Path to annotation JSON file")
    parser.add_argument("--image-dir", type=Path, required=True, help="Directory containing images")
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Path to save fixed annotations (if not provided, uses --in-place or dry run)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite original annotation file (use with --backup)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=3.0,
        help="Tolerance for out-of-bounds coordinates (default: 3.0 pixels)",
    )
    parser.add_argument("--backup", action="store_true", help="Create backup before modifying files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be fixed without making changes")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Validate arguments
    if args.in_place and args.output_file:
        print("❌ Error: Cannot use both --in-place and --output-file", file=sys.stderr)
        sys.exit(1)

    if args.in_place and not args.backup and not args.dry_run:
        response = input("⚠️  WARNING: --in-place will overwrite original file. Create backup? (y/n): ")
        if response.lower() == "y":
            args.backup = True

    # Create fixer
    try:
        fixer = PolygonCoordinateFixer(
            args.annotation_file, args.image_dir, tolerance=args.tolerance, verbose=args.verbose
        )
    except ValueError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Load annotations
    try:
        data = fixer.load_annotations()
    except Exception as e:
        print(f"❌ Error loading annotations: {e}", file=sys.stderr)
        sys.exit(1)

    # Fix annotations
    try:
        fixed_data = fixer.fix_annotations(data, dry_run=args.dry_run)
    except Exception as e:
        print(f"❌ Error fixing annotations: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)

    # Print summary
    fixer.print_summary(dry_run=args.dry_run)

    # Save fixed annotations
    if not args.dry_run:
        output_path = args.output_file if args.output_file else (args.annotation_file if args.in_place else None)

        if output_path:
            # Create backup if requested
            if args.backup:
                backup_path = args.annotation_file.with_suffix(args.annotation_file.suffix + ".backup")
                if not backup_path.exists():
                    shutil.copy2(args.annotation_file, backup_path)
                    print(f"\n💾 Backup created: {backup_path}")

            # Save fixed annotations
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(fixed_data, f, indent=2, ensure_ascii=False)

            print(f"\n✅ Fixed annotations saved to: {output_path}")
            print(f"   Fixed {fixer.stats['polygons_fixed']} polygons in {fixer.stats['images_with_fixes']} images")
        else:
            print("\n💡 Use --output-file or --in-place to save fixed annotations")
    else:
        print("\n💡 This was a dry run. Use --output-file or --in-place to apply fixes")

    # Exit with error code if fixes were needed
    if fixer.stats["polygons_fixed"] > 0:
        sys.exit(0)
    else:
        print("\n✅ No fixes needed! All polygons are within bounds.")
        sys.exit(0)


if __name__ == "__main__":
    main()

