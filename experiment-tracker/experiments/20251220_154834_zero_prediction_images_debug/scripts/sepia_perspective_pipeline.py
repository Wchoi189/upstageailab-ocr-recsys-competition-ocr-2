#!/usr/bin/env python3
"""
Sepia + Perspective Correction Pipeline.

Combines perspective correction with sepia enhancement for optimal OCR results.
Tests the hypothesis that sepia + perspective correction is the best combination
without advanced Office Lens-style enhancements.

EDS v1.0 Experiment: 20251217_024343_image_enhancements_implementation
Phase: Week 2 Day 5
Target: Validate sepia as superior enhancement for perspective-corrected images

Pipeline stages:
1. Perspective correction (document boundary detection + transformation)
2. Sepia enhancement (multiple methods)
3. Optional: Deskewing (based on experiment findings)

Reference: drp.en_ko.in_house.selectstar_000732 (problematic baseline)
           drp.en_ko.in_house.selectstar_000712_sepia.jpg (target quality)

Usage:
    # Process single image with all sepia methods
    python sepia_perspective_pipeline.py --input <image> --output <dir>

    # Process with specific sepia method
    python sepia_perspective_pipeline.py --input <image> --output <dir> --sepia-method warm

    # Process directory
    python sepia_perspective_pipeline.py --input <dir> --output <dir> --sepia-method contrast

    # Include deskewing (if needed)
    python sepia_perspective_pipeline.py --input <image> --output <dir> --deskew
"""

import argparse
import json

# Import from existing scripts
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).parent))
from sepia_enhancement import SepiaEnhancer


class SepiaPerspectivePipeline:
    """Combined perspective correction and sepia enhancement pipeline."""

    def __init__(
        self,
        sepia_method: str = "warm",
        enable_deskewing: bool = False,
        perspective_only: bool = False,
    ):
        """
        Initialize pipeline.

        Args:
            sepia_method: Sepia method [classic|adaptive|warm|contrast|all]
            enable_deskewing: Apply text deskewing after perspective correction
            perspective_only: Only do perspective correction, skip sepia
        """
        self.sepia_method = sepia_method
        self.enable_deskewing = enable_deskewing
        self.perspective_only = perspective_only

    def process(self, img: np.ndarray) -> dict[str, tuple[np.ndarray, dict]]:
        """
        Process image through full pipeline.

        Args:
            img: Input image (BGR)

        Returns:
            Dictionary of {stage_name: (image, metrics)}
        """
        results = {}
        start_time = time.time()

        # Stage 1: Perspective correction
        perspective_img, perspective_metrics = self._perspective_correction(img)
        results["perspective_corrected"] = (perspective_img, perspective_metrics)

        if self.perspective_only:
            return results

        # Stage 2: Deskewing (optional)
        if self.enable_deskewing:
            deskewed_img, deskew_metrics = self._deskew(perspective_img)
            results["deskewed"] = (deskewed_img, deskew_metrics)
            base_img = deskewed_img
        else:
            base_img = perspective_img

        # Stage 3: Sepia enhancement
        enhancer = SepiaEnhancer(method=self.sepia_method)
        sepia_results = enhancer.enhance(base_img)

        for method_name, (sepia_img, sepia_metrics) in sepia_results.items():
            stage_name = f"sepia_{method_name}"
            results[stage_name] = (sepia_img, sepia_metrics)

        # Add total pipeline time
        total_time = (time.time() - start_time) * 1000
        results["_pipeline_total_time_ms"] = total_time

        return results

    def _perspective_correction(self, img: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Apply perspective correction to document image.

        Uses contour detection to find document boundaries and applies
        perspective transformation.

        Args:
            img: Input image (BGR)

        Returns:
            Tuple of (corrected_image, metrics)
        """
        start_time = time.time()

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # No contours found, return original
            processing_time = (time.time() - start_time) * 1000
            return img.copy(), {
                "processing_time_ms": round(processing_time, 2),
                "status": "no_contours_found",
                "correction_applied": False,
            }

        # Find largest contour (assumed to be document)
        largest_contour = max(contours, key=cv2.contourArea)

        # Approximate contour to quadrilateral
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        if len(approx) != 4:
            # Not a quadrilateral, return original
            processing_time = (time.time() - start_time) * 1000
            return img.copy(), {
                "processing_time_ms": round(processing_time, 2),
                "status": "not_quadrilateral",
                "correction_applied": False,
                "contour_points": len(approx),
            }

        # Order points: [top-left, top-right, bottom-right, bottom-left]
        pts = approx.reshape(4, 2)
        rect = self._order_points(pts)

        # Calculate dimensions for output
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # Destination points for perspective transform
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        # Compute perspective transform matrix
        M = cv2.getPerspectiveTransform(rect.astype("float32"), dst)

        # Apply perspective transformation
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

        processing_time = (time.time() - start_time) * 1000

        return warped, {
            "processing_time_ms": round(processing_time, 2),
            "status": "success",
            "correction_applied": True,
            "output_size": f"{maxWidth}x{maxHeight}",
        }

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Order points in clockwise order: [TL, TR, BR, BL].

        Args:
            pts: Array of 4 points

        Returns:
            Ordered points array
        """
        rect = np.zeros((4, 2), dtype="float32")

        # Sum: top-left (smallest), bottom-right (largest)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # Diff: top-right (smallest), bottom-left (largest)
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def _deskew(self, img: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Apply text deskewing using Hough lines method.

        Args:
            img: Input image (BGR)

        Returns:
            Tuple of (deskewed_image, metrics)
        """
        start_time = time.time()

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

        if lines is None:
            processing_time = (time.time() - start_time) * 1000
            return img.copy(), {
                "processing_time_ms": round(processing_time, 2),
                "status": "no_lines_detected",
                "angle": 0.0,
                "deskewing_applied": False,
            }

        # Calculate dominant angle
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            # Filter angles near horizontal
            if -45 < angle < 45:
                angles.append(angle)

        if not angles:
            processing_time = (time.time() - start_time) * 1000
            return img.copy(), {
                "processing_time_ms": round(processing_time, 2),
                "status": "no_valid_angles",
                "angle": 0.0,
                "deskewing_applied": False,
            }

        # Use median angle
        median_angle = np.median(angles)

        # Rotate image
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        deskewed = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        processing_time = (time.time() - start_time) * 1000

        return deskewed, {
            "processing_time_ms": round(processing_time, 2),
            "status": "success",
            "angle": round(median_angle, 2),
            "deskewing_applied": True,
        }


def process_single_image(
    input_path: Path,
    sepia_method: str,
    output_dir: Path,
    enable_deskewing: bool,
    perspective_only: bool,
    verbose: bool = True,
) -> dict[str, dict]:
    """
    Process single image through pipeline.

    Args:
        input_path: Input image path
        sepia_method: Sepia method to apply
        output_dir: Output directory
        enable_deskewing: Enable deskewing stage
        perspective_only: Only perspective correction
        verbose: Print progress

    Returns:
        Dictionary of metrics
    """
    # Load image
    img = cv2.imread(str(input_path))
    if img is None:
        raise ValueError(f"Failed to load image: {input_path}")

    if verbose:
        print(f"\nProcessing: {input_path.name}")
        print(f"  Input size: {img.shape[1]}x{img.shape[0]}")
        print(f"  Sepia method: {sepia_method}")
        print(f"  Deskewing: {'enabled' if enable_deskewing else 'disabled'}")

    # Process through pipeline
    pipeline = SepiaPerspectivePipeline(
        sepia_method=sepia_method,
        enable_deskewing=enable_deskewing,
        perspective_only=perspective_only,
    )
    results = pipeline.process(img)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    all_metrics = {}

    for stage_name, result_data in results.items():
        if stage_name.startswith("_"):
            continue

        stage_img, metrics = result_data

        # Save image
        output_path = output_dir / f"{stem}_{stage_name}.jpg"
        cv2.imwrite(str(output_path), stage_img)

        if verbose:
            print(f"\n  Stage: {stage_name}")
            print(f"    Output: {output_path.name}")
            if "processing_time_ms" in metrics:
                print(f"    Processing time: {metrics['processing_time_ms']:.2f}ms")

        all_metrics[stage_name] = metrics

    if verbose and "_pipeline_total_time_ms" in results:
        print(f"\n  Total pipeline time: {results['_pipeline_total_time_ms']:.2f}ms")

    return all_metrics


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Sepia + perspective correction pipeline for document OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input image file or directory",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory",
    )

    parser.add_argument(
        "--sepia-method",
        type=str,
        default="warm",
        choices=["classic", "adaptive", "warm", "clahe", "linear_contrast", "all"],
        help="Sepia enhancement method (default: warm)",
    )

    parser.add_argument(
        "--deskew",
        action="store_true",
        help="Enable text deskewing after perspective correction",
    )

    parser.add_argument(
        "--perspective-only",
        action="store_true",
        help="Only apply perspective correction, skip sepia",
    )

    parser.add_argument(
        "--save-metrics",
        action="store_true",
        help="Save metrics to JSON file",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Process input
    if args.input.is_file():
        metrics = process_single_image(
            args.input,
            args.sepia_method,
            args.output,
            args.deskew,
            args.perspective_only,
            verbose=not args.quiet,
        )

        if args.save_metrics:
            metrics_path = args.output / f"{args.input.stem}_pipeline_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            if not args.quiet:
                print(f"\n✓ Saved metrics: {metrics_path}")

    elif args.input.is_dir():
        # Process directory
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        image_files = [
            f for f in args.input.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        if not image_files:
            raise ValueError(f"No images found in {args.input}")

        if not args.quiet:
            print(f"\nProcessing {len(image_files)} images...")

        all_metrics = {}
        for img_file in sorted(image_files):
            metrics = process_single_image(
                img_file,
                args.sepia_method,
                args.output,
                args.deskew,
                args.perspective_only,
                verbose=not args.quiet,
            )
            all_metrics[img_file.name] = metrics

        if args.save_metrics:
            metrics_path = args.output / "pipeline_metrics_all.json"
            with open(metrics_path, 'w') as f:
                json.dump(all_metrics, f, indent=2)
            if not args.quiet:
                print(f"\n✓ Saved combined metrics: {metrics_path}")

    else:
        raise ValueError(f"Input path does not exist: {args.input}")

    if not args.quiet:
        print("\n✓ Pipeline processing complete!")


if __name__ == "__main__":
    main()
