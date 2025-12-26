#!/usr/bin/env python3
"""
Text deskewing for document images.

Implements two methods to detect and correct text rotation:
1. Projection profile - Variance maximization approach
2. Hough lines - Dominant line angle detection

EDS v1.0 Experiment: 20251217_024343_image_enhancements_implementation
Phase: Week 2 Day 1-2
Target: Reduce skew from 15.0° to <2°, handle extreme angles (-83°)

Usage:
    python text_deskewing.py --input <dir> --method <method> --output <dir>
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


class TextDeskewer:
    """Document text deskewing methods."""

    def __init__(self, method: str = "projection", angle_range: tuple[float, float] = (-45, 45)):
        """
        Initialize deskewer.

        Args:
            method: Deskewing method [projection|hough|combined]
            angle_range: Angle search range in degrees (min, max)
        """
        self.method = method
        self.angle_range = angle_range
        self.valid_methods = ["projection", "hough", "combined"]

        if method not in self.valid_methods:
            raise ValueError(f"Invalid method: {method}. Choose from {self.valid_methods}")

    def deskew(self, img: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Deskew document image.

        Args:
            img: Input image (BGR format)

        Returns:
            Tuple of (deskewed_image, metrics_dict)
        """
        start_time = time.time()

        # Detect skew angle
        if self.method == "projection":
            angle = self.detect_angle_projection(img)
        elif self.method == "hough":
            angle = self.detect_angle_hough(img)
        elif self.method == "combined":
            angle_proj = self.detect_angle_projection(img)
            angle_hough = self.detect_angle_hough(img)
            # Average if both methods agree within 5 degrees
            if abs(angle_proj - angle_hough) < 5:
                angle = (angle_proj + angle_hough) / 2
            else:
                # Use projection as more reliable
                angle = angle_proj
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Rotate image
        result, rotation_matrix = self.rotate_image(img, angle)

        processing_time = (time.time() - start_time) * 1000  # ms

        # Calculate metrics
        metrics = {
            "method": self.method,
            "detected_angle_degrees": round(angle, 2),
            "processing_time_ms": round(processing_time, 2),
            "rotation_matrix": rotation_matrix.tolist() if rotation_matrix is not None else None,
        }

        return result, metrics

    def detect_angle_projection(self, img: np.ndarray, step: float = 0.5) -> float:
        """
        Detect skew angle using projection profile variance maximization.

        Hypothesis: Correctly aligned text produces maximum variance in
        horizontal projection (rows alternate between text and whitespace).

        Args:
            img: Input image (BGR)
            step: Angle step for search in degrees

        Returns:
            Detected angle in degrees
        """
        # Convert to grayscale and binarize
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Search for angle with maximum projection variance
        best_angle = 0
        max_variance = 0

        angles = np.arange(self.angle_range[0], self.angle_range[1], step)

        for angle in angles:
            # Rotate binary image
            rotated, _ = self.rotate_image(binary, angle)

            # Compute horizontal projection (sum along rows)
            projection = rotated.sum(axis=1)

            # Calculate variance of projection
            variance = np.var(projection)

            if variance > max_variance:
                max_variance = variance
                best_angle = angle

        return best_angle

    def detect_angle_hough(self, img: np.ndarray) -> float:
        """
        Detect skew angle using Hough line transform.

        Finds dominant line angles in the image and computes median.

        Args:
            img: Input image (BGR)

        Returns:
            Detected angle in degrees
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Hough line transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10,
        )

        if lines is None or len(lines) == 0:
            return 0.0

        # Calculate angles for all detected lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Skip near-vertical lines
            if abs(x2 - x1) < 1:
                continue

            # Calculate angle
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            # Normalize to -90 to 90 range
            if angle > 90:
                angle = angle - 180
            elif angle < -90:
                angle = angle + 180

            # Filter out extreme angles (likely noise)
            if abs(angle) < 45:
                angles.append(angle)

        if not angles:
            return 0.0

        # Return median angle (robust to outliers)
        return float(np.median(angles))

    def rotate_image(self, img: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Rotate image by specified angle.

        Args:
            img: Input image
            angle: Rotation angle in degrees (positive = counter-clockwise)

        Returns:
            Tuple of (rotated_image, rotation_matrix)
        """
        if abs(angle) < 0.1:
            # No significant rotation needed
            return img.copy(), None

        h, w = img.shape[:2]
        center = (w / 2, h / 2)

        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new image dimensions to fit entire rotated image
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])

        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust rotation matrix to account for translation
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]

        # Perform rotation with white background
        rotated = cv2.warpAffine(
            img,
            rotation_matrix,
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        return rotated, rotation_matrix


def process_directory(input_dir: Path, output_dir: Path, method: str, save_comparison: bool = False) -> dict:
    """
    Process all images in directory.

    Args:
        input_dir: Input directory with images
        output_dir: Output directory for processed images
        method: Deskewing method
        save_comparison: If True, save side-by-side comparison

    Returns:
        Dictionary with aggregate results
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    deskewer = TextDeskewer(method=method)

    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    if not image_files:
        print(f"No images found in {input_dir}")
        return {}

    print(f"\nProcessing {len(image_files)} images with method: {method}")
    print("=" * 60)

    results = []

    for img_path in sorted(image_files):
        print(f"Processing: {img_path.name}...", end=" ")

        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            print("ERROR: Failed to read")
            continue

        # Deskew
        deskewed, metrics = deskewer.deskew(img)

        # Save deskewed image
        output_path = output_dir / img_path.name
        cv2.imwrite(str(output_path), deskewed)

        # Save comparison if requested
        if save_comparison:
            # Resize images to same height for comparison
            h1, h2 = img.shape[0], deskewed.shape[0]
            if h1 != h2:
                target_h = min(h1, h2)
                scale1 = target_h / h1
                scale2 = target_h / h2
                img_resized = cv2.resize(img, None, fx=scale1, fy=scale1)
                deskewed_resized = cv2.resize(deskewed, None, fx=scale2, fy=scale2)
            else:
                img_resized = img
                deskewed_resized = deskewed

            # Pad to same width
            w1, w2 = img_resized.shape[1], deskewed_resized.shape[1]
            max_w = max(w1, w2)
            if w1 < max_w:
                pad = (max_w - w1) // 2
                img_resized = cv2.copyMakeBorder(img_resized, 0, 0, pad, max_w - w1 - pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            if w2 < max_w:
                pad = (max_w - w2) // 2
                deskewed_resized = cv2.copyMakeBorder(
                    deskewed_resized,
                    0,
                    0,
                    pad,
                    max_w - w2 - pad,
                    cv2.BORDER_CONSTANT,
                    value=(255, 255, 255),
                )

            comparison = np.hstack([img_resized, deskewed_resized])
            comp_path = output_dir / f"comparison_{img_path.name}"
            cv2.imwrite(str(comp_path), comparison)

        results.append({"file": img_path.name, "metrics": metrics})

        angle_str = f"{metrics['detected_angle_degrees']:+.2f}°"
        print(f"✓ ({metrics['processing_time_ms']:.1f}ms, angle: {angle_str})")

    # Calculate aggregate statistics
    if results:
        angles = [abs(r["metrics"]["detected_angle_degrees"]) for r in results]
        avg_angle = np.mean(angles)
        max_angle = np.max(angles)
        avg_processing_time = np.mean([r["metrics"]["processing_time_ms"] for r in results])

        aggregate = {
            "method": method,
            "total_images": len(results),
            "avg_abs_angle": round(avg_angle, 2),
            "max_abs_angle": round(max_angle, 2),
            "avg_processing_time_ms": round(avg_processing_time, 2),
            "results": results,
        }

        print("\n" + "=" * 60)
        print("AGGREGATE RESULTS")
        print("=" * 60)
        print(f"Method: {method}")
        print(f"Images processed: {len(results)}")
        print(f"Avg absolute angle: {avg_angle:.2f}°")
        print(f"Max absolute angle: {max_angle:.2f}°")
        print(f"Avg processing time: {avg_processing_time:.2f} ms")
        print()

        return aggregate
    else:
        return {}


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Text deskewing for document images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--input", type=Path, required=True, help="Input directory or image file")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--method",
        type=str,
        default="projection",
        choices=["projection", "hough", "combined"],
        help="Deskewing method (default: projection)",
    )
    parser.add_argument(
        "--save-comparison",
        action="store_true",
        help="Save side-by-side before/after comparison",
    )

    args = parser.parse_args()

    if args.input.is_dir():
        results = process_directory(args.input, args.output, args.method, args.save_comparison)

        # Save results JSON
        if results:
            import json
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            results_file = args.output / f"{timestamp}_deskewing-results_{args.method}.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {results_file}")
    else:
        # Single image
        deskewer = TextDeskewer(method=args.method)
        img = cv2.imread(str(args.input))

        if img is None:
            print(f"Error: Could not read image {args.input}")
            return 1

        deskewed, metrics = deskewer.deskew(img)

        args.output.mkdir(parents=True, exist_ok=True)
        output_path = args.output / args.input.name
        cv2.imwrite(str(output_path), deskewed)

        print(f"\nProcessed: {args.input.name}")
        print(f"Method: {args.method}")
        print(f"Detected angle: {metrics['detected_angle_degrees']:+.2f}°")
        print(f"Processing time: {metrics['processing_time_ms']:.2f} ms")
        print(f"Saved to: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
