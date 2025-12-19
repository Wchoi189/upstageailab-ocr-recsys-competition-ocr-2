#!/usr/bin/env python3
"""
Border removal methods for document images (STANDALONE EXPERIMENT).

This script implements and tests 3 border removal methods:
1. Canny edge detection + largest contour
2. Morphological operations + connected components
3. Hough line detection + intersection

This is an EXPERIMENT-ONLY implementation. Pipeline integration will be
handled separately in Options A/B.

Usage:
    # Run single method on one image
    python border_removal.py --input 000732.jpg --method canny --output outputs/

    # Run all methods on all border cases
    python border_removal.py --manifest border_cases_manifest.json --all-methods --output outputs/

    # Create comparison visualizations
    python border_removal.py --input 000732.jpg --all-methods --compare --output outputs/
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


class BorderRemover:
    """
    Border removal for scanned documents.

    Standalone implementation for experimentation and validation.
    Not integrated into preprocessing pipeline (yet).
    """

    def __init__(
        self,
        method: str = "canny",
        min_area_ratio: float = 0.75,
        confidence_threshold: float = 0.8,
        # Canny parameters
        canny_low: int = 50,
        canny_high: int = 150,
        # Morphological parameters
        morph_kernel_size: int = 5,
        morph_iterations: int = 2,
        # Hough parameters
        hough_threshold: int = 100,
        hough_min_line_length: int = 100,
        hough_max_line_gap: int = 10,
    ):
        """
        Initialize border remover.

        Args:
            method: Detection method ['canny', 'morph', 'hough']
            min_area_ratio: Minimum crop area vs original (safety check)
            confidence_threshold: Minimum detection confidence to crop
            canny_low: Canny lower threshold
            canny_high: Canny upper threshold
            morph_kernel_size: Morphological kernel size
            morph_iterations: Morphological closing iterations
            hough_threshold: Hough transform threshold
            hough_min_line_length: Minimum line length for Hough
            hough_max_line_gap: Maximum line gap for Hough
        """
        self.method = method
        self.min_area_ratio = min_area_ratio
        self.confidence_threshold = confidence_threshold

        # Method parameters
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.morph_kernel_size = morph_kernel_size
        self.morph_iterations = morph_iterations
        self.hough_threshold = hough_threshold
        self.hough_min_line_length = hough_min_line_length
        self.hough_max_line_gap = hough_max_line_gap

        self.valid_methods = ["canny", "morph", "hough"]
        if method not in self.valid_methods:
            raise ValueError(f"Invalid method: {method}. Choose from {self.valid_methods}")

    def remove_border(self, img: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Remove border from image.

        Args:
            img: Input image (BGR format)

        Returns:
            Tuple of (processed_image, metrics_dict)
        """
        start_time = time.time()

        # Detect border
        crop_box, confidence = self._detect_border(img)

        # Calculate metrics
        x1, y1, x2, y2 = crop_box
        original_area = img.shape[0] * img.shape[1]
        crop_area = (x2 - x1) * (y2 - y1)
        area_ratio = crop_area / original_area

        processing_time = (time.time() - start_time) * 1000  # ms

        # Decide whether to crop
        should_crop = (
            confidence >= self.confidence_threshold
            and area_ratio >= self.min_area_ratio
        )

        if should_crop:
            result = img[y1:y2, x1:x2].copy()
            cropped = True
        else:
            result = img.copy()
            cropped = False

        # Metrics
        metrics = {
            "method": self.method,
            "cropped": cropped,
            "confidence": round(confidence, 3),
            "area_ratio": round(area_ratio, 3),
            "crop_box": [int(x1), int(y1), int(x2), int(y2)],
            "original_shape": list(img.shape),
            "result_shape": list(result.shape),
            "processing_time_ms": round(processing_time, 2),
            "confidence_threshold": self.confidence_threshold,
            "min_area_ratio": self.min_area_ratio,
        }

        return result, metrics

    def _detect_border(self, img: np.ndarray) -> tuple[tuple[int, int, int, int], float]:
        """
        Detect border and return crop box.

        Args:
            img: Input image (BGR)

        Returns:
            Tuple of ((x1, y1, x2, y2), confidence)
        """
        if self.method == "canny":
            return self._detect_canny(img)
        elif self.method == "morph":
            return self._detect_morph(img)
        elif self.method == "hough":
            return self._detect_hough(img)
        else:
            # Fallback: no crop
            h, w = img.shape[:2]
            return (0, 0, w, h), 0.0

    def _detect_canny(self, img: np.ndarray) -> tuple[tuple[int, int, int, int], float]:
        """Canny edge detection + largest contour."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            h, w = img.shape[:2]
            return (0, 0, w, h), 0.0

        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Approximate to polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(approx)

        # Calculate confidence (area ratio)
        img_area = img.shape[0] * img.shape[1]
        contour_area = w * h
        confidence = min(1.0, contour_area / img_area)

        return (x, y, x + w, y + h), confidence

    def _detect_morph(self, img: np.ndarray) -> tuple[tuple[int, int, int, int], float]:
        """Morphological operations + connected components."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Otsu threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological closing
        kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=self.morph_iterations)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)

        if num_labels <= 1:
            h, w = img.shape[:2]
            return (0, 0, w, h), 0.0

        # Find largest component (skip background label 0)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

        x = stats[largest_label, cv2.CC_STAT_LEFT]
        y = stats[largest_label, cv2.CC_STAT_TOP]
        w = stats[largest_label, cv2.CC_STAT_WIDTH]
        h = stats[largest_label, cv2.CC_STAT_HEIGHT]

        # Calculate confidence
        img_area = img.shape[0] * img.shape[1]
        component_area = w * h
        confidence = min(1.0, component_area / img_area)

        return (x, y, x + w, y + h), confidence

    def _detect_hough(self, img: np.ndarray) -> tuple[tuple[int, int, int, int], float]:
        """Hough line detection + intersection."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Hough line transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap,
        )

        if lines is None or len(lines) < 4:
            h, w = img.shape[:2]
            return (0, 0, w, h), 0.0

        # Separate horizontal and vertical lines
        h_lines = []
        v_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

            if angle < 45 or angle > 135:  # Horizontal-ish
                h_lines.append((x1, y1, x2, y2))
            else:  # Vertical-ish
                v_lines.append((x1, y1, x2, y2))

        if len(h_lines) < 2 or len(v_lines) < 2:
            h, w = img.shape[:2]
            return (0, 0, w, h), 0.5  # Medium confidence

        # Find bounding box from lines
        x_coords = [x for line in v_lines for x in [line[0], line[2]]]
        y_coords = [y for line in h_lines for y in [line[1], line[3]]]

        x1, x2 = min(x_coords), max(x_coords)
        y1, y2 = min(y_coords), max(y_coords)

        # Calculate confidence
        img_area = img.shape[0] * img.shape[1]
        detected_area = (x2 - x1) * (y2 - y1)
        confidence = min(1.0, detected_area / img_area)

        return (x1, y1, x2, y2), confidence

    def create_visualization(self, img: np.ndarray, crop_box: tuple[int, int, int, int]) -> np.ndarray:
        """
        Create visualization of detected border.

        Args:
            img: Original image
            crop_box: Detected crop box (x1, y1, x2, y2)

        Returns:
            Visualization image with border box drawn
        """
        vis = img.copy()
        x1, y1, x2, y2 = crop_box

        # Draw detected border box in green
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Add label
        label = f"{self.method.upper()}"
        cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return vis


def process_single_image(
    img_path: Path,
    output_dir: Path,
    method: str,
    save_visualization: bool = False,
) -> dict:
    """Process single image with specified method."""

    print(f"\nProcessing: {img_path.name} (method: {method})")

    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        print("  ERROR: Could not read image")
        return {"error": "Could not read image"}

    # Create remover
    remover = BorderRemover(method=method)

    # Remove border
    result, metrics = remover.remove_border(img)

    # Print metrics
    print(f"  Cropped: {metrics['cropped']}")
    print(f"  Confidence: {metrics['confidence']:.3f}")
    print(f"  Area ratio: {metrics['area_ratio']:.3f}")
    print(f"  Processing time: {metrics['processing_time_ms']:.1f}ms")

    # Save result
    output_path = output_dir / f"{img_path.stem}_{method}_cropped.jpg"
    cv2.imwrite(str(output_path), result)
    print(f"  Saved: {output_path}")

    # Save visualization
    if save_visualization:
        vis = remover.create_visualization(img, metrics['crop_box'])
        vis_path = output_dir / f"{img_path.stem}_{method}_detection.jpg"
        cv2.imwrite(str(vis_path), vis)
        print(f"  Visualization: {vis_path}")

    # Add file paths to metrics
    metrics['input_path'] = str(img_path)
    metrics['output_path'] = str(output_path)
    if save_visualization:
        metrics['visualization_path'] = str(vis_path)

    return metrics


def create_comparison(
    img_path: Path,
    output_dir: Path,
    methods: list[str] = ["canny", "morph", "hough"],
) -> Path:
    """Create side-by-side comparison of all methods."""

    print(f"\nCreating comparison for: {img_path.name}")

    # Read original
    original = cv2.imread(str(img_path))
    if original is None:
        print("  ERROR: Could not read image")
        return None

    # Process with each method
    results = [original]  # Start with original
    labels = ["Original"]

    for method in methods:
        remover = BorderRemover(method=method)
        result, metrics = remover.remove_border(original)
        results.append(result)
        labels.append(f"{method.upper()}\n({metrics['confidence']:.2f})")

    # Resize all to same height
    target_h = min(img.shape[0] for img in results)
    resized = []
    for img in results:
        scale = target_h / img.shape[0]
        resized_img = cv2.resize(img, None, fx=scale, fy=scale)
        resized.append(resized_img)

    # Pad to same width
    max_w = max(img.shape[1] for img in resized)
    padded = []
    for img, label in zip(resized, labels):
        # Pad width
        pad_left = (max_w - img.shape[1]) // 2
        pad_right = max_w - img.shape[1] - pad_left
        padded_img = cv2.copyMakeBorder(
            img, 0, 0, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )

        # Add label
        cv2.putText(
            padded_img, label.split('\n')[0],
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
        )
        if '\n' in label:
            cv2.putText(
                padded_img, label.split('\n')[1],
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )

        padded.append(padded_img)

    # Concatenate horizontally
    comparison = np.hstack(padded)

    # Save
    output_path = output_dir / f"comparison_{img_path.stem}_all_methods.jpg"
    cv2.imwrite(str(output_path), comparison)
    print(f"  Saved: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Border removal experiment (standalone)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--input", type=Path, help="Input image file")
    parser.add_argument("--manifest", type=Path, help="Border cases manifest JSON")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--method",
        type=str,
        default="canny",
        choices=["canny", "morph", "hough"],
        help="Border removal method",
    )
    parser.add_argument("--all-methods", action="store_true", help="Run all 3 methods")
    parser.add_argument("--visualize", action="store_true", help="Save detection visualizations")
    parser.add_argument("--compare", action="store_true", help="Create comparison image")

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Determine methods to run
    methods = ["canny", "morph", "hough"] if args.all_methods else [args.method]

    # Process images
    if args.input:
        # Single image
        for method in methods:
            metrics = process_single_image(
                img_path=args.input,
                output_dir=args.output,
                method=method,
                save_visualization=args.visualize,
            )

        # Create comparison if requested
        if args.compare:
            create_comparison(
                img_path=args.input,
                output_dir=args.output,
                methods=methods,
            )

    elif args.manifest:
        # Process all images in manifest
        with open(args.manifest) as f:
            manifest = json.load(f)

        all_metrics = []

        for case in manifest["cases"]:
            img_path = Path(case["image_path"])

            for method in methods:
                metrics = process_single_image(
                    img_path=img_path,
                    output_dir=args.output,
                    method=method,
                    save_visualization=args.visualize,
                )
                all_metrics.append(metrics)

            # Create comparison if requested
            if args.compare:
                create_comparison(
                    img_path=img_path,
                    output_dir=args.output,
                    methods=methods,
                )

        # Save aggregate metrics
        metrics_file = args.output / "border_removal_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump({
                "methods": methods,
                "total_images": len(manifest["cases"]),
                "results": all_metrics,
            }, f, indent=2)
        print(f"\nMetrics saved to: {metrics_file}")

    else:
        print("Error: Must provide either --input or --manifest")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
