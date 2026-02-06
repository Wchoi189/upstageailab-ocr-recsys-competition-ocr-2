#!/usr/bin/env python3
"""
Background normalization for document images.

Implements three methods to correct background color tint and variation:
1. Gray-world white balance - Global color correction
2. Edge-based background estimation - Mask-based sampling
3. Illumination correction - Morphological background removal

EDS v1.0 Experiment: 20251217_024343_image_enhancements_implementation
Phase: Week 1 Day 2-3
Target: Reduce background variance from 36.5 to <10, color tint from 58.1 to <20

Usage:
    python background_normalization.py --input <dir> --method <method> --output <dir>
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


class BackgroundNormalizer:
    """Document background normalization methods."""

    def __init__(self, method: str = "gray-world"):
        """
        Initialize normalizer.

        Args:
            method: Normalization method [gray-world|edge-based|illumination]
        """
        self.method = method
        self.valid_methods = ["gray-world", "edge-based", "illumination"]

        if method not in self.valid_methods:
            raise ValueError(f"Invalid method: {method}. Choose from {self.valid_methods}")

    def normalize(self, img: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Normalize background of document image.

        Args:
            img: Input image (BGR format)

        Returns:
            Tuple of (normalized_image, metrics_dict)
        """
        start_time = time.time()

        if self.method == "gray-world":
            result = self.normalize_gray_world(img)
        elif self.method == "edge-based":
            result = self.normalize_edge_based(img)
        elif self.method == "illumination":
            result = self.normalize_illumination(img)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        processing_time = (time.time() - start_time) * 1000  # ms

        # Calculate improvement metrics
        metrics = self._calculate_metrics(img, result, processing_time)

        return result, metrics

    def normalize_gray_world(self, img: np.ndarray) -> np.ndarray:
        """
        Gray-world white balance assumption.

        Assumes average color of scene should be neutral gray.
        Scales each channel so its mean equals the global average.

        Args:
            img: Input image (BGR)

        Returns:
            Normalized image (BGR)
        """
        # Calculate channel means
        b_avg, g_avg, r_avg = img.mean(axis=(0, 1))

        # Calculate gray average
        gray_avg = (b_avg + g_avg + r_avg) / 3

        # Avoid division by zero
        if b_avg == 0 or g_avg == 0 or r_avg == 0:
            return img.copy()

        # Calculate scale factors
        scale_factors = np.array([gray_avg / b_avg, gray_avg / g_avg, gray_avg / r_avg])

        # Apply scaling
        result = img.astype(np.float32) * scale_factors
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    def normalize_edge_based(self, img: np.ndarray) -> np.ndarray:
        """
        Background sampling via edge detection.

        Detects foreground text/content using edges, then samples
        background pixels to estimate true background color.

        Args:
            img: Input image (BGR)

        Returns:
            Normalized image (BGR)
        """
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect edges (foreground content)
        edges = cv2.Canny(gray, 50, 150)

        # Dilate edges to ensure we exclude all foreground
        kernel = np.ones((5, 5), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)

        # Background mask (inverse of edges)
        bg_mask = edges_dilated == 0

        # Sample background pixels
        if bg_mask.sum() == 0:
            # Fallback to gray-world if no background detected
            return self.normalize_gray_world(img)

        bg_pixels = img[bg_mask]
        bg_mean = bg_pixels.mean(axis=0)

        # Target is pure white
        target = np.array([255, 255, 255])

        # Avoid division by zero
        bg_mean = np.maximum(bg_mean, 1.0)

        # Calculate scale factors
        scale_factors = target / bg_mean

        # Apply scaling
        result = img.astype(np.float32) * scale_factors
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    def normalize_illumination(self, img: np.ndarray) -> np.ndarray:
        """
        Morphological background estimation and subtraction.

        Uses large morphological opening to estimate background
        illumination field, then removes it.

        Args:
            img: Input image (BGR)

        Returns:
            Normalized image (BGR)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Large elliptical structuring element
        kernel_size = (51, 51)  # Must be odd
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

        # Estimate background via morphological opening
        bg_estimate = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

        # Normalize background to mid-gray (128)
        # Formula: corrected = original - (background - 128)
        corrected_gray = cv2.subtract(gray, cv2.subtract(bg_estimate, 128))

        # Apply same correction to color channels
        # Calculate per-pixel correction factors
        correction = corrected_gray.astype(np.float32) / np.maximum(gray.astype(np.float32), 1.0)

        # Apply to each channel
        result = img.astype(np.float32) * correction[:, :, np.newaxis]
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    def _calculate_metrics(self, original: np.ndarray, normalized: np.ndarray, processing_time: float) -> dict:
        """
        Calculate improvement metrics.

        Args:
            original: Original image
            normalized: Normalized image
            processing_time: Processing time in milliseconds

        Returns:
            Dictionary of metrics
        """
        # Original background metrics
        orig_bg_metrics = self._analyze_background(original)

        # Normalized background metrics
        norm_bg_metrics = self._analyze_background(normalized)

        # Calculate improvements
        variance_reduction = orig_bg_metrics["color_variance"] - norm_bg_metrics["color_variance"]
        tint_reduction = orig_bg_metrics["color_tint_score"] - norm_bg_metrics["color_tint_score"]

        return {
            "method": self.method,
            "processing_time_ms": round(processing_time, 2),
            "original": orig_bg_metrics,
            "normalized": norm_bg_metrics,
            "improvements": {
                "variance_reduction": round(variance_reduction, 2),
                "tint_reduction": round(tint_reduction, 2),
                "variance_reduction_pct": round(100 * variance_reduction / max(orig_bg_metrics["color_variance"], 0.001), 1),
                "tint_reduction_pct": round(100 * tint_reduction / max(orig_bg_metrics["color_tint_score"], 0.001), 1),
            },
        }

    def _analyze_background(self, img: np.ndarray) -> dict:
        """
        Analyze background characteristics.

        Args:
            img: Input image (BGR)

        Returns:
            Dictionary with background metrics
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        bg_mask = cv2.dilate(edges, np.ones((5, 5)), iterations=2) == 0

        if bg_mask.sum() > 0:
            bg_pixels = img[bg_mask]
            bg_mean = bg_pixels.mean(axis=0)
            bg_std = bg_pixels.std(axis=0)
            color_variance = float(np.mean(bg_std))

            # Color tint score (deviation from neutral)
            b, g, r = bg_mean
            color_tint_score = float(max(abs(b - g), abs(g - r), abs(r - b)))
        else:
            bg_mean = [0, 0, 0]
            color_variance = 0.0
            color_tint_score = 0.0

        return {
            "mean_bgr": [float(x) for x in bg_mean],
            "color_variance": round(color_variance, 2),
            "color_tint_score": round(color_tint_score, 2),
        }


def process_directory(input_dir: Path, output_dir: Path, method: str, save_comparison: bool = False) -> dict:
    """
    Process all images in directory.

    Args:
        input_dir: Input directory with images
        output_dir: Output directory for processed images
        method: Normalization method
        save_comparison: If True, save side-by-side comparison

    Returns:
        Dictionary with aggregate results
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalizer = BackgroundNormalizer(method=method)

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

        # Normalize
        normalized, metrics = normalizer.normalize(img)

        # Save normalized image
        output_path = output_dir / img_path.name
        cv2.imwrite(str(output_path), normalized)

        # Save comparison if requested
        if save_comparison:
            comparison = np.hstack([img, normalized])
            comp_path = output_dir / f"comparison_{img_path.name}"
            cv2.imwrite(str(comp_path), comparison)

        results.append({"file": img_path.name, "metrics": metrics})

        print(f"✓ ({metrics['processing_time_ms']:.1f}ms)")
        print(
            f"  Variance: {metrics['original']['color_variance']:.1f} → "
            f"{metrics['normalized']['color_variance']:.1f} "
            f"({metrics['improvements']['variance_reduction']:+.1f})"
        )
        print(
            f"  Tint: {metrics['original']['color_tint_score']:.1f} → "
            f"{metrics['normalized']['color_tint_score']:.1f} "
            f"({metrics['improvements']['tint_reduction']:+.1f})"
        )

    # Calculate aggregate statistics
    if results:
        avg_processing_time = np.mean([r["metrics"]["processing_time_ms"] for r in results])
        avg_variance_reduction = np.mean([r["metrics"]["improvements"]["variance_reduction"] for r in results])
        avg_tint_reduction = np.mean([r["metrics"]["improvements"]["tint_reduction"] for r in results])

        aggregate = {
            "method": method,
            "total_images": len(results),
            "avg_processing_time_ms": round(avg_processing_time, 2),
            "avg_variance_reduction": round(avg_variance_reduction, 2),
            "avg_tint_reduction": round(avg_tint_reduction, 2),
            "results": results,
        }

        print("\n" + "=" * 60)
        print("AGGREGATE RESULTS")
        print("=" * 60)
        print(f"Method: {method}")
        print(f"Images processed: {len(results)}")
        print(f"Avg processing time: {avg_processing_time:.2f} ms")
        print(f"Avg variance reduction: {avg_variance_reduction:.2f}")
        print(f"Avg tint reduction: {avg_tint_reduction:.2f}")
        print()

        return aggregate
    else:
        return {}


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Background normalization for document images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--input", type=Path, required=True, help="Input directory or image file")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--method",
        type=str,
        default="gray-world",
        choices=["gray-world", "edge-based", "illumination"],
        help="Normalization method (default: gray-world)",
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
            results_file = args.output / f"{timestamp}_normalization-results_{args.method}.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {results_file}")
    else:
        # Single image
        normalizer = BackgroundNormalizer(method=args.method)
        img = cv2.imread(str(args.input))

        if img is None:
            print(f"Error: Could not read image {args.input}")
            return 1

        normalized, metrics = normalizer.normalize(img)

        args.output.mkdir(parents=True, exist_ok=True)
        output_path = args.output / args.input.name
        cv2.imwrite(str(output_path), normalized)

        print(f"\nProcessed: {args.input.name}")
        print(f"Method: {args.method}")
        print(f"Processing time: {metrics['processing_time_ms']:.2f} ms")
        print(f"Variance reduction: {metrics['improvements']['variance_reduction']:.2f}")
        print(f"Tint reduction: {metrics['improvements']['tint_reduction']:.2f}")
        print(f"Saved to: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
