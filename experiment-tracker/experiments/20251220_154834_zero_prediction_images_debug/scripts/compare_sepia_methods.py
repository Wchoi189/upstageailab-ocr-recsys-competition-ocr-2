#!/usr/bin/env python3
"""
Sepia comparison script for isolated testing.

Compares sepia enhancement methods against:
- Raw image
- Gray-scale conversion
- Gray-world normalization
- Sepia methods (classic, adaptive, warm, contrast)

Generates side-by-side comparison images and detailed metrics.

EDS v1.0 Experiment: 20251217_024343_image_enhancements_implementation
Phase: Week 2 Day 4-5
Target: Validate sepia superiority over gray-scale/normalization

Reference: drp.en_ko.in_house.selectstar_000732 (problematic)
           drp.en_ko.in_house.selectstar_000712_sepia.jpg (reference)

Usage:
    python compare_sepia_methods.py --input <image> --output <dir>
    python compare_sepia_methods.py --input <image> --output <dir> --save-metrics
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
from background_normalization import BackgroundNormalizer
from sepia_enhancement import SepiaEnhancer


class ImageEnhancementComparator:
    """Compare different image enhancement methods."""

    def __init__(self):
        """Initialize comparator with all enhancement methods."""
        self.methods = {
            "raw": self._raw,
            "grayscale": self._grayscale,
            "gray_world_norm": self._gray_world_normalization,
            "sepia_classic": self._sepia_classic,
            "sepia_adaptive": self._sepia_adaptive,
            "sepia_warm": self._sepia_warm,
            "sepia_clahe": self._sepia_clahe,
            "sepia_linear_contrast": self._sepia_linear_contrast,
        }

    def compare_all(self, img: np.ndarray) -> dict[str, tuple[np.ndarray, dict]]:
        """
        Apply all enhancement methods and gather metrics.

        Args:
            img: Input image (BGR)

        Returns:
            Dictionary of {method_name: (enhanced_image, metrics_dict)}
        """
        results = {}

        for method_name, method_func in self.methods.items():
            start_time = time.time()
            enhanced = method_func(img.copy())
            processing_time = (time.time() - start_time) * 1000

            metrics = self._calculate_metrics(img, enhanced, processing_time)
            results[method_name] = (enhanced, metrics)

        return results

    def _raw(self, img: np.ndarray) -> np.ndarray:
        """Return raw image unchanged."""
        return img

    def _grayscale(self, img: np.ndarray) -> np.ndarray:
        """Convert to grayscale and back to BGR."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _gray_world_normalization(self, img: np.ndarray) -> np.ndarray:
        """Apply gray-world background normalization."""
        normalizer = BackgroundNormalizer(method="gray-world")
        result, _ = normalizer.normalize(img)
        return result

    def _sepia_classic(self, img: np.ndarray) -> np.ndarray:
        """Apply classic sepia."""
        enhancer = SepiaEnhancer(method="classic")
        results = enhancer.enhance(img)
        return results["classic"][0]

    def _sepia_adaptive(self, img: np.ndarray) -> np.ndarray:
        """Apply adaptive sepia."""
        enhancer = SepiaEnhancer(method="adaptive")
        results = enhancer.enhance(img)
        return results["adaptive"][0]

    def _sepia_warm(self, img: np.ndarray) -> np.ndarray:
        """Apply warm sepia."""
        enhancer = SepiaEnhancer(method="warm")
        results = enhancer.enhance(img)
        return results["warm"][0]

    def _sepia_clahe(self, img: np.ndarray) -> np.ndarray:
        """Apply sepia with CLAHE enhancement."""
        enhancer = SepiaEnhancer(method="clahe")
        results = enhancer.enhance(img)
        return results["clahe"][0]

    def _sepia_linear_contrast(self, img: np.ndarray) -> np.ndarray:
        """Apply sepia with linear contrast enhancement."""
        enhancer = SepiaEnhancer(method="linear_contrast")
        results = enhancer.enhance(img)
        return results["linear_contrast"][0]

    def _calculate_metrics(
        self, original: np.ndarray, enhanced: np.ndarray, processing_time: float
    ) -> dict:
        """
        Calculate comprehensive enhancement metrics.

        Args:
            original: Original image
            enhanced: Enhanced image
            processing_time: Processing time in ms

        Returns:
            Dictionary of metrics
        """
        # Color statistics
        orig_mean = original.mean(axis=(0, 1))
        enh_mean = enhanced.mean(axis=(0, 1))

        # Color tint score (deviation from neutral gray)
        orig_tint = np.std(orig_mean)
        enh_tint = np.std(enh_mean)

        # Background variance (spatial variation)
        orig_var = original.std()
        enh_var = enhanced.std()

        # Contrast
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        orig_contrast = orig_gray.std()
        enh_contrast = enh_gray.std()

        # Brightness
        orig_brightness = orig_gray.mean()
        enh_brightness = enh_gray.mean()

        # Edge strength (proxy for text clarity)
        orig_edges = cv2.Laplacian(orig_gray, cv2.CV_64F).var()
        enh_edges = cv2.Laplacian(enh_gray, cv2.CV_64F).var()

        return {
            "processing_time_ms": round(processing_time, 2),
            "color_tint": round(enh_tint, 2),
            "tint_change": round(enh_tint - orig_tint, 2),
            "tint_reduction_pct": round((orig_tint - enh_tint) / orig_tint * 100, 1) if orig_tint > 0 else 0,
            "variance": round(enh_var, 2),
            "variance_change": round(enh_var - orig_var, 2),
            "contrast": round(enh_contrast, 2),
            "contrast_change": round(enh_contrast - orig_contrast, 2),
            "contrast_improvement_pct": round((enh_contrast - orig_contrast) / orig_contrast * 100, 1) if orig_contrast > 0 else 0,
            "brightness": round(enh_brightness, 2),
            "brightness_change": round(enh_brightness - orig_brightness, 2),
            "edge_strength": round(enh_edges, 2),
            "edge_change": round(enh_edges - orig_edges, 2),
            "edge_improvement_pct": round((enh_edges - orig_edges) / orig_edges * 100, 1) if orig_edges > 0 else 0,
        }


def create_comparison_grid(
    results: dict[str, tuple[np.ndarray, dict]],
    title: str = "Image Enhancement Comparison"
) -> np.ndarray:
    """
    Create a grid visualization of all enhancement methods.

    Args:
        results: Dictionary of {method_name: (image, metrics)}
        title: Title for the comparison grid

    Returns:
        Comparison grid image
    """
    # Extract images and sort by method order
    method_order = [
        "raw",
        "grayscale",
        "gray_world_norm",
        "sepia_classic",
        "sepia_adaptive",
        "sepia_warm",
        "sepia_clahe",
        "sepia_linear_contrast",
    ]

    images = []
    labels = []

    for method in method_order:
        if method in results:
            img, metrics = results[method]
            images.append(img)

            # Create label with key metrics
            label = f"{method.replace('_', ' ').title()}\n"
            label += f"Tint: {metrics['color_tint']:.1f} ({metrics['tint_change']:+.1f})\n"
            label += f"Contrast: {metrics['contrast']:.1f} ({metrics['contrast_change']:+.1f})\n"
            label += f"Edge: {metrics['edge_improvement_pct']:+.1f}%\n"
            label += f"Time: {metrics['processing_time_ms']:.1f}ms"
            labels.append(label)

    # Determine grid size (3 columns)
    n_images = len(images)
    n_cols = 3
    n_rows = (n_images + n_cols - 1) // n_cols

    # Resize images to consistent size
    target_height = 400
    target_width = 300
    resized = []

    for img in images:
        h, w = img.shape[:2]
        scale = min(target_width / w, target_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_img = cv2.resize(img, (new_w, new_h))

        # Pad to target size
        pad_top = (target_height - new_h) // 2
        pad_bottom = target_height - new_h - pad_top
        pad_left = (target_width - new_w) // 2
        pad_right = target_width - new_w - pad_left

        padded = cv2.copyMakeBorder(
            resized_img, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        resized.append(padded)

    # Create grid
    label_height = 120
    cell_height = target_height + label_height
    cell_width = target_width

    grid = np.ones((n_rows * cell_height + 50, n_cols * cell_width, 3), dtype=np.uint8) * 255

    # Add title
    cv2.putText(
        grid, title, (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2
    )

    # Place images in grid
    for idx, (img, label) in enumerate(zip(resized, labels)):
        row = idx // n_cols
        col = idx % n_cols

        y_offset = 50 + row * cell_height
        x_offset = col * cell_width

        # Place image
        grid[y_offset:y_offset+target_height, x_offset:x_offset+target_width] = img

        # Add label below image
        label_lines = label.split('\n')
        for i, line in enumerate(label_lines):
            cv2.putText(
                grid, line,
                (x_offset + 5, y_offset + target_height + 20 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1
            )

    return grid


def print_comparison_table(results: dict[str, tuple[np.ndarray, dict]]):
    """
    Print formatted comparison table of all methods.

    Args:
        results: Dictionary of {method_name: (image, metrics)}
    """
    print("\n" + "=" * 120)
    print("IMAGE ENHANCEMENT COMPARISON")
    print("=" * 120)
    print(f"{'Method':<20} {'Tint':<10} {'Change':<10} {'Contrast':<10} {'Change':<10} {'Edge Imp%':<12} {'Time (ms)':<10}")
    print("-" * 120)

    method_order = [
        "raw",
        "grayscale",
        "gray_world_norm",
        "sepia_classic",
        "sepia_adaptive",
        "sepia_warm",
        "sepia_clahe",
        "sepia_linear_contrast",
    ]

    for method in method_order:
        if method in results:
            _, metrics = results[method]
            print(
                f"{method:<20} "
                f"{metrics['color_tint']:>8.1f}  "
                f"{metrics['tint_change']:>+8.1f}  "
                f"{metrics['contrast']:>8.1f}  "
                f"{metrics['contrast_change']:>+8.1f}  "
                f"{metrics['edge_improvement_pct']:>+10.1f}  "
                f"{metrics['processing_time_ms']:>8.1f}"
            )

    print("=" * 120)


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Compare sepia enhancement methods against alternatives",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input image file to compare",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for comparison results",
    )

    parser.add_argument(
        "--save-metrics",
        action="store_true",
        help="Save metrics to JSON file",
    )

    args = parser.parse_args()

    # Load image
    img = cv2.imread(str(args.input))
    if img is None:
        raise ValueError(f"Failed to load image: {args.input}")

    print(f"\nComparing enhancement methods on: {args.input.name}")
    print(f"Image size: {img.shape[1]}x{img.shape[0]}")

    # Run comparison
    comparator = ImageEnhancementComparator()
    results = comparator.compare_all(img)

    # Print comparison table
    print_comparison_table(results)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Save comparison grid
    grid = create_comparison_grid(results, title=f"Comparison: {args.input.name}")
    grid_path = args.output / f"{args.input.stem}_comparison_grid.jpg"
    cv2.imwrite(str(grid_path), grid)
    print(f"\n✓ Saved comparison grid: {grid_path}")

    # Save individual enhanced images
    for method_name, (enhanced, _) in results.items():
        output_path = args.output / f"{args.input.stem}_{method_name}.jpg"
        cv2.imwrite(str(output_path), enhanced)

    print(f"✓ Saved {len(results)} individual images to: {args.output}")

    # Save metrics if requested
    if args.save_metrics:
        metrics_only = {
            method: metrics
            for method, (_, metrics) in results.items()
        }

        metrics_path = args.output / f"{args.input.stem}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_only, f, indent=2)

        print(f"✓ Saved metrics: {metrics_path}")

    print("\n✓ Comparison complete!")


if __name__ == "__main__":
    main()
