#!/usr/bin/env python3
"""
Establish baseline metrics for image enhancement experiment.

EDS v1.0 compliant output:
- Generates: artifacts/YYYYMMDD_HHMM_baseline-quality-metrics.json
- Validates against EDS naming conventions

Checkpoint for OCR testing:
- Path: outputs/experiments/train/ocr/pan_resnet18_add_polygons_canonical/20241019_0033_00/checkpoints/epoch-18_step-001957.ckpt
- Performance: 97% hmean on test dataset
- Notes: Use this checkpoint for all OCR baseline testing

This script:
1. Analyzes test images for quality issues
2. Documents image characteristics (tint, skew, contrast, etc.)
3. Creates EDS-compliant baseline metrics file

Usage:
    python scripts/establish_baseline.py
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def analyze_image_quality(image_path: Path) -> dict[str, Any]:
    """
    Analyze image for quality metrics.

    Returns:
        Dict with quality metrics including:
        - background_color_mean: Average RGB of background
        - background_color_std: Standard deviation of background colors
        - brightness: Average brightness (0-255)
        - contrast: Standard deviation of brightness
        - size: Image dimensions
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return {"error": "Failed to load image"}

    height, width = img.shape[:2]

    # Convert to different color spaces for analysis
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Background estimation using edge detection
    edges = cv2.Canny(gray, 50, 150)
    bg_mask = cv2.dilate(edges, np.ones((5,5)), iterations=2) == 0

    # Calculate background color metrics
    if bg_mask.sum() > 0:
        bg_pixels = img[bg_mask]
        bg_mean = bg_pixels.mean(axis=0).tolist()  # [B, G, R]
        bg_std = bg_pixels.std(axis=0).tolist()
        bg_color_variance = float(np.mean(bg_std))
    else:
        bg_mean = [0, 0, 0]
        bg_std = [0, 0, 0]
        bg_color_variance = 0.0

    # Overall brightness and contrast
    brightness = float(gray.mean())
    contrast = float(gray.std())

    # Detect potential skew using Hough lines
    edges_binary = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges_binary, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Normalize to -90 to 90 range
            if angle > 90:
                angle = angle - 180
            elif angle < -90:
                angle = angle + 180
            angles.append(angle)

    estimated_skew = float(np.median(angles)) if angles else 0.0

    # Color cast detection (deviation from neutral gray)
    # Perfect white/gray would have equal BGR values
    bg_b, bg_g, bg_r = bg_mean
    color_cast_score = float(max(abs(bg_b - bg_g), abs(bg_g - bg_r), abs(bg_r - bg_b)))

    metrics = {
        "file": image_path.name,
        "size": {"width": width, "height": height},
        "background": {
            "mean_bgr": [float(x) for x in bg_mean],
            "std_bgr": [float(x) for x in bg_std],
            "color_variance": bg_color_variance,
            "color_cast_score": color_cast_score
        },
        "brightness": brightness,
        "contrast": contrast,
        "estimated_skew_degrees": estimated_skew,
        "edge_density": float(edges.sum() / (width * height))
    }

    # Quality assessment flags
    issues = []
    if bg_color_variance > 20:
        issues.append("high_background_color_variation")
    if color_cast_score > 30:
        issues.append("significant_color_tint")
    if abs(estimated_skew) > 2:
        issues.append("text_skew_detected")
    if contrast < 30:
        issues.append("low_contrast")
    if brightness < 80 or brightness > 200:
        issues.append("poor_brightness")

    metrics["issues"] = issues
    metrics["issue_count"] = len(issues)

    return metrics


def create_baseline_report(test_dir: Path, output_dir: Path):
    """Create comprehensive baseline report."""

    # Find all test images
    image_files = sorted(test_dir.glob("*.jpg"))

    print(f"📊 Analyzing {len(image_files)} images from {test_dir}")
    print()

    results = []

    for img_path in image_files:
        print(f"  Analyzing: {img_path.name}...", end=" ")
        metrics = analyze_image_quality(img_path)
        results.append(metrics)
        print(f"✓ ({metrics['issue_count']} issues)")

    # Create summary statistics
    summary = {
        "timestamp": datetime.now().isoformat(),
        "test_directory": str(test_dir),
        "total_images": len(results),
        "images_analyzed": results,
        "aggregate_stats": {
            "avg_background_color_variance": np.mean([r["background"]["color_variance"] for r in results]),
            "avg_color_cast_score": np.mean([r["background"]["color_cast_score"] for r in results]),
            "avg_brightness": np.mean([r["brightness"] for r in results]),
            "avg_contrast": np.mean([r["contrast"] for r in results]),
            "avg_estimated_skew": np.mean([abs(r["estimated_skew_degrees"]) for r in results]),
        },
        "common_issues": {}
    }

    # Count common issues
    all_issues = [issue for r in results for issue in r["issues"]]
    for issue in set(all_issues):
        summary["common_issues"][issue] = all_issues.count(issue)

    # Save results with EDS v1.0 compliant naming
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_file = output_dir / f"{timestamp}_baseline-quality-metrics.json"

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"📄 Baseline report saved to: {output_file}")
    print()
    print("=" * 60)
    print("BASELINE METRICS SUMMARY")
    print("=" * 60)
    print(f"Total images analyzed: {summary['total_images']}")
    print()
    print("Aggregate Statistics:")
    for key, value in summary['aggregate_stats'].items():
        print(f"  {key}: {value:.2f}")
    print()
    print("Common Issues:")
    for issue, count in sorted(summary['common_issues'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {issue}: {count}/{summary['total_images']} images ({100*count/summary['total_images']:.1f}%)")
    print()

    # Print per-image details
    print("Per-Image Analysis:")
    print()
    for r in results:
        print(f"  {r['file']}:")
        print(f"    Background variance: {r['background']['color_variance']:.1f}")
        print(f"    Color tint score: {r['background']['color_cast_score']:.1f}")
        print(f"    Estimated skew: {r['estimated_skew_degrees']:.2f}°")
        print(f"    Issues: {', '.join(r['issues']) if r['issues'] else 'None'}")
        print()


def main():
    """Main entry point."""
    # Set up paths
    experiment_dir = Path(__file__).parent.parent
    test_dir = Path("/workspaces/upstageailab-ocr-recsys-competition-ocr-2/data/zero_prediction_worst_performers")
    output_dir = experiment_dir / "artifacts"

    print()
    print("=" * 60)
    print("IMAGE ENHANCEMENT EXPERIMENT - BASELINE ESTABLISHMENT")
    print("=" * 60)
    print()

    if not test_dir.exists():
        print(f"❌ Error: Test directory not found: {test_dir}")
        return 1

    create_baseline_report(test_dir, output_dir)

    print("✅ Baseline establishment complete!")
    print()
    print("Checkpoint for OCR testing:")  # Fixed: Added missing quote
    print("  outputs/experiments/train/ocr/pan_resnet18_add_polygons_canonical/")
    print("  20241019_0033_00/checkpoints/epoch-18_step-001957.ckpt")
    print("  Performance: 97% hmean")
    print()
    print("Next steps:")
    print("  1. Review artifacts/phase1_baseline_metrics.json")  # Fixed: Removed undefined variable
    print("  2. Implement background normalization methods")
    print("  3. Run enhancement tests and compare metrics")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
