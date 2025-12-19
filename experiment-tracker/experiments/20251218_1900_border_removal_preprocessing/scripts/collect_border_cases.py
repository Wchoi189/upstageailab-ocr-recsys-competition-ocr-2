#!/usr/bin/env python3
"""
Border case dataset collection script.

Identifies images with border artifacts causing extreme skew misdetection.
Uses heuristic: abs(skew) > 20° AND zero predictions.

Usage:
    python collect_border_cases.py --output outputs/border_cases_manifest.json
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def estimate_skew_simple(img_path: Path) -> float:
    """
    Quick skew estimation using Hough lines (no ML dependencies).

    Args:
        img_path: Path to image file

    Returns:
        Estimated skew angle in degrees
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return 0.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

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

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) < 1:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if angle > 90:
            angle = angle - 180
        elif angle < -90:
            angle = angle + 180
        if abs(angle) < 45:
            angles.append(angle)

    if not angles:
        return 0.0

    return float(np.median(angles))


def has_border(img_path: Path, border_threshold: int = 20) -> tuple[bool, dict]:
    """
    Detect if image has black border artifacts.

    Uses heuristic: check if edge pixels are predominantly dark.

    Args:
        img_path: Path to image file
        border_threshold: Pixel value threshold for "black" (0-255)

    Returns:
        Tuple of (has_border: bool, metrics: dict)
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return False, {}

    h, w = img.shape[:2]

    # Sample edge pixels (top 5%, bottom 5%, left 5%, right 5%)
    edge_width = max(5, int(0.05 * w))
    edge_height = max(5, int(0.05 * h))

    top_edge = img[:edge_height, :, :]
    bottom_edge = img[-edge_height:, :, :]
    left_edge = img[:, :edge_width, :]
    right_edge = img[:, -edge_width:, :]

    # Convert to grayscale for intensity check
    edges = [
        cv2.cvtColor(top_edge, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(bottom_edge, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(left_edge, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(right_edge, cv2.COLOR_BGR2GRAY),
    ]

    # Calculate percentage of dark pixels in edges
    dark_percentages = []
    for edge in edges:
        dark_pixels = (edge < border_threshold).sum()
        total_pixels = edge.size
        dark_pct = dark_pixels / total_pixels
        dark_percentages.append(dark_pct)

    # Heuristic: border detected if >50% of edge pixels are dark on 2+ sides
    dark_sides = sum(1 for pct in dark_percentages if pct > 0.5)
    has_border_flag = dark_sides >= 2

    metrics = {
        "dark_percentages": [round(float(pct), 3) for pct in dark_percentages],
        "dark_sides": dark_sides,
        "border_threshold": border_threshold,
    }

    return has_border_flag, metrics


def collect_border_cases(
    source_dirs: list[Path],
    output_path: Path,
    skew_threshold: float = 20.0,
    max_samples: int = 50,
) -> dict:
    """
    Collect images with border artifacts.

    Args:
        source_dirs: Directories to search for images
        output_path: Path to save manifest JSON
        skew_threshold: Minimum absolute skew to consider
        max_samples: Maximum number of samples to collect

    Returns:
        Manifest dictionary
    """
    print(f"Searching for border cases in {len(source_dirs)} directories...")
    print(f"Criteria: abs(skew) > {skew_threshold}° OR visual border detection")
    print()

    border_cases = []

    for source_dir in source_dirs:
        if not source_dir.exists():
            print(f"Warning: {source_dir} does not exist, skipping...")
            continue

        image_files = (
            list(source_dir.glob("*.jpg"))
            + list(source_dir.glob("*.png"))
            + list(source_dir.glob("*.jpeg"))
        )

        print(f"Scanning {len(image_files)} images in {source_dir}...")

        for img_path in image_files:
            if len(border_cases) >= max_samples:
                break

            try:
                # Estimate skew
                skew_deg = estimate_skew_simple(img_path)

                # Detect borders
                has_border_flag, border_metrics = has_border(img_path)

                # Collect if meets criteria
                if abs(skew_deg) > skew_threshold or has_border_flag:
                    case = {
                        "image_path": str(img_path.absolute()),
                        "image_name": img_path.name,
                        "skew_deg": round(skew_deg, 2),
                        "skew_abs": round(abs(skew_deg), 2),
                        "has_border_detected": has_border_flag,
                        "border_metrics": border_metrics,
                    }
                    border_cases.append(case)
                    print(f"  ✓ {img_path.name}: skew={skew_deg:.1f}°, border={has_border_flag}")

            except Exception as e:
                print(f"  ✗ Error processing {img_path.name}: {e}")
                continue

        if len(border_cases) >= max_samples:
            print(f"\nReached max_samples ({max_samples}), stopping...")
            break

    # Sort by skew magnitude
    border_cases.sort(key=lambda x: x["skew_abs"], reverse=True)

    # Create manifest
    manifest = {
        "total_cases": len(border_cases),
        "criteria": {
            "skew_threshold_deg": skew_threshold,
            "max_samples": max_samples,
        },
        "cases": border_cases,
    }

    # Save manifest
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total border cases found: {len(border_cases)}")
    print(f"Highest skew: {border_cases[0]['skew_abs']:.1f}° ({border_cases[0]['image_name']})")
    print(f"Cases with detected borders: {sum(1 for c in border_cases if c['has_border_detected'])}")
    print(f"Manifest saved to: {output_path}")
    print()

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Collect images with border artifacts for testing"
    )

    parser.add_argument(
        "--source-dirs",
        type=Path,
        nargs="+",
        default=[Path("data/zero_prediction_worst_performers")],
        help="Directories to search for border cases (default: worst performers)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/experiments/20251218_1900_border_removal_preprocessing/border_cases_manifest.json"),
        help="Output manifest path",
    )

    parser.add_argument(
        "--skew-threshold",
        type=float,
        default=20.0,
        help="Minimum absolute skew to consider (degrees)",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Maximum number of samples to collect",
    )

    args = parser.parse_args()

    manifest = collect_border_cases(
        source_dirs=args.source_dirs,
        output_path=args.output,
        skew_threshold=args.skew_threshold,
        max_samples=args.max_samples,
    )

    return 0


if __name__ == "__main__":
    exit(main())
