#!/usr/bin/env python3
"""
Analyze failure cases and implement rembg mask-based perspective correction.

This script:
1. Extracts failure cases from test results
2. Analyzes statistical characteristics
3. Implements new approach using rembg mask to derive document boundaries
4. Tests the new approach on failure cases
"""

import argparse
import json
import logging
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import rembg
try:
    import sys

    script_dir = Path(__file__).parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from optimized_rembg import GPU_AVAILABLE, REMBG_AVAILABLE, OptimizedBackgroundRemover
except ImportError as e:
    logger.error(f"Failed to import optimized_rembg: {e}")
    REMBG_AVAILABLE = False
    GPU_AVAILABLE = False

# Setup path utils for proper path resolution
script_path = Path(__file__).resolve()
try:
    # Add tracker src to path
    tracker_root = script_path.parent.parent.parent.parent
    sys.path.insert(0, str(tracker_root / "src"))
    from etk.utils.path_utils import setup_script_paths

    TRACKER_ROOT, EXPERIMENT_ID, EXPERIMENT_PATHS = setup_script_paths(script_path)
except ImportError:
    # Fallback if path_utils not available
    TRACKER_ROOT = script_path.parent.parent.parent.parent
    EXPERIMENT_ID = None
    EXPERIMENT_PATHS = None

# Setup OCR project paths
workspace_root = tracker_root.parent
sys.path.insert(0, str(workspace_root))
try:
    from ocr.utils.path_utils import get_path_resolver

    OCR_RESOLVER = get_path_resolver()
except ImportError:
    OCR_RESOLVER = None
    PROJECT_ROOT = None

# Import perspective correction
try:
    from ocr.datasets.preprocessing.perspective import PerspectiveCorrector

    PERSPECTIVE_AVAILABLE = True
except ImportError:
    PERSPECTIVE_AVAILABLE = False
    logger.warning("Perspective correction not available.")


def extract_rembg_mask(image: np.ndarray, remover: OptimizedBackgroundRemover) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract mask from rembg output.

    Args:
        image: Input image (BGR format)
        remover: OptimizedBackgroundRemover instance

    Returns:
        Tuple of (image_with_bg_removed, mask) where mask is binary (0=background, 255=foreground)
    """
    # Convert to PIL
    if isinstance(image, np.ndarray):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
    else:
        pil_image = image

    # Get original size
    original_size = pil_image.size

    # Resize if needed (same logic as OptimizedBackgroundRemover)
    if max(original_size) != remover.max_size:
        scale = remover.max_size / max(original_size)
        new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
        pil_image_resized = pil_image.resize(new_size, Image.Resampling.LANCZOS)
    else:
        pil_image_resized = pil_image
        scale = 1.0

    # Remove background - this returns RGBA PIL Image
    from rembg import remove

    output = remove(
        pil_image_resized,
        session=remover.session,
        alpha_matting=remover.alpha_matting,
    )

    # Resize back if needed
    if scale != 1.0:
        output = output.resize(original_size, Image.Resampling.LANCZOS)

    # Convert to numpy array
    output_array = np.array(output)

    # Extract alpha channel as mask
    if output_array.shape[2] == 4:
        mask = output_array[:, :, 3]  # Alpha channel
        rgb = output_array[:, :, :3]

        # Composite on white background
        alpha = mask[:, :, np.newaxis] / 255.0
        white_bg = np.ones_like(rgb) * 255
        result = (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    else:
        # No alpha channel, create mask from non-white pixels
        result_bgr = cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

    # Convert mask to binary (0 or 255)
    mask_binary = (mask > 128).astype(np.uint8) * 255

    return result_bgr, mask_binary


def find_outer_points_from_mask(mask: np.ndarray) -> np.ndarray:
    """
    Find the outermost points of the object in the mask.

    Args:
        mask: Binary mask (0=background, 255=foreground)

    Returns:
        Array of outer points (contour points)
    """
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.array([])

    # Get the largest contour (main object)
    largest_contour = max(contours, key=cv2.contourArea)

    # Simplify contour to reduce noise
    epsilon = 0.001 * cv2.arcLength(largest_contour, True)
    simplified = cv2.approxPolyDP(largest_contour, epsilon, True)

    return simplified.reshape(-1, 2)


def fit_quadrilateral_to_points(points: np.ndarray, image_shape: tuple[int, int] | None = None) -> np.ndarray:
    """
    Fit a quadrilateral (4 corners) to a set of points using extreme points.

    This method finds the outermost points in 4 directions and fits lines to them.

    Args:
        points: Array of points (N, 2)
        image_shape: Optional (height, width) to clamp corners to image bounds

    Returns:
        Array of 4 corner points [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    """
    if len(points) < 4:
        # If we have fewer than 4 points, use bounding box
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        corners = box.astype(np.float32)
    elif len(points) == 4:
        # If we have exactly 4 points, use them
        corners = points.astype(np.float32)
    else:
        # Find 4 extreme points: topmost, rightmost, bottommost, leftmost
        # Then find intersection points of lines connecting these extremes

        # Get convex hull to reduce noise
        hull = cv2.convexHull(points)
        hull_points = hull.reshape(-1, 2)

        # Find extreme points
        topmost_idx = np.argmin(hull_points[:, 1])  # Smallest y
        bottommost_idx = np.argmax(hull_points[:, 1])  # Largest y
        leftmost_idx = np.argmin(hull_points[:, 0])  # Smallest x
        rightmost_idx = np.argmax(hull_points[:, 0])  # Largest x

        hull_points[topmost_idx]
        hull_points[bottommost_idx]
        hull_points[leftmost_idx]
        hull_points[rightmost_idx]

        # If we have 4 or fewer unique extreme points, use them directly
        extreme_indices = {topmost_idx, bottommost_idx, leftmost_idx, rightmost_idx}
        if len(extreme_indices) == 4:
            # We have 4 distinct extreme points
            extreme_points = np.array(
                [
                    hull_points[topmost_idx],
                    hull_points[rightmost_idx],
                    hull_points[bottommost_idx],
                    hull_points[leftmost_idx],
                ]
            )

            # Sort to ensure proper order: top-left, top-right, bottom-right, bottom-left
            corners = sort_corners(extreme_points)
        else:
            # Use min area rect as fallback
            rect = cv2.minAreaRect(hull_points)
            box = cv2.boxPoints(rect)
            corners = box.astype(np.float32)
            corners = sort_corners(corners)

    # Clamp corners to image bounds if provided
    if image_shape is not None:
        h, w = image_shape[:2]
        corners[:, 0] = np.clip(corners[:, 0], 0, w - 1)
        corners[:, 1] = np.clip(corners[:, 1], 0, h - 1)

    return corners


def sort_corners(corners: np.ndarray) -> np.ndarray:
    """Sort corners in order: top-left, top-right, bottom-right, bottom-left."""
    # Calculate sums and differences to identify corners
    sums = corners.sum(axis=1)
    diffs = np.diff(corners, axis=1).flatten()

    # Top-left: smallest sum
    top_left_idx = np.argmin(sums)
    # Bottom-right: largest sum
    bottom_right_idx = np.argmax(sums)

    # Top-right: smallest difference (x - y)
    top_right_idx = np.argmin(diffs)
    # Bottom-left: largest difference (x - y)
    bottom_left_idx = np.argmax(diffs)

    # Ensure all indices are unique
    indices = [top_left_idx, top_right_idx, bottom_right_idx, bottom_left_idx]
    if len(set(indices)) != 4:
        # Fallback: use simple geometric sorting
        # Sort by y first, then by x
        sorted_by_y = corners[np.argsort(corners[:, 1])]
        top_two = sorted_by_y[:2]
        bottom_two = sorted_by_y[2:]
        top_two = top_two[np.argsort(top_two[:, 0])]
        bottom_two = bottom_two[np.argsort(bottom_two[:, 0])]
        return np.vstack([top_two, bottom_two[::-1]])

    return corners[indices].astype(np.float32)


def extract_failure_cases(results_path: Path) -> list[dict[str, Any]]:
    """Extract failure cases from results JSON."""
    with open(results_path) as f:
        results = json.load(f)

    failures = []
    for result in results:
        regular_valid = result.get("regular_method", {}).get("validation", {}).get("valid", False)
        doctr_valid = result.get("doctr_method", {}).get("validation", {}).get("valid", False)

        if not regular_valid and not doctr_valid:
            failures.append(result)

    return failures


def analyze_failure_statistics(failures: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze statistical characteristics of failure cases."""
    stats = {
        "total_failures": len(failures),
        "failure_reasons": defaultdict(int),
        "pre_validation_failures": defaultdict(int),
        "post_validation_failures": defaultdict(int),
        "area_ratios": [],
        "aspect_ratios": [],
        "skew_angles": [],
    }

    for failure in failures:
        # Regular method
        regular_pre = failure.get("regular_method", {}).get("pre_validation", {})
        regular_post = failure.get("regular_method", {}).get("validation", {})

        # DocTR method
        doctr_pre = failure.get("doctr_method", {}).get("pre_validation", {})
        doctr_post = failure.get("doctr_method", {}).get("validation", {})

        # Collect pre-validation failures
        if not regular_pre.get("valid", True):
            reason = regular_pre.get("reason", "Unknown")
            stats["pre_validation_failures"][reason] += 1

        if not doctr_pre.get("valid", True):
            reason = doctr_pre.get("reason", "Unknown")
            stats["pre_validation_failures"][reason] += 1

        # Collect post-validation failures
        if not regular_post.get("valid", True):
            reason = regular_post.get("failure_reason", "Unknown")
            stats["post_validation_failures"][reason] += 1

        if not doctr_post.get("valid", True):
            reason = doctr_post.get("failure_reason", "Unknown")
            stats["post_validation_failures"][reason] += 1

        # Collect metrics (only numeric values)
        if "area_ratio" in regular_pre:
            val = regular_pre["area_ratio"]
            if isinstance(val, (int, float)) and not np.isnan(val):
                stats["area_ratios"].append(float(val))
        if "aspect_ratio" in regular_pre:
            val = regular_pre["aspect_ratio"]
            if isinstance(val, (int, float)) and not np.isnan(val):
                stats["aspect_ratios"].append(float(val))
        if "skew_angle" in regular_pre:
            val = regular_pre["skew_angle"]
            if isinstance(val, (int, float)) and not np.isnan(val):
                stats["skew_angles"].append(float(val))

        # Also check doctr method
        if "area_ratio" in doctr_pre:
            val = doctr_pre["area_ratio"]
            if isinstance(val, (int, float)) and not np.isnan(val):
                stats["area_ratios"].append(float(val))
        if "aspect_ratio" in doctr_pre:
            val = doctr_pre["aspect_ratio"]
            if isinstance(val, (int, float)) and not np.isnan(val):
                stats["aspect_ratios"].append(float(val))
        if "skew_angle" in doctr_pre:
            val = doctr_pre["skew_angle"]
            if isinstance(val, (int, float)) and not np.isnan(val):
                stats["skew_angles"].append(float(val))

    # Calculate statistics
    if stats["area_ratios"]:
        stats["area_ratio_stats"] = {
            "mean": statistics.mean(stats["area_ratios"]),
            "median": statistics.median(stats["area_ratios"]),
            "min": min(stats["area_ratios"]),
            "max": max(stats["area_ratios"]),
        }

    if stats["aspect_ratios"]:
        stats["aspect_ratio_stats"] = {
            "mean": statistics.mean(stats["aspect_ratios"]),
            "median": statistics.median(stats["aspect_ratios"]),
            "min": min(stats["aspect_ratios"]),
            "max": max(stats["aspect_ratios"]),
        }

    if stats["skew_angles"]:
        stats["skew_angle_stats"] = {
            "mean": statistics.mean(stats["skew_angles"]),
            "median": statistics.median(stats["skew_angles"]),
            "min": min(stats["skew_angles"]),
            "max": max(stats["skew_angles"]),
        }

    return stats


def test_rembg_based_correction(
    image_path: Path,
    output_dir: Path,
    remover: OptimizedBackgroundRemover,
) -> dict[str, Any]:
    """
    Test perspective correction using rembg mask-based corner detection.

    Args:
        image_path: Path to input image
        output_dir: Output directory
        remover: OptimizedBackgroundRemover instance

    Returns:
        Dictionary with results
    """
    logger.info(f"Processing: {image_path.name}")

    results = {
        "input_path": str(image_path),
        "success": False,
        "error": None,
        "method": "rembg_mask_based",
    }

    try:
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Step 1: Extract rembg mask
        image_no_bg, mask = extract_rembg_mask(image, remover)
        results["mask_extracted"] = True

        # Save mask for inspection
        mask_output = output_dir / f"{image_path.stem}_mask.jpg"
        cv2.imwrite(str(mask_output), mask)

        # Step 2: Find outer points from mask
        outer_points = find_outer_points_from_mask(mask)
        if len(outer_points) == 0:
            raise ValueError("No outer points found in mask")

        results["outer_points_count"] = len(outer_points)

        # Step 3: Fit quadrilateral (clamp to image bounds)
        corners = fit_quadrilateral_to_points(outer_points, image_no_bg.shape[:2])
        results["corners_detected"] = True
        results["corners"] = corners.tolist()

        # Visualize corners on image
        vis_image = image_no_bg.copy()
        for i, corner in enumerate(corners):
            pt = tuple(corner.astype(int))
            cv2.circle(vis_image, pt, 10, (0, 255, 0), -1)
            cv2.putText(vis_image, str(i), (pt[0] + 15, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw quadrilateral
        pts = corners.astype(int).reshape((-1, 1, 2))
        cv2.polylines(vis_image, [pts], True, (255, 0, 0), 2)

        corners_vis_output = output_dir / f"{image_path.stem}_corners.jpg"
        cv2.imwrite(str(corners_vis_output), vis_image)

        # Step 4: Apply perspective correction
        if not PERSPECTIVE_AVAILABLE:
            raise RuntimeError("Perspective correction not available")

        def ensure_doctr(feature: str) -> bool:
            return False  # Use regular method

        corrector = PerspectiveCorrector(
            logger=logger,
            ensure_doctr=ensure_doctr,
            use_doctr_geometry=False,
            doctr_assume_horizontal=False,
        )

        corrected_image, _matrix, correction_method = corrector.correct(image_no_bg, corners)
        results["correction_applied"] = True
        results["correction_method"] = correction_method

        # Save corrected image
        corrected_output = output_dir / f"{image_path.stem}_corrected.jpg"
        cv2.imwrite(str(corrected_output), corrected_image)

        # Validate result
        orig_h, orig_w = image_no_bg.shape[:2]
        corr_h, corr_w = corrected_image.shape[:2]

        orig_area = orig_h * orig_w
        corr_area = corr_h * corr_w
        area_ratio = corr_area / orig_area

        results["area_ratio"] = area_ratio
        results["original_size"] = (orig_w, orig_h)
        results["corrected_size"] = (corr_w, corr_h)

        # Check if result is valid
        if area_ratio >= 0.5:  # At least 50% area retained
            results["success"] = True
            results["valid"] = True
        else:
            results["valid"] = False
            results["failure_reason"] = f"Area loss too large: {area_ratio:.1%}"

        # Create comparison image
        max_h = max(image_no_bg.shape[0], corrected_image.shape[0])
        total_w = image_no_bg.shape[1] + corrected_image.shape[1] + 40
        comparison = np.zeros((max_h, total_w, 3), dtype=np.uint8)

        comparison[: image_no_bg.shape[0], : image_no_bg.shape[1]] = image_no_bg
        comparison[: corrected_image.shape[0], image_no_bg.shape[1] + 20 : image_no_bg.shape[1] + 20 + corrected_image.shape[1]] = (
            corrected_image
        )

        cv2.putText(comparison, "rembg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(comparison, "corrected", (image_no_bg.shape[1] + 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        comparison_output = output_dir / f"{image_path.stem}_comparison.jpg"
        cv2.imwrite(str(comparison_output), comparison)

    except Exception as e:
        results["error"] = str(e)
        logger.error(f"  ✗ Failed: {e}", exc_info=True)

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze failures and test rembg mask-based approach")
    parser.add_argument(
        "--results-json",
        type=Path,
        help="Path to results JSON file",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Input directory with original images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/rembg_mask_approach"),
        help="Output directory",
    )
    parser.add_argument(
        "--extract-failures",
        action="store_true",
        help="Extract failure cases to directory",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze statistics, don't test",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of failure cases to test",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for rembg",
    )

    args = parser.parse_args()

    if not REMBG_AVAILABLE:
        logger.error("rembg not available")
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Extract and analyze failures
    if args.results_json and args.results_json.exists():
        logger.info("Extracting failure cases...")
        failures = extract_failure_cases(args.results_json)
        logger.info(f"Found {len(failures)} failure cases")

        logger.info("Analyzing failure statistics...")
        stats = analyze_failure_statistics(failures)

        logger.info("\n" + "=" * 80)
        logger.info("FAILURE STATISTICS")
        logger.info("=" * 80)
        logger.info(f"Total failures: {stats['total_failures']}")

        logger.info("\nPre-validation failures:")
        for reason, count in sorted(stats["pre_validation_failures"].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {count}x: {reason}")

        logger.info("\nPost-validation failures:")
        for reason, count in sorted(stats["post_validation_failures"].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {count}x: {reason}")

        if "area_ratio_stats" in stats:
            logger.info(f"\nArea ratio stats: {stats['area_ratio_stats']}")
        if "aspect_ratio_stats" in stats:
            logger.info(f"Aspect ratio stats: {stats['aspect_ratio_stats']}")
        if "skew_angle_stats" in stats:
            logger.info(f"Skew angle stats: {stats['skew_angle_stats']}")

        # Save statistics
        stats_output = args.output_dir / "failure_statistics.json"
        with open(stats_output, "w") as f:
            json.dump(stats, f, indent=2, default=str)
        logger.info(f"\nStatistics saved to: {stats_output}")

        if args.analyze_only:
            return 0

        # Extract failure images if requested
        if args.extract_failures:
            failures_dir = args.output_dir / "failure_cases"
            failures_dir.mkdir(exist_ok=True)

            for i, failure in enumerate(failures[: args.num_samples]):
                input_path = Path(failure["input_path"])
                if input_path.exists():
                    import shutil

                    shutil.copy(input_path, failures_dir / input_path.name)
                    logger.info(f"Copied {input_path.name} to failures directory")

        # Test rembg-based approach on failures
        if args.input_dir and args.input_dir.exists():
            logger.info("\n" + "=" * 80)
            logger.info("TESTING REMBG MASK-BASED APPROACH")
            logger.info("=" * 80)

            # Initialize remover
            remover = OptimizedBackgroundRemover(
                model_name="silueta",
                max_size=2048,
                alpha_matting=False,
                use_gpu=args.use_gpu and GPU_AVAILABLE,
                use_tensorrt=False,
                use_int8=False,
            )

            # Find failure images
            failure_paths = []
            for failure in failures[: args.num_samples]:
                input_path = Path(failure["input_path"])
                if input_path.exists():
                    failure_paths.append(input_path)
                elif args.input_dir:
                    # Try to find in input dir
                    candidate = args.input_dir / input_path.name
                    if candidate.exists():
                        failure_paths.append(candidate)

            logger.info(f"Testing on {len(failure_paths)} failure cases...")

            test_results = []
            for image_path in failure_paths:
                result = test_rembg_based_correction(
                    image_path,
                    args.output_dir,
                    remover,
                )
                test_results.append(result)

                if result["success"]:
                    logger.info(f"  ✓ {image_path.name}: Success (area ratio: {result['area_ratio']:.2%})")
                else:
                    logger.warning(f"  ✗ {image_path.name}: {result.get('error') or result.get('failure_reason', 'Failed')}")

            # Summary
            successful = sum(1 for r in test_results if r.get("success"))
            logger.info(f"\nResults: {successful}/{len(test_results)} successful ({100 * successful / len(test_results):.1f}%)")

            # Save test results
            results_output = args.output_dir / "test_results.json"
            with open(results_output, "w") as f:
                json.dump(test_results, f, indent=2, default=str)
            logger.info(f"Test results saved to: {results_output}")

    else:
        logger.warning("No results JSON provided. Use --results-json to analyze failures.")

    return 0


if __name__ == "__main__":
    exit(main())
