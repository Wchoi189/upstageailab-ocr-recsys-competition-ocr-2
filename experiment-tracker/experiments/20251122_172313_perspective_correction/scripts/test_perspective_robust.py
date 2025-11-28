#!/usr/bin/env python3
"""
Robust perspective correction test with validation and failure detection.

This script adds:
1. Pre-correction validation (check if correction is needed)
2. Post-correction validation (check if result is reasonable)
3. Statistical measurements to detect failures
4. Fallback to skip correction if validation fails
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# Add src to path to import experiment_tracker
# Use path_utils for proper path resolution
script_path = Path(__file__).resolve()
ROOT_DIR = script_path.parent.parent.parent.parent
sys.path.append(str(ROOT_DIR / "src"))

try:
    from experiment_tracker.utils.path_utils import setup_script_paths
    TRACKER_ROOT, EXPERIMENT_ID, EXPERIMENT_PATHS = setup_script_paths(script_path)
except ImportError:
    # Fallback if path_utils not available
    TRACKER_ROOT = ROOT_DIR
    EXPERIMENT_ID = None
    EXPERIMENT_PATHS = None

# Setup OCR project paths
workspace_root = ROOT_DIR.parent
sys.path.insert(0, str(workspace_root))
try:
    from ocr.utils.path_utils import get_path_resolver, PROJECT_ROOT
    OCR_RESOLVER = get_path_resolver()
except ImportError:
    OCR_RESOLVER = None
    PROJECT_ROOT = None

try:
    from experiment_tracker import track_experiment
except ImportError:
    # Fallback if not found (e.g. running standalone without setup)
    def track_experiment(**kwargs):
        def decorator(func):
            return func

        return decorator


import cv2
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import perspective correction
try:
    from ocr.datasets.preprocessing.detector import DocumentDetector
    from ocr.datasets.preprocessing.perspective import PerspectiveCorrector

    PERSPECTIVE_AVAILABLE = True
except ImportError:
    PERSPECTIVE_AVAILABLE = False
    logger.warning("Perspective correction not available.")


def calculate_skew_angle(corners: np.ndarray) -> float:
    """Calculate the maximum skew angle from corners."""
    # Calculate angles of each edge
    edges = [
        corners[1] - corners[0],  # top edge
        corners[2] - corners[1],  # right edge
        corners[3] - corners[2],  # bottom edge
        corners[0] - corners[3],  # left edge
    ]

    angles = []
    for edge in edges:
        angle = np.arctan2(edge[1], edge[0]) * 180 / np.pi
        angles.append(angle)

    # Calculate deviation from 0/90/180/270
    deviations = []
    for angle in angles:
        # Normalize to 0-90
        angle = abs(angle) % 90
        if angle > 45:
            angle = 90 - angle
        deviations.append(angle)

    return max(deviations)


def validate_corners(corners: np.ndarray, image_shape: tuple[int, int]) -> dict[str, Any]:
    """
    Validate detected corners before applying perspective correction.

    Returns:
        Dictionary with validation results
    """
    h, w = image_shape[:2]
    image_area = h * w

    # Calculate corner area
    corner_area = cv2.contourArea(corners)
    area_ratio = corner_area / image_area

    # Check if corners are too small
    min_area_ratio = 0.3  # At least 30% of image
    if area_ratio < min_area_ratio:
        return {
            "valid": False,
            "reason": f"Corner area too small: {area_ratio:.2%} < {min_area_ratio:.0%}",
            "area_ratio": area_ratio,
        }

    # Check if corners are too close to edges (might be detecting a small region)
    margin = min(h, w) * 0.1  # 10% margin
    min_x, min_y = corners.min(axis=0)
    max_x, max_y = corners.max(axis=0)

    if min_x < margin or min_y < margin or max_x > w - margin or max_y > h - margin:
        # This is OK - corners near edges are fine
        pass

    # Check if corners form a reasonable rectangle
    # Calculate aspect ratio
    width = max(np.linalg.norm(corners[1] - corners[0]), np.linalg.norm(corners[2] - corners[3]))
    height = max(np.linalg.norm(corners[3] - corners[0]), np.linalg.norm(corners[2] - corners[1]))

    aspect_ratio = width / height if height > 0 else 0
    image_aspect = w / h

    # Aspect ratio should be similar to image (within 50%)
    if aspect_ratio > 0 and (aspect_ratio / image_aspect < 0.5 or aspect_ratio / image_aspect > 2.0):
        return {
            "valid": False,
            "reason": f"Aspect ratio mismatch: {aspect_ratio:.2f} vs image {image_aspect:.2f}",
            "aspect_ratio": aspect_ratio,
            "image_aspect": image_aspect,
        }

    # Calculate skew angle
    skew_angle = calculate_skew_angle(corners)

    # If skew is very small (< 2 degrees), correction may not be needed
    if skew_angle < 2.0:
        return {
            "valid": False,
            "reason": f"Skew angle too small: {skew_angle:.1f}° (correction not needed)",
            "skew_angle": skew_angle,
            "skip_correction": True,  # Not an error, just skip
        }

    return {
        "valid": True,
        "area_ratio": area_ratio,
        "aspect_ratio": aspect_ratio,
        "skew_angle": skew_angle,
    }


def validate_correction_result(
    original: np.ndarray,
    corrected: np.ndarray,
    corners: np.ndarray,
) -> dict[str, Any]:
    """
    Validate the result after perspective correction.

    Returns:
        Dictionary with validation results
    """
    orig_h, orig_w = original.shape[:2]
    corr_h, corr_w = corrected.shape[:2]

    orig_area = orig_h * orig_w
    corr_area = corr_h * corr_w

    # Check if output is too small (likely cropped incorrectly)
    area_ratio = corr_area / orig_area
    min_area_ratio = 0.5  # Output should be at least 50% of original

    if area_ratio < min_area_ratio:
        return {
            "valid": False,
            "reason": f"Output too small: {corr_area} vs {orig_area} ({area_ratio:.1%})",
            "area_ratio": area_ratio,
            "original_size": (orig_w, orig_h),
            "corrected_size": (corr_w, corr_h),
        }

    # Check if dimensions are reasonable
    # Width/height should not shrink by more than 50%
    width_ratio = corr_w / orig_w
    height_ratio = corr_h / orig_h

    if width_ratio < 0.5 or height_ratio < 0.5:
        return {
            "valid": False,
            "reason": f"Dimension shrinkage too large: w={width_ratio:.1%}, h={height_ratio:.1%}",
            "width_ratio": width_ratio,
            "height_ratio": height_ratio,
        }

    # Check if output is mostly empty (should have content)
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY) if len(corrected.shape) == 3 else corrected
    non_zero_ratio = np.count_nonzero(gray) / gray.size

    if non_zero_ratio < 0.1:  # Less than 10% non-zero pixels
        return {
            "valid": False,
            "reason": f"Output mostly empty: {non_zero_ratio:.1%} non-zero pixels",
            "non_zero_ratio": non_zero_ratio,
        }

    return {
        "valid": True,
        "area_ratio": area_ratio,
        "width_ratio": width_ratio,
        "height_ratio": height_ratio,
        "non_zero_ratio": non_zero_ratio,
    }


def test_robust_perspective(
    image_path: Path,
    output_dir: Path,
    skip_if_not_needed: bool = True,
) -> dict[str, Any]:
    """Test perspective correction with validation."""
    logger.info(f"Processing: {image_path.name}")

    results = {
        "input_path": str(image_path),
        "correction_applied": False,
        "correction_needed": False,
        "pre_validation": {},
        "post_validation": {},
        "error": None,
    }

    try:
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        if not PERSPECTIVE_AVAILABLE:
            logger.warning("Perspective correction not available")
            return results

        # Detect corners
        detector = DocumentDetector(
            logger=logger,
            min_area_ratio=0.1,
            use_adaptive=True,
            use_fallback=True,
        )

        corners, method = detector.detect(image)

        if corners is None:
            logger.info("  No corners detected - skipping correction")
            results["error"] = "No corners detected"
            return results

        # Pre-correction validation
        pre_validation = validate_corners(corners, image.shape[:2])
        results["pre_validation"] = pre_validation

        if not pre_validation["valid"]:
            if pre_validation.get("skip_correction"):
                logger.info(f"  Correction not needed: {pre_validation['reason']}")
                results["correction_needed"] = False
                if skip_if_not_needed:
                    # Save original as "corrected"
                    output_path = output_dir / f"{image_path.stem}_corrected.jpg"
                    cv2.imwrite(str(output_path), image)
                    results["correction_applied"] = False
                    return results
            else:
                logger.warning(f"  Pre-validation failed: {pre_validation['reason']}")
                results["error"] = pre_validation["reason"]
                return results

        results["correction_needed"] = True

        # Apply perspective correction
        def ensure_doctr(feature: str) -> bool:
            return True

        corrector = PerspectiveCorrector(
            logger=logger,
            ensure_doctr=ensure_doctr,
            use_doctr_geometry=False,
            doctr_assume_horizontal=False,
        )

        corrected_image, _matrix, _method_name = corrector.correct(image, corners)

        # Post-correction validation
        post_validation = validate_correction_result(image, corrected_image, corners)
        results["post_validation"] = post_validation

        if not post_validation["valid"]:
            logger.error(f"  Post-validation failed: {post_validation['reason']}")
            results["error"] = post_validation["reason"]
            # Save original instead of bad correction
            output_path = output_dir / f"{image_path.stem}_corrected.jpg"
            cv2.imwrite(str(output_path), image)
            results["correction_applied"] = False
            return results

        # Save corrected result
        output_path = output_dir / f"{image_path.stem}_corrected.jpg"
        cv2.imwrite(str(output_path), corrected_image)

        # Save comparison
        h1, w1 = image.shape[:2]
        h2, w2 = corrected_image.shape[:2]
        max_h = max(h1, h2)
        comparison = np.zeros((max_h, w1 + w2 + 20, 3), dtype=np.uint8)
        comparison[:h1, :w1] = image
        comparison[:h2, w1 + 20 : w1 + 20 + w2] = corrected_image
        comparison_path = output_dir / f"{image_path.stem}_comparison.jpg"
        cv2.imwrite(str(comparison_path), comparison)

        results["correction_applied"] = True
        logger.info("  ✓ Correction applied successfully")
        logger.info(f"    Pre-validation: {pre_validation.get('skew_angle', 0):.1f}° skew")
        logger.info(f"    Post-validation: {post_validation.get('area_ratio', 0):.1%} area retained")

    except Exception as e:
        results["error"] = str(e)
        logger.error(f"  ✗ Failed: {e}", exc_info=True)

    return results


@track_experiment()
def main():
    parser = argparse.ArgumentParser(description="Test robust perspective correction with validation")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input directory with images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/perspective_robust_test"),
        help="Output directory",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of images to process",
    )
    parser.add_argument(
        "--skip-if-not-needed",
        action="store_true",
        default=True,
        help="Skip correction if not needed (default: True)",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find images
    image_files = []
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        image_files.extend(args.input_dir.glob(f"*{ext}"))
        image_files.extend(args.input_dir.glob(f"*{ext.upper()}"))

    if not image_files:
        logger.error("No image files found")
        return 1

    image_files = sorted(image_files)[: args.num_samples]
    logger.info(f"Processing {len(image_files)} images")

    # Process
    all_results = []
    for image_path in image_files:
        result = test_robust_perspective(
            image_path=image_path,
            output_dir=args.output_dir,
            skip_if_not_needed=args.skip_if_not_needed,
        )
        all_results.append(result)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)

    applied = [r for r in all_results if r.get("correction_applied")]
    skipped = [r for r in all_results if not r.get("correction_applied") and not r.get("error")]
    failed = [r for r in all_results if r.get("error")]

    logger.info(f"\nTotal: {len(all_results)}")
    logger.info(f"Correction applied: {len(applied)}")
    logger.info(f"Skipped (not needed): {len(skipped)}")
    logger.info(f"Failed: {len(failed)}")

    if failed:
        logger.info("\nFailures:")
        for r in failed:
            logger.info(f"  - {Path(r['input_path']).name}: {r.get('error')}")

    logger.info("\n" + "=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
