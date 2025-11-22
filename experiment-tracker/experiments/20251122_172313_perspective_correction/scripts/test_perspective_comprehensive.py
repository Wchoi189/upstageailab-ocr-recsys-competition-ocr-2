#!/usr/bin/env python3
"""
Comprehensive perspective correction evaluation with extended testing,
failure analysis, and fallback mechanisms.

This script:
1. Tests both DocTR and regular methods on larger dataset
2. Documents failure cases and success patterns
3. Implements fallback to original rembg version
4. Generates detailed performance metrics
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import rembg
try:
    import sys
    from pathlib import Path

    script_dir = Path(__file__).parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    from optimized_rembg import OptimizedBackgroundRemover, REMBG_AVAILABLE, GPU_AVAILABLE
except ImportError as e:
    logger.error(f"Failed to import optimized_rembg: {e}")
    REMBG_AVAILABLE = False
    GPU_AVAILABLE = False

# Import perspective correction
try:
    from ocr.datasets.preprocessing.detector import DocumentDetector
    from ocr.datasets.preprocessing.perspective import PerspectiveCorrector

    PERSPECTIVE_AVAILABLE = True
except ImportError:
    PERSPECTIVE_AVAILABLE = False
    logger.warning("Perspective correction not available.")


def calculate_image_metrics(image: np.ndarray) -> dict[str, float]:
    """Calculate metrics to evaluate image quality."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Edge strength
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    edge_strength = laplacian.var()

    # Contrast
    contrast = gray.std()

    # Brightness
    brightness = gray.mean()

    # Sharpness
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sharpness = np.sqrt(grad_x**2 + grad_y**2).mean()

    return {
        "edge_strength": float(edge_strength),
        "contrast": float(contrast),
        "brightness": float(brightness),
        "sharpness": float(sharpness),
    }


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

    # Check if corners form a reasonable rectangle
    # Calculate aspect ratio
    width = max(np.linalg.norm(corners[1] - corners[0]), np.linalg.norm(corners[2] - corners[3]))
    height = max(np.linalg.norm(corners[3] - corners[0]), np.linalg.norm(corners[2] - corners[1]))

    aspect_ratio = width / height if height > 0 else 0
    image_aspect = w / h

    # Aspect ratio should be similar to image (within 50-200%)
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
    method: str,
) -> dict[str, Any]:
    """
    Validate perspective correction result.

    Returns:
        Dictionary with validation results and failure reason if invalid
    """
    orig_h, orig_w = original.shape[:2]
    corr_h, corr_w = corrected.shape[:2]

    orig_area = orig_h * orig_w
    corr_area = corr_h * corr_w

    # Check area retention
    area_ratio = corr_area / orig_area
    min_area_ratio = 0.5  # At least 50% area retained

    if area_ratio < min_area_ratio:
        return {
            "valid": False,
            "failure_reason": f"Area loss too large: {area_ratio:.1%} < {min_area_ratio:.0%}",
            "area_ratio": area_ratio,
            "original_size": (orig_w, orig_h),
            "corrected_size": (corr_w, corr_h),
        }

    # Check dimension ratios
    width_ratio = corr_w / orig_w
    height_ratio = corr_h / orig_h

    if width_ratio < 0.5 or height_ratio < 0.5:
        return {
            "valid": False,
            "failure_reason": f"Dimension shrinkage too large: w={width_ratio:.1%}, h={height_ratio:.1%}",
            "width_ratio": width_ratio,
            "height_ratio": height_ratio,
        }

    # Check content preservation
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY) if len(corrected.shape) == 3 else corrected
    non_zero_ratio = np.count_nonzero(gray) / gray.size

    if non_zero_ratio < 0.1:
        return {
            "valid": False,
            "failure_reason": f"Output mostly empty: {non_zero_ratio:.1%} non-zero pixels",
            "non_zero_ratio": non_zero_ratio,
        }

    # Calculate quality metrics
    original_metrics = calculate_image_metrics(original)
    corrected_metrics = calculate_image_metrics(corrected)

    return {
        "valid": True,
        "area_ratio": area_ratio,
        "width_ratio": width_ratio,
        "height_ratio": height_ratio,
        "non_zero_ratio": non_zero_ratio,
        "original_metrics": original_metrics,
        "corrected_metrics": corrected_metrics,
        "quality_change": {
            "contrast": corrected_metrics["contrast"] - original_metrics["contrast"],
            "sharpness": corrected_metrics["sharpness"] - original_metrics["sharpness"],
        },
    }


def test_perspective_method(
    image: np.ndarray,
    use_doctr: bool,
    method_name: str,
) -> tuple[np.ndarray | None, dict[str, Any], float]:
    """Test a perspective correction method with pre-correction validation."""
    results = {
        "method": method_name,
        "use_doctr": use_doctr,
        "success": False,
        "detection_time": 0.0,
        "correction_time": 0.0,
        "error": None,
        "pre_validation": {},
    }

    try:
        # Detect corners
        detection_start = time.perf_counter()

        detector = DocumentDetector(
            logger=logger,
            min_area_ratio=0.3,  # Increased from 0.1 to 0.3 for better detection
            use_adaptive=True,
            use_fallback=True,
        )

        corners, detection_method = detector.detect(image)
        detection_time = time.perf_counter() - detection_start

        results["detection_time"] = detection_time
        results["detection_method"] = detection_method

        if corners is None:
            results["error"] = "No corners detected"
            return None, results, 0.0

        # Pre-correction validation
        pre_validation = validate_corners(corners, image.shape)
        results["pre_validation"] = pre_validation

        if not pre_validation["valid"]:
            if pre_validation.get("skip_correction"):
                results["error"] = f"Correction not needed: {pre_validation['reason']}"
            else:
                results["error"] = f"Pre-validation failed: {pre_validation['reason']}"
            return None, results, detection_time

        # Apply perspective correction
        correction_start = time.perf_counter()

        def ensure_doctr(feature: str) -> bool:
            return use_doctr

        corrector = PerspectiveCorrector(
            logger=logger,
            ensure_doctr=ensure_doctr,
            use_doctr_geometry=use_doctr,
            doctr_assume_horizontal=False,
        )

        corrected_image, _matrix, correction_method = corrector.correct(image, corners)
        correction_time = time.perf_counter() - correction_start

        results["correction_time"] = correction_time
        results["correction_method"] = correction_method
        results["success"] = True

        total_time = detection_time + correction_time
        return corrected_image, results, total_time

    except Exception as e:
        results["error"] = str(e)
        logger.error(f"Error in {method_name}: {e}", exc_info=True)
        return None, results, 0.0


def process_image_comprehensive(
    image_path: Path,
    output_dir: Path,
    use_gpu: bool = False,
) -> dict[str, Any]:
    """Comprehensive processing with multiple methods and fallback."""
    logger.info(f"Processing: {image_path.name}")

    results = {
        "input_path": str(image_path),
        "rembg_time": 0.0,
        "regular_method": {},
        "doctr_method": {},
        "fallback_used": False,
        "final_result": None,
        "error": None,
    }

    try:
        # Load original image
        original_image = cv2.imread(str(image_path))
        if original_image is None:
            raise ValueError(f"Could not load image: {image_path}")

        original_metrics = calculate_image_metrics(original_image)

        # Step 1: Background removal
        rembg_start = time.perf_counter()

        rembg_remover = OptimizedBackgroundRemover(
            model_name="silueta",
            max_size=2048,
            alpha_matting=False,
            use_gpu=use_gpu and GPU_AVAILABLE,
            use_tensorrt=False,
            use_int8=False,
        )

        image_no_bg = rembg_remover.remove_background(original_image)
        rembg_time = time.perf_counter() - rembg_start

        results["rembg_time"] = rembg_time
        rembg_metrics = calculate_image_metrics(image_no_bg)

        # Save rembg result (this is our fallback)
        rembg_output = output_dir / f"{image_path.stem}_00_rembg.jpg"
        cv2.imwrite(str(rembg_output), image_no_bg)

        # Step 2: Test regular method
        logger.info("  Testing regular method...")
        regular_corrected, regular_results, regular_time = test_perspective_method(
            image_no_bg,
            use_doctr=False,
            method_name="regular",
        )

        if regular_corrected is not None:
            regular_validation = validate_correction_result(
                image_no_bg,
                regular_corrected,
                "regular",
            )
            regular_results["validation"] = regular_validation

            if regular_validation["valid"]:
                regular_output = output_dir / f"{image_path.stem}_01_regular.jpg"
                cv2.imwrite(str(regular_output), regular_corrected)
                logger.info(f"    ✓ Regular: {regular_time:.3f}s (valid)")
            else:
                logger.warning(f"    ✗ Regular: {regular_time:.3f}s (invalid: {regular_validation['failure_reason']})")
        else:
            regular_results["validation"] = {"valid": False, "failure_reason": regular_results.get("error", "Unknown")}
            logger.warning(f"    ✗ Regular failed: {regular_results.get('error', 'Unknown')}")

        results["regular_method"] = regular_results

        # Step 3: Test DocTR method
        logger.info("  Testing DocTR method...")
        doctr_corrected, doctr_results, doctr_time = test_perspective_method(
            image_no_bg,
            use_doctr=True,
            method_name="doctr",
        )

        if doctr_corrected is not None:
            doctr_validation = validate_correction_result(
                image_no_bg,
                doctr_corrected,
                "doctr",
            )
            doctr_results["validation"] = doctr_validation

            if doctr_validation["valid"]:
                doctr_output = output_dir / f"{image_path.stem}_02_doctr.jpg"
                cv2.imwrite(str(doctr_output), doctr_corrected)
                logger.info(f"    ✓ DocTR: {doctr_time:.3f}s (valid)")
            else:
                logger.warning(f"    ✗ DocTR: {doctr_time:.3f}s (invalid: {doctr_validation['failure_reason']})")
        else:
            doctr_results["validation"] = {"valid": False, "failure_reason": doctr_results.get("error", "Unknown")}
            logger.warning(f"    ✗ DocTR failed: {doctr_results.get('error', 'Unknown')}")

        results["doctr_method"] = doctr_results

        # Step 4: Determine best result (with fallback)
        best_result = None
        best_method = None

        # Prefer valid results
        if regular_results.get("validation", {}).get("valid"):
            best_result = regular_corrected
            best_method = "regular"
        elif doctr_results.get("validation", {}).get("valid"):
            best_result = doctr_corrected
            best_method = "doctr"
        else:
            # Fallback to rembg version
            best_result = image_no_bg
            best_method = "rembg_fallback"
            results["fallback_used"] = True
            logger.info("  → Using rembg fallback (both methods failed validation)")

        # Save final result
        final_output = output_dir / f"{image_path.stem}_03_final_{best_method}.jpg"
        cv2.imwrite(str(final_output), best_result)

        results["final_result"] = {
            "method": best_method,
            "output_path": str(final_output),
            "output_shape": best_result.shape,
        }

        # Create comparison image
        images_to_compare = [image_no_bg]
        labels = ["rembg"]

        if regular_corrected is not None:
            images_to_compare.append(regular_corrected)
            labels.append("regular")

        if doctr_corrected is not None:
            images_to_compare.append(doctr_corrected)
            labels.append("doctr")

        images_to_compare.append(best_result)
        labels.append(f"final_{best_method}")

        # Create side-by-side comparison
        max_h = max(img.shape[0] for img in images_to_compare)
        total_w = sum(img.shape[1] for img in images_to_compare) + 20 * (len(images_to_compare) - 1)
        comparison = np.zeros((max_h, total_w, 3), dtype=np.uint8)

        x_offset = 0
        for img, label in zip(images_to_compare, labels):
            h, w = img.shape[:2]
            comparison[:h, x_offset : x_offset + w] = img
            # Add label text
            cv2.putText(
                comparison,
                label,
                (x_offset + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            x_offset += w + 20

        comparison_output = output_dir / f"{image_path.stem}_04_comparison.jpg"
        cv2.imwrite(str(comparison_output), comparison)

    except Exception as e:
        results["error"] = str(e)
        logger.error(f"  ✗ Failed: {e}", exc_info=True)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive perspective correction evaluation"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input directory with original images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/perspective_comprehensive"),
        help="Output directory",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of images to process (use -1 for all)",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for rembg",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save detailed results to JSON",
    )

    args = parser.parse_args()

    if not REMBG_AVAILABLE:
        logger.error("rembg not available")
        return 1

    if not PERSPECTIVE_AVAILABLE:
        logger.error("Perspective correction not available")
        return 1

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

    # Limit samples
    if args.num_samples > 0:
        image_files = sorted(image_files)[: args.num_samples]
    else:
        image_files = sorted(image_files)

    logger.info(f"Processing {len(image_files)} images")

    # Process
    logger.info("\n" + "=" * 80)
    logger.info("COMPREHENSIVE PERSPECTIVE CORRECTION EVALUATION")
    logger.info("=" * 80)

    all_results = []

    for image_path in image_files:
        result = process_image_comprehensive(
            image_path=image_path,
            output_dir=args.output_dir,
            use_gpu=args.use_gpu,
        )
        all_results.append(result)

    # Analysis
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS ANALYSIS")
    logger.info("=" * 80)

    # Categorize results
    regular_valid = [r for r in all_results if r.get("regular_method", {}).get("validation", {}).get("valid")]
    doctr_valid = [r for r in all_results if r.get("doctr_method", {}).get("validation", {}).get("valid")]
    fallback_used = [r for r in all_results if r.get("fallback_used")]
    both_failed = [r for r in all_results if not r.get("regular_method", {}).get("validation", {}).get("valid") and
                   not r.get("doctr_method", {}).get("validation", {}).get("valid")]

    logger.info(f"\nTotal images: {len(all_results)}")
    logger.info(f"Regular method valid: {len(regular_valid)} ({100*len(regular_valid)/len(all_results):.1f}%)")
    logger.info(f"DocTR method valid: {len(doctr_valid)} ({100*len(doctr_valid)/len(all_results):.1f}%)")
    logger.info(f"Fallback used: {len(fallback_used)} ({100*len(fallback_used)/len(all_results):.1f}%)")
    logger.info(f"Both methods failed: {len(both_failed)} ({100*len(both_failed)/len(all_results):.1f}%)")

    # Performance metrics
    if regular_valid:
        avg_regular_time = np.mean([
            r["regular_method"]["detection_time"] + r["regular_method"]["correction_time"]
            for r in regular_valid
        ])
        logger.info(f"\nAverage regular time (valid only): {avg_regular_time:.3f}s")

    if doctr_valid:
        avg_doctr_time = np.mean([
            r["doctr_method"]["detection_time"] + r["doctr_method"]["correction_time"]
            for r in doctr_valid
        ])
        logger.info(f"Average DocTR time (valid only): {avg_doctr_time:.3f}s")

        if regular_valid:
            speedup = avg_regular_time / avg_doctr_time if avg_doctr_time > 0 else 0
            logger.info(f"DocTR speedup: {speedup:.2f}x")

    # Failure analysis
    if both_failed:
        logger.info("\n" + "=" * 80)
        logger.info("FAILURE ANALYSIS")
        logger.info("=" * 80)

        failure_reasons = {}
        for r in both_failed:
            regular_reason = r.get("regular_method", {}).get("validation", {}).get("failure_reason", "Unknown")
            doctr_reason = r.get("doctr_method", {}).get("validation", {}).get("failure_reason", "Unknown")

            key = f"Regular: {regular_reason[:50]} | DocTR: {doctr_reason[:50]}"
            failure_reasons[key] = failure_reasons.get(key, 0) + 1

        logger.info("\nCommon failure patterns:")
        for reason, count in sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True)[:5]:
            logger.info(f"  {count}x: {reason}")

        # List failed images
        logger.info("\nFailed images:")
        for r in both_failed[:10]:  # Show first 10
            logger.info(f"  - {Path(r['input_path']).name}")

    # Save JSON results
    if args.save_json:
        json_path = args.output_dir / "results.json"
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"\nDetailed results saved to: {json_path}")

    logger.info("\n" + "=" * 80)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())

