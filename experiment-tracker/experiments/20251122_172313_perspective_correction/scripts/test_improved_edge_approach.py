#!/usr/bin/env python3
"""
Test improved edge-based perspective correction approach.

Compares:
1. Current approach (4 extreme points)
2. Improved approach (multi-point edge detection + line fitting)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import modules
script_dir = Path(__file__).parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# Setup path utils for proper path resolution
script_path = Path(__file__).resolve()
try:
    # Add tracker src to path
    tracker_root = script_path.parent.parent.parent.parent
    sys.path.insert(0, str(tracker_root / "src"))
    from experiment_tracker.utils.path_utils import setup_script_paths
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
    from ocr.utils.path_utils import get_path_resolver, PROJECT_ROOT
    OCR_RESOLVER = get_path_resolver()
except ImportError:
    OCR_RESOLVER = None
    PROJECT_ROOT = None

from analyze_failures_rembg_approach import (
    extract_rembg_mask,
    fit_quadrilateral_to_points,
    find_outer_points_from_mask,
    OptimizedBackgroundRemover,
    GPU_AVAILABLE,
)
from mask_only_edge_detector import (
    fit_mask_rectangle,
    visualize_mask_fit,
)
from improved_edge_based_correction import validate_homography_matrix

# Import perspective correction
try:
    from ocr.datasets.preprocessing.perspective import PerspectiveCorrector
    PERSPECTIVE_AVAILABLE = True
except ImportError:
    PERSPECTIVE_AVAILABLE = False
    logger.warning("Perspective correction not available.")


# ============================================================================
# Document-aware metric helpers
# ============================================================================

def _order_corners_clockwise(corners: np.ndarray) -> Optional[np.ndarray]:
    """Return corners ordered as TL, TR, BR, BL for downstream measurements."""
    if corners is None:
        return None

    pts = np.asarray(corners, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 4:
        return None

    if pts.shape[0] > 4:
        # Keep only first 4 to avoid downstream shape surprises
        pts = pts[:4]

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]  # top-left
    ordered[2] = pts[np.argmax(s)]  # bottom-right
    ordered[1] = pts[np.argmin(diff)]  # top-right
    ordered[3] = pts[np.argmax(diff)]  # bottom-left
    return ordered


def compute_skew_deviation_degrees(corners: np.ndarray) -> Optional[float]:
    """
    Calculate mean absolute deviation from 90° between adjacent edges.
    Lower values mean the quadrilateral is closer to a rectangle.
    """
    ordered = _order_corners_clockwise(corners)
    if ordered is None:
        return None

    def _angle(v1: np.ndarray, v2: np.ndarray) -> Optional[float]:
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom < 1e-6:
            return None
        cos_val = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_val)))

    vectors = [
        ordered[1] - ordered[0],  # top
        ordered[2] - ordered[1],  # right
        ordered[3] - ordered[2],  # bottom
        ordered[0] - ordered[3],  # left
    ]

    angles = []
    for idx in range(4):
        v1 = vectors[idx]
        v2 = vectors[(idx + 1) % 4]
        angle = _angle(v1, v2)
        if angle is not None:
            angles.append(angle)

    if not angles:
        return None

    deviations = [abs(90.0 - ang) for ang in angles]
    return float(np.mean(deviations))


def warp_mask(mask: np.ndarray, matrix: np.ndarray, corrected_shape: tuple[int, ...]) -> Optional[np.ndarray]:
    """Warp mask with the same homography used for the image."""
    if matrix is None or mask is None:
        return None

    corrected_h, corrected_w = corrected_shape[:2]
    if corrected_h <= 0 or corrected_w <= 0:
        return None

    warped = cv2.warpPerspective(
        mask,
        matrix,
        (corrected_w, corrected_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped


def _foreground_stats(mask: np.ndarray) -> tuple[int, int]:
    """Return (foreground_pixel_count, bbox_area) for a mask."""
    if mask is None:
        return (0, 0)

    mask_bin = (mask > 0).astype(np.uint8)
    foreground_pixels = int(np.count_nonzero(mask_bin))
    coords = cv2.findNonZero(mask_bin)
    if coords is None:
        return (foreground_pixels, 0)

    x, y, w, h = cv2.boundingRect(coords)
    bbox_area = int(w * h)
    return (foreground_pixels, bbox_area)


def compute_document_metrics(
    mask: np.ndarray,
    matrix: np.ndarray,
    corrected_shape: tuple[int, ...],
    corners: np.ndarray,
    output_dir: Path,
    prefix: str,
) -> dict[str, Any]:
    """
    Compute document-aware metrics by warping the mask and measuring retention.
    """
    metrics: dict[str, Any] = {}

    warped_mask = warp_mask(mask, matrix, corrected_shape)
    if warped_mask is None:
        return metrics

    orig_area, orig_bbox = _foreground_stats(mask)
    warped_area, warped_bbox = _foreground_stats(warped_mask)

    if orig_area > 0:
        metrics["document_area_retention"] = warped_area / orig_area
    if orig_bbox > 0:
        metrics["bbox_retention"] = warped_bbox / orig_bbox

    skew = compute_skew_deviation_degrees(corners)
    if skew is not None:
        metrics["skew_deviation_deg"] = skew

    warped_mask_path = output_dir / f"{prefix}_warped_mask.jpg"
    try:
        cv2.imwrite(str(warped_mask_path), warped_mask)
        metrics["warped_mask_path"] = str(warped_mask_path)
    except Exception as exc:
        logger.warning("Could not save warped mask for %s: %s", prefix, exc)

    return metrics


def test_both_approaches(
    image_path: Path,
    output_dir: Path,
    remover: OptimizedBackgroundRemover,
    skip_threshold: float | None = 0.85,
) -> dict[str, Any]:
    """
    Test both current and improved approaches on the same image.

    Returns comparison results.
    """
    logger.info(f"Processing: {image_path.name}")

    results = {
        "input_path": str(image_path),
        "current_approach": {},
        "improved_approach": {},
    }

    try:
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Extract rembg mask
        image_no_bg, mask = extract_rembg_mask(image, remover)
        results["mask_extracted"] = True

        # Save mask
        mask_output = output_dir / f"{image_path.stem}_mask.jpg"
        cv2.imwrite(str(mask_output), mask)

        # ===== CURRENT APPROACH =====
        logger.info("  Testing current approach (4 extreme points)...")
        try:
            outer_points = find_outer_points_from_mask(mask)
            corners_current = fit_quadrilateral_to_points(outer_points, image_no_bg.shape[:2])

            # Apply perspective correction
            if PERSPECTIVE_AVAILABLE:
                def ensure_doctr(feature: str) -> bool:
                    return False

                corrector = PerspectiveCorrector(
                    logger=logger,
                    ensure_doctr=ensure_doctr,
                    use_doctr_geometry=False,
                    doctr_assume_horizontal=False,
                )

                corrected_current, matrix_current, _method = corrector.correct(image_no_bg, corners_current)

                # Calculate metrics
                orig_h, orig_w = image_no_bg.shape[:2]
                corr_h, corr_w = corrected_current.shape[:2]
                area_ratio_current = (corr_h * corr_w) / (orig_h * orig_w)

                document_metrics_current = compute_document_metrics(
                    mask,
                    matrix_current,
                    corrected_current.shape,
                    corners_current,
                    output_dir,
                    f"{image_path.stem}_current",
                )

                results["current_approach"] = {
                    "success": area_ratio_current >= 0.5,
                    "area_ratio": area_ratio_current,
                    "corners": corners_current.tolist(),
                    "original_size": (orig_w, orig_h),
                    "corrected_size": (corr_w, corr_h),
                    "document_metrics": document_metrics_current,
                }

                # Save result
                current_output = output_dir / f"{image_path.stem}_current_corrected.jpg"
                cv2.imwrite(str(current_output), corrected_current)

                metrics_msg = []
                if document_metrics_current.get("document_area_retention") is not None:
                    metrics_msg.append(f"mask={document_metrics_current['document_area_retention']:.2%}")
                if document_metrics_current.get("bbox_retention") is not None:
                    metrics_msg.append(f"bbox={document_metrics_current['bbox_retention']:.2%}")
                if document_metrics_current.get("skew_deviation_deg") is not None:
                    metrics_msg.append(f"skew={document_metrics_current['skew_deviation_deg']:.2f}°")
                metrics_summary = ", ".join(metrics_msg) if metrics_msg else "metrics unavailable"
                logger.info(f"    Current: area ratio = {area_ratio_current:.2%} ({metrics_summary})")
            else:
                results["current_approach"] = {"error": "Perspective correction not available"}
        except Exception as e:
            results["current_approach"] = {"error": str(e)}
            logger.error(f"    Current approach failed: {e}")

        # Decide if improved approach should run
        current_ratio = results["current_approach"].get("area_ratio")
        skip_improved = False
        skip_reason = None
        if (
            skip_threshold is not None
            and current_ratio is not None
            and current_ratio >= skip_threshold
        ):
            skip_improved = True
            skip_reason = f"Current area ratio {current_ratio:.2%} >= {skip_threshold:.0%}"

        # ===== IMPROVED APPROACH =====
        if skip_improved:
            results["improved_approach"] = {
                "skipped": True,
                "reason": skip_reason,
                "strategy": "high_confidence_baseline",
            }
            logger.info(f"    Improved: Skipped ({skip_reason})")
        else:
            logger.info("  Testing improved approach (mask-only rectangle fit)...")
            try:
                mask_geometry = fit_mask_rectangle(mask)

                if mask_geometry.corners is None:
                    results["improved_approach"] = {
                        "skipped": True,
                        "reason": mask_geometry.reason or "mask_rectangle_failed",
                    }
                    logger.info(f"    Improved: Skipped ({mask_geometry.reason})")
                else:
                    # Log validation status if validation failed but bbox fallback was used
                    if mask_geometry.reason and "validation_failed" in mask_geometry.reason:
                        logger.warning(f"    Validation failed, using mask bbox fallback: {mask_geometry.reason}")
                    line_quality_info = None
                    if mask_geometry.line_quality is not None:
                        line_quality_info = {
                            "decision": mask_geometry.line_quality.decision,
                            "metrics": mask_geometry.line_quality.metrics,
                            "passes": mask_geometry.line_quality.passes,
                            "fail_reasons": mask_geometry.line_quality.fail_reasons,
                        }
                        fail_summary = ", ".join(line_quality_info["fail_reasons"]) if line_quality_info["fail_reasons"] else "none"
                        logger.info("    Line quality: %s (fails: %s)", line_quality_info["decision"], fail_summary)
                    corners_improved = mask_geometry.corners
                    vis = visualize_mask_fit(mask, corners_improved, mask_geometry.contour, mask_geometry.hull)
                    vis_output = output_dir / f"{image_path.stem}_improved_mask_fit.jpg"
                    cv2.imwrite(str(vis_output), vis)

                    if PERSPECTIVE_AVAILABLE:
                        def ensure_doctr(feature: str) -> bool:
                            return False

                        corrector = PerspectiveCorrector(
                            logger=logger,
                            ensure_doctr=ensure_doctr,
                            use_doctr_geometry=False,
                            doctr_assume_horizontal=False,
                        )

                        corrected_improved, matrix, _method = corrector.correct(image_no_bg, corners_improved)

                        from improved_edge_based_correction import validate_homography_matrix
                        is_valid_matrix, condition_number = validate_homography_matrix(matrix)

                        if not is_valid_matrix:
                            logger.warning(
                                "BUG-20251124-002: Ill-conditioned homography matrix (condition=%.2e)",
                                condition_number,
                            )
                            results["improved_approach"] = {
                                "error": f"Ill-conditioned homography matrix (condition={condition_number:.2e})",
                                "condition_number": float(condition_number),
                            }
                            if line_quality_info:
                                results["improved_approach"]["line_quality"] = line_quality_info
                        else:
                            orig_h, orig_w = image_no_bg.shape[:2]
                            corr_h, corr_w = corrected_improved.shape[:2]
                            area_ratio_improved = (corr_h * corr_w) / (orig_h * orig_w)

                            if area_ratio_improved > 1.5:
                                logger.warning(
                                    "BUG-20251124-002: Rejecting result - area ratio %s > 150%%",
                                    f"{area_ratio_improved:.2%}",
                                )
                                results["improved_approach"] = {
                                    "error": f"Area ratio too large: {area_ratio_improved:.2%} > 150%",
                                    "area_ratio": area_ratio_improved,
                                    "rejected": True,
                                }
                                if line_quality_info:
                                    results["improved_approach"]["line_quality"] = line_quality_info
                            else:
                                document_metrics_improved = compute_document_metrics(
                                    mask,
                                    matrix,
                                    corrected_improved.shape,
                                    corners_improved,
                                    output_dir,
                                    f"{image_path.stem}_improved",
                                )

                                results["improved_approach"] = {
                                    "success": area_ratio_improved >= 0.5,
                                    "area_ratio": area_ratio_improved,
                                    "corners": corners_improved.tolist(),
                                    "original_size": (orig_w, orig_h),
                                    "corrected_size": (corr_w, corr_h),
                                    "condition_number": float(condition_number),
                                    "document_metrics": document_metrics_improved,
                                }
                                if line_quality_info:
                                    results["improved_approach"]["line_quality"] = line_quality_info

                                improved_output = output_dir / f"{image_path.stem}_improved_corrected.jpg"
                                cv2.imwrite(str(improved_output), corrected_improved)

                                metrics_msg = []
                                if document_metrics_improved.get("document_area_retention") is not None:
                                    metrics_msg.append(f"mask={document_metrics_improved['document_area_retention']:.2%}")
                                if document_metrics_improved.get("bbox_retention") is not None:
                                    metrics_msg.append(f"bbox={document_metrics_improved['bbox_retention']:.2%}")
                                if document_metrics_improved.get("skew_deviation_deg") is not None:
                                    metrics_msg.append(f"skew={document_metrics_improved['skew_deviation_deg']:.2f}°")
                                metrics_summary = ", ".join(metrics_msg) if metrics_msg else "metrics unavailable"
                                logger.info(
                                    "    Improved: area ratio = %s, condition = %.2e (%s)",
                                    f"{area_ratio_improved:.2%}",
                                    condition_number,
                                    metrics_summary,
                                )
                    else:
                        results["improved_approach"] = {"error": "Perspective correction not available"}
                        if line_quality_info:
                            results["improved_approach"]["line_quality"] = line_quality_info
            except Exception as e:
                results["improved_approach"] = {"error": str(e)}
                if "line_quality_info" in locals():
                    info = locals()["line_quality_info"]
                    if info:
                        results["improved_approach"]["line_quality"] = info
                logger.error(f"    Improved approach failed: {e}")

        # ===== COMPARISON =====
        # BUG-20251124-002: Handle skipped/rejected cases in comparison
        if "area_ratio" in results["current_approach"] and "area_ratio" in results["improved_approach"]:
            current_ratio = results["current_approach"]["area_ratio"]
            improved_ratio = results["improved_approach"]["area_ratio"]

            improvement = improved_ratio - current_ratio
            results["comparison"] = {
                "improvement": improvement,
                "improvement_percent": improvement * 100,
                "current_success": results["current_approach"].get("success", False),
                "improved_success": results["improved_approach"].get("success", False),
            }

            if improvement > 0:
                logger.info(f"    ✓ Improved by {improvement:.2%}")
            elif improvement < 0:
                logger.warning(f"    ✗ Worsened by {abs(improvement):.2%}")
            else:
                logger.info(f"    = No change")
        elif "skipped" in results.get("improved_approach", {}) or "rejected" in results.get("improved_approach", {}):
            # Passthrough or rejected case
            reason = results["improved_approach"].get("reason") or results["improved_approach"].get("error", "Unknown")
            logger.info(f"    → {reason}")

        # Create comparison image
        if "corrected_size" in results["current_approach"] and "corrected_size" in results["improved_approach"]:
            try:
                corrected_current = cv2.imread(str(output_dir / f"{image_path.stem}_current_corrected.jpg"))
                corrected_improved = cv2.imread(str(output_dir / f"{image_path.stem}_improved_corrected.jpg"))

                if corrected_current is not None and corrected_improved is not None:
                    max_h = max(image_no_bg.shape[0], corrected_current.shape[0], corrected_improved.shape[0])
                    total_w = image_no_bg.shape[1] + corrected_current.shape[1] + corrected_improved.shape[1] + 40
                    comparison = np.zeros((max_h, total_w, 3), dtype=np.uint8)

                    x_offset = 0
                    for img, label in [
                        (image_no_bg, "rembg"),
                        (corrected_current, "current"),
                        (corrected_improved, "improved"),
                    ]:
                        h, w = img.shape[:2]
                        comparison[:h, x_offset:x_offset+w] = img
                        cv2.putText(comparison, label, (x_offset + 10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        x_offset += w + 20

                    comparison_output = output_dir / f"{image_path.stem}_comparison.jpg"
                    cv2.imwrite(str(comparison_output), comparison)
            except Exception as e:
                logger.warning(f"    Could not create comparison: {e}")

    except Exception as e:
        results["error"] = str(e)
        logger.error(f"  ✗ Failed: {e}", exc_info=True)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test improved edge-based perspective correction"
    )
    # Get workspace root using path resolvers
    if OCR_RESOLVER:
        default_input_dir = OCR_RESOLVER.config.images_dir / "train"
        default_output_dir = OCR_RESOLVER.config.output_dir / "improved_edge_approach"
    else:
        workspace_root = TRACKER_ROOT.parent if EXPERIMENT_PATHS else PROJECT_ROOT if PROJECT_ROOT else Path.cwd()
        default_input_dir = workspace_root / "data" / "datasets" / "images" / "train"
        default_output_dir = workspace_root / "outputs" / "improved_edge_approach"

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input_dir,
        help="Input directory with images (used as fallback for worst performers)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Output directory",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of images to test",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for rembg",
    )
    parser.add_argument(
        "--worst-performers",
        action="store_true",
        help="Test on worst performers from previous test",
    )
    parser.add_argument(
        "--skip-threshold",
        type=float,
        default=0.85,
        help="Skip improved correction when current area ratio is above this value. Set <=0 to disable.",
    )
    parser.add_argument(
        "--force-improved",
        action="store_true",
        help="Always run improved correction, ignoring the baseline skip threshold.",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize remover
    remover = OptimizedBackgroundRemover(
        model_name="silueta",
        max_size=2048,
        alpha_matting=False,
        use_gpu=args.use_gpu and GPU_AVAILABLE,
        use_tensorrt=False,
        use_int8=False,
    )

    # Get image list
    if args.worst_performers:
        # Load worst performers list
        if OCR_RESOLVER:
            worst_list_path = OCR_RESOLVER.config.output_dir / "worst_performers_test" / "worst_performers_list.json"
        else:
            workspace_root = TRACKER_ROOT.parent if EXPERIMENT_PATHS else PROJECT_ROOT if PROJECT_ROOT else Path.cwd()
            worst_list_path = workspace_root / "outputs" / "worst_performers_test" / "worst_performers_list.json"
        if worst_list_path.exists():
            with open(worst_list_path) as f:
                worst_list = json.load(f)

            # Resolve image paths
            image_files = []
            for w in worst_list[:args.num_samples]:
                input_path = Path(w["input_path"])

                # Try absolute path first
                if input_path.exists():
                    image_files.append(input_path)
                # Try relative to workspace root
                workspace_root = OCR_RESOLVER.config.project_root if OCR_RESOLVER else (TRACKER_ROOT.parent if EXPERIMENT_PATHS else PROJECT_ROOT if PROJECT_ROOT else Path.cwd())
                if (workspace_root / input_path).exists():
                    image_files.append(workspace_root / input_path)
                # Try in input_dir by filename
                elif args.input_dir.exists():
                    candidate = args.input_dir / input_path.name
                    if candidate.exists():
                        image_files.append(candidate)
                    else:
                        logger.warning(f"  ⚠ Image not found: {input_path.name} (from {input_path})")
                else:
                    logger.warning(f"  ⚠ Image not found: {input_path}")

            if not image_files:
                logger.error("No images found from worst performers list")
                return 1
        else:
            logger.error("Worst performers list not found")
            return 1
    else:
        # Get images from input directory
        image_files = []
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            image_files.extend(args.input_dir.glob(f"*{ext}"))
            image_files.extend(args.input_dir.glob(f"*{ext.upper()}"))
        image_files = sorted(image_files)[:args.num_samples]

    logger.info(f"Testing {len(image_files)} images...")

    # Determine skip policy
    skip_threshold = None if args.force_improved or args.skip_threshold <= 0 else args.skip_threshold

    # Test each image
    all_results = []
    current_success = 0
    improved_success = 0
    improved_attempts = 0
    improved_skipped = 0
    improved_rejected = 0
    improvements = []

    for i, image_path in enumerate(image_files, 1):
        if not image_path.exists():
            logger.warning(f"  ⚠ Image not found: {image_path}")
            continue

        logger.info(f"\n[{i}/{len(image_files)}]")
        result = test_both_approaches(
            image_path,
            args.output_dir,
            remover,
            skip_threshold=skip_threshold,
        )
        all_results.append(result)

        if "area_ratio" in result.get("current_approach", {}):
            if result["current_approach"]["success"]:
                current_success += 1

        improved_info = result.get("improved_approach", {})
        if "area_ratio" in improved_info:
            improved_attempts += 1
            if improved_info.get("success"):
                improved_success += 1
        elif improved_info.get("skipped"):
            improved_skipped += 1
        elif improved_info.get("rejected"):
            improved_rejected += 1

        if "comparison" in result:
            improvements.append(result["comparison"]["improvement"])

    # Summary
    logger.info("\n" + "="*80)
    logger.info("RESULTS SUMMARY")
    logger.info("="*80)
    logger.info(f"Total tested: {len(all_results)}")
    logger.info(f"Current approach success: {current_success}/{len(all_results)} ({100*current_success/len(all_results):.1f}%)")
    if improved_attempts > 0:
        logger.info(f"Improved approach attempts: {improved_attempts}/{len(all_results)}")
        logger.info(f"Improved approach success: {improved_success}/{improved_attempts} ({100*improved_success/max(1, improved_attempts):.1f}%)")
    else:
        logger.info("Improved approach attempts: 0/0")
    if improved_skipped or improved_rejected:
        logger.info(f"Improved skipped: {improved_skipped}, rejected: {improved_rejected}")

    if improvements:
        avg_improvement = np.mean(improvements)
        logger.info(f"Average area ratio improvement: {avg_improvement:.2%}")
        positive_improvements = sum(1 for imp in improvements if imp > 0)
        logger.info(f"Cases improved: {positive_improvements}/{len(improvements)} ({100*positive_improvements/len(improvements):.1f}%)")

    # Save results
    results_output = args.output_dir / "test_results.json"
    with open(results_output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to: {results_output}")

    return 0


if __name__ == "__main__":
    exit(main())

