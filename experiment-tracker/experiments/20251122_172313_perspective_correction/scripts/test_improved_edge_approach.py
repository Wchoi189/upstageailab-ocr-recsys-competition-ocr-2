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
from typing import Any

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
try:
    # Add tracker src to path
    tracker_root = Path(__file__).resolve().parent.parent.parent.parent
    sys.path.insert(0, str(tracker_root / "src"))
    from experiment_tracker.utils.path_utils import setup_script_paths
    TRACKER_ROOT, EXPERIMENT_ID, EXPERIMENT_PATHS = setup_script_paths(Path(__file__))
except ImportError:
    # Fallback if path_utils not available
    TRACKER_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    EXPERIMENT_ID = None
    EXPERIMENT_PATHS = None

from analyze_failures_rembg_approach import (
    extract_rembg_mask,
    fit_quadrilateral_to_points,
    find_outer_points_from_mask,
    OptimizedBackgroundRemover,
    GPU_AVAILABLE,
)
from improved_edge_based_correction import (
    fit_quadrilateral_from_edges,
    visualize_edges_and_lines,
    group_edge_points,
    extract_edge_points_from_mask,
)

# Import perspective correction
try:
    from ocr.datasets.preprocessing.perspective import PerspectiveCorrector
    PERSPECTIVE_AVAILABLE = True
except ImportError:
    PERSPECTIVE_AVAILABLE = False
    logger.warning("Perspective correction not available.")


def test_both_approaches(
    image_path: Path,
    output_dir: Path,
    remover: OptimizedBackgroundRemover,
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

                corrected_current, _matrix, _method = corrector.correct(image_no_bg, corners_current)

                # Calculate metrics
                orig_h, orig_w = image_no_bg.shape[:2]
                corr_h, corr_w = corrected_current.shape[:2]
                area_ratio_current = (corr_h * corr_w) / (orig_h * orig_w)

                results["current_approach"] = {
                    "success": area_ratio_current >= 0.5,
                    "area_ratio": area_ratio_current,
                    "corners": corners_current.tolist(),
                    "original_size": (orig_w, orig_h),
                    "corrected_size": (corr_w, corr_h),
                }

                # Save result
                current_output = output_dir / f"{image_path.stem}_current_corrected.jpg"
                cv2.imwrite(str(current_output), corrected_current)

                logger.info(f"    Current: area ratio = {area_ratio_current:.2%}")
            else:
                results["current_approach"] = {"error": "Perspective correction not available"}
        except Exception as e:
            results["current_approach"] = {"error": str(e)}
            logger.error(f"    Current approach failed: {e}")

        # ===== IMPROVED APPROACH =====
        logger.info("  Testing improved approach (edge-based line fitting)...")
        try:
            corners_improved = fit_quadrilateral_from_edges(mask, image_no_bg.shape[:2], use_ransac=True)

            # Visualize edges and lines
            edge_points = extract_edge_points_from_mask(mask)
            edge_groups = group_edge_points(edge_points, image_no_bg.shape[:2])

            # Fit lines for visualization
            from improved_edge_based_correction import fit_line_ransac
            lines = {}
            for edge_name, points in edge_groups.items():
                if len(points) >= 2:
                    lines[edge_name] = fit_line_ransac(points)
                else:
                    lines[edge_name] = None

            vis = visualize_edges_and_lines(mask, edge_groups, lines, corners_improved, image_no_bg.shape[:2])
            vis_output = output_dir / f"{image_path.stem}_improved_edges.jpg"
            cv2.imwrite(str(vis_output), vis)

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

                corrected_improved, _matrix, _method = corrector.correct(image_no_bg, corners_improved)

                # Calculate metrics
                orig_h, orig_w = image_no_bg.shape[:2]
                corr_h, corr_w = corrected_improved.shape[:2]
                area_ratio_improved = (corr_h * corr_w) / (orig_h * orig_w)

                results["improved_approach"] = {
                    "success": area_ratio_improved >= 0.5,
                    "area_ratio": area_ratio_improved,
                    "corners": corners_improved.tolist(),
                    "original_size": (orig_w, orig_h),
                    "corrected_size": (corr_w, corr_h),
                }

                # Save result
                improved_output = output_dir / f"{image_path.stem}_improved_corrected.jpg"
                cv2.imwrite(str(improved_output), corrected_improved)

                logger.info(f"    Improved: area ratio = {area_ratio_improved:.2%}")
            else:
                results["improved_approach"] = {"error": "Perspective correction not available"}
        except Exception as e:
            results["improved_approach"] = {"error": str(e)}
            logger.error(f"    Improved approach failed: {e}")

        # ===== COMPARISON =====
        if "area_ratio" in results["current_approach"] and "area_ratio" in results["improved_approach"]:
            current_ratio = results["current_approach"]["area_ratio"]
            improved_ratio = results["improved_approach"]["area_ratio"]

            improvement = improved_ratio - current_ratio
            results["comparison"] = {
                "improvement": improvement,
                "improvement_percent": improvement * 100,
                "current_success": results["current_approach"]["success"],
                "improved_success": results["improved_approach"]["success"],
            }

            if improvement > 0:
                logger.info(f"    ✓ Improved by {improvement:.2%}")
            elif improvement < 0:
                logger.warning(f"    ✗ Worsened by {abs(improvement):.2%}")
            else:
                logger.info(f"    = No change")

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
    # Get workspace root (parent of tracker root)
    workspace_root = TRACKER_ROOT.parent if EXPERIMENT_PATHS else Path("/workspaces/upstageailab-ocr-recsys-competition-ocr-2")

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=workspace_root / "data" / "datasets" / "images" / "train",
        help="Input directory with images (used as fallback for worst performers)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=workspace_root / "outputs" / "improved_edge_approach",
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
        workspace_root = TRACKER_ROOT.parent if EXPERIMENT_PATHS else Path("/workspaces/upstageailab-ocr-recsys-competition-ocr-2")
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
                elif (workspace_root / input_path).exists():
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

    # Test each image
    all_results = []
    current_success = 0
    improved_success = 0
    improvements = []

    for i, image_path in enumerate(image_files, 1):
        if not image_path.exists():
            logger.warning(f"  ⚠ Image not found: {image_path}")
            continue

        logger.info(f"\n[{i}/{len(image_files)}]")
        result = test_both_approaches(image_path, args.output_dir, remover)
        all_results.append(result)

        if "area_ratio" in result.get("current_approach", {}):
            if result["current_approach"]["success"]:
                current_success += 1

        if "area_ratio" in result.get("improved_approach", {}):
            if result["improved_approach"]["success"]:
                improved_success += 1

        if "comparison" in result:
            improvements.append(result["comparison"]["improvement"])

    # Summary
    logger.info("\n" + "="*80)
    logger.info("RESULTS SUMMARY")
    logger.info("="*80)
    logger.info(f"Total tested: {len(all_results)}")
    logger.info(f"Current approach success: {current_success}/{len(all_results)} ({100*current_success/len(all_results):.1f}%)")
    logger.info(f"Improved approach success: {improved_success}/{len(all_results)} ({100*improved_success/len(all_results):.1f}%)")

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

