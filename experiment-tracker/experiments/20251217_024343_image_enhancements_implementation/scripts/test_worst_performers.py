#!/usr/bin/env python3
"""
Test perspective correction on worst performers from the list.
"""

import datetime
import json
import logging
import sys
from pathlib import Path

import cv2

# Setup experiment paths - auto-detect tracker root and experiment context
script_path = Path(__file__).resolve()
# Point to experiment-tracker/src
tracker_root = script_path.parents[3] / "src"
sys.path.insert(0, str(tracker_root))
from experiment_tracker.utils.path_utils import setup_script_paths

# Setup OCR project paths
workspace_root = tracker_root.parent.parent
sys.path.insert(0, str(workspace_root))
from ocr.utils.path_utils import get_path_resolver

# Auto-detect experiment context
TRACKER_ROOT, EXPERIMENT_ID, EXPERIMENT_PATHS = setup_script_paths(script_path)
OCR_RESOLVER = get_path_resolver()

# Import from local script
sys.path.insert(0, str(script_path.parent))

from mask_only_edge_detector import fit_mask_rectangle
from perspective_transformer import four_point_transform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_mask_file(img_id: str, search_dirs: list[Path]) -> Path | None:
    """Find mask file for given image ID in multiple possible locations."""
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        # Try different mask naming patterns
        patterns = [
            f"{img_id}_mask.jpg",
            f"{img_id}_mask.png",
            f"{img_id}.mask.jpg",
            f"{img_id}.mask.png",
        ]
        for pattern in patterns:
            mask_path = search_dir / pattern
            if mask_path.exists():
                return mask_path
    return None


def find_original_image(img_id: str, dataset_root: Path) -> Path | None:
    """Find original image file for given image ID."""
    extensions = [".jpg", ".png", ".JPG", ".jpeg", ".JPEG"]
    for ext in extensions:
        img_path = dataset_root / f"{img_id}{ext}"
        if img_path.exists():
            return img_path
    return None


def test_worst_performers(
    worst_performers_file: Path,
    output_base_dir: Path,
    dataset_root: Path,
    mask_search_dirs: list[Path],
    use_regression: bool = False,
    regression_epsilon_px: float = 10.0,
    use_dominant_extension: bool = True,
) -> dict:
    """Test perspective correction on worst performers from file."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base_dir / f"{timestamp}_worst_performers_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving outputs to {output_dir}")

    # Read worst performers list
    if not worst_performers_file.exists():
        logger.error(f"Worst performers file not found: {worst_performers_file}")
        return {"error": "File not found"}

    with open(worst_performers_file) as f:
        image_ids = [line.strip() for line in f if line.strip()]

    # Remove .jpg extension if present
    image_ids = [img_id.replace(".jpg", "") for img_id in image_ids]

    logger.info(f"Testing {len(image_ids)} worst performers")

    results = {
        "total": len(image_ids),
        "success": 0,
        "failed": 0,
        "missing_mask": 0,
        "missing_image": 0,
        "details": [],
        "timestamp": timestamp,
    }

    for img_id in image_ids:
        result_entry = {
            "image_id": img_id,
            "status": "unknown",
            "reason": None,
            "output_file": None,
        }

        # Find mask
        mask_file = find_mask_file(img_id, mask_search_dirs)
        if mask_file is None:
            logger.warning(f"Mask not found for {img_id}")
            result_entry["status"] = "missing_mask"
            result_entry["reason"] = "Mask file not found"
            results["missing_mask"] += 1
            results["details"].append(result_entry)
            continue

        # Find original image
        original_img_path = find_original_image(img_id, dataset_root)
        if original_img_path is None:
            logger.warning(f"Original image not found for {img_id}")
            result_entry["status"] = "missing_image"
            result_entry["reason"] = "Original image not found"
            results["missing_image"] += 1
            results["details"].append(result_entry)
            continue

        # Load mask
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logger.error(f"Could not load mask {mask_file}")
            result_entry["status"] = "failed"
            result_entry["reason"] = "Failed to load mask"
            results["failed"] += 1
            results["details"].append(result_entry)
            continue

        # Load original image
        original_img = cv2.imread(str(original_img_path))
        if original_img is None:
            logger.error(f"Could not load original image {original_img_path}")
            result_entry["status"] = "failed"
            result_entry["reason"] = "Failed to load original image"
            results["failed"] += 1
            results["details"].append(result_entry)
            continue

        # Run edge detector
        try:
            result = fit_mask_rectangle(
                mask,
                use_regression=use_regression,
                regression_epsilon_px=regression_epsilon_px,
                use_dominant_extension=use_dominant_extension,
            )

            if result.corners is not None:
                try:
                    # Apply perspective transform
                    warped = four_point_transform(original_img, result.corners)

                    # Save result
                    output_path = output_dir / f"{img_id}_warped.jpg"
                    cv2.imwrite(str(output_path), warped)

                    result_entry["status"] = "success"
                    result_entry["output_file"] = str(output_path)
                    result_entry["reason"] = result.reason
                    results["success"] += 1
                    logger.info(f"✓ Success: {img_id}")
                except Exception as e:
                    logger.error(f"Failed to warp {img_id}: {e}")
                    result_entry["status"] = "failed"
                    result_entry["reason"] = f"Warp failed: {str(e)}"
                    results["failed"] += 1
            else:
                logger.warning(f"Failed to detect edges for {img_id}: {result.reason}")
                result_entry["status"] = "failed"
                result_entry["reason"] = result.reason or "Edge detection failed"
                results["failed"] += 1

        except Exception as e:
            logger.error(f"Error processing {img_id}: {e}")
            result_entry["status"] = "failed"
            result_entry["reason"] = f"Processing error: {str(e)}"
            results["failed"] += 1

        results["details"].append(result_entry)

    # Save results summary
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n{'=' * 60}")
    logger.info("Test Results Summary:")
    logger.info(f"  Total: {results['total']}")
    logger.info(f"  Success: {results['success']}")
    logger.info(f"  Failed: {results['failed']}")
    logger.info(f"  Missing Mask: {results['missing_mask']}")
    logger.info(f"  Missing Image: {results['missing_image']}")
    logger.info(f"  Success Rate: {results['success'] / results['total'] * 100:.1f}%")
    logger.info(f"{'=' * 60}")
    logger.info(f"Results saved to: {results_file}")

    return results


if __name__ == "__main__":
    # Path to worst performers file
    worst_performers_file = (
        workspace_root / "experiment-tracker" / "experiments" / "20251128_220100_perspective_correction" / "worst_performers_top25.txt"
    )

    # Output directory
    output_base_dir = EXPERIMENT_PATHS.get_artifacts_path() if EXPERIMENT_PATHS else Path.cwd() / "artifacts"

    # Dataset root
    dataset_root = Path(OCR_RESOLVER.config.images_dir) / "train"
    if not dataset_root.exists():
        dataset_root = Path("/workspaces/upstageailab-ocr-recsys-competition-ocr-2/data/datasets/images/train")

    # Possible mask directories to search
    mask_search_dirs = [
        OCR_RESOLVER.config.output_dir / "improved_edge_approach" / "worst_force_improved",
        OCR_RESOLVER.config.output_dir / "improved_edge_approach",
        workspace_root / "output" / "improved_edge_approach" / "worst_force_improved",
        workspace_root / "output" / "improved_edge_approach",
        Path.cwd() / "worst_force_improved",
    ]

    # Filter to only existing directories
    mask_search_dirs = [d for d in mask_search_dirs if d.exists()]

    logger.info(f"Searching for masks in: {[str(d) for d in mask_search_dirs]}")

    results = test_worst_performers(
        worst_performers_file,
        output_base_dir,
        dataset_root,
        mask_search_dirs,
    )
