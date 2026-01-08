#!/usr/bin/env python3
import datetime
import logging
import sys
from pathlib import Path

import cv2

# Setup experiment paths - auto-detect tracker root and experiment context
script_path = Path(__file__).resolve()
# Point to experiment-tracker/src
tracker_root = script_path.parents[3] / "src"
sys.path.insert(0, str(tracker_root))
from etk.utils.path_utils import setup_script_paths

# Setup OCR project paths
workspace_root = tracker_root.parent.parent
sys.path.insert(0, str(workspace_root))
from ocr.core.utils.path_utils import get_path_resolver

# Auto-detect experiment context
TRACKER_ROOT, EXPERIMENT_ID, EXPERIMENT_PATHS = setup_script_paths(script_path)
OCR_RESOLVER = get_path_resolver()

# Import from local script
sys.path.insert(0, str(script_path.parent))

from mask_only_edge_detector import fit_mask_rectangle
from perspective_transformer import four_point_transform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_perspective_correction(
    worst_case_dir: str,
    output_base_dir: str,
    dataset_root: str,
    use_regression: bool = False,
    regression_epsilon_px: float = 10.0,
    use_dominant_extension: bool = True,
) -> Path:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base_dir) / f"{timestamp}_perspective_correction"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving outputs to {output_dir}")

    worst_case_path = Path(worst_case_dir)
    if not worst_case_path.exists():
        logger.error(f"Worst case directory not found: {worst_case_path}")
        return output_dir

    # Find mask files
    all_mask_files = list(worst_case_path.glob("*_mask.jpg"))
    # Filter out warped masks just in case
    mask_files = [f for f in all_mask_files if "_warped_mask.jpg" not in f.name]
    mask_files.sort()

    # Process a subset for testing if needed
    # selected_files = mask_files[:10]
    selected_files = mask_files  # Process all found

    logger.info(f"Found {len(all_mask_files)} masks, processing {len(selected_files)} selections")

    success_count = 0

    for mask_file in selected_files:
        img_id = mask_file.name.replace("_mask.jpg", "")

        # Load mask
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logger.error(f"Could not load mask {mask_file}")
            continue

        # Load original image
        # Try both .jpg and .png and .JPG
        found_img = False
        for ext in [".jpg", ".png", ".JPG", ".jpeg"]:
            img_path = Path(dataset_root) / f"{img_id}{ext}"
            if img_path.exists():
                found_img = True
                break

        if not found_img:
            logger.warning(f"Could not find original image for {img_id} in {dataset_root}")
            continue

        original_img = cv2.imread(str(img_path))
        if original_img is None:
            logger.error(f"Could not load original image {img_path}")
            continue

        # Run edge detector
        result = fit_mask_rectangle(
            mask, use_regression=use_regression, regression_epsilon_px=regression_epsilon_px, use_dominant_extension=use_dominant_extension
        )

        if result.corners is not None:
            try:
                # Apply perspective transform
                warped = four_point_transform(original_img, result.corners)

                # Save result
                output_path = output_dir / f"{img_id}_warped.jpg"
                cv2.imwrite(str(output_path), warped)

                success_count += 1
                if success_count % 10 == 0:
                    logger.info(f"Processed {success_count}/{len(selected_files)}")
            except Exception as e:
                logger.error(f"Failed to warp {img_id}: {e}")
        else:
            logger.warning(f"Failed to detect edges for {img_id}: {result.reason}")

    logger.info(f"Completed. Success rate: {success_count}/{len(selected_files)}")
    return output_dir


if __name__ == "__main__":
    # Use the path from evaluate_worst_performers.py logic
    # Note: Using OCR_RESOLVER logic from evaluate_worst_performers.py
    worst_case_dir = str(OCR_RESOLVER.config.output_dir / "improved_edge_approach" / "worst_force_improved")

    # Fallback if the path doesn't exist (maybe user ran it locally differently)
    # Check current directory for worst_force_improved if not found
    if not Path(worst_case_dir).exists():
        local_worst = Path.cwd() / "worst_force_improved"
        if local_worst.exists():
            worst_case_dir = str(local_worst)

    output_base_dir = str(EXPERIMENT_PATHS.get_artifacts_path() if EXPERIMENT_PATHS else Path.cwd() / "artifacts")
    dataset_root = str(Path(OCR_RESOLVER.config.images_dir) / "train")

    # Fallback for dataset root if config is wrong
    if not Path(dataset_root).exists():
        dataset_root = "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/data/datasets/images/train"

    run_perspective_correction(
        worst_case_dir,
        output_base_dir,
        dataset_root,
    )
