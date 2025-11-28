#!/usr/bin/env python3
import sys
import cv2
import numpy as np
import logging
from pathlib import Path
import datetime
import glob

# Setup paths
tracker_root = Path(__file__).resolve().parent.parent.parent.parent.parent / "src"
sys.path.insert(0, str(tracker_root))
# Import from local script
sys.path.insert(0, str(Path(__file__).parent))

from mask_only_edge_detector import fit_mask_rectangle, visualize_mask_fit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_worst_performers(
    worst_case_dir: str,
    output_base_dir: str,
    dataset_root: str,
    use_regression: bool = True,
    regression_epsilon_px: float = 10.0,
):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base_dir) / f"{timestamp}_regression_pivot"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving outputs to {output_dir}")

    worst_case_path = Path(worst_case_dir)
    # Find mask files to identify the worst performers
    # Filter out warped masks to only get the original masks
    all_mask_files = list(worst_case_path.glob("*_mask.jpg"))
    mask_files = [f for f in all_mask_files if "_warped_mask.jpg" not in f.name]

    # Sort to ensure consistent order/selection if we limit to 25
    mask_files.sort()

    # Limit to 25 samples if there are more
    selected_files = mask_files[:25]
    logger.info(f"Found {len(all_mask_files)} total mask files, filtered to {len(mask_files)} original masks. Selecting {len(selected_files)} for evaluation")

    results = []

    for mask_file in selected_files:
        # Extract image ID (e.g., drp.en_ko.in_house.selectstar_000006)
        # Filename is ID_mask.jpg
        img_id = mask_file.name.replace("_mask.jpg", "")

        # Load mask
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logger.error(f"Could not load mask {mask_file}")
            continue

        # Run fit with new parameters
        result = fit_mask_rectangle(
            mask,
            use_regression=use_regression,
            regression_epsilon_px=regression_epsilon_px
        )

        # Visualize
        vis = visualize_mask_fit(mask, result.corners, result.contour, result.hull)

        # Add text about result
        status_color = (0, 255, 0) if result.reason is None else (0, 0, 255)
        text = f"Reason: {result.reason}"
        if result.line_quality:
            text += f" | Decision: {result.line_quality.decision}"

        cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Add epsilon info
        eps_text = f"Eps: {result.used_epsilon:.2f} px (Reg)" if result.used_epsilon is not None else "Eps: N/A"
        cv2.putText(vis, eps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        output_path = output_dir / f"{img_id}_reg_vis.jpg"
        cv2.imwrite(str(output_path), vis)
        logger.info(f"Processed {img_id}: {result.reason or 'Success'} (Eps: {result.used_epsilon})")

        results.append({
            "id": img_id,
            "reason": result.reason,
            "decision": result.line_quality.decision if result.line_quality else "N/A",
            "used_epsilon": result.used_epsilon
        })

    # Summary
    success_count = sum(1 for r in results if r["reason"] is None)
    logger.info(f"Success rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")

    # Write summary to file
    with open(output_dir / "summary.txt", "w") as f:
        f.write(f"Evaluation Timestamp: {timestamp}\n")
        f.write(f"Parameters: use_regression={use_regression}, regression_epsilon_px={regression_epsilon_px}\n")
        f.write(f"Success rate: {success_count}/{len(results)}\n\n")
        for r in results:
            f.write(f"{r['id']}: {r['reason']} ({r['decision']})\n")

if __name__ == "__main__":
    worst_case_dir = "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/outputs/improved_edge_approach/worst_force_improved"
    output_base_dir = "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/experiment-tracker/experiments/20251128_005231_perspective_correction/artifacts"
    dataset_root = "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/data/datasets/images/train"

    evaluate_worst_performers(worst_case_dir, output_base_dir, dataset_root)
