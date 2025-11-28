#!/usr/bin/env python3
import sys
import cv2
import numpy as np
import logging
from pathlib import Path
import datetime
import glob
from typing import Optional

# Setup experiment paths - auto-detect tracker root and experiment context
script_path = Path(__file__).resolve()
tracker_root = script_path.parent.parent.parent.parent.parent / "src"
sys.path.insert(0, str(tracker_root))
from experiment_tracker.utils.path_utils import setup_script_paths, ExperimentPaths

# Setup OCR project paths
workspace_root = tracker_root.parent.parent
sys.path.insert(0, str(workspace_root))
from ocr.utils.path_utils import get_path_resolver, PROJECT_ROOT

# Auto-detect experiment context
TRACKER_ROOT, EXPERIMENT_ID, EXPERIMENT_PATHS = setup_script_paths(script_path)
OCR_RESOLVER = get_path_resolver()

# Import from local script
sys.path.insert(0, str(script_path.parent))

from mask_only_edge_detector import fit_mask_rectangle, visualize_mask_fit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_worst_performers(
    worst_case_dir: str,
    output_base_dir: str,
    dataset_root: str,
    use_regression: bool = False,
    regression_epsilon_px: float = 10.0,
    use_dominant_extension: bool = True,
    id_list_file: Optional[str] = None,
) -> Path:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base_dir) / f"{timestamp}_dominant_extension"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving outputs to {output_dir}")

    worst_case_path = Path(worst_case_dir)
    # Find mask files to identify the worst performers
    # Filter out warped masks to only get the original masks
    all_mask_files = list(worst_case_path.glob("*_mask.jpg"))
    mask_files = [f for f in all_mask_files if "_warped_mask.jpg" not in f.name]

    mask_files.sort()

    selected_files = mask_files[:25]
    if id_list_file:
        requested_ids = [
            line.strip()
            for line in Path(id_list_file).read_text().splitlines()
            if line.strip()
        ]
        normalized_ids = {rid.replace(".jpg", "") for rid in requested_ids}
        selected = []
        for mask_file in mask_files:
            img_id = mask_file.name.replace("_mask.jpg", "")
            if img_id in normalized_ids:
                selected.append(mask_file)
        missing = normalized_ids - {f.name.replace("_mask.jpg", "") for f in selected}
        if missing:
            logger.warning(f"{len(missing)} IDs from {id_list_file} not found in {worst_case_dir}: {sorted(missing)}")
        selected_files = selected

    logger.info(f"Found {len(all_mask_files)} masks, evaluating {len(selected_files)} selections")

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
            regression_epsilon_px=regression_epsilon_px,
            use_dominant_extension=use_dominant_extension
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
        eps_text = f"Eps: {result.used_epsilon:.2f} px (Ext)" if result.used_epsilon is not None else "Eps: N/A"
        cv2.putText(vis, eps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        output_path = output_dir / f"{img_id}_ext_vis.jpg"
        cv2.imwrite(str(output_path), vis)
        logger.info(f"Processed {img_id}: {result.reason or 'Success'} (Eps: {result.used_epsilon})")

        results.append({
            "id": img_id,
            "reason": result.reason,
            "decision": result.line_quality.decision if result.line_quality else "N/A",
            "used_epsilon": result.used_epsilon
        })

    # Summary
    total = max(len(results), 1)
    success_count = sum(1 for r in results if r["reason"] is None)
    logger.info(f"Success rate: {success_count}/{len(results)} ({success_count/total*100:.1f}%)")

    # Write summary to file
    with open(output_dir / "summary.txt", "w") as f:
        f.write(f"Evaluation Timestamp: {timestamp}\n")
        f.write(f"Parameters: use_dominant_extension={use_dominant_extension}, regression_epsilon_px={regression_epsilon_px}\n")
        f.write(f"Success rate: {success_count}/{len(results)}\n\n")
        for r in results:
            f.write(f"{r['id']}: {r['reason']} ({r['decision']})\n")

    return output_dir

if __name__ == "__main__":
    worst_case_dir = str(OCR_RESOLVER.config.output_dir / "improved_edge_approach" / "worst_force_improved")
    output_base_dir = str(EXPERIMENT_PATHS.get_artifacts_path() if EXPERIMENT_PATHS else Path.cwd() / "artifacts")
    dataset_root = str(OCR_RESOLVER.config.images_dir / "train")
    id_list_file = str(EXPERIMENT_PATHS.base_path / "worst_performers_top25.txt" if EXPERIMENT_PATHS else Path.cwd() / "worst_performers_top25.txt")

    evaluate_worst_performers(
        worst_case_dir,
        output_base_dir,
        dataset_root,
        id_list_file=id_list_file,
    )
