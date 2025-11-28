#!/usr/bin/env python3
import sys
import cv2
import numpy as np
import logging
from pathlib import Path

# Setup paths
tracker_root = Path(__file__).resolve().parent.parent.parent.parent.parent / "src"
sys.path.insert(0, str(tracker_root))
sys.path.insert(0, "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/experiment-tracker/experiments/20251122_172313_perspective_correction/scripts")

from mask_only_edge_detector import fit_mask_rectangle, _prepare_mask, _collect_edge_support_data, visualize_mask_fit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def visualize_edge_support_sampling(
    image_path: str,
    output_path: str,
    distance_threshold: float = 5.0
):
    """
    Visualize which pixels are being sampled for edge support calculation.
    Draws hits (blue) and misses (red) relative to the fitted lines.
    """
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Could not load image: {image_path}")
        return

    vis = img.copy()

    input_p = Path(image_path)
    possible_mask_loc = Path("/workspaces/upstageailab-ocr-recsys-competition-ocr-2/outputs/improved_edge_approach/worst_force_improved") / f"{input_p.stem}_mask.jpg"

    if possible_mask_loc.exists():
        mask = cv2.imread(str(possible_mask_loc), cv2.IMREAD_GRAYSCALE)
        logger.info(f"Loaded mask from {possible_mask_loc}")
    else:
        logger.warning(f"Could not find mask at {possible_mask_loc}, trying to generate simple threshold (approx)")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Run the fitter to get corners
    result = fit_mask_rectangle(mask)
    corners = result.raw_corners if result.raw_corners is not None else result.corners

    if corners is None:
        logger.error("Fitter failed to find corners")
        return

    contour = result.contour
    mask_perimeter = cv2.arcLength(contour, True)
    logger.info(f"Mask Perimeter: {mask_perimeter}")
    estimated_start_eps = mask_perimeter * 0.008
    logger.info(f"Estimated start epsilon (0.008 * peri): {estimated_start_eps}")

    # Re-run collection logic manually to visualize
    contour_pts = np.squeeze(contour, axis=1).astype(np.float32)

    # Draw all contour points in faint red first
    for pt in contour_pts:
        cv2.circle(vis, (int(pt[0]), int(pt[1])), 1, (0, 0, 255), -1) # Red for all points initially

    # Draw fitted lines
    pts = corners.astype(int).reshape((-1, 1, 2))
    cv2.polylines(vis, [pts], True, (0, 255, 255), 2) # Yellow lines

    for idx in range(4):
        p0 = corners[idx]
        p1 = corners[(idx + 1) % 4]
        edge_vec = p1 - p0
        edge_len = float(np.linalg.norm(edge_vec))

        if edge_len < 1e-3:
            continue

        dir_vec = edge_vec / edge_len
        rel = contour_pts - p0
        proj = rel @ dir_vec
        perp = rel - np.outer(proj, dir_vec)
        dist = np.linalg.norm(perp, axis=1)

        # Logic from code:
        is_supported = (
            (proj >= -distance_threshold) &
            (proj <= edge_len + distance_threshold) &
            (dist <= distance_threshold)
        )

        supported_pts = contour_pts[is_supported]

        # Draw supported points in Blue
        for pt in supported_pts:
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 2, (255, 0, 0), -1) # Blue

    cv2.imwrite(output_path, vis)
    logger.info(f"Saved debug visualization to {output_path}")

if __name__ == "__main__":
    img_name = "drp.en_ko.in_house.selectstar_000119.jpg"
    dataset_root = Path("/workspaces/upstageailab-ocr-recsys-competition-ocr-2/data/datasets/images/train")
    img_path = dataset_root / img_name

    if not img_path.exists():
        logger.error(f"Image {img_path} not found")
        sys.exit(1)

    out_dir = Path("/workspaces/upstageailab-ocr-recsys-competition-ocr-2/experiment-tracker/experiments/20251128_005231_perspective_correction/artifacts")
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / "debug_edge_support_000119.jpg"

    visualize_edge_support_sampling(str(img_path), str(out_path))
