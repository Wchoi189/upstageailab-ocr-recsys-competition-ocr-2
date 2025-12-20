#!/usr/bin/env python3
"""
Visualize mask fitting with bounding boxes and corners for debugging data loss issues.

Usage:
    python visualize_mask_fitting.py --mask path/to/mask.jpg --original path/to/original.jpg --output path/to/output.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Setup paths
workspace_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(workspace_root))

# Import mask fitting
experiment_scripts = Path(__file__).parent
sys.path.insert(0, str(experiment_scripts))

from mask_only_edge_detector import fit_mask_rectangle, visualize_mask_fit


def create_detailed_visualization(
    original_img: np.ndarray,
    mask: np.ndarray,
    fit_result,
    output_path: Path,
):
    """
    Create comprehensive visualization showing:
    - Original image with overlay
    - Mask with fitted corners and bounding boxes
    - Side-by-side comparison
    """

    # Create visualization using the existing function
    vis_mask = visualize_mask_fit(
        mask,
        fit_result.corners,
        fit_result.contour,
        fit_result.hull,
    )

    # Overlay corners on original image
    vis_original = original_img.copy()

    if fit_result.corners is not None:
        # Draw fitted rectangle
        corners_int = fit_result.corners.astype(np.int32)
        cv2.polylines(vis_original, [corners_int], True, (0, 255, 0), 3)

        # Draw corner circles
        for i, corner in enumerate(corners_int):
            x, y = corner
            cv2.circle(vis_original, (x, y), 10, (0, 0, 255), -1)
            cv2.putText(
                vis_original,
                f"C{i}",
                (x + 15, y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        # Check for data loss (corners outside mask bounds)
        mask_coords = np.argwhere(mask > 0)
        if len(mask_coords) > 0:
            mask_min_y, mask_min_x = mask_coords.min(axis=0)
            mask_max_y, mask_max_x = mask_coords.max(axis=0)

            # Draw mask bounding box in yellow
            cv2.rectangle(
                vis_original,
                (mask_min_x, mask_min_y),
                (mask_max_x, mask_max_y),
                (0, 255, 255),
                2,
            )

            # Check each corner
            data_loss = []
            for i, (x, y) in enumerate(corners_int):
                if x < mask_min_x or x > mask_max_x or y < mask_min_y or y > mask_max_y:
                    data_loss.append(f"C{i}")
                    # Highlight problematic corner in red
                    cv2.circle(vis_original, (x, y), 15, (0, 0, 255), 3)

            # Add warning text if data loss detected
            if data_loss:
                warning = f"DATA LOSS: Corners {', '.join(data_loss)} outside mask"
                cv2.putText(
                    vis_original,
                    warning,
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

    # Add metrics text
    info_text = [
        f"Contour Area: {fit_result.contour_area:.0f}",
        f"Hull Area: {fit_result.hull_area:.0f}",
        f"Mask Area: {fit_result.mask_area:.0f}",
    ]

    if fit_result.reason:
        info_text.append(f"Reason: {fit_result.reason}")

    y_offset = vis_original.shape[0] - 20 - (len(info_text) * 30)
    for i, text in enumerate(info_text):
        cv2.putText(
            vis_original,
            text,
            (20, y_offset + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    # Resize for side-by-side comparison
    target_h = 1200

    scale_orig = target_h / vis_original.shape[0]
    scale_mask = target_h / vis_mask.shape[0]

    vis_original_resized = cv2.resize(vis_original, None, fx=scale_orig, fy=scale_orig)
    vis_mask_resized = cv2.resize(vis_mask, None, fx=scale_mask, fy=scale_mask)

    # Pad to same width
    max_w = max(vis_original_resized.shape[1], vis_mask_resized.shape[1])

    def pad_image(img, target_w):
        pad_left = (target_w - img.shape[1]) // 2
        pad_right = target_w - img.shape[1] - pad_left
        return cv2.copyMakeBorder(img, 80, 20, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    vis_original_padded = pad_image(vis_original_resized, max_w)
    vis_mask_padded = pad_image(vis_mask_resized, max_w)

    # Add labels
    cv2.putText(
        vis_original_padded,
        "Original with Fitted Corners",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 0),
        2,
    )

    cv2.putText(
        vis_mask_padded,
        "Mask Fitting Visualization",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 0),
        2,
    )

    # Stack vertically
    comparison = np.vstack([vis_original_padded, vis_mask_padded])

    # Save
    cv2.imwrite(str(output_path), comparison)
    print(f"✓ Visualization saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize mask fitting with bounding boxes and corners")
    parser.add_argument("--mask", type=Path, required=True, help="Path to mask image")
    parser.add_argument("--original", type=Path, required=True, help="Path to original image")
    parser.add_argument("--output", type=Path, required=True, help="Path to output visualization")
    parser.add_argument(
        "--use-regression",
        action="store_true",
        help="Use regression-based fitting",
    )
    parser.add_argument(
        "--use-dominant-extension",
        action="store_true",
        default=True,
        help="Use dominant extension strategy (default: True)",
    )

    args = parser.parse_args()

    # Read images
    mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
    original = cv2.imread(str(args.original))

    if mask is None:
        print(f"✗ ERROR: Could not read mask: {args.mask}")
        return 1

    if original is None:
        print(f"✗ ERROR: Could not read original: {args.original}")
        return 1

    print(f"\nProcessing: {args.mask.name}")
    print(f"Original: {args.original.name}")
    print(f"Mask shape: {mask.shape}")
    print(f"Original shape: {original.shape}")

    # Fit rectangle
    print("\nFitting rectangle...")
    print(f"  Use regression: {args.use_regression}")
    print(f"  Use dominant extension: {args.use_dominant_extension}")

    fit_result = fit_mask_rectangle(
        mask,
        use_regression=args.use_regression,
        use_dominant_extension=args.use_dominant_extension,
    )

    if fit_result.corners is None:
        print("✗ ERROR: Rectangle fitting failed")
        print(f"  Reason: {fit_result.reason}")
        return 1

    print("✓ Rectangle fitted successfully")
    print(f"  Corners: {fit_result.corners.tolist()}")

    # Create visualization
    args.output.parent.mkdir(parents=True, exist_ok=True)
    create_detailed_visualization(original, mask, fit_result, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
