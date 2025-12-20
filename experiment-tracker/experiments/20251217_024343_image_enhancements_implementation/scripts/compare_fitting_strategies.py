#!/usr/bin/env python3
"""
Compare different fitting strategies side-by-side to show data loss.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mask_only_edge_detector import fit_mask_rectangle


def compare_strategies(mask_path: str, original_path: str, output_path: str):
    """Compare standard vs dominant_extension fitting."""

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    original = cv2.imread(original_path)

    # Get mask bounds
    mask_coords = np.argwhere(mask > 0)
    mask_min_y, mask_min_x = mask_coords.min(axis=0)
    mask_max_y, mask_max_x = mask_coords.max(axis=0)

    # Fit with standard strategy
    result_standard = fit_mask_rectangle(mask, use_regression=False, use_dominant_extension=False)

    # Fit with dominant extension
    result_dominant = fit_mask_rectangle(mask, use_regression=False, use_dominant_extension=True)

    # Create overlays
    def create_overlay(img, corners, label, color):
        vis = img.copy()

        # Draw mask bounding box (yellow)
        cv2.rectangle(vis, (mask_min_x, mask_min_y), (mask_max_x, mask_max_y), (0, 255, 255), 2)

        # Draw fitted rectangle
        if corners is not None:
            corners_int = corners.astype(np.int32)
            cv2.polylines(vis, [corners_int], True, color, 3)

            # Draw corners
            for i, (x, y) in enumerate(corners_int):
                cv2.circle(vis, (x, y), 8, (0, 0, 255), -1)
                cv2.putText(vis, f"C{i}", (x + 12, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Calculate data loss
            fitted_min_x = corners[:, 0].min()
            fitted_max_x = corners[:, 0].max()
            fitted_min_y = corners[:, 1].min()
            fitted_max_y = corners[:, 1].max()

            loss_left = int(fitted_min_x - mask_min_x)
            loss_right = int(mask_max_x - fitted_max_x)
            loss_top = int(fitted_min_y - mask_min_y)
            loss_bottom = int(mask_max_y - fitted_max_y)

            total_loss = loss_left + loss_right + loss_top + loss_bottom

            # Add label with data loss
            cv2.putText(vis, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(vis, f"Data Loss: {total_loss}px total", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(
                vis, f"L:{loss_left} R:{loss_right} T:{loss_top} B:{loss_bottom}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )

        return vis

    vis_standard = create_overlay(original, result_standard.corners, "Standard Strategy", (0, 255, 0))
    vis_dominant = create_overlay(original, result_dominant.corners, "Dominant Extension (CURRENT)", (255, 0, 255))

    # Stack horizontally
    target_h = 800
    scale = target_h / vis_standard.shape[0]

    vis_standard_resized = cv2.resize(vis_standard, None, fx=scale, fy=scale)
    vis_dominant_resized = cv2.resize(vis_dominant, None, fx=scale, fy=scale)

    comparison = np.hstack([vis_standard_resized, vis_dominant_resized])

    cv2.imwrite(output_path, comparison)
    print(f"✓ Strategy comparison saved: {output_path}")

    print(f"\n{'=' * 60}")
    print("COMPARISON RESULTS:")
    print(f"{'=' * 60}")
    print("Standard Strategy:")
    print(f"  Corners: {result_standard.corners.tolist() if result_standard.corners is not None else 'FAILED'}")
    if result_standard.reason:
        print(f"  Reason: {result_standard.reason}")

    print("\nDominant Extension Strategy (CURRENT):")
    print(f"  Corners: {result_dominant.corners.tolist() if result_dominant.corners is not None else 'FAILED'}")
    if result_dominant.reason:
        print(f"  Reason: {result_dominant.reason}")

    if result_standard.corners is not None and result_dominant.corners is not None:
        diff = np.abs(result_dominant.corners - result_standard.corners)
        print("\nDifferences (dominant - standard):")
        for i in range(4):
            print(f"  C{i}: {diff[i]} pixels")


if __name__ == "__main__":
    compare_strategies(
        mask_path="experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/outputs/full_pipeline_correct/drp.en_ko.in_house.selectstar_000732_step1_mask.jpg",
        original_path="data/datasets/images/test/drp.en_ko.in_house.selectstar_000732.jpg",
        output_path="experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/outputs/full_pipeline_correct/000732_STRATEGY_COMPARISON.jpg",
    )
