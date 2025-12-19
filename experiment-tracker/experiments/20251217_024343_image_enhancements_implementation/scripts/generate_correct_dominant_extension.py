#!/usr/bin/env python3
"""
Generate corrected image using the ACTUAL dominant extension algorithm from experiments.
"""
import sys

import cv2
import numpy as np

# Use experiment implementation, not production
sys.path.insert(0, 'experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/scripts')
from mask_only_edge_detector import fit_mask_rectangle
from perspective_transformer import four_point_transform

# Read original and mask
original = cv2.imread('data/datasets/images/test/drp.en_ko.in_house.selectstar_000732.jpg')
mask = cv2.imread('experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/outputs/full_pipeline_correct/drp.en_ko.in_house.selectstar_000732_step1_mask.jpg', cv2.IMREAD_GRAYSCALE)

print("Generating corrected image using EXPERIMENT dominant extension...")

# Fit with dominant extension (TRUE algorithm)
result = fit_mask_rectangle(mask, use_regression=False, use_dominant_extension=True)

if result.corners is None:
    print(f"✗ ERROR: Fitting failed - {result.reason}")
    sys.exit(1)

print(f"✓ Fitted corners: {result.corners.tolist()}")

# Apply perspective transform
corrected = four_point_transform(original, result.corners)

# Save
output_path = 'experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/outputs/full_pipeline_correct/000732_corrected_DOMINANT_EXTENSION.jpg'
cv2.imwrite(output_path, corrected)

print(f"✓ Saved corrected image: {output_path}")
print(f"  Output shape: {corrected.shape}")

# Also generate with standard strategy for comparison
result_std = fit_mask_rectangle(mask, use_regression=False, use_dominant_extension=False)
if result_std.corners is not None:
    corrected_std = four_point_transform(original, result_std.corners)
    output_std = 'experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/outputs/full_pipeline_correct/000732_corrected_STANDARD.jpg'
    cv2.imwrite(output_std, corrected_std)
    print(f"✓ Saved standard strategy: {output_std}")
    print(f"  Output shape: {corrected_std.shape}")

# Compare corner differences
mask_coords = np.argwhere(mask > 0)
mask_min_y, mask_min_x = mask_coords.min(axis=0)
mask_max_y, mask_max_x = mask_coords.max(axis=0)

print(f"\nMask bounds: x=[{mask_min_x}, {mask_max_x}], y=[{mask_min_y}, {mask_max_y}]")
print(f"Dominant extension corners: {result.corners.tolist()}")
if result_std.corners is not None:
    print(f"Standard strategy corners: {result_std.corners.tolist()}")

# Calculate data loss for dominant extension
dom_min_x = result.corners[:, 0].min()
dom_max_x = result.corners[:, 0].max()
dom_min_y = result.corners[:, 1].min()
dom_max_y = result.corners[:, 1].max()

loss_left = int(dom_min_x - mask_min_x)
loss_right = int(mask_max_x - dom_max_x)
loss_top = int(dom_min_y - mask_min_y)
loss_bottom = int(mask_max_y - dom_max_y)

print("\nDominant extension data loss:")
print(f"  Left: {loss_left}px, Right: {loss_right}px, Top: {loss_top}px, Bottom: {loss_bottom}px")
print(f"  Total: {loss_left + loss_right + loss_top + loss_bottom}px")
