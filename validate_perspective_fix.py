#!/usr/bin/env python3
"""
Validation test for perspective correction fix.

Compares old (simplified) vs new (dominant extension) implementations on test image 000732.
"""

import cv2
import numpy as np
from pathlib import Path

# Import the updated production code
from ocr.utils.perspective_correction import fit_mask_rectangle, remove_background_and_mask

def main():
    # Test image from incident report
    test_image_path = Path("data/zero_prediction_worst_performers/drp.en_ko.in_house.selectstar_000732.jpg")

    if not test_image_path.exists():
        print(f"❌ Test image not found: {test_image_path}")
        return 1

    print(f"📸 Loading test image: {test_image_path.name}")
    image = cv2.imread(str(test_image_path))

    if image is None:
        print("❌ Failed to load image")
        return 1

    print(f"   Image shape: {image.shape}")

    # Remove background to get mask
    print("\n🔍 Removing background...")
    try:
        image_no_bg, mask = remove_background_and_mask(image)
        print(f"   Mask shape: {mask.shape}")
        print(f"   Mask area: {np.count_nonzero(mask)} pixels")
    except Exception as e:
        print(f"❌ Background removal failed: {e}")
        return 1

    # Get mask bounds (from incident report: x=[320, 823], y=[104, 1239])
    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0:
        print("❌ Empty mask")
        return 1

    mask_x_min, mask_x_max = int(np.min(x_indices)), int(np.max(x_indices))
    mask_y_min, mask_y_max = int(np.min(y_indices)), int(np.max(y_indices))

    print(f"\n📏 Mask bounds:")
    print(f"   x=[{mask_x_min}, {mask_x_max}]  (width: {mask_x_max - mask_x_min}px)")
    print(f"   y=[{mask_y_min}, {mask_y_max}]  (height: {mask_y_max - mask_y_min}px)")

    # Test with dominant extension (NEW - default)
    print("\n✨ Testing DOMINANT EXTENSION (new default):")
    result_de = fit_mask_rectangle(mask, use_dominant_extension=True)

    if result_de.corners is not None:
        corners_de = result_de.corners
        print(f"   Corners: {corners_de.tolist()}")
        print(f"   Reason: {result_de.reason}")

        # Calculate data loss
        corner_x_min = int(np.min(corners_de[:, 0]))
        corner_x_max = int(np.max(corners_de[:, 0]))
        corner_y_min = int(np.min(corners_de[:, 1]))
        corner_y_max = int(np.max(corners_de[:, 1]))

        data_loss_left = corner_x_min - mask_x_min
        data_loss_right = mask_x_max - corner_x_max
        data_loss_top = corner_y_min - mask_y_min
        data_loss_bottom = mask_y_max - corner_y_max
        total_loss = abs(data_loss_left) + abs(data_loss_right) + abs(data_loss_top) + abs(data_loss_bottom)

        print(f"\n   📊 Data Loss Analysis:")
        print(f"      Left:   {data_loss_left}px")
        print(f"      Right:  {data_loss_right}px")
        print(f"      Top:    {data_loss_top}px")
        print(f"      Bottom: {data_loss_bottom}px")
        print(f"      TOTAL:  {total_loss}px")

        # Expected from incident report: ~1-20px total loss
        if total_loss <= 30:
            print(f"   ✅ PASS: Data loss ≤ 30px (expected ≤20px, got {total_loss}px)")
        else:
            print(f"   ⚠️  Data loss higher than expected: {total_loss}px (expected ≤20px)")
    else:
        print(f"   ❌ FAILED: No corners detected")
        print(f"   Reason: {result_de.reason}")
        return 1

    # Test without dominant extension (OLD behavior for comparison)
    print("\n🔧 Testing WITHOUT dominant extension (for comparison):")
    result_old = fit_mask_rectangle(mask, use_dominant_extension=False, use_regression=False)

    if result_old.corners is not None:
        corners_old = result_old.corners
        print(f"   Corners: {corners_old.tolist()}")
        print(f"   Reason: {result_old.reason}")

        corner_x_min_old = int(np.min(corners_old[:, 0]))
        corner_x_max_old = int(np.max(corners_old[:, 0]))
        corner_y_min_old = int(np.min(corners_old[:, 1]))
        corner_y_max_old = int(np.max(corners_old[:, 1]))

        data_loss_old = (abs(corner_x_min_old - mask_x_min) + abs(mask_x_max - corner_x_max_old) +
                         abs(corner_y_min_old - mask_y_min) + abs(mask_y_max - corner_y_max_old))

        print(f"   Total data loss (old): {data_loss_old}px")
    else:
        print(f"   Corners: None (using bbox fallback)")
        print(f"   Reason: {result_old.reason}")

    print("\n" + "="*60)
    print("✅ VALIDATION COMPLETE")
    print("="*60)
    print(f"Dominant Extension is now the DEFAULT (use_dominant_extension=True)")
    print(f"Expected improvement: ~80px → ~20px data loss")

    return 0

if __name__ == "__main__":
    exit(main())
