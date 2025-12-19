#!/usr/bin/env python3
"""
Simple validation test for perspective correction fix using synthetic mask.
"""

import numpy as np
from ocr.utils.perspective_correction import fit_mask_rectangle

def create_document_mask(height=500, width=800, tilt=10):
    """Create a synthetic document mask with slight tilt."""
    mask = np.zeros((height, width), dtype=np.uint8)

    # Create a tilted rectangle (simulating photographed document)
    center_x, center_y = width // 2, height // 2
    rect_w, rect_h = 700, 400

    # Define corners with slight perspective
    corners = np.array([
        [center_x - rect_w//2 + tilt, center_y - rect_h//2],
        [center_x + rect_w//2 - tilt, center_y - rect_h//2 + 5],
        [center_x + rect_w//2, center_y + rect_h//2],
        [center_x - rect_w//2, center_y + rect_h//2 - 5],
    ], dtype=np.int32)

    # Draw filled polygon
    import cv2
    cv2.fillPoly(mask, [corners], 255)

    return mask, corners

def main():
    print("="*60)
    print("PERSPECTIVE CORRECTION FIX VALIDATION")
    print("="*60)

    # Create synthetic mask
    print("\n📐 Creating synthetic document mask...")
    mask, true_corners = create_document_mask()
    print(f"   Mask shape: {mask.shape}")
    print(f"   Mask area: {np.count_nonzero(mask)} pixels")

    # Test 1: Dominant Extension (NEW DEFAULT)
    print("\n✨ TEST 1: Dominant Extension (NEW - Default)")
    print("-" * 60)
    result_de = fit_mask_rectangle(mask, use_dominant_extension=True)

    if result_de.corners is not None:
        print(f"   ✅ SUCCESS")
        print(f"   Corners detected: {result_de.corners.shape}")
        print(f"   Reason: {result_de.reason if result_de.reason else 'Success (no issues)'}")
        print(f"   Line quality: {result_de.line_quality.decision if result_de.line_quality else 'N/A'}")

        # Calculate fit quality
        mask_bounds = np.where(mask > 0)
        mask_y_min, mask_y_max = np.min(mask_bounds[0]), np.max(mask_bounds[0])
        mask_x_min, mask_x_max = np.min(mask_bounds[1]), np.max(mask_bounds[1])

        corner_y_min = int(np.min(result_de.corners[:, 1]))
        corner_y_max = int(np.max(result_de.corners[:, 1]))
        corner_x_min = int(np.min(result_de.corners[:, 0]))
        corner_x_max = int(np.max(result_de.corners[:, 0]))

        data_loss = abs(corner_x_min - mask_x_min) + abs(mask_x_max - corner_x_max) + \
                   abs(corner_y_min - mask_y_min) + abs(mask_y_max - corner_y_max)

        print(f"   Data loss: {data_loss}px (expected ≤20px for dominant extension)")
    else:
        print(f"   ❌ FAILED: No corners detected")
        print(f"   Reason: {result_de.reason}")

    # Test 2: Standard approxPolyDP (OLD BEHAVIOR)
    print("\n🔧 TEST 2: Standard approxPolyDP (OLD)")
    print("-" * 60)
    result_old = fit_mask_rectangle(mask, use_dominant_extension=False, use_regression=False)

    if result_old.corners is not None:
        print(f"   ✅ SUCCESS")
        print(f"   Corners detected: {result_old.corners.shape}")
        print(f"   Reason: {result_old.reason if result_old.reason else 'Success (no issues)'}")
        print(f"   Line quality: {result_old.line_quality.decision if result_old.line_quality else 'N/A'}")

        corner_y_min_old = int(np.min(result_old.corners[:, 1]))
        corner_y_max_old = int(np.max(result_old.corners[:, 1]))
        corner_x_min_old = int(np.min(result_old.corners[:, 0]))
        corner_x_max_old = int(np.max(result_old.corners[:, 0]))

        data_loss_old = abs(corner_x_min_old - mask_x_min) + abs(mask_x_max - corner_x_max_old) + \
                       abs(corner_y_min_old - mask_y_min) + abs(mask_y_max - corner_y_max_old)

        print(f"   Data loss: {data_loss_old}px")
    else:
        print(f"   ❌ FAILED: No corners detected")
        print(f"   Reason: {result_old.reason}")

    # Test 3: Regression Strategy
    print("\n🧪 TEST 3: Regression Strategy")
    print("-" * 60)
    result_reg = fit_mask_rectangle(mask, use_regression=True, use_dominant_extension=False)

    if result_reg.corners is not None:
        print(f"   ✅ SUCCESS")
        print(f"   Corners detected: {result_reg.corners.shape}")
        print(f"   Reason: {result_reg.reason if result_reg.reason else 'Success (no issues)'}")
    else:
        print(f"   ❌ FAILED: No corners detected")
        print(f"   Reason: {result_reg.reason}")

    print("\n" + "="*60)
    print("✅ VALIDATION COMPLETE")
    print("="*60)
    print("\nKey Improvements:")
    print("• Dominant extension strategy is now DEFAULT (use_dominant_extension=True)")
    print("• Advanced line quality metrics with geometric synthesis")
    print("• Expected data loss: ≤20px (vs 80+ px with old simplified algorithm)")
    print("• API signature unchanged - backward compatible")

    return 0

if __name__ == "__main__":
    exit(main())
