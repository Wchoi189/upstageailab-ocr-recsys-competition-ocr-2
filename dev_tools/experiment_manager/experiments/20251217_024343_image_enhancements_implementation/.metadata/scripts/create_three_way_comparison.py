#!/usr/bin/env python3
"""
Create 3-way comparison: Production vs Experiment Dominant Extension vs Standard
"""

import cv2
import numpy as np

# Load all three corrected images
prod = cv2.imread(
    "experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/outputs/full_pipeline_correct/drp.en_ko.in_house.selectstar_000732_step2_corrected.jpg"
)
dom = cv2.imread(
    "experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/outputs/full_pipeline_correct/000732_corrected_DOMINANT_EXTENSION.jpg"
)
std = cv2.imread(
    "experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/outputs/full_pipeline_correct/000732_corrected_STANDARD.jpg"
)

# Resize to same height
target_h = 800


def resize_and_label(img, label, data_loss_px):
    scale = target_h / img.shape[0]
    resized = cv2.resize(img, None, fx=scale, fy=scale)

    # Add white border at top for label
    labeled = cv2.copyMakeBorder(resized, 100, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    # Add label
    cv2.putText(labeled, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(
        labeled,
        f"Data Loss: {data_loss_px}px",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255) if data_loss_px > 10 else (0, 128, 0),
        2,
    )

    return labeled


prod_labeled = resize_and_label(prod, "Production (WRONG)", 82)  # 27+20+18+17
dom_labeled = resize_and_label(dom, "Dominant Extension (CORRECT)", 1)
std_labeled = resize_and_label(std, "Standard Strategy", 0)

# Stack horizontally
comparison = np.hstack([prod_labeled, dom_labeled, std_labeled])

output_path = "experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/outputs/full_pipeline_correct/000732_THREE_WAY_COMPARISON.jpg"
cv2.imwrite(output_path, comparison)

print(f"✓ 3-way comparison saved: {output_path}")
print("\nSummary:")
print("  Production (used by full_pipeline_demo.py): 82px data loss - WRONG ALGORITHM")
print("  Dominant Extension (experiment algorithm): 1px data loss - CORRECT")
print("  Standard Strategy: 0px data loss - MOST ACCURATE")
