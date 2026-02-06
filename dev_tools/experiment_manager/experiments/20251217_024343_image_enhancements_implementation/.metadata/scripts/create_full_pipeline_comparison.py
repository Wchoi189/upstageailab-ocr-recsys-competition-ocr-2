#!/usr/bin/env python3
"""
Create side-by-side comparison showing full enhancement pipeline.

Shows: Original → After Background Norm → After Deskewing
"""

from pathlib import Path

import cv2
import numpy as np


def create_comparison(original_path: Path, bg_norm_path: Path, deskew_path: Path, output_path: Path):
    """Create 3-panel comparison image."""

    # Read images
    original = cv2.imread(str(original_path))
    bg_norm = cv2.imread(str(bg_norm_path))
    deskew = cv2.imread(str(deskew_path))

    if original is None or bg_norm is None or deskew is None:
        print(f"ERROR: Could not read images for {original_path.stem}")
        return

    # Resize all to same height
    target_h = min(original.shape[0], bg_norm.shape[0], deskew.shape[0])
    target_h = min(target_h, 1200)  # Cap at 1200px

    images = [original, bg_norm, deskew]
    labels = ["1. Original (After Perspective)", "2. + Background Norm", "3. + Deskewing"]

    resized = []
    for img in images:
        scale = target_h / img.shape[0]
        resized_img = cv2.resize(img, None, fx=scale, fy=scale)
        resized.append(resized_img)

    # Pad to same width
    max_w = max(img.shape[1] for img in resized)
    padded = []

    for img, label in zip(resized, labels, strict=False):
        # Pad width
        pad_left = (max_w - img.shape[1]) // 2
        pad_right = max_w - img.shape[1] - pad_left
        padded_img = cv2.copyMakeBorder(img, 80, 20, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        # Add label at top
        cv2.putText(padded_img, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

        padded.append(padded_img)

    # Stack vertically
    comparison = np.vstack(padded)

    # Save
    cv2.imwrite(str(output_path), comparison)
    print(f"✓ Created: {output_path.name}")


def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    original_dir = Path("data/zero_prediction_worst_performers")
    bg_norm_dir = base_dir / "outputs/demo_pipeline/step1_bg_norm"
    deskew_dir = base_dir / "outputs/demo_pipeline/step2_deskew"
    output_dir = base_dir / "outputs/demo_pipeline/full_pipeline_comparisons"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each test image
    test_images = [
        "drp.en_ko.in_house.selectstar_000699.jpg",
        "drp.en_ko.in_house.selectstar_000712.jpg",
        "drp.en_ko.in_house.selectstar_000732.jpg",
        "drp.en_ko.in_house.selectstar_001007.jpg",
        "drp.en_ko.in_house.selectstar_001012.jpg",
        "drp.en_ko.in_house.selectstar_001161.jpg",
    ]

    for img_name in test_images:
        create_comparison(
            original_path=original_dir / img_name,
            bg_norm_path=bg_norm_dir / img_name,
            deskew_path=deskew_dir / img_name,
            output_path=output_dir / f"pipeline_{img_name}",
        )

    print(f"\n✅ All comparisons saved to: {output_dir}")
    print("   View them to see the full enhancement pipeline in action!")


if __name__ == "__main__":
    main()
