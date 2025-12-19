#!/usr/bin/env python3
"""
Full enhancement pipeline demonstration.

Pipeline:
1. Raw image → rembg background removal → mask generation
2. Mask → perspective correction (4-point transform)
3. Corrected image → background normalization (gray-world)
4. Normalized image → text deskewing (Hough lines)

Usage:
    python full_pipeline_demo.py --input data/test_images --output outputs/full_demo
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Setup paths
workspace_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(workspace_root))

from ocr.utils.perspective_correction import correct_perspective_from_mask, remove_background_and_mask

# Import enhancement modules from experiment scripts
experiment_scripts = Path(__file__).parent
sys.path.insert(0, str(experiment_scripts))

from background_normalization import BackgroundNormalizer
from text_deskewing import TextDeskewer


def process_image(input_path: Path, output_dir: Path, save_intermediates: bool = True):
    """Process single image through full pipeline."""

    img_name = input_path.stem
    print(f"\n{'='*60}")
    print(f"Processing: {input_path.name}")
    print(f"{'='*60}")

    # Read original
    original = cv2.imread(str(input_path))
    if original is None:
        print("  ✗ ERROR: Could not read image")
        return None

    results = {
        "image": str(input_path),
        "original_shape": list(original.shape),
        "steps": {},
    }

    # Step 1: Rembg + Mask generation
    print("  Step 1: Background removal + mask generation...", end=" ")
    t0 = time.time()
    try:
        no_bg_img, mask = remove_background_and_mask(original)
        step1_time = (time.time() - t0) * 1000
        print(f"✓ ({step1_time:.1f}ms)")

        results["steps"]["1_rembg"] = {
            "time_ms": round(step1_time, 2),
            "mask_shape": list(mask.shape),
        }

        if save_intermediates:
            cv2.imwrite(str(output_dir / f"{img_name}_step1_no_bg.jpg"), no_bg_img)
            cv2.imwrite(str(output_dir / f"{img_name}_step1_mask.jpg"), mask)
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return None

    # Step 2: Perspective correction
    print("  Step 2: Perspective correction (mask-based)...", end=" ")
    t0 = time.time()
    try:
        result = correct_perspective_from_mask(original, mask)
        # correct_perspective_from_mask returns (warped_img, fit_result)
        if isinstance(result, tuple):
            corrected_img, fit_result = result
        else:
            corrected_img = result
        step2_time = (time.time() - t0) * 1000
        print(f"✓ ({step2_time:.1f}ms)")

        results["steps"]["2_perspective"] = {
            "time_ms": round(step2_time, 2),
            "output_shape": list(corrected_img.shape),
        }

        if save_intermediates:
            cv2.imwrite(str(output_dir / f"{img_name}_step2_corrected.jpg"), corrected_img)
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return None

    # Step 3: Background normalization
    print("  Step 3: Background normalization (gray-world)...", end=" ")
    normalizer = BackgroundNormalizer(method="gray-world")
    normalized_img, norm_metrics = normalizer.normalize(corrected_img)
    print(f"✓ ({norm_metrics['processing_time_ms']:.1f}ms)")

    results["steps"]["3_background_norm"] = {
        "time_ms": norm_metrics["processing_time_ms"],
        "metrics": norm_metrics,
    }

    if save_intermediates:
        cv2.imwrite(str(output_dir / f"{img_name}_step3_normalized.jpg"), normalized_img)

    # Step 4: Text deskewing
    print("  Step 4: Text deskewing (Hough lines)...", end=" ")
    deskewer = TextDeskewer(method="hough")
    deskewed_img, deskew_metrics = deskewer.deskew(normalized_img)
    print(f"✓ ({deskew_metrics['processing_time_ms']:.1f}ms)")
    print(f"    Angle: {deskew_metrics['detected_angle_degrees']:.2f}°")

    results["steps"]["4_deskew"] = {
        "time_ms": deskew_metrics["processing_time_ms"],
        "angle_degrees": deskew_metrics["detected_angle_degrees"],
    }

    if save_intermediates:
        cv2.imwrite(str(output_dir / f"{img_name}_step4_deskewed.jpg"), deskewed_img)

    # Calculate total time
    total_time = sum(step.get("time_ms", 0) for step in results["steps"].values())
    results["total_time_ms"] = round(total_time, 2)

    print(f"  ✅ Complete! Total time: {total_time:.1f}ms")

    # Create comparison
    create_comparison(
        original=original,
        step2_corrected=corrected_img,
        step3_normalized=normalized_img,
        step4_deskewed=deskewed_img,
        output_path=output_dir / f"{img_name}_FULL_PIPELINE.jpg",
    )

    return results


def create_comparison(original, step2_corrected, step3_normalized, step4_deskewed, output_path):
    """Create 4-panel vertical comparison."""

    images = [original, step2_corrected, step3_normalized, step4_deskewed]
    labels = [
        "1. Original (Raw)",
        "2. + Perspective Correction (rembg mask)",
        "3. + Background Normalization (gray-world)",
        "4. + Text Deskewing (Hough lines)",
    ]

    # Resize to same height (cap at 1000px)
    target_h = min(img.shape[0] for img in images)
    target_h = min(target_h, 1000)

    resized = []
    for img in images:
        scale = target_h / img.shape[0]
        resized_img = cv2.resize(img, None, fx=scale, fy=scale)
        resized.append(resized_img)

    # Pad to same width
    max_w = max(img.shape[1] for img in resized)
    padded = []

    for img, label in zip(resized, labels):
        pad_left = (max_w - img.shape[1]) // 2
        pad_right = max_w - img.shape[1] - pad_left
        padded_img = cv2.copyMakeBorder(
            img, 80, 20, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )

        # Add label
        cv2.putText(
            padded_img, label,
            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2
        )

        padded.append(padded_img)

    # Stack vertically
    comparison = np.vstack(padded)
    cv2.imwrite(str(output_path), comparison)
    print(f"  📊 Comparison saved: {output_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Full enhancement pipeline demonstration")
    parser.add_argument("--input", type=Path, required=True, help="Input image or directory")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--no-intermediates", action="store_true", help="Skip intermediate images")

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Process images
    if args.input.is_dir():
        image_files = list(args.input.glob("*.jpg")) + list(args.input.glob("*.png"))
        image_files = [f for f in image_files if "_mask" not in f.name]
        image_files.sort()

        print(f"\nProcessing {len(image_files)} images from {args.input}")
        print(f"Output directory: {args.output}")

        all_results = []
        for img_path in image_files:
            result = process_image(img_path, args.output, save_intermediates=not args.no_intermediates)
            if result:
                all_results.append(result)

        # Save aggregate results
        results_file = args.output / "pipeline_results.json"
        with open(results_file, "w") as f:
            json.dump({
                "total_images": len(all_results),
                "avg_total_time_ms": round(sum(r["total_time_ms"] for r in all_results) / len(all_results), 2),
                "results": all_results,
            }, f, indent=2)

        print(f"\n{'='*60}")
        print(f"✅ Processed {len(all_results)} images")
        print(f"📄 Results saved to: {results_file}")
        print(f"{'='*60}")

    else:
        # Single image
        result = process_image(args.input, args.output, save_intermediates=not args.no_intermediates)

        if result:
            results_file = args.output / f"{args.input.stem}_results.json"
            with open(results_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n📄 Results saved to: {results_file}")


if __name__ == "__main__":
    main()
