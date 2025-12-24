#!/usr/bin/env python3
"""
Quick test script for perspective correction in inference pipeline.

Usage:
    python scripts/test_perspective_inference.py --image path/to/image.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2

# Setup paths
workspace_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(workspace_root))

from ocr.inference.engine import InferenceEngine


def test_perspective_correction(
    image_path: str,
    checkpoint_path: str,
    config_path: str | None = None,
    output_dir: str | None = None,
):
    """Test perspective correction in inference pipeline."""

    print("=" * 70)
    print("Testing Perspective Correction in Inference Pipeline")
    print("=" * 70)

    # Initialize engine
    print("\n1. Loading model...")
    print(f"   Checkpoint: {checkpoint_path}")
    engine = InferenceEngine()

    if not engine.load_model(checkpoint_path, config_path):
        print("   ✗ Failed to load model")
        return False
    print("   ✓ Model loaded successfully")

    # Test 1: Without perspective correction
    print("\n2. Running inference WITHOUT perspective correction...")
    result_no_persp = engine.predict_image(
        image_path=image_path,
        enable_perspective_correction=False,
        return_preview=True,
    )

    if result_no_persp:
        num_detections = len(result_no_persp.get('polygons', []))
        print(f"   ✓ Detected {num_detections} text regions")
    else:
        print("   ✗ Inference failed")
        return False

    # Test 2: With perspective correction
    print("\n3. Running inference WITH perspective correction...")
    result_with_persp = engine.predict_image(
        image_path=image_path,
        enable_perspective_correction=True,
        perspective_display_mode="corrected",
        return_preview=True,
    )

    if result_with_persp:
        num_detections = len(result_with_persp.get('polygons', []))
        print(f"   ✓ Detected {num_detections} text regions")
    else:
        print("   ✗ Inference failed")
        return False

    # Test 3: With full enhancement pipeline
    print("\n4. Running inference WITH full enhancement pipeline...")
    result_full = engine.predict_image(
        image_path=image_path,
        enable_perspective_correction=True,
        enable_background_normalization=True,
        perspective_display_mode="corrected",
        return_preview=True,
    )

    if result_full:
        num_detections = len(result_full.get('polygons', []))
        print(f"   ✓ Detected {num_detections} text regions")
    else:
        print("   ✗ Inference failed")
        return False

    # Save outputs if requested
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n5. Saving visualization outputs to {output_dir}...")

        # Save preview images
        if 'preview_image' in result_no_persp:
            cv2.imwrite(
                str(output_path / "result_no_perspective.jpg"),
                result_no_persp['preview_image']
            )
            print("   ✓ Saved: result_no_perspective.jpg")

        if 'preview_image' in result_with_persp:
            cv2.imwrite(
                str(output_path / "result_with_perspective.jpg"),
                result_with_persp['preview_image']
            )
            print("   ✓ Saved: result_with_perspective.jpg")

        if 'preview_image' in result_full:
            cv2.imwrite(
                str(output_path / "result_full_enhancement.jpg"),
                result_full['preview_image']
            )
            print("   ✓ Saved: result_full_enhancement.jpg")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Without perspective:     {len(result_no_persp.get('polygons', []))} detections")
    print(f"With perspective:        {len(result_with_persp.get('polygons', []))} detections")
    print(f"With full enhancement:   {len(result_full.get('polygons', []))} detections")
    print()

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test perspective correction in inference pipeline"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/experiments/train/ocr/pan_resnet18_add_polygons_canonical/20241019_0033_00/checkpoints/epoch-18_step-001957.ckpt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (optional)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for visualizations (optional)",
    )

    args = parser.parse_args()

    # Check if image exists
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        return 1

    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        print("Please specify a valid checkpoint with --checkpoint")
        return 1

    # Run test
    success = test_perspective_correction(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
