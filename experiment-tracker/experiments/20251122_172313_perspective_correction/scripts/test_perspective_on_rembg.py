#!/usr/bin/env python3
"""
Isolated test for perspective correction on rembg-processed images.

This script:
1. Processes images with rembg (background removal)
2. Applies perspective correction to the processed images
3. Saves intermediate results for comparison
4. Evaluates effectiveness with metrics

Usage:
    # Test on rembg-processed images
    python scripts/test_perspective_on_rembg.py --input-dir data/datasets/images/train --output-dir outputs/perspective_test

    # Test on existing rembg outputs
    python scripts/test_perspective_on_rembg.py --rembg-dir outputs/large_workload_test --output-dir outputs/perspective_test
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import rembg
try:
    import sys
    from pathlib import Path

    script_dir = Path(__file__).parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    from optimized_rembg import GPU_AVAILABLE, REMBG_AVAILABLE, OptimizedBackgroundRemover
except ImportError as e:
    logger.error(f"Failed to import optimized_rembg: {e}")
    REMBG_AVAILABLE = False
    GPU_AVAILABLE = False

# Import perspective correction
try:
    from ocr.datasets.preprocessing.detector import DocumentDetector
    from ocr.datasets.preprocessing.perspective import PerspectiveCorrector

    PERSPECTIVE_AVAILABLE = True
except ImportError:
    PERSPECTIVE_AVAILABLE = False
    logger.warning("Perspective correction not available. Install dependencies.")


def calculate_image_metrics(image: np.ndarray) -> dict[str, float]:
    """Calculate metrics to evaluate image quality."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Edge strength (variance of Laplacian)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    edge_strength = laplacian.var()

    # Contrast (standard deviation)
    contrast = gray.std()

    # Brightness (mean)
    brightness = gray.mean()

    # Sharpness (gradient magnitude)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sharpness = np.sqrt(grad_x**2 + grad_y**2).mean()

    return {
        "edge_strength": float(edge_strength),
        "contrast": float(contrast),
        "brightness": float(brightness),
        "sharpness": float(sharpness),
    }


def detect_corners_simple(image: np.ndarray) -> np.ndarray | None:
    """
    Simple corner detection for perspective correction.

    This is a fallback if DocumentDetector is not available.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Dilate edges
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Find largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Approximate to polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) == 4:
        corners = approx.reshape(4, 2)
    else:
        # Use bounding box
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        corners = box.reshape(4, 2)

    # Order corners: top-left, top-right, bottom-right, bottom-left
    corners_ordered = np.zeros((4, 2), dtype=np.float32)
    s = corners.sum(axis=1)
    corners_ordered[0] = corners[np.argmin(s)]  # top-left
    corners_ordered[2] = corners[np.argmax(s)]  # bottom-right

    diff = np.diff(corners, axis=1)
    corners_ordered[1] = corners[np.argmin(diff)]  # top-right
    corners_ordered[3] = corners[np.argmax(diff)]  # bottom-left

    return corners_ordered.astype(np.float32)


def test_perspective_on_rembg(
    image_path: Path,
    output_dir: Path,
    rembg_remover: OptimizedBackgroundRemover | None = None,
    use_gpu: bool = False,
    skip_rembg: bool = False,
) -> dict[str, Any]:
    """
    Test perspective correction on rembg-processed image.

    Args:
        image_path: Path to input image
        output_dir: Output directory for results
        rembg_remover: Optional pre-initialized rembg remover
        use_gpu: Whether to use GPU for rembg

    Returns:
        Dictionary with results and metrics
    """
    logger.info(f"Processing: {image_path.name}")

    results = {
        "input_path": str(image_path),
        "rembg_success": False,
        "perspective_success": False,
        "rembg_time": 0.0,
        "perspective_time": 0.0,
        "total_time": 0.0,
        "metrics_before": {},
        "metrics_after_rembg": {},
        "metrics_after_perspective": {},
        "error": None,
    }

    try:
        total_start = time.perf_counter()

        # Load original image
        original_image = cv2.imread(str(image_path))
        if original_image is None:
            raise ValueError(f"Could not load image: {image_path}")

        results["metrics_before"] = calculate_image_metrics(original_image)

        # Step 1: Background removal with rembg (skip if already processed)
        if skip_rembg:
            # Image is already rembg-processed
            image_no_bg = original_image
            rembg_time = 0.0
            results["rembg_success"] = True
            results["rembg_time"] = 0.0
            logger.info("  ✓ rembg: skipped (already processed)")
        else:
            rembg_start = time.perf_counter()

            if rembg_remover is None:
                rembg_remover = OptimizedBackgroundRemover(
                    model_name="silueta",
                    max_size=2048,
                    alpha_matting=False,
                    use_gpu=use_gpu,
                    use_tensorrt=False,
                    use_int8=False,
                )

            image_no_bg = rembg_remover.remove_background(original_image)
            rembg_time = time.perf_counter() - rembg_start

            results["rembg_success"] = True
            results["rembg_time"] = rembg_time
            logger.info(f"  ✓ rembg: {rembg_time:.3f}s")

        results["metrics_after_rembg"] = calculate_image_metrics(image_no_bg)

        # Save rembg result
        rembg_output = output_dir / f"{image_path.stem}_01_rembg.jpg"
        cv2.imwrite(str(rembg_output), image_no_bg)

        # Step 2: Perspective correction
        perspective_start = time.perf_counter()

        if PERSPECTIVE_AVAILABLE:
            # Use existing DocumentDetector and PerspectiveCorrector
            detector = DocumentDetector(
                logger=logger,
                min_area_ratio=0.1,  # Lower threshold for rembg-processed images
                use_adaptive=True,
                use_fallback=True,
            )

            corners, method = detector.detect(image_no_bg)

            if corners is not None:

                def ensure_doctr(feature: str) -> bool:
                    return True

                corrector = PerspectiveCorrector(
                    logger=logger,
                    ensure_doctr=ensure_doctr,
                    use_doctr_geometry=False,
                    doctr_assume_horizontal=False,
                )

                corrected_image, _matrix, _method_name = corrector.correct(image_no_bg, corners)
                perspective_success = True
            else:
                logger.warning(f"  Could not detect corners for {image_path.name}")
                corrected_image = image_no_bg
                perspective_success = False
        else:
            # Fallback: simple corner detection
            corners = detect_corners_simple(image_no_bg)
            if corners is not None:
                # Calculate destination points
                h, w = image_no_bg.shape[:2]
                max_width = int(max(np.linalg.norm(corners[0] - corners[1]), np.linalg.norm(corners[2] - corners[3])))
                max_height = int(max(np.linalg.norm(corners[0] - corners[3]), np.linalg.norm(corners[1] - corners[2])))

                dst = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], dtype=np.float32)

                matrix = cv2.getPerspectiveTransform(corners, dst)
                corrected_image = cv2.warpPerspective(image_no_bg, matrix, (max_width, max_height))
                perspective_success = True
            else:
                logger.warning(f"  Could not detect corners for {image_path.name}")
                corrected_image = image_no_bg
                perspective_success = False

        perspective_time = time.perf_counter() - perspective_start
        total_time = time.perf_counter() - total_start

        results["perspective_success"] = perspective_success
        results["perspective_time"] = perspective_time
        results["total_time"] = total_time
        results["metrics_after_perspective"] = calculate_image_metrics(corrected_image)

        # Save perspective-corrected result
        perspective_output = output_dir / f"{image_path.stem}_02_perspective.jpg"
        cv2.imwrite(str(perspective_output), corrected_image)

        # Save comparison image (side by side)
        h1, w1 = image_no_bg.shape[:2]
        h2, w2 = corrected_image.shape[:2]
        max_h = max(h1, h2)
        comparison = np.zeros((max_h, w1 + w2 + 20, 3), dtype=np.uint8)
        comparison[:h1, :w1] = image_no_bg
        comparison[:h2, w1 + 20 : w1 + 20 + w2] = corrected_image
        comparison_output = output_dir / f"{image_path.stem}_03_comparison.jpg"
        cv2.imwrite(str(comparison_output), comparison)

        logger.info(f"  ✓ perspective: {perspective_time:.3f}s (success: {perspective_success})")
        logger.info(f"  ✓ total: {total_time:.3f}s")

    except Exception as e:
        results["error"] = str(e)
        logger.error(f"  ✗ Failed: {e}", exc_info=True)

    return results


def main():
    parser = argparse.ArgumentParser(description="Test perspective correction on rembg-processed images")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Input directory with original images (will process with rembg first)",
    )
    parser.add_argument(
        "--rembg-dir",
        type=Path,
        default=None,
        help="Directory with existing rembg-processed images (skip rembg step)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/perspective_test"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of images to process",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for rembg processing",
    )
    parser.add_argument(
        "--image-extensions",
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".bmp"],
        help="Image file extensions to process",
    )

    args = parser.parse_args()

    if not REMBG_AVAILABLE and args.input_dir:
        logger.error("rembg not available. Install with: uv add rembg")
        return 1

    if not PERSPECTIVE_AVAILABLE:
        logger.warning("Perspective correction not fully available. Using fallback.")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find image files
    image_files = []
    if args.rembg_dir:
        # Use existing rembg outputs
        for ext in args.image_extensions:
            image_files.extend(args.rembg_dir.glob(f"*{ext}"))
            image_files.extend(args.rembg_dir.glob(f"*{ext.upper()}"))
        logger.info(f"Found {len(image_files)} rembg-processed images in {args.rembg_dir}")
    elif args.input_dir:
        # Process original images
        for ext in args.image_extensions:
            image_files.extend(args.input_dir.glob(f"*{ext}"))
            image_files.extend(args.input_dir.glob(f"*{ext.upper()}"))
        logger.info(f"Found {len(image_files)} images in {args.input_dir}")
    else:
        logger.error("Must specify either --input-dir or --rembg-dir")
        return 1

    if not image_files:
        logger.error("No image files found")
        return 1

    # Limit to num_samples
    image_files = sorted(image_files)[: args.num_samples]
    logger.info(f"Processing {len(image_files)} images")

    # Initialize rembg remover once if processing from original images
    rembg_remover = None
    if args.input_dir and REMBG_AVAILABLE:
        logger.info("Initializing rembg remover...")
        rembg_remover = OptimizedBackgroundRemover(
            model_name="silueta",
            max_size=2048,
            alpha_matting=False,
            use_gpu=args.use_gpu and GPU_AVAILABLE,
            use_tensorrt=False,
            use_int8=False,
        )

    # Process images
    logger.info("\n" + "=" * 80)
    logger.info("PERSPECTIVE CORRECTION TEST ON REMBG-PROCESSED IMAGES")
    logger.info("=" * 80)

    all_results = []

    for image_path in image_files:
        result = test_perspective_on_rembg(
            image_path=image_path,
            output_dir=args.output_dir,
            rembg_remover=rembg_remover,
            use_gpu=args.use_gpu and GPU_AVAILABLE,
            skip_rembg=args.rembg_dir is not None,  # Skip rembg if using existing outputs
        )
        all_results.append(result)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)

    successful = [r for r in all_results if r.get("perspective_success")]
    failed = [r for r in all_results if not r.get("perspective_success")]

    logger.info(f"\nTotal images: {len(all_results)}")
    logger.info(f"Perspective correction successful: {len(successful)}")
    logger.info(f"Perspective correction failed: {len(failed)}")

    if successful:
        avg_rembg = np.mean([r["rembg_time"] for r in successful])
        avg_perspective = np.mean([r["perspective_time"] for r in successful])
        avg_total = np.mean([r["total_time"] for r in successful])

        logger.info("\nAverage times (successful only):")
        logger.info(f"  rembg:      {avg_rembg:.3f}s")
        logger.info(f"  perspective: {avg_perspective:.3f}s")
        logger.info(f"  total:      {avg_total:.3f}s")

        # Metrics comparison
        if successful[0].get("metrics_after_rembg") and successful[0].get("metrics_after_perspective"):
            avg_contrast_rembg = np.mean([r["metrics_after_rembg"]["contrast"] for r in successful])
            avg_contrast_perspective = np.mean([r["metrics_after_perspective"]["contrast"] for r in successful])
            avg_sharpness_rembg = np.mean([r["metrics_after_rembg"]["sharpness"] for r in successful])
            avg_sharpness_perspective = np.mean([r["metrics_after_perspective"]["sharpness"] for r in successful])

            logger.info("\nAverage metrics:")
            logger.info(f"  Contrast (rembg → perspective): {avg_contrast_rembg:.2f} → {avg_contrast_perspective:.2f}")
            logger.info(f"  Sharpness (rembg → perspective): {avg_sharpness_rembg:.2f} → {avg_sharpness_perspective:.2f}")

    if failed:
        logger.info("\nFailed images:")
        for r in failed:
            logger.info(f"  - {Path(r['input_path']).name}: {r.get('error', 'Corner detection failed')}")

    logger.info("\n" + "=" * 80)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)
    logger.info("\nOutput files:")
    logger.info("  *_01_rembg.jpg       - After background removal")
    logger.info("  *_02_perspective.jpg - After perspective correction")
    logger.info("  *_03_comparison.jpg  - Side-by-side comparison")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
