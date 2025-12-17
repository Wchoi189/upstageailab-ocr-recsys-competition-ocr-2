#!/usr/bin/env python3
"""
Test DocTR-based perspective correction on rembg-processed images.

This script tests if DocTR performs better on rembg-processed images
compared to regular images.
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
    logger.warning("Perspective correction not available.")


def test_doctr_on_image(
    image: np.ndarray,
    use_doctr: bool,
    image_name: str,
) -> tuple[np.ndarray | None, dict[str, Any], float]:
    """Test DocTR vs regular detection on an image."""
    results = {
        "use_doctr": use_doctr,
        "corners_detected": False,
        "correction_applied": False,
        "detection_time": 0.0,
        "correction_time": 0.0,
        "error": None,
    }

    try:
        # Detect corners
        detection_start = time.perf_counter()

        detector = DocumentDetector(
            logger=logger,
            min_area_ratio=0.1,
            use_adaptive=True,
            use_fallback=True,
        )

        corners, method = detector.detect(image)
        detection_time = time.perf_counter() - detection_start

        results["detection_time"] = detection_time
        results["detection_method"] = method

        if corners is None:
            results["error"] = "No corners detected"
            return None, results, 0.0

        results["corners_detected"] = True

        # Apply perspective correction
        correction_start = time.perf_counter()

        def ensure_doctr(feature: str) -> bool:
            return use_doctr

        corrector = PerspectiveCorrector(
            logger=logger,
            ensure_doctr=ensure_doctr,
            use_doctr_geometry=use_doctr,
            doctr_assume_horizontal=False,
        )

        corrected_image, _matrix, correction_method = corrector.correct(image, corners)
        correction_time = time.perf_counter() - correction_start

        results["correction_time"] = correction_time
        results["correction_method"] = correction_method
        results["correction_applied"] = True

        total_time = detection_time + correction_time
        return corrected_image, results, total_time

    except Exception as e:
        results["error"] = str(e)
        logger.error(f"Error processing {image_name}: {e}", exc_info=True)
        return None, results, 0.0


def test_doctr_rembg_pipeline(
    image_path: Path,
    output_dir: Path,
    use_gpu: bool = False,
) -> dict[str, Any]:
    """Test DocTR on rembg-processed image."""
    logger.info(f"Processing: {image_path.name}")

    results = {
        "input_path": str(image_path),
        "rembg_time": 0.0,
        "regular_detection": {},
        "doctr_detection": {},
        "error": None,
    }

    try:
        # Load original image
        original_image = cv2.imread(str(image_path))
        if original_image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Step 1: Background removal
        rembg_start = time.perf_counter()

        rembg_remover = OptimizedBackgroundRemover(
            model_name="silueta",
            max_size=2048,
            alpha_matting=False,
            use_gpu=use_gpu and GPU_AVAILABLE,
            use_tensorrt=False,
            use_int8=False,
        )

        image_no_bg = rembg_remover.remove_background(original_image)
        rembg_time = time.perf_counter() - rembg_start

        results["rembg_time"] = rembg_time
        logger.info(f"  ✓ rembg: {rembg_time:.3f}s")

        # Save rembg result
        rembg_output = output_dir / f"{image_path.stem}_01_rembg.jpg"
        cv2.imwrite(str(rembg_output), image_no_bg)

        # Step 2: Test regular detection
        logger.info("  Testing regular detection...")
        regular_corrected, regular_results, regular_time = test_doctr_on_image(
            image_no_bg,
            use_doctr=False,
            image_name=image_path.name,
        )
        results["regular_detection"] = regular_results

        if regular_corrected is not None:
            regular_output = output_dir / f"{image_path.stem}_02_regular.jpg"
            cv2.imwrite(str(regular_output), regular_corrected)
            logger.info(f"    ✓ Regular: {regular_time:.3f}s (method: {regular_results.get('correction_method', 'N/A')})")
        else:
            logger.warning(f"    ✗ Regular failed: {regular_results.get('error', 'Unknown')}")

        # Step 3: Test DocTR detection
        logger.info("  Testing DocTR detection...")
        doctr_corrected, doctr_results, doctr_time = test_doctr_on_image(
            image_no_bg,
            use_doctr=True,
            image_name=image_path.name,
        )
        results["doctr_detection"] = doctr_results

        if doctr_corrected is not None:
            doctr_output = output_dir / f"{image_path.stem}_03_doctr.jpg"
            cv2.imwrite(str(doctr_output), doctr_corrected)
            logger.info(f"    ✓ DocTR: {doctr_time:.3f}s (method: {doctr_results.get('correction_method', 'N/A')})")
        else:
            logger.warning(f"    ✗ DocTR failed: {doctr_results.get('error', 'Unknown')}")

        # Create comparison image
        if regular_corrected is not None and doctr_corrected is not None:
            h1, w1 = image_no_bg.shape[:2]
            h2, w2 = regular_corrected.shape[:2]
            h3, w3 = doctr_corrected.shape[:2]
            max_h = max(h1, h2, h3)
            comparison = np.zeros((max_h, w1 + w2 + w3 + 40, 3), dtype=np.uint8)
            comparison[:h1, :w1] = image_no_bg
            comparison[:h2, w1 + 20 : w1 + 20 + w2] = regular_corrected
            comparison[:h3, w1 + w2 + 40 : w1 + w2 + 40 + w3] = doctr_corrected
            comparison_output = output_dir / f"{image_path.stem}_04_comparison.jpg"
            cv2.imwrite(str(comparison_output), comparison)

    except Exception as e:
        results["error"] = str(e)
        logger.error(f"  ✗ Failed: {e}", exc_info=True)

    return results


def main():
    parser = argparse.ArgumentParser(description="Test DocTR perspective correction on rembg-processed images")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input directory with original images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/doctr_rembg_test"),
        help="Output directory",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of images to process",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for rembg",
    )

    args = parser.parse_args()

    if not REMBG_AVAILABLE:
        logger.error("rembg not available")
        return 1

    if not PERSPECTIVE_AVAILABLE:
        logger.error("Perspective correction not available")
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find images
    image_files = []
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        image_files.extend(args.input_dir.glob(f"*{ext}"))
        image_files.extend(args.input_dir.glob(f"*{ext.upper()}"))

    if not image_files:
        logger.error("No image files found")
        return 1

    image_files = sorted(image_files)[: args.num_samples]
    logger.info(f"Processing {len(image_files)} images")

    # Process
    logger.info("\n" + "=" * 80)
    logger.info("DOCTR vs REGULAR DETECTION ON REMBG-PROCESSED IMAGES")
    logger.info("=" * 80)

    all_results = []

    for image_path in image_files:
        result = test_doctr_rembg_pipeline(
            image_path=image_path,
            output_dir=args.output_dir,
            use_gpu=args.use_gpu,
        )
        all_results.append(result)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)

    regular_success = [r for r in all_results if r.get("regular_detection", {}).get("correction_applied")]
    doctr_success = [r for r in all_results if r.get("doctr_detection", {}).get("correction_applied")]

    logger.info(f"\nTotal images: {len(all_results)}")
    logger.info(f"Regular detection success: {len(regular_success)}/{len(all_results)}")
    logger.info(f"DocTR detection success: {len(doctr_success)}/{len(all_results)}")

    if regular_success:
        avg_regular_time = np.mean(
            [r["regular_detection"]["detection_time"] + r["regular_detection"]["correction_time"] for r in regular_success]
        )
        logger.info(f"Average regular time: {avg_regular_time:.3f}s")

    if doctr_success:
        avg_doctr_time = np.mean([r["doctr_detection"]["detection_time"] + r["doctr_detection"]["correction_time"] for r in doctr_success])
        logger.info(f"Average DocTR time: {avg_doctr_time:.3f}s")

        if regular_success:
            speedup = avg_regular_time / avg_doctr_time if avg_doctr_time > 0 else 0
            logger.info(f"DocTR speedup: {speedup:.2f}x")

    logger.info("\n" + "=" * 80)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
