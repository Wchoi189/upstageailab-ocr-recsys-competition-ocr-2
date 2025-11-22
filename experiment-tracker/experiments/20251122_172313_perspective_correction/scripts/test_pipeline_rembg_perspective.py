#!/usr/bin/env python3
"""
Test Pipeline: rembg → Perspective Correction → Generate Sample Outputs

This script processes 10 sample images through:
1. Background removal using rembg
2. Perspective correction
3. Output generation with intermediate results

Usage:
    python scripts/test_pipeline_rembg_perspective.py --input-dir data/samples --output-dir outputs/pipeline_test
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Try to import rembg
try:
    from rembg import remove

    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    logger.warning("rembg not available. Install with: uv add rembg")

# Try to import existing perspective correction
try:
    from ocr.datasets.preprocessing.detector import DocumentDetector
    from ocr.datasets.preprocessing.perspective import PerspectiveCorrector

    PERSPECTIVE_AVAILABLE = True
except ImportError:
    PERSPECTIVE_AVAILABLE = False
    logger.warning("Perspective correction not available. Using fallback implementation.")


class SimplePerspectiveCorrector:
    """
    Simple perspective correction based on the reference implementation.

    Reference: https://github.com/sraddhanjali/Automated-Perspective-Correction-for-Scanned-Documents-and-Cards
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def detect_corners(self, image: np.ndarray) -> np.ndarray | None:
        """
        Detect document corners using edge detection and contour analysis.

        Args:
            image: Input image (BGR format)

        Returns:
            Array of 4 corner points [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] or None
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Dilate edges to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Check if we have 4 corners
        if len(approx) != 4:
            # Try to find 4 corners from convex hull
            hull = cv2.convexHull(largest_contour)
            epsilon = 0.02 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)

            if len(approx) < 4:
                return None

        # Extract 4 corner points
        corners = approx.reshape(4, 2)

        # Sort corners: top-left, top-right, bottom-right, bottom-left
        corners = self._sort_corners(corners)

        return corners.astype(np.float32)

    def _sort_corners(self, corners: np.ndarray) -> np.ndarray:
        """Sort corners in order: top-left, top-right, bottom-right, bottom-left."""
        # Calculate centroid
        centroid = np.mean(corners, axis=0)

        # Sort by angle from centroid
        def angle_from_centroid(point):
            return np.arctan2(point[1] - centroid[1], point[0] - centroid[0])

        # Sort corners
        sorted_indices = sorted(range(len(corners)), key=lambda i: angle_from_centroid(corners[i]))
        sorted_corners = corners[sorted_indices]

        # Find top-left (smallest x+y) and bottom-right (largest x+y)
        sums = sorted_corners.sum(axis=1)
        diffs = np.diff(sorted_corners, axis=1).flatten()

        top_left_idx = np.argmin(sums)
        bottom_right_idx = np.argmax(sums)

        # Reorder to start from top-left
        reordered = np.roll(sorted_corners, -top_left_idx, axis=0)

        return reordered

    def correct(self, image: np.ndarray, corners: np.ndarray | None = None) -> tuple[np.ndarray, bool]:
        """
        Apply perspective correction.

        Args:
            image: Input image (BGR format)
            corners: Optional pre-detected corners. If None, will detect automatically.

        Returns:
            Tuple of (corrected_image, success)
        """
        if corners is None:
            corners = self.detect_corners(image)
            if corners is None:
                self.logger.warning("Could not detect corners for perspective correction")
                return image, False

        # Calculate dimensions of output image
        width_a = np.sqrt(((corners[2][0] - corners[3][0]) ** 2) + ((corners[2][1] - corners[3][1]) ** 2))
        width_b = np.sqrt(((corners[1][0] - corners[0][0]) ** 2) + ((corners[1][1] - corners[0][1]) ** 2))
        max_width = max(int(width_a), int(width_b))

        height_a = np.sqrt(((corners[1][0] - corners[2][0]) ** 2) + ((corners[1][1] - corners[2][1]) ** 2))
        height_b = np.sqrt(((corners[0][0] - corners[3][0]) ** 2) + ((corners[0][1] - corners[3][1]) ** 2))
        max_height = max(int(height_a), int(height_b))

        # Destination points for perspective transform
        dst = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype=np.float32,
        )

        # Get perspective transform matrix
        matrix = cv2.getPerspectiveTransform(corners, dst)

        # Apply perspective transformation
        warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

        return warped, True


# Global optimized remover instance (reused across images)
_optimized_remover = None


def get_optimized_remover(
    model_name: str = "silueta",
    max_size: int = 640,
    alpha_matting: bool = False,
    use_gpu: bool = True,
    use_tensorrt: bool = True,
    use_int8: bool = False,
):
    """Get or create optimized background remover instance."""
    global _optimized_remover
    if _optimized_remover is None:
        try:
            # Import from same directory
            import sys
            from pathlib import Path

            script_dir = Path(__file__).parent
            if str(script_dir) not in sys.path:
                sys.path.insert(0, str(script_dir))

            from optimized_rembg import OptimizedBackgroundRemover

            _optimized_remover = OptimizedBackgroundRemover(
                model_name=model_name,
                max_size=max_size,
                alpha_matting=alpha_matting,
                use_gpu=use_gpu,
                use_tensorrt=use_tensorrt,
                use_int8=use_int8,
            )
        except ImportError:
            # Fallback to basic implementation
            _optimized_remover = None
    return _optimized_remover


def remove_background_rembg(image: np.ndarray, use_optimized: bool = True) -> tuple[np.ndarray, float]:
    """
    Remove background using rembg.

    Args:
        image: Input image (BGR format)
        use_optimized: Whether to use optimized remover (faster)

    Returns:
        Tuple of (image_with_bg_removed, processing_time_seconds)
    """
    if not REMBG_AVAILABLE:
        raise RuntimeError("rembg not available. Install with: uv add rembg")

    start_time = time.perf_counter()

    # Try optimized version first
    if use_optimized:
        remover = get_optimized_remover()
        if remover is not None:
            result_bgr = remover.remove_background(image)
            processing_time = time.perf_counter() - start_time
            return result_bgr, processing_time

    # Fallback to basic implementation
    # Convert BGR to RGB for PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    # Remove background
    output = remove(pil_image, model_name="u2net")

    # Convert back to numpy array
    output_array = np.array(output)

    # If RGBA, composite on white background
    if output_array.shape[2] == 4:
        rgb = output_array[:, :, :3]
        alpha = output_array[:, :, 3:4] / 255.0
        white_bg = np.ones_like(rgb) * 255
        result = (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    else:
        result_bgr = cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)

    processing_time = time.perf_counter() - start_time

    return result_bgr, processing_time


def process_image(
    image_path: Path,
    output_dir: Path,
    use_existing_perspective: bool = True,
) -> dict[str, Any]:
    """
    Process a single image through the pipeline.

    Args:
        image_path: Path to input image
        output_dir: Directory to save outputs
        use_existing_perspective: Whether to use existing PerspectiveCorrector if available

    Returns:
        Dictionary with processing results and metrics
    """
    logger.info(f"Processing: {image_path.name}")

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    results = {
        "input_path": str(image_path),
        "input_shape": image.shape,
        "rembg_time": 0.0,
        "perspective_time": 0.0,
        "total_time": 0.0,
        "success": False,
    }

    total_start = time.perf_counter()

    # Step 1: Remove background
    try:
        image_no_bg, rembg_time = remove_background_rembg(image)
        results["rembg_time"] = rembg_time
        results["rembg_success"] = True

        # Save intermediate result
        rembg_output = output_dir / f"{image_path.stem}_01_rembg.jpg"
        cv2.imwrite(str(rembg_output), image_no_bg)
        results["rembg_output"] = str(rembg_output)

    except Exception as e:
        logger.error(f"Background removal failed: {e}")
        results["rembg_success"] = False
        results["error"] = str(e)
        return results

    # Step 2: Perspective correction
    try:
        perspective_start = time.perf_counter()

        if use_existing_perspective and PERSPECTIVE_AVAILABLE:
            # Use existing implementation
            detector = DocumentDetector(
                logger=logger,
                min_area_ratio=0.18,
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
                logger.warning("Could not detect corners, skipping perspective correction")
                corrected_image = image_no_bg
                perspective_success = False
        else:
            # Use simple implementation
            simple_corrector = SimplePerspectiveCorrector()
            corrected_image, perspective_success = simple_corrector.correct(image_no_bg)

        perspective_time = time.perf_counter() - perspective_start
        results["perspective_time"] = perspective_time
        results["perspective_success"] = perspective_success

        # Save final result
        final_output = output_dir / f"{image_path.stem}_02_final.jpg"
        cv2.imwrite(str(final_output), corrected_image)
        results["final_output"] = str(final_output)

        # Save comparison (side by side)
        comparison = np.hstack([image, image_no_bg, corrected_image])
        comparison_output = output_dir / f"{image_path.stem}_03_comparison.jpg"
        cv2.imwrite(str(comparison_output), comparison)
        results["comparison_output"] = str(comparison_output)

    except Exception as e:
        logger.error(f"Perspective correction failed: {e}")
        results["perspective_success"] = False
        results["error"] = str(e)
        # Save what we have
        final_output = output_dir / f"{image_path.stem}_02_final.jpg"
        cv2.imwrite(str(final_output), image_no_bg)
        results["final_output"] = str(final_output)

    total_time = time.perf_counter() - total_start
    results["total_time"] = total_time
    results["success"] = results.get("rembg_success", False) and results.get("perspective_success", False)

    logger.info(
        f"Completed {image_path.name}: rembg={rembg_time:.2f}s, "
        f"perspective={results['perspective_time']:.2f}s, total={total_time:.2f}s"
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="Test pipeline: rembg → perspective correction")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/samples"),
        help="Input directory containing sample images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/pipeline_test"),
        help="Output directory for processed images",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to process",
    )
    parser.add_argument(
        "--use-existing-perspective",
        action="store_true",
        default=True,
        help="Use existing PerspectiveCorrector if available",
    )
    parser.add_argument(
        "--image-extensions",
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".bmp"],
        help="Image file extensions to process",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find image files
    image_files = []
    for ext in args.image_extensions:
        image_files.extend(args.input_dir.glob(f"*{ext}"))
        image_files.extend(args.input_dir.glob(f"*{ext.upper()}"))

    if not image_files:
        logger.error(f"No image files found in {args.input_dir}")
        logger.info(f"Supported extensions: {args.image_extensions}")
        return 1

    # Limit to num_samples
    image_files = sorted(image_files)[: args.num_samples]

    logger.info(f"Found {len(image_files)} images to process")

    # Process images
    all_results = []
    for image_path in image_files:
        try:
            results = process_image(
                image_path,
                args.output_dir,
                use_existing_perspective=args.use_existing_perspective,
            )
            all_results.append(results)
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}", exc_info=True)
            all_results.append(
                {
                    "input_path": str(image_path),
                    "success": False,
                    "error": str(e),
                }
            )

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 80)

    successful = sum(1 for r in all_results if r.get("success", False))
    logger.info(f"Successfully processed: {successful}/{len(all_results)}")

    if successful > 0:
        avg_rembg = np.mean([r.get("rembg_time", 0) for r in all_results if r.get("rembg_success")])
        avg_perspective = np.mean([r.get("perspective_time", 0) for r in all_results if r.get("perspective_success")])
        avg_total = np.mean([r.get("total_time", 0) for r in all_results if r.get("success")])

        logger.info(f"Average rembg time: {avg_rembg:.2f}s")
        logger.info(f"Average perspective time: {avg_perspective:.2f}s")
        logger.info(f"Average total time: {avg_total:.2f}s")

    logger.info(f"\nOutput directory: {args.output_dir}")
    logger.info("=" * 80)

    return 0 if successful == len(all_results) else 1


if __name__ == "__main__":
    exit(main())

