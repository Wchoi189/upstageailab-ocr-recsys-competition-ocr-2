#!/usr/bin/env python3
"""
Test rembg with larger workloads to demonstrate GPU acceleration.

This script tests with larger images to show GPU performance benefits.
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

# Import optimized remover
try:
    import sys
    from pathlib import Path

    script_dir = Path(__file__).parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    from optimized_rembg import (
        OptimizedBackgroundRemover,
        REMBG_AVAILABLE,
        GPU_AVAILABLE,
        TENSORRT_AVAILABLE,
    )
except ImportError as e:
    logger.error(f"Failed to import optimized_rembg: {e}")
    REMBG_AVAILABLE = False
    GPU_AVAILABLE = False
    TENSORRT_AVAILABLE = False


def test_large_image(
    image_path: Path,
    output_dir: Path,
    max_size: int,
    use_gpu: bool,
    use_tensorrt: bool,
) -> dict[str, Any]:
    """Test with a large image."""
    logger.info(f"Processing: {image_path.name} (max_size={max_size})")

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    original_h, original_w = image.shape[:2]
    original_max = max(original_h, original_w)

    results = {
        "input_path": str(image_path),
        "original_size": f"{original_w}x{original_h}",
        "max_size": max_size,
        "use_gpu": use_gpu,
        "use_tensorrt": use_tensorrt,
        "success": False,
        "processing_time": 0.0,
        "error": None,
    }

    try:
        # Create remover with specified max_size
        start_time = time.perf_counter()

        remover = OptimizedBackgroundRemover(
            model_name="silueta",
            max_size=max_size,  # Use larger size for GPU testing
            alpha_matting=False,
            use_gpu=use_gpu,
            use_tensorrt=use_tensorrt,
            use_int8=False,
        )

        # Process image
        result = remover.remove_background(image)

        processing_time = time.perf_counter() - start_time

        # Save result
        config_name = f"max{max_size}"
        if use_gpu:
            config_name += "_gpu"
        if use_tensorrt:
            config_name += "_tensorrt"

        output_path = output_dir / f"{image_path.stem}_{config_name}.jpg"
        cv2.imwrite(str(output_path), result)

        results.update(
            {
                "success": True,
                "processing_time": processing_time,
                "output_path": str(output_path),
                "output_shape": result.shape,
            }
        )

        logger.info(f"  ✓ Success: {processing_time:.3f}s")
        logger.info(f"  Original: {original_w}x{original_h} (max: {original_max}px)")
        logger.info(f"  Processed: {result.shape[1]}x{result.shape[0]}")

    except Exception as e:
        results["error"] = str(e)
        logger.error(f"  ✗ Failed: {e}", exc_info=True)

    return results


def main():
    parser = argparse.ArgumentParser(description="Test rembg with larger workloads")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/datasets/images/train"),
        help="Input directory containing sample images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/large_workload_test"),
        help="Output directory for processed images",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of images to process",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[640, 1024, 2048],
        help="Image sizes to test (max dimension)",
    )

    args = parser.parse_args()

    if not REMBG_AVAILABLE:
        logger.error("rembg not available. Install with: uv add rembg")
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find image files
    image_files = []
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        image_files.extend(args.input_dir.glob(f"*{ext}"))
        image_files.extend(args.input_dir.glob(f"*{ext.upper()}"))

    if not image_files:
        logger.error(f"No image files found in {args.input_dir}")
        return 1

    # Limit to num_samples
    image_files = sorted(image_files)[: args.num_samples]
    logger.info(f"Found {len(image_files)} images to process")

    # Test configurations
    logger.info("\n" + "=" * 80)
    logger.info("LARGE WORKLOAD GPU PERFORMANCE TEST")
    logger.info("=" * 80)

    all_results = []

    for size in args.sizes:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Testing with max_size={size}px")
        logger.info(f"{'=' * 80}")

        # Test CPU
        logger.info("\n--- CPU Baseline ---")
        cpu_results = []
        for image_path in image_files:
            try:
                result = test_large_image(
                    image_path=image_path,
                    output_dir=args.output_dir,
                    max_size=size,
                    use_gpu=False,
                    use_tensorrt=False,
                )
                cpu_results.append(result)
            except Exception as e:
                logger.error(f"Failed: {e}")

        # Test GPU
        if GPU_AVAILABLE:
            logger.info("\n--- GPU (CUDA) ---")
            gpu_results = []
            for image_path in image_files:
                try:
                    result = test_large_image(
                        image_path=image_path,
                        output_dir=args.output_dir,
                        max_size=size,
                        use_gpu=True,
                        use_tensorrt=False,
                    )
                    gpu_results.append(result)
                except Exception as e:
                    logger.error(f"Failed: {e}")

        # Test TensorRT
        if TENSORRT_AVAILABLE:
            logger.info("\n--- TensorRT ---")
            tensorrt_results = []
            for image_path in image_files:
                try:
                    result = test_large_image(
                        image_path=image_path,
                        output_dir=args.output_dir,
                        max_size=size,
                        use_gpu=True,
                        use_tensorrt=True,
                    )
                    tensorrt_results.append(result)
                except Exception as e:
                    logger.error(f"Failed: {e}")

        # Summary for this size
        logger.info(f"\n--- Summary for {size}px ---")
        if cpu_results:
            cpu_times = [r["processing_time"] for r in cpu_results if r.get("success")]
            if cpu_times:
                avg_cpu = np.mean(cpu_times)
                logger.info(f"CPU: {avg_cpu:.3f}s avg")

        if GPU_AVAILABLE and gpu_results:
            gpu_times = [r["processing_time"] for r in gpu_results if r.get("success")]
            if gpu_times:
                avg_gpu = np.mean(gpu_times)
                speedup = avg_cpu / avg_gpu if cpu_times else 0
                logger.info(f"GPU: {avg_gpu:.3f}s avg ({speedup:.2f}x speedup)")

        if TENSORRT_AVAILABLE and tensorrt_results:
            trt_times = [r["processing_time"] for r in tensorrt_results if r.get("success")]
            if trt_times:
                avg_trt = np.mean(trt_times)
                speedup = avg_cpu / avg_trt if cpu_times else 0
                logger.info(f"TensorRT: {avg_trt:.3f}s avg ({speedup:.2f}x speedup)")

        all_results.append(
            {
                "size": size,
                "cpu": cpu_results,
                "gpu": gpu_results if GPU_AVAILABLE else [],
                "tensorrt": tensorrt_results if TENSORRT_AVAILABLE else [],
            }
        )

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("FINAL PERFORMANCE SUMMARY")
    logger.info("=" * 80)

    for result_group in all_results:
        size = result_group["size"]
        cpu_times = [r["processing_time"] for r in result_group["cpu"] if r.get("success")]
        gpu_times = [r["processing_time"] for r in result_group["gpu"] if r.get("success")]
        trt_times = [r["processing_time"] for r in result_group["tensorrt"] if r.get("success")]

        logger.info(f"\n{size}px images:")
        if cpu_times:
            logger.info(f"  CPU:      {np.mean(cpu_times):.3f}s (min: {np.min(cpu_times):.3f}s, max: {np.max(cpu_times):.3f}s)")
        if gpu_times:
            speedup = np.mean(cpu_times) / np.mean(gpu_times) if cpu_times else 0
            logger.info(f"  GPU:      {np.mean(gpu_times):.3f}s (min: {np.min(gpu_times):.3f}s, max: {np.max(gpu_times):.3f}s) - {speedup:.2f}x faster")
        if trt_times:
            speedup = np.mean(cpu_times) / np.mean(trt_times) if cpu_times else 0
            logger.info(f"  TensorRT: {np.mean(trt_times):.3f}s (min: {np.min(trt_times):.3f}s, max: {np.max(trt_times):.3f}s) - {speedup:.2f}x faster")

    logger.info("\n" + "=" * 80)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())

