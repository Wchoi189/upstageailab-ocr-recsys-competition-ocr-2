#!/usr/bin/env python3
"""
Test script for optimized rembg settings:
- Model: silueta
- Image size: 640 (model training size)
- Alpha matting: disabled
- GPU/TensorRT: enabled if available
- INT8 quantization: tested if available

Usage:
    python scripts/test_optimized_rembg.py --input-dir data/samples --output-dir outputs/optimized_test
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
        GPU_AVAILABLE,
        REMBG_AVAILABLE,
        TENSORRT_AVAILABLE,
        OptimizedBackgroundRemover,
        available_providers,
    )
except ImportError as e:
    logger.error(f"Failed to import optimized_rembg: {e}")
    REMBG_AVAILABLE = False
    GPU_AVAILABLE = False
    TENSORRT_AVAILABLE = False
    available_providers = []


def check_system_capabilities():
    """Check system capabilities for optimization."""
    logger.info("=" * 80)
    logger.info("SYSTEM CAPABILITIES CHECK")
    logger.info("=" * 80)
    logger.info(f"rembg available: {REMBG_AVAILABLE}")
    logger.info(f"GPU available: {GPU_AVAILABLE}")
    logger.info(f"TensorRT available: {TENSORRT_AVAILABLE}")
    logger.info(f"Available ONNX providers: {available_providers}")
    logger.info("=" * 80)


def test_configuration(
    image_path: Path,
    output_dir: Path,
    model_name: str,
    image_size: int,
    use_gpu: bool,
    use_tensorrt: bool,
    use_int8: bool,
    resize_to: int | None = None,
) -> dict[str, Any]:
    """
    Test a specific configuration.

    Args:
        image_path: Path to input image
        output_dir: Output directory
        model_name: Model name to use
        image_size: Target image size
        use_gpu: Whether to use GPU
        use_tensorrt: Whether to use TensorRT
        use_int8: Whether to use INT8 quantization

    Returns:
        Dictionary with results and metrics
    """
    logger.info("\nTesting configuration:")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Image size: {image_size}")
    logger.info(f"  GPU: {use_gpu}")
    logger.info(f"  TensorRT: {use_tensorrt}")
    logger.info(f"  INT8: {use_int8}")

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Resize if requested
    original_shape = image.shape
    if resize_to is not None:
        h, w = image.shape[:2]
        max_dim = max(h, w)
        if max_dim > resize_to:
            scale = resize_to / max_dim
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            logger.info(f"  Resized from {original_shape[:2]} to {image.shape[:2]}")

    results = {
        "model": model_name,
        "image_size": image_size,
        "use_gpu": use_gpu,
        "use_tensorrt": use_tensorrt,
        "use_int8": use_int8,
        "input_shape": original_shape,
        "processed_shape": image.shape,
        "resize_to": resize_to,
        "success": False,
        "processing_time": 0.0,
        "error": None,
    }

    try:
        # Create optimized remover
        start_time = time.perf_counter()

        remover = OptimizedBackgroundRemover(
            model_name=model_name,
            max_size=image_size,
            alpha_matting=False,  # Disabled as per requirements
            use_gpu=use_gpu,
            use_tensorrt=use_tensorrt,
            use_int8=use_int8,
        )

        # Process image
        result = remover.remove_background(image)

        processing_time = time.perf_counter() - start_time

        # Save result
        config_name = f"{model_name}_size{image_size}"
        if use_gpu:
            config_name += "_gpu"
        if use_tensorrt:
            config_name += "_tensorrt"
        if use_int8:
            config_name += "_int8"

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
        logger.info(f"  Output: {output_path.name}")

    except Exception as e:
        results["error"] = str(e)
        logger.error(f"  ✗ Failed: {e}", exc_info=True)

    return results


def main():
    parser = argparse.ArgumentParser(description="Test optimized rembg configurations")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/samples"),
        help="Input directory containing sample images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/optimized_test"),
        help="Output directory for processed images",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of images to process",
    )
    parser.add_argument(
        "--resize-to",
        type=int,
        default=None,
        help="Resize images to this size (max dimension) before processing. If None, uses original size.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Process images in batches (for better GPU utilization)",
    )
    parser.add_argument(
        "--image-extensions",
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".bmp"],
        help="Image file extensions to process",
    )
    parser.add_argument(
        "--skip-gpu",
        action="store_true",
        help="Skip GPU tests even if GPU is available",
    )
    parser.add_argument(
        "--skip-tensorrt",
        action="store_true",
        help="Skip TensorRT tests even if TensorRT is available",
    )

    args = parser.parse_args()

    if not REMBG_AVAILABLE:
        logger.error("rembg not available. Install with: uv add rembg")
        return 1

    # Check system capabilities
    check_system_capabilities()

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
    logger.info(f"\nFound {len(image_files)} images to process")

    # Define test configurations
    configurations = [
        # Baseline: silueta, 640, CPU
        {
            "model_name": "silueta",
            "image_size": 640,
            "use_gpu": False,
            "use_tensorrt": False,
            "use_int8": False,
            "resize_to": args.resize_to,
        },
    ]

    # Add GPU configuration if available
    if GPU_AVAILABLE and not args.skip_gpu:
        configurations.append(
            {
                "model_name": "silueta",
                "image_size": 640,
                "use_gpu": True,
                "use_tensorrt": False,
                "use_int8": False,
                "resize_to": args.resize_to,
            }
        )

    # Add TensorRT configuration if available
    if TENSORRT_AVAILABLE and not args.skip_tensorrt and not args.skip_gpu:
        configurations.append(
            {
                "model_name": "silueta",
                "image_size": 640,
                "use_gpu": True,
                "use_tensorrt": True,
                "use_int8": False,
                "resize_to": args.resize_to,
            }
        )

    # Add INT8 configuration (may not work if model not available)
    configurations.append(
        {
            "model_name": "silueta",
            "image_size": 640,
            "use_gpu": GPU_AVAILABLE and not args.skip_gpu,
            "use_tensorrt": TENSORRT_AVAILABLE and not args.skip_tensorrt and not args.skip_gpu,
            "use_int8": True,
            "resize_to": args.resize_to,
        }
    )

    # Test all configurations
    all_results = []

    for config in configurations:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Testing Configuration: {config}")
        logger.info(f"{'=' * 80}")

        config_results = []
        for image_path in image_files:
            try:
                result = test_configuration(
                    image_path=image_path,
                    output_dir=args.output_dir,
                    resize_to=config.get("resize_to"),
                    **{k: v for k, v in config.items() if k != "resize_to"},
                )
                config_results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}", exc_info=True)
                config_results.append(
                    {
                        "input_path": str(image_path),
                        "success": False,
                        "error": str(e),
                        **config,
                    }
                )

        all_results.append({"config": config, "results": config_results})

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("=" * 80)

    for config_group in all_results:
        config = config_group["config"]
        results = config_group["results"]

        successful = [r for r in results if r.get("success", False)]
        if successful:
            avg_time = np.mean([r["processing_time"] for r in successful])
            min_time = np.min([r["processing_time"] for r in successful])
            max_time = np.max([r["processing_time"] for r in successful])

            config_str = (
                f"{config['model_name']}_size{config['image_size']}"
                f"{'_gpu' if config['use_gpu'] else ''}"
                f"{'_tensorrt' if config['use_tensorrt'] else ''}"
                f"{'_int8' if config['use_int8'] else ''}"
            )

            logger.info(f"\n{config_str}:")
            logger.info(f"  Success: {len(successful)}/{len(results)}")
            logger.info(f"  Avg time: {avg_time:.3f}s")
            logger.info(f"  Min time: {min_time:.3f}s")
            logger.info(f"  Max time: {max_time:.3f}s")

    logger.info("\n" + "=" * 80)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
