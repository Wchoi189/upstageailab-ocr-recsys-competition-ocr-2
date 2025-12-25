"""Benchmark full extraction pipeline throughput.

This script measures end-to-end performance of the receipt extraction pipeline
including detection, recognition, layout grouping, and field extraction.

Usage:
    uv run python scripts/benchmark_pipeline.py \
        --images-dir data/test_receipts \
        --num-iterations 100 \
        --enable-vlm

Success Criteria:
    - Throughput: ≥100 pages/min
    - VLM call rate: ≤20% of receipts
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2

LOGGER = logging.getLogger(__name__)


def benchmark_extraction_pipeline(
    images_dir: Path,
    checkpoint_path: str,
    num_iterations: int = 100,
    enable_vlm: bool = False,
) -> dict:
    """Benchmark the full extraction pipeline.

    Args:
        images_dir: Directory containing test images
        checkpoint_path: Path to model checkpoint
        num_iterations: Number of images to process
        enable_vlm: Whether to enable VLM extraction

    Returns:
        Dictionary with benchmark results
    """
    from ocr.inference.orchestrator import InferenceOrchestrator

    # Initialize orchestrator with recognition enabled
    LOGGER.info("Initializing InferenceOrchestrator...")
    orchestrator = InferenceOrchestrator(enable_recognition=True)

    # Load model
    LOGGER.info("Loading model from: %s", checkpoint_path)
    if not orchestrator.load_model(checkpoint_path):
        raise RuntimeError("Failed to load model checkpoint")

    # Enable extraction pipeline
    LOGGER.info("Enabling extraction pipeline...")
    orchestrator.enable_extraction_pipeline()

    # Collect test images
    image_paths: list[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(images_dir.glob(ext))

    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")

    # Limit to num_iterations
    image_paths = image_paths[:num_iterations]
    LOGGER.info("Found %d test images", len(image_paths))

    # Warmup (process first 5 images)
    LOGGER.info("Warming up with 5 images...")
    for img_path in image_paths[:5]:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        _ = orchestrator.predict(image, return_preview=False, enable_extraction=True)

    # Benchmark
    LOGGER.info("Starting benchmark with %d images...", len(image_paths))
    start = time.perf_counter()
    vlm_calls = 0
    successful = 0
    failed = 0

    for i, img_path in enumerate(image_paths):
        image = cv2.imread(str(img_path))
        if image is None:
            LOGGER.warning("Failed to load image: %s", img_path)
            failed += 1
            continue

        try:
            result = orchestrator.predict(
                image,
                return_preview=False,
                enable_extraction=True,
            )

            if result is None:
                failed += 1
                continue

            # Check if VLM was used
            receipt_data = result.get("receipt_data", {})
            if receipt_data:
                # Heuristic: check extraction confidence
                # Lower confidence might indicate VLM usage
                confidence = receipt_data.get("extraction_confidence", 1.0)
                if confidence > 0.7:
                    # Likely rule-based
                    pass
                else:
                    # Likely VLM
                    vlm_calls += 1

            successful += 1

        except Exception as e:
            LOGGER.error("Error processing %s: %s", img_path, e)
            failed += 1

        # Progress logging
        if (i + 1) % 10 == 0:
            elapsed = time.perf_counter() - start
            rate = (i + 1) / elapsed * 60
            LOGGER.info(
                "Progress: %d/%d images (%.1f pages/min)",
                i + 1,
                len(image_paths),
                rate,
            )

    elapsed = time.perf_counter() - start
    pages_per_min = len(image_paths) / (elapsed / 60)

    # Results
    results = {
        "total_images": len(image_paths),
        "successful": successful,
        "failed": failed,
        "elapsed_seconds": elapsed,
        "pages_per_minute": pages_per_min,
        "vlm_calls": vlm_calls,
        "vlm_call_rate": vlm_calls / len(image_paths) if image_paths else 0.0,
        "avg_time_per_image_ms": (elapsed / len(image_paths)) * 1000 if image_paths else 0.0,
    }

    return results


def print_results(results: dict) -> None:
    """Print benchmark results in a formatted table.

    Args:
        results: Benchmark results dictionary
    """
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total Images:        {results['total_images']}")
    print(f"Successful:          {results['successful']}")
    print(f"Failed:              {results['failed']}")
    print(f"Elapsed Time:        {results['elapsed_seconds']:.2f}s")
    print(f"Throughput:          {results['pages_per_minute']:.1f} pages/min")
    print(f"Avg Time/Image:      {results['avg_time_per_image_ms']:.1f}ms")
    print(f"VLM Calls:           {results['vlm_calls']}/{results['total_images']} "
          f"({results['vlm_call_rate']*100:.1f}%)")
    print("=" * 60)

    # Success criteria check
    print("\nSUCCESS CRITERIA:")
    throughput_ok = results['pages_per_minute'] >= 100
    vlm_rate_ok = results['vlm_call_rate'] <= 0.20

    print(f"  Throughput ≥100 pages/min: {'✓' if throughput_ok else '✗'} "
          f"({results['pages_per_minute']:.1f})")
    print(f"  VLM call rate ≤20%:        {'✓' if vlm_rate_ok else '✗'} "
          f"({results['vlm_call_rate']*100:.1f}%)")

    if throughput_ok and vlm_rate_ok:
        print("\n✓ All success criteria met!")
    else:
        print("\n✗ Some success criteria not met")

    print("=" * 60 + "\n")


def main():
    """Main entry point for benchmark script."""
    parser = argparse.ArgumentParser(
        description="Benchmark extraction pipeline throughput"
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory containing test images",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to model checkpoint (default: checkpoints/best_model.pth)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=100,
        help="Number of images to process (default: 100)",
    )
    parser.add_argument(
        "--enable-vlm",
        action="store_true",
        help="Enable VLM extraction for complex receipts",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # Validate inputs
    if not args.images_dir.exists():
        raise ValueError(f"Images directory not found: {args.images_dir}")

    # Run benchmark
    LOGGER.info("Starting benchmark...")
    LOGGER.info("Images directory: %s", args.images_dir)
    LOGGER.info("Checkpoint: %s", args.checkpoint)
    LOGGER.info("Num iterations: %d", args.num_iterations)
    LOGGER.info("VLM enabled: %s", args.enable_vlm)

    results = benchmark_extraction_pipeline(
        images_dir=args.images_dir,
        checkpoint_path=args.checkpoint,
        num_iterations=args.num_iterations,
        enable_vlm=args.enable_vlm,
    )

    # Print results
    print_results(results)


if __name__ == "__main__":
    main()
