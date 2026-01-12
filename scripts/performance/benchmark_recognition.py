#!/usr/bin/env python3
"""Benchmark script for text recognition backends.

This script measures throughput and VRAM usage for different recognition
backends (stub, paddleocr, etc.) to ensure they meet performance targets.

Usage:
    uv run python scripts/benchmark_recognition.py --backend paddleocr --batch-size 32
    uv run python scripts/benchmark_recognition.py --backend stub --batch-size 16
"""

from __future__ import annotations

import argparse
import time

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ocr.features.recognition.inference.recognizer import RecognitionInput, RecognizerBackend, RecognizerConfig, TextRecognizer


def create_synthetic_crops(num_crops: int, height: int = 48, width: int = 200) -> list[RecognitionInput]:
    """Create synthetic text crops for benchmarking.

    Args:
        num_crops: Number of crops to generate
        height: Height of each crop in pixels
        width: Width of each crop in pixels

    Returns:
        List of RecognitionInput instances
    """
    inputs = []
    for i in range(num_crops):
        # Create synthetic crop with random patterns
        crop = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

        # Add some white regions to simulate text
        crop[height // 4 : 3 * height // 4, 20 : width - 20] = np.random.randint(200, 255, (height // 2, width - 40, 3))

        # Create polygon
        polygon = np.array(
            [
                [0, 0],
                [width, 0],
                [width, height],
                [0, height],
            ]
        )

        inputs.append(
            RecognitionInput(
                crop=crop,
                polygon=polygon,
                detection_confidence=0.9,
                metadata={"index": i},
            )
        )

    return inputs


def benchmark_throughput(
    backend: RecognizerBackend,
    batch_size: int,
    num_batches: int = 5,
    warmup_batches: int = 2,
) -> dict:
    """Benchmark recognition throughput.

    Args:
        backend: Recognition backend to test
        batch_size: Batch size for inference
        num_batches: Number of batches to benchmark
        warmup_batches: Number of warmup batches

    Returns:
        Dictionary with benchmark results
    """
    # Create recognizer
    config = RecognizerConfig(
        backend=backend,
        max_batch_size=batch_size,
        target_height=48,
        language="korean",
        device="cpu",  # Use CPU for benchmarking
    )
    recognizer = TextRecognizer(config=config)

    # Create synthetic data
    print(f"Creating {num_batches + warmup_batches} batches of {batch_size} crops...")
    all_batches = [create_synthetic_crops(batch_size) for _ in range(num_batches + warmup_batches)]

    # Warmup
    print(f"Warming up with {warmup_batches} batches...")
    for batch in all_batches[:warmup_batches]:
        _ = recognizer.recognize_batch(batch)

    # Clear GPU cache if available
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking {num_batches} batches...")
    timings = []

    for i, batch in enumerate(all_batches[warmup_batches:], 1):
        start_time = time.perf_counter()
        outputs = recognizer.recognize_batch(batch)
        end_time = time.perf_counter()

        elapsed_ms = (end_time - start_time) * 1000
        timings.append(elapsed_ms)

        print(f"  Batch {i}/{num_batches}: {elapsed_ms:.1f}ms ({len(outputs)} crops)")

    # Calculate statistics
    mean_time = np.mean(timings)
    std_time = np.std(timings)
    min_time = np.min(timings)
    max_time = np.max(timings)

    results = {
        "backend": str(backend),
        "batch_size": batch_size,
        "num_batches": num_batches,
        "mean_time_ms": mean_time,
        "std_time_ms": std_time,
        "min_time_ms": min_time,
        "max_time_ms": max_time,
        "throughput_crops_per_sec": (batch_size * 1000) / mean_time,
    }

    # Cleanup
    recognizer.cleanup()

    return results


def measure_vram_usage(backend: RecognizerBackend, batch_size: int) -> dict:
    """Measure VRAM usage for recognition.

    Args:
        backend: Recognition backend to test
        batch_size: Batch size for inference

    Returns:
        Dictionary with VRAM measurements
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Create recognizer
    config = RecognizerConfig(
        backend=backend,
        max_batch_size=batch_size,
        target_height=48,
        language="korean",
        device="cpu",  # Use CPU for VRAM measurement (though it's not VRAM)
    )
    recognizer = TextRecognizer(config=config)

    # Measure after init
    init_vram = torch.cuda.memory_allocated() / (1024**3)

    # Run inference
    crops = create_synthetic_crops(batch_size)
    _ = recognizer.recognize_batch(crops)

    # Measure peak
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

    # Cleanup
    recognizer.cleanup()
    torch.cuda.empty_cache()

    return {
        "baseline_vram_gb": 0.0,  # Not measured
        "init_vram_gb": init_vram,
        "peak_vram_gb": peak_vram,
        "delta_vram_gb": peak_vram,
    }


def main():
    """Main benchmark entry point."""
    parser = argparse.ArgumentParser(description="Benchmark text recognition backends")

    parser.add_argument(
        "--backend",
        type=str,
        default="paddleocr",
        choices=["stub", "paddleocr"],
        help="Recognition backend to benchmark",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference",
    )

    parser.add_argument(
        "--num-batches",
        type=int,
        default=5,
        help="Number of batches to benchmark",
    )

    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=2,
        help="Number of warmup batches",
    )

    parser.add_argument(
        "--measure-vram",
        action="store_true",
        help="Measure VRAM usage (requires CUDA)",
    )

    args = parser.parse_args()

    # Convert backend string to enum
    backend = RecognizerBackend(args.backend)

    print("=" * 80)
    print(f"Recognition Benchmark: {backend}")
    print("=" * 80)
    print(f"Batch size: {args.batch_size}")
    print(f"Number of batches: {args.num_batches}")
    print(f"Warmup batches: {args.warmup_batches}")
    print()

    # Run throughput benchmark
    print("Running throughput benchmark...")
    print("-" * 80)
    results = benchmark_throughput(
        backend=backend,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        warmup_batches=args.warmup_batches,
    )

    print()
    print("Results:")
    print("-" * 80)
    print(f"Mean time:        {results['mean_time_ms']:.1f} ± {results['std_time_ms']:.1f} ms")
    print(f"Min/Max time:     {results['min_time_ms']:.1f} / {results['max_time_ms']:.1f} ms")
    print(f"Throughput:       {results['throughput_crops_per_sec']:.1f} crops/sec")

    # Check against target (400ms for batch of 32)
    if args.batch_size == 32:
        target_ms = 400
        status = "✓ PASS" if results["mean_time_ms"] <= target_ms else "✗ FAIL"
        print(f"Target check:     {status} (target: ≤{target_ms}ms, actual: {results['mean_time_ms']:.1f}ms)")

    # Measure VRAM if requested
    if args.measure_vram:
        print()
        print("Measuring VRAM usage...")
        print("-" * 80)
        vram_results = measure_vram_usage(backend=backend, batch_size=args.batch_size)

        if "error" in vram_results:
            print(f"Error: {vram_results['error']}")
        else:
            print(f"Baseline VRAM:    {vram_results['baseline_vram_gb']:.3f} GB")
            print(f"After init:       {vram_results['init_vram_gb']:.3f} GB")
            print(f"Peak VRAM:        {vram_results['peak_vram_gb']:.3f} GB")
            print(f"Delta VRAM:       {vram_results['delta_vram_gb']:.3f} GB")

            # Check against target (1GB)
            target_gb = 1.0
            status = "✓ PASS" if vram_results["delta_vram_gb"] <= target_gb else "✗ FAIL"
            print(f"Target check:     {status} (target: ≤{target_gb}GB, actual: {vram_results['delta_vram_gb']:.3f}GB)")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
