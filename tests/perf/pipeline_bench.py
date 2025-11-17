#!/usr/bin/env python3
"""Lightweight preprocessing benchmark harness."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from rembg import remove

SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")


def load_manifest(manifest_path: Path) -> list[Path]:
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return [Path(entry["relative_path"]) for entry in payload]
    raise FileNotFoundError(f"Manifest not found at {manifest_path}")


def discover_from_dir(directory: Path, limit: int) -> list[Path]:
    paths = [p for p in directory.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS]
    if limit:
        return paths[:limit]
    return paths


def run_autocontrast(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def run_blur(image: np.ndarray, kernel: int = 5) -> np.ndarray:
    return cv2.GaussianBlur(image, (kernel, kernel), 0)


def run_rembg_cpu(image: np.ndarray) -> bytes:
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Failed to encode image for rembg")
    return remove(buffer.tobytes())


def benchmark(images: Iterable[Path]) -> dict[str, float]:
    auto_contrast_times: list[float] = []
    blur_times: list[float] = []
    rembg_times: list[float] = []

    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            continue

        start = time.perf_counter()
        _ = run_autocontrast(image)
        auto_contrast_times.append((time.perf_counter() - start) * 1_000)

        start = time.perf_counter()
        _ = run_blur(image)
        blur_times.append((time.perf_counter() - start) * 1_000)

        start = time.perf_counter()
        _ = run_rembg_cpu(image)
        rembg_times.append((time.perf_counter() - start) * 1_000)

    def summarize(samples: list[float]) -> dict[str, float]:
        if not samples:
            return {"mean_ms": 0.0, "p95_ms": 0.0}
        return {
            "mean_ms": statistics.fmean(samples),
            "p95_ms": float(np.percentile(samples, 95)),
        }

    return {
        "autocontrast": summarize(auto_contrast_times),
        "gaussian_blur": summarize(blur_times),
        "rembg_client": summarize(rembg_times),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preprocessing latency benchmarks.")
    parser.add_argument("--manifest", type=str, help="Path to sample manifest produced by scripts/datasets/sample_images.py")
    parser.add_argument("--image-dir", type=str, help="Fallback directory to scan for images.")
    parser.add_argument("--limit", type=int, default=16, help="Maximum number of images to benchmark.")
    parser.add_argument("--output", type=str, default="outputs/playground/pipeline_bench.json", help="Where to store measurements.")
    args = parser.parse_args()

    if args.manifest:
        image_paths = load_manifest(Path(args.manifest))
    elif args.image_dir:
        image_paths = discover_from_dir(Path(args.image_dir), args.limit)
    else:
        raise SystemExit("Provide either --manifest or --image-dir.")

    if args.limit:
        image_paths = image_paths[: args.limit]

    metrics = benchmark(image_paths)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()


