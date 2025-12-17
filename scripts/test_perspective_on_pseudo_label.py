#!/usr/bin/env python
from __future__ import annotations

"""
Test the perspective correction pipeline on pseudo-label images.

Processes up to `--max-images` images from:
    data/pseudo_label/wildreceipt/images
and writes side-by-side comparison images to the specified output directory.
"""

import argparse
import logging
import time
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np

from ocr.utils.perspective_correction import (
    correct_perspective_from_mask,
    remove_background_and_mask,
)

LOGGER = logging.getLogger(__name__)


def iter_images(root: Path, exts: Iterable[str]) -> list[Path]:
    exts_norm = {e.lower() for e in exts}
    images: list[Path] = []
    if not root.exists():
        return images
    for path in root.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() in exts_norm:
            images.append(path)
    return sorted(images)


def process_image(path: Path, output_dir: Path) -> dict:
    LOGGER.info("Processing %s", path.name)

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")

    start = time.perf_counter()
    image_no_bg, mask = remove_background_and_mask(image)
    corrected, fit = correct_perspective_from_mask(image_no_bg, mask)
    elapsed = time.perf_counter() - start

    # Save individual stages
    base = path.stem
    out_orig = output_dir / f"{base}_00_original.jpg"
    out_nobg = output_dir / f"{base}_01_nobg.jpg"
    out_corr = output_dir / f"{base}_02_corrected.jpg"
    out_cmp = output_dir / f"{base}_03_comparison.jpg"

    cv2.imwrite(str(out_orig), image)
    cv2.imwrite(str(out_nobg), image_no_bg)
    cv2.imwrite(str(out_corr), corrected)

    # Horizontal comparison (resize if shapes differ)
    h = min(image.shape[0], image_no_bg.shape[0], corrected.shape[0])

    def resize_to_h(img: np.ndarray) -> np.ndarray:
        if img.shape[0] == h:
            return img
        scale = h / img.shape[0]
        w = int(img.shape[1] * scale)
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    cmp = np.hstack([resize_to_h(image), resize_to_h(image_no_bg), resize_to_h(corrected)])
    cv2.imwrite(str(out_cmp), cmp)

    return {
        "input": str(path),
        "time_sec": elapsed,
        "success": fit.corners is not None,
        "reason": fit.reason,
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Test perspective correction on pseudo-label images")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/pseudo_label/wildreceipt/images"),
        help="Directory containing pseudo-label images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/pseudo_label_perspective"),
        help="Directory to write outputs",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=50,
        help="Maximum number of images to process",
    )
    parser.add_argument(
        "--exts",
        nargs="+",
        default=[".jpg", ".jpeg", ".png"],
        help="Image extensions to process",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    images = iter_images(args.input_dir, args.exts)
    if not images:
        LOGGER.error("No images found in %s", args.input_dir)
        return 1

    if args.max_images > 0:
        images = images[: args.max_images]

    LOGGER.info("Processing %d images from %s", len(images), args.input_dir)

    results = []
    for idx, img_path in enumerate(images, start=1):
        try:
            res = process_image(img_path, args.output_dir)
            results.append(res)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to process %s: %s", img_path, exc)
            results.append({"input": str(img_path), "time_sec": 0.0, "success": False, "reason": str(exc)})

        if idx % 10 == 0:
            LOGGER.info("Processed %d/%d images", idx, len(images))

    total = len(results)
    success = sum(1 for r in results if r.get("success"))
    avg_time = float(sum(r.get("time_sec", 0.0) for r in results if r.get("success"))) / max(success, 1) if results else 0.0

    LOGGER.info("Completed. Success: %d/%d, avg_time=%.3fs", success, total, avg_time)
    LOGGER.info("Output directory: %s", args.output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
