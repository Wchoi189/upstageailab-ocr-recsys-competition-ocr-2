#!/usr/bin/env python
from __future__ import annotations

"""
Offline perspective-correction preprocessor for training images.

This script scans the configured training image directory, applies:
    rembg → mask generation → perspective correction (Max-Edge rule),
and writes corrected images into a parallel directory tree, preserving
relative paths.

Usage (from project root):
    uv run python scripts/offline_perspective_preprocess_train.py
"""

import argparse
import logging
from pathlib import Path
from typing import Iterable

import cv2

from ocr.utils.path_utils import get_path_resolver
from ocr.utils.perspective_correction import (
    correct_perspective_from_mask,
    remove_background_and_mask,
)

LOGGER = logging.getLogger(__name__)


def iter_images(root: Path, exts: Iterable[str]) -> Iterable[Path]:
    exts_norm = {e.lower() for e in exts}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in exts_norm:
            yield path


def process_image(src: Path, src_root: Path, dst_root: Path) -> bool:
    rel = src.relative_to(src_root)
    dst = dst_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(src))
    if image is None:
        LOGGER.warning("Failed to read image: %s", src)
        return False

    try:
        image_no_bg, mask = remove_background_and_mask(image)
        corrected, _fit = correct_perspective_from_mask(image_no_bg, mask)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Perspective preprocessing failed for %s: %s", src, exc)
        corrected = image

    ok = cv2.imwrite(str(dst), corrected)
    if not ok:
        LOGGER.warning("Failed to write corrected image: %s", dst)
        return False
    return True


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Offline perspective-correction preprocessor for training images")
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Optional cap on number of images to process (0 = no limit)",
    )
    parser.add_argument(
        "--exts",
        nargs="+",
        default=[".jpg", ".jpeg", ".png"],
        help="Image extensions to process",
    )
    args = parser.parse_args()

    resolver = get_path_resolver()
    src_root = Path(resolver.config.images_dir) / "train"
    if not src_root.exists():
        LOGGER.error("Training image directory does not exist: %s", src_root)
        return 1

    dst_root = src_root.parent / "images_perspective" / "train"
    dst_root.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Source training images: %s", src_root)
    LOGGER.info("Destination (corrected) images: %s", dst_root)

    images = list(iter_images(src_root, args.exts))
    if args.max_images > 0:
        images = images[: args.max_images]

    LOGGER.info("Found %d images to process", len(images))

    processed = 0
    for i, src in enumerate(images, start=1):
        if process_image(src, src_root, dst_root):
            processed += 1
        if i % 50 == 0:
            LOGGER.info("Processed %d/%d images", i, len(images))

    LOGGER.info("Completed. Successful: %d/%d", processed, len(images))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


