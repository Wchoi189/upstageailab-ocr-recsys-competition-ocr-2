#!/usr/bin/env python3
"""Sample train/validation images for UI previews and perf benchmarks."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Sequence

SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
DEFAULT_OUTPUT = Path("outputs") / "playground" / "samples"


def discover_images(root: Path, split: str) -> list[Path]:
    """Return all image paths for the provided split."""
    candidate_dirs: Sequence[Path] = [
        root / split,
        root / "datasets" / "images" / split,
        root / "train" / "images",
        root / "val" / "images",
        root / split / "images",
    ]

    for directory in candidate_dirs:
        if directory.exists():
            return [p for p in directory.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS]

    raise FileNotFoundError(f"Could not locate images for split '{split}' relative to {root}")


def sample_images(images: list[Path], count: int) -> list[Path]:
    if count >= len(images):
        return images
    return random.sample(images, count)


def write_manifest(paths: list[Path], output_dir: Path, split: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{split}_samples.json"
    payload = [
        {
            "relative_path": str(path),
            "filesize": path.stat().st_size,
        }
        for path in paths
    ]
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample dataset images for UI previews.")
    parser.add_argument("--project-root", default=".", help="Path to the project root.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Dataset split to sample from.")
    parser.add_argument("--count", type=int, default=16, help="Number of images to sample.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT),
        help="Directory to write the sample manifest.",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    data_root = project_root / "data"
    images = discover_images(data_root, args.split)
    selection = sample_images(images, args.count)
    manifest = write_manifest(selection, Path(args.output_dir), args.split)
    print(f"Wrote {len(selection)} samples to {manifest}")


if __name__ == "__main__":
    main()


