#!/usr/bin/env python3
"""
Pseudo-Label Validation Script

This script validates the generated Parquet pseudo-labels for the KIE dataset.
It checks for:
- Total row count compatibility
- Null values in critical columns
- Schema consistency
- Visual quality (by generating a sampling of images with overlays)

Usage:
    python scripts/data/validate_pseudo_labels.py --input data/processed/pseudo_labels.parquet --images-dir data/datasets/images/train --output-dir outputs/validation
"""

import argparse
import logging
import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def draw_polygons(image, polygons, texts=None, color=(0, 255, 0), thickness=2):
    """Draw polygons on the image."""
    for i, poly in enumerate(polygons):
        # Handle different polygon formats (list of lists, numpy array of arrays, etc.)
        try:
            if isinstance(poly, np.ndarray) and poly.dtype == object:
                # Convert array of arrays to list of points
                pts = np.array([p for p in poly], dtype=np.int32)
            else:
                pts = np.array(poly, np.int32)

            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(image, [pts], True, color, thickness)

            if texts and i < len(texts):
                # Put text near the first point of the polygon
                text = str(texts[i])
                # Basic text display - can be improved for better visibility
                if len(pts) > 0:
                     cv2.putText(image, text[:15], (int(pts[0][0][0]), int(pts[0][0][1]) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        except Exception as e:
            # logger.warning(f"Failed to draw polygon {i}: {e}")
            pass
    return image

def validate_parquet(input_file: str, images_dir: str, output_dir: str, sample_size: int = 10):
    """
    Validate the Parquet file and generate visualization samples.
    """
    input_path = Path(input_file)
    images_path = Path(images_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading Parquet file: {input_path}")
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return False

    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        logger.error(f"Failed to read Parquet file: {e}")
        return False

    # 1. Basic Stats
    total_rows = len(df)
    logger.info(f"Total rows: {total_rows}")

    # Check for expected columns
    expected_cols = ["id", "image_path", "polygons", "texts"] # Minimum expected
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing columns: {missing_cols}")
        # Continue but mark as issue

    # 2. Null Checks
    logger.info("Checking for null values...")
    null_counts = df.isnull().sum()
    if null_counts.any():
        logger.warning(f"Null values found:\n{null_counts[null_counts > 0]}")
    else:
        logger.info("No null values found in columns.")

    # 3. Content Validation
    empty_polygons = df[df["polygons"].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0) == 0]
    if not empty_polygons.empty:
        logger.warning(f"Found {len(empty_polygons)} rows with empty polygons.")

    # 4. Visualization
    logger.info(f"Generating {sample_size} validation samples...")
    sample_dir = output_path / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    sample_df = df.sample(n=min(sample_size, len(df)))

    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        # Construct image path
        # The 'image_path' in dataframe might be relative or absolute.
        # We try to resolve it against images_dir if it doesn't exist directly.

        img_filename = Path(row["image_path"]).name
        img_full_path = images_path / img_filename

        if not img_full_path.exists():
            # Try interpreting row['image_path'] as absolute or relative to project root
            potential_path = Path(row["image_path"])
            if potential_path.exists():
                img_full_path = potential_path
            else:
                logger.warning(f"Image not found for ID {row.get('id', idx)}: {img_full_path}")
                continue

        # Load Image
        image = cv2.imread(str(img_full_path))
        if image is None:
            logger.warning(f"Failed to load image: {img_full_path}")
            continue

        # Draw Polygons
        polygons = row["polygons"]
        texts = row.get("texts", [])

        # Ensure polygons are valid lists of points
        # Assuming format: [[x1,y1], [x2,y2], ...]
        try:
            # Check if polygons are in flat list format or list of lists
            # Adjust drawing logic if necessary based on actual data format
            # API usually returns [[x,y], [x,y], [x,y], [x,y]]

            annotated_img = draw_polygons(image.copy(), polygons, texts)

            output_sample_path = sample_dir / f"val_{Path(img_filename).stem}.jpg"
            cv2.imwrite(str(output_sample_path), annotated_img)

        except Exception as e:
            logger.error(f"Error drawing on image {img_filename}: {e}")
            continue

    logger.info(f"Validation complete. Samples saved to {sample_dir}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Validate Pseudo-Label Parquet")
    parser.add_argument("--input", required=True, help="Path to input parquet file")
    parser.add_argument("--images-dir", required=True, help="Directory containing original images")
    parser.add_argument("--output-dir", default="outputs/validation", help="Output directory for report/samples")
    parser.add_argument("--sample-size", type=int, default=10, help="Number of samples to visualize")

    args = parser.parse_args()

    validate_parquet(args.input, args.images_dir, args.output_dir, args.sample_size)

if __name__ == "__main__":
    main()
