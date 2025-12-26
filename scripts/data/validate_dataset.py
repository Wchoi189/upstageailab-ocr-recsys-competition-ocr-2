#!/usr/bin/env python3
"""Validate and visualize standardized OCR datasets.

This script checks adherence to the OCRStorageItem schema and provides
visual inspection of bounding boxes and text.
"""

import argparse
import logging
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from ocr.data.schemas.storage import OCRStorageItem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_file(parquet_path: Path) -> pd.DataFrame | None:
    """Validate a Parquet file against the OCRStorageItem schema."""
    logger.info(f"Validating {parquet_path}...")
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        logger.error(f"Failed to read Parquet file: {e}")
        return None

    valid_count = 0
    errors = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating rows"):
        try:
            # Convert row to dict and validate via Pydantic
            data = row.to_dict()
            # Handle numpy arrays if pyarrow returns them
            if isinstance(data.get("polygons"), np.ndarray):
                data["polygons"] = data["polygons"].tolist()
            if isinstance(data.get("texts"), np.ndarray):
                data["texts"] = data["texts"].tolist()
            if isinstance(data.get("labels"), np.ndarray):
                data["labels"] = data["labels"].tolist()
            if isinstance(data.get("metadata"), np.ndarray):
                # Metadata might be serialised as dict or numpy struct
                 data["metadata"] = data["metadata"].tolist() if hasattr(data["metadata"], "tolist") else data["metadata"]

            item = OCRStorageItem(**data)
            item.validate_lengths()
            valid_count += 1
        except Exception as e:
            errors += 1
            if errors <= 5:  # Print first 5 errors only
                logger.error(f"Row {idx} invalid: {e}")

    logger.info(f"Validation complete. Valid: {valid_count}, Invalid: {errors}")
    return df if errors == 0 else None


def visualize_samples(df: pd.DataFrame, num_samples: int = 5, output_dir: str = "validation_viz"):
    """Visualize random samples with annotation overlays."""
    if df is None or len(df) == 0:
        return

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    samples = df.sample(min(num_samples, len(df)))

    logger.info(f"Visualizing {len(samples)} samples to {output_dir}...")

    for _, row in samples.iterrows():
        img_path = row["image_path"]
        if not Path(img_path).exists():
            logger.warning(f"Image not found: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        polygons = row["polygons"]
        texts = row["texts"]
        metadata = row.get("metadata", {})

        if isinstance(polygons, np.ndarray):
             polygons = polygons.tolist()

        # Deep conversion for nested numpy arrays (pandas/pyarrow artifact)
        clean_polygons = []
        for poly in polygons:
            if isinstance(poly, np.ndarray):
                poly = poly.tolist()
            # Check for inner numpy arrays (points)
            clean_poly = []
            for pt in poly:
                if isinstance(pt, np.ndarray):
                    pt = pt.tolist()
                clean_poly.append(pt)
            clean_polygons.append(clean_poly)
        polygons = clean_polygons

        # 1. Visualize Raw (Original Image + Reprojected Boxes)
        viz_img = img.copy()
        for poly, text in zip(polygons, texts):
            try:
                pts = np.array(poly, np.int32).reshape((-1, 1, 2))
                cv2.polylines(viz_img, [pts], True, (0, 255, 0), 2)
            except Exception as e:
                logger.error(f"Error drawing poly: {poly} - {e}")
                continue

            # Simple text overlay (top-left of poly)
            if len(poly) > 0:
                x, y = int(poly[0][0]), int(poly[0][1])
                cv2.putText(viz_img, text[:20], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        out_name = output_path / f"viz_{Path(img_path).name}"
        cv2.imwrite(str(out_name), viz_img)

        # 2. Visualize Corrected (if inverse matrix exists)
        inverse_matrix_list = metadata.get("inverse_matrix") if isinstance(metadata, dict) else None

        has_inverse = False
        if inverse_matrix_list is not None:
             if isinstance(inverse_matrix_list, np.ndarray):
                 has_inverse = inverse_matrix_list.size > 0
             elif isinstance(inverse_matrix_list, list):
                 has_inverse = len(inverse_matrix_list) > 0

        if has_inverse:
            try:
                inv_matrix = np.array(inverse_matrix_list, dtype=np.float32)
                # We need Forward Matrix = Inverse(InverseMatrix)
                # Because the stored matrix maps Corrected -> Raw.
                # To visualize "Corrected", we map Raw -> Corrected.
                forward_matrix = np.linalg.inv(inv_matrix)

                # Warp Image
                h, w = img.shape[:2]
                # Note: The output size of warpPerspective should ideally match the "corrected" size.
                # We don't have that stored explicitly unless we stored it.
                # But usually 640x640 or similar for model, but here it's full res.
                # Let's assume standard perspective warp keeps approximate size or we use the bounding box.
                # For visualization, using the same size is a reasonable approximation
                # OR we could just let it clip.
                # Better: Use the bounding rect of the warped corners to determine size?
                # For simplicity, keeping original size.
                corrected_img = cv2.warpPerspective(img, forward_matrix, (w, h))

                # Warp Polygons
                for poly in polygons:
                    pts = np.array(poly, dtype=np.float32).reshape(-1, 1, 2)
                    warped_pts = cv2.perspectiveTransform(pts, forward_matrix)
                    warped_pts = warped_pts.reshape(-1, 2).astype(np.int32)

                    cv2.polylines(corrected_img, [warped_pts.reshape(-1, 1, 2)], True, (255, 0, 0), 2)

                out_name_corr = output_path / f"viz_{Path(img_path).stem}_corrected.jpg"
                cv2.imwrite(str(out_name_corr), corrected_img)
            except Exception as e:
                logger.warning(f"Failed to generate corrected visualization for {img_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Validate OCR dataset")
    parser.add_argument("--file", type=str, required=True, help="Input Parquet file")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    args = parser.parse_args()

    df = validate_file(Path(args.file))

    if args.visualize and df is not None:
        visualize_samples(df)


if __name__ == "__main__":
    main()
