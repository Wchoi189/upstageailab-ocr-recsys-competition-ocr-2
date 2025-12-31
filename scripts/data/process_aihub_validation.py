import os
import json
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def process_aihub_validation(data_dir: str, output_path: str):
    """
    Process AI Hub Public Admin OCR Validation data.

    Args:
        data_dir: Path to the validation directory (containing 'images' and 'labels' folders).
        output_path: Path to save the output parquet file.
    """
    data_path = Path(data_dir)
    images_dir = data_path / "images"
    labels_dir = data_path / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        logger.error(f"Images or labels directory not found in {data_dir}")
        return

    # 1. Index Images
    logger.info("Indexing images...")
    image_map = {}
    # Walk through images_dir
    for root, _, files in os.walk(images_dir):
        for f in files:
            image_map[f] = Path(root) / f
    logger.info(f"Indexed {len(image_map)} images.")

    # 2. Collect JSON files
    json_files = sorted(list(labels_dir.glob("**/*.json")))
    logger.info(f"Found {len(json_files)} JSON label files.")

    records = []

    for json_file in tqdm(json_files, desc="Processing JSONs"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                content = json.load(f)

            # Content structure: {"images": [...], "annotations": [...]}
            # Usually AI Hub has 1 image per JSON or multiple.
            # We map image_id to image info

            # Group annotations by file name (since 'images' has file names)
            # But annotations usually link via 'id'?
            # The example shows "image.file.name" in "images".
            # Annotations usually have "image_id" or similar reference?
            # From overview: annotations have "id", "annotation.type", "annotation.text", "annotation.bbox".
            # It doesn't explicitly show "image_id" in the example snippet, which is suspicious.
            # But usually standard formats (COCO-like) link them.
            # Let's assume there is a linkage or 1 JSON = 1 Image file name embedded.

            # Strategy: If JSON filename matches Image filename, or iterate images.

            # Let's inspect the `annotations` list.
            anns = content.get("annotations", [])

            # If multiple images in one JSON, we need to know which ann belongs to which image.
            # If annotations don't have 'image_id', then maybe 1 JSON = 1 Image?
            # But "images": [...] suggests multiple.
            # Wait, usually AI Hub OCR data is 1 JSON per 1 Image file, having same basename.

            for img_info in content.get("images", []):
                file_name = img_info.get("image.file.name")
                if not file_name:
                    continue

                # Resolve image path using index
                if file_name in image_map:
                    full_image_path = image_map[file_name]
                else:
                    # Fallback: maybe only basename matches?
                    # Or try creating a relative path if the structure mirrors
                    # But index lookup is best.
                    full_image_path = images_dir / file_name # Default failpath

                width = img_info.get("image.width")
                height = img_info.get("image.height")

                # Extract words and boxes
                # Filter anns for this image?
                # If there is no image_id in annotation, and multiple images exist, we are in trouble.
                # But typically AI hub provides 1 JSON file per Image with same name.
                # Let's assume that for now.

                boxes = []
                words = []
                labels = [] # KIE labels (if any, otherwise "O")

                for ann in anns:
                    # Optional: Check linkage if multiple images
                    # if ann.get("image_id") != img_info.get("id"): continue

                    bbox = ann.get("annotation.bbox") # [x, y, w, h] usually
                    text = ann.get("annotation.text")

                    if bbox and text:
                        # Convert [x, y, w, h] using x+w, y+h
                        x, y, w, h = bbox
                        x1, y1 = x, y
                        x2, y2 = x + w, y + h

                        boxes.append([x1, y1, x2, y2])
                        words.append(str(text))
                        labels.append("O") # Default label for now

                record = {
                    "image_path": str(full_image_path), # Construct path
                    "width": width,
                    "height": height,
                    "words": words,
                    "boxes": boxes, # List of [x1, y1, x2, y2]
                    "labels": labels, # List of strings
                    "origin_file": json_file.name
                }
                records.append(record)

        except Exception as e:
            logger.warning(f"Failed to process {json_file}: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(records)
    logger.info(f"Processed {len(df)} images.")

    # Filter empty or missing images
    def check_exists(path_str):
        return os.path.exists(path_str)

    logger.info("Verifying image existence...")
    df["exists"] = df["image_path"].apply(check_exists)
    missing = df[~df["exists"]]
    if len(missing) > 0:
        logger.warning(f"Found {len(missing)} records with missing images. Examples:\n{missing['image_path'].head()}")

    # Save valid only? Or keep all?
    # Keeping valid only is safer for training.
    df_valid = df[df["exists"]].drop(columns=["exists"])
    logger.info(f"Saving {len(df_valid)} valid records to {output_path}")

    # Save to Parquet
    # Ensure directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_valid.to_parquet(output_path, index=False)
    logger.info("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/raw/external/aihub_public_admin_doc/validation", help="Root dir of unpacked data")
    parser.add_argument("--output", type=str, default="data/processed/aihub_validation.parquet", help="Output parquet path")
    args = parser.parse_args()

    process_aihub_validation(args.data_dir, args.output)
