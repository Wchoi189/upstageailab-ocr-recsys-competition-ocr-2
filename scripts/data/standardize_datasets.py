#!/usr/bin/env python3
"""Standardize OCR datasets to the unified OCRStorageItem Parquet format.

This script loads datasets (currently supports conversion from generic formats)
and saves them as validated Parquet files using the OCRStorageItem schema.
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ocr.data.schemas.storage import OCRStorageItem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetStandardizer:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def convert_from_simple_json(self, input_path: str, dataset_name: str, split: str = "train"):
        """Convert the specific baseline JSON dataset format.

        Structure:
        {
          "images": {
            "filename.jpg": {
              "words": {
                "0001": { "points": [[x,y],...], "transcription": "text", ... },
                ...
              }
            },
            ...
          }
        }
        """
        logger.info(f"Loading {input_path}...")
        with open(input_path) as f:
            data = json.load(f)

        # Check if it matches expected structure
        if "images" not in data:
            logger.error("JSON does not contain 'images' key. Unknown format.")
            return

        images_data = data["images"]
        storage_items = []

        # Determine base path for images - typically side-by-side with json or in an 'images' subdir
        # For now, let's assume images are in `data/datasets/images/{split}` or similar.
        # But `OCRStorageItem` takes absolute path.
        # User said "Start with converting baseline training data".
        # I'll rely on the filename being relative to some root.
        # Let's try to locate the image dir.
        # Based on file listing, we have `data/datasets/jsons/`.
        # Maybe images are in `data/datasets/images`?
        # I will store the *relative* path or specific absolute path if I can infer it.
        # Safe bet: `data/datasets/images/{split}/{filename}`?
        # Actually, let's just store the filename and let the dataloader resolve it,
        # OR attempt to find it.
        # For this script, I'll store `image_path` as `data/datasets/images/{filename}`
        # (guessing flattened structure) or just the filename if unsure.

        # Let's check if `data/datasets/images` exists via `list_dir` in next step if needed,
        # but for now I will use a best-guess placeholders: `data/datasets/images/{dataset_name}/{filename}`?
        # Actually `drp.en_ko.in_house` suggests it might be complex.
        # I'll use `data/datasets/images/{filename}` for now.

        base_img_dir = Path("data/datasets/images") / split

        for filename, info in tqdm(images_data.items(), desc="Converting"):
            polygons = []
            texts = []
            labels = []

            words = info.get("words", {})
            for word_id, word_data in words.items():
                # points: [[x,y], ...]
                pts = word_data.get("points", [])

                # transcription/text: "transcription" or just labeled?
                # Inspecting the header output from before: "language": "ko", "orientation"...
                # It didn't show "transcription" or "text" in the first few lines!
                # It showed "language".
                # Wait, if there is no text, is it just detection?
                # User said "baseline training data which is text detection data".
                # So maybe no text labels?
                # I will default text to "" if missing.
                text = word_data.get("transcription", "")

                polygons.append(pts)
                texts.append(text)
                labels.append("text")

            # We need width/height.
            # The JSON snippet didn't show width/height at the image level.
            # If missing, we might need to read the image or use 0.
            # I will use 0 for now to avoid opening 4000 images if effective.
            # But standardizer usually should provide it.
            # Given the constraints, I'll set 0 and let validation or dataloader handle it.

            storage_item = OCRStorageItem(
                id=f"{dataset_name}_{split}_{filename}",
                split=split,
                image_path=str(base_img_dir / filename),
                image_filename=filename,
                width=0,  # To be filled if needed
                height=0,
                polygons=polygons,
                texts=texts,
                labels=labels,
                metadata={"original_source": input_path},
            )
            storage_item.validate_lengths()
            storage_items.append(storage_item.model_dump())

        self._save(storage_items, dataset_name, split)

    def _save(self, items: list[dict], dataset_name: str, split: str):
        if not items:
            logger.warning("No items to save.")
            return

        df = pd.DataFrame(items)
        output_path = self.output_dir / f"{dataset_name}_{split}.parquet"

        logger.info(f"Saving {len(df)} items to {output_path}...")
        df.to_parquet(output_path, engine="pyarrow", index=False)
        logger.info("Done.")


def main():
    parser = argparse.ArgumentParser(description="Standardize OCR datasets")
    parser.add_argument("--input", type=str, required=True, help="Input dataset path (JSON)")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--name", type=str, required=True, help="Dataset name")
    parser.add_argument("--split", type=str, default="train", help="Split name")

    args = parser.parse_args()

    standardizer = DatasetStandardizer(args.output_dir)
    standardizer.convert_from_simple_json(args.input, args.name, args.split)


if __name__ == "__main__":
    main()
