#!/usr/bin/env python3
"""
Script to extract text recognition training data from Hugging Face datasets for PaddleOCR.
"""

from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path("/workspaces/upstageailab-ocr-recsys-competition-ocr-2/data")
OUTPUT_DIR = DATA_DIR / "paddleocr_training_data"


def extract_paddleocr_data(dataset_name, output_subdir):
    """Extract images and create rec_gt.txt for PaddleOCR."""
    output_path = OUTPUT_DIR / output_subdir
    output_path.mkdir(parents=True, exist_ok=True)
    images_dir = output_path / "images"
    images_dir.mkdir(exist_ok=True)

    ds = load_dataset(dataset_name, split="train")

    gt_lines = []
    for i, item in enumerate(ds):
        # Assuming item has 'image' and 'text' or similar
        # For mychen76 datasets, check structure
        if "image" in item and "text" in item:
            image = item["image"]
            text = item["text"]
            image_path = images_dir / f"{i:06d}.jpg"
            image.save(image_path)
            gt_lines.append(f"{image_path}\t{text}")
        else:
            print(f"Skipping item {i}, no image/text")

    gt_file = output_path / "rec_gt.txt"
    with open(gt_file, "w") as f:
        f.write("\n".join(gt_lines))

    print(f"Extracted {len(gt_lines)} samples to {output_path}")


if __name__ == "__main__":
    extract_paddleocr_data("mychen76/wildreceipts_ocr_v1", "wildreceipts")
    extract_paddleocr_data("mychen76/receipt_cord_ocr_v2", "cord")
    extract_paddleocr_data("mychen76/ds_receipts_v2_train", "ds_receipts")
    extract_paddleocr_data("mychen76/invoices-and-receipts_ocr_v1", "invoices")
