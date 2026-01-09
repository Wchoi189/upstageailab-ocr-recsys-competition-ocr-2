#!/usr/bin/env python3
"""
Script to download and integrate Hugging Face datasets for OCR training.
"""

from pathlib import Path

from datasets import load_dataset

# Define datasets to download
DATASETS = [
    "mychen76/wildreceipts_ocr_v1",
    "mychen76/receipt_cord_ocr_v2",
    "mychen76/ds_receipts_v2_train",
    "mychen76/invoices-and-receipts_ocr_v1",
]

DATA_DIR = Path("/workspaces/upstageailab-ocr-recsys-competition-ocr-2/data")


def download_dataset(dataset_name):
    """Download a HF dataset if not cached."""
    try:
        print(f"Loading {dataset_name}...")
        ds = load_dataset(dataset_name)
        print(f"Downloaded {dataset_name} with {len(ds['train'])} entries")
        # Save to data dir if needed, but HF caches it
        return True
    except Exception as e:
        print(f"Error downloading {dataset_name}: {e}")
        return False


if __name__ == "__main__":
    for ds_name in DATASETS:
        download_dataset(ds_name)
