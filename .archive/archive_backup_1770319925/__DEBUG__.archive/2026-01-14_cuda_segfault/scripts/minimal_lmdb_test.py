#!/usr/bin/env python3
"""Minimal LMDB + DataLoader test to isolate segfault root cause.

This bypasses Hydra entirely to test if the issue is:
1. LMDB multiprocessing unsafety
2. Config/DataLoader interaction
3. Model inference during validation

Usage:
    python DEBUG_WORKSPACE/2026-01-14_cuda_segfault/scripts/minimal_lmdb_test.py
"""
import sys
sys.path.insert(0, "/workspaces/upstageailab-ocr-recsys-competition-ocr-2")

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from ocr.features.recognition.data.lmdb_dataset import LMDBRecognitionDataset
from ocr.features.recognition.data.tokenizer import KoreanOCRTokenizer

LMDB_PATH = "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/data/processed/recognition/aihub_lmdb_validation"
CHARSET_PATH = "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/charset.json"


def collate_fn(batch):
    """Simple collate function for recognition batches."""
    images = torch.stack([item["image"] for item in batch])
    text_tokens = torch.stack([item["text_tokens"] for item in batch])
    return {"images": images, "text_tokens": text_tokens}


def main():
    print("=" * 60)
    print("MINIMAL LMDB + DATALOADER TEST")
    print("=" * 60)

    # Simple transform
    transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create tokenizer
    tokenizer = KoreanOCRTokenizer(CHARSET_PATH)
    print(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    # Create dataset
    dataset = LMDBRecognitionDataset(
        lmdb_path=LMDB_PATH,
        tokenizer=tokenizer,
        transform=transform,
        max_len=25,
    )
    print(f"Dataset loaded: {len(dataset)} samples")

    # Test configurations
    configs = [
        {"num_workers": 0, "pin_memory": False},
        {"num_workers": 0, "pin_memory": True},
        {"num_workers": 2, "pin_memory": False},
        {"num_workers": 2, "pin_memory": True},
    ]

    for config in configs:
        print(f"\n--- Testing config: {config} ---")
        loader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=False,
            collate_fn=collate_fn,
            **config
        )

        try:
            for idx, batch in enumerate(loader):
                if idx >= 20:  # Test 20 batches
                    break
                if idx % 5 == 0:
                    print(f"  Batch {idx}: images={batch['images'].shape}")

            print(f"  ✅ PASSED: Completed 20 batches with {config}")

        except Exception as e:
            print(f"  ❌ FAILED at batch {idx}: {e}")
            continue

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
