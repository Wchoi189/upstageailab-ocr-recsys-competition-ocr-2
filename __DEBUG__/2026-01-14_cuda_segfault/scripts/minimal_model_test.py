#!/usr/bin/env python3
"""Minimal LMDB + Model inference test to isolate segfault root cause.

This tests if the crash is in model inference during validation.

Usage:
    python DEBUG_WORKSPACE/2026-01-14_cuda_segfault/scripts/minimal_model_test.py
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
    text_tokens = torch.nn.utils.rnn.pad_sequence(
        [item["text_tokens"] for item in batch], batch_first=True, padding_value=0
    )
    return {"images": images, "text_tokens": text_tokens}


def main():
    print("=" * 60)
    print("MINIMAL LMDB + MODEL INFERENCE TEST")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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

    # Create the actual model
    print("\n--- Loading PARSeq model ---")
    from ocr.features.recognition.models.architecture import PARSeq
    from omegaconf import OmegaConf

    # Config matching what Hydra would provide
    cfg = OmegaConf.create({
        "image_size": [32, 128],
        "encoder": {
            "_target_": "ocr.core.models.encoder.timm_backbone.TimmBackbone",
            "model_name": "vit_small_patch16_224",
            "pretrained": True,
            "out_indices": [-1],
            "features_only": True,
            "img_size": [32, 128],
        },
        "decoder": {
            "_target_": "ocr.features.recognition.models.decoder.PARSeqDecoder",
            "in_channels": 384,
            "d_model": 384,
            "nhead": 12,
            "num_layers": 12,
            "dim_feedforward": 1536,
            "dropout": 0.1,
            "vocab_size": tokenizer.vocab_size,
            "max_len": 25,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
        },
        "head": {
            "_target_": "ocr.features.recognition.models.head.PARSeqHead",
            "in_channels": 384,
            "out_channels": tokenizer.vocab_size,
        },
        "loss": {
            "_target_": "ocr.core.models.loss.cross_entropy_loss.CrossEntropyLoss",
            "vocab_size": tokenizer.vocab_size,
            "pad_token_id": 0,
        },
    })

    model = PARSeq(cfg).to(device)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # Test configurations
    configs = [
        {"num_workers": 0, "pin_memory": False},
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

                # Move to device
                images = batch["images"].to(device)
                targets = batch["text_tokens"].to(device)

                # Run inference (like validation step)
                with torch.no_grad():
                    outputs = model(images=images, targets=targets)

                if idx % 5 == 0:
                    loss = outputs.get("loss", torch.tensor(0.0))
                    print(f"  Batch {idx}: loss={loss.item():.4f}")

            print(f"  ✅ PASSED: Completed 20 batches with {config}")

        except Exception as e:
            print(f"  ❌ FAILED at batch {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
