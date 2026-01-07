"""Collate function for text recognition datasets."""
import torch
from typing import Any


def recognition_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate function for recognition datasets (e.g., LMDBRecognitionDataset).

    Expects each sample to have:
        - image: Tensor [C, H, W]
        - text_tokens: Tensor [T]
        - label: str

    Returns:
        dict with:
            - images: Tensor [B, C, H, W]
            - text_tokens: Tensor [B, T]
            - labels: list[str]
    """
    images = torch.stack([sample["image"] for sample in batch], dim=0)
    text_tokens = torch.stack([sample["text_tokens"] for sample in batch], dim=0)
    labels = [sample["label"] for sample in batch]

    return {
        "images": images,
        "text_tokens": text_tokens,
        "labels": labels,
    }
