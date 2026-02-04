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

    # Handle both Tensor and List[int] inputs
    token_list = [sample["text_tokens"] for sample in batch]
    if isinstance(token_list[0], torch.Tensor):
        text_tokens = torch.stack(token_list, dim=0)
    else:
        text_tokens = torch.tensor(token_list, dtype=torch.long)

    labels = [sample["label"] for sample in batch]

    return {
        "images": images,
        "text_tokens": text_tokens,
        "labels": labels,
    }
