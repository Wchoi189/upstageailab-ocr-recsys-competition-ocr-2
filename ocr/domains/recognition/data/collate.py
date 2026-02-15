"""Collate function for text recognition datasets."""
import logging
from typing import Any

import torch
from torch.utils.data import get_worker_info


logger = logging.getLogger(__name__)
_collate_stats_counter = 0


def _compute_trim_len(text_tokens: torch.Tensor, pad_token_id: int, min_keep_tokens: int) -> int:
    """Compute right-trim length based on max non-pad token index in batch."""
    seq_len = text_tokens.shape[1]
    non_pad = text_tokens.ne(pad_token_id)
    non_pad_counts = non_pad.sum(dim=1)
    max_non_pad = int(non_pad_counts.max().item()) if non_pad_counts.numel() > 0 else 0
    trim_len = max(max_non_pad, int(min_keep_tokens))
    trim_len = min(max(trim_len, 1), seq_len)
    return trim_len


def _log_sequence_stats(text_tokens: torch.Tensor, pad_token_id: int) -> None:
    """Log occasional sequence-length/PAD stats for observability."""
    worker_info = get_worker_info()
    if worker_info is not None and worker_info.id != 0:
        return

    global _collate_stats_counter
    _collate_stats_counter += 1

    if _collate_stats_counter > 5 and _collate_stats_counter % 200 != 0:
        return

    non_pad = text_tokens.ne(pad_token_id)
    non_pad_counts = non_pad.sum(dim=1).float()
    mean_len = float(non_pad_counts.mean().item()) if non_pad_counts.numel() > 0 else 0.0
    max_len = int(non_pad_counts.max().item()) if non_pad_counts.numel() > 0 else 0
    pad_ratio = float(1.0 - non_pad.float().mean().item()) if non_pad.numel() > 0 else 0.0

    logger.info(
        "Recognition collate sequence stats | mean_non_pad_len=%.2f max_non_pad_len=%d pad_ratio=%.3f seq_len=%d step=%d",
        mean_len,
        max_len,
        pad_ratio,
        text_tokens.shape[1],
        _collate_stats_counter,
    )


def recognition_collate_fn(
    batch: list[dict[str, Any]],
    trim_pad_to_batch_max: bool = False,
    min_keep_tokens: int = 4,
    pad_token_id: int = 0,
    log_length_stats: bool = True,
) -> dict[str, Any]:
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

    if trim_pad_to_batch_max and text_tokens.ndim == 2 and text_tokens.shape[1] > 1:
        trim_len = _compute_trim_len(text_tokens, pad_token_id, min_keep_tokens)
        text_tokens = text_tokens[:, :trim_len]

    if log_length_stats and text_tokens.ndim == 2:
        _log_sequence_stats(text_tokens, pad_token_id)

    labels = [sample["label"] for sample in batch]

    return {
        "images": images,
        "text_tokens": text_tokens,
        "labels": labels,
    }
