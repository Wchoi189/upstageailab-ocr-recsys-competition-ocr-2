"""KIE validation models and schemas.

This module provides validation contracts for Key Information Extraction (KIE) data.
It mirrors ocr/core/validation.py but focuses on token-level and layout-aware data structures.
"""

from __future__ import annotations

from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, field_validator


class KIEDataItem(BaseModel):
    """Validated dataset sample for KIE models (LayoutLMv3/LiLT)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    bbox: torch.Tensor
    pixel_values: torch.Tensor | None = None # For LayoutLMv3
    labels: torch.Tensor | None = None
    image_path: str | None = None

    @field_validator("input_ids", "attention_mask", mode="before")
    @classmethod
    def validate_long_tensors(cls, value: Any) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.long)
        if value.dtype not in (torch.long, torch.int64):
            value = value.long()
        return value

    @field_validator("bbox", mode="before")
    @classmethod
    def validate_bbox(cls, value: Any) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.long)

        # Shape check: (seq_len, 4)
        if value.ndim != 2 or value.shape[1] != 4:
            raise ValueError(f"BBox must be (seq_len, 4), got {value.shape}")

        return value

    @field_validator("pixel_values", mode="before")
    @classmethod
    def validate_pixel_values(cls, value: Any) -> torch.Tensor | None:
        if value is None:
            return None
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.float32)

        # Shape check: (C, H, W) usually (3, 224, 224) for LayoutLMv3
        if value.ndim != 3:
            raise ValueError(f"Pixel values must be (C, H, W), got {value.shape}")

        return value

    @field_validator("labels", mode="before")
    @classmethod
    def validate_labels(cls, value: Any) -> torch.Tensor | None:
        if value is None:
            return None
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.long)
        return value
