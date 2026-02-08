"""
Shared Validation Models for Core-Domain Communication

This module contains Pydantic validation models that are used across both core
modules and domain modules. These models serve as the interface layer between
core infrastructure and domain-specific implementations.

Architecture Rules:
- These are shared data contracts used across boundaries
- Core modules can import from here without violating CORE_PURITY
- Domain modules can import from here for consistent validation
- NO domain-specific logic should live here

Models:
- MapData: Cached probability/threshold maps for dataset caching
- DataItem: Validated dataset sample from OCR pipeline
- MetricConfig: Configuration validation for evaluation metrics
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from ocr.core.data.schemas import ImageMetadata


class MapData(BaseModel):
    """Cached probability/threshold maps aligned with an image sample."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prob_map: np.ndarray
    thresh_map: np.ndarray

    @field_validator("prob_map", "thresh_map", mode="before")
    @classmethod
    def validate_maps(cls, value: Any) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            value = np.asarray(value)
        if value.ndim != 3:
            raise ValueError("Maps must be rank-3 arrays shaped (C, H, W)")
        return value.astype(np.float32)

    @field_validator("thresh_map")
    @classmethod
    def ensure_shape_match(cls, thresh_map: np.ndarray, info: ValidationInfo) -> np.ndarray:
        prob_map = info.data.get("prob_map") if info.data else None
        if prob_map is not None and getattr(prob_map, "shape", None) != thresh_map.shape:
            raise ValueError("Probability and threshold maps must share identical shapes")
        return thresh_map


class DataItem(BaseModel):
    """Validated dataset sample returned by the OCR pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: Any
    polygons: list[np.ndarray] = Field(default_factory=list)
    metadata: dict[str, Any] | ImageMetadata | None = None
    prob_map: np.ndarray | None = None
    thresh_map: np.ndarray | None = None
    inverse_matrix: np.ndarray | None = None

    @field_validator("image", mode="before")
    @classmethod
    def validate_tensor(cls, value: Any) -> Any:
        if isinstance(value, (torch.Tensor, np.ndarray)):
            return value
        raise TypeError(f"Image output must be torch.Tensor or np.ndarray, got {type(value)}")

    @field_validator("polygons", mode="before")
    @classmethod
    def validate_polygons(cls, value: Any) -> list[np.ndarray]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise TypeError("Polygons must be provided as a list")
        normalized: list[np.ndarray] = []
        for poly in value:
            if not isinstance(poly, np.ndarray):
                poly = np.asarray(poly, dtype=np.float32)
            normalized.append(poly.astype(np.float32))
        return normalized

    @field_validator("inverse_matrix", mode="before")
    @classmethod
    def validate_inverse_matrix(cls, value: Any) -> np.ndarray | None:
        if value is None:
            return None
        if not isinstance(value, np.ndarray):
            value = np.asarray(value, dtype=np.float32)
        if value.shape != (3, 3):
            raise ValueError("Inverse matrix must have shape (3, 3)")
        return value.astype(np.float32)


class MetricConfig(BaseModel):
    """Configuration validation for CLEvalMetric parameters."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    dist_sync_on_step: bool = False
    case_sensitive: bool = True
    recall_gran_penalty: float = Field(default=1.0, ge=0.0, description="Recall granularity penalty")
    precision_gran_penalty: float = Field(default=1.0, ge=0.0, description="Precision granularity penalty")
    vertical_aspect_ratio_thresh: float = Field(default=0.5, ge=0.0, le=1.0, description="Vertical aspect ratio threshold")
    ap_constraint: float = Field(default=0.3, ge=0.0, le=1.0, description="AP constraint value")
    scale_wise: bool = False
    scale_bins: tuple[float, ...] = (0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.1, 0.5, 1.0)
    scale_range: tuple[float, float] = Field(default=(0.0, 1.0), description="Scale range as (min, max)")
    max_polygons: int = Field(default=500, gt=0, description="Maximum number of polygons to evaluate")

    @field_validator("scale_range")
    @classmethod
    def _validate_scale_range(cls, value: tuple[float, float]) -> tuple[float, float]:
        if len(value) != 2:
            raise ValueError("scale_range must contain exactly two values (min, max).")
        min_val, max_val = value
        if min_val >= max_val:
            raise ValueError("scale_range min must be less than max.")
        return value

    @field_validator("scale_bins")
    @classmethod
    def _validate_scale_bins(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if len(value) < 2:
            raise ValueError("scale_bins must contain at least two values.")
        if not all(value[i] <= value[i + 1] for i in range(len(value) - 1)):
            raise ValueError("scale_bins must be monotonically increasing.")
        return value


__all__ = ["MapData", "DataItem", "MetricConfig"]
