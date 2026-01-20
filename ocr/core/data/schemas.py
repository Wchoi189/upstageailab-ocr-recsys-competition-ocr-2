"""Core data schemas shared across domains.

This module defines generic Pydantic models for image data and metadata
that are used by both detection and recognition pipelines.
"""

from __future__ import annotations

from typing import Any
from pathlib import Path
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ImageLoadingConfig(BaseModel):
    """Configuration for image loading backends and fallbacks."""

    use_turbojpeg: bool = False
    turbojpeg_fallback: bool = False


class ImageMetadata(BaseModel):
    """Metadata describing the context of an image being transformed."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    filename: str | None = None
    path: Path | None = None
    original_shape: tuple[int, int]
    orientation: int = Field(ge=0, le=8, default=1)
    is_normalized: bool = False
    dtype: str
    raw_size: tuple[int, int] | None = None
    polygon_frame: str | None = None
    cache_source: str | None = None
    cache_hits: int | None = Field(default=None, ge=0)
    cache_misses: int | None = Field(default=None, ge=0)

    @field_validator("original_shape")
    @classmethod
    def validate_original_shape(cls, value: tuple[int, int]) -> tuple[int, int]:
        if len(value) != 2:
            raise ValueError("original_shape must be a tuple of (height, width)")
        height, width = value
        return (int(height), int(width))

    @field_validator("raw_size")
    @classmethod
    def validate_raw_size(cls, value: tuple[int, int] | None) -> tuple[int, int] | None:
        if value is None:
            return None
        if len(value) != 2:
            raise ValueError("raw_size must be a tuple of (width, height)")
        width, height = value
        return (int(width), int(height))


class ImageData(BaseModel):
    """Cached image payload containing decoded pixel data and metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image_array: np.ndarray
    raw_width: int
    raw_height: int
    orientation: int = Field(ge=0, le=8, default=1)
    is_normalized: bool = False

    @field_validator("image_array", mode="before")
    @classmethod
    def validate_image_array(cls, value: Any) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            value = np.asarray(value)
        if value.ndim not in (2, 3):
            raise ValueError("Cached image array must be 2D or 3D")
        return value
