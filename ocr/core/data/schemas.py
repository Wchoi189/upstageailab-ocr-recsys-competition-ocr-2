"""Core data schemas shared across domains.

This module defines generic Pydantic models for image data and metadata
that are used by both detection and recognition pipelines.
"""

from __future__ import annotations

from typing import Any
from pathlib import Path
import numpy as np
import hashlib
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


class CacheConfig(BaseModel):
    """Configuration flags controlling dataset caching behaviour.

    Includes automatic cache versioning to prevent stale cache issues when
    configuration changes affect cached data validity.
    """

    cache_images: bool = True
    cache_maps: bool = True
    cache_transformed_tensors: bool = False
    log_statistics_every_n: int | None = Field(default=None, ge=1)

    def get_cache_version(self, load_maps: bool = False) -> str:
        """Generate cache version hash from configuration.

        The cache version ensures that cached data is invalidated when configuration
        changes affect data validity. Changes to any of these settings will result
        in a new cache version:
        - cache_transformed_tensors: Affects what gets cached
        - cache_images: Affects image caching behavior
        - cache_maps: Affects map caching behavior
        - load_maps: Critical - maps must be in cached data if load_maps=True

        Args:
            load_maps: Whether maps are being loaded (from parent DatasetConfig)

        Returns:
            8-character hex string uniquely identifying this configuration

        Example:
            >>> config = CacheConfig(cache_transformed_tensors=True, load_maps=True)
            >>> version = config.get_cache_version(load_maps=True)
            >>> print(version)  # e.g., "a3f2b8c1"
        """
        # Include all configuration that affects cached data validity
        config_str = (
            f"cache_transformed_tensors={self.cache_transformed_tensors}|"
            f"cache_images={self.cache_images}|"
            f"cache_maps={self.cache_maps}|"
            f"load_maps={load_maps}"
        )
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
