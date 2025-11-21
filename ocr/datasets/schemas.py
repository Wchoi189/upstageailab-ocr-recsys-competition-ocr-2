"""Pydantic models defining data contracts for dataset transforms."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator


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


class ImageLoadingConfig(BaseModel):
    """Configuration for image loading backends and fallbacks."""

    use_turbojpeg: bool = False
    turbojpeg_fallback: bool = False


class DatasetConfig(BaseModel):
    """All runtime configuration required to build a validated OCR dataset."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image_path: Path
    annotation_path: Path | None = None
    image_extensions: list[str] = Field(default_factory=lambda: [".jpg", ".jpeg", ".png"])
    preload_maps: bool = False
    load_maps: bool = False
    preload_images: bool = False
    prenormalize_images: bool = False
    cache_config: CacheConfig = Field(default_factory=CacheConfig)
    image_loading_config: ImageLoadingConfig = Field(default_factory=ImageLoadingConfig)

    @field_validator("image_extensions", mode="before")
    @classmethod
    def normalize_extensions(cls, value: Any) -> list[str]:
        if value is None:
            return [".jpg", ".jpeg", ".png"]

        if isinstance(value, str):
            value = [value]

        extensions: list[str] = []
        for ext in value:
            if not isinstance(ext, str) or not ext.strip():
                raise ValueError("Image extensions must be non-empty strings")
            normalized = ext.lower()
            if not normalized.startswith("."):
                normalized = f".{normalized}"
            extensions.append(normalized)
        return extensions


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


class PolygonData(BaseModel):
    """Validated polygon representation with consistent shape."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    points: np.ndarray
    confidence: float | None = None
    label: str | None = None

    @field_validator("points", mode="before")
    @classmethod
    def validate_points(cls, value: Any) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            value = np.asarray(value, dtype=np.float32)

        if value.ndim == 1 and value.size % 2 == 0:
            value = value.reshape(-1, 2)
        elif value.ndim == 3 and value.shape[0] == 1:
            value = value.squeeze(0)

        if value.ndim != 2 or value.shape[1] != 2:
            raise ValueError(f"Polygon must be (N, 2) array, got shape {value.shape}")

        if value.shape[0] < 3:
            raise ValueError(f"Polygon must have at least 3 points, got {value.shape[0]}")

        return value.astype(np.float32)


class ValidatedPolygonData(PolygonData):
    """Polygon with additional bounds validation against image dimensions.

    This model extends PolygonData with validation to ensure all polygon coordinates
    are within the image boundaries, preventing out-of-bounds access errors
    (addresses BUG-20251110-001: 26.5% data corruption from invalid coordinates).

    BUG-20251116-001: Updated validation to allow boundary coordinates and clamp
    small floating-point errors, reducing excessive polygon rejection during training.

    Attributes:
        points: Polygon coordinates as (N, 2) array
        confidence: Optional confidence score for the polygon
        label: Optional text label for the polygon
        image_width: Width of the image for bounds checking
        image_height: Height of the image for bounds checking

    Raises:
        ValueError: If any coordinate falls significantly outside image bounds
            (more than 1 pixel tolerance). Coordinates at boundaries or within
            1-pixel tolerance are automatically clamped to valid range.

    Example:
        >>> polygon = ValidatedPolygonData(
        ...     points=np.array([[10, 20], [30, 40], [50, 60]]),
        ...     image_width=100,
        ...     image_height=100
        ... )  # Valid polygon
        >>> polygon = ValidatedPolygonData(
        ...     points=np.array([[10, 20], [30, 40], [100, 60]]),
        ...     image_width=100,
        ...     image_height=100
        ... )  # Valid - boundary coordinate (x=100) is allowed (BUG-20251116-001)
        >>> polygon = ValidatedPolygonData(
        ...     points=np.array([[10, 20], [30, 40], [150, 60]]),
        ...     image_width=100,
        ...     image_height=100
        ... )  # Raises ValueError: x-coordinate 150.0 significantly exceeds image width 100
    """

    image_width: int = Field(gt=0, description="Image width for bounds checking")
    image_height: int = Field(gt=0, description="Image height for bounds checking")

    @model_validator(mode="after")
    def validate_bounds(self) -> ValidatedPolygonData:
        """Validate that all polygon coordinates are within image bounds.

        This validator:
        1. Allows coordinates at exact boundaries (x=width, y=height) for edge cases
        2. Clamps coordinates slightly outside bounds (within 1 pixel tolerance) to handle
           floating-point precision errors from transformations
        3. Rejects coordinates significantly outside bounds (> 1 pixel tolerance)

        BUG-20251116-001: Fixed excessive polygon rejection by:
        - Allowing boundary coordinates (x=width, y=height) instead of exclusive upper bound
        - Adding 1-pixel tolerance for floating-point precision errors
        - Automatically clamping coordinates within tolerance to valid range

        Returns:
            The validated model instance with clamped coordinates if needed

        Raises:
            ValueError: If any coordinate is significantly out of bounds with detailed error message
        """
        points = self.points
        image_width = self.image_width
        image_height = self.image_height

        # BUG-20251116-001: Tolerance for floating-point precision errors (3 pixels)
        # This handles small coordinate errors from EXIF remapping and transformations.
        # Increased from 1.0 to 3.0 to handle real-world transformation errors that can
        # produce coordinates 2-3 pixels outside bounds due to rounding and interpolation.
        tolerance = 3.0

        # BUG-20251116-001: Check x-coordinates (width)
        # Changed from exclusive upper bound (x >= width) to allow boundary coordinates
        x_coords = points[:, 0]
        # Allow coordinates at exact boundary (x=width) and clamp small overflows
        invalid_x = x_coords < -tolerance
        significantly_out_of_bounds_x = x_coords > image_width + tolerance

        if invalid_x.any():
            invalid_indices = np.where(invalid_x)[0]
            invalid_values = x_coords[invalid_indices]
            raise ValueError(
                f"Polygon has out-of-bounds x-coordinates: "
                f"indices {invalid_indices.tolist()} have values {invalid_values.tolist()} "
                f"(must be in [-{tolerance}, {image_width + tolerance}])"
            )

        if significantly_out_of_bounds_x.any():
            invalid_indices = np.where(significantly_out_of_bounds_x)[0]
            invalid_values = x_coords[invalid_indices]
            raise ValueError(
                f"Polygon has out-of-bounds x-coordinates: "
                f"indices {invalid_indices.tolist()} have values {invalid_values.tolist()} "
                f"(must be in [-{tolerance}, {image_width + tolerance}])"
            )

        # BUG-20251116-001: Clamp coordinates within tolerance to valid range
        # Automatically corrects small floating-point errors instead of rejecting polygons
        x_coords_clamped = np.clip(x_coords, 0.0, float(image_width))
        needs_clamping = not np.allclose(x_coords, x_coords_clamped, atol=1e-6)

        # BUG-20251116-001: Check y-coordinates (height)
        # Changed from exclusive upper bound (y >= height) to allow boundary coordinates
        y_coords = points[:, 1]
        # Allow coordinates at exact boundary (y=height) and clamp small overflows
        invalid_y = y_coords < -tolerance
        significantly_out_of_bounds_y = y_coords > image_height + tolerance

        if invalid_y.any():
            invalid_indices = np.where(invalid_y)[0]
            invalid_values = y_coords[invalid_indices]
            raise ValueError(
                f"Polygon has out-of-bounds y-coordinates: "
                f"indices {invalid_indices.tolist()} have values {invalid_values.tolist()} "
                f"(must be in [-{tolerance}, {image_height + tolerance}])"
            )

        if significantly_out_of_bounds_y.any():
            invalid_indices = np.where(significantly_out_of_bounds_y)[0]
            invalid_values = y_coords[invalid_indices]
            raise ValueError(
                f"Polygon has out-of-bounds y-coordinates: "
                f"indices {invalid_indices.tolist()} have values {invalid_values.tolist()} "
                f"(must be in [-{tolerance}, {image_height + tolerance}])"
            )

        # BUG-20251116-001: Clamp coordinates within tolerance to valid range
        # Automatically corrects small floating-point errors instead of rejecting polygons
        y_coords_clamped = np.clip(y_coords, 0.0, float(image_height))
        needs_clamping = needs_clamping or not np.allclose(y_coords, y_coords_clamped, atol=1e-6)

        # BUG-20251116-001: Update points if clamping occurred
        # This preserves polygons that would otherwise be dropped due to minor coordinate errors
        if needs_clamping:
            points_clamped = points.copy()
            points_clamped[:, 0] = x_coords_clamped
            points_clamped[:, 1] = y_coords_clamped
            # Use object.__setattr__ to update field in Pydantic v2
            object.__setattr__(self, "points", points_clamped)

        return self


class TransformInput(BaseModel):
    """Input payload for the OCR transform pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: np.ndarray
    polygons: list[PolygonData] | None = None
    metadata: ImageMetadata | None = None

    @field_validator("image", mode="before")
    @classmethod
    def validate_image(cls, value: Any) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Image must be numpy array, got {type(value)}")

        if value.ndim not in (2, 3):
            raise ValueError(f"Image must be 2D or 3D, got {value.ndim}D")

        if value.ndim == 3 and value.shape[2] not in (1, 3):
            raise ValueError(f"Image must have 1 or 3 channels, got {value.shape[2]}")

        return value


class TransformOutput(BaseModel):
    """Validated output generated by the OCR transform pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: torch.Tensor
    polygons: list[np.ndarray]
    inverse_matrix: np.ndarray
    metadata: dict[str, Any] | None = None

    @field_validator("image", mode="before")
    @classmethod
    def validate_output_image(cls, value: Any) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Output image must be torch.Tensor, got {type(value)}")

        if value.ndim != 3:
            raise ValueError(f"Output image must be 3D (C, H, W), got shape {tuple(value.shape)}")

        return value

    @field_validator("polygons", mode="before")
    @classmethod
    def validate_output_polygons(cls, value: Any) -> list[np.ndarray]:
        if value is None:
            return []

        if not isinstance(value, list):
            raise TypeError(f"Output polygons must be list, got {type(value)}")

        normalized: list[np.ndarray] = []
        for idx, polygon in enumerate(value):
            if not isinstance(polygon, np.ndarray):
                polygon = np.asarray(polygon, dtype=np.float32)

            if polygon.ndim == 2:
                polygon = polygon.reshape(1, -1, 2)

            if polygon.shape[0] != 1 or polygon.shape[2] != 2:
                raise ValueError(f"Polygon at index {idx} must have shape (1, N, 2), got {polygon.shape}")

            normalized.append(polygon.astype(np.float32))

        return normalized

    @field_validator("inverse_matrix", mode="before")
    @classmethod
    def validate_matrix(cls, value: Any) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            value = np.asarray(value, dtype=np.float32)

        if value.shape != (3, 3):
            raise ValueError(f"Inverse matrix must be (3, 3), got {value.shape}")

        return value.astype(np.float32)


class TransformConfig(BaseModel):
    """Configuration for image normalization and transform probabilities."""

    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    always_apply: bool = False
    p: float = 1.0

    @field_validator("mean", "std")
    @classmethod
    def validate_mean_std(cls, value: tuple[float, ...]) -> tuple[float, float, float]:
        if len(value) != 3:
            raise ValueError("Mean and std must each provide 3 values for RGB channels")
        return tuple(float(v) for v in value)  # type: ignore[return-value]


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
        if isinstance(value, torch.Tensor | np.ndarray):
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
