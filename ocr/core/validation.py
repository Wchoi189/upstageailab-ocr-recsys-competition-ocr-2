"""Core validation models and schemas for the OCR pipeline.

Consolidates:
- ocr.validation.models (runtime validation)
- ocr.datasets.schemas (dataset contracts)

This module provides a single source of truth for all validation models.

Rationale (do not modularize): consolidation prevents circular imports and keeps
schemas + runtime validators consistent across the pipeline.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator, model_validator
from pydantic_core import InitErrorDetails, PydanticCustomError

# =============================================================================
# SECTION 1: Base Utilities & Constants (from validation/models)
# =============================================================================

VALID_EXIF_ORIENTATIONS: frozenset[int] = frozenset({0, 1, 2, 3, 4, 5, 6, 7, 8})
validator = field_validator


def _info_data(info: ValidationInfo | None) -> Mapping[str, Any]:
    """Safely extract validator context data across Pydantic versions."""
    if info is None:
        return {}
    data = getattr(info, "data", None)
    if isinstance(data, Mapping):
        return data
    return {}


def _batch_size(info: ValidationInfo | None) -> int:
    """Best-effort batch size lookup from validator context."""
    data = _info_data(info)
    for key in ("image_filename", "image_path", "shape", "inverse_matrix"):
        value = data.get(key)
        if isinstance(value, list):
            return len(value)
    return 0


def _ensure_tuple_pair(value: tuple[int, int] | Sequence[int] | None, field_name: str) -> tuple[int, int] | None:
    """Normalize a (width, height) tuple and validate that it is usable."""
    if value is None:
        return None
    if isinstance(value, tuple) and len(value) == 2:
        width, height = value
    else:
        try:
            width, height = int(value[0]), int(value[1])  # type: ignore[index]
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"{field_name} must contain exactly two integer values.") from exc
    if width < 0 or height < 0:
        raise ValueError(f"{field_name} dimensions must be non-negative.")
    return int(width), int(height)


# =============================================================================
# SECTION 2: Dataset Schemas (from datasets/schemas)
# =============================================================================


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
    enable_background_normalization: bool = False
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


# =============================================================================
# SECTION 3: Runtime Validation Models (from validation/models)
# =============================================================================


class _ModelBase(BaseModel):
    """Common configuration shared by every validation model."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)


class PolygonArray(_ModelBase):
    """Validate that a polygon is provided as an ``(N, 2)`` numpy array."""

    points: np.ndarray = Field(..., description="Polygon with shape (N, 2) and float coordinates.")

    @field_validator("points")
    @classmethod
    def _validate_points(cls, value: np.ndarray) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            raise TypeError("Polygon must be provided as a numpy.ndarray.")
        if value.ndim != 2 or value.shape[1] != 2:
            raise ValueError(f"Polygon must be shaped (N, 2); received {value.shape}.")
        if value.shape[0] < 3:
            raise ValueError("Polygon requires at least three points.")
        if value.dtype not in (np.float32, np.float64):
            value = value.astype(np.float32)
        return value


class DatasetSample(_ModelBase):
    """Dataset output prior to augmentation."""

    image: np.ndarray = Field(..., description="Raw image array shaped (H, W, 3).")
    polygons: list[np.ndarray] = Field(default_factory=list, description="List of ground-truth polygons.")
    prob_maps: np.ndarray = Field(..., description="Probability map shaped (H, W).")
    thresh_maps: np.ndarray = Field(..., description="Threshold map shaped (H, W).")
    image_filename: str
    image_path: str
    inverse_matrix: np.ndarray = Field(..., description="Homography matrix shaped (3, 3).")
    shape: tuple[int, int]

    @field_validator("image")
    @classmethod
    def _check_image(cls, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Image must be shaped (H, W, 3); received {image.shape}.")
        return image

    @field_validator("polygons")
    @classmethod
    def _check_polygons(cls, polygons: list[np.ndarray]) -> list[np.ndarray]:
        return [PolygonArray(points=polygon).points for polygon in polygons]

    @field_validator("prob_maps")
    @classmethod
    def _check_prob_maps(cls, heatmap: np.ndarray) -> np.ndarray:
        if heatmap.ndim != 2:
            raise ValueError(f"prob_maps must be 2D; received shape {heatmap.shape}.")
        return heatmap

    @field_validator("thresh_maps")
    @classmethod
    def _check_thresh_maps(cls, thresh_maps: np.ndarray, info: ValidationInfo) -> np.ndarray:
        if thresh_maps.ndim != 2:
            raise ValueError(f"thresh_maps must be 2D; received shape {thresh_maps.shape}.")
        prob_maps = _info_data(info).get("prob_maps")
        if isinstance(prob_maps, np.ndarray) and prob_maps.shape != thresh_maps.shape:
            raise ValueError("Probability and threshold maps must share the same shape.")
        return thresh_maps

    @field_validator("inverse_matrix")
    @classmethod
    def _check_inverse_matrix(cls, matrix: np.ndarray) -> np.ndarray:
        if matrix.shape != (3, 3):
            raise ValueError("Inverse matrix must be shaped (3, 3).")
        return matrix

    @field_validator("shape")
    @classmethod
    def _check_shape(cls, shape: tuple[int, int]) -> tuple[int, int]:
        if len(shape) != 2:
            raise ValueError("shape must contain exactly two dimensions.")
        return int(shape[0]), int(shape[1])


class LoaderTransformOutput(_ModelBase):
    """Output of the transform pipeline that feeds the DataLoader."""

    image: torch.Tensor = Field(..., description="Transformed image shaped (3, H, W).")
    polygons: list[np.ndarray] = Field(..., description="Polygons after augmentation.")
    prob_maps: torch.Tensor = Field(..., description="Probability map tensor shaped (1, H, W).")
    thresh_maps: torch.Tensor = Field(..., description="Threshold map tensor shaped (1, H, W).")
    inverse_matrix: np.ndarray = Field(..., description="Inverse transformation matrix shaped (3, 3).")

    @field_validator("image")
    @classmethod
    def _check_tensor_image(cls, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 3 or tensor.shape[0] != 3:
            raise ValueError("Transformed image tensor must be shaped (3, H, W).")
        return tensor

    @field_validator("polygons")
    @classmethod
    def _check_transformed_polygons(cls, polygons: list[np.ndarray]) -> list[np.ndarray]:
        return [PolygonArray(points=polygon).points for polygon in polygons]

    @field_validator("prob_maps")
    @classmethod
    def _check_tensor_prob_maps(cls, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 3 or tensor.shape[0] != 1:
            raise ValueError("prob_maps tensor must be shaped (1, H, W).")
        return tensor

    @field_validator("thresh_maps")
    @classmethod
    def _check_tensor_thresh_maps(cls, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 3 or tensor.shape[0] != 1:
            raise ValueError("thresh_maps tensor must be shaped (1, H, W).")
        return tensor

    @field_validator("inverse_matrix")
    @classmethod
    def _check_transform_matrix(cls, matrix: np.ndarray) -> np.ndarray:
        if matrix.shape != (3, 3):
            raise ValueError("Inverse matrix must be shaped (3, 3).")
        return matrix


class BatchSample(_ModelBase):
    """Single dataset sample produced before collation."""

    image: torch.Tensor
    polygons: list[np.ndarray]
    prob_maps: torch.Tensor
    thresh_maps: torch.Tensor
    image_filename: str
    image_path: str
    inverse_matrix: np.ndarray
    shape: tuple[int, int]

    @field_validator("polygons")
    @classmethod
    def _check_sample_polygons(cls, polygons: list[np.ndarray]) -> list[np.ndarray]:
        return [PolygonArray(points=polygon).points for polygon in polygons]

    @field_validator("inverse_matrix")
    @classmethod
    def _check_sample_matrix(cls, matrix: np.ndarray) -> np.ndarray:
        if matrix.shape != (3, 3):
            raise ValueError("Inverse matrix must be shaped (3, 3).")
        return matrix


class CollateOutput(_ModelBase):
    """Batch produced by the DataLoader collate function."""

    image_filename: list[str]
    image_path: list[str]
    inverse_matrix: list[np.ndarray]
    shape: list[tuple[int, int]]
    images: torch.Tensor = Field(..., description="Batch of images shaped (B, 3, H, W).")
    polygons: list[list[np.ndarray]] = Field(..., description="Polygons per image.")
    prob_maps: torch.Tensor = Field(..., description="Probability maps shaped (B, 1, H, W).")
    thresh_maps: torch.Tensor = Field(..., description="Threshold maps shaped (B, 1, H, W).")
    orientation: Sequence[int] | None = None
    raw_size: Sequence[tuple[int, int]] | None = None
    canonical_size: Sequence[tuple[int, int] | None] | None = None
    metadata: list[dict[str, Any] | ImageMetadata | None] | None = None

    @field_validator("image_filename", "image_path", "shape", "inverse_matrix", mode="before")
    @classmethod
    def _check_list_lengths(cls, value: Sequence[Any] | None) -> list[Any]:
        if value is None:
            raise ValueError("Batch metadata sequences must contain at least one entry.")
        if not isinstance(value, list):
            value = list(value)
        if not value:
            raise ValueError("Batch metadata sequences must contain at least one entry.")
        return value

    @field_validator("images")
    @classmethod
    def _check_images(cls, images: torch.Tensor, info: ValidationInfo) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images tensor must be shaped (B, 3, H, W).")
        batch = _batch_size(info)
        if batch and images.shape[0] != batch:
            raise ValueError("Number of images does not match batch metadata.")
        return images

    @field_validator("polygons")
    @classmethod
    def _check_collated_polygons(cls, polygons: list[list[np.ndarray]], info: ValidationInfo) -> list[list[np.ndarray]]:
        batch = _batch_size(info)
        if batch and len(polygons) != batch:
            raise ValueError("Polygons list must align with batch size.")
        for poly_list in polygons:
            if not isinstance(poly_list, list):
                raise TypeError("Polygons entry must be a list of np.ndarray objects.")
            for polygon in poly_list:
                PolygonArray(points=polygon)
        return polygons

    @field_validator("prob_maps")
    @classmethod
    def _check_collated_prob_maps(cls, tensor: torch.Tensor, info: ValidationInfo) -> torch.Tensor:
        if tensor.ndim != 4 or tensor.shape[1] != 1:
            raise ValueError("prob_maps tensor must be shaped (B, 1, H, W).")
        batch = _batch_size(info)
        if batch and tensor.shape[0] != batch:
            raise ValueError("prob_maps batch dimension must match batch size.")
        return tensor

    @field_validator("thresh_maps")
    @classmethod
    def _check_collated_thresh_maps(cls, tensor: torch.Tensor, info: ValidationInfo) -> torch.Tensor:
        if tensor.ndim != 4 or tensor.shape[1] != 1:
            raise ValueError("thresh_maps tensor must be shaped (B, 1, H, W).")
        batch = _batch_size(info)
        if batch and tensor.shape[0] != batch:
            raise ValueError("thresh_maps batch dimension must match batch size.")
        return tensor

    @field_validator("orientation")
    @classmethod
    def _check_orientation(cls, value: Sequence[int] | None, info: ValidationInfo) -> Sequence[int] | None:
        if value is None:
            return None
        items = list(value)
        batch = _batch_size(info)
        if batch and len(items) != batch:
            raise ValueError("orientation length must match batch size.")
        normalized: list[int] = []
        allowed = ", ".join(str(v) for v in sorted(VALID_EXIF_ORIENTATIONS))
        for idx, item in enumerate(items):
            try:
                orientation = int(item)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"orientation[{idx}] must be castable to int.") from exc
            if orientation not in VALID_EXIF_ORIENTATIONS:
                raise ValueError(f"orientation[{idx}] must be one of {{{allowed}}}.")
            normalized.append(orientation)
        return normalized

    @field_validator("raw_size")
    @classmethod
    def _check_raw_sizes(cls, value: Sequence[tuple[int, int]] | None, info: ValidationInfo) -> Sequence[tuple[int, int]] | None:
        if value is None:
            return None
        items = list(value)
        batch = _batch_size(info)
        if batch and len(items) != batch:
            raise ValueError("raw_size length must match batch size.")
        normalized: list[tuple[int, int]] = []
        for idx, item in enumerate(items):
            normalized_item = _ensure_tuple_pair(item, "raw_size")
            if normalized_item is None:
                raise ValueError("raw_size entries cannot be null.")
            normalized.append(normalized_item)
        return normalized

    @field_validator("canonical_size")
    @classmethod
    def _check_canonical_sizes(
        cls, value: Sequence[tuple[int, int] | None] | None, info: ValidationInfo
    ) -> Sequence[tuple[int, int] | None] | None:
        if value is None:
            return None
        items = list(value)
        batch = _batch_size(info)
        if batch and len(items) != batch:
            raise ValueError("canonical_size length must match batch size.")

        normalized: list[tuple[int, int] | None] = []
        for idx, item in enumerate(items):
            if item is None:
                normalized.append(None)
            else:
                normalized.append(_ensure_tuple_pair(item, "canonical_size") or None)
        return normalized

    @field_validator("metadata")
    @classmethod
    def _check_metadata(
        cls, value: list[dict[str, Any] | ImageMetadata | None] | None, info: ValidationInfo
    ) -> list[dict[str, Any] | ImageMetadata | None] | None:
        if value is None:
            return None
        batch = _batch_size(info)
        if batch and len(value) != batch:
            raise ValueError("metadata length must match batch size.")
        normalized: list[dict[str, Any] | ImageMetadata | None] = []
        for entry in value:
            if entry is None:
                normalized.append(None)
            elif isinstance(entry, (dict, ImageMetadata)):
                normalized.append(entry)
            else:
                raise TypeError("metadata entries must be dicts, ImageMetadata, or None.")
        return normalized


class ModelOutput(_ModelBase):
    """Model forward output used during training and evaluation."""

    prob_maps: torch.Tensor
    thresh_maps: torch.Tensor
    binary_maps: torch.Tensor
    loss: torch.Tensor | None = None
    loss_dict: dict[str, Any] | None = None

    @field_validator("thresh_maps", "binary_maps")
    @classmethod
    def _check_output_shapes(cls, tensor: torch.Tensor, info: ValidationInfo) -> torch.Tensor:
        data = _info_data(info)
        field_name = getattr(info, "field_name", None)
        if field_name == "thresh_maps":
            reference = data.get("binary_maps")
        elif field_name == "binary_maps":
            reference = data.get("thresh_maps")
        else:
            reference = None

        if isinstance(reference, torch.Tensor) and tensor.shape != reference.shape:
            raise ValueError("Model output tensors must share the same shape.")
        return tensor


class LightningStepPrediction(_ModelBase):
    """Prediction dictionary produced by the Lightning module for evaluation."""

    boxes: list[np.ndarray] = Field(..., description="Predicted polygons shaped (N, 2).")
    orientation: int = 1
    raw_size: tuple[int, int] | None = None
    canonical_size: tuple[int, int] | None = None
    image_path: str | None = None
    metadata: dict[str, Any] | ImageMetadata | None = None

    @field_validator("boxes")
    @classmethod
    def _validate_box(cls, polygons: list[np.ndarray]) -> list[np.ndarray]:
        return [PolygonArray(points=polygon).points for polygon in polygons]

    @field_validator("orientation")
    @classmethod
    def _validate_orientation(cls, value: int) -> int:
        orientation = int(value)
        if orientation not in VALID_EXIF_ORIENTATIONS:
            allowed = ", ".join(str(v) for v in sorted(VALID_EXIF_ORIENTATIONS))
            raise ValueError(f"Orientation must be one of {{{allowed}}}.")
        return orientation

    @field_validator("raw_size")
    @classmethod
    def _validate_raw_size(cls, value: tuple[int, int] | None) -> tuple[int, int] | None:
        return _ensure_tuple_pair(value, "raw_size")

    @field_validator("canonical_size")
    @classmethod
    def _validate_canonical_size(cls, value: tuple[int, int] | None) -> tuple[int, int] | None:
        return _ensure_tuple_pair(value, "canonical_size")

    @field_validator("image_path")
    @classmethod
    def _validate_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value:
            raise ValueError("Image path, when provided, must be a non-empty string.")
        return value

    @field_validator("metadata")
    @classmethod
    def _validate_metadata(cls, value: dict[str, Any] | ImageMetadata | None) -> dict[str, Any] | ImageMetadata | None:
        if value is None or isinstance(value, (dict, ImageMetadata)):
            return value
        raise TypeError("metadata must be a dict, ImageMetadata, or None.")


class MetricConfig(_ModelBase):
    """Configuration validation for CLEvalMetric parameters."""

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


def validate_predictions(filenames: Sequence[str], predictions: Sequence[dict[str, Any]]) -> list[LightningStepPrediction]:
    """Validate a collection of predictions against the expected schema."""

    def _make_error_details(
        *,
        error_type: str | PydanticCustomError,
        loc: tuple[Any, ...],
        original: Mapping[str, Any] | None = None,
    ) -> InitErrorDetails:
        input_value = None if original is None else original.get("input")
        ctx = None if original is None else original.get("ctx")
        if ctx:
            return InitErrorDetails(type=error_type, loc=loc, ctx=ctx, input=input_value)
        return InitErrorDetails(type=error_type, loc=loc, input=input_value)

    if len(filenames) != len(predictions):
        raise ValidationError.from_exception_data(
            LightningStepPrediction.__name__,
            line_errors=[
                _make_error_details(
                    error_type=PydanticCustomError("value_error.mismatched_lengths", "Number of filenames and predictions must match."),
                    loc=("__len__",),
                )
            ],
        )
    validated: list[LightningStepPrediction] = []
    for name, raw_pred in zip(filenames, predictions, strict=True):
        try:
            validated.append(LightningStepPrediction(**raw_pred))
        except ValidationError as exc:
            raise ValidationError.from_exception_data(
                LightningStepPrediction.__name__,
                line_errors=[
                    _make_error_details(
                        error_type=error["type"],
                        loc=("prediction", name, *error["loc"]),
                        original=error,
                    )
                    for error in exc.errors()
                ],
            ) from exc
    return validated


class ValidatedTensorData(_ModelBase):
    """Validate tensor data with shape, device, dtype, and value range checks.

    This model addresses critical tensor validation issues:
    - BUG-20251112-001: Dice loss assertion errors from out-of-range predictions
    - BUG-20251112-013: CUDA memory access errors from device mismatches

    Attributes:
        tensor: The tensor to validate
        expected_shape: Expected tensor shape (optional, validates if provided)
        expected_device: Expected device ("cpu", "cuda", "cuda:0", etc.) (optional)
        expected_dtype: Expected data type (optional)
        value_range: Valid value range as (min, max) tuple (optional)
        allow_inf: Whether to allow infinite values (default: False)
        allow_nan: Whether to allow NaN values (default: False)

    Example:
        >>> import torch
        >>> # Valid tensor with range checking
        >>> data = ValidatedTensorData(
        ...     tensor=torch.rand(2, 3, 224, 224),
        ...     expected_shape=(2, 3, 224, 224),
        ...     expected_device="cuda",
        ...     value_range=(0.0, 1.0)
        ... )
        >>> # Invalid: out of range
        >>> data = ValidatedTensorData(
        ...     tensor=torch.tensor([1.5, 2.0]),
        ...     value_range=(0.0, 1.0)
        ... )  # Raises ValidationError
    """

    tensor: torch.Tensor = Field(..., description="Tensor to validate")
    expected_shape: tuple[int, ...] | None = Field(default=None, description="Expected tensor shape")
    expected_device: torch.device | str | None = Field(default=None, description="Expected device")
    expected_dtype: torch.dtype | None = Field(default=None, description="Expected data type")
    value_range: tuple[float, float] | None = Field(default=None, description="Valid value range (min, max)")
    allow_inf: bool = Field(default=False, description="Whether to allow infinite values")
    allow_nan: bool = Field(default=False, description="Whether to allow NaN values")

    @model_validator(mode="after")
    def _validate_tensor(self) -> ValidatedTensorData:
        """Validate tensor with all constraints: type, shape, device, dtype, values."""
        tensor = self.tensor

        # Type validation (should already be validated by field type, but double-check)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor).__name__}")

        # Shape validation
        if self.expected_shape is not None:
            if tuple(tensor.shape) != tuple(self.expected_shape):
                raise ValueError(f"Tensor shape mismatch: expected {tuple(self.expected_shape)}, got {tuple(tensor.shape)}")

        # Device validation
        if self.expected_device is not None:
            expected_device_str = str(self.expected_device)
            actual_device_str = str(tensor.device)

            # Normalize device strings for comparison (e.g., "cuda" vs "cuda:0")
            if expected_device_str == "cuda" and actual_device_str.startswith("cuda"):
                pass  # Valid
            elif expected_device_str != actual_device_str:
                raise ValueError(f"Tensor device mismatch: expected {expected_device_str}, got {actual_device_str}")

        # Dtype validation
        if self.expected_dtype is not None:
            if tensor.dtype != self.expected_dtype:
                raise ValueError(f"Tensor dtype mismatch: expected {self.expected_dtype}, got {tensor.dtype}")

        # Value validation (NaN/Inf and range)
        if not self.allow_nan and torch.isnan(tensor).any():
            raise ValueError("Tensor contains NaN values (not allowed)")

        if not self.allow_inf and torch.isinf(tensor).any():
            raise ValueError("Tensor contains infinite values (not allowed)")

        # Value range validation
        if self.value_range is not None:
            min_val, max_val = self.value_range
            tensor_min = tensor.min().item()
            tensor_max = tensor.max().item()

            if tensor_min < min_val or tensor_max > max_val:
                raise ValueError(f"Tensor values out of range [{min_val}, {max_val}]: found values in [{tensor_min:.6f}, {tensor_max:.6f}]")

        return self

    @field_validator("value_range")
    @classmethod
    def _validate_value_range_format(cls, value: tuple[float, float] | None) -> tuple[float, float] | None:
        """Validate value_range format is (min, max) with min <= max."""
        if value is None:
            return None

        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError("value_range must be a tuple of (min, max)")

        min_val, max_val = value
        if min_val > max_val:
            raise ValueError(f"value_range min ({min_val}) must be <= max ({max_val})")

        return value


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Constants
    "VALID_EXIF_ORIENTATIONS",
    "validator",
    # Dataset schemas
    "CacheConfig",
    "ImageLoadingConfig",
    "DatasetConfig",
    "ImageMetadata",
    "PolygonData",
    "ValidatedPolygonData",
    "TransformInput",
    "TransformOutput",
    "TransformConfig",
    "ImageData",
    "MapData",
    "DataItem",
    # Validation models
    "PolygonArray",
    "DatasetSample",
    "BatchSample",
    "CollateOutput",
    "LoaderTransformOutput",
    "ModelOutput",
    "LightningStepPrediction",
    "MetricConfig",
    "ValidatedTensorData",
    "validate_predictions",
]
