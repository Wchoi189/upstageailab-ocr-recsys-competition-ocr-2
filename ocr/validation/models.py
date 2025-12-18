"""Runtime validation models for the OCR pipeline data contracts.

These Pydantic models aim to surface interface regressions immediately during
refactoring work on the Lightning module. They are lightweight checks that
validate tensor and polygon shapes as well as key metadata conventions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator, model_validator
from pydantic_core import InitErrorDetails, PydanticCustomError

from ocr.datasets.schemas import ImageMetadata

# Orientations align with EXIF specification; 0 represents "unknown" while
# 1-8 map to the standard rotation/mirroring states. Align this contract with
# `ocr.utils.orientation` helpers to avoid mismatches during evaluation.
VALID_EXIF_ORIENTATIONS: frozenset[int] = frozenset({0, 1, 2, 3, 4, 5, 6, 7, 8})

# Maintain a legacy alias so tools referencing the deprecated decorator name still
# resolve it, even though the implementation now uses Pydantic v2's ``field_validator``.
validator = field_validator


def _info_data(info: ValidationInfo | None) -> Mapping[str, Any]:
    """Safely extract validator context data across Pydantic versions."""
    if info is None:
        return {}
    data = getattr(info, "data", None)
    if isinstance(data, Mapping):
        return data
    return {}


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


class TransformOutput(_ModelBase):
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
        batch = len(_info_data(info).get("image_filename", []))
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images tensor must be shaped (B, 3, H, W).")
        if batch and images.shape[0] != batch:
            raise ValueError("Number of images does not match batch metadata.")
        return images

    @field_validator("polygons")
    @classmethod
    def _check_collated_polygons(cls, polygons: list[list[np.ndarray]], info: ValidationInfo) -> list[list[np.ndarray]]:
        filenames = _info_data(info).get("image_filename", [])
        if len(polygons) != len(filenames):
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
        batch = len(_info_data(info).get("image_filename", []))
        if tensor.ndim != 4 or tensor.shape[1] != 1:
            raise ValueError("prob_maps tensor must be shaped (B, 1, H, W).")
        if batch and tensor.shape[0] != batch:
            raise ValueError("prob_maps batch dimension must match batch size.")
        return tensor

    @field_validator("thresh_maps")
    @classmethod
    def _check_collated_thresh_maps(cls, tensor: torch.Tensor, info: ValidationInfo) -> torch.Tensor:
        batch = len(_info_data(info).get("image_filename", []))
        if tensor.ndim != 4 or tensor.shape[1] != 1:
            raise ValueError("thresh_maps tensor must be shaped (B, 1, H, W).")
        if batch and tensor.shape[0] != batch:
            raise ValueError("thresh_maps batch dimension must match batch size.")
        return tensor

    @field_validator("orientation")
    @classmethod
    def _check_orientation(cls, value: Sequence[int] | None, info: ValidationInfo) -> Sequence[int] | None:
        if value is None:
            return None
        batch = len(_info_data(info).get("image_filename", []))
        if len(value) != batch:
            raise ValueError("orientation length must match batch size.")
        normalized: list[int] = []
        for idx, item in enumerate(value):
            try:
                orientation = int(item)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"orientation[{idx}] must be castable to int.") from exc
            if orientation not in VALID_EXIF_ORIENTATIONS:
                allowed = ", ".join(str(v) for v in sorted(VALID_EXIF_ORIENTATIONS))
                raise ValueError(f"orientation[{idx}] must be one of {{{allowed}}}.")
            normalized.append(orientation)
        return normalized

    @field_validator("raw_size")
    @classmethod
    def _check_raw_sizes(cls, value: Sequence[tuple[int, int]] | None, info: ValidationInfo) -> Sequence[tuple[int, int]] | None:
        if value is None:
            return None
        batch = len(_info_data(info).get("image_filename", []))
        if len(value) != batch:
            raise ValueError("raw_size length must match batch size.")
        normalized: list[tuple[int, int]] = []
        for item in value:
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
        batch = len(info.data.get("image_filename", [])) if info.data else 0
        if len(value) != batch:
            raise ValueError("canonical_size length must match batch size.")
        return [_ensure_tuple_pair(item, "canonical_size") for item in value]

    @field_validator("metadata")
    @classmethod
    def _check_metadata(
        cls, value: list[dict[str, Any] | ImageMetadata | None] | None, info: ValidationInfo
    ) -> list[dict[str, Any] | ImageMetadata | None] | None:
        if value is None:
            return None
        batch = len(_info_data(info).get("image_filename", []))
        if len(value) != batch:
            raise ValueError("metadata length must match batch size.")
        normalized: list[dict[str, Any] | ImageMetadata | None] = []
        for entry in value:
            if entry is None:
                normalized.append(None)
            elif isinstance(entry, dict | ImageMetadata):
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
        reference = _info_data(info).get("prob_maps")
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
        if value is None or isinstance(value, dict | ImageMetadata):
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
