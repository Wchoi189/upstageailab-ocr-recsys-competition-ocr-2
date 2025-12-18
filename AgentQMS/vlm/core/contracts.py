"""Data Contracts for VLM Module.

PydanticV2 models for type-safe data validation throughout the VLM pipeline.
"""

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from AgentQMS.vlm.core.config import get_config

_CONFIG_CACHE = get_config()
_IMAGE_MAX_DIMENSION = _CONFIG_CACHE.image.max_dimension


def _default_analysis_max_resolution() -> int:
    return get_config().image.max_resolution


def _backend_timeout_default() -> int:
    return get_config().backend_defaults.timeout_seconds


def _backend_retries_default() -> int:
    return get_config().backend_defaults.max_retries


def _backend_max_resolution_default() -> int:
    return get_config().backend_defaults.max_resolution


class ImageFormat(str, Enum):
    """Supported image formats."""

    JPEG = "JPEG"
    PNG = "PNG"
    WEBP = "WEBP"


class AnalysisMode(str, Enum):
    """Available analysis modes."""

    DEFECT = "defect"
    INPUT = "input"
    COMPARE = "compare"
    FULL = "full"
    BUG_001 = "bug_001"
    # Image enhancement experiment modes
    IMAGE_QUALITY = "image_quality"
    ENHANCEMENT_VALIDATION = "enhancement_validation"
    PREPROCESSING_DIAGNOSIS = "preprocessing_diagnosis"


class ImageData(BaseModel):
    """Validated image file data."""

    path: Path
    format: ImageFormat
    width: int = Field(..., gt=0, le=_IMAGE_MAX_DIMENSION, description="Image width in pixels")
    height: int = Field(..., gt=0, le=_IMAGE_MAX_DIMENSION, description="Image height in pixels")
    size_bytes: int = Field(..., gt=0, description="File size in bytes")

    @field_validator("path")
    @classmethod
    def validate_path_exists(cls, v: Path) -> Path:
        """Validate that image path exists."""
        if not v.exists():
            raise ValueError(f"Image path does not exist: {v}")
        if not v.is_file():
            raise ValueError(f"Image path is not a file: {v}")
        return v

    @field_validator("format", mode="before")
    @classmethod
    def validate_format(cls, v: Any) -> ImageFormat:
        """Validate and convert image format."""
        if isinstance(v, str):
            v = v.upper()
            try:
                return ImageFormat(v)
            except ValueError:
                raise ValueError(f"Unsupported image format: {v}. Must be one of {[f.value for f in ImageFormat]}")
        return v

    class Config:
        """Pydantic configuration."""

        frozen = True
        str_strip_whitespace = True


class ProcessedImage(BaseModel):
    """Preprocessed image data ready for VLM analysis."""

    original_path: Path
    processed_path: Path | None = None
    format: ImageFormat
    width: int = Field(..., gt=0, le=_IMAGE_MAX_DIMENSION)
    height: int = Field(..., gt=0, le=_IMAGE_MAX_DIMENSION)
    original_width: int = Field(..., gt=0)
    original_height: int = Field(..., gt=0)
    resize_ratio: float = Field(..., gt=0.0, le=1.0, description="Ratio of processed to original size")
    base64_encoded: str | None = None
    size_bytes: int = Field(..., gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("resize_ratio")
    @classmethod
    def validate_resize_ratio(cls, v: float) -> float:
        """Validate resize ratio is reasonable."""
        if v <= 0 or v > 1:
            raise ValueError(f"Resize ratio must be between 0 and 1, got {v}")
        return v

    class Config:
        """Pydantic configuration."""

        frozen = True


class VIARegion(BaseModel):
    """VIA annotation region (bounding box, polygon, etc.)."""

    shape_attributes: dict[str, Any] = Field(..., description="Shape attributes (x, y, width, height, etc.)")
    region_attributes: dict[str, Any] = Field(default_factory=dict, description="Region attributes (labels, etc.)")

    class Config:
        """Pydantic configuration."""

        frozen = True


class VIAAnnotation(BaseModel):
    """VIA annotation file structure."""

    filename: str
    size: int = Field(..., gt=0, description="File size in bytes")
    regions: list[VIARegion] = Field(default_factory=list, description="Annotation regions")
    file_attributes: dict[str, Any] = Field(default_factory=dict, description="File-level attributes")

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v: str) -> str:
        """Validate filename is not empty."""
        if not v or not v.strip():
            raise ValueError("Filename cannot be empty")
        return v.strip()

    class Config:
        """Pydantic configuration."""

        frozen = True


class AnalysisRequest(BaseModel):
    """Analysis request parameters."""

    mode: AnalysisMode
    image_paths: list[Path] = Field(..., min_length=1, description="Paths to images to analyze")
    compare_with: Path | None = None
    via_annotations: Path | None = None
    initial_description: str | None = None
    few_shot_examples: Path | None = None
    template: Path | None = None
    output_format: str = Field(default="markdown", pattern="^(text|markdown|json)$")
    auto_populate: bool = False
    experiment_id: str | None = None
    incident_report: Path | None = None
    backend_preference: str | None = Field(default=None, pattern="^(dashscope|openrouter|solar_pro2|cli)$")
    max_resolution: int = Field(default_factory=_default_analysis_max_resolution, gt=0, le=_IMAGE_MAX_DIMENSION)

    @field_validator("image_paths")
    @classmethod
    def validate_image_paths(cls, v: list[Path]) -> list[Path]:
        """Validate all image paths exist."""
        for path in v:
            if not path.exists():
                raise ValueError(f"Image path does not exist: {path}")
        return v

    @field_validator("compare_with")
    @classmethod
    def validate_compare_with(cls, v: Path | None, info) -> Path | None:
        """Validate compare_with path exists if provided."""
        if v is not None:
            if not v.exists():
                raise ValueError(f"Compare image path does not exist: {v}")
            mode = info.data.get("mode")
            if mode != AnalysisMode.COMPARE:
                raise ValueError("compare_with can only be used with compare mode")
        return v

    @field_validator("via_annotations")
    @classmethod
    def validate_via_annotations(cls, v: Path | None) -> Path | None:
        """Validate VIA annotations file exists if provided."""
        if v is not None and not v.exists():
            raise ValueError(f"VIA annotations file does not exist: {v}")
        return v

    class Config:
        """Pydantic configuration."""

        frozen = True


class AnalysisResult(BaseModel):
    """VLM analysis result."""

    mode: AnalysisMode
    image_paths: list[Path]
    analysis_text: str = Field(..., description="Generated analysis text")
    structured_data: dict[str, Any] | None = None
    backend_used: str
    processing_time_seconds: float = Field(..., ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        frozen = True


class BackendConfig(BaseModel):
    """Backend configuration."""

    backend_type: str = Field(..., pattern="^(openrouter|solar_pro2|cli|dashscope)$")
    api_key: str | None = None
    model: str | None = None
    endpoint: str | None = None
    timeout_seconds: int = Field(default_factory=_backend_timeout_default, gt=0, le=300)
    max_retries: int = Field(default_factory=_backend_retries_default, ge=0, le=10)
    max_resolution: int = Field(default_factory=_backend_max_resolution_default, gt=0, le=_IMAGE_MAX_DIMENSION)

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str | None, info) -> str | None:
        """Validate API key is provided for API backends."""
        backend_type = info.data.get("backend_type")
        if backend_type in ("openrouter", "solar_pro2", "dashscope") and not v:
            raise ValueError(f"API key is required for {backend_type} backend")
        return v

    class Config:
        """Pydantic configuration."""

        frozen = True
