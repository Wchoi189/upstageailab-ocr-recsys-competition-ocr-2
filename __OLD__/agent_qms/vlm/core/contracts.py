"""Data Contracts for VLM Module.

PydanticV2 models for type-safe data validation throughout the VLM pipeline.
"""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from agent_qms.vlm.core.config import get_config

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
    processed_path: Optional[Path] = None
    format: ImageFormat
    width: int = Field(..., gt=0, le=_IMAGE_MAX_DIMENSION)
    height: int = Field(..., gt=0, le=_IMAGE_MAX_DIMENSION)
    original_width: int = Field(..., gt=0)
    original_height: int = Field(..., gt=0)
    resize_ratio: float = Field(..., gt=0.0, le=1.0, description="Ratio of processed to original size")
    base64_encoded: Optional[str] = None
    size_bytes: int = Field(..., gt=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

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

    shape_attributes: Dict[str, Any] = Field(..., description="Shape attributes (x, y, width, height, etc.)")
    region_attributes: Dict[str, Any] = Field(default_factory=dict, description="Region attributes (labels, etc.)")

    class Config:
        """Pydantic configuration."""

        frozen = True


class VIAAnnotation(BaseModel):
    """VIA annotation file structure."""

    filename: str
    size: int = Field(..., gt=0, description="File size in bytes")
    regions: List[VIARegion] = Field(default_factory=list, description="Annotation regions")
    file_attributes: Dict[str, Any] = Field(default_factory=dict, description="File-level attributes")

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
    image_paths: List[Path] = Field(..., min_length=1, description="Paths to images to analyze")
    compare_with: Optional[Path] = None
    via_annotations: Optional[Path] = None
    initial_description: Optional[str] = None
    few_shot_examples: Optional[Path] = None
    template: Optional[Path] = None
    output_format: str = Field(default="markdown", pattern="^(text|markdown|json)$")
    auto_populate: bool = False
    experiment_id: Optional[str] = None
    incident_report: Optional[Path] = None
    backend_preference: Optional[str] = Field(default=None, pattern="^(openrouter|solar_pro2|cli)$")
    max_resolution: int = Field(default_factory=_default_analysis_max_resolution, gt=0, le=_IMAGE_MAX_DIMENSION)

    @field_validator("image_paths")
    @classmethod
    def validate_image_paths(cls, v: List[Path]) -> List[Path]:
        """Validate all image paths exist."""
        for path in v:
            if not path.exists():
                raise ValueError(f"Image path does not exist: {path}")
        return v

    @field_validator("compare_with")
    @classmethod
    def validate_compare_with(cls, v: Optional[Path], info) -> Optional[Path]:
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
    def validate_via_annotations(cls, v: Optional[Path]) -> Optional[Path]:
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
    image_paths: List[Path]
    analysis_text: str = Field(..., description="Generated analysis text")
    structured_data: Optional[Dict[str, Any]] = None
    backend_used: str
    processing_time_seconds: float = Field(..., ge=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        frozen = True


class BackendConfig(BaseModel):
    """Backend configuration."""

    backend_type: str = Field(..., pattern="^(openrouter|solar_pro2|cli)$")
    api_key: Optional[str] = None
    model: Optional[str] = None
    endpoint: Optional[str] = None
    timeout_seconds: int = Field(default_factory=_backend_timeout_default, gt=0, le=300)
    max_retries: int = Field(default_factory=_backend_retries_default, ge=0, le=10)
    max_resolution: int = Field(default_factory=_backend_max_resolution_default, gt=0, le=_IMAGE_MAX_DIMENSION)

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: Optional[str], info) -> Optional[str]:
        """Validate API key is provided for API backends."""
        backend_type = info.data.get("backend_type")
        if backend_type in ("openrouter", "solar_pro2") and not v:
            raise ValueError(f"API key is required for {backend_type} backend")
        return v

    class Config:
        """Pydantic configuration."""

        frozen = True
