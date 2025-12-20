"""Pydantic v2 models for inference API data contracts.

These models provide type safety and validation for inference requests/responses.
They align with TypeScript types in frontend applications.

See: docs/artifacts/specs/shared-backend-contract.md
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Padding(BaseModel):
    """Padding applied during preprocessing."""

    top: int = Field(default=0, ge=0, description="Top padding in pixels")
    bottom: int = Field(default=0, ge=0, description="Bottom padding in pixels")
    left: int = Field(default=0, ge=0, description="Left padding in pixels")
    right: int = Field(default=0, ge=0, description="Right padding in pixels")


class InferenceMetadata(BaseModel):
    """Metadata describing coordinate system and transformations.

    Critical for frontend coordinate handling and overlay alignment.
    Provides information about image preprocessing transformations.
    """

    original_size: tuple[int, int] = Field(
        ...,
        description="Original image size (width, height) before preprocessing",
    )
    processed_size: tuple[int, int] = Field(
        ...,
        description="Processed image size (width, height) after resize/padding (typically 640x640)",
    )
    padding: Padding = Field(
        ...,
        description="Padding applied during preprocessing",
    )
    padding_position: str = Field(
        default="top_left",
        description="Padding position: 'top_left', 'center', etc.",
    )
    content_area: tuple[int, int] = Field(
        ...,
        description="Content area size (width, height) within processed_size frame",
    )
    scale: float = Field(
        ...,
        description="Scaling factor applied during resize",
        gt=0,
    )
    coordinate_system: str = Field(
        default="pixel",
        description="Coordinate system: 'pixel' (absolute) or 'normalized' (0-1)",
    )


class TextRegion(BaseModel):
    """Detected text region with polygon coordinates."""

    polygon: list[list[float]] = Field(
        ...,
        description="Polygon vertices as [[x1, y1], [x2, y2], ...] (typically 4 points)",
        min_length=3,
    )
    text: str | None = Field(
        None,
        description="Recognized text (None if recognition not performed)",
    )
    confidence: float = Field(
        ...,
        description="Detection confidence score",
        ge=0.0,
        le=1.0,
    )


class InferenceRequest(BaseModel):
    """Request body for inference endpoint."""

    checkpoint_path: str = Field(
        ...,
        description="Path to model checkpoint (.ckpt file)",
    )
    image_base64: str | None = Field(
        None,
        description="Base64-encoded image data (with or without data URL prefix)",
    )
    image_path: str | None = Field(
        None,
        description="Path to image file (absolute or relative to PROJECT_ROOT)",
    )
    confidence_threshold: float = Field(
        default=0.3,
        description="Binarization threshold (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )
    nms_threshold: float = Field(
        default=0.5,
        description="Box threshold for NMS (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )
    enable_perspective_correction: bool = Field(
        default=False,
        description="Enable rembg-based perspective correction before inference",
    )
    perspective_display_mode: str = Field(
        default="corrected",
        description="Display mode: 'corrected' shows corrected image with annotations, 'original' shows original image with transformed annotations",
        pattern="^(corrected|original)$",
    )
    enable_grayscale: bool = Field(
        default=False,
        description="Enable grayscale preprocessing (converts image to grayscale before inference)",
    )
    enable_background_normalization: bool = Field(
        default=False,
        description="Enable gray-world background normalization before inference (reduces tinted backgrounds)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "checkpoint_path": "outputs/experiments/train/ocr/checkpoint.ckpt",
                    "image_base64": "data:image/png;base64,iVBORw0KG...",
                    "confidence_threshold": 0.3,
                    "nms_threshold": 0.4,
                    "enable_perspective_correction": False,
                    "perspective_display_mode": "corrected",
                    "enable_grayscale": False,
                    "enable_background_normalization": False,
                }
            ]
        }
    }


class InferenceResponse(BaseModel):
    """Response body for inference endpoint."""

    status: str = Field(
        default="success",
        description="Response status: 'success' or 'error'",
    )
    regions: list[TextRegion] = Field(
        default_factory=list,
        description="Detected text regions",
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Inference processing time in milliseconds",
        ge=0.0,
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Optional notes or warnings",
    )
    preview_image_base64: str | None = Field(
        None,
        description="Base64-encoded preview image (JPEG) for overlay alignment",
    )
    meta: InferenceMetadata | None = Field(
        None,
        description="Coordinate system metadata (critical for frontend rendering)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "success",
                    "regions": [
                        {
                            "polygon": [[10, 10], [100, 10], [100, 50], [10, 50]],
                            "text": "Sample text",
                            "confidence": 0.95,
                        }
                    ],
                    "processing_time_ms": 150.5,
                    "notes": [],
                    "preview_image_base64": "data:image/jpeg;base64,/9j/4AAQ...",
                    "meta": {
                        "original_size": [1920, 1080],
                        "processed_size": [640, 640],
                        "padding": {"top": 0, "bottom": 280, "left": 0, "right": 0},
                        "padding_position": "top_left",
                        "content_area": [640, 360],
                        "scale": 0.333,
                        "coordinate_system": "pixel",
                    },
                }
            ]
        }
    }
